#!/usr/bin/env bash
set -u

SCRIPT="$(readlink -f "${BASH_SOURCE[0]}")"
ROOT="$(cd -- "$(dirname -- "$SCRIPT")" && pwd)"
LOG_DIR="$ROOT/logs"
SUPERVISOR_LOG="$LOG_DIR/supervisor.log"
SUPERVISOR_PIDFILE="$LOG_DIR/supervisor.pid"
BOT_PIDFILE="$LOG_DIR/bot.pid"
VENV_PYTHON="$ROOT/.venv/bin/python"
BACKOFF=(2 5 10 30 60)

mkdir -p "$LOG_DIR"

if [ -x "$VENV_PYTHON" ]; then
    PYTHON="$VENV_PYTHON"
else
    PYTHON="${PYTHON:-python3}"
fi

read_pid() {
    local file="$1"
    [ -r "$file" ] || return 1
    local pid
    pid="$(<"$file")"
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    printf '%s' "$pid"
}

cmdline_of() {
    local pid="$1"
    [ -r "/proc/$pid/cmdline" ] || return 1
    tr '\0' ' ' < "/proc/$pid/cmdline"
}

is_supervisor_pid() {
    local pid="$1" cmd
    kill -0 "$pid" 2>/dev/null || return 1
    cmd="$(cmdline_of "$pid")" || return 1
    [[ "$cmd" == *"$SCRIPT"* && "$cmd" == *" supervise"* ]]
}

is_bot_pid() {
    local pid="$1" cmd cwd
    kill -0 "$pid" 2>/dev/null || return 1
    cmd="$(cmdline_of "$pid")" || return 1
    cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null)" || return 1
    [[ "$cwd" == "$ROOT" && "$cmd" == *"-m bot.bridge"* ]]
}

check_python() {
    if ! "$PYTHON" -c 'import aiohttp, httpx, feedparser, yaml, trafilatura, newspaper, pyalex, apscheduler, openai, bs4' >/dev/null 2>&1; then
        printf '缺少依赖。请运行：\n  cd %s && python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt\n' "$ROOT" >&2
        return 1
    fi
}

foreground() {
    local supervisor_pid
    if supervisor_pid="$(read_pid "$SUPERVISOR_PIDFILE")" && is_supervisor_pid "$supervisor_pid"; then
        echo "后台 News Bot 已运行（supervisor pid $supervisor_pid），请先执行 ./run.sh stop" >&2
        return 1
    fi
    check_python || return 1
    cd "$ROOT" || return 1
    exec "$PYTHON" -m bot.bridge
}

supervise() {
    check_python || exit 1
    printf '%s\n' "$$" > "$SUPERVISOR_PIDFILE"

    local stopping=0 child_pid='' retry_index=0 started_at runtime status delay

    terminate_child() {
        stopping=1
        if [ -n "$child_pid" ] && is_bot_pid "$child_pid"; then
            kill -TERM "$child_pid" 2>/dev/null || true
        fi
    }

    cleanup() {
        if [ -n "$child_pid" ] && is_bot_pid "$child_pid"; then
            kill -TERM "$child_pid" 2>/dev/null || true
        fi
        rm -f "$BOT_PIDFILE"
        if [ -r "$SUPERVISOR_PIDFILE" ] && [ "$(<"$SUPERVISOR_PIDFILE")" = "$$" ]; then
            rm -f "$SUPERVISOR_PIDFILE"
        fi
    }

    trap terminate_child TERM INT
    trap cleanup EXIT

    while [ "$stopping" -eq 0 ]; do
        started_at="$(date +%s)"
        printf '=== %s Bot child START ===\n' "$(date '+%F %T')"
        cd "$ROOT" || exit 1
        "$PYTHON" -m bot.bridge &
        child_pid=$!
        printf '%s\n' "$child_pid" > "$BOT_PIDFILE"

        wait "$child_pid"
        status=$?
        runtime=$(( $(date +%s) - started_at ))
        rm -f "$BOT_PIDFILE"
        child_pid=''

        if [ "$stopping" -ne 0 ]; then
            break
        fi

        delay="${BACKOFF[$retry_index]}"
        printf '=== %s Bot child EXIT status=%s runtime=%ss; restart in %ss ===\n' \
            "$(date '+%F %T')" "$status" "$runtime" "$delay"
        if [ "$runtime" -ge 300 ]; then
            retry_index=0
        elif [ "$retry_index" -lt $(( ${#BACKOFF[@]} - 1 )) ]; then
            retry_index=$((retry_index + 1))
        fi
        sleep "$delay" || true
    done
}

start() {
    local pid i child_pid
    if pid="$(read_pid "$SUPERVISOR_PIDFILE")" && is_supervisor_pid "$pid"; then
        echo "News Bot 已运行（supervisor pid $pid）"
        return 0
    fi
    rm -f "$SUPERVISOR_PIDFILE" "$BOT_PIDFILE"
    check_python || return 1
    printf '=== %s Supervisor START ===\n' "$(date '+%F %T')" >> "$SUPERVISOR_LOG"
    nohup "$SCRIPT" supervise </dev/null >> "$SUPERVISOR_LOG" 2>&1 &
    pid=$!
    printf '%s\n' "$pid" > "$SUPERVISOR_PIDFILE"

    for i in $(seq 1 10); do
        if ! is_supervisor_pid "$pid"; then
            echo "News Bot supervisor 启动失败，请查看 $SUPERVISOR_LOG" >&2
            return 1
        fi
        if child_pid="$(read_pid "$BOT_PIDFILE")" && is_bot_pid "$child_pid"; then
            echo "News Bot 已启动（supervisor pid $pid, bot pid $child_pid）"
            echo "日志：$ROOT/logs/bot.log"
            return 0
        fi
        sleep 1
    done
    echo "Supervisor 已启动，但 Bot 子进程尚未就绪；请运行 ./run.sh status" >&2
    return 1
}

stop() {
    local supervisor_pid='' bot_pid='' i
    supervisor_pid="$(read_pid "$SUPERVISOR_PIDFILE" 2>/dev/null || true)"
    bot_pid="$(read_pid "$BOT_PIDFILE" 2>/dev/null || true)"

    if [ -n "$supervisor_pid" ] && is_supervisor_pid "$supervisor_pid"; then
        kill -TERM "$supervisor_pid" 2>/dev/null || true
        for i in $(seq 1 10); do
            kill -0 "$supervisor_pid" 2>/dev/null || break
            sleep 1
        done
        if is_supervisor_pid "$supervisor_pid"; then
            [ -n "$bot_pid" ] && is_bot_pid "$bot_pid" && kill -TERM "$bot_pid" 2>/dev/null || true
            kill -KILL "$supervisor_pid" 2>/dev/null || true
        fi
    elif [ -n "$bot_pid" ] && is_bot_pid "$bot_pid"; then
        kill -TERM "$bot_pid" 2>/dev/null || true
    else
        echo "News Bot 未运行"
        rm -f "$SUPERVISOR_PIDFILE" "$BOT_PIDFILE"
        return 0
    fi

    if [ -n "$bot_pid" ] && is_bot_pid "$bot_pid"; then
        kill -TERM "$bot_pid" 2>/dev/null || true
        sleep 1
        is_bot_pid "$bot_pid" && kill -KILL "$bot_pid" 2>/dev/null || true
    fi
    rm -f "$SUPERVISOR_PIDFILE" "$BOT_PIDFILE"
    echo "News Bot 已停止"
}

status() {
    local supervisor_pid='' bot_pid=''
    supervisor_pid="$(read_pid "$SUPERVISOR_PIDFILE" 2>/dev/null || true)"
    bot_pid="$(read_pid "$BOT_PIDFILE" 2>/dev/null || true)"

    if [ -n "$supervisor_pid" ] && is_supervisor_pid "$supervisor_pid"; then
        if [ -n "$bot_pid" ] && is_bot_pid "$bot_pid"; then
            echo "News Bot 运行中：supervisor pid $supervisor_pid, bot pid $bot_pid"
            ps -o pid=,ppid=,etime=,cmd= -p "$supervisor_pid" -p "$bot_pid"
            return 0
        fi
        echo "News Bot supervisor 运行中（pid $supervisor_pid），Bot 正在重启"
        return 1
    fi

    echo "News Bot 未运行"
    return 1
}

case "${1:-foreground}" in
    foreground|fg) foreground ;;
    supervise) supervise ;;
    start) start ;;
    stop) stop ;;
    restart) stop; sleep 1; start ;;
    status) status ;;
    *) echo "Usage: $0 [foreground|start|stop|restart|status]" >&2; exit 2 ;;
esac
