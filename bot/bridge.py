"""Minimal independent QQ Bot: private /ping -> pong."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import aiohttp
import httpx

from bot.commands import CommandHandler
from news.pipeline import NewsPipeline
from scheduler import DailyNewsScheduler
from settings import bind_qq_target_openid, load_settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "bot.log"

API_BASE = "https://api.sgroup.qq.com"
TOKEN_URL = "https://bots.qq.com/app/getAppAccessToken"
GATEWAY_URL = f"{API_BASE}/gateway"
CONNECT_TIMEOUT = 20
RECONNECT_BACKOFF = (2, 5, 10, 30, 60)
USER_AGENT = "wechat-news-qq-bot/0.1"
INTENTS = (1 << 25) | (1 << 30) | (1 << 12) | (1 << 26)

logger = logging.getLogger("wechat_news")


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def load_config() -> tuple[str, str]:
    settings = load_settings()
    if not settings.qq_app_id or not settings.qq_client_secret:
        raise RuntimeError(".env 必须包含 QQ_APP_ID 和 QQ_CLIENT_SECRET")
    return settings.qq_app_id, settings.qq_client_secret


def _masked_openid(openid: str) -> str:
    if len(openid) <= 8:
        return "***"
    return f"{openid[:4]}…{openid[-4:]}"


def _proxy_config() -> tuple[bool, str]:
    proxy = next(
        (
            os.environ.get(name)
            for name in (
                "HTTPS_PROXY",
                "https_proxy",
                "HTTP_PROXY",
                "http_proxy",
                "ALL_PROXY",
                "all_proxy",
            )
            if os.environ.get(name)
        ),
        None,
    )
    if not proxy:
        return True, "未检测到代理环境变量，使用直连"

    try:
        parsed = urlsplit(proxy)
        scheme = parsed.scheme.lower()
        host = parsed.hostname or "unknown"
        endpoint = f"{host}:{parsed.port}" if parsed.port else host
    except (TypeError, ValueError):
        scheme = proxy.split(":", 1)[0].lower()
        endpoint = "已脱敏"

    if scheme in {"socks", "socks4", "socks5", "socks5h"}:
        return False, f"检测到 SOCKS 代理 {scheme}://{endpoint}；aiohttp WebSocket 回退直连"
    return True, f"检测到代理 {scheme}://{endpoint}，WebSocket 使用 trust_env"


class QQNewsBot:
    def __init__(self, app_id: str, client_secret: str) -> None:
        self.app_id = app_id
        self.client_secret = client_secret
        self.http: httpx.AsyncClient | None = None
        self.access_token = ""
        self.token_expires_at = 0.0
        self.last_sequence: int | None = None
        self.message_sequences: dict[str, int] = {}
        self.seen_messages: dict[str, float] = {}
        self.websocket: aiohttp.ClientWebSocketResponse | None = None
        self.heartbeat_task: asyncio.Task[None] | None = None
        self.message_tasks: set[asyncio.Task[None]] = set()
        self.stop_event = asyncio.Event()
        self.settings = load_settings()
        self.pipeline = NewsPipeline(self.settings, logger)
        self.command_handler = CommandHandler(self.settings, self.pipeline)
        self.daily_scheduler: DailyNewsScheduler | None = None
        self.target_bind_lock = asyncio.Lock()

    async def ensure_token(self) -> str:
        if self.access_token and time.time() < self.token_expires_at - 60:
            return self.access_token
        if self.http is None:
            raise RuntimeError("HTTP client 未初始化")

        logger.info("正在获取 QQ access token")
        response = await self.http.post(
            TOKEN_URL,
            json={"appId": self.app_id, "clientSecret": self.client_secret},
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        token = data.get("access_token")
        if not isinstance(token, str) or not token:
            raise RuntimeError("QQ token 响应缺少 access_token")

        expires_in = int(data.get("expires_in", 7200))
        self.access_token = token
        self.token_expires_at = time.time() + expires_in
        logger.info("QQ access token 获取成功，expires_in=%ss", expires_in)
        return token

    async def get_gateway(self) -> str:
        if self.http is None:
            raise RuntimeError("HTTP client 未初始化")
        token = await self.ensure_token()
        logger.info("正在获取 QQ gateway")
        response = await self.http.get(
            GATEWAY_URL,
            headers={
                "Authorization": f"QQBot {token}",
                "User-Agent": USER_AGENT,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        gateway = data.get("url")
        if not isinstance(gateway, str) or not gateway:
            raise RuntimeError("QQ gateway 响应缺少 url")
        logger.info("QQ gateway 获取成功")
        return gateway

    async def send_identify(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        token = await self.ensure_token()
        await ws.send_json(
            {
                "op": 2,
                "d": {
                    "token": f"QQBot {token}",
                    "intents": INTENTS,
                    "shard": [0, 1],
                    "properties": {
                        "$os": "Linux",
                        "$browser": "wechat-news",
                        "$device": "wechat-news",
                    },
                },
            }
        )
        logger.info("QQ Identify 已发送")

    async def heartbeat_sender(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        interval: float,
    ) -> None:
        first = True
        try:
            while not self.stop_event.is_set() and not ws.closed:
                await asyncio.sleep(interval)
                if self.stop_event.is_set() or ws.closed:
                    return
                await ws.send_json({"op": 1, "d": self.last_sequence})
                if first:
                    logger.info("QQ heartbeat 已发送")
                    first = False
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("QQ heartbeat 失败：%s", exc)
            if not ws.closed:
                await ws.close()

    def is_duplicate(self, message_id: str) -> bool:
        now = time.time()
        previous = self.seen_messages.get(message_id)
        if previous is not None and now - previous < 300:
            return True
        self.seen_messages[message_id] = now
        if len(self.seen_messages) > 1000:
            self.seen_messages = {
                key: timestamp
                for key, timestamp in self.seen_messages.items()
                if now - timestamp <= 600
            }
        return False

    def next_message_sequence(self, openid: str) -> int:
        if len(self.message_sequences) > 1000 and openid not in self.message_sequences:
            self.message_sequences.clear()
        sequence = self.message_sequences.get(openid, 0) + 1
        self.message_sequences[openid] = sequence
        return sequence

    async def send_text(self, openid: str, content: str) -> bool:
        if self.http is None:
            raise RuntimeError("HTTP client 未初始化")
        chunks: list[str] = []
        remaining = content.strip()
        while remaining:
            if len(remaining) <= 3500:
                chunks.append(remaining)
                break
            split_at = remaining.rfind("\n", 0, 3500)
            if split_at < 1000:
                split_at = 3500
            chunks.append(remaining[:split_at].rstrip())
            remaining = remaining[split_at:].lstrip()

        token = await self.ensure_token()
        for chunk in chunks or [""]:
            response = await self.http.post(
                f"{API_BASE}/v2/users/{openid}/messages",
                headers={
                    "Authorization": f"QQBot {token}",
                    "Content-Type": "application/json",
                    "User-Agent": USER_AGENT,
                },
                json={
                    "markdown": {"content": chunk},
                    "msg_type": 2,
                    "msg_seq": self.next_message_sequence(openid),
                },
                timeout=30.0,
            )
            if response.status_code >= 400:
                excerpt = response.text.replace("\n", " ")[:200]
                logger.error("QQ message API 错误 status=%s body=%s", response.status_code, excerpt)
                return False
        logger.info("QQ message sent，openid=%s chunks=%s", _masked_openid(openid), len(chunks))
        return True

    async def send_pong(self, openid: str) -> bool:
        if self.http is None:
            raise RuntimeError("HTTP client 未初始化")
        token = await self.ensure_token()
        response = await self.http.post(
            f"{API_BASE}/v2/users/{openid}/messages",
            headers={
                "Authorization": f"QQBot {token}",
                "Content-Type": "application/json",
                "User-Agent": USER_AGENT,
            },
            json={
                "markdown": {"content": "pong"},
                "msg_type": 2,
                "msg_seq": self.next_message_sequence(openid),
            },
            timeout=30.0,
        )
        if response.status_code >= 400:
            excerpt = response.text.replace("\n", " ")[:200]
            logger.error("QQ pong API 错误 status=%s body=%s", response.status_code, excerpt)
            return False
        logger.info("已成功回复 pong，openid=%s", _masked_openid(openid))
        return True

    async def ensure_target_bound(self, openid: str) -> bool:
        if self.settings.qq_target_openid:
            return False
        async with self.target_bind_lock:
            if self.settings.qq_target_openid:
                return False
            bound, target = await asyncio.to_thread(bind_qq_target_openid, openid)
            self.settings.qq_target_openid = target
            if bound:
                logger.info("QQ_TARGET_OPENID 自动绑定成功，openid=%s", _masked_openid(target))
            return bound

    async def handle_c2c_message(self, data: Any) -> None:
        if not isinstance(data, dict):
            return
        message_id = str(data.get("id", ""))
        if not message_id or self.is_duplicate(message_id):
            return
        content = str(data.get("content", "")).strip()
        author = data.get("author") if isinstance(data.get("author"), dict) else {}
        openid = str(author.get("user_openid", ""))
        if not openid or not content:
            return

        try:
            await self.ensure_target_bound(openid)
        except Exception as exc:
            logger.error("QQ_TARGET_OPENID 自动绑定失败：%s", exc)

        if content == "/ping":
            logger.info("收到 /ping，message_id=%s openid=%s", message_id, _masked_openid(openid))
            try:
                await self.send_pong(openid)
            except Exception as exc:
                logger.error("回复 pong 时发生 API 错误：%s", exc)
            return

        if not content.startswith("/"):
            return
        logger.info("收到命令 %s，openid=%s", content.split()[0], _masked_openid(openid))
        try:
            response = await self.command_handler.handle(content)
            if response:
                await self.send_text(openid, response)
        except Exception as exc:
            logger.exception("命令处理失败 command=%s: %s", content.split()[0], exc)
            await self.send_text(openid, f"命令执行失败：{type(exc).__name__}: {exc}")

    def _message_task_done(self, task: asyncio.Task[None]) -> None:
        self.message_tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error("C2C 消息处理失败：%s", error)

    async def event_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        self.websocket = ws
        self.last_sequence = None
        identify_sent = False
        try:
            while not self.stop_event.is_set() and not ws.closed:
                message = await ws.receive()
                if message.type == aiohttp.WSMsgType.TEXT:
                    try:
                        payload = json.loads(message.data)
                    except (json.JSONDecodeError, TypeError):
                        logger.warning("QQ WebSocket 收到无法解析的 JSON")
                        continue

                    op = payload.get("op")
                    event_type = payload.get("t")
                    sequence = payload.get("s")
                    data = payload.get("d")
                    if isinstance(sequence, int) and (
                        self.last_sequence is None or sequence > self.last_sequence
                    ):
                        self.last_sequence = sequence

                    if op == 10:
                        hello = data if isinstance(data, dict) else {}
                        interval_ms = hello.get("heartbeat_interval", 30000)
                        interval = float(interval_ms) / 1000.0 * 0.8
                        logger.info("QQ Hello 收到，heartbeat_interval=%.1fs", interval)
                        if self.heartbeat_task and not self.heartbeat_task.done():
                            self.heartbeat_task.cancel()
                            try:
                                await self.heartbeat_task
                            except asyncio.CancelledError:
                                pass
                        self.heartbeat_task = asyncio.create_task(
                            self.heartbeat_sender(ws, interval)
                        )
                        if not identify_sent:
                            await self.send_identify(ws)
                            identify_sent = True
                        continue

                    if op == 0 and event_type:
                        if event_type == "READY":
                            logger.info("QQ READY 收到，News Bot 已上线")
                            if self.daily_scheduler is None:
                                self.daily_scheduler = DailyNewsScheduler(
                                    self.settings,
                                    self.pipeline,
                                    self.send_text,
                                    logger,
                                )
                                self.daily_scheduler.start()
                        elif event_type == "C2C_MESSAGE_CREATE":
                            task = asyncio.create_task(self.handle_c2c_message(data))
                            self.message_tasks.add(task)
                            task.add_done_callback(self._message_task_done)
                        continue

                elif message.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                    aiohttp.WSMsgType.CLOSED,
                ):
                    logger.warning("QQ WebSocket 已关闭 type=%r", message.type)
                    break
                elif message.type == aiohttp.WSMsgType.ERROR:
                    logger.error("QQ WebSocket 错误：%s", ws.exception())
                    break
        finally:
            if self.heartbeat_task and not self.heartbeat_task.done():
                self.heartbeat_task.cancel()
                try:
                    await self.heartbeat_task
                except asyncio.CancelledError:
                    pass
            self.heartbeat_task = None
            self.websocket = None

    async def wait_for_reconnect(self, delay: int) -> None:
        try:
            await asyncio.wait_for(self.stop_event.wait(), timeout=delay)
        except asyncio.TimeoutError:
            pass

    async def run(self) -> None:
        logger.info("News Bot 启动，project=%s", PROJECT_ROOT)
        logger.info("已读取独立配置文件 %s", ENV_FILE)
        self.http = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            trust_env=True,
        )
        trust_env, proxy_note = _proxy_config()
        logger.info(proxy_note)

        try:
            gateway = await self.get_gateway()
            retry_index = 0
            while not self.stop_event.is_set():
                try:
                    logger.info("QQ WebSocket connecting")
                    async with aiohttp.ClientSession(trust_env=trust_env) as session:
                        async with session.ws_connect(
                            gateway,
                            timeout=aiohttp.ClientTimeout(sock_connect=CONNECT_TIMEOUT),
                        ) as ws:
                            logger.info("QQ WebSocket connected")
                            retry_index = 0
                            await self.event_loop(ws)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if not self.stop_event.is_set():
                        logger.error("QQ WebSocket/API 错误：%s", exc)

                if self.stop_event.is_set():
                    break
                delay = RECONNECT_BACKOFF[min(retry_index, len(RECONNECT_BACKOFF) - 1)]
                logger.info("QQ WebSocket reconnect in %ss retry=%s", delay, retry_index)
                retry_index += 1
                await self.wait_for_reconnect(delay)
        finally:
            self.stop_event.set()
            if self.daily_scheduler is not None:
                self.daily_scheduler.shutdown()
            if self.websocket and not self.websocket.closed:
                await self.websocket.close()
            if self.message_tasks:
                await asyncio.gather(*self.message_tasks, return_exceptions=True)
            if self.http is not None:
                await self.http.aclose()
            logger.info("News Bot 已停止")

    def request_stop(self) -> None:
        if self.stop_event.is_set():
            return
        logger.info("收到停止信号")
        self.stop_event.set()
        if self.websocket and not self.websocket.closed:
            asyncio.create_task(self.websocket.close())


async def async_main(app_id: str, client_secret: str) -> None:
    bot = QQNewsBot(app_id, client_secret)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, bot.request_stop)
        except NotImplementedError:
            pass
    await bot.run()


def cli() -> int:
    setup_logging()
    try:
        app_id, client_secret = load_config()
        asyncio.run(async_main(app_id, client_secret))
        return 0
    except KeyboardInterrupt:
        logger.info("News Bot 被用户中断")
        return 0
    except Exception as exc:
        logger.exception("News Bot 启动失败：%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
