"""APScheduler integration for the daily 07:00 push and startup catch-up."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, time
from typing import Awaitable, Callable
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from db import utc_now
from news.pipeline import NewsPipeline, content_type_for_date, local_date
from settings import Settings

PushCallback = Callable[[str, str], Awaitable[bool]]
SCHEDULE_WEEKDAYS = {0, 2, 4}  # Monday, Wednesday, Friday


def should_run_startup_catchup(now: datetime, scheduled: time, pushed_at: str = "") -> bool:
    return now.weekday() in SCHEDULE_WEEKDAYS and now.time() >= scheduled and not pushed_at


class DailyNewsScheduler:
    def __init__(
        self,
        settings: Settings,
        pipeline: NewsPipeline,
        push_callback: PushCallback,
        logger: logging.Logger,
    ) -> None:
        self.settings = settings
        self.pipeline = pipeline
        self.push_callback = push_callback
        self.logger = logger
        self.scheduler = AsyncIOScheduler(timezone=ZoneInfo(settings.daily_timezone))
        self.started = False
        self.catchup_task: asyncio.Task[None] | None = None
        hour_text, minute_text = settings.daily_push_time.split(":", 1)
        self.hour = int(hour_text)
        self.minute = int(minute_text)

    def start(self) -> None:
        if self.started:
            return
        self.scheduler.add_job(
            self.run_daily,
            CronTrigger(
                day_of_week="mon,wed,fri",
                hour=self.hour,
                minute=self.minute,
                timezone=ZoneInfo(self.settings.daily_timezone),
            ),
            id="daily-news-push",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
            misfire_grace_time=3600,
        )
        self.scheduler.start()
        self.started = True
        self.catchup_task = asyncio.create_task(self.run_startup_catchup())
        self.logger.info(
            "Weekly scheduler enabled Mon/Wed/Fri at %s %s",
            self.settings.daily_push_time,
            self.settings.daily_timezone,
        )

    async def run_startup_catchup(self) -> None:
        zone = ZoneInfo(self.settings.daily_timezone)
        now = datetime.now(zone)
        scheduled = time(self.hour, self.minute)
        date = local_date(self.settings, now)
        content_type = content_type_for_date(date)
        run = self.pipeline.db.get_daily_run(date, content_type)
        if should_run_startup_catchup(
            now,
            scheduled,
            str((run or {}).get("pushed_at") or ""),
        ):
            self.logger.info("Weekly startup catch-up triggered")
            await self.run_daily()

    async def run_daily(self) -> None:
        date = local_date(self.settings)
        content_type = content_type_for_date(date)
        existing = self.pipeline.db.get_daily_run(date, content_type)
        if existing and existing.get("pushed_at"):
            self.logger.info("Daily push skipped: already pushed date=%s", date)
            return
        try:
            candidates = await self.pipeline.get_or_refresh(date, content_type)
            if not self.settings.qq_target_openid:
                self.pipeline.db.set_daily_run(
                    date,
                    status="blocked_no_target",
                    error="QQ_TARGET_OPENID not configured",
                    candidate_count=len(candidates),
                    content_type=content_type,
                )
                self.logger.warning("Daily push blocked: QQ_TARGET_OPENID not configured")
                return
            ok = await self.push_callback(
                self.settings.qq_target_openid,
                self.pipeline.format_news(candidates),
            )
            if not ok:
                raise RuntimeError("QQ daily push API failed")
            self.pipeline.db.set_daily_run(
                date,
                pushed_at=utc_now(),
                candidate_count=len(candidates),
                content_type=content_type,
                status="pushed",
                error="",
            )
            self.logger.info("Daily push completed date=%s", date)
        except Exception as exc:
            self.pipeline.db.set_daily_run(
                date,
                content_type=content_type,
                status="failed",
                error=f"{type(exc).__name__}: {exc}"[:2000],
            )
            self.logger.error("Daily job failed: %s", exc)

    def shutdown(self) -> None:
        if self.catchup_task and not self.catchup_task.done():
            self.catchup_task.cancel()
        self.catchup_task = None
        if self.started:
            self.scheduler.shutdown(wait=False)
            self.started = False
