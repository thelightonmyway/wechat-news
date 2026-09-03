"""Thin QQ command routing for the News Bot."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from news.feeds import canonicalize_url, extract_doi
from news.pipeline import PAPER_CONTENT, POPULAR_CONTENT, NewsPipeline, local_date
from publisher.wechat import create_draft
from settings import PROJECT_ROOT, Settings

NEWS_COMMAND = re.compile(r"^/news\s+(\d+)(?:\s+(generate|publish))?$", re.IGNORECASE)
PAPER_COMMAND = re.compile(r"^/paper\s+(\d+)(?:\s+(generate|publish))?$", re.IGNORECASE)
PAPER_URL_COMMAND = re.compile(r"^/paperurl\s+(.+)$", re.IGNORECASE)
URL_MESSAGE = re.compile(r"^https?://[^\s<>]+$", re.IGNORECASE)
ACADEMIC_PATH = re.compile(
    r"/(?:doi|article|articles|paper|papers|publication|publications|abstract|abs|full-text)(?:/|$)",
    re.IGNORECASE,
)
NEWS_USAGE = "用法：\n/news\n/news N\n/news N generate\n/news N publish"
PAPER_USAGE = (
    "用法：\n/papers\n/papers next\n/paper N\n/paper N generate\n"
    "/paper N publish\n/paperurl <论文URL或DOI>"
)


def _direct_paper_item(value: str) -> dict[str, object] | None:
    target = value.strip()
    if URL_MESSAGE.fullmatch(target):
        parsed = urlsplit(target)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            return None
        doi = extract_doi(target)
        paper_url_hint = bool(doi or ACADEMIC_PATH.search(parsed.path))
        if not paper_url_hint:
            return None
        url = target
    else:
        doi = extract_doi(target)
        if not doi or target.rstrip(".,;)").lower() != doi:
            return None
        url = f"https://doi.org/{doi}"
    canonical = canonicalize_url(url)
    return {
        "source": "Direct paper URL",
        "url": url,
        "canonical_url": canonical or url,
        "title": "",
        "summary": "",
        "published_at": "",
        "doi": doi,
        "journal": "",
        "content_type": PAPER_CONTENT,
        "paper_url_hint": True,
        "status": "direct_paper_url",
    }


def _paper_image_summary(markdown_path: Path, images: list[dict]) -> str:
    metadata_path = markdown_path.parent / "metadata.json"
    metadata: dict = {}
    if metadata_path.is_file():
        try:
            value = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                metadata = value
        except (OSError, ValueError, TypeError):
            pass
    available = sum(
        bool(image.get("publishable"))
        and bool(image.get("local_path"))
        and Path(str(image["local_path"])).is_file()
        for image in images
    )
    body_count = len(metadata.get("body_images") or [])
    first_page_path = str((metadata.get("paper_first_page") or {}).get("local_path") or "")
    first_page = bool(first_page_path and Path(first_page_path).is_file())
    return (
        f"可用图片：{available}\n"
        f"正文使用：{body_count}\n"
        f"论文首页：{'有' if first_page else '无'}"
    )


class CommandHandler:
    def __init__(self, settings: Settings, pipeline: NewsPipeline) -> None:
        self.settings = settings
        self.pipeline = pipeline

    async def handle(self, content: str) -> str | None:
        command = content.strip()
        lowered = command.lower()
        if lowered == "/news":
            candidates = await self.pipeline.get_or_refresh(
                content_type=POPULAR_CONTENT,
            )
            return self.pipeline.format_news(candidates)
        if lowered == "/papers next":
            candidates = await self.pipeline.next_paper_batch()
            return self.pipeline.format_news(candidates)
        if lowered == "/papers":
            candidates = await self.pipeline.get_or_refresh(
                content_type=PAPER_CONTENT,
            )
            return self.pipeline.format_news(candidates)
        if lowered == "/paper":
            return PAPER_USAGE
        if lowered == "/status":
            return self.status_text()
        if lowered == "/history":
            return self.history_text()
        paper_url_match = PAPER_URL_COMMAND.fullmatch(command)
        if paper_url_match:
            return await self._handle_paper_url(paper_url_match.group(1))

        news_match = NEWS_COMMAND.fullmatch(command)
        if news_match:
            return await self._handle_candidate(
                int(news_match.group(1)),
                (news_match.group(2) or "").lower(),
                POPULAR_CONTENT,
            )
        paper_match = PAPER_COMMAND.fullmatch(command)
        if paper_match:
            return await self._handle_candidate(
                int(paper_match.group(1)),
                (paper_match.group(2) or "").lower(),
                PAPER_CONTENT,
            )
        if lowered.startswith("/news"):
            return NEWS_USAGE
        if lowered.startswith("/paper"):
            return PAPER_USAGE
        return None

    async def _handle_paper_url(self, url: str) -> str | None:
        item = _direct_paper_item(url)
        if item is None:
            return None
        doi = str(item.get("doi") or "") or "未发现"
        run_date = local_date(self.settings)
        try:
            generated = await self.pipeline.generate(
                0,
                run_date,
                PAPER_CONTENT,
                item_override=item,
                output_dir=(
                    PROJECT_ROOT
                    / "articles"
                    / PAPER_CONTENT
                    / f"{run_date}-direct-{uuid.uuid4().hex[:10]}"
                ),
            )
        except LookupError as exc:
            return f"无法识别为论文或获取 metadata 失败：{exc}"
        except Exception as exc:
            return f"PAPER 生成失败：{type(exc).__name__}: {exc}"

        dossier = generated.get("dossier") or {}
        title = str(
            dossier.get("title_cn") or dossier.get("title") or "未获取到标题"
        )
        journal = str(
            (dossier.get("openalex") or {}).get("journal")
            or dossier.get("journal")
            or "未知"
        )
        doi = str(
            dossier.get("doi")
            or (dossier.get("openalex") or {}).get("doi")
            or item.get("doi")
            or "未发现"
        )
        try:
            publish_response = await self._publish_generated_paper(
                generated,
                fallback_title=title,
                source_url=url,
            )
        except Exception as exc:
            publish_response = f"已生成 PAPER，但发布到微信草稿箱失败：{type(exc).__name__}: {exc}"
        return (
            f"识别到论文：\n《{title}》\n期刊：{journal}\nDOI：{doi}\n\n"
            "正在生成 PAPER 推文……\n\n"
            f"{publish_response}"
        )

    async def _publish_generated_paper(
        self,
        generated: dict,
        *,
        fallback_title: str = "科研解读",
        source_url: str = "",
    ) -> str:
        dossier = generated.get("dossier") or {}
        markdown_path = Path(str(generated.get("markdown_path") or ""))
        try:
            article_id = int(dossier["id"])
            result = await asyncio.to_thread(
                create_draft,
                markdown_path,
                self.settings,
                title=str(
                    dossier.get("title_cn")
                    or dossier.get("title")
                    or fallback_title
                ),
                source_url=str(dossier.get("url") or source_url),
            )
            self.pipeline.db.save_publish_history(
                article_id,
                result["status"],
                result.get("draft_media_id", ""),
                result.get("error", ""),
            )
            images = self.pipeline.db.get_images(article_id)
            paper_image_summary = _paper_image_summary(markdown_path, images)
            word_count = len(markdown_path.read_text(encoding="utf-8"))
            title = str(
                dossier.get("title_cn")
                or dossier.get("title")
                or fallback_title
            )
            if result["status"] == "dry_run":
                return (
                    f"PAPER 已生成（微信发布 dry-run）\n标题：{title}\n"
                    f"字数：{word_count}\n{paper_image_summary}\n"
                    f"HTML：{result['article_html']}\n"
                    "Blocker：未配置微信公众号凭据，未创建草稿。"
                )
            return (
                f"已生成并发布到微信草稿箱\n标题：{title}\n"
                f"字数：{word_count}\n{paper_image_summary}\n"
                f"草稿 media_id：{result['draft_media_id']}"
            )
        except Exception as exc:
            try:
                self.pipeline.db.save_publish_history(
                    int(dossier["id"]),
                    "failed",
                    error=f"{type(exc).__name__}: {exc}"[:2000],
                )
            except Exception:
                pass
            raise

    async def _handle_candidate(
        self,
        rank: int,
        action: str,
        content_type: str,
    ) -> str:
        date = local_date(self.settings)
        if not action:
            dossier = await self.pipeline.paper_details(
                rank,
                date,
                content_type,
            )
            return self.pipeline.format_paper(dossier)
        if action == "generate":
            if not self.settings.model_configured:
                return "未配置 MODEL_BASE_URL / MODEL_API_KEY / MODEL_NAME，未调用模型。"
            generated = await self.pipeline.generate(
                rank,
                date,
                content_type,
            )
            return f"已生成：{generated['markdown_path']}"
        if action == "publish" and content_type == PAPER_CONTENT:
            if not self.settings.model_configured:
                return "未配置 MODEL_BASE_URL / MODEL_API_KEY / MODEL_NAME，未调用模型。"
            try:
                generated = await self.pipeline.generate(
                    rank,
                    date,
                    content_type,
                )
            except Exception as exc:
                return f"PAPER generate 失败，未调用 publish：{type(exc).__name__}: {exc}"
            candidate = self.pipeline.db.get_candidate(date, rank, content_type)
            fallback_title = str(
                (candidate or {}).get("title_cn")
                or (candidate or {}).get("title")
                or "科研解读"
            )
            try:
                return await self._publish_generated_paper(
                    generated,
                    fallback_title=fallback_title,
                    source_url=str((candidate or {}).get("url") or ""),
                )
            except Exception as exc:
                return f"已生成 PAPER，但发布到微信草稿箱失败：{type(exc).__name__}: {exc}"

        candidate = self.pipeline.db.get_candidate(date, rank, content_type)
        list_command = "/papers" if content_type == PAPER_CONTENT else "/news"
        if not candidate:
            return f"今日没有序号 {rank}；请先执行 {list_command}。"
        post = self.pipeline.db.latest_generated_post(
            int(candidate["id"]),
            content_type,
        )
        if not post or not Path(post["markdown_path"]).is_file():
            command_name = "/paper" if content_type == PAPER_CONTENT else "/news"
            return (
                f"文章尚未 generate，请先执行对应的 {command_name} {rank} generate；"
                "未调用模型。"
            )
        try:
            result = await asyncio.to_thread(
                create_draft,
                Path(post["markdown_path"]),
                self.settings,
                title=str(candidate.get("title_cn") or candidate.get("title") or "科研解读"),
                source_url=str(candidate.get("url") or ""),
            )
            self.pipeline.db.save_publish_history(
                int(candidate["id"]),
                result["status"],
                result.get("draft_media_id", ""),
                result.get("error", ""),
            )
            images = self.pipeline.db.get_images(int(candidate["id"]))
            image_count = len([image for image in images if image.get("publishable")])
            markdown_path = Path(post["markdown_path"])
            paper_image_summary = (
                _paper_image_summary(markdown_path, images)
                if content_type == PAPER_CONTENT
                else ""
            )
            word_count = len(markdown_path.read_text(encoding="utf-8"))
            dry_run_image_summary = (
                paper_image_summary
                if content_type == PAPER_CONTENT
                else f"合法图片：{image_count}"
            )
            drafted_image_summary = (
                paper_image_summary
                if content_type == PAPER_CONTENT
                else f"图片：{image_count}"
            )
            if result["status"] == "dry_run":
                return (
                    f"微信排版 dry-run 完成\n标题：{candidate.get('title_cn') or candidate.get('title')}\n"
                    f"字数：{word_count}\n{dry_run_image_summary}\nHTML：{result['article_html']}\n"
                    "Blocker：未配置微信公众号凭据，未创建草稿。"
                )
            return (
                f"微信公众号草稿已创建\n标题：{candidate.get('title_cn') or candidate.get('title')}\n"
                f"字数：{word_count}\n{drafted_image_summary}\n草稿 media_id：{result['draft_media_id']}"
            )
        except Exception as exc:
            self.pipeline.db.save_publish_history(
                int(candidate["id"]),
                "failed",
                error=f"{type(exc).__name__}: {exc}"[:2000],
            )
            raise

    def status_text(self) -> str:
        date = local_date(self.settings)
        snapshot = self.pipeline.db.status_snapshot(date)
        return "\n".join(
            [
                "## News Bot 状态",
                "Bot：在线",
                f"今日候选：{snapshot['candidate_count']}",
                f"最近抓取：{snapshot['fetched_at'] or '尚未抓取'}",
                f"今日已推送：{'是' if snapshot['pushed_at'] else '否'}",
                f"模型：{'已配置' if self.settings.model_configured else '未配置'}",
                f"OpenAlex：{'已配置' if self.settings.openalex_configured else '未配置'}",
                f"微信：{'已配置' if self.settings.wechat_configured else '未配置'}",
                f"最近错误：{snapshot['last_error'] or '无'}",
            ]
        )

    def history_text(self) -> str:
        history = self.pipeline.db.history(10)
        if not history:
            return "暂无 generated / drafted / failed 历史。"
        lines = ["## 最近文章历史"]
        for index, item in enumerate(history, start=1):
            detail = item.get("detail") or item.get("error") or ""
            lines.append(
                f"{index}. [{item['kind']}/{item['status']}] {item['title']}\n"
                f"{item['created_at']} {detail}"
            )
        return "\n".join(lines)
