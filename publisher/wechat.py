"""Wrapper around the fixed xiaohu-wechat-format external tool."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from settings import PROJECT_ROOT, Settings

TOOL_DIR = PROJECT_ROOT / "vendor" / "xiaohu-wechat-format"
FORMAT_SCRIPT = TOOL_DIR / "scripts" / "format.py"
PUBLISH_SCRIPT = TOOL_DIR / "scripts" / "publish.py"
TOOL_CONFIG = TOOL_DIR / "config.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "wechat-format"
DEFAULT_COVER = PROJECT_ROOT / "assets" / "default-cover.jpg"
QIHAI_HEADER = PROJECT_ROOT / "assets" / "qihai-header.png"


def _article_metadata(markdown_path: Path) -> dict[str, Any]:
    metadata_path = markdown_path.parent / "metadata.json"
    if not metadata_path.is_file():
        return {}
    try:
        value = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _paper_draft_title(metadata: dict[str, Any], fallback_title: str) -> str:
    article_title = str(metadata.get("title_cn") or fallback_title or "科研解读").strip()
    journal = str(metadata.get("journal") or "").strip()
    if not journal:
        return article_title[:64]
    preferred = f"{journal}：{article_title}"
    if len(preferred) <= 64:
        return preferred
    available = 64 - len(journal) - 1
    if available > 0:
        return f"{journal}：{article_title[:available]}"
    return journal[:64]


def ensure_tool_config(settings: Settings) -> None:
    if not FORMAT_SCRIPT.is_file() or not PUBLISH_SCRIPT.is_file():
        raise RuntimeError(f"xiaohu-wechat-format not installed at {TOOL_DIR}")
    config = {
        "output_dir": str(OUTPUT_DIR),
        "vault_root": str(PROJECT_ROOT),
        "image_search_paths": [str(PROJECT_ROOT / "articles")],
        "settings": {
            "default_theme": "qihai",
            "auto_open_browser": False,
            "header_author_label": settings.wechat_author,
        },
        "wechat": {
            "app_id": settings.wechat_app_id,
            "app_secret": settings.wechat_app_secret,
            "author": settings.wechat_author,
        },
        "cover": {"output_dir": str(PROJECT_ROOT / "assets"), "image_generation_script": ""},
        "smart_api": {},
    }
    TOOL_CONFIG.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    os.chmod(TOOL_CONFIG, 0o600)


def _selected_cover_path(markdown_path: Path) -> Path:
    metadata = _article_metadata(markdown_path)
    if metadata.get("content_type") == "paper":
        local_path = str((metadata.get("wechat_cover") or {}).get("local_path") or "")
        if local_path and Path(local_path).is_file():
            return Path(local_path)
    local_path = str((metadata.get("cover_image") or {}).get("local_path") or "")
    if local_path and Path(local_path).is_file():
        return Path(local_path)
    return DEFAULT_COVER


def format_markdown(
    markdown_path: Path,
    settings: Settings,
    theme: str = "qihai",
) -> dict[str, Any]:
    ensure_tool_config(settings)
    metadata = _article_metadata(markdown_path)
    paper_body = metadata.get("content_type") == "paper"
    output_base = markdown_path.parent / "wechat-format"
    source_markdown = markdown_path
    temporary_markdown: Path | None = None
    brand_header = QIHAI_HEADER.is_file()
    if brand_header:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".md",
            prefix="qihai-",
            dir=markdown_path.parent,
            delete=False,
        ) as handle:
            handle.write(f"![气海无涯]({QIHAI_HEADER})\n\n")
            handle.write(markdown_path.read_text(encoding="utf-8"))
            temporary_markdown = Path(handle.name)
            source_markdown = temporary_markdown

    command = [
        sys.executable,
        str(FORMAT_SCRIPT),
        "--input",
        str(source_markdown),
        "--theme",
        theme,
        "--vault-root",
        str(PROJECT_ROOT),
        "--output",
        str(output_base),
        "--no-open",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    finally:
        if temporary_markdown is not None:
            temporary_markdown.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"wechat formatting failed: {(result.stderr or result.stdout)[-1000:]}")
    formatted_dir = output_base / source_markdown.stem
    article_html = formatted_dir / "article.html"
    if not article_html.is_file():
        raise RuntimeError("wechat formatter did not create article.html")
    if paper_body:
        html = article_html.read_text(encoding="utf-8")
        html = re.sub(
            r"<h1\b[^>]*>.*?</h1>\s*",
            "",
            html,
            count=1,
            flags=re.IGNORECASE | re.DOTALL,
        )
        article_html.write_text(html, encoding="utf-8")
    return {
        "status": "formatted",
        "formatted_dir": str(formatted_dir),
        "article_html": str(article_html),
        "stdout": result.stdout[-1000:],
        "theme": theme,
        "brand_header": brand_header,
        "brand_header_path": str(QIHAI_HEADER) if brand_header else "",
    }


def create_draft(
    markdown_path: Path,
    settings: Settings,
    *,
    title: str,
    source_url: str,
    theme: str = "qihai",
) -> dict[str, Any]:
    formatted = format_markdown(markdown_path, settings, theme)
    metadata = _article_metadata(markdown_path)
    paper_body = metadata.get("content_type") == "paper"
    draft_title = _paper_draft_title(metadata, title) if paper_body else title[:64]
    if paper_body and len(draft_title) > 64:
        raise ValueError(
            f"paper draft title exceeds WeChat 64-character limit: {len(draft_title)}"
        )
    if not settings.wechat_configured:
        return {
            **formatted,
            "status": "dry_run",
            "draft_title": draft_title,
            "draft_media_id": "",
            "error": "WECHAT_APP_ID / WECHAT_APP_SECRET not configured",
        }
    cover_path = _selected_cover_path(markdown_path)
    if not cover_path.is_file():
        raise RuntimeError(f"cover image missing: {cover_path}")

    command = [
        sys.executable,
        str(PUBLISH_SCRIPT),
        "--dir",
        formatted["formatted_dir"],
        "--cover",
        str(cover_path),
        "--title",
        draft_title,
        "--author",
        settings.wechat_author or "",
        "--source-url",
        source_url,
        "--yes",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    output = f"{result.stdout}\n{result.stderr}"
    if result.returncode != 0:
        raise RuntimeError(f"wechat draft failed: {output[-1200:]}")
    if formatted.get("brand_header") and QIHAI_HEADER.name not in output:
        raise RuntimeError("qihai header was not uploaded into draft content")
    match = re.search(r"草稿 media_id:\s*(\S+)", output)
    if not match:
        raise RuntimeError("wechat publisher completed without a draft media_id")
    return {
        **formatted,
        "status": "drafted",
        "draft_title": draft_title,
        "draft_media_id": match.group(1),
        "cover_image_path": str(cover_path),
        "error": "",
    }
