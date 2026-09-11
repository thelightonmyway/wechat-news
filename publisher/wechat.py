"""Wrapper around the fixed xiaohu-wechat-format external tool."""

from __future__ import annotations

import html
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
        paper_first_page = metadata.get("paper_first_page") or {}
        for key in ("local_path", "wechat_cover_path"):
            local_path = str(paper_first_page.get(key) or "")
            if local_path and Path(local_path).is_file():
                return Path(local_path)
        return DEFAULT_COVER
    local_path = str((metadata.get("cover_image") or {}).get("local_path") or "")
    if local_path and Path(local_path).is_file():
        return Path(local_path)
    return DEFAULT_COVER


def _style_paper_intro(html: str) -> str:
    first_page = re.search(
        r'<img\b(?=[^>]*\balt="论文第一页")[^>]*>',
        html,
        flags=re.IGNORECASE,
    )
    if first_page is None:
        return html
    tail = html[first_page.end() :]
    first_section = re.search(r"<h2\b", tail, flags=re.IGNORECASE)
    intro_tail = tail[: first_section.start()] if first_section else tail
    candidate = re.search(
        r"<p\b[^>]*>.*?</p>|<section\b[^>]*data-role=[\"']blockquote[\"'][^>]*>.*?</section>",
        intro_tail,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if candidate is None or "<img" in candidate.group(0).lower():
        return html
    intro_style = (
        "background:#f3f4f6;padding:14px 16px;margin:20px 0 28px;"
        "border-left:4px solid #cbd5e1;border-radius:6px;"
    )
    candidate_html = candidate.group(0)
    if candidate_html.lstrip().lower().startswith("<section"):
        opening = re.match(r"<section\b[^>]*>", candidate_html, flags=re.IGNORECASE)
        if opening is None:
            return html
        opening_tag = opening.group(0)
        opening_tag = re.sub(
            r'data-role=["\']blockquote["\']',
            'data-role="paper-intro"',
            opening_tag,
            count=1,
            flags=re.IGNORECASE,
        )
        if "data-role=\"paper-intro\"" not in opening_tag:
            opening_tag = opening_tag[:-1] + ' data-role="paper-intro">'
        if re.search(r"\bstyle=[\"']", opening_tag, flags=re.IGNORECASE):
            opening_tag = re.sub(
                r"\bstyle=[\"'][^\"']*[\"']",
                f'style="{intro_style}"',
                opening_tag,
                count=1,
                flags=re.IGNORECASE,
            )
        else:
            opening_tag = opening_tag[:-1] + f' style="{intro_style}">'
        if "data-darkmode-bgcolor" not in opening_tag.lower():
            opening_tag = opening_tag[:-1] + ' data-darkmode-bgcolor="#2a2f36">'
        styled = opening_tag + candidate_html[opening.end() :]
    else:
        styled = (
            f'<section data-role="paper-intro" style="{intro_style}" '
            'data-darkmode-bgcolor="#2a2f36">'
            f"{candidate_html}</section>"
        )
    return html[: first_page.end()] + tail[: candidate.start()] + styled + tail[candidate.end() :]


def _style_paper_figure_blocks(html: str) -> str:
    """Give PAPER figure captions a quiet, mobile-friendly rhythm."""
    caption_style = (
        "font-size:12px;color:#888;line-height:1.6;text-align:center;"
        "margin:8px 0 20px;"
    )

    def replace_figure(match: re.Match[str]) -> str:
        caption = match.group("caption")
        if re.search(
            r"\bdata-role=[\"']paper-figure-caption[\"']",
            caption,
            flags=re.IGNORECASE,
        ):
            return match.group(0)
        opening = re.match(r"<p\b[^>]*>", caption, flags=re.IGNORECASE)
        if opening is None:
            return match.group(0)
        styled_opening = (
            f'<p data-role="paper-figure-caption" style="{caption_style}">'
        )
        return (
            f"{match.group('prefix')}"
            f"{styled_opening}{caption[opening.end():]}"
            f"{match.group('suffix')}"
        )

    return re.sub(
        r"(?P<prefix><section\b[^>]*data-role=[\"']img-wrapper[\"'][^>]*>.*?)"
        r"(?P<caption><p\b[^>]*>.*?</p>)"
        r"(?P<suffix>.*?</section>)",
        replace_figure,
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )


def _style_paper_visual_emphasis(article_html: str, metadata: dict[str, Any]) -> str:
    """Apply validated semantic emphasis without changing canonical article Markdown."""
    visual = metadata.get("paper_visual_emphasis") or {}
    if not isinstance(visual, dict):
        return article_html
    selections: list[tuple[str, str]] = []
    for item in visual.get("emphasis") or []:
        if isinstance(item, dict):
            text = str(item.get("text") or "").strip()
            role = str(item.get("role") or "").strip()
            if text and role in {"key_claim", "key_number"}:
                selections.append((text, role))
    for item in visual.get("takeaway") or []:
        if isinstance(item, dict):
            text = str(item.get("text") or "").strip()
            if text:
                selections.append((text, "takeaway"))
    if not selections:
        return article_html

    tokens = [match.group(0) for match in re.finditer(r"<[^>]+>|[^<]+", article_html, re.DOTALL)]
    stack: list[tuple[str, str]] = []
    paragraphs: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for index, token in enumerate(tokens):
        if not token.startswith("<"):
            if current is not None:
                current["text_indices"].append(index)
                if any(role.startswith("paper-") for _, role in stack):
                    current.setdefault("visual_text_indices", set()).add(index)
            continue
        closing = re.match(r"</\s*([A-Za-z0-9:_-]+)", token)
        opening = re.match(r"<\s*([A-Za-z0-9:_-]+)\b", token)
        if closing:
            if closing.group(1).lower() == "p" and current is not None:
                current["end_index"] = index
                current = None
            for stack_index in range(len(stack) - 1, -1, -1):
                if stack[stack_index][0] == closing.group(1).lower():
                    del stack[stack_index:]
                    break
            continue
        if not opening:
            continue
        tag = opening.group(1).lower()
        if tag == "p":
            current = {
                "opening_index": index,
                "text_indices": [],
                "roles": list(stack),
                "caption": bool(re.search(r'data-role=["\']paper-figure-caption', token, re.IGNORECASE)),
            }
            paragraphs.append(current)
        if not token.rstrip().endswith("/>"):
            role = re.search(r'data-role=["\']([^"\']+)', token, re.IGNORECASE)
            stack.append((tag, role.group(1).lower() if role else ""))

    def eligible(paragraph: dict[str, Any]) -> bool:
        if paragraph.get("caption"):
            return False
        roles = {role for _, role in paragraph.get("roles") or []}
        return not roles.intersection({"paper-intro", "img-wrapper"}) and not any(
            role.startswith("paper-") for _, role in paragraph.get("roles") or []
        )

    replacements: dict[int, list[tuple[int, int, str]]] = {}
    takeaway_openings: set[int] = set()
    for text, role in selections:
        literal = html.escape(text, quote=False)
        matches: list[tuple[dict[str, Any], int, int, int]] = []
        for paragraph in paragraphs:
            if not eligible(paragraph):
                continue
            for token_index in paragraph.get("text_indices") or []:
                if token_index in paragraph.get("visual_text_indices", set()):
                    continue
                token = tokens[token_index]
                if token.count(literal) == 1:
                    start = token.find(literal)
                    matches.append((paragraph, token_index, start, start + len(literal)))
        if len(matches) != 1:
            continue
        paragraph, token_index, start, end = matches[0]
        if role == "takeaway":
            takeaway_openings.add(int(paragraph["opening_index"]))
            continue
        color = "#1677FF" if role == "key_claim" else "#0B5FD7"
        tag = f'<strong data-role="paper-{role}" style="font-weight:700;color:{color};">'
        replacements.setdefault(token_index, []).append((start, end, f"{tag}{literal}</strong>"))

    for token_index, ranges in replacements.items():
        value = tokens[token_index]
        for start, end, replacement in sorted(ranges, reverse=True):
            value = value[:start] + replacement + value[end:]
        tokens[token_index] = value
    takeaway_style = (
        "background:rgba(22,119,255,0.06);border-left:3px solid #1677FF;"
        "padding:8px 12px;margin:12px 0 20px;text-indent:0;"
    )
    for index in takeaway_openings:
        opening = tokens[index]
        if "data-role=\"paper-takeaway\"" in opening:
            continue
        if re.search(r'\bstyle=["\']', opening, re.IGNORECASE):
            opening = re.sub(
                r'\bstyle=["\']([^"\']*)["\']',
                lambda match: f'style="{match.group(1)};{takeaway_style}"',
                opening,
                count=1,
                flags=re.IGNORECASE,
            )
        else:
            opening = opening[:-1] + f' style="{takeaway_style}">'
        opening = opening[:-1] + ' data-role="paper-takeaway">'
        tokens[index] = opening
    return "".join(tokens)


def _remove_paper_figure_attributions(html: str) -> str:
    label = r"(?:图源|图片来源|Source)\s*[:：]"
    html = re.sub(
        rf'<p\b[^>]*>\s*<em\b[^>]*>\s*{label}.*?</em>\s*</p>',
        "",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    html = re.sub(
        rf'<p\b[^>]*>\s*{label}.*?</p>',
        "",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    html = re.sub(
        rf'(?:<br\s*/?>\s*)?<em\b[^>]*>\s*{label}.*?(?:</em>|</p>)',
        "",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return html


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
        html = _style_paper_intro(html)
        html = _style_paper_figure_blocks(html)
        html = _style_paper_visual_emphasis(html, metadata)
        html = _remove_paper_figure_attributions(html)
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
    ]
    command.extend([
        "--title",
        draft_title,
        "--author",
        settings.wechat_author or "",
        "--source-url",
        source_url,
        "--yes",
    ])
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
