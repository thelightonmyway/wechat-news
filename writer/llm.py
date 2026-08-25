"""OpenAI-compatible text-only selection and writing adapter."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from settings import PROJECT_ROOT, Settings


def _json_from_text(text: str) -> Any:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def select_top_ten(
    candidates: list[dict[str, Any]],
    settings: Settings,
) -> tuple[list[dict[str, Any]], bool, str]:
    fallback = [dict(item, title_cn="") for item in candidates[:10]]
    if not candidates:
        return [], False, "no candidates"
    if not settings.model_configured:
        return fallback, False, "model not configured"

    payload = [
        {
            "index": index,
            "title": item.get("title", ""),
            "summary": item.get("summary", ""),
            "source": item.get("source", ""),
            "word_count": item.get("word_count", 0),
            "doi": item.get("doi", ""),
            "journal": item.get("journal", ""),
        }
        for index, item in enumerate(candidates[:20], start=1)
    ]
    client = OpenAI(
        api_key=settings.model_api_key,
        base_url=settings.model_base_url,
        timeout=90.0,
        max_retries=2,
    )
    try:
        response = client.chat.completions.create(
            model=settings.model_name,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是科研新闻标题翻译编辑。"
                        "如果候选不超过10篇，必须保留全部候选并按原index逐篇生成中文标题，不得遗漏；"
                        "只有候选超过10篇时才选择其中10篇。"
                        "生成中文标题时，以英文原标题为唯一依据，只做忠实翻译和轻微中文润色。"
                        "可以调整语序，使中文自然、简洁，新闻标题不必逐字直译；学术专业术语必须准确。"
                        "禁止根据摘要或其他元数据补充原标题没有的信息、原因、机制、对象或结论，"
                        "禁止为了吸引眼球扩大、强化或改写原文含义。"
                        "删除原标题末尾类似‘- Nature Climate Change’或‘- Eos’的网站、期刊名称。"
                        "风格示例：England’s Ancient Trees Are Dying in the Heat → 英格兰古树因高温而衰亡；"
                        "California’s Drought Irreversibly Damaged Sacramento Valley Aquifers → "
                        "加州干旱对萨克拉门托谷含水层造成不可逆损害；"
                        "Climate warming drives thermal shocks and accelerated freshwater habitat fragmentation → "
                        "气候变暖引发热冲击并加速淡水栖息地破碎化；"
                        "Why Marine Heat Waves and Acidification Strike Together → 海洋热浪为何与酸化同时发生。"
                        "返回严格JSON："
                        '{"items":[{"index":1,"title_cn":"..."}]}。'
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            ],
        )
        parsed = _json_from_text(response.choices[0].message.content or "")
        selected: list[dict[str, Any]] = []
        used: set[int] = set()
        title_by_index: dict[int, str] = {}
        for choice in parsed.get("items", []):
            index = int(choice.get("index", 0))
            title_cn = str(choice.get("title_cn") or "").strip()
            if index < 1 or index > len(candidates) or index in used or not title_cn:
                continue
            title_by_index[index] = title_cn
            used.add(index)

        if len(candidates) <= 10:
            if len(title_by_index) != len(candidates):
                raise ValueError("model returned an incomplete title batch")
            for index, candidate in enumerate(candidates, start=1):
                selected.append(dict(candidate, title_cn=title_by_index[index]))
        else:
            for index, title_cn in title_by_index.items():
                selected.append(dict(candidates[index - 1], title_cn=title_cn))
                if len(selected) == 10:
                    break
            if len(selected) != 10:
                raise ValueError("model returned an incomplete selection")
        return selected, True, ""
    except Exception as exc:
        return fallback, False, f"{type(exc).__name__}: {exc}"


def _title_related_image_context(title: str, summary: str) -> str:
    title_terms = {
        term
        for term in re.findall(r"[a-z0-9]+", title.lower())
        if len(term) >= 3
        and term
        not in {
            "the",
            "and",
            "for",
            "from",
            "that",
            "this",
            "with",
            "why",
            "how",
            "daily",
            "briefing",
        }
    }
    if not title_terms or not summary.strip():
        return ""
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|[\r\n]+", summary)
        if sentence.strip()
    ]
    scored: list[tuple[int, int, str]] = []
    for index, sentence in enumerate(sentences):
        sentence_terms = set(re.findall(r"[a-z0-9]+", sentence.lower()))
        overlap = len(title_terms & sentence_terms)
        if overlap >= 2:
            scored.append((overlap, -index, sentence))
    selected = [value[2] for value in sorted(scored, reverse=True)[:2]]
    return " ".join(selected)[:1200]


def generate_image_search_keywords(
    dossier: dict[str, Any],
    settings: Settings,
) -> list[str]:
    """Generate 3-5 short English search phrases from title-related text only."""
    if not settings.model_configured:
        return []
    client = OpenAI(
        api_key=settings.model_api_key,
        base_url=settings.model_base_url,
        timeout=60.0,
        max_retries=1,
    )
    response = client.chat.completions.create(
        model=settings.model_name,
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": (
                    "Generate 3 to 5 concise English visual-concept search phrases suitable for "
                    "Wikimedia Commons, NASA, or NOAA. The current article title is authoritative. "
                    "Use title_related_context only when it directly explains that title. Ignore any "
                    "other stories or topics from a Daily Briefing, roundup, or digest. Do not include "
                    "people, animals, events, mechanisms, or conclusions unrelated to the title. "
                    "Make the first keyword the broadest accurate core scientific visual concept, "
                    "followed by more specific concepts. Return strict JSON: "
                    '{"keywords":["phrase one","phrase two"]}. Do not analyze images.'
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "title": dossier.get("title", ""),
                        "title_related_context": _title_related_image_context(
                            str(dossier.get("title") or ""),
                            str(dossier.get("summary") or ""),
                        ),
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    )
    parsed = _json_from_text(response.choices[0].message.content or "")
    keywords = [
        " ".join(str(value).split())
        for value in parsed.get("keywords", [])
        if str(value).strip()
    ]
    return keywords[:5] if len(keywords) >= 3 else []


def generate_image_captions(
    images: list[dict[str, Any]],
    settings: Settings,
) -> list[str]:
    """Generate independent Chinese captions from each image's text metadata."""
    if not images or not settings.model_configured:
        return [""] * len(images)
    payload = [
        {
            "index": index,
            "title": image.get("metadata_title", ""),
            "caption": str(image.get("caption") or "")[:1500],
            "description": str(image.get("description") or image.get("alt") or "")[:1000],
            "provider": image.get("provider", ""),
        }
        for index, image in enumerate(images, start=1)
    ]
    client = OpenAI(
        api_key=settings.model_api_key,
        base_url=settings.model_base_url,
        timeout=60.0,
        max_retries=1,
    )
    try:
        response = client.chat.completions.create(
            model=settings.model_name,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "根据每张图片各自的文本metadata，独立生成简短、准确的中文图注。"
                        "必须描述该图片实际展示的内容，不能仅根据文章主题写通用句子，"
                        "不同图片不得复用同一句图注。不得输出credit、license、copyright、URL、"
                        "图库名称或英文长caption。metadata不足时caption_cn返回空字符串。"
                        "不要添加‘图1’等编号。返回严格JSON："
                        '{"items":[{"index":1,"caption_cn":"..."}]}。'
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        parsed = _json_from_text(response.choices[0].message.content or "")
        captions = [""] * len(images)
        used: set[str] = set()
        for item in parsed.get("items", []):
            index = int(item.get("index", 0))
            caption = str(item.get("caption_cn") or "").strip()
            caption = re.sub(r"^图\s*\d+\s*[.、：:]\s*", "", caption)
            if (
                index < 1
                or index > len(images)
                or not caption
                or not re.search(r"[一-鿿]", caption)
                or caption in used
            ):
                continue
            captions[index - 1] = caption[:80]
            used.add(caption)
        return captions
    except Exception:
        return [""] * len(images)


def _remove_generated_terminal_sections(markdown: str) -> str:
    lines = markdown.splitlines()
    output: list[str] = []
    skipping = False
    for line in lines:
        if re.match(r"^#{1,3}\s*(来源|参考文献)\s*$", line.strip(), re.IGNORECASE):
            skipping = True
            continue
        if skipping and re.match(r"^#{1,3}\s+", line.strip()):
            skipping = False
        if skipping:
            continue
        if re.match(r"^\s*来源\s*[：:].*$", line):
            continue
        output.append(line)
    return "\n".join(output).strip()


def _normalize_article_markdown(markdown: str, display_title: str) -> str:
    output: list[str] = []
    intro = False
    skip_section = False
    for line in markdown.splitlines():
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if heading:
            label = heading.group(2).strip()
            intro = False
            skip_section = False
            if heading.group(1) == "#":
                continue
            if any(
                marker in label
                for marker in ("简报中的其他", "其他科研", "其他新闻", "本期简报")
            ):
                skip_section = True
                continue
            if label == "导语":
                intro = True
                continue
            if label == "研究内容":
                continue
        if skip_section:
            continue
        if intro:
            output.append(f"> {line}" if line.strip() else ">")
        else:
            output.append(line)
    body = "\n".join(output).strip()
    return f"# {display_title}\n\n{body}".strip()


def generate_article_markdown(
    dossier: dict[str, Any],
    settings: Settings,
    output_dir: Path,
) -> tuple[Path, Path]:
    if not settings.model_configured:
        raise RuntimeError("MODEL_BASE_URL / MODEL_API_KEY / MODEL_NAME not configured")

    content_type = str(dossier.get("content_type") or "popular")
    display_title = str(
        dossier.get("title_cn") or dossier.get("title") or "科研解读"
    ).strip()
    safe_input = {
        "content_type": content_type,
        "display_title": display_title,
        "title": dossier.get("title", ""),
        "news_summary": dossier.get("summary", ""),
        "news_text": str(
            dossier.get("news_text")
            or (dossier.get("text") if content_type == "popular" else "")
            or ""
        )[:50000],
        "paper_text": str(dossier.get("text") if content_type == "paper" else "")[:50000],
        "doi": dossier.get("doi", ""),
        "journal": dossier.get("journal", ""),
        "authors": dossier.get("authors", []),
        "openalex": dossier.get("openalex", {}),
        "figure_captions": [
            {
                "caption": image.get("caption", ""),
                "credit": image.get("credit", ""),
                "license": image.get("license", ""),
            }
            for image in dossier.get("images", [])
        ],
    }
    client = OpenAI(
        api_key=settings.model_api_key,
        base_url=settings.model_base_url,
        timeout=180.0,
        max_retries=2,
    )
    response = client.chat.completions.create(
        model=settings.model_name,
        temperature=0.3,
        messages=[
            {
                "role": "system",
                "content": (
                    "根据资料写一篇约1000到2000中文字的原创中文科研解读稿，保持自然、克制的科研科普风格。"
                    "若content_type为paper，原论文metadata、abstract和论文正文优先于新闻报道。"
                    "不得逐句翻译，不得夸大，不得添加资料未支持的事实。"
                    "Markdown首行必须以‘# ’加用户数据中的display_title字段原文，不得改写标题。"
                    "导语使用Markdown引用块，不要使用‘导语’章节标题。"
                    "仅在确有对应内容时设置小节标题，不要强制生成‘研究内容’等固定空标题。"
                    "文章范围必须严格围绕当前title；如果资料来自Daily Briefing、roundup或digest，"
                    "必须忽略其中与当前title无关的其他新闻，不得写入‘其他科研进展’。"
                    "不要创建单独的‘来源’栏目，不要写‘来源：某媒体’，也不要自行生成参考文献或文章信息栏目。"
                    "不要插入图片；图片、图注和参考文献将由脚本统一附在正文末尾。"
                    "图片只可依据图注文字理解，不得声称看过或分析过图片。"
                ),
            },
            {"role": "user", "content": json.dumps(safe_input, ensure_ascii=False)},
        ],
    )
    markdown = _remove_generated_terminal_sections(
        (response.choices[0].message.content or "").strip()
    )
    markdown = _normalize_article_markdown(markdown, display_title)
    if not markdown:
        raise RuntimeError("model returned empty article")

    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / "article.md"
    metadata_path = output_dir / "metadata.json"
    markdown_path.write_text(markdown + "\n", encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "model": settings.model_name,
                "source": dossier.get("url", ""),
                "doi": dossier.get("doi", ""),
                "images": dossier.get("images", []),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return markdown_path, metadata_path


def article_output_dir(
    date: str,
    rank: int,
    content_type: str = "",
) -> Path:
    if content_type:
        return PROJECT_ROOT / "articles" / content_type / f"{date}-{rank:03d}"
    return PROJECT_ROOT / "articles" / f"{date}-{rank:03d}"
