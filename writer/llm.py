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
                        "逐篇判断候选是否真正涉及近地面风、风能、大气环流、边界层、陆气相互作用、"
                        "海气相互作用、检测归因、相关观测、极端天气气候机制、水汽降水机制或极地臭氧过程。"
                        "泛泛的气候模型评估、CMIP基准测试、干旱生态、海冰生态或生产力研究不得仅因宽泛关键词入选。"
                        "最多返回10篇，可以少于10篇；即使候选不足10篇也必须剔除不相关项，禁止凑数。"
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

        for index in sorted(title_by_index):
            selected.append(dict(candidates[index - 1], title_cn=title_by_index[index]))
            if len(selected) == 10:
                break
        return selected, True, ""
    except Exception as exc:
        return fallback, False, f"{type(exc).__name__}: {exc}"


def select_paper_top_ten(
    candidates: list[dict[str, Any]],
    settings: Settings,
) -> tuple[list[dict[str, Any]], bool, str]:
    fallback = [
        dict(item, title_cn=str(item.get("title_cn") or ""))
        for item in candidates
        if int(item.get("paper_local_score") or 0) >= 2
    ][:10]
    if not candidates:
        return [], False, "no candidates"
    if not settings.model_configured:
        return fallback, False, "model not configured"

    payload = [
        {
            "index": index,
            "title": item.get("title", ""),
            "abstract": str(item.get("summary") or "")[:8000],
            "publication_date": item.get("published_at", ""),
            "journal": item.get("journal", ""),
            "doi": item.get("doi", ""),
            "type": item.get("work_type", ""),
        }
        for index, item in enumerate(candidates[:30], start=1)
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
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是物理气候科学论文筛选编辑。根据每篇论文的title和abstract判断其主要科学问题，"
                        "而不是按单个关键词命中。评分：3=核心相关，2=明确相关，1=外围相关，0=无关。"
                        "优先保留风和风能、大气环流与遥相关、ENSO/NAO/SAM、极涡和层结耦合、臭氧气候、"
                        "温度热浪、降水水汽、干旱气候机制、边界层陆气、海气相互作用、极地海冰气候动力学、"
                        "气候变率可预测性、检测归因、物理气候模式评估，以及与气候机制直接相关的再分析或观测。"
                        "热带气旋、storm或extreme weather只有在主要问题明确连接气候变率/变化、ENSO/季风/遥相关、"
                        "大尺度环流、长期趋势、归因、投影或次季节/季节可预报性时才可评为2或3；纯对流动力学、"
                        "快速增强机制本身、眼墙、微物理、中尺度/风暴尺度动力学或无气候尺度联系的单次天气过程评为0。"
                        "排除没有气候机制的水文/大地测量、生态植被、生物地球化学或"
                        "海洋化学、泛环境变化、通用模型/软件benchmark，以及Reply、Correction、Editorial、"
                        "Comment、Correspondence。只返回2或3分论文，3分优先；最多10篇，允许少于10篇，禁止凑数。"
                        "title_cn必须忠实翻译英文标题，不得添加原题没有的信息。返回严格JSON："
                        '{"items":[{"index":1,"score":3,"title_cn":"...","reason":"..."}]}。'
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        parsed = _json_from_text(response.choices[0].message.content or "")
        selected: list[dict[str, Any]] = []
        seen: set[int] = set()
        scored: list[tuple[int, int, str]] = []
        for choice in parsed.get("items", []):
            index = int(choice.get("index", 0))
            score = int(choice.get("score", 0))
            title_cn = str(choice.get("title_cn") or "").strip()
            if (
                index < 1
                or index > len(payload)
                or index in seen
                or score not in {2, 3}
                or not title_cn
            ):
                continue
            scored.append((score, index, title_cn))
            seen.add(index)
        for score, index, title_cn in sorted(scored, key=lambda value: (-value[0], value[1]))[:10]:
            selected.append(
                dict(
                    candidates[index - 1],
                    title_cn=title_cn,
                    paper_relevance_score=score,
                )
            )
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


NEWS_ARTICLE_PROMPT = (
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
)

PAPER_STYLE_GUIDE = (
    "根据提供的论文metadata、abstract和正文材料，写一篇约800到1200个中文字符的中文论文解读。"
    "这是短篇解读，绝对不要扩写成长篇综述。正文最多3到4个小节，每段尽量短，一段只讲一个主要信息点。"
    "不要平均展开背景、方法、结果和讨论，优先解释论文最重要的2到4个发现、机制或判断。"
    "先理解论文，再用自然中文重新组织，禁止逐句翻译英文摘要或正文。整体应像中文科研作者给同行或研究生"
    "讲一篇新论文，而不是机翻摘要。多用短句和中等长度句，避免英文式长定语、多层从句和被动句。"
    "能直接使用动词时，不使用‘进行、开展、实现、予以’等空动词。少用‘首先、其次、此外、值得注意的是、"
    "综上所述’等模板连接词。避免‘通过对……进行分析……’、‘在……背景下……’、"
    "‘这一发现不仅……而且……’、‘从……的角度来看……’、‘研究结果表明了……’等机翻或模板句式。"
    "专业术语、数值、趋势、时间范围、模型、数据集、因果关系、机制解释和结论必须忠于输入论文材料。"
    "材料没有明确支持时不要补充常识性机制，不写空洞拔高，不添加原论文没有支持的意义。"
    "正文可以加入1到2处关键英文原文短引，但每处必须逐字复制自用户提供的论文材料，绝对不能自行生成、"
    "改写或拼接后冒充原文。每处最多约25个英文词，只选真正支撑关键结论的句子；没有合适原文时宁可不引用。"
    "引用格式为Markdown引用块：> 原文：“……”；引用后紧接自然中文解释，不要大段复制论文。"
    "结构采用简短开场，然后围绕关键发现或机制设置2到4个有信息量的小标题，最后用一小段说明真正重要的"
    "科学含义。不要固定写成‘研究背景/研究方法/研究结果/研究意义’，不要在正文重复标题或文章信息。"
    "Markdown首行仍必须以‘# ’加用户数据中的display_title字段原文，之后不得再次重复标题。"
    "不要创建来源、参考文献或文章信息栏目，不要自行插入图片；图片、图注和文章信息由现有pipeline处理。"
    "图片只可依据图注文字理解，不得声称看过或分析过图片。"
)


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
    system_prompt = (
        PAPER_STYLE_GUIDE if content_type == "paper" else NEWS_ARTICLE_PROMPT
    )
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
                "content": system_prompt,
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
