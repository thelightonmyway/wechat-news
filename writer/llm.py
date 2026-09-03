"""OpenAI-compatible text-only selection and writing adapter."""

from __future__ import annotations

import json
import re
import time
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


def _is_transient_server_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
    if status_code in {502, 503}:
        return True
    message = str(exc).lower()
    return bool(re.search(r"\b(?:502|503)\b", message))


def _paper_completion_with_retry(client: OpenAI, **kwargs: Any) -> Any:
    for attempt in range(2):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            if attempt == 0 and _is_transient_server_error(exc):
                time.sleep(3)
                continue
            raise
    raise RuntimeError("unreachable paper completion retry state")


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
        max_retries=0,
    )
    try:
        response = _paper_completion_with_retry(
            client,
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
                        "只返回筛选结果，不负责中文标题翻译。返回严格JSON："
                        '{"items":[{"index":1,"score":3,"reason":"..."}]}。'
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
            ):
                continue
            scored.append((score, index, title_cn))
            seen.add(index)
        for score, index, title_cn in sorted(scored, key=lambda value: (-value[0], value[1]))[:10]:
            selected.append(
                dict(
                    candidates[index - 1],
                    title_cn=title_cn or str(candidates[index - 1].get("title_cn") or ""),
                    paper_relevance_score=score,
                )
            )
        return selected, True, ""
    except Exception as exc:
        return fallback, False, f"{type(exc).__name__}: {exc}"


def translate_paper_titles(
    candidates: list[dict[str, Any]],
    settings: Settings,
) -> tuple[list[str], bool, str]:
    """Translate missing PAPER titles without changing candidate selection."""
    titles = [str(item.get("title_cn") or "").strip() for item in candidates]
    missing = [
        (index, str(item.get("title") or "").strip())
        for index, item in enumerate(candidates)
        if not titles[index] and str(item.get("title") or "").strip()
    ]
    if not missing:
        return titles, True, ""
    if not settings.model_configured:
        return titles, False, "model not configured"

    payload = [
        {"index": index + 1, "title": title}
        for index, title in missing
    ]
    client = OpenAI(
        api_key=settings.model_api_key,
        base_url=settings.model_base_url,
        timeout=90.0,
        max_retries=0,
    )
    try:
        response = _paper_completion_with_retry(
            client,
            model=settings.model_name,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是中文科技论文标题翻译编辑。只根据给出的英文原标题做忠实、简洁的中文翻译，"
                        "标题应完整表达原意，适合微信公众号显示，尽量控制在45个汉字以内，禁止使用省略号或半截标题；"
                        "不得扩写、解释、补充原标题没有的信息，也不要改变论文顺序。返回严格JSON："
                        '{"items":[{"index":1,"title_cn":"..."}]}。'
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        parsed = _json_from_text(response.choices[0].message.content or "")
        translated: dict[int, str] = {}
        valid_indexes = {index + 1 for index, _ in missing}
        for choice in parsed.get("items", []):
            index = int(choice.get("index", 0))
            title_cn = str(choice.get("title_cn") or "").strip()
            if index in valid_indexes and title_cn:
                translated[index] = title_cn
        for index, _ in missing:
            titles[index] = translated.get(index + 1, "")
        return titles, True, ""
    except Exception as exc:
        return titles, False, f"{type(exc).__name__}: {exc}"


def translate_paper_abstract(abstract: str, settings: Settings) -> str:
    """Translate a paper Abstract without adding claims or restructuring its findings."""
    source = re.sub(r"\s+", " ", str(abstract or "")).strip()
    if not source:
        return ""
    if not settings.model_configured:
        raise RuntimeError("MODEL_BASE_URL / MODEL_API_KEY / MODEL_NAME not configured")
    client = OpenAI(
        api_key=settings.model_api_key,
        base_url=settings.model_base_url,
        timeout=120.0,
        max_retries=0,
    )
    response = _paper_completion_with_retry(
        client,
        model=settings.model_name,
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是中文科技论文摘要翻译编辑。只根据用户提供的原始Abstract做忠实、自然的中文翻译。"
                    "短Abstract基本完整翻译；长Abstract只能删除次要细节，不能增加原文没有的结论、分类、机制或表述，"
                    "也不能自行重组科学结论。不要添加小标题、列表或解释，只返回严格JSON："
                    '{"abstract_cn":"..."}'
                ),
            },
            {"role": "user", "content": json.dumps({"abstract": source}, ensure_ascii=False)},
        ],
    )
    parsed = _json_from_text(response.choices[0].message.content or "")
    translated = re.sub(r"\s+", " ", str(parsed.get("abstract_cn") or "")).strip()
    if not translated:
        raise RuntimeError("model returned empty Chinese Abstract translation")
    return translated


def _replace_paper_lead(markdown: str, abstract_lead: str) -> str:
    lines = markdown.splitlines()
    section_index = next(
        (
            index
            for index, line in enumerate(lines)
            if re.match(r"^##\s+", line.strip())
        ),
        None,
    )
    title = lines[0] if lines and lines[0].startswith("# ") else ""
    if section_index is None:
        body = "\n".join(lines[1:]).strip() if title else markdown.strip()
        return f"{title}\n\n{abstract_lead}\n\n{body}".strip()
    sections = "\n".join(lines[section_index:]).strip()
    return f"{title}\n\n{abstract_lead}\n\n{sections}".strip()


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

PAPER_STYLE_EXAMPLE = (
    "结构示例：\n"
    "作者比较了三组模式试验。三组结果的变化方向基本一致，但幅度并不相同，其中 A 试验最强，B 试验相对较弱。"
    "差异主要出现在事件后的几个月，随后逐渐减小。\n\n"
    "论文进一步给出了敏感性试验。去掉 Z 过程后，Y 的响应明显减弱，作者据此认为，Z 是造成这组差异的重要因素。\n\n"
    "不同区域的结果也有明显差别。A 区的变化最突出，B 区相对较弱，而且这种差异具有一定的季节性，"
    "并不是全年都保持相同强度。\n"
    "只学习以上示例的句长、段落节奏、信息密度和自然推进方式，不复制具体措辞，也不把它当作固定模板。"
    "A、B、X、Y、Z 都只是占位符，绝不能进入实际文章。"
)


PAPER_STYLE_GUIDE = (
    "根据提供的论文metadata、原始abstract和正文材料，写一篇正文主体优先约650到800个中文字符的中文论文解读。"
    "这是硬性篇幅要求：标题、英文摘录、图片图注和文章信息不计入正文主体；返回前必须把正文主体压缩到650到800个中文字以内。"
    "这是短篇高信息密度解读，不要扩写成长篇综述；通常设置3到4个主要小节，每个小节约110到160个中文字，任何小节不得超过180字。每个小节只写紧凑的核心内容，不要把未来意义、限制和背景重复堆在最后一节。"
    "摘要和正文导语必须优先忠实翻译输入的原始abstract：短摘要基本完整翻译，长摘要只可删除次要细节，不得增加abstract没有的结论、分类、机制或表述，也不得自行重组科学结论；中文应自然但保持原意。若abstract为空或不可用，才可用paper_text写出有据可查的简短fallback导语。"
    "先以原始abstract的核心结果结构作为全文最高优先级提纲，正文必须覆盖abstract明确写出的主要发现；Results或paper_text只用于补充这些结论的证据、机制和数据，不能取代abstract决定的文章主线。"
    "只有当abstract明确写出two modes、first mode/second mode、two regimes、two mechanisms或同等清楚的两部分结构时，才分别覆盖对应部分并避免遗漏；如果abstract没有明确这种结构，绝对不要自行创造第一模态、第二模态、第一类、第二类或其他类似分类。"
    "对于abstract明确的每个核心mode、mechanism或regime，使用Results或paper_text补充原文支持的空间或对象特征、主要驱动因子和关键物理机制及数据；材料没有明确支持的内容不要补写。完整覆盖核心结果优先于机械保持固定section数量，section标题和正文组织应跟随论文实际科学主线，不套固定模板。"
    "优先保留研究问题、核心结果、关键机制和研究意义，主动删去冗余背景、重复解释、低价值细节和不影响结论的过程描述。"
    "不要机械截断句子或为了凑字数罗列术语，而要在生成阶段压缩表达、合并重复信息，让每段承担一个明确功能。"
    "重点呈现论文最重要的2到4个发现，不追求覆盖论文全部背景、方法、结果和讨论。"
    "文风应像中文科技媒体编辑或科研作者整理一篇刚发表的研究：直接陈述研究发现、数据和作者判断，"
    "让研究逻辑自然推进；专业准确，但不是论文摘要，也不要扮演老师给读者讲课。"
    "研究结果、关键数据和作者判断可以自然连续推进，不要求每个结果后另加解释句或总结句。"
    "不要连续多句只罗列模式名、变量、数值、数据集或术语；先理解论文，再用自然中文重新组织，"
    "禁止逐句翻译英文摘要或正文。多用短句和中等长度句，避免英文式长定语、多层从句和被动表达。"
    "所有科学事实必须来自输入论文材料。专业术语、数值、趋势、时间范围、模型、数据集、因果关系、"
    "机制解释和结论必须忠于材料；材料没有明确支持时，不补充机制，不拔高意义。"
    "如果paper_text中有适合支撑关键结论的句子，优先加入2到3组英文短引；每组保留1到2个连续且有信息量、语境完整的原文句子，"
    "不要按固定的全篇英文词数机械截断；每组应保持精炼，通常不超过80个英文词，避免复制整段论文。"
    "每处引用必须逐字复制自paper_text，绝对不能自行生成、改写或拼接；找不到合适原文就少引或不引。"
    "引用格式为自然的Markdown引用块：> “……”；不要添加‘原文：’、‘英文原文：’或‘Original text:’标签。"
    "引用可以自然嵌在相关中文段落之间，引用后可以直接继续正常叙述，不强制另写解释句，也不要大段复制论文。"
    "结构采用一段独立的中文导语开场，必须在第一个##小节之前概括全文核心发现；导语不是第一个小节的正文，不能把小节首段当作摘要。"
    "然后围绕关键发现或机制设置3到4个有信息量的小标题。"
    "不要固定写成‘研究背景/研究方法/研究结果/研究意义’，不要在正文重复标题或文章信息。"
    "Markdown首行仍必须以‘# ’加用户数据中的display_title字段原文，之后不得再次重复标题。"
    "不要创建来源、参考文献或文章信息栏目，不要自行插入图片；图片、图注和文章信息由现有pipeline处理。"
    "图片只可依据图注文字理解，不得声称看过或分析过图片。\n\n"
    + PAPER_STYLE_EXAMPLE
)


def _remove_unverified_paper_quotes(markdown: str, paper_text: str) -> str:
    """Drop or normalize English blockquotes that are not verbatim paper text."""
    normalized_source = re.sub(r"\s+", " ", paper_text).strip()
    lines = markdown.splitlines()
    output: list[str] = []
    index = 0
    while index < len(lines):
        if not lines[index].lstrip().startswith(">"):
            output.append(lines[index])
            index += 1
            continue

        end = index
        while end < len(lines) and lines[end].lstrip().startswith(">"):
            end += 1
        block_lines = lines[index:end]
        block_text = " ".join(
            re.sub(r"^\s*>\s?", "", line).strip() for line in block_lines
        ).strip()
        labelled_match = re.search(
            r"(?:英文)?原文\s*[：:]\s*[“\"](.+?)[”\"]",
            block_text,
        )
        plain_match = re.fullmatch(r"[“\"](.+?)[”\"]", block_text)
        match = labelled_match or plain_match
        if not match:
            output.extend(block_lines)
            index = end
            continue

        quote = re.sub(r"\s+", " ", match.group(1)).strip()
        english_words = re.findall(r"\b[A-Za-z]+(?:[-'][A-Za-z]+)*\b", quote)
        if not english_words:
            output.extend(block_lines)
            index = end
            continue
        if (
            quote
            and len(english_words) <= 80
            and quote in normalized_source
        ):
            output.append(f"> “{quote}”")
        index = end

    return "\n".join(output)


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
        "news_summary": dossier.get("summary", "") if content_type == "popular" else "",
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
    paper_abstract = ""
    if content_type == "paper":
        paper_abstract = str(
            dossier.get("abstract")
            or (dossier.get("openalex") or {}).get("abstract")
            or ""
        ).strip()
        safe_input["abstract"] = paper_abstract[:12000]
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
    if content_type == "paper":
        markdown = _remove_unverified_paper_quotes(
            markdown,
            str(safe_input["paper_text"]),
        )
    markdown = _normalize_article_markdown(markdown, display_title)
    if not markdown:
        raise RuntimeError("model returned empty article")
    if content_type == "paper" and paper_abstract:
        abstract_lead = translate_paper_abstract(paper_abstract, settings)
        markdown = _replace_paper_lead(markdown, abstract_lead)

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
