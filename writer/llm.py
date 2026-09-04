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
        start = stripped.find("{")
        if start < 0:
            raise
        try:
            parsed, _ = json.JSONDecoder().raw_decode(stripped[start:])
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))
        return parsed


def _normalize_evidence_anchor(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).replace("−", "-").replace("–", "-")


def _paper_source_paragraph_records(source_text: str) -> list[dict[str, Any]]:
    paragraphs = [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in re.split(r"\n+", source_text)
        if paragraph.strip()
    ]
    return [
        {"id": f"source-{index}", "text": paragraph[:1200]}
        for index, paragraph in enumerate(paragraphs[:80])
    ]


def _extract_paper_evidence_plan(markdown: str) -> tuple[dict[str, Any], str]:
    match = re.search(
        r"<!--\s*PAPER_EVIDENCE_PLAN\s*(\{.*?\})\s*-->",
        markdown,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        raise RuntimeError("PAPER evidence plan missing")
    try:
        plan = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"PAPER evidence plan is invalid JSON: {exc}") from exc
    if not isinstance(plan, dict) or not isinstance(plan.get("sections"), list):
        raise RuntimeError("PAPER evidence plan must contain a sections list")
    return plan, (markdown[: match.start()] + markdown[match.end() :]).strip()


def _paper_body_sections(markdown: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"(?m)^##\s+(.+?)\s*$", markdown))
    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        title = match.group(1).strip()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        if title in {"文章信息", "参考文献", "来源"}:
            continue
        body = markdown[match.end() : end]
        body = "\n".join(
            line
            for line in body.splitlines()
            if not line.lstrip().startswith((">", "![", "*Fig.", "*图", "图源：", "Source:"))
        )
        sections.append((title, body))
    return sections


def _validate_paper_evidence_plan(
    plan: dict[str, Any],
    markdown: str,
    valid_source_paragraph_ids: set[str] | None = None,
) -> None:
    sections = _paper_body_sections(markdown)
    planned_sections = plan.get("sections") or []
    if not sections or not planned_sections:
        raise RuntimeError("PAPER evidence plan validation failed: no body sections")
    for planned_index, section in enumerate(planned_sections):
        if not isinstance(section, dict):
            raise RuntimeError("PAPER evidence plan validation failed: invalid section")
        source_ids = section.get("source_paragraph_ids")
        if not isinstance(source_ids, list) or not source_ids or not all(
            isinstance(source_id, str) and source_id.strip() for source_id in source_ids
        ):
            raise RuntimeError(
                "PAPER evidence plan validation failed: invalid source_paragraph_ids"
            )
        if valid_source_paragraph_ids is not None:
            unknown_ids = set(source_ids) - valid_source_paragraph_ids
            if unknown_ids:
                raise RuntimeError(
                    "PAPER evidence plan validation failed: unknown source paragraph ids: "
                    + ", ".join(sorted(unknown_ids))
                )
        findings = section.get("findings") or []
        if not isinstance(findings, list):
            raise RuntimeError("PAPER evidence plan validation failed: invalid findings")
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            anchors = finding.get("anchors") or []
            if not isinstance(anchors, list):
                continue
            for anchor in anchors:
                normalized_anchor = _normalize_evidence_anchor(str(anchor))
                if not normalized_anchor or re.fullmatch(
                    r"[Pp][<>=]\d+(?:\.\d+)?", normalized_anchor
                ):
                    continue
                actual_indexes = [
                    index
                    for index, (_, body) in enumerate(sections)
                    if normalized_anchor in _normalize_evidence_anchor(body)
                ]
                if not actual_indexes:
                    continue
                if planned_index >= len(sections) or actual_indexes != [planned_index]:
                    planned_title = str(section.get("title") or section.get("id") or planned_index + 1)
                    actual_titles = ", ".join(sections[index][0] for index in actual_indexes)
                    raise RuntimeError(
                        "PAPER evidence section mismatch: "
                        f"evidence={anchor!r}; planned section={planned_title!r}; "
                        f"actual section={actual_titles!r}"
                    )


def _validate_paper_plan_structure(
    plan: dict[str, Any],
    valid_source_paragraph_ids: set[str],
) -> list[dict[str, Any]]:
    sections = plan.get("sections")
    if not isinstance(sections, list) or not sections:
        raise RuntimeError("PAPER scientific planner returned no sections")
    seen_ids: set[str] = set()
    validated: list[dict[str, Any]] = []
    for section in sections:
        if not isinstance(section, dict):
            raise RuntimeError("PAPER scientific planner returned an invalid section")
        section_id = str(section.get("id") or "").strip()
        title = str(section.get("title") or "").strip()
        role = str(section.get("role") or "").strip()
        source_ids = section.get("source_paragraph_ids")
        findings = section.get("findings")
        if not section_id or section_id in seen_ids or not title or not role:
            raise RuntimeError("PAPER scientific planner returned invalid section metadata")
        if not isinstance(source_ids, list) or not source_ids:
            raise RuntimeError("PAPER scientific planner returned invalid source paragraph ids")
        if not all(isinstance(source_id, str) and source_id in valid_source_paragraph_ids for source_id in source_ids):
            raise RuntimeError("PAPER scientific planner returned unknown source paragraph ids")
        if not isinstance(findings, list) or not findings:
            raise RuntimeError("PAPER scientific planner returned a section without findings")
        for finding in findings:
            if not isinstance(finding, dict) or not str(finding.get("evidence") or "").strip():
                raise RuntimeError("PAPER scientific planner returned an invalid finding")
            anchors = finding.get(
                "anchors",
                finding.get("quantitative_anchors", finding.get("quantitative anchors", [])),
            )
            if anchors is None:
                anchors = []
            elif isinstance(anchors, str):
                anchors = [anchors]
            if not isinstance(anchors, list):
                raise RuntimeError("PAPER scientific planner returned invalid quantitative anchors")
            finding["anchors"] = anchors
            finding.pop("quantitative_anchors", None)
            finding.pop("quantitative anchors", None)
        seen_ids.add(section_id)
        validated.append(section)
    return validated


def _paper_prune_plan_sections(
    plan: dict[str, Any],
    images: list[dict[str, Any]],
) -> dict[str, Any]:
    sections = list(plan.get("sections") or [])
    if not sections:
        raise RuntimeError("PAPER scientific planner returned no sections")
    numbered_images: dict[int, dict[str, Any]] = {}
    for index, image in enumerate(images, start=1):
        try:
            figure_number = int(image.get("figure_number"))
        except (TypeError, ValueError):
            continue
        if figure_number > 0 and image.get("publishable", True) is not False:
            numbered_images[index] = image
    referenced_indexes: list[int] = []
    section_figure_indexes: dict[str, list[int]] = {}
    for section in sections:
        section_id = str(section.get("id") or "")
        indexes: list[int] = []
        for source_id in section.get("source_paragraph_ids") or []:
            match = re.fullmatch(r"source-figure-(\d+)", str(source_id))
            if match is None:
                continue
            image_index = int(match.group(1))
            if image_index in numbered_images and image_index not in indexes:
                indexes.append(image_index)
        for figure_label in section.get("figure_ids") or section.get("selected_body_figures") or []:
            match = re.search(r"(?:Fig(?:ure)?\.?)\s*(\d+)", str(figure_label), re.IGNORECASE)
            if match is None:
                continue
            figure_number = int(match.group(1))
            image_index = next(
                (
                    index
                    for index, image in numbered_images.items()
                    if int(image.get("figure_number") or 0) == figure_number
                ),
                None,
            )
            if image_index is not None and image_index not in indexes:
                indexes.append(image_index)
        for image_index in indexes:
            if image_index not in referenced_indexes:
                referenced_indexes.append(image_index)
        section_figure_indexes[section_id] = indexes
    selected_indexes = set(referenced_indexes if len(referenced_indexes) <= 4 else referenced_indexes[:4])
    retained: list[dict[str, Any]] = []
    pruned: list[dict[str, Any]] = []
    for section in sections:
        current = dict(section)
        section_id = str(current.get("id") or "")
        selected_figures = [
            f"Fig. {int(numbered_images[index].get('figure_number'))}"
            for index in section_figure_indexes.get(section_id, [])
            if index in selected_indexes
        ]
        if selected_figures:
            current["selected_body_figures"] = selected_figures
            retained.append(current)
            continue
        transition = bool(current.get("necessary_transition"))
        transition_reason = str(current.get("transition_reason") or "").strip()
        if transition and transition_reason:
            current["retained_without_figure"] = True
            current["retention_reason"] = transition_reason
            retained.append(current)
            continue
        current["pruned"] = True
        current["prune_reason"] = "no selected body figure and not a necessary transition"
        pruned.append(current)
    if not retained:
        raise RuntimeError("PAPER section pruning removed every section")
    plan["planner_sections"] = [dict(section) for section in sections]
    plan["sections"] = retained
    plan["pruned_sections"] = pruned
    return plan


def _paper_style_exemplar() -> str:
    paths = sorted(
        PROJECT_ROOT.glob("articles/paper/*/article.md"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    excerpts: list[str] = []
    for path in paths[:2]:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        sections = _paper_body_sections(text)
        excerpt = "\n\n".join(f"## {title}\n{body.strip()}" for title, body in sections[:3])
        if excerpt:
            excerpts.append(excerpt[:1800])
    return "\n\n---\n\n".join(excerpts)


def _paper_completion_json(client: OpenAI, system_prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = _paper_completion_with_retry(
        client,
        model=payload.pop("_model"),
        temperature=payload.pop("_temperature", 0.2),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    parsed = _json_from_text(response.choices[0].message.content or "")
    if not isinstance(parsed, dict):
        raise RuntimeError("PAPER model returned a non-object JSON response")
    return parsed


def _paper_plan(
    client: OpenAI,
    abstract: str,
    paper_text: str,
    source_paragraphs: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return _paper_completion_json(
        client,
        PAPER_PLANNER_PROMPT,
        {
            "abstract": abstract,
            "paper_text": paper_text,
            "source_paragraphs": source_paragraphs,
            "metadata": metadata,
            "_model": metadata["model"],
            "_temperature": 0.1,
        },
    )


def _paper_section_body(response: dict[str, Any]) -> str:
    body = response.get("body")
    if not isinstance(body, str):
        raise RuntimeError("PAPER section writer returned invalid body")
    body = _remove_generated_terminal_sections(body).strip()
    body = "\n".join(line for line in body.splitlines() if not re.match(r"^#{1,6}\s+", line.strip()))
    if not body:
        raise RuntimeError("PAPER section writer returned empty body")
    return body.strip()


def _paper_write_section(
    client: OpenAI,
    abstract: str,
    section: dict[str, Any],
    source_paragraphs: list[dict[str, Any]],
    model: str,
) -> str:
    source_ids = set(section["source_paragraph_ids"])
    section_sources = [record for record in source_paragraphs if record["id"] in source_ids]
    payload = {
        "abstract": abstract,
        "section": section,
        "source_paragraphs": section_sources,
    }
    response = _paper_completion_with_retry(
        client,
        model=model,
        temperature=0.25,
        messages=[
            {"role": "system", "content": PAPER_STYLE_GUIDE},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    content = (response.choices[0].message.content or "").strip()
    try:
        parsed = _json_from_text(content)
    except (TypeError, json.JSONDecodeError):
        parsed = {"body": content}
    if not isinstance(parsed, dict):
        parsed = {"body": content}
    return _paper_section_body(parsed)


def _paper_assemble_markdown(display_title: str, abstract_lead: str, sections: list[dict[str, Any]]) -> str:
    chunks = [f"# {display_title}", abstract_lead.strip()]
    for section in sections:
        body = str(section.get("body") or "").strip()
        if not body:
            raise RuntimeError("PAPER assembly encountered an empty section")
        chunks.append(f"## {section['title']}\n\n{body}")
    return "\n\n".join(chunk for chunk in chunks if chunk).strip()


def _paper_extract_section_bodies(markdown: str, plan_sections: list[dict[str, Any]]) -> list[str]:
    actual = _paper_body_sections(markdown)
    if len(actual) != len(plan_sections):
        raise RuntimeError("PAPER revised draft changed the planned section count")
    return [body.strip() for _, body in actual]


def _paper_review(
    client: OpenAI,
    abstract: str,
    paper_text: str,
    source_paragraphs: list[dict[str, Any]],
    plan: dict[str, Any],
    draft: str,
    model: str,
) -> dict[str, Any]:
    response = _paper_completion_json(
        client,
        PAPER_REVIEWER_PROMPT,
        {
            "abstract": abstract,
            "paper_text": paper_text,
            "source_paragraphs": source_paragraphs,
            "paper_evidence_plan": plan,
            "draft": draft,
            "_model": model,
            "_temperature": 0.1,
        },
    )
    status = response.get("status")
    if status not in {"pass", "needs_revision"}:
        raise RuntimeError("PAPER scientific reviewer returned invalid status")
    corrections = response.get("corrections", [])
    if not isinstance(corrections, list):
        raise RuntimeError("PAPER scientific reviewer returned invalid corrections")
    if status == "needs_revision" and not corrections:
        raise RuntimeError("PAPER scientific reviewer requested revision without corrections")
    return response


def _paper_revision(
    client: OpenAI,
    abstract: str,
    paper_text: str,
    source_paragraphs: list[dict[str, Any]],
    plan: dict[str, Any],
    draft: str,
    corrections: list[Any],
    model: str,
) -> list[str]:
    payload = {
        "abstract": abstract,
        "paper_text": paper_text,
        "source_paragraphs": source_paragraphs,
        "paper_evidence_plan": plan,
        "draft": draft,
        "corrections": corrections,
    }
    response_obj = _paper_completion_with_retry(
        client,
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": PAPER_REVISION_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    content = (response_obj.choices[0].message.content or "").strip()
    try:
        response = _json_from_text(content)
    except (TypeError, json.JSONDecodeError):
        response = {"markdown": content}
    revised = response.get("sections") if isinstance(response, dict) else None
    if not isinstance(revised, list):
        return _paper_extract_section_bodies(content, plan["sections"])
    if len(revised) != len(plan["sections"]):
        raise RuntimeError("PAPER scientific revision returned invalid sections")
    by_id = {str(section.get("id") or ""): section for section in revised if isinstance(section, dict)}
    bodies: list[str] = []
    for planned in plan["sections"]:
        item = by_id.get(str(planned["id"]))
        if item is None or not isinstance(item.get("body"), str) or not item["body"].strip():
            raise RuntimeError("PAPER scientific revision omitted a planned section")
        bodies.append(_paper_section_body({"body": item["body"]}))
    return bodies


def _paper_editorial_rewrite(
    client: OpenAI,
    plan: dict[str, Any],
    draft: str,
    model: str,
    style_exemplar: str,
    lint_feedback: dict[str, int] | None = None,
) -> list[str]:
    payload = {
        "paper_evidence_plan": plan,
        "draft": draft,
        "style_exemplar": style_exemplar,
        "lint_feedback": lint_feedback or {},
    }
    response_obj = _paper_completion_with_retry(
        client,
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": PAPER_EDITOR_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    content = (response_obj.choices[0].message.content or "").strip()
    try:
        response = _json_from_text(content)
    except (TypeError, json.JSONDecodeError):
        response = {"markdown": content}
    revised = response.get("sections") if isinstance(response, dict) else None
    if not isinstance(revised, list):
        return _paper_extract_section_bodies(content, plan["sections"])
    if len(revised) != len(plan["sections"]):
        raise RuntimeError("PAPER Chinese editor returned invalid sections")
    by_id = {str(section.get("id") or ""): section for section in revised if isinstance(section, dict)}
    bodies: list[str] = []
    for planned in plan["sections"]:
        item = by_id.get(str(planned["id"]))
        if item is None or not isinstance(item.get("body"), str) or not item["body"].strip():
            raise RuntimeError("PAPER Chinese editor omitted a planned section")
        bodies.append(_paper_section_body({"body": item["body"]}))
    return bodies


def _paper_ai_style_lint(markdown: str) -> dict[str, int]:
    patterns = {
        "并非而是": r"并非[^。！？\n]{0,50}而是",
        "其原因在于": r"其原因在于",
        "也就是说": r"也就是说",
        "不只是": r"不只是",
        "不仅": r"不仅",
        "值得注意的是": r"值得注意的是",
        "进一步表明": r"进一步表明",
        "总体而言": r"总体而言",
        "由此可见": r"由此可见",
        "这意味着": r"这意味着",
    }
    return {name: len(re.findall(pattern, markdown)) for name, pattern in patterns.items()}


def _paper_ai_style_lint_failed(counts: dict[str, int]) -> bool:
    return any(count > 1 for count in counts.values()) or sum(counts.values()) > 4


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
                    "短Abstract基本完整翻译；长Abstract只能删除次要方法细节、重复背景和低优先级结果，不能增加原文没有的结论、分类、机制或表述，也不能自行重组科学结论。"
                    "中文结果优先控制在约200到300个汉字，但不得机械截断；若忠实表达需要更长，优先保留研究问题和2到3个核心结论。不要添加小标题、列表或解释，只返回严格JSON："
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


PAPER_EDITORIAL_GUIDE = (
    "中文科研表达编辑规则（仅作保守润色，服从Abstract、原文证据和section scope约束）："
    "1. 保留科学术语、数字、百分比、统计值、趋势方向和限定条件，不为追求自然而改写事实。"
    "2. 不增加原文没有的信息、机制、分类、案例或意义，优先删除冗余而不是装饰性扩写。"
    "3. 区分事实、结果、解释和推断，让表述强度与证据强度匹配；correlation不写成causation。"
    "4. 不把suggest、indicate、可能或表明强化成prove、确定或必然。"
    "5. 使用自然、克制、直接的中文，保留必要术语，避免逐句翻译和明显英文翻译腔。"
    "6. 删除模板化连接词、AI套话、空泛意义拔高和重复总结，但不牺牲必要的科学限定。"
    "7. 减少重复句式和无信息增益的句子，让每句话服务于科学主线、证据或必要解释。"
    "8. 不使用营销口号、夸张比喻或个人经历，不为了‘人味’改变科学论证。"
    "9. 保持段落主线和整体连贯性，不把细碎写作清单凌驾于科学内容之上。"
)


PAPER_STYLE_GUIDE = (
    "你是中文科研编辑，负责写一个PAPER section的正文。只依据输入的abstract、当前section的role/title、"
    "当前section的findings和source_paragraphs写作；不要引入任何未提供的科学事实、数字、机制或下一section内容。"
    "只返回当前section的中文正文，不返回标题、导语、计划、图片、参考文献或文章信息。"
    "保留数字、趋势方向、时间范围、变量关系和因果强度；correlation不写成causation。"
    "正文应自然、简洁、信息密度高，避免翻译腔、空泛总结和重复连接词。若当前section标记为retained_without_figure，只保留理解相邻主图所需的极短桥接内容，不展开次要机制或补充材料。"
    "如果有可核验的paper_text原句，可以保留短Markdown引用块，但不得改写或编造。"
)

PAPER_PLANNER_PROMPT = (
    "你是Scientific Planner，不写文章正文。根据原始Abstract、paper_text、source_paragraphs和metadata，"
    "建立唯一的paper_evidence_plan，并只返回严格JSON对象。Abstract决定全文主线，Results和source paragraphs只补充证据。"
    "每个section包含id、title、role、source_paragraph_ids和findings；title必须是适合中文成稿的简洁中文小标题；每个finding包含id、evidence、quantitative anchors。"
    "结合输入图片的Figure编号判断该section是否有正文主图；必要时可提供necessary_transition=true及简短transition_reason，说明无图section为何是理解相邻主图不可缺少的桥梁。"
    "每个核心finding只能有一个primary section。若historical/model spread、mechanism、attribution、projection或implication"
    "是不同科学问题且各有独立核心证据，必须优先拆开；不要为凑3到4节而合并attribution与future projection，也不要固定section数量。"
    "只有Abstract或Results明确支持时才拆分multiple modes/regimes，不得创造first/second mode。"
    "XGBoost、SHAP、CCA、所有R/r相关系数、百分比归因和历史模式离散度等统计/归因证据必须放入attribution或明确的统计结果section；mechanism section只写物理过程和定性响应，不承载R/r或百分比统计anchor。SSP情景、时间段和未来离散度应放入独立projection section。不要在不同role重复同一核心finding或anchor。"
    "role使用贴合论文的简洁自然标签，不要套固定taxonomy。source_paragraph_ids只能使用输入中真实存在的source id。"
    "每个finding都要有anchors数组（字段名必须是anchors，不得写成quantitative anchors或其他字段）；anchors保留原文指标大小写、R/r、符号、数值、百分号和时间段；不要使用跨section重复的P值作为anchor。"
    '返回格式：{"sections":[{"id":"section-1","title":"...","role":"attribution","source_paragraph_ids":["source-0"],"findings":[{"id":"E1","evidence":"...","anchors":["R = 0.71"]}]}]}'
)

PAPER_REVIEWER_PROMPT = (
    "你是Scientific Reviewer，只审核科学准确性和section结构，不润色文风，不重写全文。忠实翻译的Abstract导语应保持原始Abstract的结论和因果强度，不要要求改写原文已有表述；只检查导语是否新增或强化了Abstract没有的内容。检查Abstract核心结论覆盖、"
    "证据与plan归属、数字/R/r/百分比/时间段、机制与因果强度、correlation与causation区分、重复finding，以及"
    "mechanism/attribution/projection是否混在错误section，以及每个计划中的定量anchor是否在对应section实际保留。返回严格JSON："
    '{"status":"pass"|"needs_revision","corrections":[{"section_id":"section-1","issue":"...","instruction":"..."}]}'
)

PAPER_REVISION_PROMPT = (
    "你是Scientific Revision Editor。根据reviewer corrections修正科学内容，只返回严格JSON sections数组。"
    "必须逐条落实corrections，不能原样保留reviewer指出的错误句子或仅添加免责声明；修正应针对具体问题，不得因此削弱Abstract明确支持的结论强度、删除核心finding或遗漏关键数字。"
    "保留planner的section id和顺序；每个body只能写对应section的finding/evidence，不得把证据移动到别节，不得修改无问题的科学内容，不得润色成营销文案。返回："
    '{"sections":[{"id":"section-1","body":"..."}]}'
)

PAPER_EDITOR_PROMPT = (
    "你是中文科技编辑。输入稿件的科学内容已经锁定，只优化中文自然度、句式、节奏、冗余和AI套话。"
    "严格保留planner的section id、顺序、role含义、数字、趋势、相关系数、时间段、因果关系和全部核心finding；"
    "不得新增、删除、合并或移动科学证据，不得重写Abstract导语；保持正文主体约650到800个中文字符，但不要机械截断句子。避免反复使用并非而是、其原因在于、也就是说、"
    "不只是、不仅、值得注意的是、进一步表明、总体而言、由此可见、这意味着。只返回严格JSON sections数组："
    '{"sections":[{"id":"section-1","body":"..."}]}'
    + "\n\n"
    + PAPER_EDITORIAL_GUIDE
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


def _write_article_files(
    dossier: dict[str, Any],
    settings: Settings,
    output_dir: Path,
    markdown: str,
    paper_evidence_plan: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / "article.md"
    metadata_path = output_dir / "metadata.json"
    markdown_path.write_text(markdown.strip() + "\n", encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "model": settings.model_name,
                "source": dossier.get("url", ""),
                "doi": dossier.get("doi", ""),
                "images": dossier.get("images", []),
                "paper_evidence_plan": paper_evidence_plan or {},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return markdown_path, metadata_path


def _generate_paper_article_markdown(
    dossier: dict[str, Any],
    settings: Settings,
    output_dir: Path,
    display_title: str,
) -> tuple[Path, Path]:
    abstract = str(
        dossier.get("abstract")
        or (dossier.get("openalex") or {}).get("abstract")
        or ""
    ).strip()[:12000]
    paper_text = str(dossier.get("text") or "")[:50000]
    source_paragraphs = _paper_source_paragraph_records(paper_text or abstract)
    for index, image in enumerate(dossier.get("images", []), start=1):
        caption = str(image.get("caption") or image.get("description") or "").strip()
        if caption:
            source_paragraphs.append(
                {"id": f"source-figure-{index}", "text": caption[:1200]}
            )
    valid_source_ids = {str(record["id"]) for record in source_paragraphs}
    metadata = {
        "title": dossier.get("title", ""),
        "display_title": display_title,
        "doi": dossier.get("doi", ""),
        "journal": dossier.get("journal", ""),
        "authors": dossier.get("authors", []),
        "figure_captions": [
            {
                "caption": image.get("caption", ""),
                "credit": image.get("credit", ""),
                "license": image.get("license", ""),
            }
            for image in dossier.get("images", [])
        ],
        "model": settings.model_name,
    }
    client = OpenAI(
        api_key=settings.model_api_key,
        base_url=settings.model_base_url,
        timeout=180.0,
        max_retries=2,
    )
    plan = _paper_plan(client, abstract, paper_text, source_paragraphs, metadata)
    plan["sections"] = _validate_paper_plan_structure(plan, valid_source_ids)
    plan = _paper_prune_plan_sections(plan, list(dossier.get("images") or []))
    abstract_lead = translate_paper_abstract(abstract, settings) if abstract else ""

    sections: list[dict[str, Any]] = []
    for section in plan["sections"]:
        sections.append(
            {
                **section,
                "body": _paper_write_section(
                    client,
                    abstract,
                    section,
                    source_paragraphs,
                    settings.model_name,
                ),
            }
        )
    draft = _paper_assemble_markdown(display_title, abstract_lead, sections)
    review = _paper_review(
        client,
        abstract,
        paper_text,
        source_paragraphs,
        plan,
        draft,
        settings.model_name,
    )
    for revision_attempt in range(2):
        if review["status"] == "pass":
            break
        revised_bodies = _paper_revision(
            client,
            abstract,
            paper_text,
            source_paragraphs,
            plan,
            draft,
            review["corrections"],
            settings.model_name,
        )
        draft = _paper_assemble_markdown(
            display_title,
            abstract_lead,
            [
                {**section, "body": body}
                for section, body in zip(plan["sections"], revised_bodies)
            ],
        )
        review = _paper_review(
            client,
            abstract,
            paper_text,
            source_paragraphs,
            plan,
            draft,
            settings.model_name,
        )
        if review["status"] == "pass":
            break
    else:
        raise RuntimeError("PAPER scientific review failed after revision")

    style_exemplar = _paper_style_exemplar()
    editor_bodies = _paper_editorial_rewrite(
        client,
        plan,
        draft,
        settings.model_name,
        style_exemplar,
    )
    markdown = _paper_assemble_markdown(
        display_title,
        abstract_lead,
        [
            {**section, "body": body}
            for section, body in zip(plan["sections"], editor_bodies)
        ],
    )
    lint = _paper_ai_style_lint(markdown)
    if _paper_ai_style_lint_failed(lint):
        editor_bodies = _paper_editorial_rewrite(
            client,
            plan,
            markdown,
            settings.model_name,
            style_exemplar,
            lint,
        )
        markdown = _paper_assemble_markdown(
            display_title,
            abstract_lead,
            [
                {**section, "body": body}
                for section, body in zip(plan["sections"], editor_bodies)
            ],
        )
        lint = _paper_ai_style_lint(markdown)
        if _paper_ai_style_lint_failed(lint):
            raise RuntimeError(f"PAPER Chinese editorial lint failed: {lint}")

    markdown = _remove_unverified_paper_quotes(markdown, paper_text)
    markdown = _normalize_article_markdown(markdown, display_title)
    if not markdown:
        raise RuntimeError("PAPER staged pipeline returned empty article")
    _validate_paper_evidence_plan(plan, markdown, valid_source_ids)
    dossier["paper_evidence_plan"] = plan
    return _write_article_files(dossier, settings, output_dir, markdown, plan)


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
    if content_type == "paper":
        return _generate_paper_article_markdown(dossier, settings, output_dir, display_title)

    safe_input = {
        "content_type": content_type,
        "display_title": display_title,
        "title": dossier.get("title", ""),
        "news_summary": dossier.get("summary", ""),
        "news_text": str(dossier.get("news_text") or dossier.get("text") or "")[:50000],
        "paper_text": "",
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
            {"role": "system", "content": NEWS_ARTICLE_PROMPT},
            {"role": "user", "content": json.dumps(safe_input, ensure_ascii=False)},
        ],
    )
    markdown = _normalize_article_markdown(
        _remove_generated_terminal_sections((response.choices[0].message.content or "").strip()),
        display_title,
    )
    if not markdown:
        raise RuntimeError("model returned empty article")
    return _write_article_files(dossier, settings, output_dir, markdown)


def article_output_dir(
    date: str,
    rank: int,
    content_type: str = "",
) -> Path:
    if content_type:
        return PROJECT_ROOT / "articles" / content_type / f"{date}-{rank:03d}"
    return PROJECT_ROOT / "articles" / f"{date}-{rank:03d}"
