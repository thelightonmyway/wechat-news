"""OpenAI-compatible text-only selection and writing adapter."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from settings import PROJECT_ROOT, Settings


logger = logging.getLogger(__name__)


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
    normalized = re.sub(r"\s+", "", str(value or ""))
    normalized = re.sub(r"^(?:approximately|approx\.?|about|around|roughly|~|约|大约)", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"(?<=\d{4})to(?=\d{4})", "-", normalized, flags=re.IGNORECASE)
    return (
        normalized
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("‐", "-")
    )


def _paper_source_paragraph_records(source_text: str) -> list[dict[str, Any]]:
    paragraphs = [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in re.split(r"\n+", source_text)
        if paragraph.strip()
    ]
    return [
        {"id": f"source-{index}", "text": paragraph[:1200]}
        for index, paragraph in enumerate(paragraphs[:360])
    ]


def _paper_figure_id(value: Any) -> str:
    text = str(value or "").strip()
    if re.fullmatch(r"\d+", text):
        return f"Fig. {int(text)}"
    match = re.search(r"(?:Fig(?:ure)?\.?|图)\s*(\d+)", text, re.IGNORECASE)
    return f"Fig. {int(match.group(1))}" if match else text


def _paper_figure_numbers_in_text(text: str) -> set[int]:
    numbers: set[int] = set()
    for match in re.finditer(r"(?<!supplementary\s)(?<!supporting\s)(?:fig(?:ure)?\.?|图)\s*(\d+)", text, re.IGNORECASE):
        numbers.add(int(match.group(1)))
    return numbers


def _paper_quantitative_anchors(text: str) -> list[str]:
    patterns = (
        r"[Rr]\s*=\s*[−-]?\s*\d+(?:\.\d+)?",
        r"(?:(?:approximately|approx\.?|about|around|roughly|~|约|大约)\s+)?\d+(?:\.\d+)?\s*%",
        r"SSP\s*\d+(?:[-‐–]\d+(?:\.\d+)?)?",
        r"\d{4}\s*(?:[‐–—-]|to)\s*\d{4}",
    )
    anchors: list[str] = []
    for pattern in patterns:
        anchors.extend(match.group(0).strip() for match in re.finditer(pattern, text, re.IGNORECASE))
    return list(dict.fromkeys(anchors))


def _paper_anchor_sentences(text: str) -> list[tuple[str, str]]:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?。！？])\s+|\n+", text)
        if sentence.strip()
    ]
    return [
        (match.group(0).strip(), sentence)
        for sentence in sentences
        for pattern in (
            r"[Rr]\s*=\s*[−-]?\s*\d+(?:\.\d+)?",
            r"(?:(?:approximately|approx\.?|about|around|roughly|~|约|大约)\s+)?\d+(?:\.\d+)?\s*%",
            r"SSP\s*\d+(?:[-‐–]\d+(?:\.\d+)?)?",
            r"\d{4}\s*(?:[‐–—-]|to)\s*\d{4}",
        )
        for match in re.finditer(pattern, sentence, re.IGNORECASE)
    ]


def _paper_anchor_figure_refs(sentence: str, anchor: str) -> set[int]:
    clauses = re.split(r"[;；。!?]|\.(?!\d)", sentence)
    clause = next((part for part in clauses if anchor in part), sentence)
    references = list(
        re.finditer(
            r"(?<!supplementary\s)(?<!supporting\s)(?:fig(?:ure)?\.?|图)\s*(\d+)",
            clause,
            re.IGNORECASE,
        )
    )
    # Multiple unseparated main-Figure references do not establish which
    # Figure supports this anchor.  Leave the provenance contextual rather
    # than copying the value to every referenced Figure.
    if len(references) > 1:
        return set()
    return {int(match.group(1)) for match in references}


def _paper_is_figure_specific(
    sentence: str,
    caption: str = "",
    anchor: str = "",
) -> bool:
    text = f"{sentence} {caption}".lower()
    if anchor:
        anchor_index = text.find(anchor.lower())
        if anchor_index >= 0:
            text = text[max(0, anchor_index - 90) : anchor_index + len(anchor) + 90]
            temporal_context = re.search(
                r"(?:from|over|during|between|period|periods)\s*.{0,45}"
                + re.escape(anchor.lower()),
                text,
            )
            result_relation = re.search(
                r"(?:r\s*=|coefficient\s+is\s+given|corresponds|reduction|"
                r"accounting|contribution|explains|reports?\s+(?:a|an)?\s*(?:value|result)?)",
                text,
            )
            if temporal_context and not result_relation:
                return False
    return bool(
        re.search(
            r"correlat|coefficient|reconstruct|xgboost|shap|cca|canonical|"
            r"reduction|accounting|explains|contribution|corresponds|scatter|"
            r"report|\br\s*=\s*[-−]?\d|\d+(?:\.\d+)?\s*%",
            text,
        )
    )


def _paper_figure_semantic_score(sentence: str, caption: str) -> int:
    sentence_lower = sentence.lower()
    caption_lower = caption.lower()
    score = len(
        {
            token
            for token in re.findall(r"[a-z0-9]+", sentence_lower)
            if len(token) >= 4
        }
        & {
            token
            for token in re.findall(r"[a-z0-9]+", caption_lower)
            if len(token) >= 4
        }
    )
    if re.search(r"historical|1970|1979|2014", sentence_lower):
        score += 4 if re.search(r"historical|1970|1979|2014", caption_lower) else 0
        score -= 3 if re.search(r"future|projection|2025|2070|ssp", caption_lower) else 0
    if re.search(r"future|projection|2025|2070|ssp", sentence_lower):
        score += 4 if re.search(r"future|projection|2025|2070|ssp", caption_lower) else 0
        score -= 3 if re.search(r"historical|1970|1979|2014", caption_lower) else 0
    elif re.search(r"future|projection|2025|2070|ssp", caption_lower):
        score -= 2
    if "forest" in sentence_lower and "forest" in caption_lower:
        score += 2
    if re.search(r"inter[-‐– ]model spread", sentence_lower) and re.search(
        r"inter[-‐– ]model|standard deviation|forest", caption_lower
    ):
        score += 2
    return score


def _paper_figure_evidence_bundles(
    selected_images: list[dict[str, Any]],
    source_paragraphs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    figure_info: list[tuple[str, int, str]] = []
    for index, image in enumerate(selected_images, start=1):
        figure_id = _paper_figure_id(
            image.get("figure_number") or image.get("metadata_title") or index
        )
        number_match = re.search(r"(\d+)", figure_id)
        figure_info.append(
            (
                figure_id,
                int(number_match.group(1)) if number_match else index,
                str(image.get("caption") or image.get("original_caption") or image.get("alt") or "").strip(),
            )
        )

    anchor_mentions: dict[str, list[dict[str, Any]]] = {}
    raw_anchor: dict[str, str] = {}
    for record_index, record in enumerate(source_paragraphs):
        record_id = str(record.get("id") or "")
        record_text = str(record.get("text") or "")
        for anchor, sentence in _paper_anchor_sentences(record_text):
            key = _normalize_evidence_anchor(anchor)
            raw_anchor.setdefault(key, anchor)
            sentence_figures = _paper_anchor_figure_refs(sentence, anchor)
            paragraph_figures = _paper_figure_numbers_in_text(record_text)
            nearby_text = " ".join(
                [
                    record_text,
                    *[
                        str(source_paragraphs[next_index].get("text") or "")
                        for next_index in range(record_index + 1, min(len(source_paragraphs), record_index + 3))
                    ],
                ]
            )
            excluded_reference = bool(
                re.search(
                    r"supporting|supplementary|figure\s*s\d",
                    sentence + " " + nearby_text,
                    re.IGNORECASE,
                )
            )
            if not sentence_figures and len(paragraph_figures) != 1:
                paragraph_figures = set()
            if not sentence_figures and not paragraph_figures and not excluded_reference:
                for previous in range(record_index - 1, max(-1, record_index - 5), -1):
                    nearby = _paper_figure_numbers_in_text(
                        str(source_paragraphs[previous].get("text") or "")
                    )
                    if len(nearby) == 1:
                        paragraph_figures = nearby
                        break
            anchor_mentions.setdefault(key, []).append(
                {
                    "value": anchor,
                    "sentence": sentence,
                    "source_paragraph_ids": [record_id],
                    "explicit_figure_numbers": sorted(sentence_figures or paragraph_figures),
                    "excluded_reference": excluded_reference,
                    "is_figure_specific": _paper_is_figure_specific(sentence, anchor=anchor),
                    "is_global_context": record_index == 0 or record_id in {"abstract", "source-abstract"},
                    "confidence": "high" if sentence_figures else "medium",
                }
            )
    for figure_id, figure_number, caption in figure_info:
        for anchor in _paper_quantitative_anchors(caption):
            key = _normalize_evidence_anchor(anchor)
            raw_anchor.setdefault(key, anchor)
            anchor_mentions.setdefault(key, []).append(
                {
                    "value": anchor,
                    "sentence": caption,
                    "source_paragraph_ids": [f"caption:{figure_id}"],
                    "explicit_figure_numbers": [figure_number],
                    "excluded_reference": False,
                    "is_figure_specific": _paper_is_figure_specific(caption, anchor=anchor),
                    "confidence": "high",
                }
            )

    supported_by_anchor: dict[str, set[str]] = {}
    provenance_by_anchor: dict[str, list[dict[str, Any]]] = {}
    for key, mentions in anchor_mentions.items():
        records: list[dict[str, Any]] = []
        for mention in mentions:
            figures = {
                figure_id
                for figure_id, figure_number, _ in figure_info
                if figure_number in set(mention["explicit_figure_numbers"])
            }
            if mention["excluded_reference"]:
                figures = set()
            scope = "global_context" if mention.get("is_global_context") else "section_context"
            confidence = "high" if mention.get("is_global_context") else "medium"
            if figures and mention["is_figure_specific"]:
                scope = "figure_specific"
                confidence = mention["confidence"]
                supported_by_anchor.setdefault(key, set()).update(figures)
            elif mention["source_paragraph_ids"][0].startswith("caption:") or figures:
                scope = "section_context"
                confidence = "medium" if figures else "high"
            record = {
                "value": mention["value"],
                "normalized_value": key,
                "source_paragraph_ids": mention["source_paragraph_ids"],
                "source_sentence": mention["sentence"],
                "explicit_figure_refs": sorted(figures),
                "supported_figures": sorted(figures) if scope == "figure_specific" else [],
                "scope": scope,
                "provenance_confidence": confidence,
            }
            records.append(record)
        provenance_by_anchor[key] = records

    bundles: list[dict[str, Any]] = []
    for figure_id, figure_number, caption in figure_info:
        caption_ids = [
            str(record["id"])
            for record in source_paragraphs
            if str(record.get("id") or "").startswith("source-figure-")
            and figure_number in _paper_figure_numbers_in_text(str(record.get("text") or ""))
        ]
        explicit_ids = [
            str(record["id"])
            for record in source_paragraphs
            if not str(record.get("id") or "").startswith("source-figure-")
            and figure_number in _paper_figure_numbers_in_text(str(record.get("text") or ""))
        ]
        direct_ids = list(dict.fromkeys([*caption_ids, *explicit_ids]))
        figure_provenance: list[dict[str, Any]] = []
        for records in provenance_by_anchor.values():
            for record in records:
                directly_bound = (
                    figure_id in record.get("supported_figures", [])
                    or figure_id in record.get("explicit_figure_refs", [])
                )
                context_scores = {
                    candidate_id: _paper_figure_semantic_score(
                        str(record.get("source_sentence") or ""), candidate_caption
                    )
                    for candidate_id, _, candidate_caption in figure_info
                }
                best_context_score = max(context_scores.values(), default=0)
                context_bound = (
                    record.get("scope") != "figure_specific"
                    and best_context_score > 0
                    and context_scores.get(figure_id) == best_context_score
                    and list(context_scores.values()).count(best_context_score) == 1
                )
                if directly_bound or context_bound:
                    figure_provenance.append(record)
        supported_anchors = [
            raw_anchor[key]
            for key, supported_figures in supported_by_anchor.items()
            if figure_id in supported_figures
        ]
        bundles.append(
            {
                "figure_id": figure_id,
                "caption": caption,
                "explicit_source_paragraph_ids": explicit_ids,
                "directly_associated_source_paragraph_ids": direct_ids,
                "source_paragraphs": [
                    {"id": str(record.get("id") or ""), "text": str(record.get("text") or "")[:1200]}
                    for record in source_paragraphs
                    if str(record.get("id") or "") in direct_ids
                ],
                "findings": [
                    {
                        "id": f"{figure_id}-caption",
                        "evidence": caption or figure_id,
                        "quantitative_anchors": supported_anchors,
                    }
                ],
                "quantitative_anchors": supported_anchors,
                "supported_figures_by_anchor": {
                    raw_anchor[key]: sorted(supported_figures)
                    for key, supported_figures in supported_by_anchor.items()
                    if figure_id in supported_figures
                },
                "provenance": figure_provenance,
            }
        )
    return bundles


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


def _paper_validate_story_blocks(
    plan: dict[str, Any],
    sections: list[tuple[str, str]],
    supported_figures_by_anchor: dict[str, set[str]],
    provenance_by_anchor: dict[str, list[dict[str, Any]]],
) -> None:
    story_evidence = plan.get("story_evidence")
    if not isinstance(story_evidence, dict):
        return
    planned_sections = plan.get("sections") or []
    if len(sections) != len(planned_sections):
        raise RuntimeError("PAPER story block validation failed: section count changed")
    for section_index, section in enumerate(planned_sections):
        blocks = section.get("blocks")
        if not isinstance(blocks, list) or not blocks:
            raise RuntimeError("PAPER story block validation failed: missing blocks")
        beat_ids = set((section.get("story_beat") or {}).get("evidence_ids") or [])
        seen_evidence: set[str] = set()
        body_normalized = re.sub(r"\s+", "", sections[section_index][1])
        previous_position = -1
        normalized_blocks: list[tuple[dict[str, Any], str]] = []
        for block in blocks:
            if not isinstance(block, dict):
                raise RuntimeError("PAPER story block validation failed: invalid block")
            block_id = str(block.get("id") or "").strip()
            evidence_ids = [str(value).strip() for value in block.get("evidence_ids") or [] if str(value).strip()]
            text = str(block.get("text") or "").strip()
            if (
                not block_id
                or not evidence_ids
                or not text
                or not set(evidence_ids).issubset(beat_ids)
                or seen_evidence.intersection(evidence_ids)
                or any(evidence_id not in story_evidence for evidence_id in evidence_ids)
            ):
                raise RuntimeError("PAPER story block validation failed: invalid evidence assignment")
            figure_groups = {
                tuple(story_evidence[evidence_id].get("figure_ids") or [])
                for evidence_id in evidence_ids
                if story_evidence[evidence_id].get("figure_ids")
            }
            if len(figure_groups) > 1:
                raise RuntimeError("PAPER story block validation failed: mixed Figure evidence")
            block_normalized = re.sub(r"\s+", "", text)
            position = body_normalized.find(block_normalized, previous_position + 1)
            if position < 0:
                raise RuntimeError(
                    "PAPER story block validation failed: block text is not in its planned paragraph"
                )
            previous_position = position
            normalized_blocks.append((block, block_normalized))
            seen_evidence.update(evidence_ids)
        if seen_evidence != beat_ids:
            raise RuntimeError("PAPER story block validation failed: omitted or duplicated evidence")

        for evidence_id in seen_evidence:
            evidence_record = story_evidence[evidence_id]
            evidence_figures = {
                _paper_figure_id(value)
                for value in evidence_record.get("figure_ids") or []
                if str(value).strip()
            }
            for anchor in evidence_record.get("anchors") or []:
                normalized_anchor = _normalize_evidence_anchor(str(anchor))
                if (
                    not normalized_anchor
                    or re.fullmatch(r"[Pp][<>=]\d+(?:\.\d+)?", normalized_anchor)
                    or normalized_anchor in {"90%", "95%", "99%"}
                ):
                    continue
                anchor_records = provenance_by_anchor.get(normalized_anchor, [])
                figure_specific = bool(supported_figures_by_anchor.get(normalized_anchor)) or any(
                    record.get("scope") == "figure_specific" for record in anchor_records
                )
                if not figure_specific:
                    continue
                containing_blocks = [
                    block for block, block_text in normalized_blocks
                    if normalized_anchor in _normalize_evidence_anchor(block_text)
                ]
                occurrence_count = sum(
                    _normalize_evidence_anchor(block_text).count(normalized_anchor)
                    for _, block_text in normalized_blocks
                )
                if len(containing_blocks) != 1 or occurrence_count != 1:
                    raise RuntimeError(
                        "PAPER story block validation failed: figure-specific anchor is not unique to one block: "
                        f"{anchor!r}"
                    )
                block_figures = {
                    _paper_figure_id(value)
                    for value in containing_blocks[0].get("figure_ids") or evidence_figures
                    if str(value).strip()
                }
                supported = supported_figures_by_anchor.get(normalized_anchor, set())
                if supported and not block_figures.intersection(supported):
                    raise RuntimeError(
                        "PAPER evidence block figure mismatch: "
                        f"evidence={anchor!r}; block={containing_blocks[0].get('id')!r}; "
                        f"figures={sorted(block_figures)!r}; supported_figures={sorted(supported)!r}"
                    )


def _validate_paper_evidence_plan(
    plan: dict[str, Any],
    markdown: str,
    valid_source_paragraph_ids: set[str] | None = None,
    figure_evidence_bundles: list[dict[str, Any]] | None = None,
) -> None:
    sections = _paper_body_sections(markdown)
    planned_sections = plan.get("sections") or []
    if not sections or not planned_sections:
        raise RuntimeError("PAPER evidence plan validation failed: no body sections")
    bundles_by_id = {
        _paper_figure_id(bundle.get("figure_id")): bundle
        for bundle in (figure_evidence_bundles or [])
        if isinstance(bundle, dict) and bundle.get("figure_id")
    }
    supported_figures_by_anchor: dict[str, set[str]] = {}
    provenance_by_anchor: dict[str, list[dict[str, Any]]] = {}
    for figure_id, bundle in bundles_by_id.items():
        # ``quantitative_anchors`` is deliberately restricted by bundle
        # construction to figure-specific results.  Keep this compatibility
        # path for older persisted bundles that predate explicit provenance.
        for anchor in bundle.get("quantitative_anchors", []):
            supported_figures_by_anchor.setdefault(
                _normalize_evidence_anchor(str(anchor)), set()
            ).add(figure_id)
        for provenance in bundle.get("provenance", []):
            if not isinstance(provenance, dict):
                continue
            normalized = _normalize_evidence_anchor(str(provenance.get("normalized_value") or provenance.get("value") or ""))
            if not normalized:
                continue
            provenance_by_anchor.setdefault(normalized, []).append(provenance)
            if provenance.get("scope") != "figure_specific":
                continue
            supported = {
                _paper_figure_id(value)
                for value in provenance.get("supported_figures") or []
                if str(value).strip()
            }
            if supported:
                supported_figures_by_anchor.setdefault(normalized, set()).update(supported)
        for anchor, figures in (bundle.get("supported_figures_by_anchor") or {}).items():
            supported_figures_by_anchor.setdefault(
                _normalize_evidence_anchor(str(anchor)), set()
            ).update(_paper_figure_id(value) for value in figures)
    _paper_validate_story_blocks(
        plan,
        sections,
        supported_figures_by_anchor,
        provenance_by_anchor,
    )
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
        section_figure_ids = [
            _paper_figure_id(value)
            for value in (section.get("figure_ids") or section.get("selected_body_figures") or [])
            if str(value).strip()
        ]
        if bundles_by_id and not section_figure_ids:
            raise RuntimeError("PAPER evidence plan validation failed: section has no figure bundle")
        unknown_figures = set(section_figure_ids) - set(bundles_by_id)
        if unknown_figures:
            raise RuntimeError(
                "PAPER evidence plan validation failed: unknown figure bundle: "
                + ", ".join(sorted(unknown_figures))
            )
        findings = section.get("findings") or []
        if not isinstance(findings, list):
            raise RuntimeError("PAPER evidence plan validation failed: invalid findings")
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            finding_figure_ids = [
                _paper_figure_id(value)
                for value in (finding.get("figure_ids") or section_figure_ids)
                if str(value).strip()
            ]
            if bundles_by_id and not finding_figure_ids:
                raise RuntimeError("PAPER evidence plan validation failed: finding has no figure bundle")
            if bundles_by_id and not set(finding_figure_ids).issubset(set(section_figure_ids)):
                raise RuntimeError("PAPER evidence plan validation failed: finding figure mismatch")
            anchors = finding.get("anchors") or []
            if not isinstance(anchors, list):
                continue
            for anchor in anchors:
                normalized_anchor = _normalize_evidence_anchor(str(anchor))
                if (
                    not normalized_anchor
                    or re.fullmatch(r"[Pp][<>=]\d+(?:\.\d+)?", normalized_anchor)
                    or normalized_anchor in {"90%", "95%", "99%"}
                ):
                    continue
                if bundles_by_id:
                    supported_figures = supported_figures_by_anchor.get(normalized_anchor, set())
                    if supported_figures:
                        if not set(finding_figure_ids).intersection(supported_figures):
                            raise RuntimeError(
                                "PAPER evidence figure mismatch: "
                                f"evidence={anchor!r}; section={section.get('title')!r}; "
                                f"figures={finding_figure_ids!r}; supported_figures={sorted(supported_figures)!r}"
                            )
                    else:
                        contextual_records = [
                            record
                            for record in provenance_by_anchor.get(normalized_anchor, [])
                            if record.get("scope") == "section_context"
                        ]
                        if contextual_records:
                            provenance_ids = {
                                str(source_id)
                                for record in contextual_records
                                for source_id in record.get("source_paragraph_ids") or []
                            }
                            if provenance_ids and not provenance_ids.intersection(source_ids):
                                raise RuntimeError(
                                    "PAPER evidence source mismatch: "
                                    f"evidence={anchor!r}; section={section.get('title')!r}; "
                                    f"source_paragraph_ids={sorted(provenance_ids)!r}"
                                )
                actual_indexes = [
                    index
                    for index, (_, body) in enumerate(sections)
                    if normalized_anchor in _normalize_evidence_anchor(body)
                ]
                anchor_records = provenance_by_anchor.get(normalized_anchor, [])
                if not actual_indexes:
                    is_figure_specific = bool(supported_figures_by_anchor.get(normalized_anchor)) or any(
                        record.get("scope") == "figure_specific" for record in anchor_records
                    )
                    if is_figure_specific:
                        raise RuntimeError(
                            "PAPER evidence anchor missing: "
                            f"evidence={anchor!r}; planned section={section.get('title')!r}"
                        )
                    # Contextual periods, scenarios, and other anchors without
                    # Figure-specific support may be omitted from a section.
                    continue
                if not anchor_records and not supported_figures_by_anchor.get(normalized_anchor):
                    # Unclassified textual anchors are not enough to assert a
                    # unique section location; provenance remains conservative.
                    continue
                has_contextual_provenance = any(
                    record.get("scope") in {"section_context", "global_context"}
                    for record in anchor_records
                )
                if has_contextual_provenance:
                    # Contextual periods, scenarios, and other paper-level
                    # metadata may legitimately recur in several sections.
                    # They must not be forced into a single body location.
                    if any(record.get("scope") == "global_context" for record in anchor_records):
                        continue
                    if planned_index not in actual_indexes:
                        planned_title = str(section.get("title") or section.get("id") or planned_index + 1)
                        actual_titles = ", ".join(sections[index][0] for index in actual_indexes)
                        raise RuntimeError(
                            "PAPER evidence section mismatch: "
                            f"evidence={anchor!r}; planned section={planned_title!r}; "
                            f"actual section={actual_titles!r}"
                        )
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
    selected_figure_ids: set[str] | None = None,
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
        figure_ids = [
            _paper_figure_id(value)
            for value in (section.get("figure_ids") or section.get("selected_body_figures") or [])
            if str(value).strip()
        ]
        if selected_figure_ids is not None:
            if not figure_ids or not set(figure_ids).issubset(selected_figure_ids):
                raise RuntimeError("PAPER scientific planner returned an invalid figure mapping")
            section["figure_ids"] = figure_ids
        if not isinstance(findings, list) or not findings:
            raise RuntimeError("PAPER scientific planner returned a section without findings")
        for finding in findings:
            if not isinstance(finding, dict) or not str(finding.get("evidence") or "").strip():
                raise RuntimeError("PAPER scientific planner returned an invalid finding")
            finding_figure_ids = [
                _paper_figure_id(value)
                for value in (finding.get("figure_ids") or finding.get("figures") or figure_ids)
                if str(value).strip()
            ]
            if selected_figure_ids is not None:
                if not finding_figure_ids or not set(finding_figure_ids).issubset(set(figure_ids)):
                    raise RuntimeError("PAPER scientific planner returned an unbound finding")
                finding["figure_ids"] = finding_figure_ids
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


PAPER_UNFIGURED_REVIEW_PROMPT = (
    "你是Scientific Section Reviewer，只判断没有selected正文主图的候选section是否必须保留。"
    "不要评价中文文风，不要重写正文，也不要因为结构完整而保留弱内容。只有删除后会让相邻主图之间科学逻辑跳跃，"
    "或该section承载不可替代的核心科学结论时才keep，并给出简短reason；否则prune。返回严格JSON："
    '{"decisions":[{"section_id":"section-1","action":"keep"|"prune","reason":"..."}]}'
)


def _paper_remove_sections(markdown: str, keep_titles: set[str]) -> str:
    lines = markdown.splitlines()
    matches = list(
        enumerate(lines)
    )
    heading_positions = [
        index for index, line in matches if re.match(r"^##\s+", line.strip())
    ]
    if not heading_positions:
        return markdown
    output = lines[: heading_positions[0]]
    for position, start in enumerate(heading_positions):
        end = heading_positions[position + 1] if position + 1 < len(heading_positions) else len(lines)
        title = re.sub(r"^##\s+", "", lines[start].strip()).strip()
        if title in keep_titles or title in {"文章信息", "参考文献", "来源"}:
            output.extend(lines[start:end])
    return "\n".join(output).strip()


def prune_paper_sections_after_allocation(
    markdown_path: Path,
    dossier: dict[str, Any],
    allocation: dict[str, Any],
    settings: Settings,
) -> dict[str, Any]:
    plan = dossier.get("paper_evidence_plan")
    if not isinstance(plan, dict) or not isinstance(plan.get("sections"), list):
        return {"pruned": [], "retained_without_figure": []}
    sections = list(plan["sections"])
    planner_sections = [dict(section) for section in sections]
    allocation_sections = allocation.get("sections")
    if not isinstance(allocation_sections, list) or len(allocation_sections) != len(sections):
        plan["planner_sections"] = [dict(section) for section in sections]
        plan["pruning_fallback_reason"] = "allocation section mapping unavailable"
        dossier["paper_evidence_plan"] = plan
        return {"pruned": [], "retained_without_figure": []}

    body_sections = _paper_body_sections(markdown_path.read_text(encoding="utf-8"))
    if len(body_sections) != len(sections):
        plan["planner_sections"] = planner_sections
        plan["pruning_fallback_reason"] = "markdown section mapping unavailable"
        dossier["paper_evidence_plan"] = plan
        return {"pruned": [], "retained_without_figure": []}
    unfigured: list[dict[str, Any]] = []
    selected_by_index: dict[int, list[str]] = {}
    for index, record in enumerate(allocation_sections):
        selected = [str(value) for value in record.get("selected_figures") or []]
        selected_by_index[index] = selected
        if not selected:
            previous = (
                {
                    "title": sections[index - 1].get("title", ""),
                    "body": body_sections[index - 1][1],
                }
                if index > 0
                else {}
            )
            following = (
                {
                    "title": sections[index + 1].get("title", ""),
                    "body": body_sections[index + 1][1],
                }
                if index + 1 < len(sections)
                else {}
            )
            unfigured.append(
                {
                    "section_id": sections[index].get("id", ""),
                    "title": sections[index].get("title", ""),
                    "role": sections[index].get("role", ""),
                    "body": body_sections[index][1],
                    "previous_section": previous,
                    "next_section": following,
                }
            )

    decisions: dict[str, dict[str, str]] = {}
    if unfigured:
        abstract = str(
            dossier.get("abstract")
            or (dossier.get("openalex") or {}).get("abstract")
            or ""
        ).strip()
        try:
            client = OpenAI(
                api_key=settings.model_api_key,
                base_url=settings.model_base_url,
                timeout=180.0,
                max_retries=2,
            )
            response = _paper_completion_json(
                client,
                PAPER_UNFIGURED_REVIEW_PROMPT,
                {
                    "abstract": abstract,
                    "paper_evidence_plan": plan,
                    "candidates": unfigured,
                    "_model": settings.model_name,
                    "_temperature": 0.1,
                },
            )
            raw_decisions = response.get("decisions")
            if isinstance(raw_decisions, list):
                for decision in raw_decisions:
                    if not isinstance(decision, dict):
                        continue
                    section_id = str(decision.get("section_id") or "")
                    action = str(decision.get("action") or "").lower()
                    reason = str(decision.get("reason") or "").strip()
                    if section_id and action in {"keep", "prune"}:
                        decisions[section_id] = {"action": action, "reason": reason}
        except Exception:
            plan["pruning_fallback_reason"] = "unfigured section review unavailable"

    kept_indices: list[int] = []
    pruned_sections: list[dict[str, Any]] = []
    pruned_indices: list[int] = []
    retained_without_figure: list[str] = []
    for index, section in enumerate(sections):
        current = dict(section)
        section_id = str(current.get("id") or "")
        selected = selected_by_index.get(index, [])
        decision = decisions.get(section_id, {})
        if selected:
            current["selected_body_figures"] = selected
            sections[index] = current
            kept_indices.append(index)
            continue
        if decision.get("action") == "keep":
            current["retained_without_figure"] = True
            current["retention_reason"] = decision.get("reason") or "reviewer marked this as a necessary bridge or core section"
            retained_without_figure.append(section_id)
            sections[index] = current
            kept_indices.append(index)
            continue
        if decision.get("action") == "prune":
            current["pruned"] = True
            current["prune_reason"] = decision.get("reason") or "no selected body figure and not a necessary transition"
            pruned_sections.append(current)
            pruned_indices.append(index)
            continue
        # A failed or incomplete review is conservative: preserve the section.
        kept_indices.append(index)
        current["retained_without_figure"] = True
        current["retention_reason"] = "pruning reviewer did not authorize removal"
        retained_without_figure.append(section_id)
        sections[index] = current

    if not kept_indices:
        plan["pruning_fallback_reason"] = "all sections would be removed"
        kept_indices = list(range(len(sections)))
        pruned_sections = []
        pruned_indices = []
        retained_without_figure = []
    keep_titles = {str(sections[index].get("title") or "") for index in kept_indices}
    markdown_path.write_text(_paper_remove_sections(markdown_path.read_text(encoding="utf-8"), keep_titles) + "\n", encoding="utf-8")
    kept_sections = [sections[index] for index in kept_indices]
    plan["planner_sections"] = planner_sections
    plan["sections"] = kept_sections
    plan["pruned_sections"] = pruned_sections
    plan["retained_without_figure"] = retained_without_figure
    dossier["paper_evidence_plan"] = plan

    new_allocation_sections: list[dict[str, Any]] = []
    for new_index, old_index in enumerate(kept_indices):
        record = dict(allocation_sections[old_index])
        record["section_index"] = new_index
        record["section"] = str(kept_sections[new_index].get("title") or record.get("section") or "")
        new_allocation_sections.append(record)
    allocation["sections"] = new_allocation_sections
    allocation["pruned_sections"] = [
        {
            "section_id": sections[index].get("id", ""),
            "section": sections[index].get("title", ""),
            "reason": pruned_sections[position].get("prune_reason", ""),
        }
        for position, index in enumerate(pruned_indices)
    ]
    return {"pruned": [section.get("id", "") for section in pruned_sections], "retained_without_figure": retained_without_figure}


def _paper_style_exemplar() -> str:
    paths = sorted(
        PROJECT_ROOT.glob("articles/paper/*/article.md"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    preferred_tokens = (
        "2026jd046858",
        "s41467-026-77084-0",
        "2026-09-01-005",
    )
    preferred: list[Path] = []
    for token in preferred_tokens:
        match = next(
            (path for path in paths if token in path.parent.name.lower() and path not in preferred),
            None,
        )
        if match is not None:
            preferred.append(match)
    paths = preferred + [path for path in paths if path not in preferred]
    excerpts: list[str] = []
    for path in paths[:3]:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        sections = _paper_body_sections(text)
        excerpt = "\n\n".join(f"## {title}\n{body.strip()}" for title, body in sections[:3])
        if excerpt:
            excerpts.append(excerpt[:1800])
    return "\n\n---\n\n".join(excerpts)


def _paper_clean_story_text(text: str) -> str:
    """Remove backend Figure/source labels before handing evidence to story writers."""
    cleaned = str(text or "")
    cleaned = re.sub(
        r"(?i)(?<![A-Za-z])(?:supplementary|supporting)\s+(?:figure|fig)\.?\s*\d+(?:[ \t]*[a-z](?![A-Za-z]))?(?![A-Za-z0-9])",
        "",
        cleaned,
    )
    cleaned = re.sub(r"(?i)(?<![A-Za-z])(?:figure|fig)\.?\s*\d+(?:[ \t]*[a-z](?![A-Za-z]))?(?![A-Za-z0-9])", "", cleaned)
    cleaned = re.sub(r"图\s*\d+[A-Za-z]?\b", "", cleaned)
    cleaned = re.sub(r"(?i)\bsource-figure-\d+\b", "", cleaned)
    cleaned = re.sub(r"(?i)\b(?:as shown in|shown in|see)\s*,?", "", cleaned)
    cleaned = re.sub(r"如图(?:所示)?[，,：:]?", "", cleaned)
    cleaned = re.sub(r"(?i)\bpanel\s*[a-d]\b", "", cleaned)
    cleaned = re.sub(r"\(\s*[a-d]\s*\)", "", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([，。；：！？,.!?])", r"\1", cleaned)
    return cleaned.strip()


def _paper_plain_language_cleanup(text: str) -> str:
    """Translate non-essential technical shorthand without touching verified values."""
    cleaned = str(text or "")
    replacements = (
        (r"(?i)(?<![A-Za-z0-9])9[-‐]year high[-‐]pass Butterworth filter(?![A-Za-z])", "九年周期滤波方法"),
        (r"(?i)(?<![A-Za-z0-9])high[-‐]pass Butterworth filter(?![A-Za-z])", "滤波方法"),
        (r"(?i)(?<![A-Za-z0-9])standard deviation(?![A-Za-z])", "标准差"),
        (r"(?i)(?<![A-Za-z0-9])unit:\s*days(?![A-Za-z])", "单位为天"),
        (r"(?i)(?<![A-Za-z0-9])Equation\s*\d+(?![A-Za-z0-9])", "能量分解"),
        (r"(?i)(?<![A-Za-z0-9])surface albedo feedback(?![A-Za-z])", "地表反照率反馈"),
        (r"(?i)(?<![A-Za-z0-9])latent heat flux(?![A-Za-z])", "潜热通量"),
        (r"(?i)(?<![A-Za-z0-9])sensible heat flux(?![A-Za-z])", "感热通量"),
        (r"(?i)(?<![A-Za-z0-9])soil moisture(?![A-Za-z])", "土壤水分"),
        (r"(?i)(?<![A-Za-z0-9])evaporation(?![A-Za-z])", "蒸发"),
        (r"(?i)\bpositive and negative SNAO phases\b", "北大西洋涛动的正负位相"),
        (r"(?i)\bpositive Summer North Atlantic Oscillation \(SNAO\)\b", "夏季北大西洋涛动正位相"),
        (r"(?i)\bnegative SNAO years\b", "北大西洋涛动负位相年份"),
        (r"(?i)\bSummer North Atlantic Oscillation\b", "夏季北大西洋涛动"),
        (r"(?i)\bEastern Pacific \(EP\) El Niño\b", "东太平洋型厄尔尼诺"),
        (r"(?i)\bCentral Pacific \(CP\) El Niño\b", "中太平洋型厄尔尼诺"),
        (r"(?i)\bEP La Niña\b", "东太平洋型拉尼娜"),
        (r"(?i)\bCP La Niña\b", "中太平洋型拉尼娜"),
        (r"(?i)\bEl Niño\b", "厄尔尼诺"),
        (r"(?i)\bLa Niña\b", "拉尼娜"),
        (r"(?i)(?<![A-Za-z（(])ENSO(?![A-Za-z）)])", "厄尔尼诺—拉尼娜现象"),
        (r"(?i)(?<![A-Za-z（(])SNAO(?![A-Za-z）)])", "夏季北大西洋涛动"),
        (r"(?i)(?<![A-Za-z])WAF(?![A-Za-z])", "波活动通量"),
        (r"(?i)(?<![A-Za-z])JJA(?![A-Za-z])", "夏季"),
        (r"(?i)(?<![A-Za-z])DJF(?![A-Za-z])", "冬季"),
        (r"(?i)(?<![A-Za-z])SAF(?![A-Za-z])", "地表反照率反馈"),
        (r"(?i)(?<![A-Za-z])index(?![A-Za-z])", "指数"),
        (r"(?i)Correlation coefficients", "相关系数"),
        (r"(?i)regionally averaged heat-day", "区域平均高温日"),
        (r"(?i)compound hot-dry event", "复合高温干旱事件"),
        (r"(?i)dry-day", "干旱日"),
        (r"(?i)m\s*s\s*[−-]\s*1", "米/秒"),
        (r"(?i)ΔCRFs", "云辐射强迫变化"),
        (r"\(1\s*[−-]\s*α\s*\)Δ\s*S\s*↓,?\s*clr", "晴空短波辐射项"),
        (r"Δ\s*F\s*↓,?\s*clr", "晴空长波辐射项"),
        (r"Δ\s*Q", "热储存项"),
        (r"Δ\s*\(\s*H\s*\+\s*LE\s*\)", "湍流通量项"),
        (r"(?i)(?<![A-Za-z])positive(?![A-Za-z])", "正位相"),
        (r"(?i)(?<![A-Za-z])negative(?![A-Za-z])", "负位相"),
    )
    for pattern, replacement in replacements:
        cleaned = re.sub(pattern, replacement, cleaned)
    cleaned = re.sub(r"(夏季北大西洋涛动)[-‐](正位相|负位相)", r"\1\2", cleaned)
    cleaned = cleaned.replace("地表反照率反馈用于表示地表反照率反馈", "地表反照率反馈")
    cleaned = cleaned.replace("地表反照率反馈反馈", "地表反照率反馈")
    cleaned = cleaned.replace("图中的", "其中")
    cleaned = cleaned.replace("陆地—大气通量", "陆面与大气之间的交换")
    cleaned = cleaned.replace("陆地-大气通量", "陆面与大气之间的交换")
    cleaned = re.sub(r"(?i)(?<![A-Za-z])approximately\s+(?=\d+%)", "约", cleaned)
    cleaned = re.sub(r"\b((?:19|20)\d{2})\s+to\s+((?:19|20)\d{2})\b", r"\1—\2", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*[；;]\s*", "。", cleaned)
    cleaned = re.sub(r"。+", "。", cleaned)
    cleaned = re.sub(r"(?<=[一-鿿])\s+(?=[一-鿿])", "", cleaned)
    return cleaned


def _paper_clean_story_evidence(
    plan: dict[str, Any],
    source_paragraphs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, tuple[dict[str, Any], dict[str, Any]]]]:
    source_by_id = {str(record.get("id") or ""): record for record in source_paragraphs}
    evidence: list[dict[str, Any]] = []
    evidence_map: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    figure_groups: dict[tuple[str, ...], str] = {}
    counter = 1
    for section in plan.get("sections") or []:
        section_sources = [
            _paper_clean_story_text(source_by_id[source_id].get("text", ""))
            for source_id in section.get("source_paragraph_ids") or []
            if source_id in source_by_id
        ]
        for finding in section.get("findings") or []:
            if not isinstance(finding, dict):
                continue
            evidence_id = f"evidence-{counter}"
            counter += 1
            finding_figures = tuple(
                dict.fromkeys(
                    _paper_figure_id(value)
                    for value in (
                        finding.get("figure_ids")
                        or section.get("figure_ids")
                        or section.get("selected_body_figures")
                        or []
                    )
                    if str(value).strip()
                )
            )
            if finding_figures:
                if finding_figures not in figure_groups:
                    figure_groups[finding_figures] = f"evidence_group_{chr(64 + len(figure_groups) + 1)}"
                evidence_group = figure_groups[finding_figures]
            else:
                evidence_group = "context"
            source_evidence = list(dict.fromkeys(text for text in section_sources if text))
            record = {
                "evidence_id": evidence_id,
                "evidence_group": evidence_group,
                "role": str(section.get("role") or "").strip(),
                "core_finding": _paper_clean_story_text(finding.get("evidence", "")),
                "anchors": [str(anchor) for anchor in finding.get("anchors") or [] if str(anchor).strip()],
                "source_evidence": source_evidence[:8],
            }
            evidence.append(record)
            evidence_map[evidence_id] = (section, finding)
    return evidence, evidence_map


def _paper_validate_story_plan(
    story_plan: dict[str, Any],
    valid_evidence_ids: set[str],
) -> list[dict[str, Any]]:
    brief = story_plan.get("editorial_brief")
    if not isinstance(brief, dict) or not all(
        isinstance(brief.get(field), str) and brief[field].strip()
        for field in ("audience", "purpose", "tone", "reader_should_leave_with", "story_question")
    ):
        raise RuntimeError("PAPER story planner returned an invalid editorial brief")
    beats = story_plan.get("story_beats")
    if not isinstance(beats, list) or not 1 <= len(beats) <= 4:
        raise RuntimeError("PAPER story planner returned an invalid beat count")
    seen_ids: set[str] = set()
    used_evidence: set[str] = set()
    validated: list[dict[str, Any]] = []
    for beat_index, beat in enumerate(beats):
        if not isinstance(beat, dict):
            raise RuntimeError("PAPER story planner returned an invalid story beat")
        beat_id = str(beat.get("id") or "").strip()
        title = _paper_clean_story_text(beat.get("title", ""))
        reader_question = _paper_clean_story_text(beat.get("reader_question", ""))
        core_message = _paper_clean_story_text(beat.get("core_message", ""))
        transition = _paper_clean_story_text(beat.get("transition_to_next", ""))
        evidence_ids = [str(value).strip() for value in beat.get("evidence_ids") or [] if str(value).strip()]
        if (
            not beat_id
            or beat_id in seen_ids
            or not title
            or not reader_question
            or not core_message
            or not evidence_ids
            or (not transition and beat_index != len(beats) - 1)
            or not set(evidence_ids).issubset(valid_evidence_ids)
        ):
            raise RuntimeError("PAPER story planner returned invalid story beat fields")
        if used_evidence.intersection(evidence_ids):
            raise RuntimeError("PAPER story planner reused evidence across story beats")
        seen_ids.add(beat_id)
        used_evidence.update(evidence_ids)
        validated.append(
            {
                "id": beat_id,
                "title": title,
                "reader_question": reader_question,
                "core_message": core_message,
                "evidence_ids": list(dict.fromkeys(evidence_ids)),
                "transition_to_next": transition or "文章收束。",
            }
        )
    if used_evidence != valid_evidence_ids:
        raise RuntimeError("PAPER story planner omitted or invented evidence")
    story_plan["editorial_brief"] = {
        field: _paper_clean_story_text(brief[field])
        for field in ("audience", "purpose", "tone", "reader_should_leave_with", "story_question")
    }
    story_plan["story_beats"] = validated
    return validated


def _paper_story_sections(
    story_beats: list[dict[str, Any]],
    evidence_map: dict[str, tuple[dict[str, Any], dict[str, Any]]],
) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for beat in story_beats:
        source_ids: list[str] = []
        figure_ids: list[str] = []
        roles: list[str] = []
        findings: list[dict[str, Any]] = []
        seen_findings: set[int] = set()
        for evidence_id in beat["evidence_ids"]:
            original_section, finding = evidence_map[evidence_id]
            role = str(original_section.get("role") or "").strip()
            if role and role not in roles:
                roles.append(role)
            for source_id in original_section.get("source_paragraph_ids") or []:
                if source_id not in source_ids:
                    source_ids.append(source_id)
            for figure_id in original_section.get("figure_ids") or original_section.get("selected_body_figures") or []:
                normalized = _paper_figure_id(figure_id)
                if normalized not in figure_ids:
                    figure_ids.append(normalized)
            marker = id(finding)
            if marker not in seen_findings:
                findings.append(finding)
                seen_findings.add(marker)
        sections.append(
            {
                "id": beat["id"],
                "title": beat["title"],
                "role": roles[0] if len(roles) == 1 else "story",
                "figure_ids": figure_ids,
                "source_paragraph_ids": source_ids,
                "findings": findings,
                "story_beat": {
                    "reader_question": beat["reader_question"],
                    "core_message": beat["core_message"],
                    "evidence_ids": beat["evidence_ids"],
                    "transition_to_next": beat["transition_to_next"],
                },
            }
        )
    return sections


def _paper_apply_story_output(
    sections: list[dict[str, Any]],
    generated: list[dict[str, Any]],
    evidence_map: dict[str, tuple[dict[str, Any], dict[str, Any]]],
) -> None:
    """Attach block-local Figure bindings after the clean writer stage."""
    by_id = {str(item.get("id") or ""): item for item in generated}
    for section in sections:
        item = by_id.get(str(section.get("id") or ""))
        if item is None:
            raise RuntimeError("PAPER story output omitted a planned section")
        blocks: list[dict[str, Any]] = []
        raw_blocks = item.get("blocks") or []
        if not raw_blocks and isinstance(item.get("body"), str) and item["body"].strip():
            raw_blocks = [{
                "id": f"{section['id']}-block-1",
                "evidence_ids": list((section.get("story_beat") or {}).get("evidence_ids") or []),
                "text": item["body"],
            }]
        for block in raw_blocks:
            figure_ids: list[str] = []
            for evidence_id in block.get("evidence_ids") or []:
                original_section, finding = evidence_map[evidence_id]
                for value in (
                    finding.get("figure_ids")
                    or original_section.get("figure_ids")
                    or original_section.get("selected_body_figures")
                    or []
                ):
                    normalized = _paper_figure_id(value)
                    if normalized not in figure_ids:
                        figure_ids.append(normalized)
            blocks.append(
                {
                    "id": str(block["id"]),
                    "evidence_ids": list(block["evidence_ids"]),
                    "text": str(block["text"]).strip(),
                    "figure_ids": figure_ids,
                }
            )
        if not blocks:
            raise RuntimeError("PAPER story output returned no blocks")
        section["title"] = str(item["title"])
        section["blocks"] = blocks
        section["body"] = "\n\n".join(block["text"] for block in blocks)


def _paper_story_draft_blocks(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "beat_id": str(section.get("id") or ""),
            "title": str(section.get("title") or ""),
            "blocks": [
                {
                    "id": block.get("id", ""),
                    "evidence_ids": list(block.get("evidence_ids") or []),
                    "text": str(block.get("text") or ""),
                }
                for block in section.get("blocks") or []
            ],
        }
        for section in sections
    ]


def _paper_story_planner(
    client: OpenAI,
    clean_evidence: list[dict[str, Any]],
    style_exemplar: str,
    model: str,
    feedback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _paper_completion_json(
        client,
        PAPER_STORY_PLANNER_PROMPT,
        {
            "editorial_brief_request": {
                "audience": "跨专业、受过高等教育但非该领域专家的读者",
                "purpose": "用几分钟讲清论文最值得知道的科学发现",
                "tone": "清楚、自然、克制、有解释感，不像论文、汇报或营销稿",
            },
            "clean_evidence": clean_evidence,
            "style_exemplar": style_exemplar,
            "targeted_feedback": feedback or {},
            "_model": model,
            "_temperature": 0.15,
        },
    )


def _paper_normalize_story_output(
    response: dict[str, Any],
    story_plan: dict[str, Any],
    clean_evidence: list[dict[str, Any]],
    stage: str,
) -> list[dict[str, Any]]:
    sections = response.get("sections")
    if not isinstance(sections, list) or len(sections) != len(story_plan["story_beats"]):
        raise RuntimeError(f"PAPER {stage} returned invalid sections")
    evidence_by_id = {record["evidence_id"]: record for record in clean_evidence}
    by_id = {str(item.get("id") or ""): item for item in sections if isinstance(item, dict)}
    output: list[dict[str, Any]] = []
    for beat in story_plan["story_beats"]:
        item = by_id.get(beat["id"])
        if item is None:
            raise RuntimeError(f"PAPER {stage} omitted a story beat")
        raw_blocks = item.get("blocks")
        if raw_blocks is None:
            body = item.get("body")
            if not isinstance(body, str) or not body.strip():
                raise RuntimeError(f"PAPER {stage} returned no evidence blocks")
            raw_blocks = [{"id": f"{beat['id']}-block-1", "evidence_ids": beat["evidence_ids"], "text": body}]
        if not isinstance(raw_blocks, list) or not raw_blocks:
            raise RuntimeError(f"PAPER {stage} returned invalid evidence blocks")
        seen_evidence: set[str] = set()
        blocks: list[dict[str, Any]] = []
        for block_index, raw_block in enumerate(raw_blocks, start=1):
            if not isinstance(raw_block, dict):
                raise RuntimeError(f"PAPER {stage} returned an invalid evidence block")
            block_id = str(raw_block.get("id") or f"{beat['id']}-block-{block_index}").strip()
            evidence_ids = [
                str(value).strip()
                for value in raw_block.get("evidence_ids") or []
                if str(value).strip()
            ]
            text = raw_block.get("text", raw_block.get("body"))
            if (
                not block_id
                or not evidence_ids
                or not isinstance(text, str)
                or not text.strip()
                or block_id in {block["id"] for block in blocks}
                or not set(evidence_ids).issubset(set(beat["evidence_ids"]))
                or seen_evidence.intersection(evidence_ids)
            ):
                raise RuntimeError(f"PAPER {stage} returned invalid evidence block fields")
            groups = {
                str(evidence_by_id[evidence_id].get("evidence_group") or "context")
                for evidence_id in evidence_ids
                if evidence_id in evidence_by_id
                and str(evidence_by_id[evidence_id].get("evidence_group") or "context") != "context"
            }
            if len(groups) > 1:
                raise RuntimeError(f"PAPER {stage} mixed different Figure evidence groups in one block")
            clean_text = _paper_plain_language_cleanup(_paper_clean_story_text(text))
            clean_text = re.sub(r"\s*\n+\s*", " ", clean_text).strip()
            blocks.append({"id": block_id, "evidence_ids": evidence_ids, "text": clean_text})
            seen_evidence.update(evidence_ids)
        if seen_evidence != set(beat["evidence_ids"]):
            raise RuntimeError(f"PAPER {stage} omitted or duplicated evidence across blocks")
        title = _paper_clean_story_text(item.get("title") or beat["title"]) or beat["title"]
        output.append({
            "id": beat["id"],
            "title": _paper_plain_language_cleanup(title),
            "blocks": blocks,
            "body": "\n\n".join(block["text"] for block in blocks),
        })
    return output


def _paper_story_writer(
    client: OpenAI,
    story_plan: dict[str, Any],
    clean_evidence: list[dict[str, Any]],
    style_exemplar: str,
    model: str,
    feedback: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Write one story beat per completion while keeping its evidence blocks isolated."""
    evidence_by_id = {record["evidence_id"]: record for record in clean_evidence}
    output: list[dict[str, Any]] = []
    for beat in story_plan["story_beats"]:
        beat_evidence = [evidence_by_id[evidence_id] for evidence_id in beat["evidence_ids"]]
        response = _paper_completion_json(
            client,
            PAPER_STORY_WRITER_PROMPT,
            {
                "editorial_brief": story_plan["editorial_brief"],
                "story_beat": beat,
                "clean_evidence": beat_evidence,
                "style_exemplar": style_exemplar,
                "targeted_feedback": feedback or {},
                "_model": model,
                "_temperature": 0.25,
            },
        )
        if isinstance(response.get("sections"), list):
            candidate = response
        else:
            candidate = {
                "sections": [{
                    "id": beat["id"],
                    "title": response.get("title") or beat["title"],
                    "blocks": response.get("blocks"),
                }]
            }
        normalized = _paper_normalize_story_output(
            candidate,
            {"story_beats": [beat]},
            clean_evidence,
            "story writer",
        )
        output.append(normalized[0])
    return output


def _paper_humanize_story(
    client: OpenAI,
    story_plan: dict[str, Any],
    clean_evidence: list[dict[str, Any]],
    draft_blocks: list[dict[str, Any]],
    model: str,
    feedback: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Humanize one block per completion so evidence cannot cross block boundaries."""
    evidence_by_id = {record["evidence_id"]: record for record in clean_evidence}
    draft_by_beat = {
        str(item.get("beat_id") or item.get("id") or ""): item
        for item in draft_blocks
        if isinstance(item, dict)
    }
    output: list[dict[str, Any]] = []
    for beat in story_plan["story_beats"]:
        beat_id = beat["id"]
        draft_item = draft_by_beat.get(beat_id) or {}
        current_blocks = draft_item.get("blocks") or []
        if not current_blocks:
            raise RuntimeError("PAPER humanizer received no blocks for a story beat")
        humanized_blocks: list[dict[str, Any]] = []
        for block in current_blocks:
            evidence_ids = [str(value).strip() for value in block.get("evidence_ids") or [] if str(value).strip()]
            response = _paper_completion_json(
                client,
                PAPER_HUMANIZER_PROMPT,
                {
                    "story_beat": {
                        "id": beat_id,
                        "title": beat["title"],
                        "reader_question": beat["reader_question"],
                        "core_message": beat["core_message"],
                    },
                    "current_block": {
                        "id": block.get("id", ""),
                        "evidence_ids": evidence_ids,
                        "text": str(block.get("text") or ""),
                    },
                    "clean_evidence": [evidence_by_id[evidence_id] for evidence_id in evidence_ids],
                    "targeted_feedback": feedback or {},
                    "_model": model,
                    "_temperature": 0.2,
                },
            )
            result = response.get("block") if isinstance(response.get("block"), dict) else response
            if not isinstance(result, dict):
                raise RuntimeError("PAPER humanizer returned an invalid block")
            returned_ids = [str(value).strip() for value in result.get("evidence_ids") or evidence_ids if str(value).strip()]
            if returned_ids != evidence_ids:
                raise RuntimeError("PAPER humanizer changed block evidence assignment")
            text = result.get("text", result.get("body"))
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError("PAPER humanizer returned an empty block")
            clean_text = _paper_plain_language_cleanup(_paper_clean_story_text(text))
            clean_text = re.sub(r"\s*\n+\s*", " ", clean_text).strip()
            humanized_blocks.append({"id": str(block.get("id") or result.get("id") or ""), "evidence_ids": evidence_ids, "text": clean_text})
        output.append({
            "id": beat_id,
            "title": _paper_plain_language_cleanup(
                _paper_clean_story_text(str(draft_item.get("title") or beat["title"]))
            ),
            "blocks": humanized_blocks,
            "body": "\n\n".join(block["text"] for block in humanized_blocks),
        })
    return output


def _paper_completion_json(client: OpenAI, system_prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
    model = payload.pop("_model")
    temperature = payload.pop("_temperature", 0.2)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    parse_error: Exception | None = None
    for attempt in range(2):
        response = _paper_completion_with_retry(
            client,
            model=model,
            temperature=temperature,
            messages=messages,
        )
        try:
            parsed = _json_from_text(response.choices[0].message.content or "")
            if not isinstance(parsed, dict):
                raise RuntimeError("PAPER model returned a non-object JSON response")
            return parsed
        except (TypeError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
            parse_error = exc
            if attempt == 0:
                continue
    raise RuntimeError(f"PAPER model returned invalid JSON after retry: {parse_error}") from parse_error


def _paper_plan(
    client: OpenAI,
    abstract: str,
    paper_text: str,
    source_paragraphs: list[dict[str, Any]],
    metadata: dict[str, Any],
    selected_figure_ids: list[str] | None = None,
    figure_evidence_bundles: list[dict[str, Any]] | None = None,
    validation_feedback: str = "",
) -> dict[str, Any]:
    return _paper_completion_json(
        client,
        PAPER_PLANNER_PROMPT,
        {
            "abstract": abstract,
            "paper_text": paper_text,
            "source_paragraphs": source_paragraphs,
            "metadata": metadata,
            "selected_body_figures": selected_figure_ids or [],
            "figure_evidence_bundles": figure_evidence_bundles or [],
            "validation_feedback": validation_feedback,
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
    figure_evidence_bundles: list[dict[str, Any]],
    model: str,
) -> str:
    source_ids = set(section["source_paragraph_ids"])
    section_figure_ids = {
        _paper_figure_id(value)
        for value in section.get("figure_ids") or section.get("selected_body_figures") or []
    }
    section_bundles = [
        bundle
        for bundle in figure_evidence_bundles
        if _paper_figure_id(bundle.get("figure_id")) in section_figure_ids
    ]
    directly_associated_ids = {
        str(source_id)
        for bundle in section_bundles
        for source_id in bundle.get("directly_associated_source_paragraph_ids") or []
    }
    section_sources = [
        record
        for record in source_paragraphs
        if record["id"] in source_ids
        and record["id"] in directly_associated_ids
    ]
    payload = {
        # The Planner has already used the authoritative Abstract to define
        # the section findings.  Writers must not reuse unrelated Abstract
        # results as evidence for the current Figure bundle.
        "abstract": "",
        "abstract_context": "Abstract structure and lead are locked; use only the current section findings and Figure bundles below for evidence.",
        "section": section,
        "figure_evidence_bundles": section_bundles,
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
        blocks = section.get("blocks") or []
        if blocks:
            body = "\n\n".join(str(block.get("text") or "").strip() for block in blocks).strip()
        else:
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
            "figure_evidence_bundles": plan.get("figure_evidence_bundles", []),
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
    normalized_corrections: list[dict[str, str]] = []
    for correction in corrections:
        if not isinstance(correction, dict):
            raise RuntimeError("PAPER scientific reviewer returned an invalid correction")
        section = str(correction.get("section") or correction.get("section_id") or "").strip()
        evidence = str(correction.get("evidence") or "").strip()
        issue = str(correction.get("issue") or "").strip()
        instruction = str(correction.get("correction") or correction.get("instruction") or "").strip()
        if not section or not evidence or not issue or not instruction:
            raise RuntimeError("PAPER scientific reviewer returned an incomplete correction")
        normalized_corrections.append(
            {
                "section": section,
                "evidence": evidence,
                "issue": issue,
                "correction": instruction,
            }
        )
    if status == "needs_revision" and not normalized_corrections:
        raise RuntimeError("PAPER scientific reviewer requested revision without corrections")
    response["corrections"] = normalized_corrections
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
    local_source_ids: set[str] = set()
    bundles = plan.get("figure_evidence_bundles") or []
    for section in plan.get("sections") or []:
        section_figures = {
            _paper_figure_id(value)
            for value in section.get("figure_ids") or section.get("selected_body_figures") or []
        }
        local_source_ids.update(
            str(source_id)
            for bundle in bundles
            if _paper_figure_id(bundle.get("figure_id")) in section_figures
            for source_id in bundle.get("directly_associated_source_paragraph_ids") or []
        )
    local_sources = [
        record
        for record in source_paragraphs
        if str(record.get("id") or "") in local_source_ids
    ]
    payload = {
        "abstract": abstract,
        "paper_text": paper_text,
        "source_paragraphs": local_sources,
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
    original_bodies = _paper_extract_section_bodies(draft, plan["sections"])
    by_id = {str(section.get("id") or ""): section for section in revised if isinstance(section, dict)}
    bodies: list[str] = []
    changed = False
    for planned, original in zip(plan["sections"], original_bodies):
        item = by_id.get(str(planned["id"]))
        if item is None:
            bodies.append(original)
            continue
        revised_title = str(item.get("title") or "").strip()
        if revised_title:
            planned["title"] = revised_title
        if not isinstance(item.get("body"), str) or not item["body"].strip():
            raise RuntimeError("PAPER scientific revision returned an empty section")
        bodies.append(_paper_section_body({"body": item["body"]}))
        changed = True
    if not changed:
        raise RuntimeError("PAPER scientific revision returned no recognized sections")
    return bodies


def _paper_editorial_rewrite(
    client: OpenAI,
    plan: dict[str, Any],
    draft: str,
    model: str,
    style_exemplar: str,
    lint_feedback: dict[str, Any] | None = None,
    abstract: str = "",
    story_context: dict[str, Any] | None = None,
) -> list[str]:
    if story_context is not None:
        payload = {
            "abstract": abstract,
            "story_first_mode": True,
            "editorial_brief": story_context["story_plan"]["editorial_brief"],
            "story_beats": story_context["story_plan"]["story_beats"],
            "clean_evidence": story_context["clean_evidence"],
            "draft": _paper_clean_story_text(draft),
            "style_exemplar": style_exemplar,
            "audit_feedback": lint_feedback or {},
        }
    else:
        section_bundles = {
            str(section.get("id") or ""): [
                bundle
                for bundle in plan.get("figure_evidence_bundles") or []
                if _paper_figure_id(bundle.get("figure_id")) in {
                    _paper_figure_id(value)
                    for value in section.get("figure_ids") or section.get("selected_body_figures") or []
                }
            ]
            for section in plan.get("sections") or []
        }
        payload = {
            "abstract": abstract,
            "paper_evidence_plan": plan,
            "verified_key_findings": [
                {
                    "section_id": section.get("id"),
                    "title": section.get("title"),
                    "role": section.get("role"),
                    "findings": section.get("findings") or [],
                    "figure_bundles": section_bundles.get(str(section.get("id") or ""), []),
                }
                for section in plan.get("sections") or []
            ],
            "draft": draft,
            "style_exemplar": style_exemplar,
            "audit_feedback": lint_feedback or {},
        }
    response_obj = _paper_completion_with_retry(
        client,
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": PAPER_POPULAR_SCIENCE_EDITOR_PROMPT},
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
        revised_title = str(item.get("title") or "").strip()
        if revised_title:
            planned["title"] = revised_title
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


def _paper_chinese_char_count(text: str) -> int:
    return len(re.findall(r"[㐀-鿿]", str(text or "")))


def _paper_body_length_audit(markdown: str) -> dict[str, Any]:
    sections = _paper_body_sections(markdown)
    section_records = [
        {
            "title": title,
            "characters": _paper_chinese_char_count(body),
        }
        for title, body in sections
    ]
    return {
        "sections": section_records,
        "total_characters": sum(item["characters"] for item in section_records),
        "overlong_sections": [
            item["title"] for item in section_records if item["characters"] > 135
        ],
        "total_overlong": sum(item["characters"] for item in section_records) > 500,
    }


_READABILITY_FOREIGN_TERMS = (
    "spread", "model", "models", "future", "projection", "correlation", "scenario",
    "result", "results", "trend", "trends", "pattern", "patterns",
)


_READABILITY_JARGON = (
    "NSWS", "AMIP", "CMIP6", "XGBoost", "SHAP", "CCA", "EOF", "PC1", "PC2",
    "NAO", "ENSO", "SSP", "模式间离散度", "模式间离散", "模式间标准差", "趋势标准差",
    "显著性水平", "典型相关", "主成分", "回归系数", "相关系数", "地表能量收支",
    "地表静稳化",
)


def _paper_readability_audit(markdown: str) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    for title, body in _paper_body_sections(markdown):
        normalized_body = str(body or "")
        for term in _READABILITY_FOREIGN_TERMS:
            if re.search(rf"\b{re.escape(term)}\b", f"{title} {normalized_body}", re.IGNORECASE):
                issues.append({
                    "section": title,
                    "type": "english_prose_term",
                    "term": term,
                    "suggestion": "改写成普通中文，只有必要的已验证anchor或方法名可以保留。",
                })
        for acronym in re.findall(r"(?<![A-Za-z])[A-Z]{2,}\d*(?![A-Za-z])", normalized_body):
            if acronym in {"Fig", "DOI"} or re.fullmatch(r"SSP\d+", acronym):
                continue
            explained = re.search(
                rf"(?:[（(][^）)]*\b{re.escape(acronym)}\b[^）)]*[）)]|"
                rf"\b{re.escape(acronym)}\b[^。！？\n]{0,12}[（(：:])",
                normalized_body,
            )
            scenario_explained = re.search(
                rf"\b{re.escape(acronym)}(?:[-‐–]\d+(?:\.\d+)?)?[^。！？\n]{{0,16}}(?:情景|排放情景)",
                normalized_body,
            )
            if not explained and not scenario_explained and not re.search(rf"\b{re.escape(acronym)}(?:[-‐–]\d+(?:\.\d+)?)?\s*[（(]", normalized_body):
                issues.append({"section": title, "type": "unexplained_acronym", "term": acronym})
        for sentence in re.split(r"(?<=[。！？.!?])\s*", normalized_body):
            sentence = sentence.strip()
            if not sentence:
                continue
            jargon_hits = [term for term in _READABILITY_JARGON if term in sentence]
            number_hits = re.findall(r"(?:R\s*=|\d+(?:\.\d+)?%|\d{4})", sentence)
            if _paper_chinese_char_count(sentence) > 72:
                issues.append({"section": title, "type": "long_sentence", "text": sentence[:120]})
            if len(jargon_hits) >= 3 or (len(jargon_hits) >= 2 and len(number_hits) >= 2):
                issues.append({
                    "section": title,
                    "type": "dense_technical_sentence",
                    "terms": jargon_hits,
                    "text": sentence[:120],
                })
        plain_term_hits = [
            term for term in (
                "模式间离散度", "模式间离散", "模式间标准差", "趋势标准差",
                "显著性水平", "典型相关", "主成分", "地表能量收支", "地表静稳化",
            )
            if term in normalized_body
            and not any(
                other != term and term in other and other in normalized_body
                for other in (
                    "模式间离散度", "模式间标准差", "趋势标准差", "显著性水平",
                    "典型相关", "主成分", "地表能量收支", "地表静稳化",
                )
            )
        ]
        for term in plain_term_hits:
            issues.append({
                "section": title,
                "type": "technical_term_needs_explanation",
                "term": term,
                "suggestion": "先用普通中文解释，再保留必要的专业词。",
            })
    return {"issues": issues, "issue_count": len(issues)}


def _paper_title_style_lint(markdown: str) -> dict[str, Any]:
    """Flag media-style section titles without changing scientific content."""
    terms = (
        "为何", "线索", "改写", "同一片中国", "谁在主导", "真正的答案",
        "背后的秘密", "决定了什么", "正在发生什么", "揭示出", "锁定",
    )
    hits = [
        {"title": title, "term": term}
        for title, _ in _paper_body_sections(markdown)
        for term in terms
        if term in title
    ]
    return {"issues": hits, "issue_count": len(hits)}


def _paper_stop_slop_audit(markdown: str) -> dict[str, Any]:
    """Audit final story shape without attempting another full rewrite."""
    sections = _paper_body_sections(markdown)
    issues: list[dict[str, Any]] = []
    figure_refs = re.findall(
        r"(?i)(?:\b(?:figure|fig)\.?\s*\d+\s*[a-z]?\b|图\s*\d+\s*[A-Za-z]?|图中|图表|图示)",
        markdown,
    )
    if figure_refs:
        issues.append({"type": "figure_reportage", "count": len(figure_refs)})
    numbered = re.findall(r"(?:第一|第二|第三|第四|首先|其次|再次|最后)", markdown)
    if numbered:
        issues.append({"type": "numbered_structure", "terms": list(dict.fromkeys(numbered))})
    lint = _paper_ai_style_lint(markdown)
    if _paper_ai_style_lint_failed(lint):
        issues.append({"type": "ai_connective_template", "counts": lint})
    title_lint = _paper_title_style_lint(markdown)
    if title_lint["issue_count"]:
        issues.append({"type": "media_style_title", "details": title_lint["issues"]})
    semicolon_count = markdown.count("；") + markdown.count(";")
    if semicolon_count:
        issues.append({"type": "semicolon_overuse", "count": semicolon_count})
    first_sentences = []
    for _, body in sections:
        sentence = re.split(r"(?<=[。！？.!?])\s*", body.strip(), maxsplit=1)[0]
        if sentence:
            first_sentences.append(_normalize_evidence_anchor(sentence))
    if len(first_sentences) >= 3 and len(set(first_sentences)) < len(first_sentences):
        issues.append({"type": "repeated_section_opening"})
    sentence_lengths = [
        _paper_chinese_char_count(sentence.strip())
        for _, body in sections
        for sentence in re.split(r"(?<=[。！？.!?])\s*", body)
        if sentence.strip()
    ]
    if len(sentence_lengths) >= 4 and max(sentence_lengths) - min(sentence_lengths) <= 12:
        issues.append({"type": "uniform_sentence_rhythm"})
    readability = _paper_readability_audit(markdown)
    if readability["issue_count"] >= 3:
        issues.append({
            "type": "high_technical_density",
            "count": readability["issue_count"],
            "examples": readability["issues"][:5],
        })
    section_count = len(sections)
    substantial_sections = sum(_paper_chinese_char_count(body) >= 35 for _, body in sections)
    metrics = {
        "directness": max(0.0, 1.0 - min(1.0, len(figure_refs) / 2)),
        "rhythm": max(0.0, 1.0 - min(1.0, sum(item["type"] == "uniform_sentence_rhythm" for item in issues))),
        "naturalness": max(0.0, 1.0 - min(1.0, len(numbered) / 4)),
        "information_density": round(substantial_sections / section_count, 2) if section_count else 0.0,
        "template_risk": round(min(1.0, len(issues) / 5), 2),
    }
    return {"metrics": metrics, "issues": issues, "issue_count": len(issues)}


def _paper_editor_anchor_audit(
    before: str,
    after: str,
    plan: dict[str, Any],
) -> dict[str, Any]:
    before_sections = _paper_body_sections(before)
    after_sections = _paper_body_sections(after)
    issues: list[dict[str, Any]] = []
    anchors: set[str] = set()
    for section in plan.get("sections") or []:
        for finding in section.get("findings") or []:
            for anchor in finding.get("anchors") or []:
                normalized = _normalize_evidence_anchor(str(anchor))
                if normalized:
                    anchors.add(normalized)
    for anchor in sorted(anchors):
        before_indexes = [
            index for index, (_, body) in enumerate(before_sections)
            if anchor in _normalize_evidence_anchor(body)
        ]
        after_indexes = [
            index for index, (_, body) in enumerate(after_sections)
            if anchor in _normalize_evidence_anchor(body)
        ]
        if before_indexes and not after_indexes:
            issues.append({"type": "missing_anchor", "anchor": anchor})
        elif before_indexes and before_indexes != after_indexes:
            issues.append({
                "type": "anchor_section_moved",
                "anchor": anchor,
                "before_sections": before_indexes,
                "after_sections": after_indexes,
            })
    return {"issues": issues, "issue_count": len(issues)}


def _paper_editor_feedback(abstract: str, markdown: str) -> dict[str, Any]:
    body_lengths = _paper_body_length_audit(markdown)
    readability = _paper_readability_audit(markdown)
    abstract_length = _paper_chinese_char_count(abstract)
    return {
        "abstract_characters": abstract_length,
        # Abstract is a complete translation; length is informational only.
        "abstract_overlong": False,
        "body_lengths": body_lengths,
        "readability": readability,
        "title_style": _paper_title_style_lint(markdown),
    }


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
    """Translate the original paper Abstract faithfully and lock its structure."""
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
                    "你是中文科技论文摘要翻译编辑。原始Abstract是唯一来源。"
                    "按原文逻辑、顺序和段落关系做忠实中文翻译，只做必要的中文语序调整。"
                    "完整保留原文的重要背景、研究问题、方法范围、结果、数字、因果强度和限定条件。"
                    "不得压缩、删去关键结果、重新组织科学结构、自由总结、增加意义或补充原文没有的结论。"
                    "不要让公众号文风、标题风格或正文内容影响摘要。不要添加小标题、列表或解释，只返回严格JSON："
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
    "Abstract结构和导语已经锁定；正文只能使用当前section findings、Figure bundles和source_paragraphs中的证据，不得从abstract_context引入当前bundle未支持的其他结果。"
    "保留数字、趋势方向、时间范围、变量关系和因果强度；correlation不写成causation。"
    "正文应自然、简洁、信息密度高，不设置每个section的固定字数；避免翻译腔、空泛总结和重复连接词。若当前section标记为retained_without_figure，只保留理解相邻主图所需的极短桥接内容，不展开次要机制或补充材料。"
    "如果有可核验的paper_text原句，可以保留短Markdown引用块，但不得改写或编造。"
)

PAPER_FIDELITY_CONTRACT = (
    "保真约束：原始事实、数字及其修饰对象、主体和来源不变；correlation不能写成causation；"
    "uncertainty、suggest、indicate、可能、表明等限定不能强化；不新增原文没有的机制、原因或意义。"
    "能用普通中文解释术语就解释，但不能为追求自然而牺牲科学准确性。"
)

PAPER_STORY_PLANNER_PROMPT = (
    "你是Story Planner，先读懂房间，再为已经通过Figure-first科学验证的证据设计自然的公众号故事线。"
    "读者是跨专业、受过高等教育但非该领域专家的人；目的不是逐项汇报结果，而是用几分钟讲清论文最值得知道的发现。"
    "先确定editorial_brief：audience、purpose、tone、reader_should_leave_with（读者记住的2到3个观点）和story_question。"
    "再把clean_evidence组织成2到4个story beats，通常约3个但不要硬凑。每个beat包含id、title、reader_question、core_message、"
    "evidence_ids和transition_to_next。故事优先遵循问题—发现—为什么—意义/未来，而不是按资料顺序或编号排列。"
    "允许多个证据共同进入一个beat；标题必须专业、直接、简洁，优先10到22个中文字，直接陈述科学结果。避免为何、线索、改写、同一片中国、谁在主导、真正的答案、背后的秘密等媒体化措辞。不能使用第一、第二、第三、第四、首先、其次、最后，也不能提及任何图、Figure、panel或source。"
    "只学习style_exemplar的中文节奏、句长、信息密度和推进方式，不复制其中的科学事实、数字、地点、机制或句子。"
    "返回严格JSON："
    '{"editorial_brief":{"audience":"...","purpose":"...","tone":"...","reader_should_leave_with":"...","story_question":"..."},'
    '"story_beats":[{"id":"beat-1","title":"...","reader_question":"...","core_message":"...","evidence_ids":["evidence-1"],"transition_to_next":"..."}]}'
)

PAPER_STORY_WRITER_PROMPT = (
    "你是Story Writer，为跨专业科研读者写专业、简洁、易懂的中文科学公众号正文。你只能看到按beat分组的clean evidence和story beats，"
    "绝不能提及或猜测图号、Figure、panel、source id，也不要按证据编号或资料顺序逐项汇报。"
    "每个beat只能使用其对应的clean evidence，先回答读者问题，再给最重要的发现，随后用直接句解释如何理解；不要强行制造承上启下的金句。"
    "标题应专业、直接、简洁，优先10到22个中文字，直接陈述科学结果，不用为何、线索、改写、同一片中国等媒体化表达。"
    "正文不要写成论文Results、摘要扩写、图注翻译或科普新闻稿。避免第一/第二/第三/第四、首先/其次/最后、模板化排比和不必要的分号。"
    "每句话只讲一个主要科学意思，中文逗号和句号为主。保留必要专业词，第一次出现时用简短中文解释；不要为了通俗创造比喻或抽象术语。"
    "每个beat必须拆成一个或多个paragraph blocks。一个block只能使用同一个evidence_group的Figure-specific evidence；背景性的global_context或section_context可以陪同，但不能携带另一组Figure的核心定量结果。"
    "clean_evidence中的每个anchor都必须在包含对应evidence_id的block正文中原样保留；不能跨block移动、重复或删除已验证anchor。"
    "正文总量以约350到500个中文字符为目标；按故事需要分配篇幅，不要把每个beat或block机械写成等长。"
    + PAPER_FIDELITY_CONTRACT
    + "本次只写当前story beat，返回严格JSON：{\"title\":\"...\",\"blocks\":[{\"id\":\"block-1\",\"evidence_ids\":[\"evidence-1\"],\"text\":\"...\"}]}。"
)

PAPER_HUMANIZER_PROMPT = (
    "你是中文母语科学编辑，依据ai-zixun/humanizer-zh的原则，对Story Writer成稿做一次保守的人文化编辑。"
    "输入按beat再按paragraph block分组；每次只能修改当前block，不能看到或重写其他block的正文。"
    "每个block的evidence_ids和顺序是硬边界，不能合并block、拆出跨组句子、移动finding或把另一组Figure的结果带进来。"
    "保持专业、直接、简洁的科研公众号中文，去掉翻译腔、空泛总结、机械连接、过度修辞和不必要分号。"
    "标题应专业、直接、简洁，优先陈述科学结果，避免为何、线索、改写、同一片中国等媒体化措辞。"
    "每句话只讲一个主要科学意思；保留必要专业词并做简短解释，不为了通俗创造比喻、抽象术语或媒体式悬念。"
    "每个block对应的anchor必须原样保留，不能因润色而删除、改写、重复或移动。"
    "非anchor的细节可以删减，但不能新增事实、机制、意义或因果关系。"
    "若targeted_feedback指出技术密度或模板风险，优先删除方法、变量和公式清单，只保留当前block理解结论所需的信息。"
    + PAPER_FIDELITY_CONTRACT
    + "只返回严格JSON：{\"block\":{\"id\":\"block-1\",\"evidence_ids\":[\"evidence-1\"],\"text\":\"...\"}}。"
)

PAPER_PLANNER_PROMPT = (
    "你是Figure-first Scientific Planner，不写文章正文。根据Abstract、paper_text、source_paragraphs、selected_body_figures和figure_evidence_bundles，"
    "建立唯一的paper_evidence_plan，并只返回严格JSON对象。正文科学骨架必须来自selected body Figures及其真实evidence bundles；不要自行猜测Figure归属。"
    "每个section包含id、title、role、figure_ids、source_paragraph_ids和findings；title必须是适合中文成稿的简洁中文小标题；每个finding包含id、figure_ids、evidence、anchors。"
    "每个section至少包含一个selected Figure，每个finding必须明确绑定一个或多个当前section的figure_ids；一个section可以包含多张高度相关Figure。"
    "每个核心finding只能有一个primary section。若historical/model spread、mechanism、attribution、projection或implication"
    "是不同科学问题且各有独立Figure bundle证据，按真实Figure证据拆分；不要为凑section数量而合并不相关Figure，也不要固定section数量。"
    "只有Abstract或Results明确支持时才拆分multiple modes/regimes，不得创造first/second mode。"
    "每个Figure bundle的核心finding只能进入包含该Figure的section；Figure 只可通过bundle中的caption、明确引用段落和直接关联Results段落支持正文。不要把Fig.2的R=0.71写入只包含Fig.3的section。"
    "无独立主图的机制内容只能作为最相关Figure section中的2到3句解释，不要新建无图机制section；只有删除会造成明显科学逻辑断裂时才保留无图短过渡。"
    "role使用贴合论文的简洁自然标签，不要套固定taxonomy。source_paragraph_ids只能使用输入中真实存在的source id。"
    "每个finding都要有figure_ids和anchors数组（字段名必须是anchors，不得写成quantitative anchors或其他字段）；anchors保留原文指标大小写、R/r、符号、数值、百分号和时间段；不要使用跨section重复的P值作为anchor。"
    "如果输入包含validation_feedback，必须优先修复其中指出的Figure、source或anchor归属，不能重复提交同一错误计划。"
    '返回格式：{"sections":[{"id":"section-1","title":"...","role":"attribution","figure_ids":["Fig. 2"],"source_paragraph_ids":["source-0"],"findings":[{"id":"E1","figure_ids":["Fig. 2"],"evidence":"...","anchors":["R = 0.71"]}]}]}'
)

PAPER_REVIEWER_PROMPT = (
    "你是Scientific Reviewer，只审核科学准确性和section结构，不润色文风，不重写全文。忠实翻译的Abstract导语应保持原始Abstract的结论和因果强度，不要要求改写原文已有表述；只检查导语是否新增或强化了Abstract没有的内容。"
    "只检查：evidence是否错section、数字或统计量是否错误、原文没有的机制、correlation到causation的强化、Abstract核心结果遗漏、attribution/mechanism/projection混用、finding重复或科学关系错误。"
    "但正文主图最多保留4张，Abstract结果只有在当前section的selected Figure bundle或source evidence明确支持时才要求写入；若唯一证据属于未选中的Figure，省略该结果是正确的，不得因此要求补写或判定失败。"
    "还必须核对Figure对应关系：每个核心claim是否属于当前section包含的Figure bundle，数字/统计量是否来自对应Figure的caption或source evidence，是否把Fig.2的结果写入只包含Fig.3的section，以及section标题与其Figure科学内容是否明显不匹配。"
    "不要因句式、节奏、中文措辞、AI-like phrasing或其他纯文风偏好fail；这些交给Chinese Editorial Rewrite和AI-style lint。"
    "每个问题必须包含section、evidence、issue、correction四个字段；correction只能修复该问题，不能移动无关证据。返回严格JSON："
    '{"status":"pass"|"needs_revision","corrections":[{"section":"section-1","evidence":"...","issue":"...","correction":"..."}]}'
)

PAPER_REVISION_PROMPT = (
    "你是Scientific Revision Editor。根据reviewer corrections修正科学内容，只返回严格JSON。"
    "必须逐条落实corrections，不能原样保留reviewer指出的错误句子或仅添加免责声明；修正应针对具体问题，不得因此削弱Abstract明确支持的结论强度、删除核心finding或遗漏关键数字。"
    "保留planner的section id和顺序；每个body只能写对应section的finding/evidence，不得把证据移动到别节，不得修改无问题的科学内容，不得润色成营销文案。"
    "原始Abstract译文已经锁定，绝不返回或修改abstract_cn；标题也只有在明确科学范围错误时才返回。返回："
    '{"sections":[{"id":"section-1","title":"仅在需要修正标题时返回","body":"..."}]}'
)

PAPER_POPULAR_SCIENCE_EDITOR_PROMPT = (
    "你是Popular Science Editor，负责把已经通过Scientific Reviewer的论文正文改写成面向跨专业普通读者的高质量science news/explainer。"
    "科学结构、section顺序、Figure归属、证据范围、数字、anchor、趋势方向、因果强度和限定条件已经锁定，绝不能新增、删除、合并或移动科学结论。"
    "每个section围绕当前Figure回答一个读者问题：先说这张图最重要的发现，再用一两句解释为什么重要或可能如何发生；不要把Figure caption逐句翻译成结果清单。"
    "优先使用普通中文：第一次出现的缩写和专业词必须用极短中文解释，能不用缩写就不用；不要堆叠方法名、统计量或模型术语。保留SSP3-7.0这类已验证anchor时，必须写成‘SSP3-7.0这一未来排放情景’或在紧邻括号中解释，不能只留下裸缩写。"
    "使用短段落和短句，一句话只表达一个主要意思；采用直接的科学陈述，让读者第一遍就能理解。默认使用逗号和句号，避免分号、模板化连接和媒体式修辞。"
    "不要写成论文摘要或Results中文翻译，也不要为了学术感保留不必要的术语密度。XGBoost、SHAP、CCA等方法只有在解释证据为何可信时才保留。"
    "Abstract只用于核对主线和限定条件，不能把当前Figure bundle未支持的次要结果重新塞回正文。"
    "返回严格JSON sections数组，保持每个section的id和顺序；若小标题仍含英文或难懂术语，可以只改title但不得改变其Figure范围和科学含义。正文总量以350到500个中文字符为参考，不要求每节等长，也不得机械截断句子。"
    '{"sections":[{"id":"section-1","title":"可选的通俗小标题","body":"..."}]}'
    + "\n\n"
    + PAPER_EDITORIAL_GUIDE
)
# Compatibility name for callers/tests that still refer to the old stage.
PAPER_EDITOR_PROMPT = PAPER_POPULAR_SCIENCE_EDITOR_PROMPT


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
    if abstract and paper_text:
        source_paragraphs.insert(0, {"id": "source-abstract", "text": abstract})
    valid_source_ids = {str(record["id"]) for record in source_paragraphs}
    figure_first = isinstance(dossier.get("paper_selected_body_images"), list)
    selected_images = (
        list(dossier.get("paper_selected_body_images") or [])
        if figure_first
        else list(dossier.get("images") or [])
    )
    figure_evidence_bundles = _paper_figure_evidence_bundles(
        selected_images,
        source_paragraphs,
    )
    selected_figure_ids = [str(bundle["figure_id"]) for bundle in figure_evidence_bundles]
    dossier["paper_figure_evidence_bundles"] = figure_evidence_bundles
    metadata = {
        "title": dossier.get("title", ""),
        "display_title": display_title,
        "doi": dossier.get("doi", ""),
        "journal": dossier.get("journal", ""),
        "authors": dossier.get("authors", []),
        "selected_body_figures": selected_figure_ids,
        "figure_evidence_bundles": figure_evidence_bundles,
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
    plan = _paper_plan(
        client,
        abstract,
        paper_text,
        source_paragraphs,
        metadata,
        selected_figure_ids,
        figure_evidence_bundles,
    )
    plan["sections"] = _validate_paper_plan_structure(
        plan,
        valid_source_ids,
        set(selected_figure_ids) if figure_first else None,
    )
    for section in plan["sections"]:
        section.pop("necessary_transition", None)
        section.pop("transition_reason", None)
    plan["figure_evidence_bundles"] = figure_evidence_bundles
    plan["planner_sections"] = [dict(section) for section in plan["sections"]]
    # The grey Abstract is a faithful translation and is locked before any
    # story, style, or readability stage runs.
    abstract_lead = translate_paper_abstract(abstract, settings) if abstract else ""

    def write_figure_plan_draft(current_plan: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
        current_sections = [
            {
                **section,
                "body": _paper_write_section(
                    client,
                    abstract,
                    section,
                    source_paragraphs,
                    figure_evidence_bundles,
                    settings.model_name,
                ),
            }
            for section in current_plan["sections"]
        ]
        return current_sections, _paper_assemble_markdown(display_title, abstract_lead, current_sections)

    sections, draft = write_figure_plan_draft(plan)
    try:
        _validate_paper_evidence_plan(
            plan,
            draft,
            valid_source_ids,
            figure_evidence_bundles if figure_first else None,
        )
    except RuntimeError as exc:
        if not any(
            marker in str(exc)
            for marker in (
                "PAPER evidence anchor missing",
                "PAPER evidence figure mismatch",
                "PAPER evidence source mismatch",
            )
        ):
            raise
        logger.warning("PAPER planner deterministic check failed; retrying once: %s", exc)
        plan = _paper_plan(
            client,
            abstract,
            paper_text,
            source_paragraphs,
            metadata,
            selected_figure_ids,
            figure_evidence_bundles,
            str(exc),
        )
        plan["sections"] = _validate_paper_plan_structure(
            plan,
            valid_source_ids,
            set(selected_figure_ids) if figure_first else None,
        )
        for section in plan["sections"]:
            section.pop("necessary_transition", None)
            section.pop("transition_reason", None)
        plan["figure_evidence_bundles"] = figure_evidence_bundles
        plan["planner_sections"] = [dict(section) for section in plan["sections"]]
        sections, draft = write_figure_plan_draft(plan)
        _validate_paper_evidence_plan(
            plan,
            draft,
            valid_source_ids,
            figure_evidence_bundles if figure_first else None,
        )
    logger.info("PAPER deterministic evidence validation passed")
    scientific_review: dict[str, Any] = {
        "status": "pass",
        "cycles": 0,
        "unresolved_issues": [],
    }
    for cycle in range(1, 4):
        review = _paper_review(
            client,
            abstract,
            paper_text,
            source_paragraphs,
            plan,
            draft,
            settings.model_name,
        )
        corrections = review["corrections"]
        logger.info(
            "PAPER scientific review cycle %d: %s, issues=%d",
            cycle,
            review["status"],
            len(corrections),
        )
        scientific_review = {
            "status": "pass" if review["status"] == "pass" else "needs_revision",
            "cycles": cycle,
            "unresolved_issues": corrections,
        }
        if review["status"] == "pass":
            break
        if cycle == 3:
            scientific_review["status"] = "unresolved_after_max_cycles"
            logger.warning(
                "PAPER scientific review unresolved after 3 cycles; "
                "continuing because deterministic validation passed"
            )
            break
        revised_bodies = _paper_revision(
            client,
            abstract,
            paper_text,
            source_paragraphs,
            plan,
            draft,
            corrections,
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
        _validate_paper_evidence_plan(
            plan,
            draft,
            valid_source_ids,
            figure_evidence_bundles if figure_first else None,
        )
        logger.info("PAPER deterministic evidence validation passed")

    plan["scientific_review"] = scientific_review

    style_exemplar = _paper_clean_story_text(_paper_style_exemplar())
    clean_evidence, evidence_map = _paper_clean_story_evidence(plan, source_paragraphs)
    story_plan = _paper_story_planner(
        client,
        clean_evidence,
        style_exemplar,
        settings.model_name,
    )
    valid_evidence_ids = {record["evidence_id"] for record in clean_evidence}
    try:
        story_beats = _paper_validate_story_plan(story_plan, valid_evidence_ids)
    except RuntimeError as exc:
        logger.warning("PAPER story planner returned an invalid beat plan; retrying once: %s", exc)
        story_plan = _paper_story_planner(
            client,
            clean_evidence,
            style_exemplar,
            settings.model_name,
            {"validation": "Use every evidence_id exactly once across the beats; keep each anchor in its assigned beat."},
        )
        story_beats = _paper_validate_story_plan(story_plan, valid_evidence_ids)
    story_sections = _paper_story_sections(story_beats, evidence_map)
    plan["story_plan"] = story_plan
    plan["sections"] = story_sections
    plan["story_evidence"] = {
        evidence_id: {
            "figure_ids": list(
                dict.fromkeys(
                    _paper_figure_id(value)
                    for value in (
                        finding.get("figure_ids")
                        or original_section.get("figure_ids")
                        or original_section.get("selected_body_figures")
                        or []
                    )
                    if str(value).strip()
                )
            ),
            "anchors": list(finding.get("anchors") or []),
        }
        for evidence_id, (original_section, finding) in evidence_map.items()
    }
    story_writer_retry_count = 0
    try:
        story_output = _paper_story_writer(
            client,
            story_plan,
            clean_evidence,
            style_exemplar,
            settings.model_name,
        )
    except RuntimeError as exc:
        if not str(exc).startswith("PAPER story writer"):
            raise
        story_writer_retry_count = 1
        logger.warning("PAPER story writer returned invalid block structure; retrying once: %s", exc)
        story_output = _paper_story_writer(
            client,
            story_plan,
            clean_evidence,
            style_exemplar,
            settings.model_name,
            {"structure": "Return every story beat exactly once and partition every evidence_id into non-mixed Figure blocks."},
        )
    _paper_apply_story_output(plan["sections"], story_output, evidence_map)
    draft = _paper_assemble_markdown(display_title, abstract_lead, plan["sections"])
    try:
        _validate_paper_evidence_plan(
            plan,
            draft,
            valid_source_ids,
            figure_evidence_bundles if figure_first else None,
        )
    except RuntimeError as exc:
        if not any(
            marker in str(exc)
            for marker in (
                "PAPER evidence anchor missing",
                "PAPER evidence figure mismatch",
                "PAPER evidence source mismatch",
            )
        ):
            raise
        story_writer_retry_count = 1
        logger.warning("PAPER story writer deterministic check failed; retrying once: %s", exc)
        story_output = _paper_story_writer(
            client,
            story_plan,
            clean_evidence,
            style_exemplar,
            settings.model_name,
            {
                "deterministic_validation": "A deterministic evidence check found an anchor placement issue. Preserve every supplied anchor exactly and keep it with its evidence."
            },
        )
        _paper_apply_story_output(plan["sections"], story_output, evidence_map)
        draft = _paper_assemble_markdown(display_title, abstract_lead, plan["sections"])
        _validate_paper_evidence_plan(
            plan,
            draft,
            valid_source_ids,
            figure_evidence_bundles if figure_first else None,
        )
    logger.info("PAPER deterministic evidence validation passed")
    plan["story_writer_retry_count"] = story_writer_retry_count

    markdown = draft
    lint = _paper_ai_style_lint(markdown)
    popular_feedback = _paper_editor_feedback(abstract_lead, markdown)
    popular_feedback["anchor_preservation"] = _paper_editor_anchor_audit(
        draft,
        markdown,
        plan,
    )

    def popular_audit_failed() -> bool:
        return bool(
            _paper_ai_style_lint_failed(lint)
            or popular_feedback["abstract_overlong"]
            or popular_feedback["body_lengths"]["total_overlong"]
            or popular_feedback["readability"]["issue_count"]
            or popular_feedback["title_style"]["issue_count"]
            or popular_feedback["anchor_preservation"]["issue_count"]
        )

    popular_retry_count = 0
    if popular_audit_failed():
        popular_retry_count = 1
        story_output = _paper_story_writer(
            client,
            story_plan,
            clean_evidence,
            style_exemplar,
            settings.model_name,
            {"style_lint": lint, **popular_feedback},
        )
        _paper_apply_story_output(plan["sections"], story_output, evidence_map)
        markdown = _paper_assemble_markdown(display_title, abstract_lead, plan["sections"])
        _validate_paper_evidence_plan(
            plan,
            markdown,
            valid_source_ids,
            figure_evidence_bundles if figure_first else None,
        )
        logger.info("PAPER deterministic evidence validation passed")
        lint = _paper_ai_style_lint(markdown)
        popular_feedback = _paper_editor_feedback(abstract_lead, markdown)
        popular_feedback["anchor_preservation"] = _paper_editor_anchor_audit(
            draft,
            markdown,
            plan,
        )

    popular_science_audit = {
        "status": "warning" if popular_audit_failed() else "pass",
        "retry_count": popular_retry_count,
        "unresolved_issues": {
            "style_lint": lint if _paper_ai_style_lint_failed(lint) else {},
            "feedback": popular_feedback if popular_audit_failed() else {},
        },
    }
    if popular_science_audit["status"] == "warning":
        logger.warning("PAPER popular science audit unresolved after retry; continuing with warning")

    humanizer_baseline = markdown
    humanized_output = _paper_humanize_story(
        client,
        story_plan,
        clean_evidence,
        _paper_story_draft_blocks(plan["sections"]),
        settings.model_name,
    )
    _paper_apply_story_output(plan["sections"], humanized_output, evidence_map)
    markdown = _paper_assemble_markdown(display_title, abstract_lead, plan["sections"])
    _validate_paper_evidence_plan(
        plan,
        markdown,
        valid_source_ids,
        figure_evidence_bundles if figure_first else None,
    )
    logger.info("PAPER deterministic evidence validation passed")

    final_humanizer_feedback = _paper_editor_feedback(abstract_lead, markdown)
    final_humanizer_feedback["anchor_preservation"] = _paper_editor_anchor_audit(
        humanizer_baseline,
        markdown,
        plan,
    )
    humanizer_retry_count = 0
    if (
        _paper_ai_style_lint_failed(_paper_ai_style_lint(markdown))
        or final_humanizer_feedback["abstract_overlong"]
        or final_humanizer_feedback["body_lengths"]["total_overlong"]
        or final_humanizer_feedback["readability"]["issue_count"]
        or final_humanizer_feedback["title_style"]["issue_count"]
        or final_humanizer_feedback["anchor_preservation"]["issue_count"]
    ):
        humanizer_retry_count = 1
        targeted_output = _paper_humanize_story(
            client,
            story_plan,
            clean_evidence,
            _paper_story_draft_blocks(plan["sections"]),
            settings.model_name,
            {"readability": final_humanizer_feedback},
        )
        _paper_apply_story_output(plan["sections"], targeted_output, evidence_map)
        markdown = _paper_assemble_markdown(display_title, abstract_lead, plan["sections"])
        _validate_paper_evidence_plan(
            plan,
            markdown,
            valid_source_ids,
            figure_evidence_bundles if figure_first else None,
        )
        logger.info("PAPER deterministic evidence validation passed")
        final_humanizer_feedback = _paper_editor_feedback(abstract_lead, markdown)
        final_humanizer_feedback["anchor_preservation"] = _paper_editor_anchor_audit(
            humanizer_baseline,
            markdown,
            plan,
        )

    plan["humanizer_retry_count"] = humanizer_retry_count
    stop_slop_feedback = _paper_stop_slop_audit(markdown)
    stop_slop_retry_count = 0
    if stop_slop_feedback["issue_count"]:
        stop_slop_retry_count = 1
        targeted_output = _paper_humanize_story(
            client,
            story_plan,
            clean_evidence,
            _paper_story_draft_blocks(plan["sections"]),
            settings.model_name,
            {"stop_slop": stop_slop_feedback},
        )
        _paper_apply_story_output(plan["sections"], targeted_output, evidence_map)
        markdown = _paper_assemble_markdown(display_title, abstract_lead, plan["sections"])
        _validate_paper_evidence_plan(
            plan,
            markdown,
            valid_source_ids,
            figure_evidence_bundles if figure_first else None,
        )
        logger.info("PAPER deterministic evidence validation passed")
        stop_slop_feedback = _paper_stop_slop_audit(markdown)
    plan["stop_slop_audit"] = {
        **stop_slop_feedback,
        "retry_count": stop_slop_retry_count,
    }
    if stop_slop_feedback["issue_count"]:
        logger.warning("PAPER stop-slop style audit unresolved after retry; continuing with warning")

    lint = _paper_ai_style_lint(markdown)
    popular_feedback = _paper_editor_feedback(abstract_lead, markdown)
    popular_feedback["anchor_preservation"] = _paper_editor_anchor_audit(
        humanizer_baseline,
        markdown,
        plan,
    )
    final_popular_issue = bool(
        _paper_ai_style_lint_failed(lint)
        or popular_feedback["abstract_overlong"]
        or popular_feedback["body_lengths"]["total_overlong"]
        or popular_feedback["readability"]["issue_count"]
        or popular_feedback["title_style"]["issue_count"]
        or popular_feedback["anchor_preservation"]["issue_count"]
    )
    if final_popular_issue:
        popular_science_audit["status"] = "warning"
        popular_science_audit["unresolved_issues"] = {
            "style_lint": lint if _paper_ai_style_lint_failed(lint) else {},
            "feedback": popular_feedback,
        }

    markdown = _remove_unverified_paper_quotes(markdown, paper_text)
    markdown = _normalize_article_markdown(markdown, display_title)
    if not markdown:
        raise RuntimeError("PAPER staged pipeline returned empty article")
    plan["popular_science_audit"] = popular_science_audit
    _validate_paper_evidence_plan(
        plan,
        markdown,
        valid_source_ids,
        figure_evidence_bundles if figure_first else None,
    )
    logger.info("PAPER deterministic evidence validation passed")
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
