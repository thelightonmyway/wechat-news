"""OpenAI-compatible text-only selection and writing adapter."""

from __future__ import annotations

import copy
import hashlib
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


def _paper_stable_evidence_id(prefix: str, *parts: Any) -> str:
    payload = "\x1f".join(str(part or "") for part in parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"evidence-{prefix}-{digest}"


def _paper_canonical_evidence_registry(
    source_paragraphs: list[dict[str, Any]],
    figure_evidence_bundles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build immutable contextual and quantitative evidence records before planning."""
    registry: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(source_paragraphs):
        source_id = str(record.get("id") or "")
        text = str(record.get("text") or "").strip()
        if not source_id or not text:
            continue
        figure_numbers = _paper_figure_numbers_in_text(text)
        supported_figures = (
            [f"Fig. {number}" for number in sorted(figure_numbers)]
            if len(figure_numbers) == 1
            else []
        )
        scope = "global_context" if index == 0 or source_id in {"abstract", "source-abstract"} else "section_context"
        if source_id.startswith("source-figure-") and supported_figures:
            scope = "figure_specific"
        evidence_id = f"evidence-source-{source_id}"
        registry[evidence_id] = {
            "evidence_id": evidence_id,
            "value": text,
            "normalized_value": _normalize_evidence_anchor(text),
            "source_paragraph_ids": [source_id],
            "source_sentence": text,
            "scope": scope,
            "supported_figures": supported_figures,
            # Paragraph records provide context only. Quantitative ownership
            # belongs to the atomic evidence-anchor records below.
            "anchors": [],
        }

    owner_keys: set[tuple[str, tuple[str, ...], str, tuple[str, ...]]] = set()

    def add_quantitative_owner(
        value: str,
        normalized: str,
        source_ids: tuple[str, ...],
        source_sentence: str,
        scope: str,
        supported_figures: tuple[str, ...],
    ) -> None:
        if not value or not normalized or not source_ids:
            return
        key = (normalized, source_ids, scope, supported_figures)
        if key in owner_keys:
            return
        owner_keys.add(key)
        evidence_id = _paper_stable_evidence_id(
            "anchor", normalized, source_ids, scope, supported_figures
        )
        registry.setdefault(
            evidence_id,
            {
                "evidence_id": evidence_id,
                "value": value,
                "normalized_value": normalized,
                "source_paragraph_ids": list(source_ids),
                "source_sentence": source_sentence or value,
                "scope": scope,
                "supported_figures": list(supported_figures),
                "anchors": [value],
            },
        )

    for bundle in figure_evidence_bundles:
        for provenance in bundle.get("provenance") or []:
            if not isinstance(provenance, dict):
                continue
            value = str(provenance.get("value") or "").strip()
            normalized = _normalize_evidence_anchor(
                str(provenance.get("normalized_value") or value)
            )
            source_ids = tuple(
                dict.fromkeys(
                    str(source_id)
                    for source_id in provenance.get("source_paragraph_ids") or []
                    if str(source_id).strip()
                )
            )
            scope = str(provenance.get("scope") or "section_context")
            supported_figures = tuple(
                dict.fromkeys(
                    _paper_figure_id(figure)
                    for figure in provenance.get("supported_figures") or []
                    if str(figure).strip()
                )
            )
            add_quantitative_owner(
                value,
                normalized,
                source_ids,
                str(provenance.get("source_sentence") or value),
                scope,
                supported_figures,
            )

    # Bundle provenance is authoritative for Figure associations, but it does
    # not cover source-only values or papers without selected Figures. Create a
    # conservative contextual owner for every uncovered source occurrence.
    represented_occurrences = {
        (normalized, source_ids)
        for normalized, source_ids, _scope, _supported_figures in owner_keys
    }
    for index, record in enumerate(source_paragraphs):
        source_id = str(record.get("id") or "").strip()
        text = str(record.get("text") or "").strip()
        if not source_id or not text or source_id.startswith("source-figure-"):
            continue
        scope = "global_context" if index == 0 or source_id in {"abstract", "source-abstract"} else "section_context"
        for value, sentence in _paper_anchor_sentences(text):
            normalized = _normalize_evidence_anchor(value)
            occurrence = (normalized, (source_id,))
            if occurrence in represented_occurrences:
                continue
            add_quantitative_owner(value, normalized, (source_id,), sentence, scope, ())
            represented_occurrences.add(occurrence)

    return list(registry.values())


def _paper_evidence_by_id(registry: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(record.get("evidence_id") or ""): record
        for record in registry
        if str(record.get("evidence_id") or "")
    }


def _paper_canonical_anchor_owners(
    evidence_registry: list[dict[str, Any]],
) -> dict[tuple[str, str], str]:
    """Assign one literal owner per repeated occurrence of the same scientific claim."""
    evidence_by_id = _paper_evidence_by_id(evidence_registry)
    source_context = {
        str(record.get("evidence_id") or "")[len("evidence-source-"):]: str(
            record.get("source_sentence") or record.get("value") or ""
        )
        for record in evidence_registry
        if str(record.get("evidence_id") or "").startswith("evidence-source-")
    }
    groups: dict[tuple[str, tuple[str, ...], tuple[str, ...]], list[dict[str, Any]]] = {}
    for evidence_id, record in evidence_by_id.items():
        source_ids = tuple(
            str(source_id).strip()
            for source_id in record.get("source_paragraph_ids") or []
            if str(source_id).strip()
        )
        sentence = str(record.get("source_sentence") or record.get("value") or "")
        context = " ".join(
            [sentence]
            + [source_context.get(source_id, "") for source_id in source_ids]
        )
        explicit_figures = {
            _paper_figure_id(value)
            for value in record.get("supported_figures") or []
            if str(value).strip()
        }
        explicit_figures.update(
            f"Fig. {match.group(1)}"
            for source_id in source_ids
            for match in [re.search(r"source-figure-(\d+)", source_id)]
            if match
        )
        figure_matches = list(re.finditer(
            r"\b(?:Fig(?:ure)?\.?\s*\d+)", context, flags=re.IGNORECASE
        ))
        label_matches = list(re.finditer(
            r"\b(?:zone|region)\s+\d+\b", context, flags=re.IGNORECASE
        ))
        for anchor in record.get("anchors") or []:
            normalized = _normalize_evidence_anchor(str(anchor))
            if not normalized:
                continue
            anchor_matches = list(re.finditer(re.escape(str(anchor)), context, flags=re.IGNORECASE))
            anchor_positions = [match.start() for match in anchor_matches] or [0]
            figure_context = set(explicit_figures)
            if figure_matches and not figure_context:
                nearest_figure = min(
                    figure_matches,
                    key=lambda match: min(
                        abs(match.start() - position) for position in anchor_positions
                    ),
                )
                figure_context.add(_paper_figure_id(nearest_figure.group(0)))
            nearby_labels: set[str] = set()
            if label_matches:
                label_candidates: list[tuple[int, re.Match[str]]] = []
                for position in anchor_positions:
                    preceding = [match for match in label_matches if match.start() <= position]
                    selected = preceding[-1] if preceding else label_matches[0]
                    label_candidates.append((abs(selected.start() - position), selected))
                nearest = min(label_candidates, key=lambda item: item[0])[1]
                nearby_labels.add(nearest.group(0).lower())
            # Labels such as “zone 14” distinguish two different claims that
            # happen to share the same percentage.  When no stable label is
            # available, source identity remains part of the key.
            claim_identity = (
                ("label", *sorted(nearby_labels))
                if nearby_labels
                else ("source", *source_ids, _normalize_evidence_anchor(sentence))
            )
            key = (normalized, claim_identity, tuple(sorted(figure_context)))
            groups.setdefault(key, []).append(record)

    owners: dict[tuple[str, str], str] = {}
    for records in groups.values():
        owner = min(
            records,
            key=lambda record: (
                0 if str(record.get("scope") or "") == "section_context" else 1,
                0 if not any(
                    str(source_id).startswith("source-figure-")
                    for source_id in record.get("source_paragraph_ids") or []
                ) else 1,
                str(record.get("evidence_id") or ""),
            ),
        )
        owner_id = str(owner.get("evidence_id") or "")
        for record in records:
            evidence_id = str(record.get("evidence_id") or "")
            for anchor in record.get("anchors") or []:
                normalized = _normalize_evidence_anchor(str(anchor))
                if normalized:
                    owners[(evidence_id, normalized)] = owner_id
    return owners


def _paper_registry_for_llm(registry: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: record.get(key)
            for key in (
                "evidence_id",
                "value",
                "normalized_value",
                "source_sentence",
                "scope",
                "supported_figures",
                "anchors",
            )
        }
        for record in registry
    ]


def _paper_figure_backed_evidence_ids(
    registry: list[dict[str, Any]],
    selected_figure_ids: list[str] | set[str],
) -> dict[str, list[str]]:
    """Expose the canonical Figure-backed evidence choices without selecting claims."""
    selected = {_paper_figure_id(value) for value in selected_figure_ids if str(value).strip()}
    allowed = {figure_id: [] for figure_id in selected}
    for record in registry:
        evidence_id = str(record.get("evidence_id") or "").strip()
        if not evidence_id:
            continue
        for value in record.get("supported_figures") or []:
            figure_id = _paper_figure_id(value)
            if figure_id in allowed and evidence_id not in allowed[figure_id]:
                allowed[figure_id].append(evidence_id)
    return {figure_id: evidence_ids for figure_id, evidence_ids in allowed.items()}


def _paper_clean_evidence_for_llm(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record.get(key)
        for key in (
            "evidence_id",
            "evidence_group",
            "role",
            "core_finding",
            "anchors",
            "source_evidence",
        )
    }


def _paper_derived_source_ids(
    evidence_ids: list[str],
    evidence_by_id: dict[str, dict[str, Any]],
    valid_source_ids: set[str] | None = None,
) -> list[str]:
    """Resolve only real source-paragraph IDs; keep caption IDs evidence-local."""
    source_set: set[str] = set()
    for evidence_id in evidence_ids:
        record = evidence_by_id.get(evidence_id)
        if record is None:
            raise RuntimeError(f"PAPER unknown evidence_id: {evidence_id}")
        for raw_source_id in record.get("source_paragraph_ids") or []:
            source_id = str(raw_source_id).strip()
            if not source_id or source_id.startswith("caption:"):
                continue
            if not source_id.startswith("source-"):
                raise RuntimeError(f"PAPER invalid source paragraph id: {source_id}")
            if valid_source_ids is not None and source_id not in valid_source_ids:
                raise RuntimeError(f"PAPER unknown source paragraph id: {source_id}")
            source_set.add(source_id)
    source_ids: list[str] = []
    for record in evidence_by_id.values():
        for raw_source_id in record.get("source_paragraph_ids") or []:
            source_id = str(raw_source_id).strip()
            if source_id in source_set and source_id not in source_ids:
                source_ids.append(source_id)
    for source_id in sorted(source_set):
        if source_id not in source_ids:
            source_ids.append(source_id)
    return source_ids


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
    evidence_registry: list[dict[str, Any]] | None = None,
    valid_source_paragraph_ids: set[str] | None = None,
) -> None:
    story_evidence = plan.get("story_evidence")
    if not isinstance(story_evidence, dict):
        return
    evidence_by_id = _paper_evidence_by_id(evidence_registry or [])
    canonical_mode = bool(evidence_registry)
    canonical_anchor_owners = _paper_canonical_anchor_owners(evidence_registry or [])
    planned_sections = plan.get("sections") or []
    seen_block_ids: set[str] = set()
    if len(sections) != len(planned_sections):
        raise RuntimeError("PAPER story block validation failed: section count changed")
    for section_index, section in enumerate(planned_sections):
        blocks = section.get("blocks")
        if not isinstance(blocks, list) or not blocks:
            raise RuntimeError("PAPER story block validation failed: missing blocks")
        beat_ids = set((section.get("story_beat") or {}).get("evidence_ids") or [])
        seen_evidence: set[str] = set()
        body_normalized = re.sub(r"\s+", "", sections[section_index][1])
        contract_body_normalized = _paper_story_contract_text(sections[section_index][1])
        paragraph_text_by_block: dict[str, str] = {}
        planned_paragraphs = section.get("paragraphs")
        if planned_paragraphs is not None:
            if not isinstance(planned_paragraphs, list) or not planned_paragraphs:
                raise RuntimeError("PAPER story block validation failed: invalid paragraphs")
            paragraph_position = -1
            for paragraph in planned_paragraphs:
                if not isinstance(paragraph, dict):
                    raise RuntimeError("PAPER story block validation failed: invalid paragraph")
                paragraph_block_ids = paragraph.get("block_ids")
                paragraph_text = str(paragraph.get("text") or "").strip()
                if (
                    not isinstance(paragraph_block_ids, list)
                    or not paragraph_block_ids
                    or not paragraph_text
                    or any(not isinstance(block_id, str) or not block_id.strip() for block_id in paragraph_block_ids)
                    or len({block_id for block_id in paragraph_block_ids if isinstance(block_id, str)}) != len(paragraph_block_ids)
                    or any(block_id in paragraph_text_by_block for block_id in paragraph_block_ids)
                ):
                    raise RuntimeError("PAPER story block validation failed: invalid paragraph ownership")
                normalized_paragraph = _paper_story_contract_text(paragraph_text)
                paragraph_position = contract_body_normalized.find(
                    normalized_paragraph, paragraph_position + 1
                )
                if paragraph_position < 0:
                    raise RuntimeError(
                        "PAPER story block validation failed: paragraph text is not in article"
                    )
                for block_id in paragraph_block_ids:
                    paragraph_text_by_block[block_id] = normalized_paragraph
        previous_position = -1
        normalized_blocks: list[tuple[dict[str, Any], str]] = []
        blocks_by_evidence: dict[str, list[tuple[dict[str, Any], str]]] = {}
        for block in blocks:
            if not isinstance(block, dict):
                raise RuntimeError("PAPER story block validation failed: invalid block")
            block_id = str(block.get("id") or "").strip()
            if block_id in seen_block_ids:
                raise RuntimeError("PAPER story block validation failed: duplicate block id")
            seen_block_ids.add(block_id)
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
            if canonical_mode:
                expected_source_ids = _paper_derived_source_ids(
                    evidence_ids, evidence_by_id, valid_source_paragraph_ids
                )
                actual_source_ids = list(block.get("source_paragraph_ids") or [])
                if actual_source_ids != expected_source_ids:
                    raise RuntimeError(
                        "PAPER evidence block source mismatch: "
                        f"block={block_id!r}; source_paragraph_ids={actual_source_ids!r}; "
                        f"expected={expected_source_ids!r}"
                    )
            figure_groups = {
                tuple(story_evidence[evidence_id].get("figure_ids") or [])
                for evidence_id in evidence_ids
                if story_evidence[evidence_id].get("figure_ids")
            }
            if len(figure_groups) > 1:
                raise RuntimeError("PAPER story block validation failed: mixed Figure evidence")
            block_normalized = paragraph_text_by_block.get(block_id) or re.sub(r"\s+", "", text)
            if block_id not in paragraph_text_by_block:
                position = body_normalized.find(block_normalized, previous_position + 1)
                if position < 0:
                    raise RuntimeError(
                        "PAPER story block validation failed: block text is not in its planned paragraph"
                    )
                previous_position = position
            normalized_blocks.append((block, block_normalized))
            for evidence_id in evidence_ids:
                blocks_by_evidence.setdefault(evidence_id, []).append(
                    (block, block_normalized)
                )
            seen_evidence.update(evidence_ids)
        if planned_paragraphs is not None and set(paragraph_text_by_block) != {
            str(block.get("id") or "") for block in blocks if isinstance(block, dict)
        }:
            raise RuntimeError("PAPER story block validation failed: paragraph block coverage changed")
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
                anchor_records = provenance_by_anchor.get(normalized_anchor, [])
                supported_figures = supported_figures_by_anchor.get(normalized_anchor, set())
                if (
                    not normalized_anchor
                    or re.fullmatch(r"[Pp][<>=]\d+(?:\.\d+)?", normalized_anchor)
                    or normalized_anchor in {"90%", "95%", "99%"}
                ):
                    continue
                owned_anchor_records = [
                    record
                    for record in anchor_records
                    if str(record.get("evidence_id") or "") == evidence_id
                ]
                figure_specific = bool(evidence_figures) or any(
                    record.get("scope") == "figure_specific"
                    for record in owned_anchor_records
                )
                if not canonical_mode:
                    figure_specific = figure_specific or bool(supported_figures)
                owning_blocks = blocks_by_evidence.get(evidence_id, [])
                literal_anchor_owner = canonical_anchor_owners.get(
                    (str(evidence_id), normalized_anchor), str(evidence_id)
                ) == str(evidence_id)
                if canonical_mode and literal_anchor_owner:
                    if len(owning_blocks) != 1:
                        raise RuntimeError(
                            "PAPER story block validation failed: evidence is not bound to one block: "
                            f"evidence_id={evidence_id!r}"
                        )
                    if normalized_anchor not in _normalize_evidence_anchor(owning_blocks[0][1]):
                        raise RuntimeError(
                            "PAPER evidence anchor missing from bound block: "
                            f"evidence_id={evidence_id!r}; anchor={anchor!r}; "
                            f"block={owning_blocks[0][0].get('id')!r}"
                        )
                if not figure_specific:
                    continue
                owner_blocks = owning_blocks
                if not owner_blocks:
                    owner_blocks = [
                        (block, block_text)
                        for block, block_text in normalized_blocks
                        if normalized_anchor in _normalize_evidence_anchor(block_text)
                    ]
                if not owner_blocks:
                    raise RuntimeError(
                        "PAPER evidence anchor missing from bound block: "
                        f"evidence_id={evidence_id!r}; anchor={anchor!r}"
                    )
                supported = supported_figures_by_anchor.get(normalized_anchor, set())
                owner_figure_sets = [
                    {
                        _paper_figure_id(value)
                        for value in owner_block.get("figure_ids") or evidence_figures
                        if str(value).strip()
                    }
                    for owner_block, _ in owner_blocks
                ]
                if supported and not any(
                    figure_ids.intersection(supported) for figure_ids in owner_figure_sets
                ):
                    owner_block = owner_blocks[0][0]
                    block_figures = set().union(*owner_figure_sets)
                    raise RuntimeError(
                        "PAPER evidence block figure mismatch: "
                        f"evidence={anchor!r}; block={owner_block.get('id')!r}; "
                        f"figures={sorted(block_figures)!r}; supported_figures={sorted(supported)!r}"
                    )


def _validate_paper_evidence_plan(
    plan: dict[str, Any],
    markdown: str,
    valid_source_paragraph_ids: set[str] | None = None,
    figure_evidence_bundles: list[dict[str, Any]] | None = None,
    evidence_registry: list[dict[str, Any]] | None = None,
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
    evidence_by_id = _paper_evidence_by_id(evidence_registry or [])
    canonical_mode = bool(evidence_registry)
    canonical_anchor_owners = _paper_canonical_anchor_owners(evidence_registry or [])
    anchor_evidence_ids: dict[str, set[str]] = {}
    if canonical_mode:
        for evidence_id, evidence in evidence_by_id.items():
            for anchor in evidence.get("anchors") or []:
                normalized_anchor = _normalize_evidence_anchor(str(anchor))
                if normalized_anchor:
                    anchor_evidence_ids.setdefault(normalized_anchor, set()).add(evidence_id)
    _paper_validate_story_blocks(
        plan,
        sections,
        supported_figures_by_anchor,
        provenance_by_anchor,
        evidence_registry,
        valid_source_paragraph_ids,
    )
    for planned_index, section in enumerate(planned_sections):
        if not isinstance(section, dict):
            raise RuntimeError("PAPER evidence plan validation failed: invalid section")
        source_ids = section.get("source_paragraph_ids")
        if not isinstance(source_ids, list) or any(
            not isinstance(source_id, str) or not source_id.strip()
            for source_id in source_ids
        ) or (not canonical_mode and not source_ids):
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
        if canonical_mode:
            section_evidence_ids = [
                str(value).strip()
                for finding in section.get("findings") or []
                for value in finding.get("evidence_ids") or []
                if str(value).strip()
            ]
            unique_evidence_ids = list(dict.fromkeys(section_evidence_ids))
            expected_source_ids = _paper_derived_source_ids(
                unique_evidence_ids, evidence_by_id, valid_source_paragraph_ids
            )
            for evidence_id in unique_evidence_ids:
                record = evidence_by_id[evidence_id]
                source_sentence = _normalize_evidence_anchor(
                    str(record.get("source_sentence") or record.get("value") or "")
                )
                for anchor in record.get("anchors") or []:
                    normalized_anchor = _normalize_evidence_anchor(str(anchor))
                    if normalized_anchor and normalized_anchor not in source_sentence:
                        raise RuntimeError(
                            "PAPER evidence source mismatch: "
                            f"evidence_id={evidence_id!r}; anchor={anchor!r}; "
                            f"source_paragraph_ids={record.get('source_paragraph_ids')!r}"
                        )
                logger.debug(
                    "PAPER evidence provenance evidence_id=%s sources=%s scope=%s",
                    evidence_id,
                    record.get("source_paragraph_ids") or [],
                    record.get("scope") or "",
                )
            if source_ids != expected_source_ids:
                raise RuntimeError(
                    "PAPER evidence source mismatch: "
                    f"section={section.get('title')!r}; source_paragraph_ids={source_ids!r}; "
                    f"expected={expected_source_ids!r}"
                )
        section_figure_ids = [
            _paper_figure_id(value)
            for value in (section.get("figure_ids") or section.get("selected_body_figures") or [])
            if str(value).strip()
        ]
        if bundles_by_id and not section_figure_ids:
            raise RuntimeError("PAPER evidence plan validation failed: section has no figure bundle")
        if bundles_by_id:
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
                anchor_records = provenance_by_anchor.get(normalized_anchor, [])
                supported_figures = supported_figures_by_anchor.get(normalized_anchor, set())
                if (
                    not normalized_anchor
                    or re.fullmatch(r"[Pp][<>=]\d+(?:\.\d+)?", normalized_anchor)
                    or normalized_anchor in {"90%", "95%", "99%"}
                ):
                    continue
                if bundles_by_id:
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
                            for record in anchor_records
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
                if canonical_mode:
                    bound_evidence_ids = {
                        evidence_id
                        for evidence_id in section_evidence_ids
                        if normalized_anchor in {
                            _normalize_evidence_anchor(str(value))
                            for value in evidence_by_id[evidence_id].get("anchors") or []
                        }
                    }
                    duplicate_anchor = len(anchor_evidence_ids.get(normalized_anchor, set())) > 1
                    if bound_evidence_ids and (
                        duplicate_anchor
                        or bool(supported_figures)
                        or any(record.get("scope") == "figure_specific" for record in anchor_records)
                    ):
                        # Block validation above has already checked the anchor
                        # against its immutable evidence binding. A global
                        # string search is ambiguous when another evidence
                        # record uses the same normalized value.
                        continue
                actual_indexes = [
                    index
                    for index, (_, body) in enumerate(sections)
                    if normalized_anchor in _normalize_evidence_anchor(body)
                ]
                if not actual_indexes:
                    is_figure_specific = bool(supported_figures) or any(
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


def _paper_planner_structure_retryable(error: str) -> bool:
    """Limit planner retries to model structure/evidence selection errors."""
    return any(
        marker in error
        for marker in (
            "returned no sections",
            "invalid section metadata",
            "section without findings",
            "invalid evidence_ids",
            "unknown evidence_id",
            "invalid finding",
            "invalid quantitative anchors",
            "no Figure-backed evidence",
        )
    )


def _validate_paper_plan_structure(
    plan: dict[str, Any],
    valid_source_paragraph_ids: set[str],
    selected_figure_ids: set[str] | None = None,
    evidence_registry: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    sections = plan.get("sections")
    if not isinstance(sections, list) or not sections:
        raise RuntimeError("PAPER scientific planner returned no sections")
    seen_ids: set[str] = set()
    validated: list[dict[str, Any]] = []
    evidence_by_id = _paper_evidence_by_id(evidence_registry or [])
    canonical_mode = bool(evidence_registry)
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
        if canonical_mode:
            if source_ids is not None:
                raise RuntimeError("PAPER scientific planner returned source provenance")
        else:
            if not isinstance(source_ids, list) or not source_ids:
                raise RuntimeError("PAPER scientific planner returned invalid source paragraph ids")
            if not all(
                isinstance(source_id, str) and source_id in valid_source_paragraph_ids
                for source_id in source_ids
            ):
                raise RuntimeError("PAPER scientific planner returned unknown source paragraph ids")
        if not isinstance(findings, list) or not findings:
            raise RuntimeError("PAPER scientific planner returned a section without findings")

        # In canonical mode, Figure ownership is derived exclusively from the
        # immutable evidence registry. Any Figure fields returned by the model
        # are compatibility noise and must never affect validation or output.
        figure_ids = (
            []
            if canonical_mode
            else [
                _paper_figure_id(value)
                for value in (
                    section.get("figure_ids")
                    or section.get("selected_body_figures")
                    or []
                )
                if str(value).strip()
            ]
        )
        if not canonical_mode and selected_figure_ids is not None:
            if not figure_ids or not set(figure_ids).issubset(selected_figure_ids):
                raise RuntimeError("PAPER scientific planner returned an invalid figure mapping")
            section["figure_ids"] = figure_ids

        derived_section_evidence_ids: list[str] = []
        derived_section_figures: list[str] = []
        for finding in findings:
            if not isinstance(finding, dict):
                raise RuntimeError("PAPER scientific planner returned an invalid finding")
            if canonical_mode:
                if "source_paragraph_ids" in finding or "source_sentence" in finding:
                    raise RuntimeError("PAPER scientific planner returned source provenance")
                evidence_ids = [
                    str(value).strip()
                    for value in finding.get("evidence_ids") or []
                    if str(value).strip()
                ]
                if not evidence_ids or len(set(evidence_ids)) != len(evidence_ids):
                    raise RuntimeError("PAPER scientific planner returned invalid evidence_ids")
                records = [evidence_by_id.get(evidence_id) for evidence_id in evidence_ids]
                if any(record is None for record in records):
                    raise RuntimeError("PAPER scientific planner returned unknown evidence_id")
                supported_figures: list[str] = []
                for record in records:
                    if record is None:
                        continue
                    for value in record.get("supported_figures") or []:
                        figure_id = _paper_figure_id(value)
                        if figure_id and figure_id not in supported_figures:
                            supported_figures.append(figure_id)
                if selected_figure_ids is not None and not set(supported_figures).issubset(
                    selected_figure_ids
                ):
                    raise RuntimeError(
                        "PAPER canonical evidence references an unselected Figure: "
                        + ", ".join(sorted(set(supported_figures) - selected_figure_ids))
                    )
                finding["evidence_ids"] = evidence_ids
                # Keep finding.figure_ids as direct support only. Contextual
                # evidence stays unfigured; section placement is handled by
                # the enclosing section and never changes its provenance.
                finding["figure_ids"] = supported_figures
                finding["evidence"] = " ".join(
                    str(record.get("source_sentence") or record.get("value") or "").strip()
                    for record in records
                    if record is not None
                ).strip()
                finding["anchors"] = list(
                    dict.fromkeys(
                        str(anchor)
                        for record in records
                        if record is not None
                        for anchor in record.get("anchors") or []
                        if str(anchor).strip()
                    )
                )
                finding["source_paragraph_ids"] = _paper_derived_source_ids(
                    evidence_ids, evidence_by_id, valid_source_paragraph_ids
                )
                finding["source_sentence"] = [
                    str(record.get("source_sentence") or "")
                    for record in records
                    if record is not None
                ]
                for figure_id in supported_figures:
                    if figure_id not in derived_section_figures:
                        derived_section_figures.append(figure_id)
                for evidence_id in evidence_ids:
                    if evidence_id not in derived_section_evidence_ids:
                        derived_section_evidence_ids.append(evidence_id)
            elif not str(finding.get("evidence") or "").strip():
                raise RuntimeError("PAPER scientific planner returned an invalid finding")
            if not canonical_mode:
                finding_figure_ids = [
                    _paper_figure_id(value)
                    for value in (
                        finding.get("figure_ids")
                        or finding.get("figures")
                        or figure_ids
                    )
                    if str(value).strip()
                ]
                if selected_figure_ids is not None:
                    if not finding_figure_ids or not set(finding_figure_ids).issubset(
                        set(figure_ids)
                    ):
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

        if canonical_mode:
            if selected_figure_ids is not None and not derived_section_figures:
                raise RuntimeError(
                    "PAPER scientific planner section has no Figure-backed evidence"
                )
            section["figure_ids"] = derived_section_figures
            section["source_paragraph_ids"] = _paper_derived_source_ids(
                derived_section_evidence_ids, evidence_by_id, valid_source_paragraph_ids
            )
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


_PAPER_STYLE_CORPUS_CHAR_BUDGET = 30000
_PAPER_STYLE_DOCUMENT_EXCLUSIONS = frozenset({
    "readme.md",
    "style_guide.md",
    "guide.md",
    "instructions.md",
    "notes.md",
    "manifest.md",
    "index.md",
})


def _paper_style_corpus_paths(exemplar_dir: Path) -> list[Path]:
    """Discover writing documents without coupling the loader to exemplar names."""
    return sorted(
        path
        for path in exemplar_dir.rglob("*.md")
        if path.is_file()
        and path.name.lower() not in _PAPER_STYLE_DOCUMENT_EXCLUSIONS
        and not path.name.startswith("_")
    )


def _paper_style_document_excerpt(text: str, budget: int, title: str) -> str:
    """Compress one document while retaining its title, opening, middle, and ending."""
    text = text.strip()
    if len(text) <= budget:
        return text
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]
    heading = next(
        (paragraph for paragraph in paragraphs if paragraph.startswith("#")),
        f"# {title}",
    )
    prose = [paragraph for paragraph in paragraphs if not paragraph.startswith("#")]
    opening = prose[0] if prose else text
    middle = prose[len(prose) // 2] if prose else opening
    ending = prose[-1] if prose else opening
    if len({opening, middle, ending}) == 1:
        middle = ""
    fixed = len(heading) + len("\n\n开头：\n\n正文代表段落：\n\n结尾：\n")
    available = max(0, budget - fixed)
    opening_budget = available * 3 // 8
    middle_budget = available * 1 // 4
    ending_budget = max(0, available - opening_budget - middle_budget)
    return "\n\n".join(
        part
        for part in (
            heading,
            f"开头：\n{opening[:opening_budget]}" if opening_budget else "",
            f"正文代表段落：\n{middle[:middle_budget]}" if middle and middle_budget else "",
            f"结尾：\n{ending[:ending_budget]}" if ending_budget else "",
        )
        if part
    )[:budget]


def _paper_style_exemplar() -> str:
    """Load the full extensible writing corpus, never generated PAPER articles."""
    exemplar_dir = PROJECT_ROOT / "writer" / "exemplars"
    try:
        guide = (exemplar_dir / "STYLE_GUIDE.md").read_text(encoding="utf-8").strip()
    except OSError:
        guide = ""
    documents: list[tuple[str, str]] = []
    for path in _paper_style_corpus_paths(exemplar_dir):
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if text:
            documents.append((path.relative_to(exemplar_dir).as_posix(), text))
    corpus_size = sum(len(text) + len(name) + 20 for name, text in documents)
    if corpus_size <= _PAPER_STYLE_CORPUS_CHAR_BUDGET:
        corpus = [f"范文文件：{name}\n{text}" for name, text in documents]
    else:
        per_document = max(1, _PAPER_STYLE_CORPUS_CHAR_BUDGET // max(1, len(documents)))
        corpus = [
            f"范文文件：{name}\n{_paper_style_document_excerpt(text, per_document, name)}"
            for name, text in documents
        ]
    parts = []
    if guide:
        parts.append(
            "STYLE_GUIDE（辅助规则；范文原文是主要风格参考）\n" + guide
        )
    parts.append(
        "STYLE CORPUS（主要参考：学习句法、段落长度、信息密度、叙事推进、"
        "术语解释和自然中文；禁止复制其中的事实、数字、人物、地点和结论）\n"
        + "\n\n---\n\n".join(corpus)
    )
    return "\n\n=== STYLE PACKAGE ===\n\n".join(parts)


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


def _paper_protected_proper_nouns(text: str) -> list[str]:
    """Return no automatic geography guesses; translation choice belongs to the LLM."""
    return []


def _paper_restore_proper_noun_markers(
    text: str,
    markers: dict[str, str],
) -> str:
    """Restore complete protected spans and reject partial or missing restores."""
    restored = str(text or "")
    for marker, value in markers.items():
        marker_name = marker[2:-2]
        restored = re.sub(
            rf"\[\[\s*{re.escape(marker_name)}\s*\]\]",
            value,
            restored,
        )
    if any(
        re.search(rf"\[\[\s*{re.escape(marker[2:-2])}\s*\]\]", restored)
        for marker in markers
    ) or any(value not in restored for value in markers.values()):
        raise RuntimeError("PAPER Abstract proper noun placeholder integrity check failed")
    return restored


def _paper_remove_inline_citation_markers(text: str) -> str:
    """Remove bibliographic superscripts while preserving scientific unit exponents."""
    cleaned = str(text or "")
    cluster = r"(?:⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)(?:[˒,、·.\-–—−⁻][⁰¹²³⁴⁵⁶⁷⁸⁹]+)*"
    unit_token = re.compile(
        r"(?i)(?:w|m|cm|mm|km|s|ms|kg|g|k|pa|hz|n|j)$"
    )

    def replace(match: re.Match[str]) -> str:
        prefix = cleaned[: match.start()]
        previous_token = re.search(r"[A-Za-z]+$", prefix)
        if previous_token and (
            unit_token.fullmatch(previous_token.group(0))
            or (
                previous_token.group(0).isupper()
                and len(previous_token.group(0)) <= 3
            )
        ):
            return match.group(0)
        return ""

    cleaned = re.sub(cluster, replace, cleaned)

    ascii_cluster = re.compile(
        r"(?<![\d.])"
        r"(?P<gap>[ \t]*)"
        r"(?P<cluster>[1-9]\d?(?:\s*[,，]\s*[1-9]\d?|\s*[-–—]\s*[1-9]\d?)*)"
        r"\s*(?P<punct>[，,。！？.!?])(?!\d)"
    )

    def remove_ascii_citation(match: re.Match[str]) -> str:
        before = cleaned[: match.start()].rstrip()
        if re.search(r"(?i)(?:\bfig(?:ure)?\.?)$", before):
            return match.group(0)
        return match.group("punct")

    cleaned = ascii_cluster.sub(remove_ascii_citation, cleaned)
    cleaned = re.sub(
        r"(?<=[。！？.!?])\s*[1-4](?:\s*[,，]\s*[1-4])*(?:\s*[-–—]\s*[1-4])?(?=\s*$|\s+[㐀-鿿A-Za-z])",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"(?<=[㐀-鿿A-Za-z)])\s*\[\s*\d{1,3}(?:\s*[,;、]\s*\d{1,3})*(?:\s*[-–—]\s*\d{1,3})?\s*\]",
        "",
        cleaned,
    )
    return cleaned


def _paper_story_contract_text(text: str) -> str:
    """Normalize only publication-layer whitespace and citation-only surface markers."""
    return re.sub(
        r"\s+", "", _paper_remove_inline_citation_markers(str(text or ""))
    )


def _paper_plain_language_cleanup(text: str) -> str:
    """Translate non-essential technical shorthand without touching verified values."""
    cleaned = str(text or "")
    replacements = (
        (r"(?<![A-Za-z0-9])(\d{3})\s*百帕", r"\1 hPa"),
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
    cleaned = re.sub(r"(?i)(?<![A-Za-z])(?:approximately|roughly|about|around)\s+(?=\d+%)", "约", cleaned)
    cleaned = re.sub(r"约占([^。！？\n]{0,40})的约(?=\d+%)", r"约占\1的", cleaned)
    cleaned = re.sub(r"\b((?:19|20)\d{2})\s+to\s+((?:19|20)\d{2})\b", r"\1—\2", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*[；;]\s*", "。", cleaned)
    cleaned = re.sub(r"。+", "。", cleaned)
    cleaned = re.sub(r"(?<=[一-鿿])\s+(?=[一-鿿])", "", cleaned)
    return cleaned


def _paper_clean_story_evidence(
    plan: dict[str, Any],
    source_paragraphs: list[dict[str, Any]],
    evidence_registry: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, tuple[dict[str, Any], dict[str, Any]]]]:
    source_by_id = {str(record.get("id") or ""): record for record in source_paragraphs}
    evidence_by_id = _paper_evidence_by_id(evidence_registry or [])
    canonical_mode = bool(evidence_registry)
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
            finding_evidence_ids = (
                [str(value).strip() for value in finding.get("evidence_ids") or [] if str(value).strip()]
                if canonical_mode
                else [f"evidence-{counter}"]
            )
            if not finding_evidence_ids:
                continue
            if not canonical_mode:
                counter += 1
            if canonical_mode:
                finding_figures = tuple(
                    dict.fromkeys(
                        _paper_figure_id(value)
                        for evidence_id in finding_evidence_ids
                        for value in evidence_by_id[evidence_id].get("supported_figures") or []
                        if str(value).strip()
                    )
                )
            else:
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
            for evidence_id in finding_evidence_ids:
                canonical_records = (
                    [evidence_by_id[evidence_id]]
                    if canonical_mode and evidence_id in evidence_by_id
                    else []
                )
                if canonical_mode and not canonical_records:
                    raise RuntimeError(f"PAPER unknown evidence_id: {evidence_id}")
                source_evidence = list(
                    dict.fromkeys(
                        _paper_clean_story_text(
                            str(record.get("source_sentence") or record.get("value") or "")
                        )
                        for record in canonical_records
                        if str(record.get("source_sentence") or record.get("value") or "").strip()
                    )
                ) if canonical_mode else list(dict.fromkeys(text for text in section_sources if text))
                source_ids = (
                    _paper_derived_source_ids(
                        [evidence_id], evidence_by_id, set(source_by_id)
                    )
                    if canonical_mode
                    else list(section.get("source_paragraph_ids") or [])
                )
                source_sentences = [
                    str(record.get("source_sentence") or record.get("value") or "")
                    for record in canonical_records
                ] if canonical_mode else source_evidence[:]
                scopes = list(
                    dict.fromkeys(
                        str(record.get("scope") or "section_context")
                        for record in canonical_records
                    )
                ) if canonical_mode else []
                supported_figures = list(
                    dict.fromkeys(
                        _paper_figure_id(value)
                        for record in canonical_records
                        for value in record.get("supported_figures") or []
                        if str(value).strip()
                    )
                ) if canonical_mode else []
                record = {
                    "evidence_id": evidence_id,
                    "evidence_group": evidence_group,
                    "role": str(section.get("role") or "").strip(),
                    "core_finding": (
                        _paper_clean_story_text(source_sentences[0])
                        if canonical_mode and source_sentences
                        else _paper_clean_story_text(finding.get("evidence", ""))
                    ),
                    "anchors": (
                        [str(anchor) for anchor in canonical_records[0].get("anchors") or [] if str(anchor).strip()]
                        if canonical_mode
                        else [str(anchor) for anchor in finding.get("anchors") or [] if str(anchor).strip()]
                    ),
                    "source_evidence": source_evidence[:8],
                    "source_paragraph_ids": source_ids,
                    "source_sentence": source_sentences,
                    "scope": scopes,
                    "supported_figures": supported_figures,
                }
                evidence.append(record)
                evidence_map[evidence_id] = (section, finding)
    return evidence, evidence_map


def _paper_story_skeleton(
    plan: dict[str, Any],
    clean_evidence: list[dict[str, Any]],
    evidence_map: dict[str, tuple[dict[str, Any], dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Build fixed beat ownership from the validated scientific plan."""
    clean_by_id = {
        str(record.get("evidence_id") or ""): record
        for record in clean_evidence
        if str(record.get("evidence_id") or "")
    }
    skeleton: list[dict[str, Any]] = []
    for index, section in enumerate(plan.get("sections") or [], start=1):
        section_id = str(section.get("id") or "")
        evidence_ids: list[str] = []
        owner_ids = None
        if evidence_map is not None:
            owner_ids = {
                evidence_id
                for evidence_id, (owner_section, _) in evidence_map.items()
                if owner_section is section
            }
        for finding in section.get("findings") or []:
            if not isinstance(finding, dict):
                continue
            for value in finding.get("evidence_ids") or []:
                evidence_id = str(value).strip()
                if (
                    evidence_id in clean_by_id
                    and (owner_ids is None or evidence_id in owner_ids)
                    and evidence_id not in evidence_ids
                ):
                    evidence_ids.append(evidence_id)
        if evidence_map is not None:
            for evidence_id, (owner_section, _) in evidence_map.items():
                if owner_section is section and evidence_id in clean_by_id and evidence_id not in evidence_ids:
                    evidence_ids.append(evidence_id)
        if not evidence_ids:
            raise RuntimeError(
                "PAPER story skeleton section has no evidence: "
                f"section_id={section_id!r}"
            )
        facts: list[str] = []
        summaries = [str(section.get("title") or "").strip()]
        for evidence_id in evidence_ids:
            record = clean_by_id[evidence_id]
            summaries.append(str(record.get("core_finding") or "").strip())
            for anchor in record.get("anchors") or []:
                fact = _paper_plain_language_cleanup(str(anchor)).strip()
                if fact and fact not in facts:
                    facts.append(fact)
        skeleton.append({
            "beat_id": f"beat-{index}",
            "evidence_ids": evidence_ids,
            "summary": "；".join(part for part in summaries if part)[:1200],
            "required_facts": facts,
        })
    if not skeleton:
        raise RuntimeError("PAPER story skeleton has no evidence beats")
    return skeleton


def _paper_validate_story_plan(
    story_plan: dict[str, Any],
    story_skeleton: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    brief = story_plan.get("editorial_brief")
    if not isinstance(brief, dict) or not all(
        isinstance(brief.get(field), str) and brief[field].strip()
        for field in ("audience", "purpose", "tone", "reader_should_leave_with", "story_question")
    ):
        raise RuntimeError("PAPER story planner returned an invalid editorial brief")
    beats = story_plan.get("story_beats")
    expected_ids = [str(item.get("beat_id") or "") for item in story_skeleton]
    if not isinstance(beats, list) or len(beats) != len(expected_ids):
        raise RuntimeError("PAPER story planner returned an invalid beat count")
    returned_ids = [str(beat.get("id") or "").strip() if isinstance(beat, dict) else "" for beat in beats]
    if returned_ids != expected_ids:
        raise RuntimeError(
            "PAPER story planner returned invalid beat ids: "
            f"expected={expected_ids!r}; actual={returned_ids!r}"
        )
    validated: list[dict[str, Any]] = []
    for beat_index, (beat, skeleton) in enumerate(zip(beats, story_skeleton)):
        if not isinstance(beat, dict):
            raise RuntimeError("PAPER story planner returned an invalid story beat")
        title = _paper_clean_story_text(beat.get("title", ""))
        reader_question = _paper_clean_story_text(beat.get("reader_question", ""))
        core_message = _paper_clean_story_text(beat.get("core_message", ""))
        transition = _paper_clean_story_text(beat.get("transition_to_next", ""))
        if (
            not title
            or not reader_question
            or not core_message
            or (not transition and beat_index != len(beats) - 1)
        ):
            raise RuntimeError("PAPER story planner returned invalid story beat fields")
        validated.append({
            "id": skeleton["beat_id"],
            "title": title,
            "reader_question": reader_question,
            "core_message": core_message,
            "evidence_ids": list(skeleton["evidence_ids"]),
            "transition_to_next": transition or "文章收束。",
        })
    story_plan["editorial_brief"] = {
        field: _paper_clean_story_text(brief[field])
        for field in ("audience", "purpose", "tone", "reader_should_leave_with", "story_question")
    }
    story_plan["story_beats"] = validated
    return validated


def _paper_story_sections(
    story_beats: list[dict[str, Any]],
    evidence_map: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    evidence_registry: list[dict[str, Any]] | None = None,
    valid_source_paragraph_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    evidence_by_id = _paper_evidence_by_id(evidence_registry or [])
    canonical_mode = bool(evidence_registry)
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
            if not canonical_mode:
                for source_id in original_section.get("source_paragraph_ids") or []:
                    if source_id not in source_ids:
                        source_ids.append(source_id)
            for figure_id in original_section.get("figure_ids") or original_section.get("selected_body_figures") or []:
                normalized = _paper_figure_id(figure_id)
                if normalized not in figure_ids:
                    figure_ids.append(normalized)
            marker = id(finding)
            if marker not in seen_findings:
                finding_output = dict(finding)
                if canonical_mode:
                    finding_output["evidence_ids"] = [
                        candidate_id
                        for candidate_id in beat["evidence_ids"]
                        if evidence_map.get(candidate_id, (None, None))[1] is finding
                    ]
                    finding_output["source_paragraph_ids"] = _paper_derived_source_ids(
                        finding_output["evidence_ids"],
                        evidence_by_id,
                        valid_source_paragraph_ids,
                    )
                findings.append(finding_output)
                seen_findings.add(marker)
        if canonical_mode:
            source_ids = _paper_derived_source_ids(
                beat["evidence_ids"], evidence_by_id, valid_source_paragraph_ids
            )
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


def _paper_story_block_specs(
    story_plan: dict[str, Any],
    clean_evidence: list[dict[str, Any]],
    evidence_map: dict[str, tuple[dict[str, Any], dict[str, Any]]] | None,
    evidence_registry: list[dict[str, Any]] | None = None,
    valid_source_paragraph_ids: set[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Build immutable Python-owned block bindings for each validated beat."""
    clean_by_id = {
        str(record.get("evidence_id") or ""): record
        for record in clean_evidence
        if str(record.get("evidence_id") or "")
    }
    evidence_by_id = _paper_evidence_by_id(evidence_registry or [])
    canonical_anchor_owners = _paper_canonical_anchor_owners(evidence_registry or [])
    specs_by_beat: dict[str, list[dict[str, Any]]] = {}
    for beat in story_plan.get("story_beats") or []:
        beat_id = str(beat.get("id") or "")
        runs: list[tuple[str, list[str]]] = []
        for evidence_id in beat.get("evidence_ids") or []:
            evidence_id = str(evidence_id).strip()
            record = clean_by_id.get(evidence_id)
            if record is None:
                raise RuntimeError(f"PAPER unknown evidence_id: {evidence_id}")
            group = str(record.get("evidence_group") or "context")
            # Keep every immutable evidence item independently addressable so
            # the Article Editor can later merge related blocks into prose.
            runs.append((group, [evidence_id]))

        specs: list[dict[str, Any]] = []
        for index, (group, evidence_ids) in enumerate(runs, start=1):
            figure_ids: list[str] = []
            anchors: list[str] = []
            source_ids: list[str] = []
            for evidence_id in evidence_ids:
                original_section, finding = (
                    evidence_map[evidence_id] if evidence_map is not None else ({}, {})
                )
                if evidence_registry:
                    figure_values = evidence_by_id.get(evidence_id, {}).get("supported_figures") or []
                    # Contextual evidence is placed in the section's prose for
                    # rendering, but its canonical support remains unfigured.
                    if not figure_values:
                        figure_values = (
                            original_section.get("figure_ids")
                            or original_section.get("selected_body_figures")
                            or []
                        )
                else:
                    figure_values = (
                        finding.get("figure_ids")
                        or original_section.get("figure_ids")
                        or original_section.get("selected_body_figures")
                        or []
                    )
                for value in figure_values:
                    normalized = _paper_figure_id(value)
                    if normalized and normalized not in figure_ids:
                        figure_ids.append(normalized)
                anchors.extend(
                    str(anchor)
                    for anchor in clean_by_id[evidence_id].get("anchors") or []
                    if str(anchor).strip()
                    and canonical_anchor_owners.get(
                        (evidence_id, _normalize_evidence_anchor(str(anchor))), evidence_id
                    ) == evidence_id
                )
                if evidence_registry:
                    resolved_sources = _paper_derived_source_ids(
                        [evidence_id], evidence_by_id, valid_source_paragraph_ids
                    )
                else:
                    resolved_sources = list(
                        original_section.get("source_paragraph_ids") or []
                    )
                for source_id in resolved_sources:
                    if source_id not in source_ids:
                        source_ids.append(source_id)
            if evidence_registry:
                source_ids = _paper_derived_source_ids(
                    evidence_ids, evidence_by_id, valid_source_paragraph_ids
                )
            specs.append(
                {
                    "block_id": f"{beat_id}-block-{index}",
                    "beat_id": beat_id,
                    "evidence_group": group,
                    "evidence_ids": tuple(evidence_ids),
                    "figure_ids": tuple(figure_ids),
                    "anchors": tuple(dict.fromkeys(anchors)),
                    "source_paragraph_ids": tuple(source_ids),
                }
            )
        if not specs:
            raise RuntimeError(f"PAPER story beat has no evidence blocks: {beat_id}")
        specs_by_beat[beat_id] = specs
    return specs_by_beat


def _paper_block_specs_by_id(
    block_specs: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for specs in block_specs.values():
        for spec in specs:
            block_id = str(spec.get("block_id") or "")
            if not block_id or block_id in by_id:
                raise RuntimeError(f"PAPER duplicate block_id: {block_id}")
            by_id[block_id] = spec
    return by_id


def _paper_story_block_map(
    raw_blocks: Any,
    expected_ids: list[str],
    stage: str,
    beat_id: str,
) -> dict[str, dict[str, Any]]:
    """Validate only the structural block contract and return blocks by id."""
    missing = list(expected_ids)
    unknown: list[str] = []
    duplicate: list[str] = []
    invalid_text: list[str] = []
    returned_ids: list[str] = []
    if not isinstance(raw_blocks, list):
        invalid_text.append("<blocks>")
    else:
        seen_ids: set[str] = set()
        for index, raw_block in enumerate(raw_blocks):
            if not isinstance(raw_block, dict):
                invalid_text.append(f"<index:{index}>")
                continue
            block_id = str(raw_block.get("block_id") or "").strip()
            if not block_id:
                invalid_text.append(f"<index:{index}>")
            else:
                returned_ids.append(block_id)
                if block_id in seen_ids:
                    if block_id not in duplicate:
                        duplicate.append(block_id)
                else:
                    seen_ids.add(block_id)
            text = raw_block.get("text")
            if not isinstance(text, str) or not text.strip():
                invalid_text.append(block_id or f"<index:{index}>")
        expected_set = set(expected_ids)
        unknown = list(dict.fromkeys(block_id for block_id in returned_ids if block_id not in expected_set))
        missing = [block_id for block_id in expected_ids if block_id not in set(returned_ids)]
    if missing or unknown or duplicate or invalid_text:
        raise RuntimeError(
            f"PAPER {stage} invalid blocks: "
            f"missing={missing!r}; unknown={unknown!r}; duplicate={duplicate!r}; "
            f"invalid_text={invalid_text!r}; beat_id={beat_id!r}"
        )
    return {
        block_id: raw_block
        for block_id, raw_block in zip(returned_ids, raw_blocks)
    }


def _paper_apply_story_output(
    sections: list[dict[str, Any]],
    generated: list[dict[str, Any]],
    evidence_map: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    evidence_registry: list[dict[str, Any]] | None = None,
    block_specs: dict[str, list[dict[str, Any]]] | None = None,
) -> None:
    """Attach immutable Python-owned Figure and source bindings to text blocks."""
    by_id = {str(item.get("id") or ""): item for item in generated}
    if block_specs is None:
        raise RuntimeError("PAPER story output missing immutable block specs")
    specs_by_id = _paper_block_specs_by_id(block_specs)
    seen_block_ids: set[str] = set()
    for section in sections:
        section_id = str(section.get("id") or "")
        item = by_id.get(section_id)
        if item is None:
            raise RuntimeError("PAPER story output omitted a planned section")
        expected_specs = block_specs.get(section_id) or []
        expected_ids = [str(spec.get("block_id") or "") for spec in expected_specs]
        raw_by_id = _paper_story_block_map(
            item.get("blocks"), expected_ids, "story output", section_id
        )
        blocks: list[dict[str, Any]] = []
        for block_id in expected_ids:
            raw_block = raw_by_id[block_id]
            if block_id in seen_block_ids:
                raise RuntimeError(f"PAPER duplicate block_id: {block_id}")
            seen_block_ids.add(block_id)
            spec = specs_by_id[block_id]
            blocks.append(
                {
                    "id": block_id,
                    "evidence_ids": list(spec["evidence_ids"]),
                    "text": raw_block["text"].strip(),
                    "figure_ids": list(spec["figure_ids"]),
                    "source_paragraph_ids": list(spec["source_paragraph_ids"]),
                }
            )
        section["title"] = str(item.get("title") or section.get("title") or "")
        section["blocks"] = blocks
        section["body"] = "\n\n".join(block["text"] for block in blocks)


def _paper_apply_story_candidate(
    plan: dict[str, Any],
    generated: list[dict[str, Any]],
    evidence_map: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    evidence_registry: list[dict[str, Any]],
    block_specs: dict[str, list[dict[str, Any]]],
    display_title: str,
    abstract_lead: str,
    valid_source_ids: set[str],
    figure_evidence_bundles: list[dict[str, Any]] | None,
    current_markdown: str,
    stage: str,
    rollback_on_failure: bool = True,
) -> tuple[bool, str]:
    """Apply a story candidate on a copy and commit it only after hard validation."""
    candidate_plan = copy.deepcopy(plan)
    candidate_sections = candidate_plan.get("sections")
    if not isinstance(candidate_sections, list):
        raise RuntimeError("PAPER story candidate has no sections")
    try:
        _paper_apply_story_output(
            candidate_sections,
            generated,
            evidence_map,
            evidence_registry,
            block_specs,
        )
        candidate_markdown = _paper_assemble_markdown(
            display_title, abstract_lead, candidate_sections
        )
        _validate_paper_evidence_plan(
            candidate_plan,
            candidate_markdown,
            valid_source_ids,
            figure_evidence_bundles,
            evidence_registry,
        )
    except Exception as exc:
        if not rollback_on_failure:
            raise
        diagnostics = []
        evidence_by_id = _paper_evidence_by_id(evidence_registry)
        for beat_specs in block_specs.values():
            for spec in beat_specs:
                diagnostics.append({
                    "block_id": spec.get("block_id"),
                    "evidence_ids": list(spec.get("evidence_ids") or []),
                    "figure_ids": list(spec.get("figure_ids") or []),
                    "source_paragraph_ids": list(spec.get("source_paragraph_ids") or []),
                    "owners": [
                        {
                            "evidence_id": evidence_id,
                            "owner_type": (
                                "quantitative"
                                if evidence_by_id.get(evidence_id, {}).get("anchors")
                                else "contextual"
                            ),
                            "scope": evidence_by_id.get(evidence_id, {}).get("scope"),
                            "source_paragraph_ids": evidence_by_id.get(evidence_id, {}).get("source_paragraph_ids"),
                            "anchors": evidence_by_id.get(evidence_id, {}).get("anchors"),
                        }
                        for evidence_id in spec.get("evidence_ids") or []
                        if evidence_id in evidence_by_id
                    ],
                })
        logger.warning(
            "PAPER %s rewrite rejected; retaining previous valid draft: %s; diagnostics=%s",
            stage,
            exc,
            diagnostics,
        )
        return False, current_markdown
    plan["sections"] = candidate_sections
    return True, candidate_markdown


def _paper_story_draft_blocks(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "beat_id": str(section.get("id") or ""),
            "title": str(section.get("title") or ""),
            "blocks": [
                {
                    "block_id": block.get("id", ""),
                    "text": str(block.get("text") or ""),
                }
                for block in section.get("blocks") or []
            ],
        }
        for section in sections
    ]


def _paper_story_planner(
    client: OpenAI,
    story_skeleton: list[dict[str, Any]],
    style_exemplar: str,
    model: str,
    feedback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_skeleton = [
        {
            "beat_id": skeleton["beat_id"],
            "summary": skeleton["summary"],
            "required_facts": list(skeleton.get("required_facts") or []),
        }
        for skeleton in story_skeleton
    ]
    return _paper_completion_json(
        client,
        PAPER_STORY_PLANNER_PROMPT,
        {
            "editorial_brief_request": {
                "audience": "跨专业、受过高等教育但非该领域专家的读者",
                "purpose": "用几分钟讲清论文最值得知道的科学发现",
                "tone": "清楚、自然、克制、有解释感，不像论文、汇报或营销稿",
            },
            "story_skeleton": model_skeleton,
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
    block_specs: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    sections = response.get("sections")
    if not isinstance(sections, list) or len(sections) != len(story_plan["story_beats"]):
        raise RuntimeError(f"PAPER {stage} returned invalid sections")
    if block_specs is None:
        raise RuntimeError(f"PAPER {stage} missing immutable block specs")
    by_id = {str(item.get("id") or ""): item for item in sections if isinstance(item, dict)}
    output: list[dict[str, Any]] = []
    for beat in story_plan["story_beats"]:
        item = by_id.get(beat["id"])
        if item is None:
            raise RuntimeError(f"PAPER {stage} omitted a story beat")
        specs = block_specs.get(beat["id"]) or []
        expected_ids = [str(spec.get("block_id") or "") for spec in specs]
        raw_by_id = _paper_story_block_map(
            item.get("blocks"), expected_ids, stage, beat["id"]
        )
        blocks: list[dict[str, Any]] = []
        for block_id in expected_ids:
            text = raw_by_id[block_id]["text"]
            clean_text = _paper_plain_language_cleanup(_paper_clean_story_text(text))
            clean_text = re.sub(r"\s*\n+\s*", " ", clean_text).strip()
            blocks.append({"block_id": block_id, "text": clean_text})
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
    block_specs: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Write text for Python-owned blocks, never assign evidence to blocks."""
    evidence_by_id = {record["evidence_id"]: record for record in clean_evidence}
    block_specs = block_specs or _paper_story_block_specs(
        story_plan, clean_evidence, None
    )
    output: list[dict[str, Any]] = []
    for beat in story_plan["story_beats"]:
        beat_id = beat["id"]
        beat_specs = block_specs.get(beat_id) or []
        expected_ids = [spec["block_id"] for spec in beat_specs]
        beat_feedback = feedback or {}
        beat_error = ""
        for attempt in range(2):
            response = _paper_completion_json(
                client,
                PAPER_STORY_WRITER_PROMPT,
                {
                    "editorial_brief": story_plan["editorial_brief"],
                    "story_beat": {
                        key: value
                        for key, value in beat.items()
                        if key != "evidence_ids"
                    },
                    "blocks": [
                        {
                            "block_id": spec["block_id"],
                            "clean_evidence": [
                                _paper_clean_evidence_for_llm(evidence_by_id[evidence_id])
                                for evidence_id in spec["evidence_ids"]
                            ],
                        }
                        for spec in beat_specs
                    ],
                    "style_exemplar": style_exemplar,
                    "targeted_feedback": beat_feedback,
                    "_model": model,
                    "_temperature": 0.25,
                },
            )
            candidate = response if isinstance(response.get("sections"), list) else {
                "sections": [{
                    "id": beat_id,
                    "title": response.get("title") or beat["title"],
                    "blocks": response.get("blocks"),
                }]
            }
            try:
                normalized = _paper_normalize_story_output(
                    candidate,
                    {"story_beats": [beat]},
                    clean_evidence,
                    "story writer",
                    {beat_id: beat_specs},
                )
                output.append(normalized[0])
                break
            except RuntimeError as exc:
                beat_error = str(exc)
                if attempt:
                    fallback_blocks: list[dict[str, str]] = []
                    for spec in beat_specs:
                        block_id = spec["block_id"]
                        fallback_response = _paper_completion_json(
                            client,
                            PAPER_STORY_WRITER_PROMPT,
                            {
                                "editorial_brief": story_plan["editorial_brief"],
                                "story_beat": {
                                    key: value
                                    for key, value in beat.items()
                                    if key != "evidence_ids"
                                },
                                "block": {
                                    "block_id": block_id,
                                    "clean_evidence": [
                                        _paper_clean_evidence_for_llm(evidence_by_id[evidence_id])
                                        for evidence_id in spec["evidence_ids"]
                                    ],
                                },
                                "style_exemplar": style_exemplar,
                                "targeted_feedback": {
                                    **(feedback or {}),
                                    "structure": (
                                        "The beat-level response was structurally invalid. "
                                        f"Write only text for block_id={block_id!r}. "
                                        f"Previous diagnostic: {beat_error}"
                                    ),
                                },
                                "_model": model,
                                "_temperature": 0.25,
                            },
                        )
                        raw_block = fallback_response.get("block")
                        if isinstance(raw_block, dict):
                            text = raw_block.get("text")
                        elif isinstance(fallback_response.get("blocks"), list):
                            fallback_blocks_response = fallback_response["blocks"]
                            text = (
                                fallback_blocks_response[0].get("text")
                                if fallback_blocks_response
                                and isinstance(fallback_blocks_response[0], dict)
                                else None
                            )
                        else:
                            text = fallback_response.get("text")
                        if not isinstance(text, str) or not text.strip():
                            raise RuntimeError(
                                "PAPER story writer per-block fallback invalid text: "
                                f"block_id={block_id!r}"
                            )
                        clean_text = _paper_plain_language_cleanup(
                            _paper_clean_story_text(text)
                        )
                        fallback_blocks.append({
                            "block_id": block_id,
                            "text": re.sub(r"\s*\n+\s*", " ", clean_text).strip(),
                        })
                    output.append({
                        "id": beat_id,
                        "title": _paper_plain_language_cleanup(
                            _paper_clean_story_text(beat["title"])
                        ),
                        "blocks": fallback_blocks,
                        "body": "\n\n".join(
                            block["text"] for block in fallback_blocks
                        ),
                    })
                    break
                beat_feedback = {
                    "structure": (
                        f"Return these block_id values in any order; Python will restore order: "
                        f"{expected_ids!r}. Return only block_id and text; do not return metadata. "
                        f"Error: {beat_error}"
                    )
                }
        else:
            raise RuntimeError(
                f"PAPER story writer failed for beat={beat_id!r}"
            )
    return output


def _paper_article_editor_sections(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for section in sections:
        paragraphs = section.get("paragraphs") or []
        if not paragraphs:
            paragraphs = [
                {
                    "block_ids": [
                        str(block.get("id") or block.get("block_id") or "")
                    ],
                    "text": str(block.get("text") or ""),
                }
                for block in section.get("blocks") or []
                if isinstance(block, dict)
            ]
        output.append({
            "section_id": str(section.get("id") or ""),
            "title": str(section.get("title") or ""),
            "paragraphs": [
                {
                    "block_ids": list(paragraph.get("block_ids") or []),
                    "text": str(paragraph.get("text") or ""),
                }
                for paragraph in paragraphs
                if isinstance(paragraph, dict)
            ],
        })
    return output


def _paper_normalize_article_editor_output(
    response: dict[str, Any],
    plan: dict[str, Any],
    block_specs: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if set(response) != {"sections"} or not isinstance(response.get("sections"), list):
        raise RuntimeError("PAPER article editor returned invalid fields")
    expected_section_ids = {
        str(section.get("id") or "") for section in plan.get("sections") or []
    }
    expected_block_ids = set(_paper_block_specs_by_id(block_specs))
    seen_sections: set[str] = set()
    seen_blocks: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for raw_section in response["sections"]:
        if not isinstance(raw_section, dict) or set(raw_section) != {"section_id", "title", "paragraphs"}:
            raise RuntimeError("PAPER article editor returned invalid section fields")
        section_id = str(raw_section.get("section_id") or "").strip()
        title = str(raw_section.get("title") or "").strip()
        raw_paragraphs = raw_section.get("paragraphs")
        if (
            not section_id
            or section_id not in expected_section_ids
            or section_id in seen_sections
            or not title
            or "\n" in title
            or title.startswith("#")
            or any(_paper_metadata_leakage_lint(title).values())
            or not isinstance(raw_paragraphs, list)
            or not raw_paragraphs
        ):
            raise RuntimeError(
                "PAPER article editor returned invalid section structure: "
                f"section_id={section_id!r}"
            )
        seen_sections.add(section_id)
        paragraphs: list[dict[str, Any]] = []
        for raw_paragraph in raw_paragraphs:
            if not isinstance(raw_paragraph, dict) or set(raw_paragraph) != {"block_ids", "text"}:
                raise RuntimeError("PAPER article editor returned invalid paragraph fields")
            block_ids = raw_paragraph.get("block_ids")
            text = raw_paragraph.get("text")
            normalized_ids = [block_id.strip() for block_id in block_ids] if isinstance(block_ids, list) and all(isinstance(block_id, str) for block_id in block_ids) else []
            if (
                not normalized_ids
                or len(set(normalized_ids)) != len(normalized_ids)
                or any(
                    not block_id
                    or block_id not in expected_block_ids
                    or block_id in seen_blocks
                    for block_id in normalized_ids
                )
                or not isinstance(text, str)
                or not text.strip()
            ):
                raise RuntimeError("PAPER article editor returned invalid paragraph structure")
            seen_blocks.update(normalized_ids)
            clean_text = _paper_plain_language_cleanup(_paper_clean_story_text(text)).strip()
            clean_text = re.sub(r"\s*\n+\s*", " ", clean_text)
            if (
                not clean_text
                or re.search(r"(?m)^\s*(?:#{1,6}\s|>\s|!\[|[-*+]\s|<\/?[A-Za-z])", clean_text)
                or any(_paper_metadata_leakage_lint(clean_text).values())
            ):
                raise RuntimeError("PAPER article editor returned non-prose or metadata text")
            paragraphs.append({"block_ids": normalized_ids, "text": clean_text})
        normalized.append({"id": section_id, "title": title, "paragraphs": paragraphs})
    if seen_sections != expected_section_ids:
        raise RuntimeError(
            "PAPER article editor omitted or duplicated sections: "
            f"missing={sorted(expected_section_ids - seen_sections)!r}; "
            f"unknown={sorted(seen_sections - expected_section_ids)!r}"
        )
    if seen_blocks != expected_block_ids:
        raise RuntimeError(
            "PAPER article editor omitted or duplicated blocks: "
            f"missing={sorted(expected_block_ids - seen_blocks)!r}; "
            f"unknown={sorted(seen_blocks - expected_block_ids)!r}"
        )
    return normalized


def _paper_article_editor(
    client: OpenAI,
    plan: dict[str, Any],
    draft: str,
    abstract: str,
    style_exemplar: str,
    model: str,
    block_specs: dict[str, list[dict[str, Any]]],
    feedback: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    writing_facts = [
        {
            "block_id": spec["block_id"],
            "required_facts": list(spec.get("anchors") or []),
            "has_figure": bool(spec.get("figure_ids")),
        }
        for specs in block_specs.values()
        for spec in specs
    ]
    response = _paper_completion_json(
        client,
        PAPER_ARTICLE_EDITOR_PROMPT,
        {
            "abstract": abstract,
            "draft": draft,
            "sections": _paper_article_editor_sections(plan.get("sections") or []),
            "writing_facts": writing_facts,
            "style_exemplar": style_exemplar,
            "article_feedback": feedback or {},
            "_model": model,
            "_temperature": 0.2,
        },
    )
    return _paper_normalize_article_editor_output(response, plan, block_specs)


def _paper_article_editor_rebuild_sections(
    generated: list[dict[str, Any]],
    plan: dict[str, Any],
    evidence_map: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    evidence_registry: list[dict[str, Any]],
    block_specs: dict[str, list[dict[str, Any]]],
    valid_source_ids: set[str],
) -> list[dict[str, Any]]:
    baseline_by_id = {
        str(section.get("id") or ""): section
        for section in plan.get("sections") or []
    }
    specs_by_id = _paper_block_specs_by_id(block_specs)
    spec_by_evidence: dict[str, dict[str, Any]] = {}
    for spec in specs_by_id.values():
        for evidence_id in spec.get("evidence_ids") or []:
            if evidence_id in spec_by_evidence:
                raise RuntimeError(f"PAPER duplicate evidence ownership: {evidence_id}")
            spec_by_evidence[evidence_id] = spec
    evidence_by_id = _paper_evidence_by_id(evidence_registry)
    rebuilt: list[dict[str, Any]] = []
    for item in generated:
        section_id = str(item.get("id") or "")
        section = copy.deepcopy(baseline_by_id[section_id])
        paragraphs: list[dict[str, Any]] = []
        blocks: list[dict[str, Any]] = []
        evidence_ids: list[str] = []
        for raw_paragraph in item["paragraphs"]:
            paragraph_block_ids = list(raw_paragraph["block_ids"])
            paragraph_text = str(raw_paragraph["text"]).strip()
            paragraph_evidence_ids: list[str] = []
            paragraph_source_ids: list[str] = []
            paragraph_figure_ids: list[str] = []
            for block_id in paragraph_block_ids:
                spec = specs_by_id[block_id]
                for evidence_id in spec.get("evidence_ids") or []:
                    if evidence_id not in evidence_ids:
                        evidence_ids.append(evidence_id)
                    if evidence_id not in paragraph_evidence_ids:
                        paragraph_evidence_ids.append(evidence_id)
                for source_id in spec.get("source_paragraph_ids") or []:
                    source_id = str(source_id).strip()
                    if source_id and source_id not in paragraph_source_ids:
                        paragraph_source_ids.append(source_id)
                for figure_id in spec.get("figure_ids") or []:
                    figure_id = _paper_figure_id(figure_id)
                    if figure_id and figure_id not in paragraph_figure_ids:
                        paragraph_figure_ids.append(figure_id)
            paragraphs.append({
                "block_ids": paragraph_block_ids,
                "text": paragraph_text,
                "evidence_ids": paragraph_evidence_ids,
                "source_paragraph_ids": paragraph_source_ids,
                "figure_ids": paragraph_figure_ids,
            })
            for block_id in paragraph_block_ids:
                spec = specs_by_id[block_id]
                blocks.append({
                    "id": block_id,
                    "evidence_ids": list(spec.get("evidence_ids") or []),
                    "text": paragraph_text,
                    "figure_ids": list(spec.get("figure_ids") or []),
                    "source_paragraph_ids": list(spec.get("source_paragraph_ids") or []),
                })
        section["paragraphs"] = paragraphs
        section["blocks"] = blocks
        section["body"] = "\n\n".join(paragraph["text"] for paragraph in paragraphs)
        section["figure_ids"] = list(dict.fromkeys(
            figure_id
            for block in blocks
            for figure_id in block.get("figure_ids") or []
        ))
        section["source_paragraph_ids"] = _paper_derived_source_ids(
            evidence_ids, evidence_by_id, valid_source_ids
        )
        finding_groups: dict[int, dict[str, Any]] = {}
        finding_order: list[int] = []
        for evidence_id in evidence_ids:
            original_section, original_finding = evidence_map[evidence_id]
            marker = id(original_finding)
            if marker not in finding_groups:
                finding_groups[marker] = {
                    "finding": copy.deepcopy(original_finding),
                    "evidence_ids": [],
                }
                finding_order.append(marker)
            finding_groups[marker]["evidence_ids"].append(evidence_id)
        findings: list[dict[str, Any]] = []
        for marker in finding_order:
            grouped = finding_groups[marker]
            finding = grouped["finding"]
            grouped_ids = grouped["evidence_ids"]
            finding["evidence_ids"] = grouped_ids
            finding["figure_ids"] = list(dict.fromkeys(
                figure_id
                for evidence_id in grouped_ids
                for figure_id in spec_by_evidence[evidence_id].get("figure_ids") or []
            )) or list(finding.get("figure_ids") or [])
            finding["source_paragraph_ids"] = _paper_derived_source_ids(
                grouped_ids, evidence_by_id, valid_source_ids
            )
            if evidence_registry:
                records = [evidence_by_id[evidence_id] for evidence_id in grouped_ids]
                finding["evidence"] = " ".join(
                    str(record.get("source_sentence") or record.get("value") or "").strip()
                    for record in records
                ).strip()
                finding["source_sentence"] = [
                    str(record.get("source_sentence") or "") for record in records
                ]
                finding["anchors"] = list(dict.fromkeys(
                    str(anchor)
                    for record in records
                    for anchor in record.get("anchors") or []
                    if str(anchor).strip()
                ))
            findings.append(finding)
        section["findings"] = findings
        story_beat = dict(section.get("story_beat") or {})
        story_beat["evidence_ids"] = evidence_ids
        section["story_beat"] = story_beat
        section["title"] = str(item["title"]).strip()
        rebuilt.append(section)
    return rebuilt


def _paper_apply_article_editor_candidate(
    plan: dict[str, Any],
    generated: list[dict[str, Any]],
    evidence_map: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    evidence_registry: list[dict[str, Any]],
    block_specs: dict[str, list[dict[str, Any]]],
    display_title: str,
    abstract_lead: str,
    valid_source_ids: set[str],
    figure_evidence_bundles: list[dict[str, Any]] | None,
    current_markdown: str,
    stage: str,
) -> tuple[bool, str]:
    candidate_plan = copy.deepcopy(plan)
    try:
        candidate_plan["sections"] = _paper_article_editor_rebuild_sections(
            generated,
            plan,
            evidence_map,
            evidence_registry,
            block_specs,
            valid_source_ids,
        )
        candidate_markdown = _paper_assemble_markdown(
            display_title, abstract_lead, candidate_plan["sections"]
        )
        _validate_paper_evidence_plan(
            candidate_plan,
            candidate_markdown,
            valid_source_ids,
            figure_evidence_bundles,
            evidence_registry,
        )
    except Exception as exc:
        logger.warning("PAPER %s rewrite rejected; retaining prior draft: %s", stage, exc)
        return False, current_markdown
    plan["sections"] = candidate_plan["sections"]
    return True, candidate_markdown


def _paper_humanize_story(
    client: OpenAI,
    story_plan: dict[str, Any],
    clean_evidence: list[dict[str, Any]],
    draft_blocks: list[dict[str, Any]],
    model: str,
    feedback: dict[str, Any] | None = None,
    block_specs: dict[str, list[dict[str, Any]]] | None = None,
    style_exemplar: str = "",
    target_block_ids: set[str] | None = None,
    feedback_by_block: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Humanize selected text blocks while preserving the complete immutable block list."""
    evidence_by_id = {record["evidence_id"]: record for record in clean_evidence}
    block_specs = block_specs or _paper_story_block_specs(
        story_plan, clean_evidence, None
    )
    draft_by_beat = {
        str(item.get("beat_id") or item.get("id") or ""): item
        for item in draft_blocks
        if isinstance(item, dict)
    }
    output: list[dict[str, Any]] = []
    for beat in story_plan["story_beats"]:
        beat_id = beat["id"]
        draft_item = draft_by_beat.get(beat_id) or {}
        current_by_id = {
            str(block.get("block_id") or block.get("id") or ""): block
            for block in draft_item.get("blocks") or []
            if isinstance(block, dict)
        }
        humanized_blocks: list[dict[str, Any]] = []
        beat_specs = block_specs.get(beat_id) or []
        for block_index, spec in enumerate(beat_specs):
            block_id = spec["block_id"]
            current = current_by_id.get(block_id)
            if current is None:
                raise RuntimeError(f"PAPER humanizer missing block: {block_id}")
            if target_block_ids is not None and block_id not in target_block_ids:
                humanized_blocks.append({
                    "block_id": block_id,
                    "text": str(current.get("text") or "").strip(),
                })
                continue
            adjacent_blocks: list[dict[str, str]] = []
            for neighbor_index in (block_index - 1, block_index + 1):
                if 0 <= neighbor_index < len(beat_specs):
                    neighbor_id = beat_specs[neighbor_index]["block_id"]
                    neighbor = current_by_id.get(neighbor_id)
                    if neighbor is not None:
                        adjacent_blocks.append({
                            "block_id": neighbor_id,
                            "text": str(neighbor.get("text") or "")[:240],
                        })
            block_feedback = dict((feedback_by_block or {}).get(block_id) or {})
            if not block_feedback:
                block_feedback = feedback or {}
            for attempt in range(2):
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
                            "block_id": block_id,
                            "text": str(current.get("text") or ""),
                        },
                        "clean_evidence": [
                            _paper_clean_evidence_for_llm(evidence_by_id[evidence_id])
                            for evidence_id in spec["evidence_ids"]
                        ],
                        "adjacent_blocks": adjacent_blocks,
                        "style_exemplar": style_exemplar,
                        "targeted_feedback": block_feedback,
                        "_model": model,
                        "_temperature": 0.2,
                    },
                )
                try:
                    result = response.get("block") if isinstance(response.get("block"), dict) else response
                    if (
                        not isinstance(result, dict)
                        or set(result) - {"block_id", "text"}
                        or str(result.get("block_id") or "").strip() != block_id
                    ):
                        raise RuntimeError("PAPER humanizer returned an invalid block structure")
                    text = result.get("text")
                    if not isinstance(text, str) or not text.strip():
                        raise RuntimeError("PAPER humanizer returned an empty block")
                    clean_text = _paper_plain_language_cleanup(_paper_clean_story_text(text))
                    clean_text = re.sub(r"\s*\n+\s*", " ", clean_text).strip()
                    humanized_blocks.append({"block_id": block_id, "text": clean_text})
                    break
                except RuntimeError as exc:
                    if attempt:
                        raise
                    block_feedback = {
                        "structure": (
                            f"Return only block_id={block_id!r} and text; "
                            f"do not return evidence_ids or other metadata. Error: {exc}"
                        )
                    }
        output.append({
            "id": beat_id,
            "title": _paper_plain_language_cleanup(
                _paper_clean_story_text(str(draft_item.get("title") or beat["title"]))
            ),
            "blocks": humanized_blocks,
            "body": "\n\n".join(block["text"] for block in humanized_blocks),
        })
    return output


def _paper_style_issue_block_ids(sections: list[dict[str, Any]]) -> set[str]:
    """Map deterministic prose warnings to blocks without exposing metadata to the model."""
    target_ids: set[str] = set()
    for section in sections:
        for block in section.get("blocks") or []:
            if not isinstance(block, dict):
                continue
            block_id = str(block.get("id") or block.get("block_id") or "").strip()
            text = str(block.get("text") or "")
            if not block_id:
                continue
            if _paper_ai_style_lint_failed(_paper_ai_style_lint(text)):
                target_ids.add(block_id)
                continue
            if (
                "；" in text
                or ";" in text
                or re.search(r"(?:第一|第二|第三|第四|首先|其次|再次|最后|图\s*\d+|Fig\.?\s*\d+)", text)
                or _paper_readability_audit(f"## section\n\n{text}")["issue_count"]
            ):
                target_ids.add(block_id)
    return target_ids


def _paper_aggregate_article_review_issues(
    issues: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Coalesce one article-level defect into one actionable revision item."""
    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        text = f"{issue.get('issue', '')} {issue.get('instruction', '')}".lower()
        if any(term in text for term in (
            "机制", "重复", "复述", "暖池", "罗斯贝", "高压", "水汽", "增雪", "遥相关",
            "mechanism", "repeat", "restate", "moisture", "snowfall", "high-pressure", "rossby",
        )):
            key = "mechanism"
        elif any(term in text for term in (
            "术语", "地名", "译名", "名称", "中英文", "英文名", "单位", "混用", "中文名",
            "专名", "实体", "写法", "proper noun", "inconsisten", "terminolog",
        )):
            key = "terminology"
        elif any(term in text for term in (
            "翻译腔", "results", "方法", "统计", "变量", "句式", "ai", "translation", "awkward", "stylist",
        )):
            key = "results_style"
        else:
            key = f"issue:{len(order)}"
        if key not in grouped:
            grouped[key] = {
                "block_ids": [],
                "issue": str(issue.get("issue") or "").strip(),
                "instruction": str(issue.get("instruction") or "").strip(),
            }
            order.append(key)
        target = grouped[key]
        for block_id in issue.get("block_ids") or []:
            if block_id not in target["block_ids"]:
                target["block_ids"].append(block_id)
        current_issue = str(issue.get("issue") or "").strip()
        current_instruction = str(issue.get("instruction") or "").strip()
        if current_issue and current_issue != target["issue"] and current_issue not in target["issue"]:
            target["issue"] = f"{target['issue']}；{current_issue}".strip("；")
        if current_instruction and current_instruction != target["instruction"] and current_instruction not in target["instruction"]:
            target["instruction"] = f"{target['instruction']}；{current_instruction}".strip("；")
    return [grouped[key] for key in order]


def _paper_bounded_article_review(
    review_fn: Any,
    revision_fn: Any,
    max_revisions: int = 2,
) -> tuple[dict[str, Any], list[dict[str, Any]], int, int]:
    latest_review: dict[str, Any] = {"issues": []}
    issues: list[dict[str, Any]] = []
    review_count = 0
    revision_count = 0
    for _ in range(max_revisions):
        reviewed = review_fn()
        latest_review = dict(reviewed or {})
        issues = list(latest_review.get("issues") or [])
        review_count += 1
        if not issues:
            break
        if not revision_fn(issues):
            break
        revision_count += 1
        if revision_count >= max_revisions:
            break
    latest_review["review_count"] = review_count
    latest_review["revision_count"] = revision_count
    return latest_review, issues, review_count, revision_count


def _paper_style_review(
    client: OpenAI,
    style_exemplar: str,
    sections: list[dict[str, Any]],
    _markdown: str,
    model: str,
    article_level: bool = False,
    feedback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Review prose only; never accept edits to text or scientific metadata."""
    review_sections = []
    for section in sections:
        if article_level:
            paragraphs = [
                {
                    "block_ids": list(paragraph.get("block_ids") or []),
                    "text": str(paragraph.get("text") or ""),
                }
                for paragraph in section.get("paragraphs") or []
                if isinstance(paragraph, dict)
            ]
            if not paragraphs:
                paragraphs = [
                    {
                        "block_ids": [str(block.get("id") or block.get("block_id") or "")],
                        "text": str(block.get("text") or ""),
                    }
                    for block in section.get("blocks") or []
                    if isinstance(block, dict)
                ]
            review_sections.append({
                "section_id": str(section.get("id") or ""),
                "title": str(section.get("title") or ""),
                "paragraphs": paragraphs,
            })
        else:
            review_sections.append({
                "section_id": str(section.get("id") or ""),
                "title": str(section.get("title") or ""),
                "blocks": [
                    {
                        "block_id": str(block.get("id") or block.get("block_id") or ""),
                        "text": str(block.get("text") or ""),
                    }
                    for block in section.get("blocks") or []
                    if isinstance(block, dict)
                ],
            })
    article_body = "\n\n".join(
        f"## {section['title']}\n\n"
        + "\n\n".join(
            item["text"]
            for item in (section.get("paragraphs") or section.get("blocks") or [])
        )
        for section in review_sections
    )
    response = _paper_completion_json(
        client,
        PAPER_ARTICLE_STYLE_REVIEWER_PROMPT if article_level else PAPER_STYLE_REVIEWER_PROMPT,
        {
            "style_exemplar": style_exemplar,
            "article_body": article_body,
            "sections": review_sections,
            **({"article_feedback": feedback or {}} if article_level else {}),
            "_model": model,
            "_temperature": 0.1,
        },
    )
    if set(response) != {"issues"} or not isinstance(response.get("issues"), list):
        raise RuntimeError("PAPER style reviewer returned invalid fields")
    if article_level:
        valid_ids = {
            block_id
            for section in review_sections
            for paragraph in section["paragraphs"]
            for block_id in paragraph["block_ids"]
            if block_id
        }
        issues: list[dict[str, Any]] = []
        for issue in response["issues"]:
            if not isinstance(issue, dict) or set(issue) != {"block_ids", "issue", "instruction"}:
                raise RuntimeError("PAPER article style reviewer returned invalid issue fields")
            block_ids = issue.get("block_ids")
            problem = issue.get("issue")
            instruction = issue.get("instruction")
            if (
                not isinstance(block_ids, list)
                or not block_ids
                or len(set(block_ids)) != len(block_ids)
                or any(not isinstance(block_id, str) or block_id not in valid_ids for block_id in block_ids)
                or not isinstance(problem, str)
                or not problem.strip()
                or not isinstance(instruction, str)
                or not instruction.strip()
            ):
                raise RuntimeError("PAPER article style reviewer returned invalid block issue")
            issues.append({
                "block_ids": block_ids,
                "issue": problem.strip(),
                "instruction": instruction.strip(),
            })
        return {"issues": _paper_aggregate_article_review_issues(issues)}
    valid_ids = {
        block["block_id"]
        for section in review_sections
        for block in section["blocks"]
        if block["block_id"]
    }
    issues: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for issue in response["issues"]:
        if not isinstance(issue, dict) or set(issue) != {"block_id", "issue", "instruction"}:
            raise RuntimeError("PAPER style reviewer returned invalid issue fields")
        block_id = issue.get("block_id")
        problem = issue.get("issue")
        instruction = issue.get("instruction")
        if (
            not isinstance(block_id, str)
            or not block_id.strip()
            or block_id not in valid_ids
            or block_id in seen_ids
            or not isinstance(problem, str)
            or not problem.strip()
            or not isinstance(instruction, str)
            or not instruction.strip()
        ):
            raise RuntimeError("PAPER style reviewer returned an invalid block issue")
        seen_ids.add(block_id)
        issues.append({
            "block_id": block_id,
            "issue": problem.strip(),
            "instruction": instruction.strip(),
        })
    return {"issues": issues}


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
    started = time.perf_counter()
    logger.info(
        "PAPER stage=scientific_planner %s",
        "retry" if validation_feedback else "start",
    )
    response = _paper_completion_json(
        client,
        PAPER_PLANNER_PROMPT,
        {
            "abstract": abstract,
            "paper_text": paper_text,
            "source_paragraphs": source_paragraphs,
            "metadata": metadata,
            "selected_body_figures": selected_figure_ids or [],
            "figure_evidence_bundles": figure_evidence_bundles or [],
            "evidence_registry": _paper_registry_for_llm(metadata.get("evidence_registry") or []),
            "figure_backed_evidence_ids": _paper_figure_backed_evidence_ids(
                metadata.get("evidence_registry") or [],
                selected_figure_ids or [],
            ),
            "validation_feedback": validation_feedback,
            "_model": metadata["model"],
            "_temperature": 0.1,
        },
    )
    raw_sections = response.get("sections") if isinstance(response, dict) else None
    logger.info(
        "PAPER stage=scientific_planner response raw_sections=%d elapsed=%.3f",
        len(raw_sections) if isinstance(raw_sections, list) else 0,
        time.perf_counter() - started,
    )
    return response


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
    section_sources = [
        record
        for record in source_paragraphs
        if record["id"] in source_ids
    ]
    section_for_llm = {
        key: value
        for key, value in section.items()
        if key not in {"source_paragraph_ids", "source_sentence", "scope"}
    }
    section_for_llm["findings"] = [
        {
            key: value
            for key, value in finding.items()
            if key not in {"source_paragraph_ids", "source_sentence", "scope"}
        }
        for finding in section.get("findings") or []
    ]
    payload = {
        # The Planner has already used the authoritative Abstract to define
        # the section findings.  Writers must not reuse unrelated Abstract
        # results as evidence for the current Figure bundle.
        "abstract": "",
        "abstract_context": "Abstract structure and lead are locked; use only the current section findings and Figure bundles below for evidence.",
        "section": section_for_llm,
        "figure_evidence_bundles": section_bundles,
        "source_paragraphs": [{"text": record.get("text", "")} for record in section_sources],
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
        paragraphs = section.get("paragraphs") or []
        blocks = section.get("blocks") or []
        if paragraphs:
            body = "\n\n".join(
                str(paragraph.get("text") or "").strip()
                for paragraph in paragraphs
            ).strip()
        elif blocks:
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
        local_source_ids.update(
            str(source_id)
            for source_id in section.get("source_paragraph_ids") or []
            if str(source_id).strip()
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


def _paper_style_lint_text(markdown: str, include_abstract: bool = False) -> str:
    """Lint generated prose while keeping metadata, titles, and source quotes out."""
    sections = _paper_body_sections(markdown)
    body = "\n\n".join(body for _, body in sections)
    if not include_abstract:
        return body or markdown
    first_section = re.search(r"(?m)^##\s+", markdown)
    lead = markdown[: first_section.start()] if first_section else markdown
    lead = re.sub(r"(?m)^#\s+.*?$", "", lead).strip()
    return "\n\n".join(part for part in (lead, body) if part) or markdown


_PAPER_SINGLE_HIT_STYLE_KEYS = frozenset({
    "不是而是",
    "并不是而是",
    "并非而是",
    "并非",
    "并不是",
    "而不是",
    "不在而在",
    "不只是更是",
    "不仅更",
    "真正不是而是",
    "与其说不如说",
    "roughly",
    "由此可见",
    "metadata_leakage",
})


def _paper_metadata_leakage_lint(text: str) -> dict[str, int]:
    patterns = {
        "证据锚点": r"证据锚点",
        "锚点为": r"锚点为",
        "对应的锚点": r"对应的锚点",
        "mandatory anchor": r"mandatory[_ ]+anchor",
        "required fact": r"required[_ ]+fact",
        "evidence_id": r"evidence_id",
        "block_id": r"block_id",
        "provenance": r"provenance",
    }
    return {
        name: len(re.findall(pattern, str(text or ""), flags=re.IGNORECASE))
        for name, pattern in patterns.items()
    }


def _paper_ai_style_lint(markdown: str, include_abstract: bool = False) -> dict[str, int]:
    prose = _paper_style_lint_text(markdown, include_abstract=include_abstract)
    patterns = {
        "不是而是": r"不是[^。！？\n]{0,50}而是",
        "并不是而是": r"并不是[^。！？\n]{0,50}而是",
        "并非而是": r"并非[^。！？\n]{0,50}而是",
        "并非": r"并非",
        "并不是": r"并不是",
        "而不是": r"而不是",
        "不在而在": r"不在[^。！？\n]{0,50}而在",
        "不只是更是": r"不只是[^。！？\n]{0,50}更是",
        "不仅更": r"不仅[^。！？\n]{0,50}更",
        "真正不是而是": r"真正[^。！？\n]{0,50}不是[^。！？\n]{0,50}而是",
        "与其说不如说": r"与其说[^。！？\n]{0,50}不如说",
        "其原因在于": r"其原因在于",
        "也就是说": r"也就是说",
        "不只是": r"不只是",
        "不仅": r"不仅",
        "研究发现": r"研究发现",
        "结果表明": r"结果表明",
        "进一步分析": r"进一步分析",
        "值得注意的是": r"值得注意的是",
        "进一步表明": r"进一步表明",
        "这一发现表明": r"这一发现表明",
        "总体而言": r"总体而言",
        "由此可见": r"由此可见",
        "这意味着": r"这意味着",
        "综上所述": r"综上所述",
        "roughly": r"(?i)\broughly\b",
        "metadata_leakage": (
            r"(?:证据锚点|锚点为|对应的锚点|mandatory[_ ]?anchor|required[_ ]?fact|"
            r"evidence_id|block_id|provenance)"
        ),
        "作者式第一人称": (
            r"(?:我室|咱们|我们|研究团队在此(?:表明|显示|发现)|在此(?:表明|显示|发现)|"
            r"本研究(?:发现|展示|使用|进一步分析|的结果)|"
            r"本文(?:发现|展示|使用|进一步分析|的结果))"
        ),
    }
    counts = {name: len(re.findall(pattern, prose)) for name, pattern in patterns.items()}
    if include_abstract:
        title_text = "\n".join(title for title, _ in _paper_body_sections(markdown))
        counts["metadata_leakage"] += len(
            re.findall(patterns["metadata_leakage"], title_text)
        )
    return counts


def _paper_ai_style_lint_failed(counts: dict[str, int]) -> bool:
    return bool(counts.get("作者式第一人称")) or any(
        counts.get(key, 0) > 0 for key in _PAPER_SINGLE_HIT_STYLE_KEYS
    ) or any(
        count > 1 for key, count in counts.items()
        if key not in _PAPER_SINGLE_HIT_STYLE_KEYS and key != "作者式第一人称"
    ) or sum(counts.values()) > 4


def _paper_require_clean_final_style_lint(counts: dict[str, int]) -> None:
    if _paper_ai_style_lint_failed(counts):
        raise RuntimeError("PAPER final style lint failed after local fallback")


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
    if lint.get("作者式第一人称", 0):
        issues.append({
            "type": "author_voice",
            "count": lint["作者式第一人称"],
        })
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
    paragraph_openings: list[str] = []
    for _, body in sections:
        for paragraph in re.split(r"\n\s*\n", body.strip()):
            first_sentence = re.split(r"(?<=[。！？.!?])\s*", paragraph.strip(), maxsplit=1)[0]
            opening = re.sub(r"\s+", "", first_sentence)[:16]
            if opening:
                paragraph_openings.append(opening)
    repeated_paragraph_openings = {
        opening for opening in paragraph_openings if paragraph_openings.count(opening) > 1
    }
    if repeated_paragraph_openings:
        issues.append({
            "type": "repeated_paragraph_opening",
            "openings": sorted(repeated_paragraph_openings),
        })
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


def _is_title_translation_retryable(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
    try:
        if int(status_code) == 429 or int(status_code) >= 500:
            return True
    except (TypeError, ValueError):
        pass
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    return (
        isinstance(exc, TimeoutError)
        or "timeout" in name
        or "timed out" in name
        or "timeout" in message
        or "timed out" in message
    )


def _paper_title_completion_with_retry(client: OpenAI, **kwargs: Any) -> Any:
    for attempt in range(2):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            if attempt == 0 and _is_title_translation_retryable(exc):
                time.sleep(3)
                continue
            raise
    raise RuntimeError("unreachable paper title translation retry state")


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


def select_paper_ranked(
    candidates: list[dict[str, Any]],
    settings: Settings,
    *,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], bool, str]:
    fallback = [
        dict(item, title_cn=str(item.get("title_cn") or ""))
        for item in candidates
        if int(item.get("paper_local_score") or 0) >= 2
    ]
    if limit is not None:
        fallback = fallback[:limit]
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
        for index, item in enumerate(candidates, start=1)
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
                        "Comment、Correspondence。只返回评分为2或3的论文，3分优先；返回所有符合条件的论文，"
                        "不要截断为固定数量，也不要凑数。"
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
        ranked = sorted(scored, key=lambda value: (-value[0], value[1]))
        if limit is not None:
            ranked = ranked[:limit]
        for score, index, title_cn in ranked:
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


def select_paper_top_ten(
    candidates: list[dict[str, Any]],
    settings: Settings,
) -> tuple[list[dict[str, Any]], bool, str]:
    return select_paper_ranked(candidates, settings, limit=10)


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
        response = _paper_title_completion_with_retry(
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


def _paper_abstract_grammar_cleanup(text: str) -> str:
    cleaned = re.sub(
        r"(在[^，。！？]{2,30}年间)，\s*然而，",
        r"然而，\1，",
        str(text or ""),
    )
    cleaned = re.sub(
        r"((?:\d{4}[—–-]\d{4}|\d{4})年(?:间)?)[，,]\s*然而，",
        r"然而，\1，",
        cleaned,
    )
    cleaned = re.sub(r"(?<![A-Za-z0-9])(\d{3})\s*百帕", r"\1 hPa", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _paper_deauthor_abstract(text: str) -> str:
    """Remove first-person author voice without changing Abstract information."""
    cleaned = _paper_abstract_grammar_cleanup(text)
    replacements = (
        (r"我们在此表明", "研究显示"),
        (r"我们在此显示", "研究显示"),
        (r"我们在此发现", "结果显示"),
        (r"在此[，,]\s*我们表明", "研究显示"),
        (r"在此[，,]\s*我们显示", "研究显示"),
        (r"在此[，,]\s*我们发现", "结果显示"),
        (r"我们的结果", "研究结果"),
        (r"我们的研究", "该研究"),
        (r"我们的分析", "研究分析"),
        (r"我们发现", "结果显示"),
        (r"我们表明", "研究显示"),
        (r"我们显示", "结果显示"),
        (r"我们观察到", "观察结果显示"),
        (r"我们使用", "研究使用"),
        (r"我们分析", "研究分析"),
        (r"我们估计", "研究估计"),
        (r"我们提出", "研究提出"),
        (r"我们认为", "研究认为"),
        (r"在此[，,]\s*我们", "研究"),
        (r"我们", "该研究"),
    )
    for pattern, replacement in replacements:
        cleaned = re.sub(pattern, replacement, cleaned)
    return _paper_abstract_grammar_cleanup(cleaned)


def translate_paper_abstract(abstract: str, settings: Settings) -> str:
    """Translate the original paper Abstract faithfully and remove author voice."""
    source = re.sub(r"\s+", " ", str(abstract or "")).strip()
    source = _paper_remove_inline_citation_markers(source)
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
                    "地名按语义判断：有标准且明确中文译名的地理名称正常翻译，例如East Antarctica译为东南极、"
                    "West Antarctica译为西南极、Indian Ocean译为印度洋、Southern Ocean译为南大洋、"
                    "North Atlantic译为北大西洋、Arctic译为北极；生僻、具体、命名型地理实体如果中文译名不确定，"
                    "保留原始英文，例如Queen Mary Land、Wilkes Land，但不得把Antarctic moistening、"
                    "Antarctic precipitation、Antarctic warming或Rossby wave当作地名，必须译为中文。"
                    "不要让公众号文风、标题风格或正文内容影响摘要。不要添加小标题、列表或解释，只返回严格JSON："
                    '{"abstract_cn":"..."}'
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "abstract": source,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    )
    parsed = _json_from_text(response.choices[0].message.content or "")
    translated = re.sub(r"\s+", " ", str(parsed.get("abstract_cn") or "")).strip()
    if not translated:
        raise RuntimeError("model returned empty Chinese Abstract translation")
    return _paper_remove_inline_citation_markers(_paper_deauthor_abstract(translated))


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


def _paper_safe_caption(text: str, max_chars: int = 160) -> str:
    caption = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(caption) <= max_chars:
        return caption
    boundaries = [
        match.end()
        for match in re.finditer(r"[。！？；.!?;]", caption)
        if match.end() <= max_chars
    ]
    return caption[: max(boundaries)] if boundaries else caption


def generate_image_captions(
    images: list[dict[str, Any]],
    settings: Settings,
    terminology_context: str = "",
    targeted_english_fallback: bool = False,
) -> list[str]:
    """Generate independent Chinese captions from each image's text metadata."""
    if not images or not settings.model_configured:
        return [""] * len(images)
    payload = [
        {
            "index": index,
            "title": image.get("metadata_title", ""),
            "caption": str(
                (
                    image.get("original_caption") or image.get("caption") or ""
                )
                if targeted_english_fallback
                else (image.get("caption") or "")
            )[:1500],
            "description": str(image.get("description") or image.get("alt") or "")[:1000],
            "provider": image.get("provider", ""),
        }
        for index, image in enumerate(images, start=1)
    ]
    request_payload: dict[str, Any] = {"images": payload}
    if terminology_context.strip():
        request_payload["terminology_context"] = re.sub(
            r"\s+", " ", terminology_context
        ).strip()[:6000]
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
                        (
                            "将每张图片的英文原始caption忠实翻译成简短中文图注；不得增添原文没有的科学内容。"
                            if targeted_english_fallback
                            else "根据每张图片各自的文本metadata，独立生成简短、准确的中文图注。"
                        )
                        + "必须描述该图片实际展示的内容，不能仅根据文章主题写通用句子，"
                        "不同图片不得复用同一句图注。不得输出外部图片元数据或 URL、"
                        "图库名称或英文长caption。metadata不足时caption_cn返回空字符串。"
                        "如果提供terminology_context，同一实体沿用其中已经使用的中文名称，不要自行重新翻译。"
                        "地名按语义判断：有标准且明确中文译名的大尺度地理区域正常翻译，例如East Antarctica译为东南极、"
                        "West Antarctica译为西南极、Indian Ocean译为印度洋；生僻、具体、命名型地理实体若中文译名不确定则保留准确英文。"
                        "普通科学表达如Antarctic moistening、Antarctic precipitation、Antarctic warming和Rossby wave必须翻译。"
                        "不要添加‘图1’等编号。返回严格JSON："
                        '{"items":[{"index":1,"caption_cn":"..."}]}。'
                    ),
                },
                {"role": "user", "content": json.dumps(request_payload, ensure_ascii=False)},
            ],
        )
        parsed = _json_from_text(response.choices[0].message.content or "")
        captions = [""] * len(images)
        used: set[str] = set()
        for item in parsed.get("items", []):
            index = int(item.get("index", 0))
            caption = _paper_plain_language_cleanup(str(item.get("caption_cn") or "")).strip()
            caption = re.sub(r"^图\s*\d+\s*[.、：:]\s*", "", caption)
            if (
                index < 1
                or index > len(images)
                or not caption
                or not re.search(r"[一-鿿]", caption)
                or caption in used
            ):
                continue
            captions[index - 1] = _paper_safe_caption(caption)
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

PAPER_NARRATOR_CONTRACT = (
    "叙述者契约：PAPER是第三方科学公众号编辑，不是论文作者。正文不得出现作者式第一人称“我们”或“咱们”，"
    "也不得使用“我室”、“本研究发现”、“本文发现”等作者口吻；包括“我们发现”、“我们看到”、“我们可以看到”、"
    "“我们展示”、“我们使用”、“我们进一步分析”、“我们注意到”和“我们的结果”。"
    "应改为自然的第三方表达，但不要因此反复制造“研究发现”或“结果表明”等模板句。"
    "英文原文引用和论文题目保持原样，不要改写其中的第一人称。"
)

PAPER_STORY_PLANNER_PROMPT = (
    "你是Story Planner，先读懂房间，再为已经通过Figure-first科学验证的证据设计自然的公众号故事线。"
    "读者是对科学感兴趣但非本领域专家的普通读者；目的不是逐项汇报Results，而是像人在解释一个值得知道的科学发现。"
    "style_exemplar中的范文原文是主要写作参考，STYLE_GUIDE只是辅助规则；学习句法、段落长度、信息密度、叙事推进、术语解释和自然中文，禁止复制范文事实、数字、人物、地点和结论。"
    "先确定editorial_brief：audience、purpose、tone、reader_should_leave_with（读者记住的2到3个观点）和story_question。"
    "故事形状由当前证据决定，不套固定的A→B→C公式；可以从现象、结果、机制或影响切入，结论和方法的先后以自然表达为准，方法只保留帮助理解结果的部分。"
    "Python已经根据科学验证结果建立固定story skeleton，每个scientific section对应一个固定beat；不要重排、合并、拆分或重新分配beat。"
    "你只需为每个固定beat填写title、reader_question、core_message和transition_to_next，不能返回任何evidence、source、Figure或provenance字段。故事顺序和证据归属由Python保持。"
    "每个beat的summary和required_facts仅用于理解该固定section的科学主题；不要在输出中复述系统字段或任何内部标识。标题必须专业、直接、简洁，优先10到22个中文字，直接陈述科学结果。避免为何、线索、改写、同一片中国、谁在主导、真正的答案、背后的秘密等媒体化措辞。不能使用第一、第二、第三、第四、首先、其次、最后，也不能提及任何图、Figure、panel或source。"
    "只学习style_exemplar的中文节奏、句长、信息密度和推进方式，不复制其中的科学事实、数字、地点、机制或句子。"
    "返回严格JSON："
    '{"editorial_brief":{"audience":"...","purpose":"...","tone":"...","reader_should_leave_with":"...","story_question":"..."},'
    '"story_beats":[{"id":"beat-1","title":"...","reader_question":"...","core_message":"...","transition_to_next":"..."}]}'
)

PAPER_STORY_WRITER_PROMPT = (
    "你是Story Writer，为跨专业科研读者写专业、简洁、易懂的中文科学公众号正文。你只能看到按beat分组的clean evidence和story beats，"
    "绝不能提及或猜测图号、Figure、panel、source id，也不要按证据编号或资料顺序逐项汇报。"
    "每个beat只能使用其对应的clean evidence；在固定evidence/block边界内参考style corpus自然组织正文，不要为了结构完整硬加转折、总结句或解释句。"
    "style_exemplar中的范文原文是主要写作参考，STYLE_GUIDE只是辅助规则；学习句法、段落长度、信息密度、叙事推进、术语解释和自然中文，禁止复制范文事实、数字、人物、地点和结论。"
    "标题和正文以style corpus的自然表达为主要参考，清楚、克制即可，不把10到22字、固定小标题或禁用词清单当成硬模板。"
    "正文不要写成论文Results、摘要扩写、图注翻译或营销型自媒体；在当前evidence/block边界内自然组织，不为了结构完整硬加转折、总结句或解释句。方法、变量清单和统计术语只保留确实有助于理解的部分。"
    "句式、句长和段落节奏参考style corpus自然变化；专业词在需要时顺手解释，避免连续堆缩写、模型名和参数，不把人工规则写成排比模板。"
    "每个beat必须按输入的固定blocks分别写作。Python已经决定每个block_id及其对应的科学证据边界；不得新增、删除、重排、合并或拆分block，不得分配或返回evidence_ids。"
    "每个block只能使用输入中该block的clean_evidence；不能把不同Figure group的证据混入，也不能把anchor移动到另一个block。clean_evidence中的每个anchor必须在对应block正文中原样保留。"
    "本次只写当前story beat，按证据需要保持紧凑；全篇长度由Python汇总审计，不要把每个beat或block机械写成等长。"
    + PAPER_FIDELITY_CONTRACT
    + PAPER_NARRATOR_CONTRACT
    + "本次只写当前story beat，严格按输入blocks顺序返回；每个block只能包含block_id和text。返回严格JSON：{\"title\":\"...\",\"blocks\":[{\"block_id\":\"beat-1-block-1\",\"text\":\"...\"}]}。"
)

PAPER_ARTICLE_EDITOR_PROMPT = (
    "你是中文科学新闻的Article Editor。一次性阅读完整PAPER草稿、Abstract、全部sections和全部自然段，"
    "并把五篇范文全文当作主要参考，学习整篇文章的组织、信息推进、段落节奏、句法和自然中文。"
    "根据当前论文自己的科学内容重新组织叙事：可以重排sections和自然段，也可以把多个相邻或相关block合并到同一个自然段，"
    "让已解释的机制只完整出现一次，后文用简短承接推进新证据或意义。不要把文章写成Figure目录，不要按图号顺序汇报，"
    "不要复制范文事实、数字、人物、地点或结论；方法只保留帮助理解结论所需的部分。"
    "术语和地名必须全文一致：地名按语义判断，有标准且明确中文译名的地理名称正常翻译，例如East Antarctica译为东南极、"
    "West Antarctica译为西南极、Indian Ocean译为印度洋、Southern Ocean译为南大洋、North Atlantic译为北大西洋、"
    "Arctic译为北极；生僻、具体、命名型地理实体如果中文译名不确定，保留准确的原始英文，例如Queen Mary Land、Wilkes Land。"
    "普通科学表达必须翻译，例如Antarctic moistening、Antarctic precipitation、Antarctic warming和Rossby wave分别使用自然中文，"
    "不得因为包含Antarctic或Antarctica就保留整段英文。Abstract和正文首次明确采用的写法优先沿用，后文不得交替使用英文名或不同中文译名。"
    "writing_facts中的required_facts只是必须保留的论文事实，has_figure只是附近需要承载图的提示；"
    "不要在正文提到required fact、evidence、证据、锚点、约束、block、metadata、provenance或任何系统概念。"
    "不得发明事实、因果、意义、数字或来源。每个原有block_id必须在全文一个且仅一个paragraph的block_ids中出现，"
    "不得新增、删除或重复block_id；block_ids只用于Python恢复事实归属，不要把它们写进正文。"
    "返回严格JSON且只能包含sections。每个section使用section_id、title和paragraphs；每个paragraph只能使用block_ids和text。"
    "所有原有section_id和block_id必须各出现恰好一次，允许sections、paragraphs和block_ids任意重排。"
    + PAPER_FIDELITY_CONTRACT
    + PAPER_NARRATOR_CONTRACT
    + "禁止明显AI或论文翻译腔，包括我们、咱们、并非、并不是、而不是、不是……而是、值得注意的是、这意味着、综上所述等。"
    + "输出格式：{\"sections\":[{\"section_id\":\"...\",\"title\":\"...\",\"paragraphs\":[{\"block_ids\":[\"...\"],\"text\":\"...\"}]}]}"
)


PAPER_ARTICLE_STYLE_REVIEWER_PROMPT = (
    "你是只读的整篇中文科学新闻Style Reviewer。完整style_exemplar中的五篇范文原文是主要参考。"
    "检查整篇文章的组织、信息推进、段落节奏、自然中文和科学新闻感，而不是逐block挑句子。"
    "必须检查：跨section机制是否重复完整解释、后文是否只是换词复述、是否隐含按Figure顺序、开头和结尾是否重复、"
    "术语和地名是否一致：Abstract和正文首次采用的标准中文译名优先沿用；中文译名不确定的生僻命名型地理实体可保持准确英文，后文不得交替使用不同写法；"
    "普通科学概念如Rossby wave应统一译为‘罗斯贝波’，不得中英文或多个中文译名混用；"
    "还要检查中英文是否异常混杂、roughly或300百帕等不自然表达、以及Results翻译腔和AI对立句。"
    "检查普通英文 prose 或科学短语是否意外未翻译；不要把合法缩写、单位、拉丁学名或有意保留的生僻命名型地理名称误报为问题。"
    "如果前文已经完整解释暖池增温—罗斯贝波—高压—水汽—增雪机制，后文只应补充recurrence、反馈、暂时性或长期边界，不得再次完整复述起点到终点。"
    "每个跨paragraph问题必须一次性列出全部受影响的已有block_ids，并给出一条整篇revision_instruction；不要分别制造局部修句任务。"
    "不要修改科学事实，不要发明证据，不要修改或返回evidence/source/Figure元数据。没有明确问题返回空issues。"
    "返回严格JSON且只能包含issues："
    "{\"issues\":[{\"block_ids\":[\"block-id\"],\"issue\":\"article-level issue\",\"instruction\":\"whole-article revision instruction\"}]}"
)


PAPER_HUMANIZER_PROMPT = (
    "你是中文母语科学编辑，依据ai-zixun/humanizer-zh的原则，对Story Writer成稿做一次保守的人文化编辑。"
    "输入按beat再按paragraph block分组；每次只能修改当前block，可只读少量相邻block文本帮助衔接，但不能重写、合并或移动其他block的正文。"
    "每个block_id及其证据边界是Python设定的硬边界，不能合并block、拆出跨组句子、移动finding或把另一组Figure的结果带进来。"
    "style_exemplar中的范文原文是主要写作参考，STYLE_GUIDE、humanizer、stop-slop、shuorenhua和人工规则都只是辅助校对；"
    "优先模仿范文的句法、段落节奏、信息密度、叙事推进、术语解释和自然中文。"
    "只做保守校对：修正明显翻译腔、作者式第一人称、AI套话、机械重复、方法/变量堆积和生硬衔接，不重新设计段落结构。"
    "不要把正文改成论文Results转述，也不要为了‘像公众号’硬加结论、转折、总结或解释句；段落形状和句式以范文自然写法为准。"
    "标题清楚、克制即可，不把长度或禁用词清单当成硬模板；专业词在需要时顺手解释，避免连续堆缩写。"
    "每个block对应的anchor必须原样保留，不能因润色而删除、改写、重复或移动。"
    "非anchor的细节可以删减，但不能新增事实、机制、意义或因果关系；不新增事实。"
    "若targeted_feedback指出技术密度或模板风险，优先删除方法、变量和公式清单，只保留当前block理解结论所需的信息。"
    + PAPER_FIDELITY_CONTRACT
    + PAPER_NARRATOR_CONTRACT
    + "只返回严格JSON：{\"block\":{\"block_id\":\"beat-1-block-1\",\"text\":\"...\"}}。"
)

PAPER_STYLE_REVIEWER_PROMPT = (
    "你是只读的中文科学写作Style Reviewer。完整style_exemplar是主要参考，STYLE_GUIDE和人工规则只作辅助校对。"
    "检查完整正文是否像自然的science news/explainer，而不是论文Results、摘要扩写或图注翻译。"
    "只指出结果翻译腔、AI语气、机械对仗或转折、作者式第一人称、重复同构句式、方法/变量堆叠、生硬衔接，"
    "以及与范文语法、段落节奏、信息密度和叙事推进明显偏离的问题。不要评价或改动科学事实。"
    "不能移动证据、调整block或section、修改Figure和anchor、改变来源、重排章节或新增科学结论。"
    "每个问题只指定一个已有block；没有明确问题就返回空issues。每个block最多一个问题。"
    "返回严格JSON且只能包含issues字段："
    '{"issues":[{"block_id":"beat-1-block-1","issue":"...","instruction":"..."}]}'
)


PAPER_PLANNER_PROMPT = (
    "你是Figure-first Scientific Planner，不写文章正文。根据Abstract、paper_text、source_paragraphs、selected_body_figures和figure_evidence_bundles，"
    "建立唯一的paper_evidence_plan，并只返回严格JSON对象。正文科学骨架必须来自selected body Figures及其真实evidence bundles；Figure ownership is Python-managed，不要根据自己的判断重新分配evidence到Figure。"
    "每个section包含id、title、role和findings；source provenance与真实Figure mapping均由Python根据evidence_ids和canonical evidence registry推导，禁止返回source_paragraph_ids、source_sentence或其他source字段。title必须是适合中文成稿的简洁中文小标题；每个finding包含id和evidence_ids。"
    "不要返回或依赖figure_ids、selected_body_figures等Figure ownership字段；即使兼容旧JSON格式返回这些字段，Python也会忽略它们。evidence_ids必须来自输入registry，Figure mapping将由Python根据canonical supported_figures自动恢复。"
    "输入中的figure_backed_evidence_ids按selected Figure列出合法的Figure-backed evidence_id；有selected Figure时，每个section至少选择一条对应列表中的evidence_id，可搭配global_context或section_context，但不要由模型重新推断Figure归属。若selected_body_figures或figure_evidence_bundles为空，仍必须根据paper_text和source_paragraphs规划至少一个有证据支持的section。"
    "每个核心finding只能有一个primary section。若historical/model spread、mechanism、attribution、projection或implication"
    "是不同科学问题且各有独立Figure bundle证据，按真实Figure证据拆分；不要为凑section数量而合并不相关Figure，也不要固定section数量。"
    "只有Abstract或Results明确支持时才拆分multiple modes/regimes，不得创造first/second mode。"
    "每个Figure bundle的核心finding只能进入包含该Figure的section；Figure 只可通过bundle中的caption、明确引用段落和直接关联Results段落支持正文。不要把Fig.2的R=0.71写入只包含Fig.3的section。"
    "无独立主图的机制内容只能作为最相关Figure section中的2到3句解释，不要新建无图机制section；只有删除会造成明显科学逻辑断裂时才保留无图短过渡。"
    "role使用贴合论文的简洁自然标签，不要套固定taxonomy。只能引用输入registry中真实存在的evidence_id，不能生成或修改source provenance。"
    "每个finding都要有evidence_ids数组；不要返回anchors、source_paragraph_ids、source_sentence或scope，anchors和全部source provenance由Python registry推导。"
    "涉及数字、百分比或统计量时，必须引用对应的quantitative evidence-anchor记录；evidence-source-source-*段落记录仅作上下文，anchors为空，不能替代数字owner。"
    "如果输入包含validation_feedback，必须优先修复其中指出的Figure、source或anchor归属，不能重复提交同一错误计划。"
    '返回格式：{"sections":[{"id":"section-1","title":"...","role":"attribution","findings":[{"id":"E1","evidence_ids":["复制registry中的真实evidence_id"]}]}]}'
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
    + PAPER_NARRATOR_CONTRACT
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
    selected_body_images = dossier.get("paper_selected_body_images")
    figure_first = isinstance(selected_body_images, list) and bool(selected_body_images)
    selected_images = (
        list(selected_body_images)
        if figure_first
        else list(dossier.get("images") or [])
    )
    figure_evidence_bundles = _paper_figure_evidence_bundles(
        selected_images,
        source_paragraphs,
    )
    evidence_registry = _paper_canonical_evidence_registry(
        source_paragraphs,
        figure_evidence_bundles,
    )
    evidence_by_id = _paper_evidence_by_id(evidence_registry)
    selected_figure_ids = [str(bundle["figure_id"]) for bundle in figure_evidence_bundles]
    dossier["paper_evidence_registry"] = evidence_registry
    dossier["paper_figure_evidence_bundles"] = figure_evidence_bundles
    metadata = {
        "title": dossier.get("title", ""),
        "display_title": display_title,
        "doi": dossier.get("doi", ""),
        "journal": dossier.get("journal", ""),
        "authors": dossier.get("authors", []),
        "selected_body_figures": selected_figure_ids,
        "figure_evidence_bundles": figure_evidence_bundles,
        "evidence_registry": evidence_registry,
        "figure_captions": [
            {"caption": image.get("caption", "")}
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
    scientific_planner_started = time.perf_counter()
    planner_retry_count = 0
    plan = _paper_plan(
        client,
        abstract,
        paper_text,
        source_paragraphs,
        metadata,
        selected_figure_ids,
        figure_evidence_bundles,
    )
    if not isinstance(plan.get("sections"), list) or not plan["sections"]:
        planner_retry_count += 1
        logger.warning("PAPER scientific planner returned empty sections; retrying once")
        plan = _paper_plan(
            client,
            abstract,
            paper_text,
            source_paragraphs,
            metadata,
            selected_figure_ids,
            figure_evidence_bundles,
            "The previous planner response had no sections. Return at least one section supported by the supplied paper_text or source_paragraphs. Use Figure-specific evidence when available, but allow global_context or section_context when no unique Figure supports the content. Do not invent results.",
        )
    try:
        plan["sections"] = _validate_paper_plan_structure(
            plan,
            valid_source_ids,
            set(selected_figure_ids) if figure_first else None,
            evidence_registry,
        )
    except RuntimeError as exc:
        if not _paper_planner_structure_retryable(str(exc)):
            raise
        planner_retry_count += 1
        logger.warning(
            "PAPER planner structure validation failed; retrying once: %s",
            exc,
        )
        allowed_figure_evidence = _paper_figure_backed_evidence_ids(
            evidence_registry,
            selected_figure_ids,
        )
        validation_feedback = (
            f"The previous planner response failed deterministic validation: {exc}. "
            "Return only valid section metadata and evidence_ids from the supplied registry. "
            "Figure ownership is Python-managed: do not return or rely on figure_ids or selected_body_figures. "
            "If a section has no Figure-backed evidence, select at least one supplied evidence record "
            "from the following allowed Figure-backed evidence IDs, without inventing or remapping provenance: "
            f"{json.dumps(allowed_figure_evidence, ensure_ascii=False, sort_keys=True)}"
        )
        plan = _paper_plan(
            client,
            abstract,
            paper_text,
            source_paragraphs,
            metadata,
            selected_figure_ids,
            figure_evidence_bundles,
            validation_feedback,
        )
        plan["sections"] = _validate_paper_plan_structure(
            plan,
            valid_source_ids,
            set(selected_figure_ids) if figure_first else None,
            evidence_registry,
        )
    logger.info(
        "PAPER stage=scientific_planner normalized_sections=%d selected_figure_ids=%s "
        "planner_evidence_sections=%s derived_section_figures=%s retry_count=%d elapsed=%.3f",
        len(plan["sections"]),
        selected_figure_ids,
        [
            {
                "section_id": section.get("id"),
                "evidence_ids": [
                    evidence_id
                    for finding in section.get("findings") or []
                    for evidence_id in finding.get("evidence_ids") or []
                ],
            }
            for section in plan["sections"]
        ],
        [
            {
                "section_id": section.get("id"),
                "figure_ids": section.get("figure_ids") or [],
            }
            for section in plan["sections"]
        ],
        planner_retry_count,
        time.perf_counter() - scientific_planner_started,
    )
    for section in plan["sections"]:
        section.pop("necessary_transition", None)
        section.pop("transition_reason", None)
    plan["figure_evidence_bundles"] = figure_evidence_bundles
    plan["evidence_registry"] = evidence_registry
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
            evidence_registry,
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
            evidence_registry,
        )
        logger.info(
            "PAPER stage=scientific_planner normalized_sections=%d",
            len(plan["sections"]),
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
            evidence_registry,
        )
    logger.info(
        "PAPER stage=scientific_planner done elapsed=%.3f",
        time.perf_counter() - scientific_planner_started,
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
            evidence_registry,
        )
        logger.info("PAPER deterministic evidence validation passed")

    plan["scientific_review"] = scientific_review

    style_exemplar = _paper_style_exemplar()
    clean_evidence, evidence_map = _paper_clean_story_evidence(
        plan,
        source_paragraphs,
        evidence_registry,
    )
    story_skeleton = _paper_story_skeleton(plan, clean_evidence, evidence_map)
    story_plan = _paper_story_planner(
        client,
        story_skeleton,
        style_exemplar,
        settings.model_name,
    )
    try:
        story_beats = _paper_validate_story_plan(story_plan, story_skeleton)
    except RuntimeError as exc:
        logger.warning("PAPER story planner returned an invalid beat plan; retrying once: %s", exc)
        story_plan = _paper_story_planner(
            client,
            story_skeleton,
            style_exemplar,
            settings.model_name,
            {"validation": "Return exactly one beat for each fixed beat_id in the supplied skeleton, in the same order. Do not return evidence, source, Figure, or provenance fields."},
        )
        story_beats = _paper_validate_story_plan(story_plan, story_skeleton)
    story_sections = _paper_story_sections(
        story_beats,
        evidence_map,
        evidence_registry,
        valid_source_ids,
    )
    block_specs = _paper_story_block_specs(
        story_plan,
        clean_evidence,
        evidence_map,
        evidence_registry,
        valid_source_ids,
    )
    plan["story_plan"] = story_plan
    plan["sections"] = story_sections
    story_evidence: dict[str, dict[str, Any]] = {}
    for evidence_id, (original_section, finding) in evidence_map.items():
        canonical = evidence_by_id.get(evidence_id)
        if canonical is None:
            raise RuntimeError(f"PAPER unknown evidence_id: {evidence_id}")
        story_evidence[evidence_id] = {
            "figure_ids": list(
                dict.fromkeys(
                    _paper_figure_id(value)
                    for value in (
                        canonical.get("supported_figures")
                        if evidence_registry
                        else (
                            finding.get("figure_ids")
                            or original_section.get("figure_ids")
                            or original_section.get("selected_body_figures")
                            or []
                        )
                    )
                    if str(value).strip()
                )
            ),
            "anchors": [
                str(anchor)
                for anchor in canonical.get("anchors") or []
                if str(anchor).strip()
            ],
            "source_paragraph_ids": list(canonical.get("source_paragraph_ids") or []),
            "source_sentence": str(canonical.get("source_sentence") or ""),
            "scope": str(canonical.get("scope") or "section_context"),
            "supported_figures": list(canonical.get("supported_figures") or []),
        }
    plan["story_evidence"] = story_evidence
    story_writer_retry_count = 0
    story_output = _paper_story_writer(
        client,
        story_plan,
        clean_evidence,
        style_exemplar,
        settings.model_name,
        block_specs=block_specs,
    )
    try:
        _, draft = _paper_apply_story_candidate(
            plan,
            story_output,
            evidence_map,
            evidence_registry,
            block_specs,
            display_title,
            abstract_lead,
            valid_source_ids,
            figure_evidence_bundles if figure_first else None,
            "",
            "story writer",
            rollback_on_failure=False,
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
                "deterministic_validation": "A deterministic evidence check found an anchor placement issue. Preserve every supplied anchor exactly and keep it with its fixed block."
            },
            block_specs=block_specs,
        )
        _, draft = _paper_apply_story_candidate(
            plan,
            story_output,
            evidence_map,
            evidence_registry,
            block_specs,
            display_title,
            abstract_lead,
            valid_source_ids,
            figure_evidence_bundles if figure_first else None,
            "",
            "story writer retry",
            rollback_on_failure=False,
        )
    logger.info("PAPER deterministic evidence validation passed")
    plan["story_writer_retry_count"] = story_writer_retry_count

    markdown = draft
    article_editor_succeeded = False
    try:
        editor_output = _paper_article_editor(
            client,
            plan,
            markdown,
            abstract_lead,
            style_exemplar,
            settings.model_name,
            block_specs,
        )
        article_editor_succeeded, candidate_markdown = _paper_apply_article_editor_candidate(
            plan,
            editor_output,
            evidence_map,
            evidence_registry,
            block_specs,
            display_title,
            abstract_lead,
            valid_source_ids,
            figure_evidence_bundles if figure_first else None,
            markdown,
            "article editor",
        )
        if article_editor_succeeded:
            markdown = candidate_markdown
            logger.info("PAPER article-level editor validation passed")
    except Exception as exc:
        logger.warning("PAPER article-level editor unavailable; retaining Story Writer draft: %s", exc)
    lint = _paper_ai_style_lint(markdown, include_abstract=True)
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
    if not article_editor_succeeded and popular_audit_failed():
        popular_retry_count = 1
        story_output = _paper_story_writer(
            client,
            story_plan,
            clean_evidence,
            style_exemplar,
            settings.model_name,
            {"style_lint": lint, **popular_feedback},
            block_specs=block_specs,
        )
        accepted, candidate_markdown = _paper_apply_story_candidate(
            plan,
            story_output,
            evidence_map,
            evidence_registry,
            block_specs,
            display_title,
            abstract_lead,
            valid_source_ids,
            figure_evidence_bundles if figure_first else None,
            markdown,
            "popular style",
        )
        if accepted:
            markdown = candidate_markdown
            logger.info("PAPER deterministic evidence validation passed")
        lint = _paper_ai_style_lint(markdown, include_abstract=True)
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

    def safe_humanize(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        try:
            return _paper_humanize_story(*args, **kwargs)
        except Exception as exc:
            logger.warning("PAPER style rewrite generation failed; retaining prior draft: %s", exc)
            return []

    humanized_output = (
        safe_humanize(
            client,
            story_plan,
            clean_evidence,
            _paper_story_draft_blocks(plan["sections"]),
            settings.model_name,
            block_specs=block_specs,
            style_exemplar=style_exemplar,
        )
        if not article_editor_succeeded
        else []
    )
    accepted, candidate_markdown = _paper_apply_story_candidate(
        plan,
        humanized_output,
        evidence_map,
        evidence_registry,
        block_specs,
        display_title,
        abstract_lead,
        valid_source_ids,
        figure_evidence_bundles if figure_first else None,
        markdown,
        "humanizer",
    )
    if accepted:
        markdown = candidate_markdown
        logger.info("PAPER deterministic evidence validation passed")
    elif not article_editor_succeeded:
        retry_output = safe_humanize(
            client,
            story_plan,
            clean_evidence,
            _paper_story_draft_blocks(plan["sections"]),
            settings.model_name,
            {"deterministic_validation": "The candidate dropped or changed a required hard anchor. Preserve every anchor exactly."},
            block_specs=block_specs,
            style_exemplar=style_exemplar,
        )
        accepted, candidate_markdown = _paper_apply_story_candidate(
            plan,
            retry_output,
            evidence_map,
            evidence_registry,
            block_specs,
            display_title,
            abstract_lead,
            valid_source_ids,
            figure_evidence_bundles if figure_first else None,
            markdown,
            "humanizer retry",
        )
        if accepted:
            markdown = candidate_markdown
            logger.info("PAPER deterministic evidence validation passed")

    style_review: dict[str, Any] = {"issues": [], "status": "skipped"}
    review_issues: list[dict[str, Any]] = []
    if article_editor_succeeded:
        article_feedback = {
            "style_lint": lint,
            "editor_feedback": popular_feedback,
            "stop_slop": _paper_stop_slop_audit(markdown),
        }
        review_error = ""
        revision_error = ""
        revision_rollback = False
        article_revision_attempts = 0

        def review_article() -> dict[str, Any]:
            nonlocal review_error
            try:
                return _paper_style_review(
                    client,
                    style_exemplar,
                    plan["sections"],
                    markdown,
                    settings.model_name,
                    article_level=True,
                    feedback=article_feedback,
                )
            except Exception as exc:
                review_error = str(exc)
                logger.warning(
                    "PAPER article style reviewer returned unusable output; continuing without revision: %s",
                    exc,
                )
                return {"issues": []}

        def revise_article(issues: list[dict[str, Any]]) -> bool:
            nonlocal markdown, revision_error, revision_rollback, article_revision_attempts
            try:
                editor_revision = _paper_article_editor(
                    client,
                    plan,
                    markdown,
                    abstract_lead,
                    style_exemplar,
                    settings.model_name,
                    block_specs,
                    {"style_review": issues, **article_feedback},
                )
                accepted, candidate_markdown = _paper_apply_article_editor_candidate(
                    plan,
                    editor_revision,
                    evidence_map,
                    evidence_registry,
                    block_specs,
                    display_title,
                    abstract_lead,
                    valid_source_ids,
                    figure_evidence_bundles if figure_first else None,
                    markdown,
                    f"article style revision {article_revision_attempts + 1}",
                )
                if accepted:
                    markdown = candidate_markdown
                    article_revision_attempts += 1
                    logger.info("PAPER article-level style revision validation passed")
                    return True
                revision_rollback = True
            except Exception as exc:
                revision_error = str(exc)
                revision_rollback = True
                logger.warning(
                    "PAPER article style revision failed; retaining prior draft: %s",
                    exc,
                )
            return False

        style_review, review_issues, review_count, revision_count = _paper_bounded_article_review(
            review_article,
            revise_article,
            max_revisions=2,
        )
        style_review["status"] = "pass" if not review_issues else "needs_revision"
        if review_error or revision_error or revision_rollback:
            style_review["status"] = "warning"
        if review_error:
            style_review["error"] = review_error
        if revision_error:
            style_review["error"] = revision_error
        if revision_rollback:
            style_review["rollback"] = True
        if revision_count >= 2 and review_issues:
            style_review["revision_limit_reached"] = True
        # Article-level issues are handled only by the bounded whole-article
        # revision loop; never downgrade them to per-block humanizer tasks.
        review_issues = []
    else:
        try:
            style_review = {
                "status": "pass",
                **_paper_style_review(
                    client,
                    style_exemplar,
                    plan["sections"],
                    markdown,
                    settings.model_name,
                ),
            }
            review_issues = style_review.get("issues") or []
        except Exception as exc:
            logger.warning("PAPER style reviewer returned unusable output; continuing without rewrite: %s", exc)
            style_review = {"status": "warning", "issues": [], "error": str(exc)}
    if review_issues:
        review_targets = {issue["block_id"] for issue in review_issues}
        review_feedback = {
            issue["block_id"]: {
                "style_review": {
                    "issue": issue["issue"],
                    "instruction": issue["instruction"],
                }
            }
            for issue in review_issues
        }
        reviewed_output = safe_humanize(
            client,
            story_plan,
            clean_evidence,
            _paper_story_draft_blocks(plan["sections"]),
            settings.model_name,
            block_specs=block_specs,
            style_exemplar=style_exemplar,
            target_block_ids=review_targets,
            feedback_by_block=review_feedback,
        )
        accepted, candidate_markdown = _paper_apply_story_candidate(
            plan,
            reviewed_output,
            evidence_map,
            evidence_registry,
            block_specs,
            display_title,
            abstract_lead,
            valid_source_ids,
            figure_evidence_bundles if figure_first else None,
            markdown,
            "style reviewer",
        )
        if not accepted:
            reviewed_output = safe_humanize(
                client,
                story_plan,
                clean_evidence,
                _paper_story_draft_blocks(plan["sections"]),
                settings.model_name,
                {"style_review": review_feedback},
                block_specs=block_specs,
                style_exemplar=style_exemplar,
                target_block_ids=review_targets,
                feedback_by_block=review_feedback,
            )
            accepted, candidate_markdown = _paper_apply_story_candidate(
                plan,
                reviewed_output,
                evidence_map,
                evidence_registry,
                block_specs,
                display_title,
                abstract_lead,
                valid_source_ids,
                figure_evidence_bundles if figure_first else None,
                markdown,
                "style reviewer retry",
            )
        if accepted:
            markdown = candidate_markdown
            logger.info("PAPER deterministic evidence validation passed")
            style_review["rewritten_block_ids"] = sorted(review_targets)
        else:
            style_review["status"] = "warning"
            style_review["rollback"] = True
    plan["style_review"] = style_review

    final_humanizer_feedback = _paper_editor_feedback(abstract_lead, markdown)
    final_humanizer_feedback["anchor_preservation"] = _paper_editor_anchor_audit(
        humanizer_baseline,
        markdown,
        plan,
    )
    humanizer_retry_count = 0
    humanizer_targets = _paper_style_issue_block_ids(plan["sections"])
    if (
        not article_editor_succeeded
        and humanizer_targets
        and (
            _paper_ai_style_lint_failed(_paper_ai_style_lint(markdown))
            or final_humanizer_feedback["abstract_overlong"]
            or final_humanizer_feedback["body_lengths"]["total_overlong"]
            or final_humanizer_feedback["readability"]["issue_count"]
            or final_humanizer_feedback["title_style"]["issue_count"]
            or final_humanizer_feedback["anchor_preservation"]["issue_count"]
        )
    ):
        humanizer_retry_count = 1
        targeted_output = safe_humanize(
            client,
            story_plan,
            clean_evidence,
            _paper_story_draft_blocks(plan["sections"]),
            settings.model_name,
            {"readability": final_humanizer_feedback},
            block_specs=block_specs,
            style_exemplar=style_exemplar,
            target_block_ids=humanizer_targets,
        )
        accepted, candidate_markdown = _paper_apply_story_candidate(
            plan,
            targeted_output,
            evidence_map,
            evidence_registry,
            block_specs,
            display_title,
            abstract_lead,
            valid_source_ids,
            figure_evidence_bundles if figure_first else None,
            markdown,
            "targeted humanizer",
        )
        if not accepted:
            targeted_output = safe_humanize(
                client,
                story_plan,
                clean_evidence,
                _paper_story_draft_blocks(plan["sections"]),
                settings.model_name,
                {"deterministic_validation": final_humanizer_feedback},
                block_specs=block_specs,
                style_exemplar=style_exemplar,
                target_block_ids=humanizer_targets,
            )
            accepted, candidate_markdown = _paper_apply_story_candidate(
                plan,
                targeted_output,
                evidence_map,
                evidence_registry,
                block_specs,
                display_title,
                abstract_lead,
                valid_source_ids,
                figure_evidence_bundles if figure_first else None,
                markdown,
                "targeted humanizer retry",
            )
        if accepted:
            markdown = candidate_markdown
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
    stop_slop_targets = _paper_style_issue_block_ids(plan["sections"])
    if not article_editor_succeeded and stop_slop_feedback["issue_count"] and stop_slop_targets:
        stop_slop_retry_count = 1
        targeted_output = safe_humanize(
            client,
            story_plan,
            clean_evidence,
            _paper_story_draft_blocks(plan["sections"]),
            settings.model_name,
            {"stop_slop": stop_slop_feedback},
            block_specs=block_specs,
            style_exemplar=style_exemplar,
            target_block_ids=stop_slop_targets,
        )
        accepted, candidate_markdown = _paper_apply_story_candidate(
            plan,
            targeted_output,
            evidence_map,
            evidence_registry,
            block_specs,
            display_title,
            abstract_lead,
            valid_source_ids,
            figure_evidence_bundles if figure_first else None,
            markdown,
            "stop-slop",
        )
        if not accepted:
            targeted_output = safe_humanize(
                client,
                story_plan,
                clean_evidence,
                _paper_story_draft_blocks(plan["sections"]),
                settings.model_name,
                {"stop_slop": stop_slop_feedback, "deterministic_validation": "Preserve every hard anchor exactly."},
                block_specs=block_specs,
                style_exemplar=style_exemplar,
                target_block_ids=stop_slop_targets,
            )
            accepted, candidate_markdown = _paper_apply_story_candidate(
                plan,
                targeted_output,
                evidence_map,
                evidence_registry,
                block_specs,
                display_title,
                abstract_lead,
                valid_source_ids,
                figure_evidence_bundles if figure_first else None,
                markdown,
                "stop-slop retry",
            )
        if accepted:
            markdown = candidate_markdown
            logger.info("PAPER deterministic evidence validation passed")
        stop_slop_feedback = _paper_stop_slop_audit(markdown)
    plan["stop_slop_audit"] = {
        **stop_slop_feedback,
        "retry_count": stop_slop_retry_count,
    }
    if stop_slop_feedback["issue_count"]:
        logger.warning("PAPER stop-slop style audit unresolved after retry; continuing with warning")

    lint = _paper_ai_style_lint(markdown, include_abstract=True)
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
    markdown = _paper_remove_inline_citation_markers(markdown)
    final_lint = _paper_ai_style_lint(markdown, include_abstract=True)
    final_author_rewrite_rolled_back = False
    if _paper_ai_style_lint_failed(final_lint):
        logger.warning("PAPER final prose style lint remains; triggering local fallback rewrite")
        author_baseline_sections = copy.deepcopy(plan["sections"])
        author_baseline_markdown = markdown
        author_baseline_lint = dict(final_lint)
        author_targets = _paper_style_issue_block_ids(plan["sections"])
        targeted_output = safe_humanize(
            client,
            story_plan,
            clean_evidence,
            _paper_story_draft_blocks(plan["sections"]),
            settings.model_name,
            {"author_voice": final_lint},
            block_specs=block_specs,
            style_exemplar=style_exemplar,
            target_block_ids=author_targets,
        )
        accepted, candidate_markdown = _paper_apply_story_candidate(
            plan,
            targeted_output,
            evidence_map,
            evidence_registry,
            block_specs,
            display_title,
            abstract_lead,
            valid_source_ids,
            figure_evidence_bundles if figure_first else None,
            markdown,
            "author-voice",
        )
        if not accepted:
            targeted_output = safe_humanize(
                client,
                story_plan,
                clean_evidence,
                _paper_story_draft_blocks(plan["sections"]),
                settings.model_name,
                {"author_voice": final_lint, "deterministic_validation": "Preserve every hard anchor exactly."},
                block_specs=block_specs,
                style_exemplar=style_exemplar,
                target_block_ids=author_targets,
            )
            accepted, candidate_markdown = _paper_apply_story_candidate(
                plan,
                targeted_output,
                evidence_map,
                evidence_registry,
                block_specs,
                display_title,
                abstract_lead,
                valid_source_ids,
                figure_evidence_bundles if figure_first else None,
                markdown,
                "author-voice retry",
            )
        if accepted:
            candidate_markdown = _normalize_article_markdown(
                _remove_unverified_paper_quotes(candidate_markdown, paper_text),
                display_title,
            )
            candidate_lint = _paper_ai_style_lint(
                candidate_markdown,
                include_abstract=True,
            )
            if _paper_ai_style_lint_failed(candidate_lint):
                logger.warning(
                    "PAPER local fallback did not clear final style lint; rolling back"
                )
                plan["sections"] = author_baseline_sections
                markdown = author_baseline_markdown
                final_lint = author_baseline_lint
                final_author_rewrite_rolled_back = True
            else:
                markdown = candidate_markdown
                final_lint = candidate_lint
        else:
            final_author_rewrite_rolled_back = True
    plan["final_style_lint"] = final_lint
    _paper_require_clean_final_style_lint(final_lint)
    if not markdown:
        raise RuntimeError("PAPER staged pipeline returned empty article")
    plan["popular_science_audit"] = popular_science_audit
    _validate_paper_evidence_plan(
        plan,
        markdown,
        valid_source_ids,
        figure_evidence_bundles if figure_first else None,
        evidence_registry,
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
            {"caption": image.get("caption", "")}
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
