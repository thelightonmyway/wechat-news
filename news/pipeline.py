"""V1 news pipeline: RSS, deterministic ranking, optional one-shot LLM selection."""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import re
from datetime import date as date_type, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from db import Database, utc_now
from images.policy import apply_policy, format_license_label, is_no_derivatives_license
from images.search import search_public_images
from news.extract import download_publishable_images, extract_article
from news.feeds import fetch_all_feeds, normalize_title
from papers.doi import resolve_doi_landing_page
from papers.first_page import render_paper_first_page
from papers.oa_mirror import resolve_oa_html_mirror
from papers.openalex import OpenAlexAdapter
from papers.pdf_figures import discover_pdf_source, extract_pdf_figures
from settings import PROJECT_ROOT, Settings
from writer.llm import (
    article_output_dir,
    generate_article_markdown,
    generate_image_captions,
    generate_image_search_keywords,
    select_top_ten,
)

SOURCE_WEIGHTS = {
    "Nature News": 13,
    "Science.org Latest News": 13,
    "Guardian Science": 9,
    "NYT Science": 9,
    "ScienceDaily Top Science": 7,
}
PRIMARY_SOURCES = {
    "Guardian Science",
    "Nature News",
    "Science.org Latest News",
    "NYT Science",
    "ScienceDaily Top Science",
}
SECONDARY_SOURCES = {
    "Nature Atmospheric Science",
    "Nature Atmospheric Dynamics",
    "Nature Climate Sciences",
    "Nature Ocean Sciences",
    "Nature Physical Oceanography",
    "Nature Climate Change",
    "Nature Geoscience",
    "Eos / AGU",
    "Carbon Brief",
}
RESEARCH_TERMS = (
    "study",
    "research",
    "researchers",
    "scientists",
    "published",
    "journal",
    "experiment",
    "trial",
    "evidence",
    "discovery",
)
CORE_TOPIC_GROUPS = {
    "surface_wind": (
        "near-surface wind speed",
        "near surface wind speed",
        "surface wind",
        "terrestrial stilling",
        "wind recovery",
        "wind power density",
        "wind resource",
        "wind energy",
        "extreme wind",
    ),
    "atmospheric_circulation": (
        "atmospheric circulation",
        "jet stream",
        "polar vortex",
        "north atlantic oscillation",
        "nao",
        "southern annular mode",
        "sam",
        "monsoon",
        "subtropical high",
        "walker circulation",
        "hadley circulation",
    ),
    "boundary_land": (
        "planetary boundary layer",
        "boundary layer",
        "surface roughness",
        "land cover",
        "vegetation cover",
        "vegetation feedback",
        "forest cover",
        "land-atmosphere interaction",
        "land atmosphere interaction",
    ),
    "climate_attribution": (
        "climate change",
        "cmip5",
        "cmip6",
        "large ensemble",
        "climate model",
        "model evaluation",
        "detection and attribution",
        "anthropogenic forcing",
        "greenhouse gas forcing",
        "internal variability",
    ),
    "observations": (
        "era5",
        "reanalysis",
        "station observations",
        "satellite observations",
        "observational coverage",
        "climate observations",
    ),
    "temperature_extremes": (
        "temperature",
        "air temperature",
        "surface air temperature",
        "temperature extreme",
        "temperature extremes",
        "extreme temperature",
        "heatwave",
        "heatwaves",
        "heat wave",
        "heat waves",
        "extreme wind",
        "compound extremes",
        "compound climate extremes",
        "drought",
    ),
    "ocean_air_sea": (
        "southern ocean",
        "ocean circulation",
        "air-sea interaction",
        "air sea interaction",
        "ocean surface wind",
        "sea ice",
        "ocean-atmosphere coupling",
        "ocean atmosphere coupling",
        "wind work",
        "ocean energy input",
    ),
    "polar_ozone": (
        "antarctic climate",
        "antarctic warming",
        "antarctic sea ice",
        "antarctic ozone",
        "arctic climate",
        "arctic warming",
        "arctic sea ice",
        "polar climate",
        "ozone depletion",
        "ozone recovery",
        "southern hemisphere westerlies",
    ),
    "moisture_precipitation": (
        "precipitation",
        "heavy precipitation",
        "extreme precipitation",
        "precipitation extreme",
        "precipitation extremes",
        "atmospheric moisture",
        "water vapor",
        "water vapour",
        "humidity",
        "moisture transport",
        "moisture convergence",
        "soil moisture",
        "evapotranspiration",
        "vertical motion",
        "precipitation mechanism",
        "precipitation variability",
        "precipitation change",
        "asian monsoon",
    ),
    "cloud_radiation": (
        "cloud",
        "clouds",
        "cloud radiative effect",
        "cloud radiative effects",
        "radiation",
        "radiative forcing",
    ),
    "snow": (
        "snow",
        "snow cover",
    ),
    "storms": (
        "tropical cyclone",
        "tropical cyclones",
        "cyclone",
        "cyclones",
        "typhoon",
        "typhoons",
        "hurricane",
        "hurricanes",
        "storm",
        "storms",
        "storm dynamics",
        "fire weather",
    ),
}
TOPIC_TERMS = tuple(
    term for terms in CORE_TOPIC_GROUPS.values() for term in terms
)
WILDFIRE_TERMS = ("wildfire", "wildfires")
WILDFIRE_WEATHER_TERMS = (
    "temperature",
    "heat",
    "drought",
    "humidity",
    "wind",
    "precipitation",
    "fire weather",
    "climate variability",
    "climate change",
)
WILDFIRE_EXCLUDED_CONTEXTS = (
    "post-fire vegetation",
    "post fire vegetation",
    "vegetation recovery",
    "forest management",
    "fire management",
    "ecological succession",
    "social impact",
    "social impacts",
    "economic impact",
    "economic impacts",
)
BROAD_CONTEXT_TERMS = {
    "climate change",
    "cmip5",
    "cmip6",
    "large ensemble",
    "climate model",
    "model evaluation",
    "era5",
    "reanalysis",
    "station observations",
    "satellite observations",
    "observational coverage",
    "climate observations",
    "drought",
    "monsoon",
    "asian monsoon",
    "sea ice",
    "antarctic sea ice",
    "arctic sea ice",
    "southern ocean",
    "land cover",
    "vegetation cover",
    "vegetation feedback",
    "forest cover",
    "precipitation variability",
    "precipitation change",
}
EXCLUDED_TERMS = (
    "battery",
    "batteries",
    "electric vehicle",
    "electric vehicles",
    "ev",
    "solar",
    "renewable energy investment",
    "renewable energy investments",
    "renewable energy market",
    "renewable energy markets",
    "renewable energy policy",
    "renewable energy policies",
    "renewable energy project",
    "renewable energy projects",
    "clean energy investment",
    "clean energy investments",
    "artificial intelligence",
    "ai",
    "defense",
    "military",
    "politics",
    "political",
    "trump",
    "economics",
    "economic policy",
    "alzheimer",
    "dementia",
    "cancer",
    "tumor",
    "disease",
    "patient",
    "clinical trial",
    "medicine",
    "medical",
    "biomedical",
    "neuroscience",
    "neurolog",
    "brain cell",
    "gene therapy",
    "vaccine",
    "cognitive",
    "vitamin",
    "retina",
    "retinal",
    "eye tissue",
    "neuron",
    "animal physiology",
    "black hole",
    "galaxy",
    "exoplanet",
    "astronomy",
    "archaeology",
    "archaeological",
    "ancient burial",
    "wrf",
    "numerical simulation",
    "regional climate model",
    "parameterization",
)
POPULAR_CONTENT = "popular"
PAPER_CONTENT = "paper"
LOOKBACK_HOURS = (48, 168, 720)


def content_type_for_date(date_value: str) -> str:
    return PAPER_CONTENT if datetime.fromisoformat(date_value).weekday() == 2 else POPULAR_CONTENT


def _contains_term(text: str, term: str) -> bool:
    pattern = r"(?<![a-z0-9])" + re.escape(term).replace(r"\ ", r"[\s-]+") + r"(?![a-z0-9])"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _topic_text(item: dict[str, Any]) -> str:
    openalex = item.get("openalex") or {}
    return " ".join(
        [
            str(item.get("title") or ""),
            str(item.get("summary") or ""),
            str(openalex.get("title") or ""),
            str(openalex.get("abstract") or ""),
            " ".join(openalex.get("topics") or []),
            " ".join(openalex.get("keywords") or []),
        ]
    ).lower()


def matched_topic_groups(item: dict[str, Any]) -> set[str]:
    text = _topic_text(item)
    if any(_contains_term(text, term) for term in EXCLUDED_TERMS):
        return set()
    wildfire = any(_contains_term(text, term) for term in WILDFIRE_TERMS)
    if wildfire and any(context in text for context in WILDFIRE_EXCLUDED_CONTEXTS):
        return set()
    groups = {
        group
        for group, terms in CORE_TOPIC_GROUPS.items()
        if any(
            term not in BROAD_CONTEXT_TERMS and _contains_term(text, term)
            for term in terms
        )
    }
    if wildfire and any(_contains_term(text, term) for term in WILDFIRE_WEATHER_TERMS):
        groups.add("wildfire_weather")
    return groups


def is_relevant_topic(item: dict[str, Any]) -> bool:
    return bool(matched_topic_groups(item))


def prioritize_candidates(
    items: list[dict[str, Any]],
    limit: int = 10,
) -> list[dict[str, Any]]:
    def ranked(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(values, key=lambda item: float(item.get("score") or 0), reverse=True)

    primary = ranked(
        [item for item in items if str(item.get("source") or "") in PRIMARY_SOURCES]
    )
    secondary = ranked(
        [item for item in items if str(item.get("source") or "") in SECONDARY_SOURCES]
    )
    other = ranked(
        [
            item
            for item in items
            if str(item.get("source") or "") not in PRIMARY_SOURCES | SECONDARY_SOURCES
        ]
    )
    return (primary + secondary + other)[:limit]


def _broad_image_search_keywords(
    keywords: list[str],
    title: str = "",
) -> list[str]:
    title_text = title.lower()
    if "sleep" in title_text and ("hot" in title_text or "heat" in title_text):
        return ["nighttime heat"]
    science_terms = (
        "temperature",
        "heatwave",
        "heat wave",
        "climate",
        "weather",
        "atmosphere",
        "wind",
        "ocean",
        "precipitation",
        "drought",
        "sea ice",
        "ozone",
    )
    normalized_keywords = [
        keyword.lower().replace("temperatures", "temperature")
        for keyword in keywords
    ]
    if any(
        "nighttime" in keyword and "temperature" in keyword
        for keyword in normalized_keywords
    ):
        return ["nighttime temperature"]
    if any(
        "nighttime" in keyword and "heat" in keyword
        for keyword in normalized_keywords
    ):
        return ["nighttime heat"]
    for keyword in normalized_keywords:
        if any(term in keyword for term in science_terms):
            return [keyword]
    return keywords[:1]


def _image_visual_score(image: dict[str, Any], content_type: str) -> int:
    text = " ".join(
        str(image.get(key) or "")
        for key in ("metadata_title", "caption", "alt", "provider", "image_role")
    ).lower()
    score = 0
    if str(image.get("local_path") or "").lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        score += 4
    if content_type == PAPER_CONTENT:
        role = str(image.get("image_role") or "").lower()
        if role in {"hero", "graphical_abstract", "article_cover", "cover"}:
            score += 40
        for term in ("cover", "hero", "graphical abstract", "main figure", "figure 1"):
            if term in text:
                score += 8
    else:
        for term in ("photo", "city", "night", "weather", "heat wave", "landscape"):
            if term in text:
                score += 3
        for term in ("dataset", "data set", "chart", "graph", "complex diagram"):
            if term in text:
                score -= 4
    return score


def _image_information_tokens(image: dict[str, Any]) -> set[str]:
    text = " ".join(
        str(image.get(key) or "") for key in ("metadata_title", "caption", "alt")
    ).lower()
    stopwords = {
        "file",
        "global",
        "data",
        "set",
        "2013",
        "summer",
        "south",
        "north",
        "america",
        "europe",
        "asia",
        "africa",
        "oceania",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text)
        if len(token) >= 3 and token not in stopwords
    }


def _text_relevance_score(image: dict[str, Any], context: str) -> int:
    image_tokens = _image_information_tokens(image)
    context_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", context.lower())
        if len(token) >= 3
    }
    if not image_tokens or not context_tokens:
        return 0
    overlap = image_tokens & context_tokens
    return len(overlap) * 4 + round(10 * len(overlap) / len(image_tokens))


def _images_redundant(
    first: dict[str, Any],
    second: dict[str, Any],
) -> bool:
    if str(first.get("url") or "") == str(second.get("url") or ""):
        return True
    first_tokens = _image_information_tokens(first)
    second_tokens = _image_information_tokens(second)
    if not first_tokens or not second_tokens:
        return False
    overlap = len(first_tokens & second_tokens) / min(
        len(first_tokens),
        len(second_tokens),
    )
    return overlap >= 0.8


def _paper_html_access_failed(
    extracted: dict[str, Any],
    landing: dict[str, Any],
) -> bool:
    error = str(extracted.get("extraction_error") or "").lower()
    access_markers = (
        "401",
        "403",
        "forbidden",
        "unauthorized",
        "timeout",
        "timed out",
        "fetch failed",
    )
    return bool(
        not landing.get("accessible", True)
        or any(marker in error for marker in access_markers)
    )


def _apply_article_license_to_html_figures(
    images: list[dict[str, Any]],
    article_license: str,
) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    for image in images:
        record = dict(image)
        if record.get("image_source") == "html_figure":
            record["license"] = article_license or record.get("license") or ""
            record = apply_policy(record, allow_no_derivatives=True)
            if (
                record.get("publishable")
                and is_no_derivatives_license(str(record.get("license") or ""))
                and int(record.get("figure_image_count") or 1) != 1
            ):
                record["publishable"] = False
                record["cover_eligible"] = False
                record["reason"] = (
                    "ND figure contains multiple image regions; complete unmodified figure "
                    "cannot be guaranteed"
                )
        updated.append(record)
    return updated


def _has_publishable_html_image(images: list[dict[str, Any]]) -> bool:
    rejected_terms = ("logo", "icon", "advertisement", "tracking", "banner", "sprite")
    for image in images:
        if not image.get("publishable") or image.get("image_source") == "pdf_figure":
            continue
        descriptor = " ".join(
            str(image.get(key) or "")
            for key in ("url", "caption", "alt", "metadata_title", "image_role")
        ).lower()
        if not any(term in descriptor for term in rejected_terms):
            return True
    return False


def _select_article_images(
    images: list[dict[str, Any]],
    content_type: str,
    context: str = "",
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], int]:
    ranked = sorted(
        images,
        key=lambda image: (
            _image_visual_score(image, content_type) + _text_relevance_score(image, context)
        ),
        reverse=True,
    )
    cover_ranked = [
        image
        for image in ranked
        if not is_no_derivatives_license(str(image.get("license") or ""))
    ]
    cover: dict[str, Any] | None = cover_ranked[0] if cover_ranked else None
    if content_type == PAPER_CONTENT and cover_ranked:
        preferred_html = [
            image
            for image in cover_ranked
            if image.get("image_source") != "pdf_figure"
            and str(image.get("image_role") or "").lower()
            in {"hero", "graphical_abstract", "article_cover", "cover"}
        ]
        pdf_figures = sorted(
            [image for image in cover_ranked if image.get("image_source") == "pdf_figure"],
            key=lambda image: int(image.get("figure_number") or 0),
        )
        if preferred_html:
            cover = preferred_html[0]
        elif pdf_figures:
            if len(pdf_figures) > 4:
                first = pdf_figures[0]
                last = pdf_figures[-1]
                cover = (
                    last
                    if _text_relevance_score(last, context)
                    > _text_relevance_score(first, context)
                    else first
                )
            else:
                cover = pdf_figures[0]

    limit = 4 if content_type == PAPER_CONTENT else 2
    body: list[dict[str, Any]] = []
    redundant_count = 0
    for image in ranked:
        if any(_images_redundant(image, selected) for selected in body):
            redundant_count += 1
            continue
        body.append(image)
        if len(body) == limit:
            break
    return cover, body, redundant_count


def _paper_nd_attribution(image: dict[str, Any], dossier: dict[str, Any]) -> str:
    if dossier.get("content_type") != PAPER_CONTENT or not is_no_derivatives_license(
        str(image.get("license") or "")
    ):
        return ""

    credit = " ".join(str(image.get("credit") or "").split())
    openalex = dossier.get("openalex") or {}
    authors = openalex.get("authors") or dossier.get("authors") or []
    creator = credit
    if not creator and authors:
        creator = str(authors[0]).strip()
        if creator and len(authors) > 1:
            creator = f"{creator} et al."

    year = ""
    for value in (
        openalex.get("publication_year"),
        openalex.get("publication_date"),
        dossier.get("published_at"),
    ):
        match = re.search(r"\b(?:19|20)\d{2}\b", str(value or ""))
        if match:
            year = match.group(0)
            break
    if creator and year and year not in creator:
        creator = f"{creator} ({year})"

    license_label = format_license_label(
        str(image.get("license") or ""),
        str(image.get("license_url") or ""),
    )
    parts = [part for part in (creator, license_label) if part]
    return f"图源：{', '.join(parts)}" if parts else ""


def _paper_wechat_cover(paper_first_page: dict[str, Any]) -> dict[str, Any]:
    if not paper_first_page.get("wechat_cover_path") or is_no_derivatives_license(
        str(paper_first_page.get("license") or "")
    ):
        return {}
    return {
        "image_source": "paper_first_page",
        "local_path": paper_first_page.get("wechat_cover_path", ""),
        "source_pdf": paper_first_page.get("source_pdf", ""),
        "page": 1,
    }


def _prepare_paper_markdown(
    markdown_path: Path,
    paper_first_page: dict[str, Any] | None,
) -> None:
    markdown = markdown_path.read_text(encoding="utf-8")
    markdown = re.sub(r"\A\s*#\s+[^\n]+\n+", "", markdown, count=1)
    if paper_first_page and paper_first_page.get("local_path"):
        local_path = Path(str(paper_first_page["local_path"]))
        try:
            relative_path = local_path.relative_to(markdown_path.parent).as_posix()
        except ValueError:
            relative_path = local_path.as_posix()
        markdown = f"![论文第一页]({relative_path})\n\n{markdown.lstrip()}"
    markdown_path.write_text(markdown.rstrip() + "\n", encoding="utf-8")


def _is_published_article(
    item: dict[str, Any],
    identifiers: dict[str, set[Any]],
) -> bool:
    doi = str(item.get("doi") or "").strip().lower()
    canonical = str(item.get("canonical_url") or item.get("url") or "").strip()
    article_id = item.get("article_id") or item.get("id")
    return bool(
        (doi and doi in identifiers["dois"])
        or (canonical and canonical in identifiers["canonical_urls"])
        or (article_id and int(article_id) in identifiers["article_ids"])
    )


def reference_markdown(dossier: dict[str, Any]) -> str:
    openalex = dossier.get("openalex") or {}
    title = str(openalex.get("title") or dossier.get("title") or "")
    source = str(
        openalex.get("journal")
        or dossier.get("journal")
        or dossier.get("source")
        or ""
    )
    published_at = str(
        openalex.get("publication_date") or dossier.get("published_at") or ""
    )[:10]
    doi = str(openalex.get("doi") or dossier.get("doi") or "")
    url = str(dossier.get("url") or "")
    lines = [
        "## 文章信息",
        "",
        f"- 原文标题：{title}",
        f"- 来源/期刊：{source}",
    ]
    if published_at:
        lines.append(f"- 日期：{published_at}")
    if doi:
        lines.append(f"- DOI：{doi}")
    if url:
        lines.append(f"- 原文链接：{url}")
    return "\n".join(lines)


def local_date(settings: Settings, now: datetime | None = None) -> str:
    current = now or datetime.now(timezone.utc)
    return current.astimezone(ZoneInfo(settings.daily_timezone)).date().isoformat()


def _paper_publication_within_window(
    metadata: dict[str, Any],
    run_date: str,
    days: int = 30,
) -> bool:
    try:
        published = date_type.fromisoformat(str(metadata.get("publication_date") or ""))
        current = date_type.fromisoformat(run_date)
    except ValueError:
        return False
    return current - timedelta(days=days) <= published <= current


def _parse_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return datetime.now(timezone.utc)


def deterministic_score(item: dict[str, Any], now: datetime | None = None) -> float:
    current = now or datetime.now(timezone.utc)
    published = _parse_datetime(str(item.get("published_at") or ""))
    age_hours = max(0.0, (current - published.astimezone(timezone.utc)).total_seconds() / 3600)
    score = max(0.0, 168.0 - age_hours) * 0.2
    score += max(0.0, 48.0 - age_hours) * 1.0
    score += SOURCE_WEIGHTS.get(str(item.get("source") or ""), 5)

    text = f"{item.get('title', '')} {item.get('summary', '')}".lower()
    score += min(24, sum(4 for term in RESEARCH_TERMS if term in text))
    score += min(24, len(matched_topic_groups(item)) * 4)
    if item.get("doi"):
        score += 12

    word_count = int(item.get("word_count") or 0)
    if 400 <= word_count <= 1800:
        score += 12
    elif 1800 < word_count <= 2500:
        score += 5
    elif word_count > 2500:
        score -= min(12, (word_count - 2500) / 250)
    elif 0 < word_count < 400:
        score -= 4
    return round(score, 3)


def deduplicate(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(items, key=lambda item: item.get("published_at", ""), reverse=True)
    kept: list[dict[str, Any]] = []
    seen_doi: set[str] = set()
    seen_url: set[str] = set()
    seen_title: set[str] = set()

    for item in ordered:
        doi = str(item.get("doi") or "").lower()
        canonical = str(item.get("canonical_url") or "")
        normalized = str(item.get("normalized_title") or normalize_title(str(item.get("title") or "")))
        if doi and doi in seen_doi:
            continue
        if canonical and canonical in seen_url:
            continue
        if normalized and normalized in seen_title:
            continue
        if normalized and any(
            difflib.SequenceMatcher(None, normalized, str(other.get("normalized_title") or "")).ratio() >= 0.88
            for other in kept
        ):
            continue
        item["normalized_title"] = normalized
        kept.append(item)
        if doi:
            seen_doi.add(doi)
        if canonical:
            seen_url.add(canonical)
        if normalized:
            seen_title.add(normalized)
    return kept


class NewsPipeline:
    def __init__(self, settings: Settings, logger: logging.Logger | None = None) -> None:
        self.settings = settings
        self.logger = logger or logging.getLogger("wechat_news.pipeline")
        self.db = Database(settings.database_path)
        self.openalex = OpenAlexAdapter(settings.openalex_api_key)
        self.feed_config = PROJECT_ROOT / "config" / "feeds.yaml"
        self.refresh_lock = asyncio.Lock()
        self.last_source_counts: dict[str, int] = {}

    async def _extract_shortlist(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        preliminary = sorted(items, key=deterministic_score, reverse=True)[:40]
        semaphore = asyncio.Semaphore(5)

        async def one(item: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                return await asyncio.to_thread(extract_article, item)

        extracted = await asyncio.gather(*(one(item) for item in preliminary), return_exceptions=True)
        by_url: dict[str, dict[str, Any]] = {
            str(item.get("canonical_url") or item.get("url")): item for item in items
        }
        for original, result in zip(preliminary, extracted):
            key = str(original.get("canonical_url") or original.get("url"))
            if isinstance(result, Exception):
                failed = dict(original)
                failed["extraction_error"] = f"{type(result).__name__}: {result}"
                by_url[key] = failed
            else:
                by_url[key] = result
        return list(by_url.values())

    async def _published_papers(
        self,
        items: list[dict[str, Any]],
        run_date: str,
    ) -> list[dict[str, Any]]:
        if not self.openalex.configured:
            self.logger.warning("Wednesday paper selection skipped: OPENALEX_API_KEY not configured")
            return []
        semaphore = asyncio.Semaphore(5)

        async def verify(item: dict[str, Any]) -> dict[str, Any] | None:
            if not item.get("doi") or not is_relevant_topic(item):
                return None
            async with semaphore:
                try:
                    metadata = await asyncio.to_thread(self.openalex.lookup_doi, item.get("doi"))
                except Exception as exc:
                    self.logger.warning("OpenAlex DOI lookup failed: %s", exc)
                    return None
            merged = dict(item)
            merged["openalex"] = metadata
            run_day = date_type.fromisoformat(run_date)
            if not self.openalex.is_formally_published(
                metadata,
                today=run_day,
            ) or not _paper_publication_within_window(metadata, run_date):
                return None
            merged["source_published_at"] = merged.get("published_at") or ""
            merged["paper_publication_date"] = metadata.get("publication_date") or ""
            merged["title"] = metadata.get("title") or merged.get("title") or ""
            merged["normalized_title"] = normalize_title(str(merged["title"]))
            merged["summary"] = metadata.get("abstract") or merged.get("summary") or ""
            merged["journal"] = metadata.get("journal") or ""
            merged["source"] = metadata.get("journal") or merged.get("source") or "OpenAlex"
            if metadata.get("publication_date"):
                merged["published_at"] = f"{metadata['publication_date']}T00:00:00+00:00"
            if not is_relevant_topic(merged):
                return None
            merged["status"] = "published_paper"
            return merged

        verified = await asyncio.gather(*(verify(item) for item in items))
        return [item for item in verified if item is not None]

    async def refresh(
        self,
        date: str | None = None,
        content_type: str | None = None,
    ) -> list[dict[str, Any]]:
        run_date = date or local_date(self.settings)
        run_type = content_type or content_type_for_date(run_date)
        if run_type not in {POPULAR_CONTENT, PAPER_CONTENT}:
            raise ValueError(f"unsupported content type: {run_type}")
        async with self.refresh_lock:
            self.db.set_daily_run(
                run_date,
                content_type=run_type,
                status="running",
                error="",
            )
            try:
                published = self.db.published_article_identifiers()
                feed_errors: list[str] = []
                source_counts: dict[str, int] = {}
                topic_items: list[dict[str, Any]] = []
                enriched: list[dict[str, Any]] = []
                lookback_hours = LOOKBACK_HOURS[0]
                for window_hours in LOOKBACK_HOURS:
                    items, window_errors, window_counts = await asyncio.to_thread(
                        fetch_all_feeds,
                        self.feed_config,
                        window_hours,
                    )
                    feed_errors = list(dict.fromkeys([*feed_errors, *window_errors]))
                    source_counts = window_counts
                    lookback_hours = window_hours
                    topic_items = [
                        item
                        for item in items
                        if is_relevant_topic(item)
                        and not _is_published_article(item, published)
                    ]
                    normalized = deduplicate(topic_items)
                    enriched = await self._extract_shortlist(normalized)
                    enriched = [
                        item for item in deduplicate(enriched) if is_relevant_topic(item)
                    ]
                    if run_type == PAPER_CONTENT:
                        enriched = await self._published_papers(enriched, run_date)
                        enriched = deduplicate(enriched)
                    if len(enriched) >= 10:
                        break
                self.last_source_counts = source_counts

                eligible: list[dict[str, Any]] = []
                for item in enriched:
                    if _is_published_article(item, published):
                        continue
                    article_id = self.db.upsert_article(item)
                    item["article_id"] = article_id
                    if _is_published_article(item, published):
                        continue
                    if item.get("images") is not None:
                        self.db.replace_images(article_id, item.get("images") or [])
                    item["score"] = deterministic_score(item)
                    eligible.append(item)
                enriched = eligible

                prioritized = prioritize_candidates(enriched)
                selected, used_model, llm_error = await asyncio.to_thread(
                    select_top_ten,
                    prioritized,
                    self.settings,
                )
                selected = prioritize_candidates(selected)
                self.db.replace_candidates(run_date, selected, run_type)
                errors = list(feed_errors)
                if self.settings.model_configured and llm_error:
                    errors.append(f"LLM selection fallback: {llm_error}")
                status = "success" if selected else "empty"
                if feed_errors and selected:
                    status = "partial"
                self.db.set_daily_run(
                    run_date,
                    fetched_at=utc_now(),
                    candidate_count=len(selected),
                    content_type=run_type,
                    status=status,
                    error=" | ".join(errors)[:2000],
                )
                self.logger.info(
                    "News refresh complete date=%s type=%s lookback=%sh feeds=%s topic=%s unique=%s candidates=%s llm=%s",
                    run_date,
                    run_type,
                    lookback_hours,
                    source_counts,
                    len(topic_items),
                    len(enriched),
                    len(selected),
                    used_model,
                )
                return self.db.get_candidates(run_date, run_type)
            except Exception as exc:
                self.db.set_daily_run(
                    run_date,
                    fetched_at=utc_now(),
                    content_type=run_type,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}"[:2000],
                )
                raise

    async def get_or_refresh(
        self,
        date: str | None = None,
        content_type: str | None = None,
    ) -> list[dict[str, Any]]:
        run_date = date or local_date(self.settings)
        run_type = content_type or content_type_for_date(run_date)
        existing = self.db.get_candidates(run_date, run_type)
        run = self.db.get_daily_run(run_date, run_type)
        if existing and run and run.get("content_type") == run_type:
            published = self.db.published_article_identifiers()
            if not any(_is_published_article(item, published) for item in existing):
                if self.settings.model_configured and any(
                    not str(item.get("title_cn") or "").strip() for item in existing
                ):
                    title_candidates = [
                        dict(item, article_id=int(item["id"])) for item in existing
                    ]
                    titled, used_model, title_error = await asyncio.to_thread(
                        select_top_ten,
                        title_candidates,
                        self.settings,
                    )
                    if used_model:
                        self.db.replace_candidates(run_date, titled, run_type)
                        return self.db.get_candidates(run_date, run_type)
                    self.logger.warning(
                        "Chinese title batch fallback date=%s error=%s",
                        run_date,
                        title_error,
                    )
                return existing
        return await self.refresh(run_date, run_type)

    def format_news(self, candidates: list[dict[str, Any]]) -> str:
        if not candidates:
            return "今日暂无可用科研新闻候选。"
        content_type = str(candidates[0].get("content_type") or POPULAR_CONTENT)
        heading = "今日已发表论文 Top 10" if content_type == PAPER_CONTENT else "今日科普新闻 Top 10"
        lines = [f"## {heading}"]
        for item in candidates[:10]:
            title_cn = str(item.get("title_cn") or "").strip()
            english = str(item.get("title") or "")
            published = str(item.get("published_at") or "")[:16].replace("T", " ")
            lines.append(f"\n**{item['rank']}. {title_cn or english}**")
            if title_cn and title_cn != english:
                lines.append(english)
            lines.append(f"{item.get('source', '')} · {published}")
        return "\n".join(lines)

    async def paper_details(
        self,
        rank: int,
        date: str | None = None,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        run_date = date or local_date(self.settings)
        run_type = content_type or content_type_for_date(run_date)
        item = self.db.get_candidate(run_date, rank, run_type)
        if not item:
            command = "/papers" if run_type == PAPER_CONTENT else "/news"
            raise LookupError(f"今日没有序号 {rank}；请先执行 {command}")

        extracted = await asyncio.to_thread(extract_article, item)
        article_id = self.db.upsert_article(extracted)
        extracted["id"] = article_id
        extracted["rank"] = rank
        extracted["date"] = run_date
        extracted["content_type"] = item.get("content_type") or POPULAR_CONTENT

        try:
            openalex = await asyncio.to_thread(
                self.openalex.lookup_doi,
                extracted.get("doi"),
            )
        except Exception as exc:
            self.logger.warning("OpenAlex metadata lookup skipped: %s", exc)
            openalex = {
                "configured": self.openalex.configured,
                "found": False,
                "doi": str(extracted.get("doi") or ""),
                "error": f"{type(exc).__name__}: {exc}"[:1000],
            }
        extracted["openalex"] = openalex
        if openalex.get("found"):
            extracted["authors"] = openalex.get("authors") or extracted.get("authors") or []
            extracted["journal"] = openalex.get("journal") or extracted.get("journal") or ""
            if not extracted.get("summary"):
                extracted["summary"] = openalex.get("abstract") or ""

        if extracted["content_type"] == PAPER_CONTENT and openalex.get("found"):
            news_text = extracted.get("text") or ""
            paper_url = openalex.get("oa_url") or f"https://doi.org/{extracted.get('doi', '')}"
            try:
                doi_landing = await asyncio.to_thread(
                    resolve_doi_landing_page,
                    str(extracted.get("doi") or ""),
                    str(paper_url),
                )
            except Exception as exc:
                doi_landing = {
                    "resolved": False,
                    "landing_url": str(paper_url),
                    "accessible": False,
                    "error": f"{type(exc).__name__}: {exc}"[:1000],
                }
            resolved_url = str(doi_landing.get("landing_url") or paper_url)
            paper_item = {
                **extracted,
                "url": resolved_url,
                "canonical_url": resolved_url,
                "title": openalex.get("title") or extracted.get("title") or "",
                "summary": openalex.get("abstract") or extracted.get("summary") or "",
                "published_at": openalex.get("publication_date") or extracted.get("published_at") or "",
                "journal": openalex.get("journal") or "",
                "source": openalex.get("journal") or extracted.get("source") or "OpenAlex",
                "images": [],
            }
            publisher_extracted = await asyncio.to_thread(extract_article, paper_item)
            publisher_extracted["images"] = _apply_article_license_to_html_figures(
                list(publisher_extracted.get("images") or []),
                str(openalex.get("license") or ""),
            )
            paper_extracted = publisher_extracted
            oa_mirror: dict[str, Any] = {
                "attempted": False,
                "found": False,
                "url": "",
                "error": "",
            }
            actual_image_source = (
                "publisher_html"
                if any(
                    image.get("image_source") == "html_figure"
                    for image in publisher_extracted.get("images") or []
                )
                else "none"
            )
            if _paper_html_access_failed(publisher_extracted, doi_landing):
                oa_mirror["attempted"] = True
                try:
                    oa_mirror.update(
                        await asyncio.to_thread(
                            resolve_oa_html_mirror,
                            openalex,
                            resolved_url,
                        )
                    )
                except Exception as exc:
                    oa_mirror["error"] = f"{type(exc).__name__}: {exc}"[:1000]
                if oa_mirror.get("found") and oa_mirror.get("url"):
                    mirror_url = str(oa_mirror["url"])
                    mirror_item = {
                        **paper_item,
                        "url": mirror_url,
                        "canonical_url": mirror_url,
                        "images": [],
                    }
                    mirror_extracted = await asyncio.to_thread(extract_article, mirror_item)
                    mirror_extracted["images"] = _apply_article_license_to_html_figures(
                        list(mirror_extracted.get("images") or []),
                        str(openalex.get("license") or ""),
                    )
                    mirror_figures = [
                        image
                        for image in mirror_extracted.get("images") or []
                        if image.get("image_source") == "html_figure"
                    ]
                    if not _paper_html_access_failed(
                        mirror_extracted,
                        {"accessible": True},
                    ) and (mirror_extracted.get("text") or mirror_figures):
                        paper_extracted = mirror_extracted
                        actual_image_source = "oa_mirror"

            paper_extracted["url"] = resolved_url
            paper_extracted["canonical_url"] = resolved_url
            paper_extracted["id"] = article_id
            paper_extracted["rank"] = rank
            paper_extracted["date"] = run_date
            paper_extracted["content_type"] = PAPER_CONTENT
            paper_extracted["news_text"] = news_text
            paper_extracted["openalex"] = openalex
            paper_extracted["doi_landing"] = doi_landing
            paper_extracted["publisher_url"] = resolved_url
            paper_extracted["oa_mirror"] = oa_mirror
            paper_extracted["oa_mirror_url"] = str(oa_mirror.get("url") or "")
            paper_extracted["actual_image_source"] = actual_image_source
            paper_extracted["publisher_extraction_error"] = str(
                publisher_extracted.get("extraction_error") or ""
            )
            paper_extracted["authors"] = openalex.get("authors") or paper_extracted.get("authors") or []
            paper_extracted["journal"] = openalex.get("journal") or ""
            paper_extracted["title"] = openalex.get("title") or paper_extracted.get("title") or ""
            paper_extracted["summary"] = openalex.get("abstract") or paper_extracted.get("summary") or ""
            extracted = paper_extracted

        image_records = list(extracted.get("images") or [])
        self.db.replace_images(article_id, image_records)
        extracted["images"] = (
            image_records
            if extracted.get("content_type") == PAPER_CONTENT
            else self.db.get_images(article_id)
        )
        return extracted

    def format_paper(self, dossier: dict[str, Any]) -> str:
        openalex = dossier.get("openalex") or {}
        images = dossier.get("images") or []
        publishable = sum(1 for image in images if image.get("publishable"))
        authors = openalex.get("authors") or dossier.get("authors") or []
        summary = str(openalex.get("abstract") or dossier.get("summary") or "")
        if len(summary) > 900:
            summary = summary[:897] + "..."
        return "\n".join(
            [
                f"## {dossier.get('title', '')}",
                f"来源：{dossier.get('source', '')}",
                f"发布时间：{str(dossier.get('published_at', ''))[:16].replace('T', ' ')}",
                f"URL：{dossier.get('url', '')}",
                f"word count：{dossier.get('word_count', 0)}",
                f"summary：{summary or '无'}",
                f"DOI：{dossier.get('doi') or '未发现'}",
                f"Journal：{openalex.get('journal') or dossier.get('journal') or '未知'}",
                f"作者：{', '.join(authors) if authors else '未知'}",
                f"OA status：{openalex.get('oa_status', '未查询')}",
                f"License：{openalex.get('license', '未知')}",
                f"图片：{len(images)}；可自动使用：{publishable}",
            ]
        )

    async def generate(
        self,
        rank: int,
        date: str | None = None,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        dossier = await self.paper_details(rank, date, content_type)
        output_dir = article_output_dir(
            dossier["date"],
            rank,
            str(dossier.get("content_type") or ""),
        )
        if dossier.get("content_type") == PAPER_CONTENT:
            dossier["pdf_figure_fallback"] = {
                "attempted": False,
                "reason": "legal HTML image available",
            }
            if not _has_publishable_html_image(list(dossier.get("images") or [])):
                dossier["pdf_figure_fallback"] = {
                    "attempted": True,
                    "reason": "no legal HTML hero, graphical abstract, cover, or figure",
                }
                try:
                    openalex = dossier.get("openalex") or {}
                    source = await asyncio.to_thread(
                        discover_pdf_source,
                        str(dossier.get("url") or ""),
                        str(dossier.get("doi") or ""),
                        str(openalex.get("license") or ""),
                    )
                    dossier["pdf_figure_source"] = source
                    if source.get("pdf_url"):
                        pdf_figures, pdf_metadata = await asyncio.to_thread(
                            extract_pdf_figures,
                            str(source["pdf_url"]),
                            output_dir / "images",
                            article_url=str(source.get("landing_url") or dossier.get("url") or ""),
                            article_license=str(source.get("license") or ""),
                            license_url=str(source.get("license_url") or ""),
                        )
                        dossier["images"] = list(dossier.get("images") or []) + pdf_figures
                        dossier["pdf_figure_metadata"] = pdf_metadata
                        dossier["pdf_figure_fallback"]["matched_figures"] = len(pdf_figures)
                        if pdf_figures:
                            dossier["actual_image_source"] = "pdf_figure"
                    else:
                        dossier["pdf_figure_fallback"]["error"] = "no formal/reference PDF found"
                except Exception as exc:
                    dossier["pdf_figure_fallback"]["error"] = (
                        f"{type(exc).__name__}: {exc}"[:1000]
                    )
                    self.logger.warning("PDF Figure fallback failed: %s", exc)
        if dossier.get("content_type") == POPULAR_CONTENT:
            try:
                keywords = await asyncio.to_thread(
                    generate_image_search_keywords,
                    dossier,
                    self.settings,
                )
                self.logger.info("Image search keywords: %s", keywords)
                public_images = await asyncio.to_thread(search_public_images, keywords, 5)
                self.logger.info(
                    "Public image candidates: %s (primary search)",
                    len(public_images),
                )
                if not public_images and keywords:
                    broad_keywords = _broad_image_search_keywords(
                        keywords,
                        str(dossier.get("title") or ""),
                    )
                    public_images = await asyncio.to_thread(
                        search_public_images,
                        broad_keywords,
                        5,
                    )
                    self.logger.info(
                        "Public image candidates: %s (broad search keywords=%s)",
                        len(public_images),
                        broad_keywords,
                    )
                dossier["image_search_keywords"] = keywords
                dossier["images"] = list(dossier.get("images") or []) + public_images
            except Exception as exc:
                self.logger.warning("Public image metadata search failed: %s", exc)
                dossier["image_search_keywords"] = []
        markdown_path, metadata_path = await asyncio.to_thread(
            generate_article_markdown,
            dossier,
            self.settings,
            output_dir,
        )
        if dossier.get("content_type") == PAPER_CONTENT:
            paper_first_page: dict[str, Any] = {}
            source_pdf = output_dir / "source_reference.pdf"
            if source_pdf.is_file():
                try:
                    paper_first_page = await asyncio.to_thread(
                        render_paper_first_page,
                        source_pdf,
                        output_dir / "images",
                    )
                    openalex = dossier.get("openalex") or {}
                    paper_first_page.update(
                        {
                            "doi": dossier.get("doi", ""),
                            "journal": openalex.get("journal") or dossier.get("journal") or "",
                            "license": (dossier.get("pdf_figure_source") or {}).get("license")
                            or openalex.get("license")
                            or "",
                            "license_url": (dossier.get("pdf_figure_source") or {}).get(
                                "license_url", ""
                            ),
                        }
                    )
                except Exception as exc:
                    self.logger.warning("Paper first-page rendering failed: %s", exc)
                    dossier["paper_first_page_error"] = f"{type(exc).__name__}: {exc}"[:1000]
            dossier["paper_first_page"] = paper_first_page
            dossier["wechat_cover"] = _paper_wechat_cover(paper_first_page)
            await asyncio.to_thread(
                _prepare_paper_markdown,
                markdown_path,
                paper_first_page,
            )
        source_images = list(dossier.get("images") or [])
        local_flags = [
            bool(image.get("local_path") and Path(str(image["local_path"])).is_file())
            for image in source_images
        ]
        remote_images = [
            image for image, is_local in zip(source_images, local_flags) if not is_local
        ]
        downloaded_remote = await asyncio.to_thread(
            download_publishable_images,
            remote_images,
            str(output_dir / "images"),
        )
        remote_iterator = iter(downloaded_remote)
        downloaded_images = [
            dict(image) if is_local else next(remote_iterator)
            for image, is_local in zip(source_images, local_flags)
        ]
        legal_images = [
            image
            for image in downloaded_images
            if image.get("publishable") and image.get("local_path")
        ]
        selection_context = (
            " ".join(
                (
                    str(dossier.get("title") or ""),
                    str(dossier.get("summary") or ""),
                    markdown_path.read_text(encoding="utf-8")[:20000],
                )
            )
            if dossier.get("content_type") == PAPER_CONTENT
            else ""
        )
        cover_image, body_images, redundant_count = _select_article_images(
            legal_images,
            str(dossier.get("content_type") or POPULAR_CONTENT),
            selection_context,
        )
        body_image_captions = await asyncio.to_thread(
            generate_image_captions,
            body_images,
            self.settings,
        )
        dossier["cover_image"] = cover_image or {}
        dossier["body_images"] = body_images
        dossier["body_image_captions"] = body_image_captions
        dossier["redundant_images_removed"] = redundant_count
        self.logger.info(
            "Publishable downloaded images: %s (downloaded records=%s body=%s redundant=%s)",
            len(legal_images),
            len(downloaded_images),
            len(body_images),
            redundant_count,
        )
        terminal_sections: list[str] = []
        if body_images:
            image_section: list[str] = []
            for index, image in enumerate(body_images, start=1):
                local_name = Path(str(image["local_path"])).name
                image_section.append(f"![图 {index}](images/{local_name})")
                caption = body_image_captions[index - 1]
                if caption:
                    image_section.append(f"*图{index}. {caption}*")
                attribution = _paper_nd_attribution(image, dossier)
                if attribution:
                    image_section.append(f"*{attribution}*")
                image_section.append("")
            terminal_sections.append("\n".join(image_section).rstrip())
        references = reference_markdown(dossier)
        if references:
            terminal_sections.append(references)
        if terminal_sections:
            with markdown_path.open("a", encoding="utf-8") as handle:
                handle.write("\n\n---\n\n" + "\n\n".join(terminal_sections) + "\n")
        dossier["images"] = downloaded_images
        self.db.replace_images(int(dossier["id"]), downloaded_images)
        metadata_path.write_text(
            json.dumps(
                {
                    "model": self.settings.model_name,
                    "source": dossier.get("url", ""),
                    "doi": dossier.get("doi", ""),
                    "content_type": dossier.get("content_type", POPULAR_CONTENT),
                    "title_cn": dossier.get("title_cn", ""),
                    "journal": (dossier.get("openalex") or {}).get("journal")
                    or dossier.get("journal", ""),
                    "doi_landing": dossier.get("doi_landing", {}),
                    "publisher_url": dossier.get("publisher_url", dossier.get("url", "")),
                    "oa_mirror_url": dossier.get("oa_mirror_url", ""),
                    "oa_mirror": dossier.get("oa_mirror", {}),
                    "actual_image_source": dossier.get("actual_image_source", "none"),
                    "image_search_keywords": dossier.get("image_search_keywords", []),
                    "paper_first_page": dossier.get("paper_first_page", {}),
                    "paper_first_page_error": dossier.get("paper_first_page_error", ""),
                    "wechat_cover": dossier.get("wechat_cover", {}),
                    "cover_image": dossier.get("cover_image", {}),
                    "body_images": dossier.get("body_images", []),
                    "body_image_captions": dossier.get("body_image_captions", []),
                    "redundant_images_removed": dossier.get("redundant_images_removed", 0),
                    "pdf_figure_fallback": dossier.get("pdf_figure_fallback", {}),
                    "pdf_figure_source": dossier.get("pdf_figure_source", {}),
                    "pdf_figure_metadata": dossier.get("pdf_figure_metadata", {}),
                    "images": downloaded_images,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self.db.save_generated_post(
            int(dossier["id"]),
            str(markdown_path),
            self.settings.model_name,
            "generated",
        )
        return {
            "dossier": dossier,
            "markdown_path": markdown_path,
            "metadata_path": metadata_path,
        }
