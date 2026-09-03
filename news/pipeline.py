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
from papers.openalex import (
    OpenAlexAdapter,
    is_allowed_paper_journal,
    journal_display_name,
)
from papers.pdf_figures import (
    discover_pdf_source,
    download_pdf_with_wiley_tdm,
    extract_pdf_figures,
)
from settings import PROJECT_ROOT, Settings
from writer.llm import (
    article_output_dir,
    generate_article_markdown,
    generate_image_captions,
    generate_image_search_keywords,
    select_paper_top_ten,
    select_top_ten,
    translate_paper_titles,
)

SOURCE_WEIGHTS = {
    "Nature News": 13,
    "Science.org Latest News": 13,
    "NASA Earth Observatory": 11,
    "Copernicus Climate": 10,
    "NOAA NOS News": 10,
    "NOAA NOS Newsroom": 10,
    "Guardian Science": 9,
    "Guardian Climate Crisis": 9,
    "NYT Science": 9,
    "Inside Climate News": 8,
    "ScienceDaily Top Science": 7,
}
PRIMARY_SOURCES = {
    "Guardian Science",
    "Guardian Climate Crisis",
    "Nature News",
    "Science.org Latest News",
    "NYT Science",
    "ScienceDaily Top Science",
}
SECONDARY_SOURCES = {
    "NASA Earth Observatory",
    "NOAA NOS News",
    "NOAA NOS Newsroom",
    "Copernicus Climate",
    "Inside Climate News",
    "Eos / AGU",
    "Carbon Brief",
}
PAPER_ONLY_SOURCES = {
    "Nature Atmospheric Science",
    "Nature Atmospheric Dynamics",
    "Nature Climate Sciences",
    "Nature Ocean Sciences",
    "Nature Physical Oceanography",
    "Nature Climate Change",
    "Nature Geoscience",
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


def source_allowed_for_content(source: str, content_type: str) -> bool:
    return content_type != POPULAR_CONTENT or source not in PAPER_ONLY_SOURCES


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


NEWS_RSS_PREFILTER_TERMS = (
    "climate",
    "weather",
    "temperature",
    "warming",
    "heat",
    "precipitation",
    "rainfall",
    "snow",
    "drought",
    "wind",
    "storm",
    "storms",
    "ocean",
    "sea",
    "arctic",
    "antarctic",
    "ice",
    "atmosphere",
    "atmospheric",
    "cloud",
    "clouds",
    "monsoon",
    "enso",
    "wildfire",
    "wildfires",
    "humidity",
    "moisture",
    "water vapor",
    "water vapour",
)
NEWS_RSS_PREFILTER_EXCLUSIONS = (
    "alzheimer",
    "dementia",
    "cancer",
    "tumor",
    "patient",
    "clinical trial",
    "medicine",
    "medical",
    "gene therapy",
    "vaccine",
    "artificial intelligence",
    "battery",
    "electric vehicle",
    "black hole",
    "galaxy",
    "exoplanet",
    "archaeology",
    "smartphone",
    "gaming",
    "celebrity",
)


def is_relevant_news_rss_prefilter(item: dict[str, Any]) -> bool:
    text = f"{item.get('title', '')} {item.get('summary', '')}".lower()
    if any(_contains_term(text, term) for term in NEWS_RSS_PREFILTER_EXCLUSIONS):
        return False
    return is_relevant_topic(item) or any(
        _contains_term(text, term) for term in NEWS_RSS_PREFILTER_TERMS
    )


def is_relevant_news_after_extraction(item: dict[str, Any]) -> bool:
    probe = dict(item)
    probe["summary"] = " ".join(
        (
            str(item.get("summary") or ""),
            str(item.get("text") or "")[:50000],
        )
    )
    return is_relevant_topic(probe)


PAPER_DISALLOWED_TYPES = {
    "reply",
    "correction",
    "editorial",
    "comment",
    "correspondence",
}
PAPER_DISALLOWED_TITLE_PREFIXES = (
    "reply to ",
    "correction to ",
    "author correction",
    "publisher correction",
    "editorial:",
    "comment on ",
    "correspondence:",
)
PAPER_RELEVANCE_GROUPS = {
    "wind": (
        "surface wind",
        "near-surface wind",
        "near surface wind",
        "wind energy",
        "wind power",
        "terrestrial stilling",
    ),
    "circulation_teleconnection": (
        "atmospheric circulation",
        "jet stream",
        "teleconnection",
        "teleconnections",
        "el niño",
        "el nino",
        "enso",
        "north atlantic oscillation",
        "nao",
        "southern annular mode",
        "sam",
        "walker circulation",
        "hadley circulation",
        "polar vortex",
    ),
    "stratosphere_ozone": (
        "stratosphere-troposphere",
        "stratosphere troposphere",
        "stratospheric ozone",
        "ozone depletion",
        "ozone recovery",
        "ozone-climate",
        "ozone climate",
    ),
    "temperature": (
        "surface air temperature",
        "air temperature",
        "temperature extreme",
        "temperature extremes",
        "heatwave",
        "heatwaves",
        "heat wave",
        "heat waves",
    ),
    "precipitation_moisture": (
        "precipitation",
        "water vapor",
        "water vapour",
        "atmospheric moisture",
        "moisture transport",
        "moisture convergence",
    ),
    "climate_extremes": (
        "climate extreme",
        "climate extremes",
        "compound extreme",
        "compound extremes",
    ),
    "boundary_land": (
        "boundary layer",
        "land-atmosphere interaction",
        "land atmosphere interaction",
        "surface roughness",
    ),
    "air_sea": (
        "air-sea interaction",
        "air sea interaction",
        "ocean-atmosphere interaction",
        "ocean atmosphere interaction",
        "ocean-atmosphere coupling",
        "ocean atmosphere coupling",
        "ocean circulation",
        "wind work",
    ),
    "polar_climate": (
        "polar climate",
        "arctic climate",
        "antarctic climate",
        "sea ice dynamics",
        "sea-ice dynamics",
        "sea ice variability",
    ),
    "variability_predictability": (
        "climate variability",
        "climate predictability",
        "predictability",
        "internal variability",
    ),
    "attribution": (
        "detection and attribution",
        "climate attribution",
        "anthropogenic forcing",
        "greenhouse gas forcing",
    ),
}
PAPER_CONDITIONAL_GROUPS = {
    "land_surface_hydrology": (
        "drought",
        "soil moisture",
        "evapotranspiration",
        "catchment",
        "hydrology",
    ),
    "model_evaluation": (
        "climate model",
        "model evaluation",
        "model benchmarking",
        "cmip5",
        "cmip6",
    ),
    "observations": (
        "reanalysis",
        "era5",
        "station observations",
        "satellite observations",
    ),
    "sea_ice": ("sea ice", "sea-ice"),
}
PAPER_DOMAIN_EXCLUSIONS = (
    "ground displacement",
    "geodesy",
    "geodetic",
    "ecological succession",
    "vegetation biology",
    "species richness",
    "biodiversity",
    "biogeochemistry",
    "ocean chemistry",
    "software benchmark",
    "software benchmarking",
)
PAPER_MECHANISM_MARKERS = (
    "mechanism",
    "dynamics",
    "feedback",
    "interaction",
    "coupling",
    "circulation",
    "variability",
    "predictability",
    "attribution",
    "forcing",
)
PAPER_STORM_TERMS = (
    "tropical cyclone",
    "tropical cyclones",
    "hurricane",
    "hurricanes",
    "typhoon",
    "typhoons",
    "storm",
    "storms",
    "extreme weather",
)
PAPER_STORM_CLIMATE_SCALE_TERMS = (
    "climate variability",
    "climate change",
    "interannual variability",
    "decadal variability",
    "long-term trend",
    "long term trend",
    "climate trend",
    "attribution",
    "projection",
    "projections",
    "predictability",
    "subseasonal",
    "sub-seasonal",
    "seasonal prediction",
    "seasonal predictability",
    "enso",
    "el niño",
    "el nino",
    "la niña",
    "la nina",
    "monsoon",
    "teleconnection",
    "teleconnections",
    "large-scale circulation",
    "large scale circulation",
    "atmospheric circulation",
    "walker circulation",
    "hadley circulation",
    "climatology",
)


def paper_relevance_score(item: dict[str, Any]) -> int:
    title = str(item.get("title") or "").strip().lower()
    text = _topic_text(item)
    work_type = str(
        item.get("work_type")
        or item.get("type")
        or (item.get("openalex") or {}).get("work_type")
        or ""
    ).lower()
    if work_type in PAPER_DISALLOWED_TYPES or any(
        title.startswith(prefix) for prefix in PAPER_DISALLOWED_TITLE_PREFIXES
    ):
        return 0

    storm_scale = any(_contains_term(text, term) for term in PAPER_STORM_TERMS)
    storm_climate_scale = any(
        _contains_term(text, term) for term in PAPER_STORM_CLIMATE_SCALE_TERMS
    )
    if storm_scale and not storm_climate_scale:
        return 0

    physical_groups = {
        group
        for group, terms in PAPER_RELEVANCE_GROUPS.items()
        if any(_contains_term(text, term) for term in terms)
    }
    if storm_scale and storm_climate_scale:
        physical_groups.add("storm_climate")
    conditional_groups = {
        group
        for group, terms in PAPER_CONDITIONAL_GROUPS.items()
        if any(_contains_term(text, term) for term in terms)
    }
    if not physical_groups:
        return 0
    if any(marker in text for marker in PAPER_DOMAIN_EXCLUSIONS) and not any(
        marker in text for marker in PAPER_MECHANISM_MARKERS
    ):
        return 0

    if len(physical_groups) >= 2:
        return 3
    if physical_groups & {
        "circulation_teleconnection",
        "stratosphere_ozone",
        "variability_predictability",
        "attribution",
    }:
        return 3
    if conditional_groups and not any(marker in text for marker in PAPER_MECHANISM_MARKERS):
        return 1
    return 2


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
        str(image.get(key) or "")
        for key in (
            "metadata_title",
            "caption",
            "original_caption",
            "figure_title",
            "description",
            "alt",
        )
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


_PAPER_GENERIC_FIGURE_TOKENS = {
    "analysis",
    "climate",
    "comparison",
    "distribution",
    "figure",
    "model",
    "models",
    "pattern",
    "patterns",
    "result",
    "results",
    "response",
    "simulation",
    "study",
    "variability",
}


def _paper_figure_match_score(image: dict[str, Any], context: str) -> int:
    """Score a figure only when its caption has specific textual evidence."""
    score = _text_relevance_score(image, context)
    if not score:
        return 0
    image_tokens = _image_information_tokens(image)
    context_tokens = {
        token for token in re.findall(r"[a-z0-9]+", context.lower()) if len(token) >= 3
    }
    meaningful_overlap = (image_tokens & context_tokens) - _PAPER_GENERIC_FIGURE_TOKENS
    image_text = " ".join(
        str(image.get(key) or "")
        for key in ("metadata_title", "caption", "original_caption", "figure_title", "description", "alt")
    )
    identifier_tokens = {
        token.lower()
        for token in re.findall(
            r"(?<![A-Za-z0-9])(?:[A-Z]{2,}[A-Za-z0-9-]*|[A-Za-z]{2,}\d[A-Za-z0-9-]*)(?![A-Za-z0-9])",
            image_text,
        )
    }
    context_identifiers = {
        token.lower()
        for token in re.findall(
            r"(?<![A-Za-z0-9])(?:[A-Z]{2,}[A-Za-z0-9-]*|[A-Za-z]{2,}\d[A-Za-z0-9-]*)(?![A-Za-z0-9])",
            context,
        )
    }
    identifier_overlap = identifier_tokens & context_identifiers
    if not meaningful_overlap and not identifier_overlap:
        return 0
    # One specific identifier (for example NAO, GRIP, or GeoB25206) can
    # establish a match even when the surrounding text is Chinese.
    minimum_overlap = 1 if len(image_tokens) <= 4 else 2
    if len(meaningful_overlap) < minimum_overlap and not identifier_overlap:
        return 0
    return score


_PAPER_SCIENTIFIC_CONCEPTS: dict[str, tuple[str, ...]] = {
    "nao": ("nao", "north atlantic oscillation", "大西洋涛动"),
    "niobium": ("niobium", "niobium content", "nb", "铌"),
    "hydroclimate": ("hydroclimate", "precipitation", "river discharge", "水文", "降水", "径流"),
    "temperature": ("temperature", "warming", "cooling", "δ18o", "温度", "变暖", "变冷"),
    "sediment": ("sediment", "sedimentary", "bioturbation", "ird", "density", "沉积", "岩芯", "冰筏", "密度"),
    "deglaciation": ("deglaciation", "deglaciated", "ice-free", "glacier", "glacial", "退冰", "冰退", "冰后退缩", "冰川", "无冰"),
    "mass_balance": ("surface mass balance", "smb", "mass balance", "accumulation", "ablation", "质量平衡", "积雪", "消融", "冰量"),
    "ice_core": ("ice core", "grip", "dye-3", "δ18o", "冰芯", "同位素"),
    "holocene": ("holocene", "early holocene", "late holocene", "全新世", "早全新世", "晚全新世"),
    "geology": ("geology", "geomorphology", "intrusion", "landscape", "地质", "地貌", "景观"),
    "model": ("simulation", "simulations", "model", "modeled", "模式", "模拟"),
    "anomaly": ("anomaly", "anomalies", "异常", "湿冷", "更湿", "更冷", "更暖", "更干"),
    "proxy": ("proxy", "marker", "代用", "指标"),
    "reconstruction": ("reconstruction", "reconstructed", "重建", "对比"),
    "late_holocene_event": ("little ice age", "medieval warm period", "lia", "小冰期", "中世纪暖期"),
}

_PAPER_CONCEPT_WEIGHTS = {
    "nao": 10,
    "niobium": 10,
    "hydroclimate": 8,
    "temperature": 6,
    "sediment": 5,
    "deglaciation": 20,
    "mass_balance": 8,
    "ice_core": 6,
    "holocene": 2,
    "geology": 2,
    "model": 2,
    "anomaly": 8,
    "proxy": 8,
    "reconstruction": 6,
    "late_holocene_event": 8,
}


def _paper_scientific_concepts(text: str) -> set[str]:
    lowered = text.lower()
    return {
        concept
        for concept, aliases in _PAPER_SCIENTIFIC_CONCEPTS.items()
        if any(alias.lower() in lowered for alias in aliases)
    }


def _paper_heading_match_score(image: dict[str, Any], heading: str) -> int:
    image_text = " ".join(
        str(image.get(key) or "")
        for key in ("metadata_title", "caption", "original_caption", "figure_title", "description", "alt")
    )
    shared = _paper_scientific_concepts(image_text) & _paper_scientific_concepts(heading)
    return sum(_PAPER_CONCEPT_WEIGHTS[concept] for concept in shared)


def _paper_scientific_match_score(image: dict[str, Any], context: str) -> int:
    image_text = " ".join(
        str(image.get(key) or "")
        for key in ("metadata_title", "caption", "original_caption", "figure_title", "description", "alt")
    )
    image_concepts = _paper_scientific_concepts(image_text)
    context_concepts = _paper_scientific_concepts(context)
    shared = image_concepts & context_concepts
    score = sum(_PAPER_CONCEPT_WEIGHTS[concept] for concept in shared)
    heading = context.splitlines()[0] if context.splitlines() else ""
    heading_shared = image_concepts & _paper_scientific_concepts(heading)
    # A concept stated in the section heading is more diagnostic than a
    # generic concept repeated in the body.
    score += 12 * len(heading_shared)
    return score


def _paper_source_paragraphs(source_text: str) -> list[str]:
    return [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in re.split(r"\n+", source_text)
        if paragraph.strip()
    ]


def _paper_source_paragraph_records(source_text: str) -> list[dict[str, Any]]:
    return [
        {
            "id": f"source-{index}",
            "index": index,
            "text": paragraph,
            "figure_references": sorted(_paper_figure_reference_numbers(paragraph)),
        }
        for index, paragraph in enumerate(_paper_source_paragraphs(source_text))
    ]


def _paper_source_heading(text: str) -> bool:
    stripped = text.strip()
    return bool(
        stripped
        and len(stripped) <= 120
        and not re.search(r"[.!?。！？:]$", stripped)
        and len(re.findall(r"[A-Za-z0-9]+", stripped)) <= 16
    )


def _paper_match_source_paragraphs(
    section_context: str,
    source_text: str,
    section_index: int,
    section_count: int,
) -> list[dict[str, Any]]:
    """Return a small, section-specific source window instead of full-text evidence."""
    records = _paper_source_paragraph_records(source_text)
    if not records:
        return []
    section_concepts = _paper_scientific_concepts(section_context)
    context_lines = section_context.splitlines()
    heading = context_lines[0] if context_lines else ""
    heading_concepts = _paper_scientific_concepts(heading)
    focus_text = next(
        (line for line in context_lines[1:] if line.strip()),
        "",
    )
    focus_concepts = _paper_scientific_concepts(focus_text)
    groups: list[dict[str, Any]] = []
    heading_indexes = [
        index for index, record in enumerate(records) if _paper_source_heading(record["text"])
    ]
    methods_indexes = [
        index
        for index in heading_indexes
        if records[index]["text"].strip().lower() == "methods"
    ]
    if methods_indexes:
        heading_indexes = [index for index in heading_indexes if index < methods_indexes[0]]
    for group_number, start in enumerate(heading_indexes):
        end = heading_indexes[group_number + 1] if group_number + 1 < len(heading_indexes) else len(records)
        members = [
            record
            for record in records[start + 1 : end]
            if not _paper_source_heading(record["text"])
        ]
        if not members:
            continue
        source_heading_concepts = _paper_scientific_concepts(records[start]["text"])
        heading_score = sum(
            _PAPER_CONCEPT_WEIGHTS[concept]
            for concept in source_heading_concepts & section_concepts
        )
        heading_match_score = sum(
            _PAPER_CONCEPT_WEIGHTS[concept]
            for concept in source_heading_concepts & heading_concepts
        )
        focus_heading_score = sum(
            _PAPER_CONCEPT_WEIGHTS[concept]
            for concept in source_heading_concepts & focus_concepts
        )
        paragraph_scores = []
        for member in members:
            shared = _paper_scientific_concepts(member["text"]) & section_concepts
            score = sum(_PAPER_CONCEPT_WEIGHTS[concept] for concept in shared)
            paragraph_scores.append((score, member))
        best_body_score = max((score for score, _ in paragraph_scores), default=0)
        if heading_score or best_body_score:
            groups.append(
                {
                    "heading": records[start],
                    "members": members,
                    "heading_score": heading_score,
                    "heading_match_score": heading_match_score,
                    "focus_heading_score": focus_heading_score,
                    "best_body_score": best_body_score,
                    "start": start,
                }
            )
    if not groups:
        scored_records = []
        for record in records:
            shared = _paper_scientific_concepts(record["text"]) & section_concepts
            score = sum(_PAPER_CONCEPT_WEIGHTS[concept] for concept in shared)
            if score:
                scored_records.append((score, record))
        return [
            record
            for _, record in sorted(
                scored_records,
                key=lambda item: (item[0], -item[1]["index"]),
                reverse=True,
            )[:3]
        ]
    groups.sort(
        key=lambda group: (
            int(group["heading_match_score"] > 0),
            group["heading_match_score"] * 100
            + group["focus_heading_score"] * 20
            + group["heading_score"] * 2
            + group["best_body_score"],
            -abs(group["start"] - round((section_index + 1) * len(records) / max(section_count + 1, 1))),
        ),
        reverse=True,
    )
    selected_groups = [groups[0]]
    selected_members: dict[int, dict[str, Any]] = {}
    for selected_group in selected_groups:
        members = selected_group["members"]
        scored_members = []
        for member in members:
            shared = _paper_scientific_concepts(member["text"]) & section_concepts
            score = sum(_PAPER_CONCEPT_WEIGHTS[concept] for concept in shared)
            score += 12 * len(_paper_scientific_concepts(member["text"]) & heading_concepts)
            if score:
                scored_members.append((score, member))
        for _, member in sorted(
            scored_members,
            key=lambda item: (item[0], -item[1]["index"]),
            reverse=True,
        )[:3]:
            selected_members[member["index"]] = member
            member_index = member["index"]
            for neighbor in members:
                if (
                    abs(neighbor["index"] - member_index) == 1
                    and not _paper_source_heading(neighbor["text"])
                ):
                    selected_members[neighbor["index"]] = neighbor
    return [selected_members[index] for index in sorted(selected_members)]


def _paper_source_evidence(
    image: dict[str, Any],
    section_context: str,
    source_paragraphs: list[dict[str, Any]],
) -> tuple[int, str, list[dict[str, Any]]]:
    figure_number = _numbered_paper_figure(image)
    if figure_number is None or not source_paragraphs:
        return 0, "", []
    section_concepts = _paper_scientific_concepts(section_context)
    if not section_concepts:
        return 0, "", []
    heading = section_context.splitlines()[0] if section_context.splitlines() else ""
    heading_concepts = _paper_scientific_concepts(heading)
    image_concepts = _paper_scientific_concepts(
        " ".join(
            str(image.get(key) or "")
            for key in ("metadata_title", "caption", "original_caption", "figure_title", "description", "alt")
        )
    )
    section_image_shared = section_concepts & image_concepts
    if not section_image_shared:
        return 0, "", []
    best_score = 0
    best_reason = ""
    best_evidence: list[dict[str, Any]] = []
    for record in source_paragraphs:
        paragraph = str(record.get("text") or "")
        paragraph_concepts = _paper_scientific_concepts(paragraph)
        paragraph_image_shared = paragraph_concepts & image_concepts
        shared = section_image_shared & paragraph_concepts
        references = _paper_figure_reference_numbers(paragraph)
        referenced = figure_number in references
        # A Figure citation inside the already matched source window is strong
        # evidence even when the paragraph names the mechanism rather than the
        # exact caption vocabulary. Unreferenced evidence still needs direct
        # section/Figure concept overlap to avoid cross-Figure contamination.
        if referenced:
            section_paragraph_shared = section_concepts & paragraph_concepts
            if not section_paragraph_shared:
                continue
            score = sum(
                _PAPER_CONCEPT_WEIGHTS[concept]
                for concept in section_paragraph_shared
            )
            score += sum(
                _PAPER_CONCEPT_WEIGHTS[concept]
                for concept in paragraph_image_shared
            ) // 2
            score += 30 * len(section_image_shared & heading_concepts)
            score += 90
        else:
            if not shared or not paragraph_image_shared:
                continue
            score = sum(_PAPER_CONCEPT_WEIGHTS[concept] for concept in shared)
            score += sum(_PAPER_CONCEPT_WEIGHTS[concept] for concept in paragraph_image_shared) // 2
            score += 30 * len(section_image_shared & heading_concepts)
            if not references:
                score += 8
            else:
                continue
        if score > best_score:
            best_score = score
            best_reason = (
                "source paragraph explicitly cites this Figure and shares section concepts"
                if referenced
                else "source paragraph and Figure content share section concepts"
            )
            best_evidence = [
                {
                    "id": record.get("id", ""),
                    "text": paragraph[:600],
                    "figure_references": record.get("figure_references", []),
                }
            ]
    return best_score, best_reason, best_evidence


def _images_redundant(
    first: dict[str, Any],
    second: dict[str, Any],
) -> bool:
    first_url = str(first.get("url") or "")
    second_url = str(second.get("url") or "")
    if first_url and first_url == second_url:
        return True
    if (
        str(first.get("image_role") or "").lower() == "figure"
        and str(second.get("image_role") or "").lower() == "figure"
    ):
        first_number = first.get("figure_number")
        second_number = second.get("figure_number")
        if first_number is not None and second_number is not None:
            return str(first_number) == str(second_number)
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
        local_path = str(image.get("local_path") or "")
        if not local_path or not Path(local_path).is_file():
            continue
        descriptor = " ".join(
            str(image.get(key) or "")
            for key in ("url", "caption", "alt", "metadata_title", "image_role")
        ).lower()
        if not any(term in descriptor for term in rejected_terms):
            return True
    return False


def _numbered_paper_figure(image: dict[str, Any]) -> int | None:
    if str(image.get("image_role") or "").lower() != "figure":
        return None
    try:
        number = int(image.get("figure_number"))
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _paper_body_image_order(images: list[dict[str, Any]]) -> list[int]:
    order = list(range(len(images)))
    numbered_positions = [
        position
        for position, image in enumerate(images)
        if _numbered_paper_figure(image) is not None
    ]
    numbered_sources = sorted(
        numbered_positions,
        key=lambda position: _numbered_paper_figure(images[position]) or 0,
    )
    for target, source in zip(numbered_positions, numbered_sources):
        order[target] = source
    return order


def _paper_figure_reference_numbers(context: str) -> set[int]:
    return {
        int(match.group("number"))
        for match in re.finditer(
            r"\b(?:(?P<supplementary>supplementary)\s+)?"
            r"fig(?:ure)?\.?\s*(?P<number>\d+)\s*[a-z]?\b",
            context,
            re.IGNORECASE,
        )
        if not match.group("supplementary")
    }


def _paper_image_label(image: dict[str, Any]) -> str:
    figure_number = _numbered_paper_figure(image)
    if figure_number is not None:
        return f"Fig. {figure_number}"
    return str(image.get("metadata_title") or image.get("url") or "image").strip()


def _paper_candidate_sort_key(candidate: dict[str, Any]) -> tuple[int, int, int, int, int, int, int, int, int]:
    return (
        int(candidate["explicit_reference"]),
        int(candidate.get("heading_score", 0)),
        int(candidate.get("scientific_score", 0)),
        int(candidate.get("source_score", 0)),
        int(candidate["semantic_score"] > 0),
        int(candidate["semantic_score"]),
        int(candidate["keyword_score"]),
        min(int(candidate.get("structural_score", 0)), 8),
        -int(candidate.get("figure_number") or 0),
    )


def _paper_allocate_images(
    images: list[dict[str, Any]],
    context: str,
    limit: int = 4,
    source_context: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Map every legal PAPER image before applying the global image limit."""
    lines = context.splitlines()
    section_slots = _paper_section_slots(lines) if context.strip() else []
    numbered_images = [
        image for image in images if _numbered_paper_figure(image) is not None
    ]
    if not numbered_images:
        selected_images = images[:limit]
        return selected_images, {
            "max_images": limit,
            "input_image_count": len(images),
            "numbered_figure_count": 0,
            "sections": [],
            "selected_figures": [
                {
                    "figure": _paper_image_label(image),
                    "reason": "没有编号 Figure，保留相关性排序靠前的图片",
                }
                for image in selected_images
            ],
            "discarded_figures": [
                {
                    "figure": _paper_image_label(image),
                    "reason": "超过全局最多4张图片限制",
                    "candidate_sections": [],
                }
                for image in images[limit:]
            ],
        }
    figure_numbers = sorted(
        {_numbered_paper_figure(image) for image in numbered_images if _numbered_paper_figure(image) is not None}
    )
    candidates_by_section: dict[int, list[dict[str, Any]]] = {}
    all_candidates: list[dict[str, Any]] = []
    all_scores_by_section: dict[int, list[dict[str, Any]]] = {}
    matched_source_paragraphs_by_section: dict[int, list[dict[str, Any]]] = {}
    for section_index, slot in enumerate(section_slots):
        references = _paper_figure_reference_numbers(slot[3])
        matched_source_paragraphs = _paper_match_source_paragraphs(
            slot[3],
            source_context,
            section_index,
            len(section_slots),
        )
        matched_source_paragraphs_by_section[section_index] = matched_source_paragraphs
        for image_index, image in enumerate(images):
            figure_number = _numbered_paper_figure(image)
            explicit = bool(figure_number is not None and figure_number in references)
            semantic_score = (
                _paper_figure_match_score(image, slot[3])
                if figure_number is not None
                else 0
            )
            scientific_score = _paper_scientific_match_score(image, slot[3])
            heading_score = _paper_heading_match_score(
                image,
                slot[3].splitlines()[0] if slot[3].splitlines() else "",
            )
            source_score, source_reason, source_evidence = _paper_source_evidence(
                image,
                slot[3],
                matched_source_paragraphs,
            )
            keyword_score = _text_relevance_score(image, slot[3])
            structural_score = 0
            if figure_number in figure_numbers and len(section_slots) > 1:
                figure_rank = figure_numbers.index(figure_number)
                expected_section = round(
                    figure_rank * (len(section_slots) - 1) / max(len(figure_numbers) - 1, 1)
                )
                # Figure order is only a weak tie-breaker; it is never a
                # relationship on its own.
                structural_score = max(0, 8 - abs(section_index - expected_section) * 2)
            if figure_number is not None:
                image_tokens = _image_information_tokens(image)
                context_tokens = {
                    token
                    for token in re.findall(r"[a-z0-9]+", slot[3].lower())
                    if len(token) >= 3
                }
                meaningful_keyword_overlap = (
                    image_tokens & context_tokens
                ) - _PAPER_GENERIC_FIGURE_TOKENS
                related = (
                    explicit
                    or source_score > 0
                    or scientific_score > 0
                    or semantic_score > 0
                    or (keyword_score >= 10 and bool(meaningful_keyword_overlap))
                )
            else:
                related = keyword_score > 0
            method = (
                "explicit_reference"
                if explicit
                else "source_paragraph"
                if source_score > 0
                else "semantic"
                if scientific_score > 0 or semantic_score > 0
                else "keyword"
                if related
                else "fallback"
            )
            score = max(
                source_score,
                scientific_score,
                semantic_score,
                keyword_score,
            )
            score_record = {
                "image_index": image_index,
                "section_index": section_index,
                "figure_number": figure_number,
                "label": _paper_image_label(image),
                "explicit_reference": explicit,
                "source_score": source_score,
                "source_reason": source_reason,
                "source_evidence": source_evidence,
                "heading_score": heading_score,
                "scientific_score": scientific_score,
                "semantic_score": semantic_score,
                "keyword_score": keyword_score,
                "structural_score": structural_score,
                "score": score,
                "match_method": method,
            }
            all_scores_by_section.setdefault(section_index, []).append(score_record)
            if not related:
                continue
            candidates_by_section.setdefault(section_index, []).append(score_record)
            all_candidates.append(score_record)

    unique_candidates: dict[int, dict[str, Any]] = {}
    for candidate in all_candidates:
        current = unique_candidates.get(candidate["image_index"])
        if current is None or _paper_candidate_sort_key(candidate) > _paper_candidate_sort_key(current):
            unique_candidates[candidate["image_index"]] = candidate
    section_records = []
    for section_index, slot in enumerate(section_slots):
        candidates = candidates_by_section.get(section_index, [])
        section_records.append(
            {
                "section_index": section_index,
                "section": (
                    re.sub(r"^#{2,3}\s+", "", slot[2].splitlines()[0]).strip()
                    or f"section_{section_index + 1}"
                ),
                "source_paragraphs": [
                    {
                        "id": paragraph.get("id", ""),
                        "text": paragraph.get("text", "")[:600],
                        "figure_references": paragraph.get("figure_references", []),
                    }
                    for paragraph in matched_source_paragraphs_by_section.get(section_index, [])
                ],
                "candidates": sorted(
                    candidates,
                    key=_paper_candidate_sort_key,
                    reverse=True,
                ),
                "figure_scores": sorted(
                    all_scores_by_section.get(section_index, []),
                    key=lambda candidate: (
                        -int(candidate.get("score", 0)),
                        -int(candidate.get("source_score", 0)),
                        -int(candidate.get("scientific_score", 0)),
                        int(candidate.get("figure_number") or 0),
                    ),
                ),
            }
        )
    selected_indices: set[int] = set()
    selected_reasons: dict[int, str] = {}
    selected_assignment_by_image: dict[int, int] = {}
    if section_slots and unique_candidates:
        numbered_image_indices = sorted(
            {
                candidate["image_index"]
                for candidate in unique_candidates.values()
                if candidate.get("figure_number") is not None
            },
            key=lambda image_index: _numbered_paper_figure(images[image_index]) or 0,
        )
        candidates_by_image: dict[int, list[dict[str, Any]]] = {}
        for candidate in all_candidates:
            if candidate.get("figure_number") is not None:
                candidates_by_image.setdefault(candidate["image_index"], []).append(candidate)

        # Select a monotonic path through Figure-number order. This preserves
        # section coverage without allowing a late Figure to force an earlier
        # section assignment that the insertion stage cannot honor.
        states: dict[tuple[int, int, int], tuple[int, int, list[dict[str, Any]]]] = {
            (0, -1, 0): (0, 0, [])
        }
        for image_index in numbered_image_indices:
            next_states = dict(states)
            for (count, last_section, covered_mask), (current_count, current_quality, path) in states.items():
                if current_count >= limit:
                    continue
                for candidate in candidates_by_image.get(image_index, []):
                    section_index = int(candidate["section_index"])
                    if section_index < last_section:
                        continue
                    new_count = current_count + 1
                    new_mask = covered_mask | (1 << section_index)
                    quality = (
                        int(candidate.get("explicit_reference", False)) * 100000
                        + int(candidate.get("heading_score", 0)) * 1000
                        + int(candidate.get("scientific_score", 0)) * 100
                        + int(candidate.get("source_score", 0))
                        + int(candidate.get("semantic_score", 0)) * 2
                        + int(candidate.get("keyword_score", 0))
                    )
                    new_state = (new_count, section_index, new_mask)
                    new_value = (
                        new_count,
                        current_quality + quality,
                        [*path, candidate],
                    )
                    old_value = next_states.get(new_state)
                    if old_value is None or (
                        new_value[0],
                        new_value[1],
                        len(new_value[2]),
                    ) > (
                        old_value[0],
                        old_value[1],
                        len(old_value[2]),
                    ):
                        next_states[new_state] = new_value
            states = next_states
        best_path = max(
            states.values(),
            key=lambda value: (
                len({candidate["section_index"] for candidate in value[2]}),
                value[0],
                value[1],
            ),
        )[2]
        for candidate in best_path:
            image_index = candidate["image_index"]
            selected_indices.add(image_index)
            selected_assignment_by_image[image_index] = int(candidate["section_index"])
            selected_reasons[image_index] = (
                f"匹配 {section_records[int(candidate['section_index'])]['section']}，"
                "按科学证据和 section 覆盖选择"
            )

        non_numbered_candidates = sorted(
            (
                candidate
                for candidate in unique_candidates.values()
                if candidate.get("figure_number") is None
            ),
            key=_paper_candidate_sort_key,
            reverse=True,
        )
        for candidate in non_numbered_candidates:
            if len(selected_indices) >= limit:
                break
            image_index = candidate["image_index"]
            if image_index in selected_indices:
                continue
            selected_indices.add(image_index)
            selected_reasons[image_index] = "覆盖 section 后的最高优先级剩余候选"

    if not section_slots or not unique_candidates:
        selected_indices = set(range(min(limit, len(images))))
        selected_reasons = {
            index: "正文没有可用 mapping，使用视觉/相关性排序兜底"
            for index in selected_indices
        }

    selected_images = [images[index] for index in sorted(selected_indices)]
    selected_images = [
        selected_images[index] for index in _paper_body_image_order(selected_images)
    ]
    selected_image_indices = {id(image): index for index, image in enumerate(images)}
    selected_by_index = {
        selected_image_indices[id(image)] for image in selected_images
    }
    for record in section_records:
        selected_candidates = [
            candidate
            for candidate in record["candidates"]
            if candidate["image_index"] in selected_by_index
            and selected_assignment_by_image.get(candidate["image_index"])
            == record["section_index"]
        ]
        record["selected_figures"] = [
            candidate["label"]
            for candidate in sorted(
                selected_candidates,
                key=lambda candidate: candidate.get("figure_number") or 0,
            )
        ]
    discarded: list[dict[str, Any]] = []
    for image_index, image in enumerate(images):
        if image_index in selected_by_index:
            continue
        related_sections = [
            record["section"]
            for record in section_records
            if any(
                candidate["image_index"] == image_index
                for candidate in record["candidates"]
            )
        ]
        discarded.append(
            {
                "figure": _paper_image_label(image),
                "reason": (
                    "全局最多4张，已优先覆盖主要 section"
                    if related_sections
                    else "没有足够的正文对应关系"
                ),
                "candidate_sections": related_sections,
            }
        )
    for record in section_records:
        def public_score(candidate: dict[str, Any]) -> dict[str, Any]:
            return {
                "figure": candidate["label"],
                "score": candidate["score"],
                "explicit_reference": candidate["explicit_reference"],
                "source_score": candidate.get("source_score", 0),
                "source_reason": candidate.get("source_reason", ""),
                "source_evidence": candidate.get("source_evidence", []),
                "heading_score": candidate.get("heading_score", 0),
                "scientific_score": candidate.get("scientific_score", 0),
                "semantic_score": candidate["semantic_score"],
                "structural_score": candidate["structural_score"],
                "keyword_score": candidate["keyword_score"],
                "match_method": candidate["match_method"],
            }
        record["candidates"] = [
            public_score(candidate) for candidate in record["candidates"]
        ]
        record["figure_scores"] = [
            public_score(candidate) for candidate in record.get("figure_scores", [])
        ]
    diagnostics = {
        "max_images": limit,
        "input_image_count": len(images),
        "numbered_figure_count": len(numbered_images),
        "sections": section_records,
        "selected_figures": [
            {
                "figure": _paper_image_label(image),
                "reason": selected_reasons.get(
                    selected_image_indices[id(image)], "section coverage"
                ),
            }
            for image in selected_images
        ],
        "discarded_figures": discarded,
    }
    return selected_images, diagnostics


def _select_article_images(
    images: list[dict[str, Any]],
    content_type: str,
    context: str = "",
    allocation: dict[str, Any] | None = None,
    source_context: str = "",
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
    if content_type == PAPER_CONTENT:
        body, diagnostics = _paper_allocate_images(
            body,
            context,
            limit,
            source_context,
        )
        if allocation is not None:
            allocation.clear()
            allocation.update(diagnostics)
    else:
        body = body[:limit]
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


def _paper_figure_caption(
    image: dict[str, Any],
    generated_caption: str,
    display_index: int,
) -> tuple[int, str]:
    figure_number = int(image.get("figure_number") or display_index)
    caption = generated_caption.strip() or str(
        image.get("original_caption")
        or image.get("caption")
        or image.get("alt")
        or image.get("metadata_title")
        or image.get("figure_title")
        or ""
    ).strip()
    caption = re.sub(
        r"^\s*fig(?:ure)?\.?\s*\d+\s*(?:[|:.-]\s*)?",
        "",
        caption,
        flags=re.IGNORECASE,
    ).strip()
    return figure_number, caption or str(image.get("figure_title") or f"Figure {figure_number}")


def _paper_text_slots(lines: list[str]) -> list[tuple[int, int, str]]:
    """Return paragraph insertion points in document order."""
    heading_indexes = [
        index
        for index, line in enumerate(lines)
        if re.match(r"^#{2,3}\s+", line.strip())
    ]
    ranges: list[tuple[int, int, str]] = []
    if heading_indexes:
        for position, start in enumerate(heading_indexes):
            end = (
                heading_indexes[position + 1]
                if position + 1 < len(heading_indexes)
                else len(lines)
            )
            ranges.append((start, end, lines[start].strip()))
    else:
        ranges.append((0, len(lines), ""))

    slots: list[tuple[int, int, str]] = []
    for section_index, (section_start, section_end, heading) in enumerate(ranges):
        index = section_start + (1 if heading else 0)
        paragraph_starts: list[tuple[int, int]] = []
        while index < section_end:
            while index < section_end and not lines[index].strip():
                index += 1
            if index >= section_end:
                break
            start = index
            while index < section_end and lines[index].strip():
                index += 1
            paragraph_starts.append((start, index))
        if not paragraph_starts and heading:
            paragraph_starts.append((section_start, section_end))
        for paragraph_number, (start, end) in enumerate(paragraph_starts):
            paragraph_lines = lines[start:end]
            if paragraph_lines and all(line.strip().startswith("![") for line in paragraph_lines):
                continue
            context = "\n".join(paragraph_lines).strip()
            if paragraph_number == 0 and heading:
                context = f"{heading}\n{context}".strip()
            if context:
                slots.append((end, section_index, context))
    if not slots:
        slots.append((len(lines), 0, "\n".join(lines)))
    return slots


def _paper_section_slots(
    lines: list[str],
) -> list[tuple[int, int, str, str]]:
    """Return one aggregate text slot for each major ``##`` section."""
    all_heading_indexes = [
        index
        for index, line in enumerate(lines)
        if re.match(r"^##\s+", line.strip())
    ]
    heading_indexes = [
        index
        for index in all_heading_indexes
        if not re.match(
            r"^##\s+(文章信息|来源|参考文献)\s*$",
            lines[index].strip(),
        )
    ]
    if not heading_indexes:
        return []
    sections: list[tuple[int, int, str, str]] = []
    for start in heading_indexes:
        following_headings = [index for index in all_heading_indexes if index > start]
        end = following_headings[0] if following_headings else len(lines)
        heading = lines[start].strip()
        body_lines: list[str] = []
        for line in lines[start + 1 : end]:
            stripped = line.strip()
            if not stripped or stripped.startswith(">"):
                continue
            if stripped.startswith("![") or stripped.startswith("*Fig."):
                continue
            body_lines.append(stripped)
        context = "\n".join([heading, *body_lines]).strip()
        sections.append((start, end, heading, context or heading))
    return sections


def _paper_section_heading_text(heading: str) -> str:
    return re.sub(r"^##\s+", "", heading.strip()).strip()


def _insert_paper_figures(
    markdown_path: Path,
    images: list[dict[str, Any]],
    generated_captions: list[str],
    dossier: dict[str, Any],
) -> list[str]:
    if not images:
        return []
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    slots = _paper_text_slots(lines)
    section_slots = _paper_section_slots(lines)
    image_order = _paper_body_image_order(images)
    ordered_entries = [
        (
            images[source_index],
            generated_captions[source_index]
            if source_index < len(generated_captions)
            else "",
        )
        for source_index in image_order
    ]
    insertions: dict[int, list[str]] = {}
    retained_images: list[dict[str, Any]] = []
    retained_generated_captions: list[str] = []
    effective_captions: list[str] = []
    last_insertion_index = -1
    last_numbered_section = -1
    allocation = dossier.get("paper_image_allocation")
    mapped_sections: dict[str, list[dict[str, Any]]] = {}
    if isinstance(allocation, dict):
        for section in allocation.get("sections", []):
            for figure in section.get("selected_figures", []):
                mapped_sections.setdefault(str(figure), []).append(
                    {
                        "section_index": section.get("section_index"),
                        "section": str(section.get("section") or ""),
                    }
                )
    used_slots: set[int] = set()
    inserted_image_keys: set[str] = set()
    inserted_section_records: list[dict[str, Any]] = []
    for index, (image, generated) in enumerate(ordered_entries, start=1):
        figure_number, caption = _paper_figure_caption(image, generated, index)
        numbered = _numbered_paper_figure(image) is not None
        if numbered and section_slots:
            image_key = _paper_image_label(image)
            if image_key in inserted_image_keys:
                continue
            mapped_section = None
            mapped_targets = mapped_sections.get(image_key, [])
            for target in mapped_targets:
                target_heading = str(target.get("section") or "").strip()
                matching_sections = [
                    section_index
                    for section_index, section in enumerate(section_slots)
                    if _paper_section_heading_text(section[2]) == target_heading
                    and section_index >= last_numbered_section
                ]
                if matching_sections:
                    mapped_section = matching_sections[0]
                    break
                try:
                    target_index = int(target.get("section_index"))
                except (TypeError, ValueError):
                    target_index = -1
                if (
                    0 <= target_index < len(section_slots)
                    and target_index >= last_numbered_section
                ):
                    mapped_section = target_index
                    break
            if mapped_section is not None:
                section_index = mapped_section
            else:
                eligible_sections = [
                    (section_index, section)
                    for section_index, section in enumerate(section_slots)
                    if section_index >= last_numbered_section
                    and section[1] >= last_insertion_index
                ]
                relevance = {
                    section_index: _paper_figure_match_score(image, section[3])
                    for section_index, section in eligible_sections
                }
                if not relevance or max(relevance.values()) <= 0:
                    # Do not invent a placement for a figure whose caption is not
                    # supported by any later major section.
                    continue
                section_index = max(
                    relevance,
                    key=lambda candidate: (relevance[candidate], -candidate),
                )
            insertion_index = section_slots[section_index][1]
            last_numbered_section = section_index
        else:
            eligible = [
                (slot_index, slot)
                for slot_index, slot in enumerate(slots)
                if slot_index not in used_slots and slot[0] > last_insertion_index
            ]
            if not eligible:
                eligible = [
                    (slot_index, slot)
                    for slot_index, slot in enumerate(slots)
                    if slot_index not in used_slots
                ]
            slot_index = max(
                eligible,
                key=lambda candidate: (
                    _text_relevance_score(image, candidate[1][2]),
                    -candidate[0],
                ),
            )[0]
            insertion_index = slots[slot_index][0]
        if not numbered or not section_slots:
            used_slots.add(slot_index)
        last_insertion_index = insertion_index
        inserted_image_keys.add(
            _paper_image_label(image) if numbered else str(id(image))
        )
        retained_images.append(image)
        retained_generated_captions.append(generated)
        effective_captions.append(caption)
        if numbered and section_slots:
            inserted_section_records.append(
                {
                    "figure": _paper_image_label(image),
                    "section_index": section_index,
                    "section": _paper_section_heading_text(
                        section_slots[section_index][2]
                    ),
                }
            )
        local_name = Path(str(image["local_path"])).name
        block = [
            f"![Fig. {figure_number}](images/{local_name})",
            f"*Fig. {figure_number} | {caption}*",
        ]
        attribution = _paper_nd_attribution(image, dossier)
        if attribution:
            block.append(f"*{attribution}*")
        insertions.setdefault(insertion_index, []).append("\n".join(block))

    if dossier.get("content_type") == PAPER_CONTENT:
        dossier["body_images"] = retained_images
        dossier["generated_body_image_captions"] = retained_generated_captions
        if isinstance(allocation, dict):
            allocation["final_inserted_sections"] = inserted_section_records
    output: list[str] = []
    for index in range(len(lines) + 1):
        for block in insertions.get(index, []):
            if output and output[-1].strip():
                output.append("")
            output.extend(block.splitlines())
            output.append("")
        if index < len(lines):
            output.append(lines[index])
    markdown_path.write_text("\n".join(output).rstrip() + "\n", encoding="utf-8")
    return effective_captions


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


def _openalex_discovery_item(record: dict[str, Any], run_date: str) -> dict[str, Any]:
    doi = str(record.get("doi") or "").strip().lower()
    url = f"https://doi.org/{doi}" if doi else ""
    publication_date = str(record.get("publication_date") or "")
    journal = journal_display_name(str(record.get("journal") or ""))
    publisher = str(record.get("publisher") or "")
    work_type = str(record.get("type") or "")
    return {
        "source": journal or "OpenAlex",
        "url": url,
        "canonical_url": url,
        "title": str(record.get("title") or ""),
        "normalized_title": normalize_title(str(record.get("title") or "")),
        "summary": str(record.get("abstract") or ""),
        "published_at": f"{publication_date}T00:00:00+00:00" if publication_date else "",
        "doi": doi,
        "journal": journal,
        "publisher": publisher,
        "word_count": 0,
        "work_type": work_type,
        "discovery_origin": "openalex",
        "status": "openalex_discovered",
        "discovered_at": f"{run_date}T00:00:00+00:00",
        "openalex": {
            "configured": True,
            "found": True,
            "doi": doi,
            "title": str(record.get("title") or ""),
            "abstract": str(record.get("abstract") or ""),
            "journal": journal,
            "publisher": publisher,
            "publication_date": publication_date,
            "work_type": work_type,
        },
    }


def merge_paper_candidate_pool(
    rss_candidates: list[dict[str, Any]],
    openalex_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return deduplicate([*rss_candidates, *openalex_candidates])


class NewsPipeline:
    def __init__(self, settings: Settings, logger: logging.Logger | None = None) -> None:
        self.settings = settings
        self.logger = logger or logging.getLogger("wechat_news.pipeline")
        self.db = Database(settings.database_path)
        self.openalex = OpenAlexAdapter(settings.openalex_api_key)
        self.feed_config = PROJECT_ROOT / "config" / "feeds.yaml"
        self.refresh_lock = asyncio.Lock()
        self.last_source_counts: dict[str, int] = {}
        self.last_paper_discovery_stats: dict[str, int] = {}
        self.last_paper_journal_counts: dict[str, int] = {}
        self.last_paper_refresh_warning = ""
        self.last_paper_batch_total = 0
        self.last_paper_batch_only = False

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
            if not item.get("doi") or paper_relevance_score(item) == 0:
                return None
            metadata = item.get("openalex") or {}
            if not metadata.get("found"):
                async with semaphore:
                    try:
                        metadata = await asyncio.to_thread(
                            self.openalex.lookup_doi,
                            item.get("doi"),
                        )
                    except Exception as exc:
                        self.logger.warning("OpenAlex DOI lookup failed: %s", exc)
                        return None
            journal = journal_display_name(str(metadata.get("journal") or ""))
            if not is_allowed_paper_journal(
                journal,
                str(metadata.get("publisher") or ""),
            ):
                return None
            merged = dict(item)
            merged["openalex"] = dict(metadata, journal=journal)
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
            merged["journal"] = journal
            merged["publisher"] = metadata.get("publisher") or ""
            merged["source"] = journal or merged.get("source") or "OpenAlex"
            merged["work_type"] = metadata.get("work_type") or merged.get("work_type") or ""
            if metadata.get("publication_date"):
                merged["published_at"] = f"{metadata['publication_date']}T00:00:00+00:00"
            local_score = paper_relevance_score(merged)
            if local_score == 0:
                return None
            merged["paper_local_score"] = local_score
            merged["discovery_origin"] = merged.get("discovery_origin") or "rss"
            merged["status"] = "published_paper"
            return merged

        verified = await asyncio.gather(*(verify(item) for item in items))
        return [item for item in verified if item is not None]

    async def refresh(
        self,
        date: str | None = None,
        content_type: str | None = None,
        *,
        exclude_seen: bool = False,
        append: bool = False,
    ) -> list[dict[str, Any]]:
        run_date = date or local_date(self.settings)
        run_type = content_type or content_type_for_date(run_date)
        if run_type not in {POPULAR_CONTENT, PAPER_CONTENT}:
            raise ValueError(f"unsupported content type: {run_type}")
        async with self.refresh_lock:
            self.last_paper_refresh_warning = ""
            if not append:
                self.last_paper_batch_total = 0
                self.last_paper_batch_only = False
            existing_paper_candidates = (
                self.db.get_candidates(run_date, PAPER_CONTENT)
                if run_type == PAPER_CONTENT
                else []
            )
            seen_paper_ids = (
                self.db.get_seen_candidate_ids(run_date, PAPER_CONTENT)
                if run_type == PAPER_CONTENT and exclude_seen
                else set()
            )
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
                rss_paper_count = 0
                openalex_added_count = 0
                journal_first_count = 0
                topic_openalex_count = 0
                merged_unique_count = 0
                after_journal_whitelist_count = 0
                after_published_seen_count = 0
                journal_counts: dict[str, int] = {}
                for window_hours in LOOKBACK_HOURS:
                    items, window_errors, window_counts = await asyncio.to_thread(
                        fetch_all_feeds,
                        self.feed_config,
                        window_hours,
                    )
                    feed_errors = list(dict.fromkeys([*feed_errors, *window_errors]))
                    source_counts = window_counts
                    lookback_hours = window_hours
                    if run_type == PAPER_CONTENT:
                        openalex_papers: list[dict[str, Any]] = []
                        if self.openalex.configured:
                            run_day = date_type.fromisoformat(run_date)
                            window_days = max(1, (window_hours + 23) // 24)
                            discovered = await asyncio.to_thread(
                                self.openalex.discover_recent_papers,
                                run_day - timedelta(days=window_days),
                                run_day,
                            )
                            openalex_papers = await self._published_papers(
                                [
                                    _openalex_discovery_item(record, run_date)
                                    for record in discovered
                                ],
                                run_date,
                            )
                            journal_first_count = getattr(
                                self.openalex,
                                "last_journal_first_count",
                                0,
                            )
                            topic_openalex_count = getattr(
                                self.openalex,
                                "last_topic_count",
                                0,
                            )
                            journal_counts = dict(
                                getattr(self.openalex, "last_journal_first_counts", {})
                            )
                        topic_items = [
                            item
                            for item in items
                            if item.get("doi")
                            and paper_relevance_score(item) > 0
                            and not _is_published_article(item, published)
                        ]
                        normalized = deduplicate(topic_items)
                        rss_enriched = await self._extract_shortlist(normalized)
                        rss_enriched = [
                            item
                            for item in deduplicate(rss_enriched)
                            if paper_relevance_score(item) > 0
                        ]
                        rss_papers = deduplicate(
                            await self._published_papers(rss_enriched, run_date)
                        )
                        rss_paper_count = len(rss_papers)
                        merged_papers = merge_paper_candidate_pool(
                            rss_papers,
                            openalex_papers,
                        )
                        merged_unique_count = len(merged_papers)
                        after_journal_whitelist_count = len(merged_papers)
                        openalex_added_count = sum(
                            item.get("discovery_origin") == "openalex"
                            for item in merged_papers
                        )
                        enriched = sorted(
                            merged_papers,
                            key=lambda item: (
                                int(item.get("paper_local_score") or 0),
                                deterministic_score(item),
                            ),
                            reverse=True,
                        )
                        eligible: list[dict[str, Any]] = []
                        for item in enriched:
                            if _is_published_article(item, published):
                                continue
                            article_id = self.db.upsert_article(item)
                            item["article_id"] = article_id
                            if article_id in seen_paper_ids:
                                continue
                            if _is_published_article(item, published):
                                continue
                            if item.get("images") is not None:
                                self.db.replace_images(article_id, item.get("images") or [])
                            item["score"] = deterministic_score(item)
                            eligible.append(item)
                        enriched = eligible
                        after_published_seen_count = len(enriched)
                    else:
                        topic_items = [
                            item
                            for item in items
                            if source_allowed_for_content(
                                str(item.get("source") or ""),
                                run_type,
                            )
                            and is_relevant_news_rss_prefilter(item)
                            and not _is_published_article(item, published)
                        ]
                        normalized = deduplicate(topic_items)
                        enriched = await self._extract_shortlist(normalized)
                        enriched = [
                            item
                            for item in deduplicate(enriched)
                            if is_relevant_news_after_extraction(item)
                        ]
                    if len(enriched) >= 10:
                        break
                self.last_source_counts = source_counts
                if run_type == PAPER_CONTENT:
                    self.last_paper_journal_counts = journal_counts
                    self.last_paper_discovery_stats = {
                        "lookback": lookback_hours,
                        "journal_first": journal_first_count,
                        "topic_openalex": topic_openalex_count,
                        "rss": rss_paper_count,
                        "merged_unique": merged_unique_count,
                        "after_journal_whitelist": after_journal_whitelist_count,
                        "after_published_seen": after_published_seen_count,
                        "after_local_relevance": len(enriched),
                        "ai_examined": 0,
                        "ai_kept": 0,
                        "final": 0,
                        "rss_candidates": rss_paper_count,
                        "openalex_added": openalex_added_count,
                        "coarse_filtered": len(enriched),
                    }

                if run_type != PAPER_CONTENT:
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

                title_error = ""
                if run_type == PAPER_CONTENT:
                    selected: list[dict[str, Any]] = []
                    used_model = True
                    llm_error = ""
                    ai_examined = 0
                    if not self.settings.model_configured:
                        selected = [
                            dict(item, title_cn=str(item.get("title_cn") or ""))
                            for item in enriched
                            if int(item.get("paper_local_score") or 0) >= 2
                        ][:10]
                        used_model = False
                        llm_error = "model not configured"
                    else:
                        for offset in range(0, len(enriched), 30):
                            batch = enriched[offset : offset + 30]
                            if not batch:
                                break
                            ai_examined += len(batch)
                            batch_selected, batch_used, batch_error = await asyncio.to_thread(
                                select_paper_top_ten,
                                batch,
                                self.settings,
                            )
                            if not batch_used:
                                used_model = False
                                llm_error = batch_error
                                selected = [
                                    dict(item, title_cn=str(item.get("title_cn") or ""))
                                    for item in enriched
                                    if int(item.get("paper_local_score") or 0) >= 2
                                ][:10]
                                break
                            selected.extend(batch_selected)
                            if len(selected) >= 10:
                                break
                    selected = sorted(
                        selected,
                        key=lambda item: (
                            int(item.get("paper_relevance_score") or 0),
                            float(item.get("score") or deterministic_score(item)),
                        ),
                        reverse=True,
                    )[:10]
                    if self.last_paper_discovery_stats:
                        self.last_paper_discovery_stats.update(
                            {
                                "ai_examined": ai_examined,
                                "ai_kept": len(selected) if used_model else 0,
                                "final": len(selected),
                            }
                        )
                    if used_model and selected:
                        translated, _, title_error = await asyncio.to_thread(
                            translate_paper_titles,
                            selected,
                            self.settings,
                        )
                        selected = [
                            dict(item, title_cn=translated[index] or str(item.get("title_cn") or ""))
                            for index, item in enumerate(selected)
                        ]
                        if self.last_paper_discovery_stats:
                            self.last_paper_discovery_stats["final"] = len(selected)
                else:
                    prioritized = prioritize_candidates(enriched)
                    selected, used_model, llm_error = await asyncio.to_thread(
                        select_top_ten,
                        prioritized,
                        self.settings,
                    )
                    selected = prioritize_candidates(selected)
                errors = list(feed_errors)
                if (
                    run_type == PAPER_CONTENT
                    and (not used_model or not selected)
                    and existing_paper_candidates
                ):
                    self.last_paper_refresh_warning = (
                        "⚠ 本次 AI 刷新失败，继续使用今日最近一次成功结果"
                    )
                    if llm_error:
                        errors.append(f"LLM selection fallback: {llm_error}")
                    self.db.set_daily_run(
                        run_date,
                        fetched_at=utc_now(),
                        candidate_count=len(existing_paper_candidates),
                        content_type=run_type,
                        status="degraded",
                        error=" | ".join(errors)[:2000],
                    )
                    return existing_paper_candidates
                if self.settings.model_configured and llm_error:
                    errors.append(f"LLM selection fallback: {llm_error}")
                if title_error:
                    errors.append(f"LLM title translation warning: {title_error}")
                    self.logger.warning(
                        "PAPER title translation unavailable date=%s error=%s",
                        run_date,
                        title_error,
                    )
                if run_type == PAPER_CONTENT and not used_model:
                    self.last_paper_refresh_warning = (
                        "⚠ AI 筛选暂时不可用，当前显示本地筛选结果"
                    )
                if append:
                    self.db.append_candidates(run_date, selected, run_type)
                else:
                    self.db.replace_candidates(run_date, selected, run_type)
                if run_type == PAPER_CONTENT and selected:
                    self.db.add_seen_candidates(
                        run_date,
                        PAPER_CONTENT,
                        [int(item["article_id"]) for item in selected],
                    )
                stored_candidates = self.db.get_candidates(run_date, run_type)
                status = "success" if selected else "empty"
                if feed_errors and selected:
                    status = "partial"
                self.db.set_daily_run(
                    run_date,
                    fetched_at=utc_now(),
                    candidate_count=len(stored_candidates),
                    content_type=run_type,
                    status=status,
                    error=" | ".join(errors)[:2000],
                )
                if run_type == PAPER_CONTENT:
                    self.logger.info(
                        "PAPER discovery stats lookback=%sh journal_first=%s topic_openalex=%s rss=%s "
                        "merged_unique=%s after_journal_whitelist=%s after_published_seen=%s "
                        "after_local_relevance=%s ai_examined=%s ai_kept=%s final=%s journals=%s",
                        lookback_hours,
                        self.last_paper_discovery_stats.get("journal_first", 0),
                        self.last_paper_discovery_stats.get("topic_openalex", 0),
                        self.last_paper_discovery_stats.get("rss", 0),
                        self.last_paper_discovery_stats.get("merged_unique", 0),
                        self.last_paper_discovery_stats.get("after_journal_whitelist", 0),
                        self.last_paper_discovery_stats.get("after_published_seen", 0),
                        self.last_paper_discovery_stats.get("after_local_relevance", 0),
                        self.last_paper_discovery_stats.get("ai_examined", 0),
                        self.last_paper_discovery_stats.get("ai_kept", 0),
                        self.last_paper_discovery_stats.get("final", 0),
                        self.last_paper_journal_counts,
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
        self.last_paper_refresh_warning = ""
        self.last_paper_batch_total = 0
        self.last_paper_batch_only = False
        existing = self.db.get_candidates(run_date, run_type)
        run = self.db.get_daily_run(run_date, run_type)
        if existing and (
            run_type == PAPER_CONTENT
            or (run and run.get("content_type") == run_type)
        ):
            if run_type != PAPER_CONTENT:
                published = self.db.published_article_identifiers()
                if any(_is_published_article(item, published) for item in existing):
                    return await self.refresh(run_date, run_type)
            if self.settings.model_configured and any(
                not str(item.get("title_cn") or "").strip() for item in existing
            ):
                if run_type == PAPER_CONTENT:
                    titled, used_model, title_error = await asyncio.to_thread(
                        translate_paper_titles,
                        existing,
                        self.settings,
                    )
                    if used_model:
                        titled_candidates = [
                            dict(
                                item,
                                article_id=int(item["id"]),
                                title_cn=titled[index] or str(item.get("title_cn") or ""),
                            )
                            for index, item in enumerate(existing)
                        ]
                        self.db.replace_candidates(
                            run_date,
                            titled_candidates,
                            run_type,
                        )
                        return self.db.get_candidates(run_date, run_type)
                    self.logger.warning(
                        "PAPER title translation unavailable date=%s error=%s",
                        run_date,
                        title_error,
                    )
                else:
                    title_candidates = [
                        dict(
                            item,
                            article_id=int(item["id"]),
                            paper_local_score=item.get("paper_local_score", 0),
                        )
                        for item in existing
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

    async def next_paper_batch(
        self,
        date: str | None = None,
    ) -> list[dict[str, Any]]:
        run_date = date or local_date(self.settings)
        self.last_paper_batch_total = 0
        self.last_paper_batch_only = False
        current = self.db.get_candidates(run_date, PAPER_CONTENT)
        if not current:
            return await self.refresh(run_date, PAPER_CONTENT)
        current_ids = {int(item["id"]) for item in current}
        current_count = len(current)
        self.db.add_seen_candidates(
            run_date,
            PAPER_CONTENT,
            current_ids,
        )
        result = await self.refresh(
            run_date,
            PAPER_CONTENT,
            exclude_seen=True,
            append=True,
        )
        if len(result) > current_count:
            self.last_paper_batch_total = len(result)
            self.last_paper_batch_only = True
            return result[current_count:]
        self.last_paper_refresh_warning = "⚠ 换一批失败，继续保留当前论文列表"
        return result

    def format_news(self, candidates: list[dict[str, Any]]) -> str:
        if not candidates:
            return "今日暂无可用科研新闻候选。"
        content_type = str(candidates[0].get("content_type") or POPULAR_CONTENT)
        if content_type == PAPER_CONTENT:
            heading = (
                f"今日已发表论文（本批次新增{len(candidates)}篇，累计{self.last_paper_batch_total}篇）"
                if self.last_paper_batch_only
                else f"今日已发表论文（共{len(candidates)}篇）"
            )
            visible_candidates = candidates
        else:
            heading = "今日科普新闻 Top 10"
            visible_candidates = candidates[:10]
        lines = []
        if content_type == PAPER_CONTENT and self.last_paper_refresh_warning:
            lines.append(self.last_paper_refresh_warning)
        lines.append(f"## {heading}")
        for item in visible_candidates:
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
            html_images = list(dossier.get("images") or [])
            html_local_flags = [
                bool(image.get("local_path") and Path(str(image["local_path"])).is_file())
                for image in html_images
            ]
            html_remote_images = [
                image
                for image, is_local in zip(html_images, html_local_flags)
                if not is_local
            ]
            downloaded_html_images = await asyncio.to_thread(
                download_publishable_images,
                html_remote_images,
                str(output_dir / "images"),
            )
            html_remote_iterator = iter(downloaded_html_images)
            dossier["images"] = [
                dict(image) if is_local else next(html_remote_iterator)
                for image, is_local in zip(html_images, html_local_flags)
            ]
            dossier["pdf_figure_fallback"] = {
                "attempted": False,
                "reason": "usable legal HTML image downloaded",
            }
            if not _has_publishable_html_image(list(dossier.get("images") or [])):
                dossier["pdf_figure_fallback"] = {
                    "attempted": True,
                    "reason": (
                        "no usable downloaded HTML hero, graphical abstract, cover, or figure"
                    ),
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
                            doi=str(dossier.get("doi") or ""),
                            wiley_tdm_token=self.settings.wiley_tdm_api_token,
                        )
                        dossier["images"] = list(dossier.get("images") or []) + pdf_figures
                        dossier["pdf_figure_metadata"] = pdf_metadata
                        dossier["pdf_figure_fallback"]["matched_figures"] = len(pdf_figures)
                        dossier["pdf_figure_fallback"]["pdf_download"] = pdf_metadata.get(
                            "pdf_download", {}
                        )
                        download_info = dossier["pdf_figure_fallback"]["pdf_download"]
                        dossier["pdf_figure_fallback"].update(
                            {
                                "source": download_info.get("source", "ordinary_pdf"),
                                "status": download_info.get("status"),
                            }
                        )
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
            paper_pdf_source = dict(dossier.get("pdf_figure_source") or {})
            fallback_attempted = bool(
                (dossier.get("pdf_figure_fallback") or {}).get("attempted")
            )
            if not source_pdf.is_file() and not fallback_attempted:
                try:
                    if not paper_pdf_source.get("pdf_url"):
                        openalex = dossier.get("openalex") or {}
                        paper_pdf_source = await asyncio.to_thread(
                            discover_pdf_source,
                            str(dossier.get("url") or ""),
                            str(dossier.get("doi") or ""),
                            str(openalex.get("license") or ""),
                        )
                    if paper_pdf_source.get("pdf_url"):
                        paper_pdf_download = await asyncio.to_thread(
                            download_pdf_with_wiley_tdm,
                            str(paper_pdf_source["pdf_url"]),
                            source_pdf,
                            doi=str(dossier.get("doi") or ""),
                            token=self.settings.wiley_tdm_api_token,
                            article_url=str(dossier.get("url") or ""),
                        )
                        dossier["paper_pdf_download"] = paper_pdf_download
                        if not paper_pdf_download.get("success"):
                            raise RuntimeError(
                                str(paper_pdf_download.get("error") or "PDF download failed")
                            )
                    else:
                        dossier["paper_first_page_error"] = "no formal/reference PDF found"
                except Exception as exc:
                    self.logger.warning("Paper first-page PDF download failed: %s", exc)
                    dossier["paper_first_page_error"] = f"{type(exc).__name__}: {exc}"[:1000]
            dossier["paper_first_page_pdf_source"] = paper_pdf_source
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
                            "license": paper_pdf_source.get("license")
                            or openalex.get("license")
                            or "",
                            "license_url": paper_pdf_source.get("license_url", ""),
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
        paper_image_allocation: dict[str, Any] = {}
        cover_image, body_images, redundant_count = _select_article_images(
            legal_images,
            str(dossier.get("content_type") or POPULAR_CONTENT),
            selection_context,
            paper_image_allocation if dossier.get("content_type") == PAPER_CONTENT else None,
            str(dossier.get("text") or "")
            if dossier.get("content_type") == PAPER_CONTENT
            else "",
        )
        body_image_captions = await asyncio.to_thread(
            generate_image_captions,
            body_images,
            self.settings,
        )
        dossier["cover_image"] = cover_image or {}
        dossier["body_images"] = body_images
        dossier["generated_body_image_captions"] = body_image_captions
        if dossier.get("content_type") == PAPER_CONTENT:
            dossier["paper_image_allocation"] = paper_image_allocation
        if dossier.get("content_type") == PAPER_CONTENT and body_images:
            body_image_captions = _insert_paper_figures(
                markdown_path,
                body_images,
                body_image_captions,
                dossier,
            )
            allocation = dossier.get("paper_image_allocation")
            if isinstance(allocation, dict):
                allocation["final_inserted_figures"] = [
                    _paper_image_label(image)
                    for image in dossier.get("body_images") or []
                ]
            body_images = list(dossier.get("body_images") or [])
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
        if body_images and dossier.get("content_type") != PAPER_CONTENT:
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
                    "paper_first_page_pdf_source": dossier.get(
                        "paper_first_page_pdf_source", {}
                    ),
                    "wechat_cover": dossier.get("wechat_cover", {}),
                    "cover_image": dossier.get("cover_image", {}),
                    "body_images": dossier.get("body_images", []),
                    "generated_body_image_captions": dossier.get(
                        "generated_body_image_captions", []
                    ),
                    "body_image_captions": dossier.get("body_image_captions", []),
                    "paper_image_allocation": dossier.get("paper_image_allocation", {}),
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
