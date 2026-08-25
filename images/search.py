"""Text-metadata-only search for clearly reusable public images."""

from __future__ import annotations

import json
import re
import subprocess
from typing import Any
from urllib.parse import urlsplit

import httpx
from bs4 import BeautifulSoup

from images.policy import apply_policy

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
NASA_API = "https://images-api.nasa.gov/search"
USER_AGENT = "wechat-news-qq-bot/0.1 image-metadata-search"
WIKIMEDIA_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 wechat-news/1.0"
)


def _plain(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("value", "")
    return " ".join(BeautifulSoup(str(value or ""), "html.parser").get_text(" ", strip=True).split())


def _wikimedia_json(params: dict[str, Any]) -> dict[str, Any]:
    with httpx.Client(timeout=30.0, follow_redirects=True, trust_env=True) as client:
        response = client.get(
            COMMONS_API,
            params=params,
            headers={"User-Agent": WIKIMEDIA_USER_AGENT},
        )
        if response.status_code != 403:
            response.raise_for_status()
            return response.json()

    command = [
        "curl",
        "--fail",
        "--silent",
        "--show-error",
        "--location",
        "--user-agent",
        WIKIMEDIA_USER_AGENT,
        "--get",
        COMMONS_API,
    ]
    for key, value in params.items():
        command.extend(["--data-urlencode", f"{key}={value}"])
    result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)
    return json.loads(result.stdout)


def search_wikimedia_commons(query: str, limit: int = 12) -> list[dict[str, Any]]:
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,
        "gsrlimit": min(50, max(1, limit)),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "format": "json",
        "formatversion": 2,
        "origin": "*",
    }
    pages = _wikimedia_json(params).get("query", {}).get("pages", [])

    records: list[dict[str, Any]] = []
    for page in pages:
        image_info = (page.get("imageinfo") or [{}])[0]
        metadata = image_info.get("extmetadata") or {}
        license_value = _plain(metadata.get("LicenseShortName") or metadata.get("UsageTerms"))
        description = _plain(metadata.get("ImageDescription") or metadata.get("ObjectName"))
        credit = _plain(metadata.get("Credit"))
        artist = _plain(metadata.get("Artist"))
        record = apply_policy(
            {
                "url": image_info.get("url") or "",
                "local_path": "",
                "caption": description or str(page.get("title") or "").removeprefix("File:"),
                "alt": description,
                "credit": " · ".join(part for part in (artist, credit, "Wikimedia Commons") if part),
                "license": license_value,
                "provider": "Wikimedia Commons",
                "metadata_title": str(page.get("title") or ""),
            }
        )
        if record["url"]:
            records.append(record)
    return records


def search_nasa_images(query: str, limit: int = 12) -> list[dict[str, Any]]:
    with httpx.Client(timeout=30.0, follow_redirects=True, trust_env=True) as client:
        response = client.get(
            NASA_API,
            params={"q": query, "media_type": "image", "page_size": min(100, limit)},
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
        items = response.json().get("collection", {}).get("items", [])[:limit]

    records: list[dict[str, Any]] = []
    for item in items:
        data = (item.get("data") or [{}])[0]
        links = item.get("links") or []
        image_url = next((link.get("href") for link in links if link.get("render") == "image"), "")
        license_value = data.get("license") or data.get("copyright") or ""
        record = apply_policy(
            {
                "url": image_url,
                "local_path": "",
                "caption": str(data.get("description") or data.get("title") or ""),
                "alt": str(data.get("title") or ""),
                "credit": " · ".join(
                    str(part)
                    for part in (data.get("photographer"), data.get("center"), "NASA")
                    if part
                ),
                "license": str(license_value),
                "provider": "NASA Images",
                "metadata_title": str(data.get("title") or ""),
            }
        )
        if record["url"]:
            records.append(record)
    return records


def _metadata_matches_keywords(record: dict[str, Any], keywords: list[str]) -> bool:
    stopwords = {
        "and",
        "the",
        "for",
        "with",
        "during",
        "effects",
        "conditions",
        "high",
        "map",
    }

    def normalize_term(term: str) -> str:
        if len(term) > 4 and term.endswith("ies"):
            return term[:-3] + "y"
        if len(term) > 4 and term.endswith("s"):
            return term[:-1]
        return term

    terms = {
        normalize_term(term)
        for keyword in keywords
        for term in re.findall(r"[a-z0-9]+", keyword.lower())
        if len(term) >= 4 and term not in stopwords
    }
    if len(terms) < 2:
        return True
    metadata_title = str(record.get("metadata_title") or "")
    text = metadata_title or str(record.get("caption") or "")
    record_terms = {
        normalize_term(term)
        for term in re.findall(r"[a-z0-9]+", text.lower())
    }
    return len(terms & record_terms) >= 2


def search_public_images(keywords: list[str], max_images: int = 5) -> list[dict[str, Any]]:
    """Search by text only and return only relevant, policy-approved metadata records."""
    clean_keywords = [" ".join(keyword.split()) for keyword in keywords if keyword.strip()][:5]
    if not clean_keywords:
        return []
    query = " ".join(clean_keywords)
    found: list[dict[str, Any]] = []
    errors: list[str] = []
    for search in (
        lambda: search_wikimedia_commons(query, 20),
        lambda: search_wikimedia_commons(f"NOAA {query}", 12),
        lambda: search_wikimedia_commons(f"NASA {query}", 12),
        lambda: search_nasa_images(query, 12),
    ):
        try:
            found.extend(search())
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

    approved: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in found:
        url = str(record.get("url") or "")
        path = urlsplit(url).path.lower()
        if (
            not record.get("publishable")
            or not url
            or url in seen
            or path.endswith((".pdf", ".svg"))
            or not _metadata_matches_keywords(record, clean_keywords)
        ):
            continue
        approved.append(record)
        seen.add(url)
        if len(approved) >= max_images:
            break
    return approved
