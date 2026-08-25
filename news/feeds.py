"""Fixed V1 RSS feed discovery using feedparser."""

from __future__ import annotations

import calendar
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import feedparser
import httpx
import yaml
from bs4 import BeautifulSoup

DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
TRACKING_KEYS = {"fbclid", "gclid", "mc_cid", "mc_eid", "cmpid", "campaign"}
USER_AGENT = "wechat-news-qq-bot/0.1 RSS"


def canonicalize_url(url: str) -> str:
    if not url:
        return ""
    parts = urlsplit(url.strip())
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in TRACKING_KEYS
    ]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, urlencode(query), ""))


def normalize_title(title: str) -> str:
    title = BeautifulSoup(title or "", "html.parser").get_text(" ", strip=True).lower()
    title = re.sub(r"[^\w\s]", " ", title, flags=re.UNICODE)
    return " ".join(title.split())


def clean_html_text(value: str) -> str:
    return " ".join(BeautifulSoup(value or "", "html.parser").get_text(" ", strip=True).split())


def extract_doi(*values: str) -> str:
    for value in values:
        match = DOI_RE.search(value or "")
        if match:
            return match.group(0).rstrip(".,;)").lower()
    return ""


def _entry_datetime(entry: Any, now: datetime) -> datetime:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed:
        return datetime.fromtimestamp(calendar.timegm(parsed), tz=timezone.utc)
    return now


def load_feeds(path: Path) -> list[dict[str, str]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    feeds = data.get("feeds") or []
    return [feed for feed in feeds if feed.get("name") and feed.get("url")]


def fetch_all_feeds(
    config_path: Path,
    hours: int = 36,
    now: datetime | None = None,
) -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
    current = now or datetime.now(timezone.utc)
    cutoff = current - timedelta(hours=hours)
    items: list[dict[str, Any]] = []
    errors: list[str] = []
    source_counts: dict[str, int] = {}

    with httpx.Client(timeout=25.0, follow_redirects=True, trust_env=True) as client:
        for feed in load_feeds(config_path):
            name = str(feed["name"])
            try:
                response = client.get(feed["url"], headers={"User-Agent": USER_AGENT})
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                if "text/html" in content_type:
                    page = BeautifulSoup(response.text, "html.parser")
                    alternate = page.find(
                        "link",
                        attrs={"rel": lambda value: value and "alternate" in value},
                        type=lambda value: value and "rss" in value.lower(),
                    )
                    if alternate and alternate.get("href"):
                        response = client.get(
                            urljoin(str(response.url), str(alternate["href"])),
                            headers={"User-Agent": USER_AGENT},
                        )
                        response.raise_for_status()
                parsed = feedparser.parse(response.content)
                if not parsed.entries and getattr(parsed, "bozo", False):
                    raise RuntimeError(str(getattr(parsed, "bozo_exception", "invalid feed")))
                count = 0
                for entry in parsed.entries:
                    title = clean_html_text(str(entry.get("title", "")))
                    url = str(entry.get("link", "")).strip()
                    if not title or not url:
                        continue
                    published = _entry_datetime(entry, current)
                    if published < cutoff:
                        continue
                    summary = clean_html_text(
                        str(entry.get("summary") or entry.get("description") or "")
                    )
                    doi = extract_doi(
                        str(entry.get("prism_doi", "")),
                        str(entry.get("dc_identifier", "")),
                        url,
                        title,
                        summary,
                    )
                    items.append(
                        {
                            "source": name,
                            "url": url,
                            "canonical_url": canonicalize_url(url),
                            "title": title,
                            "normalized_title": normalize_title(title),
                            "summary": summary,
                            "published_at": published.isoformat(),
                            "doi": doi,
                            "journal": "",
                            "word_count": 0,
                            "status": "discovered",
                            "discovered_at": current.isoformat(),
                        }
                    )
                    count += 1
                source_counts[name] = count
            except Exception as exc:
                source_counts[name] = 0
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
    return items, errors, source_counts
