"""Select an accessible open HTML mirror from existing paper metadata."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

import httpx

USER_AGENT = "Mozilla/5.0 (compatible; wechat-news/0.1; +local-research-bot)"


def _pmc_url(value: str) -> str:
    match = re.search(r"(?:PMC)?(\d+)", value, re.IGNORECASE)
    return f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{match.group(1)}/" if match else ""


def _is_pmc(url: str) -> bool:
    lower = url.lower()
    return "pmc.ncbi.nlm.nih.gov/articles/" in lower or "/pmc/articles/" in lower


def _usable_html_candidate(url: str, publisher_url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.lower()
    return bool(
        parsed.scheme in {"http", "https"}
        and parsed.netloc
        and parsed.netloc.lower() not in {"doi.org", "dx.doi.org"}
        and url.rstrip("/") != publisher_url.rstrip("/")
        and not path.endswith(".pdf")
    )


def resolve_oa_html_mirror(
    metadata: dict[str, Any],
    publisher_url: str = "",
) -> dict[str, Any]:
    """Return the first accessible OA HTML mirror, preferring PMC."""
    candidates: list[tuple[int, str, str]] = []
    pmc_url = str(metadata.get("pmc_url") or "")
    if pmc_url:
        candidates.append((0, pmc_url, "PubMed Central"))
    pmcid = str(metadata.get("pmcid") or "")
    if pmcid:
        candidates.append((0, _pmc_url(pmcid), "PubMed Central"))

    for location in metadata.get("oa_locations") or []:
        if not isinstance(location, dict) or not location.get("is_oa"):
            continue
        url = str(location.get("landing_page_url") or "")
        source = str(location.get("source") or "OA repository")
        priority = 0 if _is_pmc(url) or "pubmed central" in source.lower() else 10
        if priority == 0:
            match = re.search(r"/pmc/articles/(?:PMC)?(\d+)", url, re.IGNORECASE)
            if match:
                url = _pmc_url(match.group(1))
        candidates.append((priority, url, source))

    unique: list[tuple[int, str, str]] = []
    seen: set[str] = set()
    for priority, url, source in sorted(candidates, key=lambda item: item[0]):
        normalized = url.rstrip("/")
        if not url or normalized in seen or not _usable_html_candidate(url, publisher_url):
            continue
        seen.add(normalized)
        unique.append((priority, url, source))

    errors: list[str] = []
    with httpx.Client(timeout=30.0, follow_redirects=True, trust_env=True) as client:
        for _, candidate, source in unique:
            try:
                response = client.get(
                    candidate,
                    headers={
                        "User-Agent": USER_AGENT,
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    },
                )
            except Exception as exc:
                errors.append(f"{candidate}: {type(exc).__name__}: {exc}"[:500])
                continue
            content_type = str(response.headers.get("content-type") or "").lower()
            if response.is_success and "html" in content_type:
                return {
                    "found": True,
                    "url": str(response.url),
                    "source": source,
                    "status_code": response.status_code,
                    "error": "",
                }
            errors.append(f"{candidate}: HTTP {response.status_code}"[:500])

    return {
        "found": False,
        "url": "",
        "source": "",
        "status_code": 0,
        "error": "; ".join(errors)[:1000] or "no accessible OA HTML mirror",
    }
