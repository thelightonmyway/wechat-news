"""Resolve DOI URLs to publisher-neutral final article landing pages."""

from __future__ import annotations

from urllib.parse import urlparse

import httpx

USER_AGENT = "Mozilla/5.0 (compatible; wechat-news/0.1; +local-research-bot)"


def _is_doi_url(url: str) -> bool:
    return urlparse(url).netloc.lower() in {"doi.org", "dx.doi.org"}


def _is_html_landing(url: str, content_type: str) -> bool:
    path = urlparse(url).path.lower()
    return bool(url and not _is_doi_url(url) and not path.endswith(".pdf") and "pdf" not in content_type)


def resolve_doi_landing_page(
    doi: str,
    current_url: str = "",
) -> dict[str, object]:
    """Follow DOI redirects and return the final publisher article URL without raising."""
    doi_value = doi.strip()
    candidates: list[str] = []
    if doi_value:
        candidates.append(f"https://doi.org/{doi_value}")
    if current_url:
        candidates.append(current_url)

    last_error = ""
    fallback_url = current_url or (candidates[0] if candidates else "")
    with httpx.Client(timeout=30.0, follow_redirects=True, trust_env=True) as client:
        for candidate in dict.fromkeys(candidates):
            try:
                response = client.get(
                    candidate,
                    headers={
                        "User-Agent": USER_AGENT,
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    },
                )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"[:1000]
                continue

            final_url = str(response.url)
            content_type = str(response.headers.get("content-type") or "").lower()
            if _is_html_landing(final_url, content_type):
                return {
                    "resolved": True,
                    "landing_url": final_url,
                    "status_code": response.status_code,
                    "accessible": response.is_success,
                    "error": "" if response.is_success else f"HTTP {response.status_code}",
                }
            fallback_url = final_url or fallback_url
            if response.is_error:
                last_error = f"HTTP {response.status_code}"

    return {
        "resolved": False,
        "landing_url": fallback_url,
        "status_code": 0,
        "accessible": False,
        "error": last_error or "DOI landing page could not be resolved",
    }
