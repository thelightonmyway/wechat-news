"""Minimal PyAlex adapter."""

from __future__ import annotations

from datetime import date
import logging
import re
import time
from typing import Any

from pyalex import Works, config, invert_abstract

LOGGER = logging.getLogger("wechat_news.openalex")
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_RETRIES = 2
DISCOVERY_QUERIES = (
    "surface wind wind energy",
    "atmospheric circulation jet stream teleconnection",
    "ENSO NAO SAM climate predictability",
    "polar vortex stratosphere troposphere coupling",
    "ozone climate stratosphere",
    "temperature heatwave climate extremes",
    "precipitation water vapor moisture transport",
    "drought atmospheric climate mechanism",
    "tropical cyclone extreme weather",
    "boundary layer land atmosphere interaction",
    "air sea interaction ocean atmosphere",
    "polar climate sea ice dynamics",
    "climate variability predictability detection attribution",
    "physical climate model evaluation reanalysis observations",
)
DISCOVERY_SELECT = (
    "title,doi,publication_date,type,primary_location,abstract_inverted_index"
)


def normalize_doi(value: str | None) -> str:
    doi = (value or "").strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.lower().startswith(prefix):
            doi = doi[len(prefix):]
            break
    return doi.strip().lower()


def _status_code(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None) or getattr(exc, "status_code", None)
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    match = re.search(r"\b(429|500|502|503|504)\b", str(exc))
    return int(match.group(1)) if match else None


def _retryable_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return (
        _status_code(exc) in RETRYABLE_STATUS_CODES
        or isinstance(exc, TimeoutError)
        or "timeout" in text
        or "timed out" in text
    )


class OpenAlexAdapter:
    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key.strip()
        if self.api_key:
            config.api_key = self.api_key

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def discover_recent_papers(
        self,
        from_date: date,
        to_date: date,
        *,
        per_query: int = 15,
    ) -> list[dict[str, Any]]:
        if not self.configured:
            return []

        discovered: list[dict[str, Any]] = []
        seen_dois: set[str] = set()
        for query in DISCOVERY_QUERIES:
            works = None
            for attempt in range(MAX_RETRIES + 1):
                try:
                    works = (
                        Works()
                        .search(query)
                        .filter(
                            from_publication_date=from_date.isoformat(),
                            to_publication_date=to_date.isoformat(),
                        )
                        .select(DISCOVERY_SELECT)
                        .get(per_page=per_query)
                    )
                    break
                except Exception as exc:
                    if _retryable_error(exc) and attempt < MAX_RETRIES:
                        time.sleep(0.5 * (2**attempt))
                        continue
                    LOGGER.warning(
                        "OpenAlex discovery unavailable query=%r attempts=%s error=%s",
                        query,
                        attempt + 1,
                        exc,
                    )
                    works = []
                    break

            for work in works or []:
                doi = normalize_doi(work.get("doi"))
                if not doi or doi in seen_dois:
                    continue
                publication_date = str(work.get("publication_date") or "")
                try:
                    published = date.fromisoformat(publication_date)
                except ValueError:
                    continue
                if published < from_date or published > to_date:
                    continue
                primary_location = work.get("primary_location") or {}
                source = primary_location.get("source") or {}
                abstract_index = work.get("abstract_inverted_index")
                discovered.append(
                    {
                        "title": work.get("title") or "",
                        "abstract": invert_abstract(abstract_index) if abstract_index else "",
                        "publication_date": publication_date,
                        "journal": source.get("display_name") or "",
                        "doi": doi,
                        "type": work.get("type") or "",
                    }
                )
                seen_dois.add(doi)
        return discovered

    def lookup_doi(self, doi_value: str | None) -> dict[str, Any]:
        doi = normalize_doi(doi_value)
        if not doi:
            return {"configured": self.configured, "found": False, "doi": ""}
        if not self.configured:
            return {
                "configured": False,
                "found": False,
                "doi": doi,
                "error": "OPENALEX_API_KEY not configured",
            }

        works = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                works = Works().filter(doi=f"https://doi.org/{doi}").get(per_page=1)
                break
            except Exception as exc:
                if _retryable_error(exc) and attempt < MAX_RETRIES:
                    delay = 0.5 * (2**attempt)
                    LOGGER.warning(
                        "OpenAlex DOI lookup retry doi=%s attempt=%s/%s delay=%.1fs error=%s",
                        doi,
                        attempt + 1,
                        MAX_RETRIES,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
                    continue
                LOGGER.warning(
                    "OpenAlex DOI lookup unavailable doi=%s attempts=%s error=%s",
                    doi,
                    attempt + 1,
                    exc,
                )
                return {
                    "configured": True,
                    "found": False,
                    "doi": doi,
                    "error": f"{type(exc).__name__}: {exc}"[:1000],
                }
        if not works:
            return {"configured": True, "found": False, "doi": doi}
        work = works[0]
        authors = [
            authorship.get("author", {}).get("display_name", "")
            for authorship in work.get("authorships", [])
            if authorship.get("author", {}).get("display_name")
        ]
        primary_location = work.get("primary_location") or {}
        best_oa = work.get("best_oa_location") or {}
        source = primary_location.get("source") or {}
        open_access = work.get("open_access") or {}
        abstract_index = work.get("abstract_inverted_index")
        abstract = invert_abstract(abstract_index) if abstract_index else ""
        license_value = best_oa.get("license") or primary_location.get("license") or ""
        ids = work.get("ids") or {}
        locations = []
        pmc_url = ""
        pmcid = ""
        for location in work.get("locations") or []:
            landing_page_url = str(location.get("landing_page_url") or "")
            pdf_url = str(location.get("pdf_url") or "")
            location_source = location.get("source") or {}
            locations.append(
                {
                    "landing_page_url": landing_page_url,
                    "pdf_url": pdf_url,
                    "license": location.get("license") or "",
                    "is_oa": bool(location.get("is_oa")),
                    "version": location.get("version") or "",
                    "source": location_source.get("display_name") or "",
                }
            )
            match = re.search(r"/pmc/articles/(?:PMC)?(\d+)", landing_page_url, re.IGNORECASE)
            if match and not pmc_url:
                pmcid = f"PMC{match.group(1)}"
                pmc_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
        topics = [
            topic.get("display_name", "")
            for topic in work.get("topics", [])
            if topic.get("display_name")
        ]
        keywords = [
            keyword.get("display_name", "")
            for keyword in work.get("keywords", [])
            if keyword.get("display_name")
        ]
        return {
            "configured": True,
            "found": True,
            "doi": doi,
            "title": work.get("title") or "",
            "authors": authors,
            "journal": source.get("display_name") or "",
            "publication_date": work.get("publication_date") or "",
            "abstract": abstract or "",
            "oa_status": open_access.get("oa_status") or "unknown",
            "is_oa": bool(open_access.get("is_oa")),
            "oa_url": open_access.get("oa_url") or best_oa.get("landing_page_url") or best_oa.get("pdf_url") or "",
            "license": license_value or "unknown",
            "openalex_id": work.get("id") or "",
            "pmid": ids.get("pmid") or "",
            "pmcid": pmcid,
            "pmc_url": pmc_url,
            "oa_locations": locations,
            "work_type": work.get("type") or "",
            "publication_year": work.get("publication_year") or "",
            "source_type": source.get("type") or "",
            "topics": topics,
            "keywords": keywords,
        }

    @staticmethod
    def is_formally_published(metadata: dict[str, Any], today: date | None = None) -> bool:
        if not metadata.get("found"):
            return False
        publication_date = str(metadata.get("publication_date") or "")
        try:
            published = date.fromisoformat(publication_date)
        except ValueError:
            return False
        if published > (today or date.today()):
            return False
        work_type = str(metadata.get("work_type") or "").lower()
        if work_type not in {"article", "letter", "review", "peer-review"}:
            return False
        return bool(metadata.get("doi") and metadata.get("journal"))
