"""Minimal PyAlex adapter."""

from __future__ import annotations

from datetime import date
import logging
import re
import time
from typing import Any

from pyalex import Sources, Works, config, invert_abstract

LOGGER = logging.getLogger("wechat_news.openalex")
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_RETRIES = 2
DISCOVERY_QUERIES = (
    "near-surface wind speed",
    "near surface wind speed",
    "surface wind speed",
    "10 m wind",
    "10-m wind",
    "terrestrial stilling",
    "wind stilling",
    "wind recovery",
    "surface wind trend",
    "wind speed trend",
    "near-surface wind trend",
    "wind climatology",
    "wind power density",
    "wind resource",
    "wind energy climate",
    "extreme wind",
    "sfcWind",
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
JOURNAL_FIRST_TARGETS = (
    "Nature",
    "Nature Climate Change",
    "Nature Geoscience",
    "Nature Communications",
    "Communications Earth & Environment",
    "npj Climate and Atmospheric Science",
    "Science",
    "Science Advances",
    "PNAS",
    "Geophysical Research Letters",
    "Earth's Future",
    "AGU Advances",
    "Journal of Geophysical Research: Atmospheres",
    "The Innovation",
    "Atmospheric Chemistry and Physics",
    "Weather and Climate Dynamics",
    "Earth System Dynamics",
    "Climate Dynamics",
    "Environmental Research Letters",
)
DISCOVERY_SELECT = (
    "title,doi,publication_date,type,primary_location,abstract_inverted_index"
)
ALLOWED_JOURNALS = {
    "nature",
    "science",
    "science advances",
    "proceedings of the national academy of sciences",
    "proceedings of the national academy of sciences of the united states of america",
    "pnas",
    "geophysical research letters",
    "earths future",
    "agu advances",
    "journal of geophysical research atmospheres",
    "the innovation",
    "atmospheric chemistry and physics",
    "weather and climate dynamics",
    "earth system dynamics",
    "climate dynamics",
    "environmental research letters",
}
NATURE_PUBLISHER_MARKERS = (
    "nature portfolio",
    "nature publishing group",
    "springer nature",
)
AAAS_PUBLISHER_MARKERS = (
    "american association for the advancement of science",
    "aaas",
)


def normalize_doi(value: str | None) -> str:
    doi = (value or "").strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.lower().startswith(prefix):
            doi = doi[len(prefix):]
            break
    return doi.strip().lower()


def normalize_journal_name(value: str | None) -> str:
    text = (
        (value or "")
        .casefold()
        .replace("&", " and ")
        .replace("’", "'")
        .replace("'", "")
    )
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


JOURNAL_ALIASES = {
    "jgr atmospheres": "journal of geophysical research atmospheres",
    "journal of geophysical research atmospheres": "journal of geophysical research atmospheres",
    "pnas": "proceedings of the national academy of sciences",
    "proceedings of the national academy of sciences of the united states of america": (
        "proceedings of the national academy of sciences"
    ),
}
JOURNAL_DISPLAY_NAMES = {
    "journal of geophysical research atmospheres": "Journal of Geophysical Research: Atmospheres",
}


def canonical_journal_name(value: str | None) -> str:
    normalized = normalize_journal_name(value)
    return JOURNAL_ALIASES.get(normalized, normalized)


def journal_display_name(value: str | None) -> str:
    normalized = canonical_journal_name(value)
    return JOURNAL_DISPLAY_NAMES.get(normalized, str(value or "").strip())


def is_allowed_paper_journal(
    journal: str | None,
    publisher: str | None = None,
) -> bool:
    journal_name = canonical_journal_name(journal)
    publisher_name = normalize_journal_name(publisher)
    if journal_name in ALLOWED_JOURNALS:
        return True

    nature_publisher = any(
        marker in publisher_name for marker in NATURE_PUBLISHER_MARKERS
    )
    if nature_publisher and (
        journal_name.startswith("nature ")
        or journal_name.startswith("communications ")
        or journal_name.startswith("npj ")
    ):
        return True

    aaas_publisher = any(marker in publisher_name for marker in AAAS_PUBLISHER_MARKERS)
    return aaas_publisher and journal_name.startswith("science ")


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
        self._source_id_cache: dict[str, str | None] = {}
        self.last_journal_first_counts: dict[str, int] = {}
        self.last_journal_first_count = 0
        self.last_topic_count = 0

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    @staticmethod
    def _source_id(value: Any) -> str:
        text = str(value or "").strip().rstrip("/")
        return text.rsplit("/", 1)[-1] if text else ""

    def _resolve_source_id(self, journal: str) -> str | None:
        target = canonical_journal_name(journal)
        if target in self._source_id_cache:
            return self._source_id_cache[target]
        source_id: str | None = None
        if self.configured:
            for attempt in range(MAX_RETRIES + 1):
                try:
                    sources = Sources().search(journal).get(per_page=100)
                    for source in sources or []:
                        display = journal_display_name(str(source.get("display_name") or ""))
                        publisher = (
                            source.get("host_organization_name")
                            or source.get("host_organization_display_name")
                            or ""
                        )
                        if (
                            canonical_journal_name(display) == target
                            and is_allowed_paper_journal(display, publisher)
                        ):
                            candidate_id = self._source_id(source.get("id"))
                            if candidate_id:
                                source_id = candidate_id
                                break
                    break
                except Exception as exc:
                    if _retryable_error(exc) and attempt < MAX_RETRIES:
                        time.sleep(0.5 * (2**attempt))
                        continue
                    LOGGER.warning(
                        "OpenAlex source search unavailable journal=%r attempts=%s error=%s",
                        journal,
                        attempt + 1,
                        exc,
                    )
                    break
        self._source_id_cache[target] = source_id
        return source_id

    @staticmethod
    def _work_record(
        work: dict[str, Any],
        from_date: date,
        to_date: date,
        default_journal: str = "",
    ) -> dict[str, Any] | None:
        doi = normalize_doi(work.get("doi"))
        if not doi:
            return None
        publication_date = str(work.get("publication_date") or "")
        try:
            published = date.fromisoformat(publication_date)
        except ValueError:
            return None
        if published < from_date or published > to_date:
            return None
        work_type = str(work.get("type") or "").casefold()
        if work_type not in {"article", "research-article"}:
            return None
        primary_location = work.get("primary_location") or {}
        source = primary_location.get("source") or {}
        journal = journal_display_name(str(source.get("display_name") or default_journal))
        publisher = (
            source.get("host_organization_name")
            or source.get("host_organization_display_name")
            or ""
        )
        if not is_allowed_paper_journal(journal, publisher):
            return None
        abstract_index = work.get("abstract_inverted_index")
        return {
            "title": work.get("title") or "",
            "abstract": invert_abstract(abstract_index) if abstract_index else "",
            "publication_date": publication_date,
            "journal": journal,
            "publisher": publisher,
            "doi": doi,
            "type": work_type,
            "discovery_origin": "openalex",
        }

    def _discover_journal_works(
        self,
        journal: str,
        source_id: str,
        from_date: date,
        to_date: date,
    ) -> list[dict[str, Any]]:
        discovered: list[dict[str, Any]] = []
        cursor = "*"
        while cursor:
            works = None
            for attempt in range(MAX_RETRIES + 1):
                try:
                    works = (
                        Works()
                        .filter(
                            **{
                                "primary_location.source.id": source_id,
                                "from_publication_date": from_date.isoformat(),
                                "to_publication_date": to_date.isoformat(),
                            }
                        )
                        .select(DISCOVERY_SELECT)
                        .get(per_page=100, cursor=cursor)
                    )
                    break
                except Exception as exc:
                    if _retryable_error(exc) and attempt < MAX_RETRIES:
                        time.sleep(0.5 * (2**attempt))
                        continue
                    LOGGER.warning(
                        "OpenAlex journal discovery unavailable journal=%r attempts=%s error=%s",
                        journal,
                        attempt + 1,
                        exc,
                    )
                    works = []
                    cursor = ""
                    break
            if works is None:
                break
            for work in works or []:
                record = self._work_record(work, from_date, to_date, journal)
                if record:
                    discovered.append(record)
            metadata = getattr(works, "meta", {}) or {}
            next_cursor = metadata.get("next_cursor") if isinstance(metadata, dict) else None
            if not next_cursor or next_cursor == cursor:
                break
            cursor = str(next_cursor)
        return discovered

    def _discover_topic_works(
        self,
        from_date: date,
        to_date: date,
        *,
        per_query: int,
    ) -> list[dict[str, Any]]:
        discovered: list[dict[str, Any]] = []
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
                record = self._work_record(work, from_date, to_date)
                if record:
                    discovered.append(record)
        return discovered

    def discover_recent_papers(
        self,
        from_date: date,
        to_date: date,
        *,
        per_query: int = 15,
    ) -> list[dict[str, Any]]:
        self.last_journal_first_counts = {}
        self.last_journal_first_count = 0
        self.last_topic_count = 0
        if not self.configured:
            return []

        journal_papers: list[dict[str, Any]] = []
        for journal in JOURNAL_FIRST_TARGETS:
            source_id = self._resolve_source_id(journal)
            papers = (
                self._discover_journal_works(journal, source_id, from_date, to_date)
                if source_id
                else []
            )
            self.last_journal_first_counts[journal_display_name(journal)] = len(papers)
            journal_papers.extend(papers)
        topic_papers = self._discover_topic_works(from_date, to_date, per_query=per_query)
        self.last_journal_first_count = len(journal_papers)
        self.last_topic_count = len(topic_papers)

        discovered: list[dict[str, Any]] = []
        seen_dois: set[str] = set()
        for record in [*journal_papers, *topic_papers]:
            doi = normalize_doi(record.get("doi"))
            if not doi or doi in seen_dois:
                continue
            seen_dois.add(doi)
            discovered.append(record)
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
        publisher = (
            source.get("host_organization_name")
            or source.get("host_organization_display_name")
            or ""
        )
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
            "publisher": publisher,
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
