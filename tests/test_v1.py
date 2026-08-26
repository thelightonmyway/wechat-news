from __future__ import annotations

import asyncio
import copy
import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, time, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from bot.bridge import QQNewsBot
from bot.commands import CommandHandler, NEWS_USAGE, PAPER_USAGE
from db import Database
from images.policy import apply_policy, assess_image
from news.feeds import canonicalize_url, normalize_title
from news.pipeline import (
    PAPER_CONTENT,
    POPULAR_CONTENT,
    NewsPipeline,
    _apply_article_license_to_html_figures,
    _paper_wechat_cover,
    _prepare_paper_markdown,
    _select_article_images,
    content_type_for_date,
    deduplicate,
    deterministic_score,
    is_relevant_topic,
    prioritize_candidates,
)
from papers.doi import resolve_doi_landing_page
from papers.oa_mirror import resolve_oa_html_mirror
from papers.openalex import OpenAlexAdapter
from papers.pdf_figures import extract_pdf_figures
from publisher.wechat import _paper_draft_title, _selected_cover_path, format_markdown
from scheduler import should_run_startup_catchup
from settings import bind_qq_target_openid, load_settings
from writer.llm import (
    _normalize_article_markdown,
    generate_image_captions,
    generate_image_search_keywords,
    select_top_ten,
)


class V1Tests(unittest.TestCase):
    def test_sqlite_schema_and_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "news.db")
            article_id = db.upsert_article(
                {
                    "source": "test",
                    "url": "https://example.test/story",
                    "canonical_url": "https://example.test/story",
                    "title": "A study",
                    "summary": "research",
                    "published_at": "2026-08-24T00:00:00+00:00",
                    "doi": "",
                    "journal": "",
                    "word_count": 700,
                    "status": "extracted",
                    "discovered_at": "2026-08-24T00:00:00+00:00",
                }
            )
            db.replace_candidates("2026-08-24", [{"article_id": article_id, "score": 10, "title_cn": "研究"}])
            self.assertEqual(db.get_candidate("2026-08-24", 1)["title_cn"], "研究")

    def test_url_and_title_normalization(self):
        self.assertEqual(
            canonicalize_url("HTTPS://Example.COM/a/?utm_source=x&keep=1#frag"),
            "https://example.com/a?keep=1",
        )
        self.assertEqual(normalize_title("A  New: Study!"), "a new study")

    def test_dedup_threshold(self):
        items = [
            {"title": "A new study of cells", "normalized_title": "a new study of cells", "canonical_url": "https://a/1", "doi": "", "published_at": "2"},
            {"title": "A new study of cells!", "normalized_title": "a new study of cells", "canonical_url": "https://b/2", "doi": "", "published_at": "1"},
        ]
        self.assertEqual(len(deduplicate(items)), 1)

    def test_deterministic_score(self):
        score = deterministic_score(
            {
                "title": "Researchers published a new study",
                "summary": "scientists report research",
                "source": "Nature News",
                "published_at": "2026-08-24T00:00:00+00:00",
                "doi": "10.1000/test",
                "word_count": 900,
            }
        )
        self.assertGreater(score, 30)

    def test_weekly_content_types(self):
        self.assertEqual(content_type_for_date("2026-08-24"), POPULAR_CONTENT)  # Monday
        self.assertEqual(content_type_for_date("2026-08-26"), PAPER_CONTENT)  # Wednesday
        self.assertEqual(content_type_for_date("2026-08-28"), POPULAR_CONTENT)  # Friday

    def test_popular_and_paper_candidates_share_date_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "typed-candidates.db")
            popular_id = db.upsert_article(
                {
                    "source": "Nature News",
                    "url": "https://example.test/news",
                    "canonical_url": "https://example.test/news",
                    "title": "Popular wind news",
                    "summary": "Near-surface wind observations",
                    "published_at": "2026-08-25T00:00:00+00:00",
                    "doi": "",
                    "journal": "",
                    "word_count": 800,
                    "status": "extracted",
                    "discovered_at": "2026-08-25T00:00:00+00:00",
                }
            )
            paper_id = db.upsert_article(
                {
                    "source": "Journal of Climate",
                    "url": "https://example.test/paper",
                    "canonical_url": "https://example.test/paper",
                    "title": "Published circulation paper",
                    "summary": "Atmospheric circulation attribution",
                    "published_at": "2026-08-25T00:00:00+00:00",
                    "doi": "10.1000/typed-paper",
                    "journal": "Journal of Climate",
                    "word_count": 900,
                    "status": "published_paper",
                    "discovered_at": "2026-08-25T00:00:00+00:00",
                }
            )
            db.replace_candidates(
                "2026-08-25",
                [{"article_id": popular_id, "score": 20, "title_cn": "科普新闻"}],
                POPULAR_CONTENT,
            )
            db.replace_candidates(
                "2026-08-25",
                [{"article_id": paper_id, "score": 30, "title_cn": "正式论文"}],
                PAPER_CONTENT,
            )
            db.set_daily_run(
                "2026-08-25",
                content_type=POPULAR_CONTENT,
                candidate_count=1,
                status="success",
            )
            db.set_daily_run(
                "2026-08-25",
                content_type=PAPER_CONTENT,
                candidate_count=1,
                status="success",
            )

            news = db.get_candidate("2026-08-25", 1, POPULAR_CONTENT)
            paper = db.get_candidate("2026-08-25", 1, PAPER_CONTENT)
            self.assertEqual(news["id"], popular_id)
            self.assertEqual(news["content_type"], POPULAR_CONTENT)
            self.assertEqual(paper["id"], paper_id)
            self.assertEqual(paper["content_type"], PAPER_CONTENT)

    def test_news_and_paper_command_routes_stay_isolated(self):
        class FakePipeline:
            def __init__(self):
                self.calls = []

            async def get_or_refresh(self, date=None, content_type=None):
                self.calls.append(("list", content_type))
                return [
                    {
                        "rank": 1,
                        "content_type": content_type,
                        "title": f"{content_type} title",
                        "source": "test",
                        "published_at": "2026-08-25T00:00:00+00:00",
                    }
                ]

            def format_news(self, candidates):
                return f"list:{candidates[0]['content_type']}"

            async def paper_details(self, rank, date=None, content_type=None):
                self.calls.append(("detail", rank, content_type))
                return {"rank": rank, "content_type": content_type}

            def format_paper(self, dossier):
                return f"detail:{dossier['content_type']}:{dossier['rank']}"

            async def generate(self, rank, date=None, content_type=None):
                self.calls.append(("generate", rank, content_type))
                return {"markdown_path": Path(f"/{content_type}-{rank}.md")}

        async def check():
            settings = replace(
                load_settings(),
                model_base_url="https://model.example/v1",
                model_api_key="test-key",
                model_name="test-model",
            )
            pipeline = FakePipeline()
            handler = CommandHandler(settings, pipeline)

            self.assertEqual(await handler.handle("/news"), "list:popular")
            self.assertEqual(await handler.handle("/news 1"), "detail:popular:1")
            self.assertEqual(await handler.handle("/papers"), "list:paper")
            self.assertEqual(await handler.handle("/paper 1"), "detail:paper:1")
            self.assertIn("/popular-1.md", await handler.handle("/news 1 generate"))
            self.assertIn("/paper-1.md", await handler.handle("/paper 1 generate"))

            await handler.handle("/news")
            await handler.handle("/papers")
            self.assertEqual(await handler.handle("/news 1"), "detail:popular:1")
            self.assertEqual(await handler.handle("/paper 1"), "detail:paper:1")
            self.assertEqual(await handler.handle("/paper"), PAPER_USAGE)
            self.assertEqual(await handler.handle("/news abc"), NEWS_USAGE)
            self.assertEqual(await handler.handle("/paper abc"), PAPER_USAGE)

        asyncio.run(check())

    def test_strict_research_topic_filter(self):
        allowed = [
            {"title": "Near-surface wind speed recovery across station observations", "summary": "A terrestrial stilling assessment"},
            {"title": "Jet stream shifts alter atmospheric circulation", "summary": "Links to the North Atlantic Oscillation"},
            {"title": "Detection and attribution of climate change", "summary": "CMIP6 large ensembles separate anthropogenic forcing and internal variability"},
            {"title": "Southern Ocean air-sea interaction", "summary": "Ocean circulation and wind work change ocean energy input"},
            {"title": "Antarctic sea ice and ozone recovery", "summary": "Southern Hemisphere westerlies respond to polar climate change"},
            {"title": "Compound extremes intensify", "summary": "Extreme precipitation, heatwave and drought mechanisms"},
            {"title": "Planetary boundary layer responds to land cover", "summary": "Surface roughness and vegetation feedback alter land-atmosphere interaction"},
            {"title": "ERA5 reanalysis evaluated against station observations", "summary": "Satellite observations improve climate observational coverage"},
            {"title": "Moisture transport controls rainfall", "summary": "Moisture convergence and vertical motion explain precipitation mechanisms"},
        ]
        for item in allowed:
            self.assertTrue(is_relevant_topic(item), item["title"])

        rejected = [
            {"title": "Trump Shuns E.V.s and Wind Power, But He’s Pouring Billions Into Batteries", "summary": "Government policy funds battery manufacturing for AI and defense"},
            {"title": "Alzheimer risk gene changes brain cells", "summary": "Medical neuroscience study"},
            {"title": "Renewable energy investment rises", "summary": "Economics of solar and electric vehicles"},
            {"title": "AI improves regional climate model", "summary": "WRF numerical simulation and parameterization"},
            {"title": "A black hole tears apart a star", "summary": "Astronomy discovery"},
            {"title": "Arctic shark migration reveals new feeding grounds", "summary": "Marine biology tracks animal behavior"},
            {"title": "Renewable energy projects expand", "summary": "Wind energy capacity attracts private investment"},
            {"title": "Solar panels respond to climate change", "summary": "Manufacturers announce a new commercial product"},
            {"title": "A climate window for architecture", "summary": "The window design improves a building"},
        ]
        for item in rejected:
            self.assertFalse(is_relevant_topic(item), item["title"])

    def test_primary_sources_fill_all_slots_before_secondary(self):
        primary = [
            {"source": "Guardian Science", "title": f"Primary {index}", "score": index}
            for index in range(12)
        ]
        secondary = [
            {"source": "Nature Climate Change", "title": "Secondary", "score": 999}
        ]
        selected = prioritize_candidates(primary + secondary)
        self.assertEqual(len(selected), 10)
        self.assertTrue(all(item["source"] == "Guardian Science" for item in selected))

    def test_secondary_sources_supplement_primary_shortfall(self):
        primary = [
            {"source": "Nature News", "title": f"Primary {index}", "score": index}
            for index in range(3)
        ]
        secondary = [
            {
                "source": "Eos / AGU",
                "title": f"Secondary {index}",
                "score": index,
            }
            for index in range(10)
        ]
        selected = prioritize_candidates(primary + secondary)
        self.assertEqual(len(selected), 10)
        self.assertTrue(all(item["source"] == "Nature News" for item in selected[:3]))
        self.assertEqual(
            [item["score"] for item in selected[3:]],
            [9, 8, 7, 6, 5, 4, 3],
        )

    def test_48h_expands_to_7d_without_filling_irrelevant_items(self):
        def item(index, title, summary):
            return {
                "source": "test",
                "url": f"https://example.test/{index}",
                "canonical_url": f"https://example.test/{index}",
                "title": title,
                "normalized_title": normalize_title(title),
                "summary": summary,
                "published_at": "2026-08-24T00:00:00+00:00",
                "doi": "",
                "journal": "",
                "word_count": 0,
                "status": "discovered",
                "discovered_at": "2026-08-24T00:00:00+00:00",
            }

        recent = [
            item(1, "Near-surface wind speed recovery", "Station observations show terrestrial stilling reversal"),
            item(2, "Jet stream and atmospheric circulation", "The NAO shifts extreme wind patterns"),
            item(3, "Wind Power and Batteries", "Political investment in EV battery factories"),
        ]
        expanded = recent + [
            item(4, "CMIP6 detection and attribution", "Climate change and anthropogenic forcing"),
            item(5, "Southern Ocean air-sea interaction", "Wind work affects ocean circulation"),
            item(6, "Antarctic sea ice change", "Ozone recovery and Southern Hemisphere westerlies"),
            item(7, "Alzheimer treatment trial", "Medical neuroscience disease research"),
            item(8, "Solar investment outlook", "Generic renewable energy economics"),
        ]
        calls = []

        def fake_fetch(_path, hours):
            calls.append(hours)
            values = recent if hours == 48 else expanded
            return copy.deepcopy(values), [], {"test": len(values)}

        async def check():
            with tempfile.TemporaryDirectory() as tmp:
                settings = replace(
                    load_settings(),
                    database_path=Path(tmp) / "window.db",
                    model_base_url="",
                    model_api_key="",
                    model_name="",
                    openalex_api_key="",
                )
                pipeline = NewsPipeline(settings)

                async def fake_extract(items):
                    return [
                        {**value, "word_count": 800, "text": "research", "images": []}
                        for value in items
                    ]

                pipeline._extract_shortlist = fake_extract
                with patch("news.pipeline.fetch_all_feeds", side_effect=fake_fetch):
                    candidates = await pipeline.refresh("2026-08-24", POPULAR_CONTENT)
                self.assertEqual(calls, [48, 168])
                self.assertEqual(len(candidates), 5)
                self.assertTrue(all(is_relevant_topic(value) for value in candidates))
                self.assertFalse(any("Batter" in value["title"] for value in candidates))
                self.assertFalse(any("Alzheimer" in value["title"] for value in candidates))
                self.assertFalse(any("Solar" in value["title"] for value in candidates))

        asyncio.run(check())

    def test_refresh_excludes_drafted_url_and_doi_but_keeps_failed(self):
        def item(index, title, *, doi="", canonical_url=None):
            url = canonical_url or f"https://example.test/{index}"
            return {
                "source": "Nature News",
                "url": url,
                "canonical_url": url,
                "title": title,
                "normalized_title": normalize_title(title),
                "summary": "Station observations document near-surface wind speed change",
                "published_at": "2026-08-24T00:00:00+00:00",
                "doi": doi,
                "journal": "",
                "word_count": 0,
                "status": "discovered",
                "discovered_at": "2026-08-24T00:00:00+00:00",
            }

        async def check():
            with tempfile.TemporaryDirectory() as tmp:
                settings = replace(
                    load_settings(),
                    database_path=Path(tmp) / "published.db",
                    model_base_url="",
                    model_api_key="",
                    model_name="",
                    openalex_api_key="",
                )
                pipeline = NewsPipeline(settings)

                used_url = item(
                    1,
                    "Published surface wind observations",
                    canonical_url="https://example.test/published-url",
                )
                used_url_id = pipeline.db.upsert_article(used_url)
                pipeline.db.save_publish_history(used_url_id, "drafted", "media-url")

                used_doi = item(
                    2,
                    "Published atmospheric circulation attribution",
                    doi="10.1000/shared-paper",
                )
                used_doi_id = pipeline.db.upsert_article(used_doi)
                pipeline.db.save_publish_history(used_doi_id, "drafted", "media-doi")

                failed = item(
                    3,
                    "Retryable ERA5 wind analysis",
                    doi="10.1000/failed-paper",
                )
                failed_id = pipeline.db.upsert_article(failed)
                pipeline.db.save_publish_history(
                    failed_id,
                    "failed",
                    error="draft creation failed",
                )

                incoming = [
                    item(
                        4,
                        "The already published URL returns",
                        canonical_url="https://example.test/published-url",
                    ),
                    item(
                        5,
                        "Another source reports the same DOI",
                        doi="10.1000/shared-paper",
                        canonical_url="https://other.test/shared-paper",
                    ),
                    item(
                        6,
                        "Retryable ERA5 wind analysis",
                        doi="10.1000/failed-paper",
                    ),
                    item(7, "Fresh boundary layer wind observations"),
                ]

                async def fake_extract(values):
                    return [
                        {**value, "word_count": 800, "text": "research", "images": []}
                        for value in values
                    ]

                pipeline._extract_shortlist = fake_extract
                with patch(
                    "news.pipeline.fetch_all_feeds",
                    return_value=(copy.deepcopy(incoming), [], {"Nature News": 4}),
                ):
                    candidates = await pipeline.refresh("2026-08-24", POPULAR_CONTENT)

                self.assertEqual(len(candidates), 2)
                self.assertEqual(
                    {candidate["doi"] for candidate in candidates},
                    {"10.1000/failed-paper", ""},
                )
                self.assertFalse(
                    any(
                        candidate["canonical_url"]
                        == "https://example.test/published-url"
                        for candidate in candidates
                    )
                )
                self.assertFalse(
                    any(candidate["doi"] == "10.1000/shared-paper" for candidate in candidates)
                )

        asyncio.run(check())

    def test_get_or_refresh_rebuilds_after_candidate_is_drafted(self):
        async def check():
            with tempfile.TemporaryDirectory() as tmp:
                settings = replace(
                    load_settings(),
                    database_path=Path(tmp) / "news-command.db",
                )
                pipeline = NewsPipeline(settings)
                article_id = pipeline.db.upsert_article(
                    {
                        "source": "Nature News",
                        "url": "https://example.test/used",
                        "canonical_url": "https://example.test/used",
                        "title": "Used near-surface wind study",
                        "summary": "Station observations",
                        "published_at": "2026-08-24T00:00:00+00:00",
                        "doi": "10.1000/used-command",
                        "journal": "",
                        "word_count": 800,
                        "status": "extracted",
                        "discovered_at": "2026-08-24T00:00:00+00:00",
                    }
                )
                pipeline.db.replace_candidates(
                    "2026-08-24",
                    [{"article_id": article_id, "score": 10, "title_cn": ""}],
                    POPULAR_CONTENT,
                )
                pipeline.db.set_daily_run(
                    "2026-08-24",
                    content_type=POPULAR_CONTENT,
                    status="success",
                )
                pipeline.db.save_publish_history(article_id, "drafted", "media-used")
                calls = []

                async def fake_refresh(date, content_type):
                    calls.append((date, content_type))
                    return []

                pipeline.refresh = fake_refresh
                candidates = await pipeline.get_or_refresh(
                    "2026-08-24",
                    POPULAR_CONTENT,
                )
                self.assertEqual(candidates, [])
                self.assertEqual(calls, [("2026-08-24", POPULAR_CONTENT)])

        asyncio.run(check())

    def test_openalex_504_retries_twice_and_generate_continues(self):
        class GatewayTimeout(Exception):
            def __init__(self):
                super().__init__("504 Gateway Timeout")
                self.response = SimpleNamespace(status_code=504)

        async def check():
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                settings = replace(
                    load_settings(),
                    database_path=root / "openalex-timeout.db",
                    openalex_api_key="test-openalex-key",
                )
                pipeline = NewsPipeline(settings)
                article_id = pipeline.db.upsert_article(
                    {
                        "source": "Nature News",
                        "url": "https://example.test/openalex-timeout",
                        "canonical_url": "https://example.test/openalex-timeout",
                        "title": "Near-surface wind observations",
                        "summary": "Station data show a recent wind-speed change.",
                        "published_at": "2026-08-25T00:00:00+00:00",
                        "doi": "10.1000/openalex-timeout",
                        "journal": "",
                        "word_count": 800,
                        "status": "extracted",
                        "discovered_at": "2026-08-25T00:00:00+00:00",
                    }
                )
                pipeline.db.replace_candidates(
                    "2026-08-25",
                    [{"article_id": article_id, "score": 10, "title_cn": "近地面风观测"}],
                    POPULAR_CONTENT,
                )
                pipeline.db.set_daily_run(
                    "2026-08-25",
                    content_type=POPULAR_CONTENT,
                    status="success",
                )
                output_dir = root / "article"

                def fake_extract(value):
                    return {
                        **value,
                        "text": "Existing extracted article content.",
                        "images": [],
                        "authors": [],
                    }

                def fake_generate_markdown(dossier, _settings, destination):
                    destination.mkdir(parents=True, exist_ok=True)
                    markdown = destination / "article.md"
                    metadata = destination / "metadata.json"
                    markdown.write_text(dossier["text"], encoding="utf-8")
                    metadata.write_text("{}", encoding="utf-8")
                    return markdown, metadata

                works = MagicMock()
                works.filter.return_value.get.side_effect = GatewayTimeout()
                with (
                    patch("papers.openalex.Works", return_value=works),
                    patch("papers.openalex.time.sleep") as sleep,
                    patch("news.pipeline.extract_article", side_effect=fake_extract),
                    patch("news.pipeline.article_output_dir", return_value=output_dir),
                    patch("news.pipeline.generate_image_search_keywords", return_value=[]),
                    patch("news.pipeline.search_public_images", return_value=[]),
                    patch("news.pipeline.generate_article_markdown", side_effect=fake_generate_markdown),
                    patch("news.pipeline.download_publishable_images", return_value=[]),
                ):
                    generated = await pipeline.generate(1, "2026-08-25")

                self.assertTrue(generated["markdown_path"].is_file())
                self.assertEqual(works.filter.return_value.get.call_count, 3)
                self.assertEqual(sleep.call_count, 2)
                dossier = generated["dossier"]
                self.assertFalse(dossier["openalex"]["found"])
                self.assertEqual(dossier["title"], "Near-surface wind observations")
                self.assertEqual(
                    dossier["summary"],
                    "Station data show a recent wind-speed change.",
                )
                self.assertEqual(dossier["source"], "Nature News")
                self.assertEqual(dossier["url"], "https://example.test/openalex-timeout")

        asyncio.run(check())

    def test_openalex_success_still_supplements_metadata(self):
        async def check():
            with tempfile.TemporaryDirectory() as tmp:
                settings = replace(
                    load_settings(),
                    database_path=Path(tmp) / "openalex-success.db",
                    openalex_api_key="test-openalex-key",
                )
                pipeline = NewsPipeline(settings)
                article_id = pipeline.db.upsert_article(
                    {
                        "source": "Nature News",
                        "url": "https://example.test/openalex-success",
                        "canonical_url": "https://example.test/openalex-success",
                        "title": "Atmospheric circulation study",
                        "summary": "",
                        "published_at": "2026-08-25T00:00:00+00:00",
                        "doi": "10.1000/openalex-success",
                        "journal": "",
                        "word_count": 800,
                        "status": "extracted",
                        "discovered_at": "2026-08-25T00:00:00+00:00",
                    }
                )
                pipeline.db.replace_candidates(
                    "2026-08-25",
                    [{"article_id": article_id, "score": 10, "title_cn": ""}],
                    POPULAR_CONTENT,
                )
                pipeline.db.set_daily_run(
                    "2026-08-25",
                    content_type=POPULAR_CONTENT,
                    status="success",
                )
                metadata = {
                    "configured": True,
                    "found": True,
                    "doi": "10.1000/openalex-success",
                    "authors": ["Researcher One"],
                    "journal": "Journal of Climate",
                    "abstract": "OpenAlex abstract supplement.",
                }
                pipeline.openalex.lookup_doi = MagicMock(return_value=metadata)

                def fake_extract(value):
                    return {**value, "text": "Article text", "images": [], "authors": []}

                with patch("news.pipeline.extract_article", side_effect=fake_extract):
                    dossier = await pipeline.paper_details(1, "2026-08-25")

                self.assertEqual(dossier["authors"], ["Researcher One"])
                self.assertEqual(dossier["journal"], "Journal of Climate")
                self.assertEqual(dossier["summary"], "OpenAlex abstract supplement.")

        asyncio.run(check())

    def test_doi_resolver_keeps_final_html_url_even_when_publisher_blocks_fetch(self):
        response = MagicMock()
        response.url = "https://www.science.org/doi/10.1126/sciadv.adn9389"
        response.status_code = 403
        response.headers = {"content-type": "text/html; charset=UTF-8"}
        response.is_success = False
        response.is_error = True
        client = MagicMock()
        client.__enter__.return_value = client
        client.get.return_value = response
        with patch("papers.doi.httpx.Client", return_value=client):
            result = resolve_doi_landing_page("10.1126/sciadv.adn9389")

        self.assertTrue(result["resolved"])
        self.assertFalse(result["accessible"])
        self.assertEqual(
            result["landing_url"],
            "https://www.science.org/doi/10.1126/sciadv.adn9389",
        )

    def test_paper_uses_resolved_doi_landing_before_html_extraction(self):
        async def check():
            with tempfile.TemporaryDirectory() as tmp:
                settings = replace(
                    load_settings(),
                    database_path=Path(tmp) / "doi-landing.db",
                    openalex_api_key="test-openalex-key",
                )
                pipeline = NewsPipeline(settings)
                article_id = pipeline.db.upsert_article(
                    {
                        "source": "Eos / AGU",
                        "url": "https://example.test/news",
                        "canonical_url": "https://example.test/news",
                        "title": "California drought article",
                        "summary": "Aquifer observations.",
                        "published_at": "2026-08-25T00:00:00+00:00",
                        "doi": "10.1126/sciadv.adn9389",
                        "journal": "",
                        "word_count": 800,
                        "status": "extracted",
                        "discovered_at": "2026-08-25T00:00:00+00:00",
                    }
                )
                pipeline.db.replace_candidates(
                    "2026-08-25",
                    [{"article_id": article_id, "score": 10, "title_cn": "干旱研究"}],
                    PAPER_CONTENT,
                )
                pipeline.openalex.lookup_doi = MagicMock(
                    return_value={
                        "configured": True,
                        "found": True,
                        "doi": "10.1126/sciadv.adn9389",
                        "title": "Anthropogenic warming and western droughts",
                        "journal": "Science Advances",
                        "abstract": "Paper abstract.",
                        "publication_date": "2024-11-06",
                        "oa_url": "https://doi.org/10.1126/sciadv.adn9389",
                        "license": "cc-by-nc",
                        "authors": ["Researcher One"],
                    }
                )
                landing_url = "https://www.science.org/doi/10.1126/sciadv.adn9389"
                calls = []

                def fake_extract(value):
                    calls.append(value["url"])
                    return {
                        **value,
                        "text": "Extracted text.",
                        "images": [],
                        "authors": [],
                    }

                with (
                    patch(
                        "news.pipeline.resolve_doi_landing_page",
                        return_value={
                            "resolved": True,
                            "landing_url": landing_url,
                            "accessible": True,
                            "status_code": 200,
                            "error": "",
                        },
                    ),
                    patch("news.pipeline.extract_article", side_effect=fake_extract),
                ):
                    dossier = await pipeline.paper_details(
                        1,
                        "2026-08-25",
                        PAPER_CONTENT,
                    )

                self.assertEqual(calls[1], landing_url)
                self.assertEqual(dossier["url"], landing_url)
                self.assertEqual(dossier["canonical_url"], landing_url)
                self.assertEqual(dossier["doi_landing"]["landing_url"], landing_url)
                self.assertEqual(dossier["journal"], "Science Advances")

        asyncio.run(check())

    def test_oa_mirror_prefers_accessible_pmc_url(self):
        response = MagicMock()
        response.url = "https://pmc.ncbi.nlm.nih.gov/articles/PMC11540010/"
        response.status_code = 200
        response.headers = {"content-type": "text/html; charset=UTF-8"}
        response.is_success = True
        client = MagicMock()
        client.__enter__.return_value = client
        client.get.return_value = response
        metadata = {
            "pmcid": "PMC11540010",
            "pmc_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11540010/",
            "oa_locations": [],
        }
        with patch("papers.oa_mirror.httpx.Client", return_value=client):
            result = resolve_oa_html_mirror(
                metadata,
                "https://www.science.org/doi/10.1126/sciadv.adn9389",
            )

        self.assertTrue(result["found"])
        self.assertEqual(
            result["url"],
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC11540010/",
        )

    def test_publisher_access_failure_uses_oa_mirror_html_figures(self):
        async def check():
            with tempfile.TemporaryDirectory() as tmp:
                settings = replace(
                    load_settings(),
                    database_path=Path(tmp) / "oa-mirror.db",
                    openalex_api_key="test-openalex-key",
                )
                pipeline = NewsPipeline(settings)
                article_id = pipeline.db.upsert_article(
                    {
                        "source": "Eos / AGU",
                        "url": "https://example.test/news",
                        "canonical_url": "https://example.test/news",
                        "title": "California drought article",
                        "summary": "Aquifer observations.",
                        "published_at": "2026-08-25T00:00:00+00:00",
                        "doi": "10.1126/sciadv.adn9389",
                        "journal": "",
                        "word_count": 800,
                        "status": "extracted",
                        "discovered_at": "2026-08-25T00:00:00+00:00",
                    }
                )
                pipeline.db.replace_candidates(
                    "2026-08-25",
                    [{"article_id": article_id, "score": 10, "title_cn": "干旱研究"}],
                    PAPER_CONTENT,
                )
                publisher_url = "https://www.science.org/doi/10.1126/sciadv.adn9389"
                mirror_url = "https://pmc.ncbi.nlm.nih.gov/articles/PMC11540010/"
                pipeline.openalex.lookup_doi = MagicMock(
                    return_value={
                        "configured": True,
                        "found": True,
                        "doi": "10.1126/sciadv.adn9389",
                        "title": "Anthropogenic warming and western droughts",
                        "journal": "Science Advances",
                        "abstract": "Paper abstract.",
                        "publication_date": "2024-11-06",
                        "oa_url": "https://doi.org/10.1126/sciadv.adn9389",
                        "license": "cc-by-nc",
                        "authors": ["Researcher One"],
                        "pmcid": "PMC11540010",
                        "pmc_url": mirror_url,
                        "oa_locations": [],
                    }
                )
                calls = []

                def fake_extract(value):
                    calls.append(value["url"])
                    if value["url"] == publisher_url:
                        return {
                            **value,
                            "text": "",
                            "images": [],
                            "authors": [],
                            "extraction_error": "fetch failed: HTTP 403 Forbidden",
                        }
                    if value["url"] == mirror_url:
                        return {
                            **value,
                            "text": "Full mirrored paper text.",
                            "images": [
                                {
                                    "url": "https://cdn.example.test/figure-1.jpg",
                                    "local_path": "",
                                    "caption": "Fig. 1. Drought severity time series.",
                                    "credit": "",
                                    "license": "",
                                    "image_source": "html_figure",
                                    "image_role": "figure",
                                    "metadata_title": "Fig. 1",
                                    "publishable": False,
                                    "reason": "unknown or non-reusable license",
                                }
                            ],
                            "authors": [],
                            "extraction_error": "",
                        }
                    return {
                        **value,
                        "text": "News text.",
                        "images": [],
                        "authors": [],
                        "extraction_error": "",
                    }

                with (
                    patch(
                        "news.pipeline.resolve_doi_landing_page",
                        return_value={
                            "resolved": True,
                            "landing_url": publisher_url,
                            "accessible": False,
                            "status_code": 403,
                            "error": "HTTP 403",
                        },
                    ),
                    patch(
                        "news.pipeline.resolve_oa_html_mirror",
                        return_value={
                            "found": True,
                            "url": mirror_url,
                            "source": "PubMed Central",
                            "status_code": 200,
                            "error": "",
                        },
                    ),
                    patch("news.pipeline.extract_article", side_effect=fake_extract),
                ):
                    dossier = await pipeline.paper_details(
                        1,
                        "2026-08-25",
                        PAPER_CONTENT,
                    )

                self.assertEqual(calls[1:], [publisher_url, mirror_url])
                self.assertEqual(dossier["publisher_url"], publisher_url)
                self.assertEqual(dossier["oa_mirror_url"], mirror_url)
                self.assertEqual(dossier["actual_image_source"], "oa_mirror")
                self.assertEqual(len(dossier["images"]), 1)
                self.assertTrue(dossier["images"][0]["publishable"])
                self.assertEqual(dossier["images"][0]["reason"], "CC BY-NC")

        asyncio.run(check())

    def test_formally_published_openalex_policy(self):
        metadata = {
            "found": True,
            "doi": "10.1000/test",
            "journal": "Journal of Climate",
            "publication_date": "2026-08-20",
            "work_type": "article",
        }
        self.assertTrue(OpenAlexAdapter.is_formally_published(metadata))
        self.assertFalse(OpenAlexAdapter.is_formally_published({**metadata, "work_type": "preprint"}))
        self.assertFalse(OpenAlexAdapter.is_formally_published({**metadata, "doi": ""}))

    def test_weekly_startup_catchup(self):
        scheduled = time(7, 0)
        monday = datetime(2026, 8, 24, 8, 0, tzinfo=timezone.utc)
        tuesday = datetime(2026, 8, 25, 8, 0, tzinfo=timezone.utc)
        wednesday = datetime(2026, 8, 26, 8, 0, tzinfo=timezone.utc)
        friday = datetime(2026, 8, 28, 8, 0, tzinfo=timezone.utc)
        self.assertTrue(should_run_startup_catchup(monday, scheduled))
        self.assertFalse(should_run_startup_catchup(tuesday, scheduled))
        self.assertTrue(should_run_startup_catchup(wednesday, scheduled))
        self.assertTrue(should_run_startup_catchup(friday, scheduled))
        self.assertFalse(should_run_startup_catchup(friday, scheduled, "already"))

    def test_article_markdown_keeps_candidate_title_and_removes_empty_sections(self):
        markdown = _normalize_article_markdown(
            "# Model title\n\n## 导语\n\nIntro paragraph.\n\n"
            "## 研究内容\n\nActual body.\n\n## 科学意义\n\nMeaning.\n\n"
            "## 简报中的其他科研进展\n\nDolphin and slavery stories.",
            "每日简报：“热得睡不着”有害健康",
        )
        self.assertTrue(markdown.startswith("# 每日简报：“热得睡不着”有害健康"))
        self.assertNotIn("# Model title", markdown)
        self.assertNotIn("## 导语", markdown)
        self.assertNotIn("## 研究内容", markdown)
        self.assertIn("> Intro paragraph.", markdown)
        self.assertIn("## 科学意义", markdown)
        self.assertNotIn("简报中的其他科研进展", markdown)
        self.assertNotIn("Dolphin and slavery stories", markdown)

    def test_body_image_captions_are_independent_and_batched(self):
        settings = replace(
            load_settings(),
            model_base_url="https://model.example/v1",
            model_api_key="test-key",
            model_name="test-model",
        )
        images = [
            {
                "metadata_title": "Summer Nighttime Urban-Rural Temperature Difference",
                "caption": "2013 urban heat island temperature difference",
                "provider": "Wikimedia Commons",
            },
            {
                "metadata_title": "Average Summer Nighttime Minimum Surface Temperature",
                "caption": "2013 nighttime minimum land surface temperature",
                "provider": "Wikimedia Commons",
            },
        ]
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "items": [
                                    {
                                        "index": 1,
                                        "caption_cn": "2013年夏季夜间城乡地表温差分布，反映城市热岛强度。",
                                    },
                                    {
                                        "index": 2,
                                        "caption_cn": "2013年夏季夜间最低地表温度分布，反映背景热环境。",
                                    },
                                ]
                            },
                            ensure_ascii=False,
                        )
                    )
                )
            ]
        )
        client = MagicMock()
        client.chat.completions.create.return_value = response
        with patch("writer.llm.OpenAI", return_value=client):
            captions = generate_image_captions(images, settings)

        self.assertEqual(client.chat.completions.create.call_count, 1)
        self.assertEqual(len(captions), 2)
        self.assertNotEqual(captions[0], captions[1])
        self.assertIn("城乡地表温差", captions[0])
        self.assertIn("最低地表温度", captions[1])

    def test_news_body_images_are_limited_and_deduplicated(self):
        def image(index, title):
            return {
                "url": f"https://example.test/{index}.jpg",
                "local_path": f"/tmp/{index}.jpg",
                "metadata_title": title,
                "caption": title,
            }

        images = [
            image(1, "Summer Nighttime Minimum Land Surface Temperature 2013 South America"),
            image(2, "Summer Nighttime Minimum Land Surface Temperature 2013 Oceania"),
            image(3, "Summer Nighttime Urban-Rural Temperature Difference 2013 Global"),
            image(4, "Summer Nighttime Minimum Land Surface Temperature 2013 Europe"),
        ]
        cover, body, redundant_count = _select_article_images(images, POPULAR_CONTENT)
        self.assertEqual(cover, images[0])
        self.assertEqual(len(body), 2)
        self.assertEqual(body[0], images[0])
        self.assertEqual(body[1], images[2])
        self.assertGreaterEqual(redundant_count, 1)

    def test_daily_briefing_image_keywords_use_only_title_related_context(self):
        settings = replace(
            load_settings(),
            model_base_url="https://model.example/v1",
            model_api_key="test-key",
            model_name="test-model",
        )
        dossier = {
            "title": "Daily briefing: ‘Too hot to sleep’ is harmful to your health",
            "summary": (
                "Hot bedrooms and high nighttime temperatures can disrupt sleep and health. "
                "Dolphin calves learn hunting techniques from their mothers. "
                "A separate study examines slavery and US health disparities."
            ),
        }
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "keywords": [
                                    "nighttime heat",
                                    "hot bedroom sleep",
                                    "high temperature sleep",
                                ]
                            }
                        )
                    )
                )
            ]
        )
        client = MagicMock()
        client.chat.completions.create.return_value = response
        with patch("writer.llm.OpenAI", return_value=client):
            keywords = generate_image_search_keywords(dossier, settings)

        payload = json.loads(
            client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        )
        context = payload["title_related_context"].lower()
        self.assertIn("hot bedrooms", context)
        self.assertNotIn("dolphin", context)
        self.assertNotIn("slavery", context)
        self.assertEqual(len(keywords), 3)

    def test_public_image_search_retries_once_with_broad_keyword(self):
        async def check():
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                settings = replace(
                    load_settings(),
                    database_path=root / "image-fallback.db",
                )
                pipeline = NewsPipeline(settings)
                article_id = pipeline.db.upsert_article(
                    {
                        "source": "Nature News",
                        "url": "https://example.test/hot-sleep",
                        "canonical_url": "https://example.test/hot-sleep",
                        "title": "Too hot to sleep",
                        "summary": "Nighttime heat disrupts sleep.",
                        "published_at": "2026-08-25T00:00:00+00:00",
                        "doi": "",
                        "journal": "",
                        "word_count": 800,
                        "status": "extracted",
                        "discovered_at": "2026-08-25T00:00:00+00:00",
                    }
                )
                dossier = {
                    "id": article_id,
                    "rank": 1,
                    "date": "2026-08-25",
                    "content_type": POPULAR_CONTENT,
                    "title": "Too hot to sleep",
                    "summary": "Nighttime heat disrupts sleep.",
                    "text": "Article text",
                    "url": "https://example.test/hot-sleep",
                    "doi": "",
                    "images": [],
                }

                async def fake_details(_rank, _date, _content_type=None):
                    return copy.deepcopy(dossier)

                def fake_markdown(value, _settings, destination):
                    destination.mkdir(parents=True, exist_ok=True)
                    markdown = destination / "article.md"
                    metadata = destination / "metadata.json"
                    markdown.write_text(value["text"], encoding="utf-8")
                    metadata.write_text("{}", encoding="utf-8")
                    return markdown, metadata

                pipeline.paper_details = fake_details
                public_image = {
                    "url": "https://example.test/heat.jpg",
                    "local_path": "",
                    "caption": "Nighttime heat",
                    "credit": "Public archive",
                    "license": "Public Domain",
                    "publishable": True,
                }
                downloaded = {
                    **public_image,
                    "local_path": str(root / "heat.jpg"),
                }
                with (
                    patch(
                        "news.pipeline.generate_image_search_keywords",
                        return_value=[
                            "nighttime heat",
                            "hot bedroom sleep",
                            "high temperature sleep",
                        ],
                    ),
                    patch(
                        "news.pipeline.search_public_images",
                        side_effect=[[], [public_image]],
                    ) as search,
                    patch("news.pipeline.article_output_dir", return_value=root / "article"),
                    patch(
                        "news.pipeline.generate_image_captions",
                        return_value=["夜间地表温度分布。"],
                    ),
                    patch("news.pipeline.generate_article_markdown", side_effect=fake_markdown),
                    patch(
                        "news.pipeline.download_publishable_images",
                        return_value=[downloaded],
                    ),
                ):
                    generated = await pipeline.generate(1, "2026-08-25")

                self.assertEqual(search.call_count, 2)
                self.assertEqual(
                    search.call_args_list[1].args[0],
                    ["nighttime heat"],
                )
                self.assertEqual(len(generated["dossier"]["images"]), 1)
                markdown_text = generated["markdown_path"].read_text(encoding="utf-8")
                metadata_text = generated["metadata_path"].read_text(encoding="utf-8")
                self.assertNotIn("Public Domain", markdown_text)
                self.assertNotIn("Public archive", markdown_text)
                self.assertNotIn("https://example.test/heat.jpg", markdown_text)
                self.assertNotIn("正文配图", markdown_text)
                self.assertIn("图1. 夜间地表温度分布。", markdown_text)
                self.assertIn("## 文章信息", markdown_text)
                metadata = json.loads(metadata_text)
                self.assertIn("Public Domain", metadata_text)
                self.assertIn("Public archive", metadata_text)
                self.assertEqual(metadata["cover_image"]["url"], public_image["url"])
                self.assertEqual(len(metadata["body_images"]), 1)

        asyncio.run(check())

    def test_paper_title_first_page_and_cover_use_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            images.mkdir()
            first_page = images / "paper-first-page.png"
            first_page.write_bytes(b"png")
            wechat_cover = images / "paper-first-page-cover.png"
            wechat_cover.write_bytes(b"png")
            figure_cover = images / "figure-02.png"
            figure_cover.write_bytes(b"png")
            markdown = root / "article.md"
            title_cn = "夏季风变率驱动中国北方毛乌素沙地绿化和新石器时代社会变迁"
            markdown.write_text(f"# {title_cn}\n\n> 导语内容。\n", encoding="utf-8")
            metadata = {
                "content_type": PAPER_CONTENT,
                "title_cn": title_cn,
                "journal": "Communications Earth & Environment",
                "wechat_cover": {"local_path": str(wechat_cover)},
                "cover_image": {"local_path": str(figure_cover)},
            }
            (root / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False),
                encoding="utf-8",
            )

            _prepare_paper_markdown(
                markdown,
                {"local_path": str(first_page)},
            )
            markdown_text = markdown.read_text(encoding="utf-8")
            self.assertTrue(markdown_text.startswith("![论文第一页](images/paper-first-page.png)"))
            self.assertNotIn(f"# {title_cn}", markdown_text)
            self.assertEqual(_selected_cover_path(markdown), wechat_cover)
            self.assertEqual(
                _paper_draft_title(metadata, title_cn),
                "Communications Earth & Environment："
                "夏季风变率驱动中国北方毛乌素沙地绿化和新石器时代社会变迁",
            )
            self.assertEqual(
                _paper_draft_title({"content_type": PAPER_CONTENT}, title_cn),
                f"最新成果丨{title_cn}",
            )

    def test_paper_formatter_removes_duplicate_h1_after_brand_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            images.mkdir()
            first_page = images / "paper-first-page.png"
            first_page.write_bytes(b"png")
            markdown = root / "article.md"
            markdown.write_text(
                "![论文第一页](images/paper-first-page.png)\n\n> 导语内容。\n",
                encoding="utf-8",
            )
            (root / "metadata.json").write_text(
                json.dumps({"content_type": PAPER_CONTENT}),
                encoding="utf-8",
            )
            header = root / "qihai-header.png"
            header.write_bytes(b"png")
            settings = replace(load_settings(), wechat_app_id="", wechat_app_secret="")
            with (
                patch("publisher.wechat.QIHAI_HEADER", header),
                patch("publisher.wechat.ensure_tool_config"),
            ):
                result = format_markdown(markdown, settings)

            html = Path(result["article_html"]).read_text(encoding="utf-8")
            self.assertNotIn("<h1", html.lower())
            self.assertLess(html.index("qihai-header.png"), html.index("paper-first-page.png"))

    def test_qihai_theme_applies_xiaohu_styles_and_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            body_image = root / "result.png"
            body_image.write_bytes(b"png")
            markdown = root / "article.md"
            markdown.write_text(
                "# 测试文章\n\n> 导语内容。\n\n## 主要发现\n\n正文段落。\n\n"
                "![结果图](result.png)\n\n*图片说明*\n",
                encoding="utf-8",
            )
            header = root / "qihai-header.png"
            header.write_bytes(b"png")
            settings = replace(
                load_settings(),
                wechat_app_id="",
                wechat_app_secret="",
            )
            with (
                patch("publisher.wechat.QIHAI_HEADER", header),
                patch("publisher.wechat.ensure_tool_config"),
            ):
                result = format_markdown(markdown, settings)

            html = Path(result["article_html"]).read_text(encoding="utf-8")
            self.assertEqual(result["theme"], "qihai")
            self.assertTrue(result["brand_header"])
            self.assertIn("font-size:15px", html)
            self.assertIn("line-height:1.75", html)
            self.assertIn("letter-spacing:1.2px", html)
            self.assertIn("text-indent:2em", html)
            self.assertIn("font-size:18px", html)
            self.assertIn("width:95%", html)
            self.assertIn("background:#F5F7FA", html)
            self.assertIn("border-left:3px solid #C7D0DA", html)
            self.assertIn("linear-gradient(90deg, #1677FF 0%, #13A8D8 100%)", html)
            self.assertIn("font-size:12px", html)
            self.assertIn("color:#888888", html)
            self.assertIn("qihai-header.png", html)
            self.assertTrue(
                (Path(result["formatted_dir"]) / "images" / "qihai-header.png").is_file()
            )

    def test_pdf_figure_mapping_uses_number_and_adjacent_text_boxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def fake_download(_url, destination):
                destination.write_bytes(b"%PDF-test")

            def fake_layout(document, **kwargs):
                raw_dir = Path(kwargs["image_path"])
                raw_dir.mkdir(parents=True, exist_ok=True)
                (raw_dir / f"{Path(document).name}-0002-01.png").write_bytes(b"png")
                return {
                    "page_count": 2,
                    "pages": [
                        {
                            "page_number": 2,
                            "boxes": [
                                {
                                    "x0": 481.0,
                                    "y0": 212.0,
                                    "x1": 492.0,
                                    "y1": 222.0,
                                    "boxclass": "picture",
                                    "image": None,
                                    "table": None,
                                    "textlines": [],
                                },
                                {
                                    "x0": 90.0,
                                    "y0": 50.0,
                                    "x1": 510.0,
                                    "y1": 426.0,
                                    "boxclass": "picture",
                                    "image": None,
                                    "table": None,
                                    "textlines": [],
                                },
                                {
                                    "x0": 39.0,
                                    "y0": 434.0,
                                    "x1": 294.0,
                                    "y1": 492.0,
                                    "boxclass": "text",
                                    "image": None,
                                    "table": None,
                                    "textlines": [
                                        {"spans": [{"text": "Fig. 3 | Composite climate records."}]}
                                    ],
                                },
                                {
                                    "x0": 306.0,
                                    "y0": 433.5,
                                    "x1": 561.0,
                                    "y1": 480.0,
                                    "boxclass": "text",
                                    "image": None,
                                    "table": None,
                                    "textlines": [
                                        {"spans": [{"text": "Panels a-j show independent records."}]}
                                    ],
                                },
                            ],
                        }
                    ],
                }

            with (
                patch("papers.pdf_figures._download_pdf", side_effect=fake_download),
                patch("papers.pdf_figures.pymupdf4llm.to_json", side_effect=fake_layout),
            ):
                figures, metadata = extract_pdf_figures(
                    "https://example.test/article_reference.pdf",
                    root / "images",
                    article_license="CC BY 4.0",
                )

            self.assertEqual(metadata["layout_picture_regions"], 2)
            self.assertEqual(metadata["matched_figures"], 1)
            self.assertEqual(len(metadata["rejected_picture_regions"]), 1)
            self.assertEqual(figures[0]["figure_number"], 3)
            self.assertEqual(figures[0]["caption_boxclasses"], ["text", "text"])
            self.assertIn("Panels a-j", figures[0]["original_caption"])
            self.assertTrue(figures[0]["publishable"])
            self.assertTrue(Path(figures[0]["local_path"]).is_file())

    def test_paper_cover_fallback_compares_only_first_and_last_when_many(self):
        images = [
            {
                "url": f"https://example.test/figure-{number}.png",
                "local_path": f"/tmp/figure-{number}.png",
                "metadata_title": f"Figure {number}",
                "caption": (
                    "Monsoon variability, desert greening, and societal change summary"
                    if number == 5
                    else f"Supporting analysis panel {number}"
                ),
                "image_source": "pdf_figure",
                "image_role": "figure",
                "figure_number": number,
            }
            for number in range(1, 6)
        ]
        cover, body, _ = _select_article_images(
            images,
            PAPER_CONTENT,
            "Monsoon variability drove desert greening and societal change",
        )
        self.assertEqual(cover["figure_number"], 5)
        self.assertLessEqual(len(body), 4)
        tied_cover, _, _ = _select_article_images(images, PAPER_CONTENT, "")
        self.assertEqual(tied_cover["figure_number"], 1)

    def test_image_policy(self):
        self.assertTrue(assess_image("CC BY 4.0")[0])
        self.assertTrue(assess_image("CC BY-SA 4.0")[0])
        self.assertTrue(assess_image("CC BY-NC 4.0")[0])
        self.assertTrue(assess_image("cc-by-nc-sa")[0])
        self.assertTrue(assess_image("CC0")[0])
        self.assertTrue(assess_image("Public Domain")[0])
        self.assertFalse(assess_image("CC BY-NC-ND 4.0")[0])
        self.assertFalse(assess_image("CC BY-ND 4.0")[0])
        self.assertTrue(
            assess_image("CC BY-NC-ND 4.0", allow_no_derivatives=True)[0]
        )
        self.assertTrue(assess_image("CC BY-ND 4.0", allow_no_derivatives=True)[0])
        self.assertFalse(assess_image("CC BY-NC", credit="Getty Images")[0])
        self.assertFalse(assess_image("CC BY-NC", credit="Reproduced with permission")[0])
        self.assertFalse(assess_image("CC BY-NC", credit="Based on Google Earth imagery")[0])
        self.assertFalse(assess_image("unknown")[0])

    def test_paper_nd_figures_allow_body_but_not_cover(self):
        nd = apply_policy(
            {
                "url": "https://example.test/nd.png",
                "local_path": "/tmp/nd.png",
                "license": "CC BY-NC-ND 4.0",
                "caption": "Complete Figure 1",
                "image_source": "html_figure",
                "image_role": "figure",
            },
            allow_no_derivatives=True,
        )
        reusable = apply_policy(
            {
                "url": "https://example.test/by.png",
                "local_path": "/tmp/by.png",
                "license": "CC BY 4.0",
                "caption": "Complete Figure 2",
                "image_source": "html_figure",
                "image_role": "figure",
            }
        )

        self.assertTrue(nd["publishable"])
        self.assertFalse(nd["derivatives_allowed"])
        self.assertFalse(nd["cover_eligible"])
        cover, body, _ = _select_article_images([nd, reusable], PAPER_CONTENT)
        self.assertEqual(cover, reusable)
        self.assertIn(nd, body)
        nd_only_cover, nd_only_body, _ = _select_article_images([nd], PAPER_CONTENT)
        self.assertIsNone(nd_only_cover)
        self.assertEqual(nd_only_body, [nd])
        self.assertEqual(
            _paper_wechat_cover(
                {
                    "wechat_cover_path": "/tmp/cropped-cover.png",
                    "source_pdf": "/tmp/paper.pdf",
                    "license": "CC BY-NC-ND 4.0",
                }
            ),
            {},
        )

    def test_paper_nd_figure_specific_restrictions_take_priority(self):
        records = _apply_article_license_to_html_figures(
            [
                {
                    "image_source": "html_figure",
                    "figure_image_count": 1,
                    "caption": "Figure 1",
                    "credit": "Reproduced with permission",
                    "license": "",
                }
            ],
            "CC BY-NC-ND 4.0",
        )
        self.assertFalse(records[0]["publishable"])
        self.assertIn("third-party marker", records[0]["reason"])

    def test_target_openid_binds_only_when_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("QQ_APP_ID=1\nQQ_TARGET_OPENID=\nMODEL_NAME=x\n", encoding="utf-8")
            bound, target = bind_qq_target_openid("full-openid-123", env_path)
            self.assertTrue(bound)
            self.assertEqual(target, "full-openid-123")
            self.assertIn("QQ_TARGET_OPENID=full-openid-123", env_path.read_text())
            self.assertEqual(env_path.stat().st_mode & 0o777, 0o600)

            bound, target = bind_qq_target_openid("different-openid", env_path)
            self.assertFalse(bound)
            self.assertEqual(target, "full-openid-123")
            self.assertNotIn("different-openid", env_path.read_text())

    def test_runtime_settings_update_after_binding(self):
        async def check():
            bot = QQNewsBot("app", "secret")
            bot.settings.qq_target_openid = ""
            with patch(
                "bot.bridge.bind_qq_target_openid",
                return_value=(True, "full-openid-456"),
            ):
                self.assertTrue(await bot.ensure_target_bound("full-openid-456"))
            self.assertEqual(bot.settings.qq_target_openid, "full-openid-456")
            self.assertIs(bot.pipeline.settings, bot.settings)
            self.assertIs(bot.command_handler.settings, bot.settings)

        asyncio.run(check())

    def test_batch_chinese_titles_use_one_model_call_and_keep_order(self):
        settings = replace(
            load_settings(),
            model_base_url="https://model.example/v1",
            model_api_key="test-key",
            model_name="test-model",
        )
        candidates = [
            {"title": f"English title {index}", "source": "Nature News", "score": 20 - index}
            for index in range(1, 11)
        ]
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "items": [
                                    {"index": index, "title_cn": f"中文标题{index}"}
                                    for index in range(10, 0, -1)
                                ]
                            },
                            ensure_ascii=False,
                        )
                    )
                )
            ]
        )
        client = MagicMock()
        client.chat.completions.create.return_value = response
        with patch("writer.llm.OpenAI", return_value=client):
            selected, used_model, error = select_top_ten(candidates, settings)

        self.assertTrue(used_model)
        self.assertEqual(error, "")
        self.assertEqual(client.chat.completions.create.call_count, 1)
        system_prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn("以英文原标题为唯一依据", system_prompt)
        self.assertIn("只做忠实翻译和轻微中文润色", system_prompt)
        self.assertIn("禁止根据摘要或其他元数据补充", system_prompt)
        self.assertIn("禁止为了吸引眼球扩大、强化或改写原文含义", system_prompt)
        self.assertIn("删除原标题末尾", system_prompt)
        self.assertEqual(
            [item["title_cn"] for item in selected],
            [f"中文标题{index}" for index in range(1, 11)],
        )
        self.assertEqual(
            [item["title"] for item in selected],
            [f"English title {index}" for index in range(1, 11)],
        )

    def test_news_restores_chinese_title_and_keeps_english_title(self):
        async def check():
            with tempfile.TemporaryDirectory() as tmp:
                settings = replace(
                    load_settings(),
                    database_path=Path(tmp) / "titles.db",
                    model_base_url="https://model.example/v1",
                    model_api_key="test-key",
                    model_name="test-model",
                )
                pipeline = NewsPipeline(settings)
                article_id = pipeline.db.upsert_article(
                    {
                        "source": "Nature News",
                        "url": "https://example.test/title",
                        "canonical_url": "https://example.test/title",
                        "title": "English original title",
                        "summary": "Near-surface wind station observations",
                        "published_at": "2026-08-25T00:00:00+00:00",
                        "doi": "",
                        "journal": "",
                        "word_count": 800,
                        "status": "extracted",
                        "discovered_at": "2026-08-25T00:00:00+00:00",
                    }
                )
                pipeline.db.replace_candidates(
                    "2026-08-25",
                    [{"article_id": article_id, "score": 10, "title_cn": ""}],
                    POPULAR_CONTENT,
                )
                pipeline.db.set_daily_run(
                    "2026-08-25",
                    content_type=POPULAR_CONTENT,
                    status="success",
                )

                def fake_titles(candidates, _settings):
                    return [dict(candidates[0], title_cn="中文标题")], True, ""

                with patch("news.pipeline.select_top_ten", side_effect=fake_titles) as model_call:
                    candidates = await pipeline.get_or_refresh(
                        "2026-08-25",
                        POPULAR_CONTENT,
                    )
                text = pipeline.format_news(candidates)
                self.assertEqual(model_call.call_count, 1)
                self.assertIn("**1. 中文标题**\nEnglish original title\n", text)

        asyncio.run(check())

    def test_chinese_title_model_failure_falls_back_to_english(self):
        settings = replace(
            load_settings(),
            model_base_url="https://model.example/v1",
            model_api_key="test-key",
            model_name="test-model",
        )
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("model unavailable")
        with patch("writer.llm.OpenAI", return_value=client):
            selected, used_model, error = select_top_ten(
                [{"title": "English fallback", "source": "Nature News", "score": 10}],
                settings,
            )
        self.assertFalse(used_model)
        self.assertEqual(selected[0]["title_cn"], "")
        self.assertIn("model unavailable", error)

    def test_llm_fallback(self):
        settings = load_settings()
        items = [{"title": str(i), "score": 20 - i} for i in range(20)]
        selected, used_model, error = select_top_ten(items, settings)
        if not settings.model_configured:
            self.assertFalse(used_model)
            self.assertEqual(len(selected), 10)
            self.assertIn("not configured", error)


if __name__ == "__main__":
    unittest.main()
