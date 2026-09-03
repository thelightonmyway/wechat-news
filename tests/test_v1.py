from __future__ import annotations

import asyncio
import copy
import json
import tempfile
import unittest
from dataclasses import replace
from datetime import date as date_type, datetime, time, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from bot.bridge import QQNewsBot
from bot.commands import CommandHandler, NEWS_USAGE, PAPER_USAGE, _paper_image_summary
from db import Database
from images.policy import apply_policy, assess_image
from images.search import normalize_search_result, search_public_images
from news.extract import discover_figure_images
from news.feeds import canonicalize_url, load_feeds, normalize_title
from news.pipeline import (
    PAPER_CONTENT,
    PAPER_ONLY_SOURCES,
    POPULAR_CONTENT,
    PRIMARY_SOURCES,
    SECONDARY_SOURCES,
    NewsPipeline,
    _apply_article_license_to_html_figures,
    _images_redundant,
    _insert_paper_figures,
    _paper_publication_within_window,
    _paper_match_source_paragraphs,
    _paper_wechat_cover,
    _prepare_paper_markdown,
    _select_article_images,
    content_type_for_date,
    deduplicate,
    deterministic_score,
    is_relevant_news_after_extraction,
    is_relevant_news_rss_prefilter,
    is_relevant_topic,
    merge_paper_candidate_pool,
    paper_relevance_score,
    prioritize_candidates,
    source_allowed_for_content,
)
from papers.doi import resolve_doi_landing_page
from papers.oa_mirror import resolve_oa_html_mirror
from papers.openalex import (
    OpenAlexAdapter,
    is_allowed_paper_journal,
    journal_display_name,
)
from papers.pdf_figures import (
    _download_pdf,
    discover_pdf_source,
    download_pdf_with_wiley_tdm,
    extract_pdf_figures,
)
from publisher.wechat import _paper_draft_title, _selected_cover_path, format_markdown
from scheduler import should_run_startup_catchup
from settings import bind_qq_target_openid, load_settings
from writer.llm import (
    _normalize_article_markdown,
    generate_article_markdown,
    generate_image_captions,
    generate_image_search_keywords,
    select_paper_top_ten,
    select_top_ten,
    translate_paper_titles,
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

    def test_news_rss_prefilter_defers_strict_relevance_until_after_extraction(self):
        rss_item = {
            "title": "Why the ocean is changing faster than expected",
            "summary": "Scientists explain what happened.",
        }
        self.assertTrue(is_relevant_news_rss_prefilter(rss_item))
        self.assertFalse(is_relevant_topic(rss_item))
        extracted = {
            **rss_item,
            "text": (
                "The full report identifies air-sea interaction and atmospheric "
                "circulation as the mechanism driving the observed change."
            ),
        }
        self.assertTrue(is_relevant_news_after_extraction(extracted))

        medical = {
            "title": "Heat treatment improves cancer outcomes",
            "summary": "A clinical trial of a new medical therapy.",
        }
        self.assertFalse(is_relevant_news_rss_prefilter(medical))

    def test_shared_topic_relevance_requires_specific_process(self):
        self.assertFalse(
            is_relevant_topic(
                {
                    "content_type": POPULAR_CONTENT,
                    "title": "Generic climate model benchmarking",
                    "summary": "CMIP6 model evaluation",
                }
            )
        )
        self.assertFalse(
            is_relevant_topic(
                {
                    "content_type": PAPER_CONTENT,
                    "title": "Generic climate model benchmarking",
                    "summary": "CMIP6 model evaluation",
                }
            )
        )
        self.assertTrue(
            is_relevant_topic(
                {
                    "title": "Climate model projections of surface wind",
                    "summary": "Near-surface wind speed changes",
                }
            )
        )
        for title in (
            "Heatwaves and temperature extremes",
            "Mechanisms of extreme precipitation",
            "Temperature-soil moisture feedback",
            "Tropical cyclone dynamics",
            "Wildfire risk during heat and drought",
        ):
            self.assertTrue(is_relevant_topic({"title": title, "summary": ""}), title)
        self.assertFalse(is_relevant_topic({"title": "Wildfire impacts", "summary": ""}))
        self.assertFalse(
            is_relevant_topic(
                {
                    "title": "Post-fire vegetation recovery during heat",
                    "summary": "Ecological succession after wildfire",
                }
            )
        )

    def test_paper_relevance_regressions_and_types(self):
        self.assertEqual(
            paper_relevance_score(
                {
                    "title": (
                        "Drought-induced soil moisture declines in Andean catchments "
                        "inferred from satellite-derived ground displacement"
                    ),
                    "summary": "A hydrology and geodetic analysis of catchment storage.",
                    "work_type": "article",
                }
            ),
            0,
        )
        self.assertEqual(
            paper_relevance_score(
                {
                    "title": (
                        "Central Pacific El Niño-driven Pacific–Atlantic teleconnections "
                        "are an important source of North Atlantic Oscillation predictability"
                    ),
                    "summary": "ENSO teleconnections control NAO climate predictability.",
                    "work_type": "article",
                }
            ),
            3,
        )
        self.assertGreaterEqual(
            paper_relevance_score(
                {
                    "title": (
                        "Role of natural halogen chemistry on the evolution of global "
                        "stratospheric ozone depletion"
                    ),
                    "summary": "Ozone-climate interactions in the stratosphere.",
                    "work_type": "article",
                }
            ),
            2,
        )
        self.assertEqual(
            paper_relevance_score(
                {
                    "title": "Reply to comments on atmospheric circulation",
                    "summary": "A reply to the original article.",
                    "work_type": "reply",
                }
            ),
            0,
        )
        self.assertEqual(
            paper_relevance_score(
                {
                    "title": "Correction to a study of surface wind",
                    "summary": "Publisher correction.",
                    "work_type": "correction",
                }
            ),
            0,
        )

    def test_paper_storms_require_climate_scale_connection(self):
        self.assertEqual(
            paper_relevance_score(
                {
                    "title": (
                        "Convective butterflies lead to tropical cyclone rapid "
                        "intensification"
                    ),
                    "summary": "Convective and storm-scale intensification dynamics.",
                    "work_type": "article",
                }
            ),
            0,
        )
        self.assertEqual(
            paper_relevance_score(
                {
                    "title": "A storm-scale microphysics mechanism in an eyewall",
                    "summary": "A single extreme weather event.",
                    "work_type": "article",
                }
            ),
            0,
        )
        self.assertGreaterEqual(
            paper_relevance_score(
                {
                    "title": "ENSO controls interannual variability of tropical cyclone activity",
                    "summary": "El Niño teleconnections regulate seasonal cyclone activity.",
                    "work_type": "article",
                }
            ),
            2,
        )
        self.assertGreaterEqual(
            paper_relevance_score(
                {
                    "title": "Climate change alters tropical cyclone frequency and intensity",
                    "summary": "Long-term projections attribute changes in cyclone climatology.",
                    "work_type": "article",
                }
            ),
            2,
        )

    def test_news_feed_configuration_and_paper_only_sources(self):
        feeds = {
            item["name"]: item["url"]
            for item in load_feeds(
                Path(__file__).resolve().parents[1] / "config" / "feeds.yaml"
            )
        }
        expected = {
            "Guardian Climate Crisis": (
                "https://www.theguardian.com/environment/climate-crisis/rss"
            ),
            "NASA Earth Observatory": (
                "https://earthobservatory.nasa.gov/feeds/earth-observatory.rss"
            ),
            "NOAA NOS News": "https://oceanservice.noaa.gov/rss/nosnews.xml",
            "NOAA NOS Newsroom": (
                "https://oceanservice.noaa.gov/newsroom/nosmedia.xml"
            ),
            "Copernicus Climate": "https://climate.copernicus.eu/rss.xml",
            "Inside Climate News": "https://insideclimatenews.org/feed/",
        }
        for name, url in expected.items():
            with self.subTest(source=name):
                self.assertEqual(feeds.get(name), url)
                self.assertIn(name, PRIMARY_SOURCES | SECONDARY_SOURCES)
                self.assertTrue(source_allowed_for_content(name, POPULAR_CONTENT))

        for name in PAPER_ONLY_SOURCES:
            with self.subTest(source=name):
                self.assertFalse(source_allowed_for_content(name, POPULAR_CONTENT))
                self.assertTrue(source_allowed_for_content(name, PAPER_CONTENT))

        for name in ("Nature News", "Eos / AGU", "Carbon Brief"):
            self.assertTrue(source_allowed_for_content(name, POPULAR_CONTENT))

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

            async def next_paper_batch(self, date=None):
                self.calls.append(("next", PAPER_CONTENT))
                return [
                    {
                        "rank": 1,
                        "content_type": PAPER_CONTENT,
                        "title": "next paper",
                        "source": "test",
                        "published_at": "2026-08-25T00:00:00+00:00",
                    }
                ]

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
            self.assertEqual(await handler.handle("/papers next"), "list:paper")
            self.assertIn(("next", PAPER_CONTENT), pipeline.calls)
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

    def test_papers_stays_frozen_after_current_candidate_is_published(self):
        async def check():
            with tempfile.TemporaryDirectory() as tmp:
                settings = replace(load_settings(), database_path=Path(tmp) / "frozen.db")
                pipeline = NewsPipeline(settings)
                ids = [
                    pipeline.db.upsert_article(
                        {
                            "source": "Journal of Climate",
                            "url": f"https://example.test/frozen-{index}",
                            "canonical_url": f"https://example.test/frozen-{index}",
                            "title": f"Frozen climate paper {index}",
                            "summary": "Atmospheric circulation climate mechanism",
                            "published_at": "2026-08-26T00:00:00+00:00",
                            "doi": f"10.1000/frozen-{index}",
                            "journal": "Journal of Climate",
                            "word_count": 800,
                            "status": "published_paper",
                            "discovered_at": "2026-08-26T00:00:00+00:00",
                        }
                    )
                    for index in range(1, 11)
                ]
                pipeline.db.replace_candidates(
                    "2026-08-26",
                    [
                        {"article_id": article_id, "score": 20 - index, "title_cn": f"论文{index}"}
                        for index, article_id in enumerate(ids, start=1)
                    ],
                    PAPER_CONTENT,
                )
                pipeline.db.set_daily_run(
                    "2026-08-26",
                    content_type=PAPER_CONTENT,
                    candidate_count=10,
                    status="success",
                )
                pipeline.db.save_publish_history(ids[3], "drafted", "draft-4")
                original = pipeline.db.get_candidates("2026-08-26", PAPER_CONTENT)
                pipeline.refresh = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    AssertionError("published current candidate must not refresh /papers")
                )
                current = await pipeline.get_or_refresh("2026-08-26", PAPER_CONTENT)
                self.assertEqual(
                    [(item["rank"], item["id"]) for item in current],
                    [(item["rank"], item["id"]) for item in original],
                )
                self.assertEqual(current[3]["id"], ids[3])

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

    def test_news_expands_48h_to_7d_and_30d_without_filling(self):
        def item(index, title, summary, source="test"):
            return {
                "source": source,
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
            item(
                9,
                "Surface wind changes in a climate model",
                "Near-surface wind and atmospheric circulation",
                "Nature Climate Change",
            ),
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
                self.assertEqual(calls, [48, 168, 720])
                self.assertEqual(len(candidates), 5)
                self.assertTrue(all(is_relevant_topic(value) for value in candidates))
                self.assertFalse(any("Batter" in value["title"] for value in candidates))
                self.assertFalse(any("Alzheimer" in value["title"] for value in candidates))
                self.assertFalse(any("Solar" in value["title"] for value in candidates))
                self.assertFalse(
                    any(value["source"] in PAPER_ONLY_SOURCES for value in candidates)
                )

        asyncio.run(check())

    def test_paper_expands_48h_to_7d_and_30d(self):
        titles = {
            1: "Near-surface wind speed recovery",
            2: "Polar vortex circulation dynamics",
            3: "Air-sea interaction and ocean energy input",
        }

        def item(index):
            return {
                "source": "test",
                "url": f"https://example.test/paper-{index}",
                "canonical_url": f"https://example.test/paper-{index}",
                "title": titles[index],
                "normalized_title": normalize_title(titles[index]),
                "summary": titles[index],
                "published_at": "2026-08-26T00:00:00+00:00",
                "doi": f"10.1000/paper-{index}",
                "journal": "Journal of Climate",
                "word_count": 800,
                "status": "discovered",
                "discovered_at": "2026-08-26T00:00:00+00:00",
            }

        windows = {
            48: [item(1)],
            168: [item(1), item(2)],
            720: [item(1), item(2), item(3)],
        }
        calls = []

        def fake_fetch(_path, hours):
            calls.append(hours)
            values = windows[hours]
            return copy.deepcopy(values), [], {"test": len(values)}

        async def check():
            with tempfile.TemporaryDirectory() as tmp:
                settings = replace(
                    load_settings(),
                    database_path=Path(tmp) / "paper-window.db",
                    model_base_url="",
                    model_api_key="",
                    model_name="",
                    openalex_api_key="test-key",
                )
                pipeline = NewsPipeline(settings)
                pipeline.openalex.discover_recent_papers = MagicMock(
                    return_value=[
                        {
                            "title": titles[1],
                            "abstract": titles[1],
                            "publication_date": "2026-08-25",
                            "journal": "Journal of Climate",
                            "doi": "10.1000/paper-1",
                            "type": "article",
                        },
                        {
                            "title": "Stratospheric ozone depletion evolution",
                            "abstract": "Ozone-climate interactions in the stratosphere",
                            "publication_date": "2026-08-24",
                            "journal": "Atmospheric Chemistry and Physics",
                            "doi": "10.1000/openalex-ozone",
                            "type": "article",
                        },
                    ]
                )

                async def fake_extract(values):
                    return copy.deepcopy(values)

                async def fake_published(values, _run_date):
                    return [
                        {**value, "paper_local_score": 2, "work_type": "article"}
                        for value in copy.deepcopy(values)
                    ]

                pipeline._extract_shortlist = fake_extract
                pipeline._published_papers = fake_published
                with patch("news.pipeline.fetch_all_feeds", side_effect=fake_fetch):
                    candidates = await pipeline.refresh("2026-08-26", PAPER_CONTENT)

                self.assertEqual(calls, [48, 168, 720])
                self.assertEqual(len(candidates), 4)
                self.assertEqual(pipeline.openalex.discover_recent_papers.call_count, 3)
                self.assertEqual(
                    pipeline.openalex.discover_recent_papers.call_args_list,
                    [
                        unittest.mock.call(date_type(2026, 8, 24), date_type(2026, 8, 26)),
                        unittest.mock.call(date_type(2026, 8, 19), date_type(2026, 8, 26)),
                        unittest.mock.call(date_type(2026, 7, 27), date_type(2026, 8, 26)),
                    ],
                )
                self.assertEqual(pipeline.last_paper_discovery_stats["rss_candidates"], 3)
                self.assertEqual(pipeline.last_paper_discovery_stats["openalex_added"], 1)

        asyncio.run(check())

    def test_paper_publication_date_must_be_within_30_days(self):
        self.assertTrue(
            _paper_publication_within_window(
                {"publication_date": "2026-07-27"},
                "2026-08-26",
            )
        )
        self.assertFalse(
            _paper_publication_within_window(
                {"publication_date": "2026-07-26"},
                "2026-08-26",
            )
        )

    def test_openalex_papers_merge_with_rss_and_deduplicate_doi(self):
        rss = [
            {
                "title": "Surface wind variability",
                "summary": "Near-surface wind mechanism",
                "doi": "10.1000/shared",
                "canonical_url": "https://publisher.test/shared",
                "published_at": "2026-08-25T00:00:00+00:00",
                "discovery_origin": "rss",
            }
        ]
        openalex = [
            {
                "title": "Surface wind variability",
                "summary": "Near-surface wind mechanism",
                "doi": "10.1000/shared",
                "canonical_url": "https://doi.org/10.1000/shared",
                "published_at": "2026-08-24T00:00:00+00:00",
                "discovery_origin": "openalex",
            },
            {
                "title": "Stratospheric ozone depletion evolution",
                "summary": "Ozone-climate interactions",
                "doi": "10.1000/ozone",
                "canonical_url": "https://doi.org/10.1000/ozone",
                "published_at": "2026-08-23T00:00:00+00:00",
                "discovery_origin": "openalex",
            },
        ]
        merged = merge_paper_candidate_pool(rss, openalex)
        self.assertEqual(len(merged), 2)
        self.assertEqual({item["doi"] for item in merged}, {"10.1000/shared", "10.1000/ozone"})
        self.assertEqual(
            sum(item.get("discovery_origin") == "openalex" for item in merged),
            1,
        )

    def test_paper_journal_whitelist(self):
        allowed = [
            ("Nature", ""),
            ("Nature Climate Change", "Springer Nature"),
            ("Communications Earth & Environment", "Springer Nature"),
            ("npj Climate and Atmospheric Science", "Nature Portfolio"),
            ("Science", "AAAS"),
            ("Science Advances", "AAAS"),
            (
                "Science Translational Medicine",
                "American Association for the Advancement of Science",
            ),
            ("Proceedings of the National Academy of Sciences", ""),
            ("Geophysical Research Letters", "American Geophysical Union"),
            ("Earth's Future", "American Geophysical Union"),
            ("AGU Advances", "American Geophysical Union"),
            ("The Innovation", ""),
            ("Atmospheric Chemistry and Physics", "Copernicus Publications"),
            ("Weather and Climate Dynamics", "Copernicus Publications"),
            ("Earth System Dynamics", "Copernicus Publications"),
            ("Climate Dynamics", "Springer Nature"),
            ("Environmental Research Letters", "IOP Publishing"),
        ]
        for journal, publisher in allowed:
            with self.subTest(journal=journal):
                self.assertTrue(is_allowed_paper_journal(journal, publisher))

        rejected = [
            ("Britain International of Exact Sciences (BIoEx) Journal", ""),
            ("SOLA", ""),
            ("Agricultural Water Management", "Elsevier"),
            ("Geoscientific Model Development", "Copernicus Publications"),
            ("Theoretical and Applied Climatology", "Springer Nature"),
        ]
        for journal, publisher in rejected:
            with self.subTest(journal=journal):
                self.assertFalse(is_allowed_paper_journal(journal, publisher))

    def test_jgra_alias_whitelist_and_source_match(self):
        aliases = (
            "Journal of Geophysical Research: Atmospheres",
            "Journal of Geophysical Research - Atmospheres",
            "JGR: Atmospheres",
            "JGR Atmospheres",
        )
        for alias in aliases:
            with self.subTest(alias=alias):
                self.assertTrue(
                    is_allowed_paper_journal(alias, "American Geophysical Union")
                )
                self.assertEqual(
                    journal_display_name(alias),
                    "Journal of Geophysical Research: Atmospheres",
                )
        self.assertFalse(
            is_allowed_paper_journal(
                "Journal of Geophysical Research: Oceans",
                "American Geophysical Union",
            )
        )

        class FakeSources:
            def __init__(self):
                self.query = ""
                self.calls = []

            def search(self, query):
                self.query = query
                return self

            def get(self, **kwargs):
                self.calls.append((self.query, kwargs))
                return [
                    {
                        "id": "https://openalex.org/S-JGRA",
                        "display_name": "Journal of Geophysical Research: Atmospheres",
                        "host_organization_name": "American Geophysical Union",
                    },
                    {
                        "id": "https://openalex.org/S-OCEANS",
                        "display_name": "Journal of Geophysical Research: Oceans",
                        "host_organization_name": "American Geophysical Union",
                    },
                ]

        sources = FakeSources()
        with patch("papers.openalex.Sources", return_value=sources):
            adapter = OpenAlexAdapter("test-openalex-key")
            self.assertEqual(adapter._resolve_source_id("JGR Atmospheres"), "S-JGRA")
            self.assertEqual(
                adapter._resolve_source_id("Journal of Geophysical Research: Atmospheres"),
                "S-JGRA",
            )
        self.assertEqual(len(sources.calls), 1)
        self.assertEqual(sources.calls[0][0], "JGR Atmospheres")

    def test_openalex_journal_first_paginates_target_sources(self):
        class FakeSources:
            def search(self, query):
                self.query = query
                return self

            def get(self, **_kwargs):
                return [
                    {
                        "id": f"https://openalex.org/S-{self.query}",
                        "display_name": self.query,
                        "host_organization_name": "American Geophysical Union",
                    }
                ]

        works_pages = [
            [
                {
                    "title": "GRL paper outside topic shortlist",
                    "doi": "https://doi.org/10.1000/grl-1",
                    "publication_date": "2026-08-26",
                    "type": "article",
                    "primary_location": {
                        "source": {
                            "display_name": "Geophysical Research Letters",
                            "host_organization_name": "American Geophysical Union",
                        }
                    },
                }
            ],
            [
                {
                    "title": "ESD paper page two",
                    "doi": "https://doi.org/10.1000/esd-2",
                    "publication_date": "2026-08-25",
                    "type": "article",
                    "primary_location": {
                        "source": {
                            "display_name": "Earth System Dynamics",
                            "host_organization_name": "Copernicus Publications",
                        }
                    },
                }
            ],
            [
                {
                    "title": "Science paper",
                    "doi": "https://doi.org/10.1000/science-3",
                    "publication_date": "2026-08-24",
                    "type": "article",
                    "primary_location": {
                        "source": {
                            "display_name": "Science",
                            "host_organization_name": "AAAS",
                        }
                    },
                }
            ],
            [
                {
                    "title": "PNAS paper",
                    "doi": "https://doi.org/10.1000/pnas-4",
                    "publication_date": "2026-08-23",
                    "type": "article",
                    "primary_location": {
                        "source": {
                            "display_name": "Proceedings of the National Academy of Sciences",
                            "host_organization_name": "National Academy of Sciences",
                        }
                    },
                }
            ],
        ]

        class FakeWorks:
            def __init__(self):
                self.filters = []
                self.page = 0

            def filter(self, **kwargs):
                self.filters.append(kwargs)
                return self

            def select(self, _value):
                return self

            def get(self, **_kwargs):
                page = works_pages[self.page]
                self.page += 1
                return page

        with (
            patch("papers.openalex.JOURNAL_FIRST_TARGETS", (
                "Geophysical Research Letters",
                "Earth System Dynamics",
                "Science",
                "PNAS",
            )),
            patch("papers.openalex.DISCOVERY_QUERIES", ()),
            patch("papers.openalex.Sources", return_value=FakeSources()),
            patch("papers.openalex.Works", return_value=FakeWorks()),
        ):
            adapter = OpenAlexAdapter("test-openalex-key")
            records = adapter.discover_recent_papers(
                date_type(2026, 8, 23),
                date_type(2026, 8, 26),
            )

        self.assertEqual(
            {record["doi"] for record in records},
            {
                "10.1000/grl-1",
                "10.1000/esd-2",
                "10.1000/science-3",
                "10.1000/pnas-4",
            },
        )
        self.assertEqual(adapter.last_journal_first_count, 4)
        self.assertEqual(adapter.last_topic_count, 0)

    def test_openalex_journal_first_follows_cursor_until_exhausted(self):
        class Page(list):
            def __init__(self, values, next_cursor):
                super().__init__(values)
                self.meta = {"next_cursor": next_cursor}

        class FakeSources:
            def search(self, _query):
                return self

            def get(self, **_kwargs):
                return [{
                    "id": "https://openalex.org/S-GRL",
                    "display_name": "Geophysical Research Letters",
                    "host_organization_name": "American Geophysical Union",
                }]

        def work(doi):
            return {
                "title": f"GRL work {doi}",
                "doi": f"https://doi.org/{doi}",
                "publication_date": "2026-08-26",
                "type": "article",
                "primary_location": {
                    "source": {
                        "display_name": "Geophysical Research Letters",
                        "host_organization_name": "American Geophysical Union",
                    }
                },
            }

        class FakeWorks:
            def __init__(self):
                self.get_calls = []

            def filter(self, **_kwargs):
                return self

            def select(self, _value):
                return self

            def get(self, **kwargs):
                self.get_calls.append(kwargs)
                if kwargs["cursor"] == "*":
                    return Page([work("10.1000/page-1")], "cursor-2")
                return Page([work("10.1000/page-2")], None)

        works = FakeWorks()
        with (
            patch("papers.openalex.JOURNAL_FIRST_TARGETS", ("Geophysical Research Letters",)),
            patch("papers.openalex.DISCOVERY_QUERIES", ()),
            patch("papers.openalex.Sources", return_value=FakeSources()),
            patch("papers.openalex.Works", return_value=works),
        ):
            records = OpenAlexAdapter("test-openalex-key").discover_recent_papers(
                date_type(2026, 8, 26),
                date_type(2026, 8, 26),
            )
        self.assertEqual(len(records), 2)
        self.assertEqual([call["per_page"] for call in works.get_calls], [100, 100])
        self.assertEqual([call["cursor"] for call in works.get_calls], ["*", "cursor-2"])

    def test_openalex_nsws_query_reaches_ai_candidate_pool(self):
        work = {
            "title": "Near-surface wind speed trend over land",
            "doi": "https://doi.org/10.1000/nsws",
            "publication_date": "2026-08-26",
            "type": "article",
            "primary_location": {
                "source": {
                    "display_name": "Geophysical Research Letters",
                    "host_organization_name": "American Geophysical Union",
                }
            },
        }
        class FakeWorks:
            def search(self, query):
                self.query = query
                return self

            def filter(self, **_kwargs):
                return self

            def select(self, _value):
                return self

            def get(self, **_kwargs):
                return [work] if self.query == "near-surface wind speed trend" else []

        with (
            patch("papers.openalex.JOURNAL_FIRST_TARGETS", ()),
            patch("papers.openalex.DISCOVERY_QUERIES", ("near-surface wind speed trend",)),
            patch("papers.openalex.Works", return_value=FakeWorks()),
        ):
            records = OpenAlexAdapter("test-openalex-key").discover_recent_papers(
                date_type(2026, 8, 26),
                date_type(2026, 8, 26),
                per_query=15,
            )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["doi"], "10.1000/nsws")

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

    def test_paper_prompt_uses_short_natural_style_and_verbatim_quote_rules(self):
        settings = replace(
            load_settings(),
            model_base_url="https://model.example/v1",
            model_api_key="test-key",
            model_name="test-model",
        )
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            "# 测试标题\n\n## 关键结果\n\n简短正文。\n\n"
                            "> “The supplied paper states an exact scientific result.”\n\n"
                            "这句引文支持上述判断。\n\n"
                            "> “The ensemble spread remains stable across the tested regions while forecast errors decline during the validation period and improve seasonal predictability.”\n\n"
                            "> “This quotation was invented by the model.”\n\n"
                            "这条引文不应保留。"
                        )
                    )
                )
            ]
        )
        client = MagicMock()
        client.chat.completions.create.return_value = response
        with tempfile.TemporaryDirectory() as tmp, patch(
            "writer.llm.OpenAI",
            return_value=client,
        ):
            root = Path(tmp)
            paper_path, _ = generate_article_markdown(
                {
                    "content_type": PAPER_CONTENT,
                    "title": "Test paper",
                    "title_cn": "测试标题",
                    "text": (
                        "The supplied paper states an exact scientific result. "
                        "The ensemble spread remains stable across the tested regions while "
                        "forecast errors decline during the validation period and improve "
                        "seasonal predictability."
                    ),
                    "summary": "Paper abstract",
                    "openalex": {"abstract": "Paper abstract"},
                    "images": [],
                },
                settings,
                root / "paper",
            )
            paper_prompt = client.chat.completions.create.call_args.kwargs["messages"][0][
                "content"
            ]
            paper_markdown = paper_path.read_text(encoding="utf-8")
            client.chat.completions.create.reset_mock()
            generate_article_markdown(
                {
                    "content_type": POPULAR_CONTENT,
                    "title": "Test news",
                    "title_cn": "测试新闻",
                    "text": "News article text",
                    "summary": "News summary",
                    "images": [],
                },
                settings,
                root / "news",
            )
            news_prompt = client.chat.completions.create.call_args.kwargs["messages"][0][
                "content"
            ]

        self.assertIn("正文主体优先约650到800个中文字符", paper_prompt)
        self.assertIn("标题、英文摘录、图片图注和文章信息不计入正文主体", paper_prompt)
        self.assertIn("必须把正文主体压缩到650到800个中文字以内", paper_prompt)
        self.assertIn("每个小节约110到160个中文字", paper_prompt)
        self.assertIn("任何小节不得超过180字", paper_prompt)
        self.assertIn("通常设置3到4个主要小节", paper_prompt)
        self.assertIn("主动删去冗余背景、重复解释、低价值细节", paper_prompt)
        self.assertIn("不要机械截断句子", paper_prompt)
        self.assertIn(
            "像中文科技媒体编辑或科研作者整理一篇刚发表的研究",
            paper_prompt,
        )
        self.assertIn("直接陈述研究发现、数据和作者判断", paper_prompt)
        self.assertIn("专业准确，但不是论文摘要，也不要扮演老师给读者讲课", paper_prompt)
        self.assertNotIn("解释这个结果说明什么", paper_prompt)
        self.assertIn("作者比较了三组模式试验", paper_prompt)
        self.assertIn("去掉 Z 过程后，Y 的响应明显减弱", paper_prompt)
        self.assertIn("不同区域的结果也有明显差别", paper_prompt)
        self.assertIn("句长、段落节奏、信息密度和自然推进方式", paper_prompt)
        self.assertIn("不把它当作固定模板", paper_prompt)
        self.assertIn("A、B、X、Y、Z 都只是占位符", paper_prompt)
        self.assertIn("所有科学事实必须来自输入论文材料", paper_prompt)
        self.assertIn("禁止逐句翻译英文", paper_prompt)
        self.assertIn("必须逐字复制自paper_text", paper_prompt)
        self.assertIn("优先加入2到3组英文短引", paper_prompt)
        self.assertIn("每组保留1到2个连续且有信息量、语境完整的原文句子", paper_prompt)
        self.assertIn("不要按固定的全篇英文词数机械截断", paper_prompt)
        self.assertIn("每组应保持精炼，通常不超过80个英文词", paper_prompt)
        self.assertNotIn("每处最多25个英文词", paper_prompt)
        self.assertNotIn("全篇英文引用总量尽量控制在约25个英文词以内", paper_prompt)
        self.assertIn("引用格式为自然的Markdown引用块", paper_prompt)
        self.assertIn("找不到合适原文就少引或不引", paper_prompt)
        self.assertNotIn("> 原文：", paper_prompt)
        self.assertIn("最重要的2到4个发现", paper_prompt)
        self.assertIn(
            "> “The supplied paper states an exact scientific result.”",
            paper_markdown,
        )
        self.assertIn(
            "> “The ensemble spread remains stable across the tested regions while forecast errors decline during the validation period and improve seasonal predictability.”",
            paper_markdown,
        )
        self.assertNotIn("原文：", paper_markdown)
        self.assertNotIn("This quotation was invented by the model.", paper_markdown)
        self.assertIn("约1000到2000中文字", news_prompt)
        self.assertNotIn("约800到1200个中文字符", news_prompt)
        self.assertNotIn("正文主体优先约650到800个中文字符", news_prompt)
        self.assertNotIn("像中文科技媒体编辑或科研作者", news_prompt)
        self.assertNotIn("作者比较了三组模式试验", news_prompt)
        self.assertNotIn("A、B、X、Y、Z 都只是占位符", news_prompt)

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

    def test_paper_pdf_fallback_runs_after_html_download_failure(self):
        async def check():
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                settings = replace(
                    load_settings(),
                    database_path=root / "paper-fallback.db",
                )
                pipeline = NewsPipeline(settings)
                article_id = pipeline.db.upsert_article(
                    {
                        "source": "Nature Communications",
                        "url": "https://example.test/paper",
                        "canonical_url": "https://example.test/paper",
                        "title": "Atmospheric circulation paper",
                        "summary": "Atmospheric circulation mechanism",
                        "published_at": "2026-08-25T00:00:00+00:00",
                        "doi": "10.1000/paper-fallback",
                        "journal": "Nature Communications",
                        "word_count": 800,
                        "status": "published_paper",
                        "discovered_at": "2026-08-25T00:00:00+00:00",
                    }
                )
                dossier = {
                    "id": article_id,
                    "rank": 1,
                    "date": "2026-08-25",
                    "content_type": PAPER_CONTENT,
                    "title": "Atmospheric circulation paper",
                    "title_cn": "大气环流论文",
                    "summary": "Atmospheric circulation mechanism",
                    "text": "Paper text",
                    "url": "https://example.test/paper",
                    "doi": "10.1000/paper-fallback",
                    "journal": "Nature Communications",
                    "authors": ["Author One"],
                    "openalex": {
                        "journal": "Nature Communications",
                        "license": "CC BY 4.0",
                    },
                    "images": [
                        {
                            "url": "https://example.test/html-figure.png",
                            "local_path": "",
                            "caption": "Complete Figure 1",
                            "credit": "Author One",
                            "license": "CC BY 4.0",
                            "publishable": True,
                            "image_source": "html_figure",
                            "image_role": "figure",
                        }
                    ],
                }

                async def fake_details(_rank, _date, _content_type=None):
                    return copy.deepcopy(dossier)

                def fake_markdown(value, _settings, destination):
                    destination.mkdir(parents=True, exist_ok=True)
                    markdown = destination / "article.md"
                    metadata = destination / "metadata.json"
                    markdown.write_text(
                        "# 大气环流论文\n\nAtmospheric circulation paper body",
                        encoding="utf-8",
                    )
                    metadata.write_text("{}", encoding="utf-8")
                    return markdown, metadata

                def fake_download(records, _destination):
                    return [
                        {
                            **record,
                            "local_path": "",
                            "publishable": False,
                            "reason": "download failed",
                        }
                        for record in records
                    ]

                def fake_pdf_figures(_url, output_dir, **_kwargs):
                    output_dir.mkdir(parents=True, exist_ok=True)
                    path = output_dir / "figure-01.png"
                    path.write_bytes(b"png")
                    figure = {
                        "url": "https://example.test/paper.pdf#figure=1",
                        "local_path": str(path),
                        "caption": "Fig. 1 | Atmospheric circulation.",
                        "credit": "Author One",
                        "license": "CC BY 4.0",
                        "publishable": True,
                        "image_source": "pdf_figure",
                        "image_role": "figure",
                        "figure_number": 1,
                    }
                    return [figure], {"matched_figures": 1, "figures": [figure]}

                pipeline.paper_details = fake_details
                with (
                    patch("news.pipeline.article_output_dir", return_value=root / "article"),
                    patch("news.pipeline.download_publishable_images", side_effect=fake_download),
                    patch(
                        "news.pipeline.discover_pdf_source",
                        return_value={
                            "pdf_url": "https://example.test/paper.pdf",
                            "landing_url": "https://example.test/paper",
                            "license": "CC BY 4.0",
                            "license_url": "https://creativecommons.org/licenses/by/4.0/",
                        },
                    ),
                    patch(
                        "news.pipeline.extract_pdf_figures",
                        side_effect=fake_pdf_figures,
                    ) as extract_pdf,
                    patch("news.pipeline.generate_article_markdown", side_effect=fake_markdown),
                    patch("news.pipeline.generate_image_captions", return_value=["大气环流。"]),
                ):
                    generated = await pipeline.generate(
                        1,
                        "2026-08-25",
                        PAPER_CONTENT,
                    )

                self.assertEqual(extract_pdf.call_count, 1)
                self.assertTrue(generated["dossier"]["pdf_figure_fallback"]["attempted"])
                self.assertEqual(
                    generated["dossier"]["body_images"][0]["image_source"],
                    "pdf_figure",
                )

        asyncio.run(check())

    def test_paper_figures_are_inserted_with_original_numbers_and_caption_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "images"
            images_dir.mkdir()
            first_page = images_dir / "paper-first-page.png"
            first_page.write_bytes(b"png")
            figure_five = images_dir / "figure-05.png"
            figure_five.write_bytes(b"png")
            figure_two = images_dir / "figure-02.png"
            figure_two.write_bytes(b"png")
            markdown = root / "article.md"
            markdown.write_text(
                "![论文第一页](images/paper-first-page.png)\n\n"
                "## Atmospheric circulation\n\n"
                "Surface wind responds to large-scale atmospheric circulation.\n\n"
                "## Precipitation mechanisms\n\n"
                "Moisture transport controls extreme precipitation.\n",
                encoding="utf-8",
            )
            body_images = [
                {
                    "local_path": str(figure_five),
                    "url": "https://example.test/figure-5.png",
                    "image_role": "figure",
                    "figure_number": 5,
                    "caption": "Fig. 5 | Moisture transport and precipitation.",
                    "publishable": True,
                },
                {
                    "local_path": str(figure_two),
                    "url": "https://example.test/figure-2.png",
                    "image_role": "figure",
                    "figure_number": 2,
                    "caption": "Fig. 2 | Atmospheric circulation and surface wind.",
                    "publishable": True,
                },
            ]
            captions = _insert_paper_figures(
                markdown,
                body_images,
                ["大尺度环流与近地面风。", ""],
                {"content_type": PAPER_CONTENT},
            )
            text = markdown.read_text(encoding="utf-8")
            self.assertEqual(
                captions,
                ["Atmospheric circulation and surface wind.", "大尺度环流与近地面风。"],
            )
            self.assertLess(text.index("论文第一页"), text.index("## Atmospheric circulation"))
            self.assertGreater(
                text.index("Fig. 2 | Atmospheric circulation and surface wind."),
                text.index("## Atmospheric circulation"),
            )
            self.assertLess(
                text.index("Fig. 2 | Atmospheric circulation and surface wind."),
                text.index("Fig. 5 | 大尺度环流与近地面风。"),
            )
            self.assertGreater(
                text.index("Fig. 5 | 大尺度环流与近地面风。"),
                text.index("## Precipitation mechanisms"),
            )
            self.assertNotIn("图1.", text)

            (root / "metadata.json").write_text(
                json.dumps(
                    {
                        "body_images": body_images,
                        "paper_first_page": {"local_path": str(first_page)},
                    }
                ),
                encoding="utf-8",
            )
            summary = _paper_image_summary(markdown, body_images)
            self.assertEqual(
                summary,
                "可用图片：2\n正文使用：2\n论文首页：有",
            )

            plain_markdown = root / "plain-article.md"
            plain_markdown.write_text(
                "![论文第一页](images/paper-first-page.png)\n\n"
                "Surface wind responds to atmospheric circulation.\n\n"
                "The study compares several climate mechanisms.\n\n"
                "Moisture transport controls extreme precipitation.\n",
                encoding="utf-8",
            )
            _insert_paper_figures(
                plain_markdown,
                body_images,
                ["大尺度环流与近地面风。", ""],
                {"content_type": PAPER_CONTENT},
            )
            plain_lines = plain_markdown.read_text(encoding="utf-8").splitlines()
            figure_lines = [
                (index, line)
                for index, line in enumerate(plain_lines)
                if line.startswith("![Fig. ")
            ]
            self.assertEqual([line for _, line in figure_lines], [
                "![Fig. 2](images/figure-02.png)",
                "![Fig. 5](images/figure-05.png)",
            ])
            self.assertLess(figure_lines[0][0], figure_lines[1][0])

    def test_paper_numbered_figures_sort_before_monotonic_insertion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            captions_by_number = {
                1: "Baseline circulation response",
                3: "Moisture transport evidence",
                6: "Surface wind mechanism",
                8: "Future projection signal",
            }
            selected = []
            for number in (8, 1, 3, 6):
                path = root / f"figure-{number:02d}.png"
                path.write_bytes(b"png")
                selected.append(
                    {
                        "url": f"https://example.test/figure-{number}.png",
                        "local_path": str(path),
                        "image_role": "figure",
                        "figure_number": number,
                        "metadata_title": f"Figure {number} {captions_by_number[number]}",
                        "caption": captions_by_number[number],
                    }
                )

            _, body, _ = _select_article_images(
                selected,
                PAPER_CONTENT,
                " ".join(captions_by_number.values()),
            )
            self.assertEqual(
                [image["figure_number"] for image in body],
                [1, 3, 6, 8],
            )

            markdown = root / "ordered.md"
            markdown.write_text(
                "## Baseline circulation\n\nBaseline circulation response is identified.\n\n"
                "## Moisture transport\n\nMoisture transport evidence is quantified.\n\n"
                "## Surface wind\n\nThe surface wind mechanism is evaluated.\n\n"
                "## Future projection\n\nThe future projection signal is compared.\n",
                encoding="utf-8",
            )
            effective_captions = _insert_paper_figures(
                markdown,
                selected,
                [f"图{number}说明。" for number in (8, 1, 3, 6)],
                {"content_type": PAPER_CONTENT},
            )
            text = markdown.read_text(encoding="utf-8")
            figure_numbers = [
                int(line.split("![Fig. ", 1)[1].split("]", 1)[0])
                for line in text.splitlines()
                if line.startswith("![Fig. ")
            ]
            self.assertEqual(figure_numbers, [1, 3, 6, 8])
            self.assertEqual(
                effective_captions,
                ["图1说明。", "图3说明。", "图6说明。", "图8说明。"],
            )
            figure_positions = [text.index(f"![Fig. {number}]") for number in figure_numbers]
            self.assertEqual(figure_positions, sorted(figure_positions))
            for number, heading in (
                (1, "## Baseline circulation"),
                (3, "## Moisture transport"),
                (6, "## Surface wind"),
                (8, "## Future projection"),
            ):
                self.assertGreater(
                    text.index(f"![Fig. {number}]"),
                    text.index(heading),
                )

    def test_paper_image_allocation_covers_later_sections_before_four_image_cap(self):
        context = (
            "## Ensemble spread\n\n"
            "The ensemble spread quantifies forecast errors across the experiments.\n\n"
            "## Circulation and precipitation\n\n"
            "Large-scale circulation controls regional precipitation changes.\n\n"
            "## Projection and attribution\n\n"
            "Figure 4 and Figure 5 show future projection warming trends explained by radiative forcing.\n"
        )
        captions = {
            1: "Ensemble spread and forecast errors.",
            2: "Forecast uncertainty in ensemble spread.",
            3: "Circulation and precipitation response.",
            4: "Future projection warming trend.",
            5: "Projection warming trend attribution.",
            6: "Ocean chlorophyll concentration.",
        }
        images = [
            {
                "url": f"https://example.test/{number}.png",
                "local_path": f"/tmp/{number}.png",
                "image_role": "figure",
                "figure_number": number,
                "caption": captions[number],
            }
            for number in (1, 2, 3, 6, 4, 5)
        ]
        allocation = {}
        _, selected, _ = _select_article_images(
            images,
            PAPER_CONTENT,
            context,
            allocation,
        )
        selected_numbers = [image["figure_number"] for image in selected]
        self.assertEqual(len(selected), 4)
        self.assertEqual(selected_numbers, [1, 3, 4, 5])
        self.assertNotEqual(selected_numbers, [1, 2, 3, 6])
        self.assertEqual(allocation["input_image_count"], 6)
        self.assertEqual(allocation["max_images"], 4)
        self.assertEqual(
            [section["section"] for section in allocation["sections"]],
            ["Ensemble spread", "Circulation and precipitation", "Projection and attribution"],
        )
        self.assertEqual(
            [section["selected_figures"] for section in allocation["sections"]],
            [["Fig. 1"], ["Fig. 3"], ["Fig. 4", "Fig. 5"]],
        )
        self.assertEqual(
            [candidate["figure"] for candidate in allocation["sections"][0]["candidates"]],
            ["Fig. 1", "Fig. 2"],
        )
        discarded = {item["figure"]: item["reason"] for item in allocation["discarded_figures"]}
        self.assertIn("Fig. 2", discarded)
        self.assertIn("Fig. 6", discarded)
        self.assertIn("全局最多4张", discarded["Fig. 2"])
        self.assertIn("没有足够的正文对应关系", discarded["Fig. 6"])

        with tempfile.TemporaryDirectory() as tmp:
            markdown = Path(tmp) / "article.md"
            markdown.write_text(context, encoding="utf-8")
            insertion_dossier = {
                "content_type": PAPER_CONTENT,
                "body_images": selected,
            }
            _insert_paper_figures(
                markdown,
                selected,
                ["" for _ in selected],
                insertion_dossier,
            )
            rendered = markdown.read_text(encoding="utf-8")
            self.assertEqual(
                [
                    int(line.split("![Fig. ", 1)[1].split("]", 1)[0])
                    for line in rendered.splitlines()
                    if line.startswith("![Fig. ")
                ],
                [1, 3, 4, 5],
            )
            self.assertGreater(rendered.index("![Fig. 1]"), rendered.index("ensemble spread"))
            self.assertGreater(rendered.index("![Fig. 3]"), rendered.index("regional precipitation"))
            self.assertGreater(rendered.index("![Fig. 4]"), rendered.index("future projection"))
            self.assertGreater(rendered.index("![Fig. 5]"), rendered.index("![Fig. 4]"))

    def test_paper_scoring_prefers_scientific_and_source_evidence_over_structure(self):
        context = (
            "## Nb proxy for Holocene hydroclimate\n\n"
            "The niobium record tracks Holocene hydroclimate with GRIP and DYE-3.\n\n"
            "## Positive NAO response\n\n"
            "Positive NAO phases produce wetter but colder conditions and alter precipitation.\n"
        )
        source = (
            "The niobium record is compared with GRIP temperatures and DYE-3 isotope data (Fig. 3).\n\n"
            "Positive NAO phases correspond to higher precipitation and lower temperature in southwestern Greenland (Fig. 4).\n"
        )
        captions = {
            1: "Regional geology and landscape setting.",
            2: "Sediment properties and IRD concentration during Holocene climate evolution.",
            3: "Niobium content as a marker for Holocene hydroclimate variability compared with GRIP and DYE-3.",
            4: "NAO-driven anomalies in temperature and precipitation.",
            5: "Niobium compared with NAO reconstructions and regional ice accumulation.",
        }
        images = [
            {
                "url": f"https://example.test/{number}.png",
                "local_path": f"/tmp/{number}.png",
                "image_role": "figure",
                "figure_number": number,
                "caption": captions[number],
            }
            for number in (1, 2, 3, 4, 5)
        ]
        allocation = {}
        _, selected, _ = _select_article_images(
            images,
            PAPER_CONTENT,
            context,
            allocation,
            source,
        )
        self.assertEqual(
            [image["figure_number"] for image in selected[:4]],
            [2, 3, 4, 5],
        )
        nao_section = allocation["sections"][1]
        nao_scores = {
            score["figure"]: score for score in nao_section["figure_scores"]
        }
        self.assertEqual(
            max(nao_scores, key=lambda figure: nao_scores[figure]["score"]),
            "Fig. 4",
        )
        self.assertEqual(nao_section["selected_figures"][0], "Fig. 4")
        self.assertGreater(nao_scores["Fig. 4"]["source_score"], nao_scores["Fig. 2"]["source_score"])
        self.assertEqual(nao_scores["Fig. 4"]["match_method"], "source_paragraph")
        self.assertLessEqual(nao_scores["Fig. 2"]["structural_score"], 8)
        self.assertEqual(
            max(
                allocation["sections"][0]["figure_scores"],
                key=lambda score: score["score"],
            )["figure"],
            "Fig. 3",
        )

    def test_paper_source_mapping_is_section_local(self):
        source = (
            "Results\n"
            "Narsaq Sound deglaciated before surrounding land\n"
            "The IRD layer indicates that the sound became periodically ice-free (Fig. 2).\n"
            "Niobium as marker for hydroclimate change in southern Greenland\n"
            "The niobium record agrees with GRIP temperature and DYE-3 isotope data (Fig. 3).\n"
            "Hydroclimate in southern Greenland driven by NAO variability\n"
            "Positive NAO phases produce wetter but colder conditions (Fig. 4).\n"
            "Late Holocene ice accumulation and NAO reconstruction\n"
            "The Little Ice Age accumulation history is compared with the NAO record (Fig. 5).\n"
        )
        nb_section = (
            "## Nb proxy for Holocene hydroclimate\n"
            "The niobium record tracks Holocene hydroclimate and agrees with GRIP and DYE-3."
        )
        deglaciation_section = (
            "## Early Holocene deglaciation\n"
            "The IRD layer shows that the sound became periodically ice-free."
        )
        nb_matches = _paper_match_source_paragraphs(nb_section, source, 0, 2)
        deglaciation_matches = _paper_match_source_paragraphs(
            deglaciation_section,
            source,
            1,
            2,
        )
        self.assertEqual(
            [paragraph["id"] for paragraph in nb_matches],
            ["source-4"],
        )
        self.assertEqual(nb_matches[0]["figure_references"], [3])
        self.assertEqual(
            [paragraph["id"] for paragraph in deglaciation_matches],
            ["source-2"],
        )
        self.assertEqual(deglaciation_matches[0]["figure_references"], [2])

    def test_paper_insertion_uses_allocated_section_heading_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = []
            for number in (1, 2, 4):
                path = root / f"figure-{number}.png"
                path.write_bytes(b"png")
                images.append(
                    {
                        "url": f"https://example.test/{number}.png",
                        "local_path": str(path),
                        "image_role": "figure",
                        "figure_number": number,
                        "caption": f"Figure {number} caption",
                    }
                )
            markdown = root / "article.md"
            markdown.write_text(
                "## First section\n\n"
                "First正文段落。\n\n"
                "> quote must remain inside the first section.\n\n"
                "![existing placeholder](placeholder.png)\n\n"
                "## Second section\n\n"
                "Second正文段落。\n\n"
                "## Third section\n\n"
                "Third正文段落。\n",
                encoding="utf-8",
            )
            allocation = {
                "sections": [
                    {
                        "section_index": 0,
                        "section": "First section",
                        "selected_figures": ["Fig. 1"],
                    },
                    {
                        # Deliberately stale indexes model an insertion offset;
                        # section headings remain the authoritative targets.
                        "section_index": 0,
                        "section": "Second section",
                        "selected_figures": ["Fig. 2"],
                    },
                    {
                        "section_index": 1,
                        "section": "Third section",
                        "selected_figures": ["Fig. 4"],
                    },
                ]
            }
            dossier = {
                "content_type": PAPER_CONTENT,
                "paper_image_allocation": allocation,
            }
            _insert_paper_figures(
                markdown,
                images,
                ["", "", ""],
                dossier,
            )
            rendered = markdown.read_text(encoding="utf-8")
            headings = [
                "## First section",
                "## Second section",
                "## Third section",
            ]
            positions = {heading: rendered.index(heading) for heading in headings}
            positions["END"] = len(rendered)
            figure_positions = {
                number: rendered.index(f"![Fig. {number}]") for number in (1, 2, 4)
            }
            self.assertLess(positions[headings[0]], figure_positions[1])
            self.assertLess(figure_positions[1], positions[headings[1]])
            self.assertLess(positions[headings[1]], figure_positions[2])
            self.assertLess(figure_positions[2], positions[headings[2]])
            self.assertLess(positions[headings[2]], figure_positions[4])
            self.assertLess(figure_positions[4], positions["END"])
            self.assertEqual(
                allocation["final_inserted_sections"],
                [
                    {
                        "figure": "Fig. 1",
                        "section_index": 0,
                        "section": "First section",
                    },
                    {
                        "figure": "Fig. 2",
                        "section_index": 1,
                        "section": "Second section",
                    },
                    {
                        "figure": "Fig. 4",
                        "section_index": 2,
                        "section": "Third section",
                    },
                ],
            )
            self.assertEqual(
                [image["figure_number"] for image in dossier["body_images"]],
                [1, 2, 4],
            )

    def test_paper_mapping_aggregates_major_section_without_quote_slots(self):
        context = (
            "## Main result\n\n"
            "The ensemble spread tracks forecast errors in the tested region.\n\n"
            "> “The ensemble spread tracks forecast errors in the tested region.”\n\n"
            "A later paragraph reports the same signal across seasons.\n\n"
            "## Mechanism\n\n"
            "Circulation controls regional precipitation through moisture transport.\n"
        )
        images = [
            {
                "url": "https://example.test/1.png",
                "local_path": "/tmp/1.png",
                "image_role": "figure",
                "figure_number": 1,
                "caption": "Ensemble spread and forecast errors.",
            },
            {
                "url": "https://example.test/2.png",
                "local_path": "/tmp/2.png",
                "image_role": "figure",
                "figure_number": 2,
                "caption": "Circulation and precipitation.",
            },
        ]
        allocation = {}
        _select_article_images(images, PAPER_CONTENT, context, allocation)
        self.assertEqual(
            [section["section"] for section in allocation["sections"]],
            ["Main result", "Mechanism"],
        )
        self.assertEqual(
            [section["candidates"][0]["figure"] for section in allocation["sections"]],
            ["Fig. 1", "Fig. 2"],
        )

    def test_paper_figures_match_specific_paragraphs_and_skip_unrelated_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            markdown = root / "matched.md"
            markdown.write_text(
                "## Forecast uncertainty\n\n"
                "The ensemble spread quantifies forecast errors across the experiments.\n\n"
                "## Hydroclimate response\n\n"
                "Large-scale circulation controls regional precipitation changes.\n\n"
                "## Background\n\n"
                "The paper describes the observational period.\n",
                encoding="utf-8",
            )
            images = []
            captions = {
                1: "Ensemble spread and forecast errors.",
                2: "Circulation and precipitation response.",
                4: "Ocean chlorophyll concentration.",
            }
            for number in (4, 2, 1):
                path = root / f"figure-{number:02d}.png"
                path.write_bytes(b"png")
                images.append(
                    {
                        "url": f"https://example.test/{number}.png",
                        "local_path": str(path),
                        "image_role": "figure",
                        "figure_number": number,
                        "caption": captions[number],
                    }
                )
            dossier = {"content_type": PAPER_CONTENT, "body_images": images}
            effective = _insert_paper_figures(
                markdown,
                images,
                ["", "", ""],
                dossier,
            )
            text = markdown.read_text(encoding="utf-8")
            self.assertEqual(
                [image["figure_number"] for image in dossier["body_images"]],
                [1, 2],
            )
            self.assertEqual(effective, [captions[1], captions[2]])
            self.assertEqual(
                [
                    int(line.split("![Fig. ", 1)[1].split("]", 1)[0])
                    for line in text.splitlines()
                    if line.startswith("![Fig. ")
                ],
                [1, 2],
            )
            self.assertGreater(
                text.index("![Fig. 1]"),
                text.index("ensemble spread quantifies forecast errors"),
            )
            self.assertGreater(
                text.index("![Fig. 2]"),
                text.index("circulation controls regional precipitation"),
            )
            self.assertLess(text.index("![Fig. 1]"), text.index("![Fig. 2]"))
            self.assertNotIn("Ocean chlorophyll", text)

    def test_paper_images_without_figure_numbers_keep_relevance_order(self):
        ordinary_images = [
            {
                "url": "https://example.test/supporting.png",
                "local_path": "/tmp/supporting.png",
                "image_role": "article_image",
                "caption": "Supporting appendix material",
            },
            {
                "url": "https://example.test/hero.png",
                "local_path": "/tmp/hero.png",
                "image_role": "hero",
                "caption": "Central circulation result",
            },
        ]
        _, body, _ = _select_article_images(
            ordinary_images,
            PAPER_CONTENT,
            "Central circulation result",
        )
        self.assertEqual(
            [image["url"] for image in body],
            [
                "https://example.test/hero.png",
                "https://example.test/supporting.png",
            ],
        )

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
            draft_title = _paper_draft_title(metadata, title_cn)
            self.assertEqual(
                draft_title,
                "Communications Earth & Environment："
                "夏季风变率驱动中国北方毛乌素沙地绿化和新石器时代社会变迁",
            )
            self.assertNotIn("最新成果", draft_title)
            self.assertEqual(
                _paper_draft_title({"content_type": PAPER_CONTENT}, title_cn),
                title_cn,
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

    def test_wiley_pdf_discovery_and_existing_pdf_validation(self):
        landing = "https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2026GL125002"
        pdf_url = "https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2026GL125002"

        def discover(html):
            response = MagicMock()
            response.url = landing
            response.headers = {"content-type": "text/html"}
            response.text = html
            response.raise_for_status.return_value = None
            client = MagicMock()
            client.__enter__.return_value = client
            client.__exit__.return_value = False
            client.get.return_value = response
            with patch("papers.pdf_figures.httpx.Client", return_value=client):
                return discover_pdf_source(
                    landing,
                    "10.1029/2026GL125002",
                )

        linked = discover(
            '<html><a href="/doi/pdf/10.1029/2026GL125002">Download PDF</a></html>'
        )
        self.assertEqual(linked["pdf_url"], pdf_url)
        constructed = discover("<html><body>No directly parseable link</body></html>")
        self.assertEqual(constructed["pdf_url"], pdf_url)

        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "paper.pdf"
            response = MagicMock(content=b"<html>Access denied</html>")
            response.raise_for_status.return_value = None
            client = MagicMock()
            client.__enter__.return_value = client
            client.__exit__.return_value = False
            client.get.return_value = response
            with patch("papers.pdf_figures.httpx.Client", return_value=client):
                with self.assertRaisesRegex(ValueError, "not a PDF"):
                    _download_pdf(pdf_url, destination)

            response.content = b"%PDF-1.7 test"
            with patch("papers.pdf_figures.httpx.Client", return_value=client):
                _download_pdf(pdf_url, destination)
            self.assertTrue(destination.read_bytes().startswith(b"%PDF"))

    def test_wiley_tdm_is_skipped_without_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "paper.pdf"
            with (
                patch(
                    "papers.pdf_figures._download_pdf",
                    side_effect=RuntimeError("HTTP 403"),
                ),
                patch("papers.pdf_figures.httpx.Client") as client,
            ):
                result = download_pdf_with_wiley_tdm(
                    "https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/example",
                    destination,
                    doi="10.1029/example",
                )
            self.assertFalse(result["success"])
            self.assertFalse(result["attempted"])
            client.assert_not_called()
            self.assertFalse(destination.exists())

    def test_wiley_tdm_uses_encoded_doi_and_validates_pdf(self):
        class Response:
            def __init__(self, status_code, content):
                self.status_code = status_code
                self.content = content

        client = MagicMock()
        client.__enter__.return_value = client
        client.__exit__.return_value = False
        client.get.return_value = Response(200, b"%PDF-1.7 redirected PDF")
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "paper.pdf"
            with (
                patch(
                    "papers.pdf_figures._download_pdf",
                    side_effect=RuntimeError("HTTP 403"),
                ),
                patch("papers.pdf_figures.httpx.Client", return_value=client) as http_client,
            ):
                result = download_pdf_with_wiley_tdm(
                    "https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2025GL121477",
                    destination,
                    doi="10.1029/2025GL121477",
                    token="test-tdm-token",
                )
            self.assertTrue(result["success"])
            self.assertEqual(result["source"], "wiley_tdm")
            self.assertEqual(result["status"], 200)
            http_client.assert_called_once_with(
                timeout=60.0,
                follow_redirects=True,
                trust_env=True,
            )
            request_url, request_kwargs = client.get.call_args.args[0], client.get.call_args.kwargs
            self.assertEqual(
                request_url,
                "https://api.wiley.com/onlinelibrary/tdm/v1/articles/10.1029%2F2025GL121477",
            )
            self.assertEqual(
                request_kwargs["headers"],
                {"Wiley-TDM-Client-Token": "test-tdm-token"},
            )
            self.assertTrue(destination.read_bytes().startswith(b"%PDF"))

    def test_wiley_tdm_rejects_html_and_non_wiley_urls(self):
        class Response:
            status_code = 200
            content = b"<html>Access denied</html>"

        client = MagicMock()
        client.__enter__.return_value = client
        client.__exit__.return_value = False
        client.get.return_value = Response()
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "paper.pdf"
            with (
                patch(
                    "papers.pdf_figures._download_pdf",
                    side_effect=RuntimeError("HTTP 403"),
                ),
                patch("papers.pdf_figures.httpx.Client", return_value=client),
            ):
                result = download_pdf_with_wiley_tdm(
                    "https://example.org/doi/pdf/example",
                    destination,
                    doi="10.1029/example",
                    token="test-tdm-token",
                )
            self.assertFalse(result["success"])
            self.assertFalse(result["attempted"])
            self.assertFalse(destination.exists())

            with (
                patch(
                    "papers.pdf_figures._download_pdf",
                    side_effect=RuntimeError("HTTP 403"),
                ),
                patch("papers.pdf_figures.httpx.Client", return_value=client),
            ):
                result = download_pdf_with_wiley_tdm(
                    "https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/example",
                    destination,
                    doi="10.1029/example",
                    token="test-tdm-token",
                )
            self.assertFalse(result["success"])
            self.assertEqual(result["source"], "wiley_tdm")
            self.assertIn("not a PDF", result["error"])
            self.assertFalse(destination.exists())

    def test_wiley_tdm_retries_5xx_once_and_stops_on_403(self):
        class Response:
            def __init__(self, status_code, content=b""):
                self.status_code = status_code
                self.content = content

        client = MagicMock()
        client.__enter__.return_value = client
        client.__exit__.return_value = False
        client.get.side_effect = [Response(503), Response(403)]
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch(
                    "papers.pdf_figures._download_pdf",
                    side_effect=RuntimeError("HTTP 403"),
                ),
                patch("papers.pdf_figures.httpx.Client", return_value=client),
                patch("papers.pdf_figures.time.sleep") as sleep,
            ):
                result = download_pdf_with_wiley_tdm(
                    "https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/example",
                    Path(tmp) / "paper.pdf",
                    doi="10.1029/example",
                    token="test-tdm-token",
                )
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], 403)
        self.assertEqual(
            client.get.call_args_list[0].args[0],
            "https://api.wiley.com/onlinelibrary/tdm/v1/articles/10.1029%2Fexample",
        )
        self.assertEqual(
            client.get.call_args_list[0].kwargs["headers"],
            {"Wiley-TDM-Client-Token": "test-tdm-token"},
        )
        self.assertEqual(client.get.call_count, 2)
        sleep.assert_called_once_with(1)

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

    def test_paper_figure_numbers_prevent_false_deduplication(self):
        figure_one = {
            "url": "https://example.test/figure-1.png",
            "image_role": "figure",
            "figure_number": 1,
            "caption": "Atmospheric circulation response under warming",
        }
        figure_two = {
            "url": "https://example.test/figure-2.png",
            "image_role": "figure",
            "figure_number": 2,
            "caption": "Atmospheric circulation response under warming",
        }
        self.assertFalse(_images_redundant(figure_one, figure_two))
        self.assertTrue(
            _images_redundant(
                figure_one,
                {**figure_two, "figure_number": 1},
            )
        )
        self.assertTrue(
            _images_redundant(
                figure_one,
                {**figure_two, "url": figure_one["url"]},
            )
        )
        self.assertTrue(
            _images_redundant(
                {**figure_one, "image_role": "article_image", "figure_number": None},
                {**figure_two, "image_role": "article_image", "figure_number": None},
            )
        )

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

    def test_springer_html_figures_parse_full_caption_and_number(self):
        html = """
        <div class="c-article-section__figure" id="figure-1">
          <p class="c-article-section__figure-caption">Fig. 1</p>
          <div class="c-article-section__figure-description">
            Full caption describing atmospheric circulation and surface wind.
          </div>
          <figure><img src="/article/figure-1.png" alt="Figure one"></figure>
        </div>
        <div class="c-article-section__figure" id="figure-2">
          <p class="c-article-section__figure-caption">Fig. 2</p>
          <div class="c-article-section__figure-description">
            Full caption describing precipitation and temperature feedbacks.
          </div>
          <figure><img src="/article/figure-2.png" alt="Figure two"></figure>
        </div>
        """
        figures = discover_figure_images(
            html,
            "https://link.springer.com/article/10.1007/example",
            "CC BY 4.0",
        )
        self.assertEqual(len(figures), 2)
        self.assertEqual([item["figure_number"] for item in figures], [1, 2])
        self.assertEqual(
            figures[0]["caption"],
            "Full caption describing atmospheric circulation and surface wind.",
        )
        self.assertEqual(
            figures[1]["caption"],
            "Full caption describing precipitation and temperature feedbacks.",
        )
        self.assertTrue(all(item["publishable"] for item in figures))
        self.assertTrue(all(item["image_role"] == "figure" for item in figures))

    def test_wiley_html_figures_parse_title_caption_and_large_image_url(self):
        html = """
        <div class="article-section__full">
          <figure>
            <div class="figure__title">Figure 1</div>
            <div class="figure__caption-text">Large-scale circulation response.</div>
            <img data-lg-src="/images/figure-1-large.png" src="/images/figure-1-small.png"
                 alt="Circulation response">
          </figure>
          <figure>
            <div class="figure__title">Figure 2</div>
            <div class="figure__caption-text">Surface wind anomalies.</div>
            <img src="images/figure-2.png" alt="Surface wind anomalies">
          </figure>
        </div>
        """
        figures = discover_figure_images(
            html,
            "https://agupubs.onlinelibrary.wiley.com/doi/10.1029/example/",
            "CC BY 4.0",
        )
        self.assertEqual(len(figures), 2)
        self.assertEqual([item["figure_number"] for item in figures], [1, 2])
        self.assertEqual(
            [item["caption"] for item in figures],
            ["Large-scale circulation response.", "Surface wind anomalies."],
        )
        self.assertEqual(
            figures[0]["url"],
            "https://agupubs.onlinelibrary.wiley.com/images/figure-1-large.png",
        )
        self.assertEqual(figures[0]["image_url_source"], "data-lg-src")
        self.assertEqual(
            figures[1]["url"],
            "https://agupubs.onlinelibrary.wiley.com/doi/10.1029/example/images/figure-2.png",
        )
        self.assertEqual(figures[1]["image_url_source"], "src")
        self.assertTrue(all(item["publishable"] for item in figures))
        self.assertTrue(all(item["image_role"] == "figure" for item in figures))

    def test_news_public_search_preserves_metadata_and_rejects_unknown_license(self):
        licensed = {
            "original_url": "https://upload.wikimedia.org/marine-heat-wave.jpg",
            "source_url": "https://commons.wikimedia.org/wiki/File:Marine_heat_wave.jpg",
            "source": "Wikimedia Commons",
            "license_short_name": "CC BY-SA 4.0",
            "metadata_title": "Marine heat wave and ocean acidification",
            "description": "Marine heat wave conditions in the ocean",
            "credit": "Example Author",
        }
        unknown = {
            "original_url": "https://example.test/unknown.jpg",
            "source": "Unknown archive",
            "metadata_title": "Marine heat wave and ocean acidification",
            "description": "Unknown-license image",
        }
        mapped = normalize_search_result(licensed)
        self.assertEqual(mapped["provider"], "Wikimedia Commons")
        self.assertEqual(mapped["source"], "Wikimedia Commons")
        self.assertEqual(mapped["image_source"], "public_search")
        self.assertEqual(mapped["license"], "CC BY-SA 4.0")
        self.assertEqual(mapped["url"], licensed["original_url"])
        self.assertEqual(mapped["source_url"], licensed["source_url"])
        self.assertEqual(mapped["credit"], "Example Author")
        self.assertTrue(mapped["publishable"])
        self.assertFalse(normalize_search_result(unknown)["publishable"])

        with (
            patch(
                "images.search.search_wikimedia_commons",
                return_value=[licensed, unknown],
            ),
            patch("images.search.search_nasa_images", return_value=[]),
        ):
            approved = search_public_images(
                ["marine heat wave", "ocean acidification"],
                max_images=5,
            )

        self.assertEqual(len(approved), 1)
        self.assertEqual(approved[0]["provider"], "Wikimedia Commons")
        self.assertTrue(approved[0]["publishable"])
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "marine-heat-wave.jpg"
            image_path.write_bytes(b"image")
            approved[0]["local_path"] = str(image_path)
            _, body_images, _ = _select_article_images(approved, POPULAR_CONTENT)
            self.assertEqual(len(body_images), 1)

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

    def test_model_can_select_fewer_than_ten_and_keep_order(self):
        settings = replace(
            load_settings(),
            model_base_url="https://model.example/v1",
            model_api_key="test-key",
            model_name="test-model",
        )
        candidates = [
            {"title": f"English title {index}", "source": "Nature News", "score": 20 - index}
            for index in range(1, 8)
        ]
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "items": [
                                    {"index": 5, "title_cn": "中文标题5"},
                                    {"index": 2, "title_cn": "中文标题2"},
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
        self.assertIn("最多返回10篇，可以少于10篇", system_prompt)
        self.assertIn("禁止凑数", system_prompt)
        self.assertEqual(
            [item["title_cn"] for item in selected],
            ["中文标题2", "中文标题5"],
        )
        self.assertEqual(
            [item["title"] for item in selected],
            ["English title 2", "English title 5"],
        )

    def test_paper_llm_keeps_only_scores_two_and_three_without_filling(self):
        settings = replace(
            load_settings(),
            model_base_url="https://model.example/v1",
            model_api_key="test-key",
            model_name="test-model",
        )
        candidates = [
            {
                "title": f"Paper {index}",
                "summary": "Physical climate mechanism",
                "paper_local_score": 2,
            }
            for index in range(1, 13)
        ]
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "items": [
                                    {"index": 1, "score": 3, "title_cn": "核心论文"},
                                    {"index": 2, "score": 2, "title_cn": "相关论文"},
                                    {"index": 3, "score": 1, "title_cn": "外围论文"},
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
            selected, used_model, error = select_paper_top_ten(candidates, settings)

        self.assertTrue(used_model)
        self.assertEqual(error, "")
        system_prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn("纯对流动力学", system_prompt)
        self.assertIn("次季节/季节可预报性", system_prompt)
        self.assertEqual(len(selected), 2)
        self.assertEqual(
            [item["paper_relevance_score"] for item in selected],
            [3, 2],
        )
        self.assertEqual(
            [item["title_cn"] for item in selected],
            ["核心论文", "相关论文"],
        )

    def test_paper_refresh_expands_window_after_seen_filter(self):
        async def check():
            with tempfile.TemporaryDirectory() as tmp:
                settings = replace(
                    load_settings(),
                    database_path=Path(tmp) / "seen-window.db",
                    model_base_url="https://model.example/v1",
                    model_api_key="test-key",
                    model_name="test-model",
                    openalex_api_key="",
                )
                pipeline = NewsPipeline(settings)

                def make_item(index):
                    return {
                        "source": "Journal of Climate",
                        "url": f"https://example.test/window-{index}",
                        "canonical_url": f"https://example.test/window-{index}",
                        "title": f"Window paper {index}",
                        "summary": "Near-surface wind climate mechanism",
                        "published_at": "2026-08-26T00:00:00+00:00",
                        "doi": f"10.1000/window-{index}",
                        "journal": "Journal of Climate",
                        "word_count": 800,
                        "status": "discovered",
                        "discovered_at": "2026-08-26T00:00:00+00:00",
                    }

                seen_items = [make_item(index) for index in range(2)]
                seen_ids = [pipeline.db.upsert_article(item) for item in seen_items]
                pipeline.db.add_seen_candidates("2026-08-26", PAPER_CONTENT, seen_ids)
                windows = {
                    48: seen_items + [make_item(index) for index in range(2, 8)],
                    168: [make_item(index) for index in range(8, 20)],
                }
                calls = []

                def fake_fetch(_path, hours):
                    calls.append(hours)
                    return copy.deepcopy(windows.get(hours, [])), [], {"test": len(windows.get(hours, []))}

                pipeline._extract_shortlist = lambda values: asyncio.sleep(
                    0, result=copy.deepcopy(values)
                )
                pipeline._published_papers = lambda values, _date: asyncio.sleep(
                    0,
                    result=[dict(value, paper_local_score=2) for value in values],
                )
                with (
                    patch("news.pipeline.fetch_all_feeds", side_effect=fake_fetch),
                    patch(
                        "news.pipeline.select_paper_top_ten",
                        side_effect=lambda values, _settings: (
                            [dict(value, paper_relevance_score=3) for value in values[:10]],
                            True,
                            "",
                        ),
                    ),
                    patch(
                        "news.pipeline.translate_paper_titles",
                        side_effect=lambda values, _settings: (
                            ["" for _ in values],
                            True,
                            "",
                        ),
                    ),
                    patch("news.pipeline.deduplicate", side_effect=lambda values: values),
                ):
                    selected = await pipeline.refresh(
                        "2026-08-26",
                        PAPER_CONTENT,
                        exclude_seen=True,
                    )

                self.assertEqual(calls, [48, 168])
                self.assertEqual(len(selected), 10)
                self.assertTrue(
                    all(item["id"] not in set(seen_ids) for item in selected)
                )

        asyncio.run(check())

    def test_paper_ai_selection_processes_multiple_30_item_batches(self):
        async def check():
            with tempfile.TemporaryDirectory() as tmp:
                settings = replace(
                    load_settings(),
                    database_path=Path(tmp) / "ai-batches.db",
                    model_base_url="https://model.example/v1",
                    model_api_key="test-key",
                    model_name="test-model",
                    openalex_api_key="",
                )
                pipeline = NewsPipeline(settings)
                items = [
                    {
                        "source": "Journal of Climate",
                        "url": f"https://example.test/ai-{index}",
                        "canonical_url": f"https://example.test/ai-{index}",
                        "title": f"Distinct paper {index}",
                        "summary": "Near-surface wind climate mechanism",
                        "published_at": "2026-08-26T00:00:00+00:00",
                        "doi": f"10.1000/ai-{index}",
                        "journal": "Journal of Climate",
                        "word_count": 800,
                        "status": "discovered",
                        "discovered_at": "2026-08-26T00:00:00+00:00",
                    }
                    for index in range(60)
                ]
                pipeline._extract_shortlist = lambda values: asyncio.sleep(
                    0, result=copy.deepcopy(values)
                )
                pipeline._published_papers = lambda values, _date: asyncio.sleep(
                    0,
                    result=[dict(value, paper_local_score=2) for value in values],
                )
                batch_sizes = []

                def select_batch(values, _settings):
                    batch_sizes.append(len(values))
                    keep = 3 if len(batch_sizes) == 1 else 7
                    return [
                        dict(value, paper_relevance_score=3)
                        for value in values[:keep]
                    ], True, ""

                with (
                    patch("news.pipeline.fetch_all_feeds", return_value=(items, [], {"test": 60})),
                    patch("news.pipeline.select_paper_top_ten", side_effect=select_batch),
                    patch(
                        "news.pipeline.translate_paper_titles",
                        side_effect=lambda values, _settings: (
                            ["" for _ in values],
                            True,
                            "",
                        ),
                    ),
                    patch("news.pipeline.deduplicate", side_effect=lambda values: values),
                ):
                    selected = await pipeline.refresh("2026-08-26", PAPER_CONTENT)

                self.assertEqual(batch_sizes, [30, 30])
                self.assertEqual(len(selected), 10)
                self.assertEqual(
                    pipeline.last_paper_discovery_stats["ai_examined"],
                    60,
                )
                self.assertEqual(pipeline.last_paper_discovery_stats["ai_kept"], 10)

        asyncio.run(check())

    def test_paper_refresh_returns_six_after_30_day_pool_is_exhausted(self):
        async def check():
            with tempfile.TemporaryDirectory() as tmp:
                settings = replace(
                    load_settings(),
                    database_path=Path(tmp) / "six.db",
                    model_base_url="https://model.example/v1",
                    model_api_key="test-key",
                    model_name="test-model",
                    openalex_api_key="",
                )
                pipeline = NewsPipeline(settings)
                items = [
                    {
                        "source": "Journal of Climate",
                        "url": f"https://example.test/six-{index}",
                        "canonical_url": f"https://example.test/six-{index}",
                        "title": f"Strict paper {index}",
                        "summary": "Near-surface wind climate mechanism",
                        "published_at": "2026-08-26T00:00:00+00:00",
                        "doi": f"10.1000/six-{index}",
                        "journal": "Journal of Climate",
                        "word_count": 800,
                        "status": "discovered",
                        "discovered_at": "2026-08-26T00:00:00+00:00",
                    }
                    for index in range(6)
                ]
                calls = []

                def fake_fetch(_path, hours):
                    calls.append(hours)
                    return copy.deepcopy(items), [], {"test": len(items)}

                pipeline._extract_shortlist = lambda values: asyncio.sleep(
                    0, result=copy.deepcopy(values)
                )
                pipeline._published_papers = lambda values, _date: asyncio.sleep(
                    0,
                    result=[dict(value, paper_local_score=2) for value in values],
                )
                with (
                    patch("news.pipeline.fetch_all_feeds", side_effect=fake_fetch),
                    patch(
                        "news.pipeline.select_paper_top_ten",
                        side_effect=lambda values, _settings: (
                            [dict(value, paper_relevance_score=3) for value in values],
                            True,
                            "",
                        ),
                    ),
                    patch(
                        "news.pipeline.translate_paper_titles",
                        side_effect=lambda values, _settings: (
                            ["" for _ in values],
                            True,
                            "",
                        ),
                    ),
                    patch("news.pipeline.deduplicate", side_effect=lambda values: values),
                ):
                    selected = await pipeline.refresh("2026-08-26", PAPER_CONTENT)

                self.assertEqual(calls, [48, 168, 720])
                self.assertEqual(len(selected), 6)
                self.assertEqual(pipeline.last_paper_discovery_stats["final"], 6)

        asyncio.run(check())

    def test_paper_refresh_failure_keeps_same_day_last_known_good(self):
        async def check():
            with tempfile.TemporaryDirectory() as tmp:
                settings = replace(
                    load_settings(),
                    database_path=Path(tmp) / "last-good.db",
                    model_base_url="https://model.example/v1",
                    model_api_key="test-key",
                    model_name="test-model",
                    openalex_api_key="test-openalex-key",
                )
                pipeline = NewsPipeline(settings)
                old_id = pipeline.db.upsert_article(
                    {
                        "source": "Journal of Climate",
                        "url": "https://example.test/old",
                        "canonical_url": "https://example.test/old",
                        "title": "Previously selected paper",
                        "summary": "Near-surface wind climate mechanism",
                        "published_at": "2026-08-25T00:00:00+00:00",
                        "doi": "10.1000/old",
                        "journal": "Journal of Climate",
                        "word_count": 800,
                        "status": "published_paper",
                        "discovered_at": "2026-08-25T00:00:00+00:00",
                    }
                )
                pipeline.db.replace_candidates(
                    "2026-08-26",
                    [{"article_id": old_id, "score": 20, "title_cn": "昨日成功结果"}],
                    PAPER_CONTENT,
                )
                new_item = {
                    "source": "Journal of Climate",
                    "url": "https://example.test/new",
                    "canonical_url": "https://example.test/new",
                    "title": "New candidate B",
                    "summary": "Near-surface wind climate mechanism",
                    "published_at": "2026-08-26T00:00:00+00:00",
                    "doi": "10.1000/new",
                    "journal": "Journal of Climate",
                    "word_count": 800,
                    "status": "discovered",
                    "discovered_at": "2026-08-26T00:00:00+00:00",
                }
                pipeline._extract_shortlist = lambda items: asyncio.sleep(
                    0, result=copy.deepcopy(items)
                )
                pipeline._published_papers = lambda items, _date: asyncio.sleep(
                    0,
                    result=[dict(item, paper_local_score=2) for item in items],
                )
                with (
                    patch.object(
                        pipeline.openalex,
                        "discover_recent_papers",
                        return_value=[],
                    ),
                    patch(
                        "news.pipeline.fetch_all_feeds",
                        return_value=([new_item], [], {"test": 1}),
                    ),
                    patch(
                        "news.pipeline.select_paper_top_ten",
                        return_value=(
                            [dict(new_item, article_id=999, score=10)],
                            False,
                            "502 server_is_overloaded",
                        ),
                    ),
                ):
                    candidates = await pipeline.refresh("2026-08-26", PAPER_CONTENT)

                self.assertEqual([item["title"] for item in candidates], ["Previously selected paper"])
                self.assertEqual(
                    [item["title"] for item in pipeline.db.get_candidates("2026-08-26", PAPER_CONTENT)],
                    ["Previously selected paper"],
                )
                self.assertIn("继续使用今日最近一次成功结果", pipeline.format_news(candidates))

        asyncio.run(check())

    def test_paper_refresh_failure_uses_fallback_when_no_same_day_candidates(self):
        async def check():
            with tempfile.TemporaryDirectory() as tmp:
                settings = replace(
                    load_settings(),
                    database_path=Path(tmp) / "fallback.db",
                    model_base_url="https://model.example/v1",
                    model_api_key="test-key",
                    model_name="test-model",
                    openalex_api_key="test-openalex-key",
                )
                pipeline = NewsPipeline(settings)
                item = {
                    "source": "Journal of Climate",
                    "url": "https://example.test/fallback",
                    "canonical_url": "https://example.test/fallback",
                    "title": "Local fallback paper",
                    "summary": "Near-surface wind climate mechanism",
                    "published_at": "2026-08-26T00:00:00+00:00",
                    "doi": "10.1000/fallback",
                    "journal": "Journal of Climate",
                    "word_count": 800,
                    "status": "discovered",
                    "discovered_at": "2026-08-26T00:00:00+00:00",
                }
                pipeline._extract_shortlist = lambda items: asyncio.sleep(
                    0, result=copy.deepcopy(items)
                )
                pipeline._published_papers = lambda items, _date: asyncio.sleep(
                    0,
                    result=[dict(value, paper_local_score=2) for value in items],
                )
                with (
                    patch.object(
                        pipeline.openalex,
                        "discover_recent_papers",
                        return_value=[],
                    ),
                    patch(
                        "news.pipeline.fetch_all_feeds",
                        return_value=([item], [], {"test": 1}),
                    ),
                    patch(
                        "news.pipeline.select_paper_top_ten",
                        return_value=(
                            [dict(item, article_id=1, score=10, title_cn="")],
                            False,
                            "429 usage_limit_reached",
                        ),
                    ),
                ):
                    candidates = await pipeline.refresh("2026-08-26", PAPER_CONTENT)

                self.assertEqual([value["title"] for value in candidates], ["Local fallback paper"])
                self.assertIn("当前显示本地筛选结果", pipeline.format_news(candidates))
                self.assertEqual(
                    len(pipeline.db.get_candidates("2026-08-26", PAPER_CONTENT)),
                    1,
                )

        asyncio.run(check())

    def test_paper_selection_saves_before_independent_title_translation_failure(self):
        async def check():
            with tempfile.TemporaryDirectory() as tmp:
                settings = replace(
                    load_settings(),
                    database_path=Path(tmp) / "translation-failure.db",
                    model_base_url="https://model.example/v1",
                    model_api_key="test-key",
                    model_name="test-model",
                    openalex_api_key="test-openalex-key",
                )
                pipeline = NewsPipeline(settings)
                items = [
                    {
                        "source": "Journal of Climate",
                        "url": f"https://example.test/selected-{index}",
                        "canonical_url": f"https://example.test/selected-{index}",
                        "title": (
                            "Near-surface wind circulation variability"
                            if index == 1
                            else "Stratospheric ozone climate coupling"
                        ),
                        "summary": "Near-surface wind climate mechanism",
                        "published_at": "2026-08-26T00:00:00+00:00",
                        "doi": f"10.1000/selected-{index}",
                        "journal": "Journal of Climate",
                        "word_count": 800,
                        "status": "discovered",
                        "discovered_at": "2026-08-26T00:00:00+00:00",
                    }
                    for index in (1, 2)
                ]
                pipeline._extract_shortlist = lambda values: asyncio.sleep(
                    0, result=copy.deepcopy(values)
                )
                pipeline._published_papers = lambda values, _date: asyncio.sleep(
                    0,
                    result=[dict(value, paper_local_score=2) for value in values],
                )
                def fake_selection(values, _settings):
                    return [dict(values[1]), dict(values[0])], True, ""

                with (
                    patch.object(
                        pipeline.openalex,
                        "discover_recent_papers",
                        return_value=[],
                    ),
                    patch(
                        "news.pipeline.fetch_all_feeds",
                        return_value=(items, [], {"test": 2}),
                    ),
                    patch(
                        "news.pipeline.select_paper_top_ten",
                        side_effect=fake_selection,
                    ),
                    patch(
                        "news.pipeline.translate_paper_titles",
                        return_value=(['', ''], False, "502 server_is_overloaded"),
                    ) as translate,
                ):
                    candidates = await pipeline.refresh("2026-08-26", PAPER_CONTENT)

                self.assertEqual(translate.call_count, 1)
                self.assertEqual(
                    [value["title"] for value in candidates],
                    [
                        "Stratospheric ozone climate coupling",
                        "Near-surface wind circulation variability",
                    ],
                )
                self.assertEqual([value["title_cn"] for value in candidates], ["", ""])

        asyncio.run(check())

    def test_paper_title_translation_only_fills_missing_without_selection(self):
        async def check():
            with tempfile.TemporaryDirectory() as tmp:
                settings = replace(
                    load_settings(),
                    database_path=Path(tmp) / "translation-success.db",
                    model_base_url="https://model.example/v1",
                    model_api_key="test-key",
                    model_name="test-model",
                )
                pipeline = NewsPipeline(settings)
                article_ids = [
                    pipeline.db.upsert_article(
                        {
                            "source": "Journal of Climate",
                            "url": f"https://example.test/title-{index}",
                            "canonical_url": f"https://example.test/title-{index}",
                            "title": f"English title {index}",
                            "summary": "Near-surface wind climate mechanism",
                            "published_at": "2026-08-26T00:00:00+00:00",
                            "doi": f"10.1000/title-{index}",
                            "journal": "Journal of Climate",
                            "word_count": 800,
                            "status": "published_paper",
                            "discovered_at": "2026-08-26T00:00:00+00:00",
                        }
                    )
                    for index in (1, 2, 3)
                ]
                pipeline.db.replace_candidates(
                    "2026-08-26",
                    [
                        {"article_id": article_ids[0], "score": 30, "title_cn": "已有标题"},
                        {"article_id": article_ids[1], "score": 20, "title_cn": ""},
                        {"article_id": article_ids[2], "score": 10, "title_cn": ""},
                    ],
                    PAPER_CONTENT,
                )
                pipeline.db.set_daily_run(
                    "2026-08-26",
                    content_type=PAPER_CONTENT,
                    status="success",
                )
                with (
                    patch(
                        "news.pipeline.translate_paper_titles",
                        return_value=(
                            ["已有标题", "补充标题2", "补充标题3"],
                            True,
                            "",
                        ),
                    ) as translate,
                    patch("news.pipeline.select_paper_top_ten") as select,
                ):
                    candidates = await pipeline.get_or_refresh("2026-08-26", PAPER_CONTENT)

                self.assertEqual(translate.call_count, 1)
                self.assertEqual(select.call_count, 0)
                self.assertEqual(
                    [(value["rank"], value["title_cn"]) for value in candidates],
                    [(1, "已有标题"), (2, "补充标题2"), (3, "补充标题3")],
                )

        asyncio.run(check())

    def test_paper_model_retries_502_once_but_not_429(self):
        settings = replace(
            load_settings(),
            model_base_url="https://model.example/v1",
            model_api_key="test-key",
            model_name="test-model",
        )
        candidates = [
            {
                "title": "Climate paper",
                "summary": "Near-surface wind climate mechanism",
                "paper_local_score": 2,
            }
        ]
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps({"items": [{"index": 1, "score": 2}]})
                    )
                )
            ]
        )

        class HttpError(Exception):
            def __init__(self, status_code, message):
                super().__init__(message)
                self.status_code = status_code

        retry_client = MagicMock()
        retry_client.chat.completions.create.side_effect = [
            HttpError(502, "server_is_overloaded"),
            response,
        ]
        with (
            patch("writer.llm.OpenAI", return_value=retry_client),
            patch("writer.llm.time.sleep") as sleep,
        ):
            selected, used_model, error = select_paper_top_ten(candidates, settings)
        self.assertTrue(used_model)
        self.assertEqual(error, "")
        self.assertEqual(retry_client.chat.completions.create.call_count, 2)
        sleep.assert_called_once_with(3)
        self.assertEqual(len(selected), 1)

        rate_client = MagicMock()
        rate_client.chat.completions.create.side_effect = HttpError(
            429,
            "usage_limit_reached",
        )
        with (
            patch("writer.llm.OpenAI", return_value=rate_client),
            patch("writer.llm.time.sleep") as sleep,
        ):
            selected, used_model, error = select_paper_top_ten(candidates, settings)
        self.assertFalse(used_model)
        self.assertIn("usage_limit_reached", error)
        self.assertEqual(rate_client.chat.completions.create.call_count, 1)
        sleep.assert_not_called()

        title_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {"items": [{"index": 1, "title_cn": "中文标题"}]},
                            ensure_ascii=False,
                        )
                    )
                )
            ]
        )
        title_client = MagicMock()
        title_client.chat.completions.create.side_effect = [
            HttpError(503, "temporarily unavailable"),
            title_response,
        ]
        with (
            patch("writer.llm.OpenAI", return_value=title_client),
            patch("writer.llm.time.sleep") as sleep,
        ):
            titles, used_model, error = translate_paper_titles(
                [{"title": "English title", "title_cn": ""}],
                settings,
            )
        self.assertTrue(used_model)
        self.assertEqual(error, "")
        self.assertEqual(titles, ["中文标题"])
        self.assertEqual(title_client.chat.completions.create.call_count, 2)
        sleep.assert_called_once_with(3)

    def test_paper_title_translation_sends_only_missing_titles_in_order(self):
        settings = replace(
            load_settings(),
            model_base_url="https://model.example/v1",
            model_api_key="test-key",
            model_name="test-model",
        )
        candidates = [
            {"title": "Already translated", "title_cn": "已有标题"},
            {"title": "Second English title", "title_cn": ""},
            {"title": "Third English title", "title_cn": ""},
        ]
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "items": [
                                    {"index": 2, "title_cn": "第二个标题"},
                                    {"index": 3, "title_cn": "第三个标题"},
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
            titles, used_model, error = translate_paper_titles(candidates, settings)
        self.assertTrue(used_model)
        self.assertEqual(error, "")
        self.assertEqual(titles, ["已有标题", "第二个标题", "第三个标题"])
        payload = json.loads(
            client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        )
        self.assertEqual(payload, [
            {"index": 2, "title": "Second English title"},
            {"index": 3, "title": "Third English title"},
        ])

    def test_papers_next_excludes_seen_batches_and_preserves_failed_batch(self):
        async def check():
            with tempfile.TemporaryDirectory() as tmp:
                settings = replace(
                    load_settings(),
                    database_path=Path(tmp) / "next.db",
                    model_base_url="https://model.example/v1",
                    model_api_key="test-key",
                    model_name="test-model",
                    openalex_api_key="",
                )
                pipeline = NewsPipeline(settings)
                items = [
                    {
                        "source": "Journal of Climate",
                        "url": f"https://example.test/next-{index}",
                        "canonical_url": f"https://example.test/next-{index}",
                        "title": f"Paper {index}",
                        "summary": "Near-surface wind climate mechanism",
                        "published_at": "2026-08-26T00:00:00+00:00",
                        "doi": f"10.1000/next-{index}",
                        "journal": "Journal of Climate",
                        "word_count": 800,
                        "status": "discovered",
                        "discovered_at": "2026-08-26T00:00:00+00:00",
                    }
                    for index in range(1, 25)
                ]
                pipeline._extract_shortlist = lambda values: asyncio.sleep(
                    0, result=copy.deepcopy(values)
                )
                pipeline._published_papers = lambda values, _date: asyncio.sleep(
                    0,
                    result=[dict(value, paper_local_score=2) for value in values],
                )

                def select_batch(values, _settings):
                    return [dict(value) for value in values[:10]], True, ""

                def translate_batch(values, _settings):
                    return [str(value.get("title_cn") or "") for value in values], True, ""

                with (
                    patch("news.pipeline.fetch_all_feeds", return_value=(items, [], {"test": 24})),
                    patch("news.pipeline.select_paper_top_ten", side_effect=select_batch),
                    patch("news.pipeline.translate_paper_titles", side_effect=translate_batch),
                    patch("news.pipeline.deduplicate", side_effect=lambda values: values),
                ):
                    first = await pipeline.next_paper_batch("2026-08-26")
                    second = await pipeline.next_paper_batch("2026-08-26")
                    second_text = pipeline.format_news(second)
                    third = await pipeline.next_paper_batch("2026-08-26")

                self.assertEqual([item["title"] for item in first], [f"Paper {i}" for i in range(1, 11)])
                self.assertEqual([item["title"] for item in second], [f"Paper {i}" for i in range(11, 21)])
                self.assertEqual([item["title"] for item in third], [f"Paper {i}" for i in range(21, 25)])
                self.assertIn("本批次新增10篇，累计20篇", second_text)
                self.assertIn("11.", second_text)
                with patch(
                    "news.pipeline.translate_paper_titles",
                    return_value=(['' for _ in range(24)], True, ""),
                ):
                    all_current = await pipeline.get_or_refresh("2026-08-26", PAPER_CONTENT)
                self.assertEqual(len(all_current), 24)
                self.assertIn("今日已发表论文（共24篇）", pipeline.format_news(all_current))
                self.assertEqual(
                    len(pipeline.db.get_seen_candidate_ids("2026-08-26", PAPER_CONTENT)),
                    24,
                )
                all_titles = [item["title"] for item in (*first, *second, *third)]
                self.assertEqual(len(all_titles), len(set(all_titles)))

                detail_calls = []

                async def fake_details(rank, date=None, content_type=None):
                    detail_calls.append((rank, date, content_type))
                    return {"rank": rank, "content_type": content_type}

                pipeline.paper_details = fake_details
                handler = CommandHandler(settings, pipeline)
                with patch("bot.commands.local_date", return_value="2026-08-26"):
                    detail = await handler.handle("/paper 1")
                self.assertEqual(detail_calls, [(1, "2026-08-26", PAPER_CONTENT)])
                self.assertTrue(detail.startswith("## "))

                with (
                    patch("news.pipeline.fetch_all_feeds", return_value=(items, [], {"test": 24})),
                    patch(
                        "news.pipeline.select_paper_top_ten",
                        return_value=([], False, "429 usage_limit_reached"),
                    ),
                    patch("news.pipeline.deduplicate", side_effect=lambda values: values),
                ):
                    failed = await pipeline.next_paper_batch("2026-08-26")
                self.assertEqual(
                    [item["title"] for item in failed],
                    [f"Paper {i}" for i in range(1, 25)],
                )
                self.assertIn("⚠ 换一批失败，继续保留当前论文列表", pipeline.format_news(failed))
                self.assertEqual(
                    len(pipeline.db.get_seen_candidate_ids("2026-08-26", PAPER_CONTENT)),
                    24,
                )

        asyncio.run(check())

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
