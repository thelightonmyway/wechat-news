from __future__ import annotations

import asyncio
import copy
import io
import json
import re
import tempfile
import unittest
from dataclasses import replace
from datetime import date as date_type, datetime, time, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pymupdf
from PIL import Image

from bot.bridge import QQNewsBot
from bot.commands import (
    CommandHandler,
    NEWS_USAGE,
    PAPER_USAGE,
    _direct_paper_item,
    _paper_image_summary,
)
from db import Database
from images.policy import apply_policy, assess_image
from images.search import normalize_search_result, search_public_images
from news.extract import discover_figure_images, download_images
from news.feeds import canonicalize_url, load_feeds, normalize_title
from news.pipeline import (
    PAPER_CONTENT,
    PAPER_ONLY_SOURCES,
    POPULAR_CONTENT,
    PRIMARY_SOURCES,
    SECONDARY_SOURCES,
    NewsPipeline,
    PAPER_LOOKBACK_HOURS,
    _images_redundant,
    _insert_paper_figures,
    _paper_publication_within_window,
    _paper_figure_reference_numbers,
    _paper_match_source_paragraphs,
    _prepare_paper_markdown,
    _select_article_images,
    content_type_for_date,
    deduplicate,
    deterministic_score,
    is_broad_journal_first_paper,
    is_relevant_news_after_extraction,
    is_relevant_news_rss_prefilter,
    is_relevant_topic,
    merge_paper_candidate_pool,
    paper_journal_tier,
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
    _expanded_crop_rect,
    _refine_figure_crop_bounds,
    discover_pdf_source,
    download_pdf_with_wiley_tdm,
    extract_pdf_figures,
)
from publisher.wechat import (
    DEFAULT_COVER,
    _paper_draft_title,
    _remove_paper_figure_attributions,
    _selected_cover_path,
    _style_paper_intro,
    format_markdown,
)
from scheduler import should_run_startup_catchup
from settings import bind_qq_target_openid, load_settings
from writer.llm import (
    PAPER_PLANNER_PROMPT,
    PAPER_POPULAR_SCIENCE_EDITOR_PROMPT,
    PAPER_STORY_PLANNER_PROMPT,
    PAPER_STORY_WRITER_PROMPT,
    PAPER_HUMANIZER_PROMPT,
    PAPER_ARTICLE_EDITOR_PROMPT,
    PAPER_STYLE_REVIEWER_PROMPT,
    PAPER_STYLE_GUIDE,
    _extract_paper_evidence_plan,
    _paper_ai_style_lint,
    _paper_ai_style_lint_failed,
    _paper_require_clean_final_style_lint,
    _paper_apply_article_editor_candidate,
    _paper_article_editor,
    _paper_article_editor_rebuild_sections,
    _paper_deauthor_abstract,
    _paper_normalize_article_editor_output,
    _paper_apply_story_output,
    _paper_apply_story_candidate,
    _paper_body_length_audit,
    _paper_chinese_char_count,
    _paper_canonical_evidence_registry,
    _paper_clean_story_evidence,
    _paper_clean_story_text,
    _paper_evidence_by_id,
    _paper_derived_source_ids,
    _paper_story_sections,
    _paper_story_block_specs,
    _paper_plain_language_cleanup,
    _paper_metadata_leakage_lint,
    _paper_editor_anchor_audit,
    _paper_editor_feedback,
    _paper_figure_evidence_bundles,
    _paper_readability_audit,
    _paper_stop_slop_audit,
    _paper_stable_evidence_id,
    _paper_style_exemplar,
    _paper_title_style_lint,
    _paper_story_sections,
    _paper_story_writer,
    _paper_normalize_story_output,
    _paper_humanize_story,
    _paper_style_review,
    _paper_validate_story_plan,
    _validate_paper_plan_structure,
    prune_paper_sections_after_allocation,
    _normalize_article_markdown,
    _paper_revision,
    _paper_review,
    _validate_paper_evidence_plan,
    generate_article_markdown,
    generate_image_captions,
    generate_image_search_keywords,
    select_paper_ranked,
    select_paper_top_ten,
    select_top_ten,
    translate_paper_abstract,
    translate_paper_titles,
)


def _story_plan_for_evidence(count: int) -> dict[str, object]:
    beats = [
        {
            "id": f"beat-{index}",
            "title": f"发现{index}",
            "reader_question": f"读者问题{index}",
            "core_message": f"核心发现{index}",
            "evidence_ids": [f"evidence-{index}"],
            "transition_to_next": "继续解释这一发现。",
        }
        for index in range(1, count + 1)
    ]
    return {
        "editorial_brief": {
            "audience": "跨专业读者",
            "purpose": "解释核心科学发现",
            "tone": "清楚自然",
            "reader_should_leave_with": "记住核心发现及其意义",
            "story_question": "这项研究回答了什么问题？",
        },
        "story_beats": beats,
    }


def _story_output_for_evidence(count: int, bodies: list[str] | None = None) -> list[dict[str, str]]:
    return [
        {
            "id": f"beat-{index}",
            "title": f"发现{index}",
            "blocks": [{
                "block_id": f"beat-{index}-block-1",
                "text": (bodies or [f"核心发现{item}。" for item in range(1, count + 1)])[index - 1],
            }],
        }
        for index in range(1, count + 1)
    ]


def _story_block_output(
    beat_id: str,
    blocks: list[tuple[str, list[str], str]],
    title: str = "专业结果",
) -> list[dict[str, object]]:
    return [{
        "id": beat_id,
        "title": title,
        "blocks": [
            {"block_id": block_id, "text": text}
            for block_id, evidence_ids, text in blocks
        ],
    }]


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

    def test_direct_paper_url_recognizes_doi_and_ignores_ordinary_url(self):
        item = _direct_paper_item(
            "https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025GL120559"
        )
        self.assertIsNotNone(item)
        self.assertEqual(item["doi"], "10.1029/2025gl120559")
        self.assertEqual(item["content_type"], PAPER_CONTENT)
        doi_item = _direct_paper_item("10.1029/2025GL120559")
        self.assertIsNotNone(doi_item)
        self.assertEqual(doi_item["url"], "https://doi.org/10.1029/2025gl120559")
        self.assertIsNone(_direct_paper_item("https://example.com/news/story"))

    def test_direct_paper_url_generates_then_publishes(self):
        class FakePipeline:
            def __init__(self):
                self.calls = []

            async def generate(self, rank, date=None, content_type=None, **kwargs):
                self.calls.append(("generate", rank, date, content_type, kwargs))
                return {
                    "dossier": {
                        "id": 123,
                        "title": "A paper title",
                        "journal": "Journal of Climate",
                    },
                    "markdown_path": Path("/tmp/direct-paper.md"),
                }

        async def check():
            settings = replace(
                load_settings(),
                model_base_url="https://model.example/v1",
                model_api_key="test-key",
                model_name="test-model",
            )
            pipeline = FakePipeline()
            handler = CommandHandler(settings, pipeline)

            async def fake_publish(generated, **kwargs):
                pipeline.calls.append(("publish", generated, kwargs))
                return "已生成并发布到微信草稿箱"

            handler._publish_generated_paper = fake_publish
            response = await handler.handle(
                "/paperurl https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025GL120559"
            )
            self.assertIn("识别到论文", response)
            self.assertIn("已生成并发布到微信草稿箱", response)
            self.assertEqual([call[0] for call in pipeline.calls], ["generate", "publish"])
            generate_call = pipeline.calls[0]
            self.assertEqual(generate_call[1], 0)
            self.assertEqual(generate_call[3], PAPER_CONTENT)
            self.assertEqual(
                generate_call[4]["item_override"]["doi"], "10.1029/2025gl120559"
            )
            self.assertFalse(any("publish" in str(value) for value in generate_call[4]))
            self.assertIsNone(
                await handler.handle(
                    "https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025GL120559"
                )
            )
            self.assertIn("/paperurl <论文URL或DOI>", PAPER_USAGE)

        asyncio.run(check())

    def test_paper_publish_generates_before_publish(self):
        class FakeDB:
            def get_candidate(self, date, rank, content_type):
                return {"title": "Candidate", "url": "https://example.com/paper"}

        class FakePipeline:
            def __init__(self):
                self.calls = []
                self.db = FakeDB()

            async def generate(self, rank, date=None, content_type=None):
                self.calls.append("generate")
                return {"dossier": {"id": 456}, "markdown_path": Path("/tmp/paper.md")}

        async def check():
            settings = replace(
                load_settings(),
                model_base_url="https://model.example/v1",
                model_api_key="test-key",
                model_name="test-model",
            )
            pipeline = FakePipeline()
            handler = CommandHandler(settings, pipeline)

            async def fake_publish(generated, **kwargs):
                pipeline.calls.append("publish")
                return "published"

            handler._publish_generated_paper = fake_publish
            self.assertEqual(await handler.handle("/paper 1 publish"), "published")
            self.assertEqual(pipeline.calls, ["generate", "publish"])

        asyncio.run(check())

    def test_paper_publish_generate_failure_skips_publish(self):
        class FakePipeline:
            def __init__(self):
                self.calls = []
                self.db = SimpleNamespace()

            async def generate(self, rank, date=None, content_type=None):
                self.calls.append("generate")
                raise RuntimeError("generation failed")

        async def check():
            settings = replace(
                load_settings(),
                model_base_url="https://model.example/v1",
                model_api_key="test-key",
                model_name="test-model",
            )
            pipeline = FakePipeline()
            handler = CommandHandler(settings, pipeline)

            async def fake_publish(generated, **kwargs):
                pipeline.calls.append("publish")
                return "published"

            handler._publish_generated_paper = fake_publish
            response = await handler.handle("/paper 1 publish")
            self.assertIn("PAPER generate 失败，未调用 publish", response)
            self.assertEqual(pipeline.calls, ["generate"])

        asyncio.run(check())

    def test_direct_paper_generate_translates_title_before_markdown(self):
        settings = replace(
            load_settings(),
            database_path=Path(tempfile.mkdtemp()) / "direct-title.db",
            model_base_url="https://model.example/v1",
            model_api_key="test-key",
            model_name="test-model",
        )
        pipeline = NewsPipeline(settings)
        captured = {}

        def fake_markdown(dossier, _settings, output_dir):
            captured["title_cn"] = dossier.get("title_cn")
            dossier["paper_evidence_plan"] = {
                "sections": [
                    {
                        "id": "section-1",
                        "title": "关键结果",
                        "role": "attribution",
                        "source_paragraph_ids": ["source-0"],
                        "findings": [],
                    }
                ]
            }
            output_dir.mkdir(parents=True, exist_ok=True)
            markdown_path = output_dir / "article.md"
            metadata_path = output_dir / "metadata.json"
            markdown_path.write_text("# 中文标题\n\n正文。\n", encoding="utf-8")
            metadata_path.write_text("{}", encoding="utf-8")
            return markdown_path, metadata_path

        dossier = {
            "id": 789,
            "rank": 0,
            "date": "2026-09-03",
            "content_type": PAPER_CONTENT,
            "title": "An English paper title",
            "title_cn": "",
            "summary": "Paper abstract",
            "text": "Paper text",
            "doi": "10.1029/example",
            "url": "https://doi.org/10.1029/example",
            "images": [],
            "openalex": {},
        }
        dossier["id"] = pipeline.db.upsert_article(dossier)
        with (
            patch("news.pipeline.translate_paper_titles", return_value=(['中文完整标题'], True, "")),
            patch("news.pipeline.generate_article_markdown", side_effect=fake_markdown),
            patch("news.pipeline.download_images", return_value=[]),
            patch("news.pipeline.generate_image_captions", return_value=[]),
            patch("news.pipeline._select_article_images", return_value=({}, [], 0)),
            patch("news.pipeline.discover_pdf_source", return_value={"pdf_url": ""}),
            patch("news.pipeline._prepare_paper_markdown"),
        ):
            with tempfile.TemporaryDirectory() as tmp:
                generated = asyncio.run(
                    pipeline.generate(
                        0,
                        "2026-09-03",
                        PAPER_CONTENT,
                        item_override={"paper_url_hint": True},
                        dossier_override=dossier,
                        output_dir=Path(tmp) / "paper",
                    )
                )
                self.assertEqual(captured["title_cn"], "中文完整标题")
                metadata = json.loads(generated["metadata_path"].read_text(encoding="utf-8"))
                self.assertEqual(
                    metadata["paper_evidence_plan"]["sections"][0]["id"],
                    "section-1",
                )

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
            PAPER_LOOKBACK_HOURS: [item(1), item(2), item(3)],
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

                self.assertEqual(calls, [PAPER_LOOKBACK_HOURS])
                self.assertEqual(len(candidates), 4)
                self.assertEqual(pipeline.openalex.discover_recent_papers.call_count, 1)
                self.assertEqual(
                    pipeline.openalex.discover_recent_papers.call_args_list,
                    [
                        unittest.mock.call(date_type(2026, 5, 28), date_type(2026, 8, 26)),
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
                                    "image_source": "html_figure",
                                    "image_role": "figure",
                                    "metadata_title": "Fig. 1",
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
                self.assertEqual(dossier["images"][0]["image_source"], "html_figure")

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

    def test_paper_evidence_validator_rejects_anchor_in_wrong_section(self):
        plan = {
            "sections": [
                {
                    "id": "section-1",
                    "title": "现象",
                    "role": "phenomenon",
                    "source_paragraph_ids": ["source-0"],
                    "findings": [],
                },
                {
                    "id": "section-2",
                    "title": "机制",
                    "role": "mechanism",
                    "source_paragraph_ids": ["source-1"],
                    "findings": [],
                },
                {
                    "id": "section-3",
                    "title": "归因",
                    "role": "attribution",
                    "source_paragraph_ids": ["source-2"],
                    "findings": [
                        {
                            "id": "E1",
                            "evidence": "cross-model attribution",
                            "anchors": ["-0.77"],
                        }
                    ],
                },
            ]
        }
        markdown = (
            "# 测试标题\n\n摘要导语。\n\n"
            "## 现象\n\n现象正文。\n\n"
            "## 机制\n\n相关系数为−0.77。\n\n"
            "## 归因\n\n归因正文。"
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"evidence='-0.77'.*planned section='归因'.*actual section='机制'",
        ):
            _validate_paper_evidence_plan(plan, markdown)

    def test_paper_evidence_plan_is_extracted_from_markdown(self):
        raw = (
            '<!-- PAPER_EVIDENCE_PLAN {"sections":[{"id":"section-1",'
            '"title":"现象","source_paragraph_ids":["source-0"],"findings":[]}]} -->\n'
            "# 标题\n\n## 现象\n\n正文。"
        )
        plan, markdown = _extract_paper_evidence_plan(raw)
        self.assertEqual(plan["sections"][0]["id"], "section-1")
        self.assertNotIn("PAPER_EVIDENCE_PLAN", markdown)

    def test_paper_planner_separates_attribution_and_projection_roles(self):
        synthetic_input = {
            "abstract": (
                "Historical model spread is attributed to forest-cover changes; "
                "future SSP3-7.0 projections retain a separate uncertainty signal."
            ),
            "paper_text": (
                "Attribution explains the historical spread. Future projection results "
                "cover 2025-2054 and 2070-2099 under SSP3-7.0."
            ),
        }
        self.assertIn('"role":"attribution"', PAPER_PLANNER_PROMPT)
        self.assertIn("title必须是适合中文成稿的简洁中文小标题", PAPER_PLANNER_PROMPT)
        self.assertIn("各有独立Figure bundle证据", PAPER_PLANNER_PROMPT)
        self.assertIn("不要为凑section数量而合并不相关Figure", PAPER_PLANNER_PROMPT)
        self.assertIn("每个Figure bundle的核心finding只能进入包含该Figure的section", PAPER_PLANNER_PROMPT)
        self.assertIn("无独立主图的机制内容只能作为最相关Figure section中的2到3句解释", PAPER_PLANNER_PROMPT)
        self.assertIn("projection", PAPER_PLANNER_PROMPT)
        synthetic_text = json.dumps(synthetic_input, ensure_ascii=False).lower()
        self.assertIn("attribution", synthetic_text)
        self.assertIn("projection", synthetic_text)

    def test_paper_scientific_planner_retries_empty_sections(self):
        empty_plan = {"sections": []}
        valid_plan = {
            "sections": [{
                "id": "section-1",
                "title": "总体结果",
                "role": "phenomenon",
                "figure_ids": [],
                "findings": [{"id": "E1", "figure_ids": [], "evidence_ids": ["evidence-source-source-0"]}],
            }]
        }
        story_plan = _story_plan_for_evidence(1)
        story_plan["story_beats"][0]["evidence_ids"] = ["evidence-source-source-0"]
        settings = replace(load_settings(), model_base_url="https://model.example/v1", model_api_key="test-key", model_name="test-model")
        with tempfile.TemporaryDirectory() as tmp, patch(
            "writer.llm._paper_plan", side_effect=[empty_plan, valid_plan]
        ) as planner, patch(
            "writer.llm.translate_paper_abstract", return_value="忠实摘要翻译。"
        ), patch(
            "writer.llm._paper_write_section", return_value="总体结果。"
        ), patch(
            "writer.llm._paper_review", return_value={"status": "pass", "corrections": []}
        ), patch(
            "writer.llm._paper_story_planner", return_value=story_plan
        ), patch(
            "writer.llm._paper_story_writer", return_value=_story_output_for_evidence(1, ["总体结果。"])
        ), patch(
            "writer.llm._paper_humanize_story", return_value=_story_output_for_evidence(1, ["总体结果。"])
        ):
            path, _ = generate_article_markdown(
                {
                    "content_type": PAPER_CONTENT,
                    "title": "Sparse paper",
                    "title_cn": "稀疏证据论文",
                    "text": "source",
                    "openalex": {"abstract": "Abstract"},
                    "images": [],
                    "paper_selected_body_images": [],
                },
                settings,
                Path(tmp) / "paper",
            )
            article_text = path.read_text(encoding="utf-8")
        self.assertEqual(planner.call_count, 2)
        self.assertIn("总体结果", article_text)
        self.assertIn("global_context", PAPER_PLANNER_PROMPT)
        self.assertIn("section_context", PAPER_PLANNER_PROMPT)

    def test_paper_figure_first_writers_receive_only_current_bundles(self):
        planner = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "sections": [
                                    {
                                        "id": "section-1",
                                        "title": "机器学习重构",
                                        "role": "attribution",
                                        "figure_ids": ["Fig. 2"],
                                        "findings": [{"id": "E1", "figure_ids": ["Fig. 2"], "evidence_ids": ["evidence-source-source-figure-1"]}],
                                    },
                                    {
                                        "id": "section-2",
                                        "title": "森林相关",
                                        "role": "attribution",
                                        "figure_ids": ["Fig. 3"],
                                        "findings": [{"id": "E2", "figure_ids": ["Fig. 3"], "evidence_ids": ["evidence-source-source-figure-2"]}],
                                    },
                                ]
                            },
                            ensure_ascii=False,
                        )
                    )
                )
            ]
        )
        responses = [
            planner,
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"abstract_cn": "摘要翻译"}, ensure_ascii=False))) ]),
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"body": "机器学习重构R = 0.71。"}, ensure_ascii=False))) ]),
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"body": "森林相关R = −0.77。"}, ensure_ascii=False))) ]),
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"status": "pass", "corrections": []}, ensure_ascii=False))) ]),
        ]
        images = [
            {"figure_number": 2, "caption": "Figure 2. XGBoost reconstruction R = 0.71."},
            {"figure_number": 3, "caption": "Figure 3. Forest correlation R = −0.77."},
        ]
        settings = replace(load_settings(), model_base_url="https://model.example/v1", model_api_key="test-key", model_name="test-model")
        client = MagicMock()
        client.chat.completions.create.side_effect = responses
        story_plan = _story_plan_for_evidence(2)
        story_plan["story_beats"][0]["evidence_ids"] = ["evidence-source-source-figure-1"]
        story_plan["story_beats"][1]["evidence_ids"] = ["evidence-source-source-figure-2"]
        with tempfile.TemporaryDirectory() as tmp, patch("writer.llm.OpenAI", return_value=client), patch(
            "writer.llm._paper_story_planner", return_value=story_plan
        ), patch(
            "writer.llm._paper_story_writer",
            return_value=_story_output_for_evidence(2, ["机器学习重构R = 0.71。", "森林相关R = −0.77。"]),
        ), patch(
            "writer.llm._paper_editorial_rewrite", return_value=["机器学习重构R = 0.71。", "森林相关R = −0.77。"]
        ), patch(
            "writer.llm._paper_humanize_story",
            return_value=_story_output_for_evidence(2, ["机器学习重构R = 0.71。", "森林相关R = −0.77。"]),
        ):
            path, _ = generate_article_markdown(
                {
                    "content_type": PAPER_CONTENT,
                    "title": "Test paper",
                    "title_cn": "测试标题",
                    "text": "Results refer to Figure 2 and Figure 3.",
                    "openalex": {"abstract": "Abstract"},
                    "images": images,
                    "paper_selected_body_images": images,
                },
                settings,
                Path(tmp) / "paper",
            )
            calls = client.chat.completions.create.call_args_list
            final_text = path.read_text(encoding="utf-8")
        first_writer = json.loads(calls[2].kwargs["messages"][1]["content"])
        second_writer = json.loads(calls[3].kwargs["messages"][1]["content"])
        self.assertEqual([bundle["figure_id"] for bundle in first_writer["figure_evidence_bundles"]], ["Fig. 2"])
        self.assertEqual([bundle["figure_id"] for bundle in second_writer["figure_evidence_bundles"]], ["Fig. 3"])
        self.assertNotIn("R = −0.77", json.dumps(first_writer, ensure_ascii=False))
        self.assertIn("R = 0.71", final_text)

    def test_paper_figure_evidence_bundle_binds_anchor_to_figure(self):
        bundles = [
            {
                "figure_id": "Fig. 2",
                "caption": "XGBoost reconstruction, R = 0.71.",
                "source_paragraphs": [],
                "quantitative_anchors": ["R = 0.71"],
            },
            {
                "figure_id": "Fig. 3",
                "caption": "Forest correlation, R = −0.77.",
                "source_paragraphs": [],
                "quantitative_anchors": ["R = −0.77"],
            },
        ]
        plan = {
            "sections": [
                {
                    "id": "section-1",
                    "title": "机器学习重构",
                    "figure_ids": ["Fig. 2"],
                    "source_paragraph_ids": ["source-0"],
                    "findings": [{"id": "E1", "figure_ids": ["Fig. 2"], "evidence": "重构", "anchors": []}],
                },
                {
                    "id": "section-2",
                    "title": "森林相关",
                    "figure_ids": ["Fig. 3"],
                    "source_paragraph_ids": ["source-0"],
                    "findings": [{"id": "E2", "figure_ids": ["Fig. 3"], "evidence": "错误串位", "anchors": ["R = 0.71"]}],
                },
            ]
        }
        markdown = "# 标题\n\n## 机器学习重构\n\n重构结果。\n\n## 森林相关\n\nR = 0.71。"
        with self.assertRaisesRegex(RuntimeError, "PAPER evidence figure mismatch"):
            _validate_paper_evidence_plan(plan, markdown, {"source-0"}, bundles)

    def test_paper_74_percent_anchor_binds_to_fig3(self):
        images = [
            {"figure_number": 2, "caption": "Figure 2. XGBoost reconstruction R = 0.71; CCA R = 0.84."},
            {"figure_number": 3, "caption": "Figure 3. Forest trend relation and percentage reduction in inter-model spread."},
        ]
        source_paragraphs = [
            {"id": "source-0", "text": "XGBoost reconstruction is shown in Figure 2 with R = 0.71."},
            {"id": "source-1", "text": "The forest relation is shown in Figure 3b with R = −0.77."},
            {"id": "source-2", "text": "This corresponds to approximately 74% reduction in the inter-model spread."},
            {"id": "source-3", "text": "(Figure 3d)."},
        ]
        bundles = _paper_figure_evidence_bundles(images, source_paragraphs)
        by_id = {bundle["figure_id"]: bundle for bundle in bundles}
        self.assertIn("approximately 74%", by_id["Fig. 3"]["quantitative_anchors"])
        self.assertNotIn("approximately 74%", by_id["Fig. 2"]["quantitative_anchors"])
        self.assertIn("R = −0.77", by_id["Fig. 3"]["quantitative_anchors"])
        self.assertNotIn("R = −0.77", by_id["Fig. 2"]["quantitative_anchors"])
        self.assertIn("R = 0.71", by_id["Fig. 2"]["quantitative_anchors"])
        self.assertIn("R = 0.84", by_id["Fig. 2"]["quantitative_anchors"])
        self.assertEqual(by_id["Fig. 3"]["supported_figures_by_anchor"]["approximately 74%"], ["Fig. 3"])

    def test_paper_figure_bundle_contains_caption_and_source_links(self):
        images = [{"figure_number": 2, "caption": "Figure 2. XGBoost reconstruction R = 0.71."}]
        source_paragraphs = [{"id": "source-0", "text": "Results refer to Figure 2 and report R = 0.71."}]
        bundles = _paper_figure_evidence_bundles(images, source_paragraphs)
        self.assertEqual(bundles[0]["figure_id"], "Fig. 2")
        self.assertEqual(bundles[0]["explicit_source_paragraph_ids"], ["source-0"])
        self.assertIn("R = 0.71", bundles[0]["quantitative_anchors"])

    def test_paper_canonical_registry_keeps_anchor_source_provenance(self):
        source_paragraphs = [
            {"id": "source-0", "text": "Abstract context."},
            {"id": "source-8", "text": "U ISV improves Niño-3.4 reconstruction (r = 0.73)."},
            {"id": "source-47", "text": "The combined predictor improves reconstruction to r = 0.73 (Fig. 2)."},
        ]
        bundles = _paper_figure_evidence_bundles(
            [{"figure_number": 2, "caption": "Figure 2. ENSO reconstruction."}],
            source_paragraphs,
        )
        registry = _paper_canonical_evidence_registry(source_paragraphs, bundles)
        records = [
            record for record in registry if record["normalized_value"] == "r=0.73"
        ]
        self.assertTrue(records)
        self.assertEqual(records[0]["source_paragraph_ids"], ["source-8"])
        self.assertIn("r = 0.73", records[0]["source_sentence"])
        self.assertEqual(records[0]["scope"], "section_context")

    def test_paper_canonical_registry_assigns_quantitative_ownership_once(self):
        source_paragraphs = [{"id": "source-52", "text": "The model agreement reaches 80 % in this case."}]
        registry = _paper_canonical_evidence_registry(source_paragraphs, [])
        generic = next(record for record in registry if record["evidence_id"] == "evidence-source-source-52")
        owners = [record for record in registry if "80%" in record["normalized_value"] and record.get("anchors")]
        self.assertEqual(generic["anchors"], [])
        self.assertEqual(len(owners), 1)
        self.assertEqual(owners[0]["source_paragraph_ids"], ["source-52"])
        self.assertEqual(owners[0]["anchors"], ["80 %"])

    def test_paper_canonical_registry_deduplicates_same_provenance_owner(self):
        source_paragraphs = [
            {"id": "source-0", "text": "Abstract context."},
            {"id": "source-52", "text": "The model agreement reaches 80 % in this case."},
        ]
        bundles = [{"provenance": [
            {
                "value": "80 %", "normalized_value": "80%",
                "source_paragraph_ids": ["source-52"],
                "source_sentence": "The model agreement reaches 80 % in this case.",
                "scope": "figure_specific", "supported_figures": ["Fig. 2"],
            },
        ]}]
        registry = _paper_canonical_evidence_registry(source_paragraphs, bundles)
        owners = [record for record in registry if record.get("anchors")]
        self.assertEqual(len(owners), 1)
        self.assertEqual(owners[0]["evidence_id"], _paper_stable_evidence_id("anchor", "80%", ("source-52",), "figure_specific", ("Fig. 2",)))

    def test_paper_canonical_registry_keeps_same_value_from_distinct_sources(self):
        source_paragraphs = [
            {"id": "source-A", "text": "Evidence A reports 80 %."},
            {"id": "source-B", "text": "Evidence B reports 80 %."},
        ]
        registry = _paper_canonical_evidence_registry(source_paragraphs, [])
        owners = [record for record in registry if record.get("anchors")]
        self.assertEqual({tuple(record["source_paragraph_ids"]) for record in owners}, {("source-A",), ("source-B",)})

    def test_paper_style_candidate_rolls_back_anchor_loss_and_keeps_metadata(self):
        registry = [{
            "evidence_id": "evidence-anchor", "value": "80 %", "normalized_value": "80%",
            "source_paragraph_ids": ["source-52"], "source_sentence": "Evidence reports 80 %.",
            "scope": "section_context", "supported_figures": [], "anchors": ["80 %"],
        }]
        section = {
            "id": "beat-1", "title": "结果", "role": "result", "figure_ids": [],
            "source_paragraph_ids": ["source-52"],
            "findings": [{"id": "finding-1", "evidence_ids": ["evidence-anchor"]}],
            "story_beat": {"evidence_ids": ["evidence-anchor"]},
            "blocks": [{"id": "beat-1-block-1", "evidence_ids": ["evidence-anchor"],
                        "figure_ids": [], "source_paragraph_ids": ["source-52"],
                        "text": "结果为80 %。"}],
        }
        plan = {"sections": [section], "story_evidence": {
            "evidence-anchor": {"figure_ids": [], "anchors": ["80 %"], "source_paragraph_ids": ["source-52"]}
        }}
        evidence_map = {"evidence-anchor": (section, section["findings"][0])}
        block_specs = {"beat-1": [{"block_id": "beat-1-block-1", "evidence_ids": ["evidence-anchor"], "figure_ids": [], "source_paragraph_ids": ["source-52"]}]}
        baseline = copy.deepcopy(plan["sections"])
        for candidate_text in ("结果显著增加。", "结果约八成。"):
            accepted, markdown = _paper_apply_story_candidate(
                plan,
                [{"id": "beat-1", "title": "结果", "blocks": [{
                    "block_id": "beat-1-block-1", "text": candidate_text,
                }]}],
                evidence_map, registry, block_specs, "标题", "摘要",
                {"source-52"}, None, "# 标题\n\n## 结果\n\n结果为80 %。", "humanizer",
            )
            self.assertFalse(accepted)
            self.assertEqual(markdown, "# 标题\n\n## 结果\n\n结果为80 %。")
            self.assertEqual(plan["sections"], baseline)
        accepted, markdown = _paper_apply_story_candidate(
            plan,
            [{"id": "beat-1", "title": "结果", "blocks": [{
                "block_id": "wrong-block", "text": "结果为80 %。",
            }]}],
            evidence_map, registry, block_specs, "标题", "摘要",
            {"source-52"}, None, "# 标题\n\n## 结果\n\n结果为80 %。", "style reviewer",
        )
        self.assertFalse(accepted)
        self.assertEqual(markdown, "# 标题\n\n## 结果\n\n结果为80 %。")
        self.assertEqual(plan["sections"], baseline)

    def test_paper_style_candidate_commits_valid_anchor_and_python_metadata(self):
        registry = [{
            "evidence_id": "evidence-anchor", "value": "80 %", "normalized_value": "80%",
            "source_paragraph_ids": ["source-52"], "source_sentence": "Evidence reports 80 %.",
            "scope": "section_context", "supported_figures": [], "anchors": ["80 %"],
        }]
        section = {
            "id": "beat-1", "title": "结果", "role": "result", "figure_ids": [],
            "source_paragraph_ids": ["source-52"],
            "findings": [{"id": "finding-1", "evidence_ids": ["evidence-anchor"]}],
            "story_beat": {"evidence_ids": ["evidence-anchor"]},
            "blocks": [{"id": "beat-1-block-1", "evidence_ids": ["evidence-anchor"],
                        "figure_ids": [], "source_paragraph_ids": ["source-52"],
                        "text": "结果为80 %。"}],
        }
        plan = {"sections": [section], "story_evidence": {
            "evidence-anchor": {"figure_ids": [], "anchors": ["80 %"], "source_paragraph_ids": ["source-52"]}
        }}
        accepted, markdown = _paper_apply_story_candidate(
            plan,
            [{"id": "beat-1", "title": "结果", "blocks": [{
                "block_id": "beat-1-block-1", "text": "结果仍为80 %。",
                # Model metadata must never override the Python-owned binding.
                "evidence_ids": ["rogue-evidence"],
                "figure_ids": ["Fig. 99"],
                "source_paragraph_ids": ["source-rogue"],
            }]}],
            {"evidence-anchor": (section, section["findings"][0])}, registry,
            {"beat-1": [{"block_id": "beat-1-block-1", "evidence_ids": ["evidence-anchor"], "figure_ids": [], "source_paragraph_ids": ["source-52"]}]},
            "标题", "摘要", {"source-52"}, None, "# 标题\n\n## 结果\n\n结果为80 %。", "humanizer",
        )
        self.assertTrue(accepted)
        self.assertIn("结果仍为80 %", markdown)
        self.assertEqual(plan["sections"][0]["blocks"][0]["evidence_ids"], ["evidence-anchor"])
        self.assertEqual(plan["sections"][0]["blocks"][0]["source_paragraph_ids"], ["source-52"])

    def test_paper_canonical_owner_anchor_is_required_in_bound_block(self):
        registry = [{
            "evidence_id": "evidence-anchor", "value": "80 %", "normalized_value": "80%",
            "source_paragraph_ids": ["source-52"], "source_sentence": "Evidence reports 80 %.",
            "scope": "section_context", "supported_figures": [], "anchors": ["80 %"],
        }]
        section = {
            "id": "beat-1", "title": "结果", "role": "result", "figure_ids": [],
            "source_paragraph_ids": ["source-52"],
            "findings": [{"id": "finding-1", "evidence_ids": ["evidence-anchor"]}],
            "story_beat": {"evidence_ids": ["evidence-anchor"]},
            "blocks": [{"id": "beat-1-block-1", "evidence_ids": ["evidence-anchor"],
                        "source_paragraph_ids": ["source-52"], "text": "结果为80 %。"}],
        }
        plan = {"sections": [section], "story_evidence": {
            "evidence-anchor": {"figure_ids": [], "anchors": ["80 %"], "source_paragraph_ids": ["source-52"]}
        }}
        _validate_paper_evidence_plan(plan, "# 标题\n\n## 结果\n\n结果为80 %。", {"source-52"}, evidence_registry=registry)
        section["blocks"][0]["text"] = "结果显著增加。"
        with self.assertRaisesRegex(RuntimeError, "anchor missing from bound block"):
            _validate_paper_evidence_plan(plan, "# 标题\n\n## 结果\n\n结果显著增加。", {"source-52"}, evidence_registry=registry)

    def test_paper_caption_provenance_stays_out_of_source_paragraph_ids(self):
        source_paragraphs = [{
            "id": "source-figure-3",
            "text": "Fig. 3. Forest trend relation and approximately 74% reduction.",
        }]
        bundles = _paper_figure_evidence_bundles(
            [{"figure_number": 3, "caption": source_paragraphs[0]["text"]}],
            source_paragraphs,
        )
        registry = _paper_canonical_evidence_registry(source_paragraphs, bundles)
        caption_record = next(
            record
            for record in registry
            if record["source_paragraph_ids"] == ["caption:Fig. 3"]
        )
        self.assertEqual(caption_record["supported_figures"], ["Fig. 3"])
        self.assertIn("74%", caption_record["source_sentence"])
        plan = {
            "sections": [{
                "id": "section-1",
                "title": "森林变化",
                "role": "result",
                "figure_ids": ["Fig. 3"],
                "findings": [{
                    "id": "E1",
                    "figure_ids": ["Fig. 3"],
                    "evidence_ids": [caption_record["evidence_id"]],
                }],
            }]
        }
        validated = _validate_paper_plan_structure(
            plan,
            {"source-figure-3"},
            {"Fig. 3"},
            registry,
        )
        self.assertEqual(validated[0]["source_paragraph_ids"], [])
        self.assertNotIn("caption:Fig. 3", validated[0]["source_paragraph_ids"])
        evidence_map = {
            caption_record["evidence_id"]: (
                validated[0],
                validated[0]["findings"][0],
            )
        }
        story_plan = {
            "story_beats": [{
                "id": "beat-1",
                "title": "森林变化",
                "reader_question": "森林变化如何影响结果？",
                "core_message": "森林变化与结果相关。",
                "evidence_ids": [caption_record["evidence_id"]],
                "transition_to_next": "文章收束。",
            }]
        }
        story_sections = _paper_story_sections(
            story_plan["story_beats"],
            evidence_map,
            registry,
            {"source-figure-3"},
        )
        block_specs = _paper_story_block_specs(
            story_plan,
            [{
                "evidence_id": caption_record["evidence_id"],
                "evidence_group": "evidence_group_A",
                "anchors": ["approximately 74%"],
            }],
            evidence_map,
            registry,
            {"source-figure-3"},
        )
        self.assertEqual(story_sections[0]["source_paragraph_ids"], [])
        self.assertEqual(block_specs["beat-1"][0]["source_paragraph_ids"], ())
        _validate_paper_evidence_plan(
            {
                "sections": validated,
            },
            "# 标题\n\n## 森林变化\n\n约74%的差异来自森林变化。",
            {"source-figure-3"},
            bundles,
            registry,
        )
        with self.assertRaisesRegex(RuntimeError, "unknown source paragraph id"):
            _paper_derived_source_ids(
                ["bad-evidence"],
                {"bad-evidence": {"source_paragraph_ids": ["source-999"]}},
                {"source-figure-3"},
            )

    def test_paper_planner_cannot_return_source_provenance(self):
        registry = [{
            "evidence_id": "evidence-anchor",
            "value": "r = 0.73",
            "normalized_value": "r=0.73",
            "source_paragraph_ids": ["source-8"],
            "source_sentence": "The reconstruction reaches r = 0.73.",
            "scope": "section_context",
            "supported_figures": [],
            "anchors": ["r = 0.73"],
        }]
        plan = {
            "sections": [{
                "id": "section-1",
                "title": "结果",
                "role": "result",
                "source_paragraph_ids": ["source-1"],
                "findings": [{"id": "E1", "evidence_ids": ["evidence-anchor"]}],
            }]
        }
        with self.assertRaisesRegex(RuntimeError, "source provenance"):
            _validate_paper_plan_structure(plan, {"source-1", "source-8"}, None, registry)

    def test_paper_canonical_section_sources_derive_from_evidence_ids(self):
        registry = [{
            "evidence_id": "evidence-anchor",
            "value": "r = 0.73",
            "normalized_value": "r=0.73",
            "source_paragraph_ids": ["source-8"],
            "source_sentence": "The reconstruction reaches r = 0.73.",
            "scope": "section_context",
            "supported_figures": [],
            "anchors": ["r = 0.73"],
        }]
        plan = {
            "sections": [{
                "id": "section-1",
                "title": "结果",
                "role": "result",
                "figure_ids": [],
                "findings": [{"id": "E1", "evidence_ids": ["evidence-anchor"]}],
            }]
        }
        sections = _validate_paper_plan_structure(plan, {"source-8"}, None, registry)
        self.assertEqual(sections[0]["source_paragraph_ids"], ["source-8"])
        self.assertEqual(sections[0]["findings"][0]["anchors"], ["r = 0.73"])

    def test_paper_canonical_validation_allows_duplicate_anchors_in_bound_blocks(self):
        registry = [
            {
                "evidence_id": "evidence-A",
                "value": "80 %",
                "normalized_value": "80%",
                "source_paragraph_ids": ["source-A"],
                "source_sentence": "Evidence A reports 80 %.",
                "scope": "section_context",
                "supported_figures": [],
                "anchors": ["80 %"],
            },
            {
                "evidence_id": "evidence-B",
                "value": "80 %",
                "normalized_value": "80%",
                "source_paragraph_ids": ["source-B"],
                "source_sentence": "Evidence B reports 80 %.",
                "scope": "section_context",
                "supported_figures": [],
                "anchors": ["80 %"],
            },
        ]

        def make_plan(a_text, b_text):
            return {
                "sections": [
                    {
                        "id": "section-A",
                        "title": "结果A",
                        "role": "result",
                        "figure_ids": [],
                        "source_paragraph_ids": ["source-A"],
                        "findings": [{
                            "id": "finding-A",
                            "evidence_ids": ["evidence-A"],
                            "anchors": ["80 %"],
                        }],
                        "story_beat": {"evidence_ids": ["evidence-A"]},
                        "blocks": [{
                            "id": "block-A",
                            "evidence_ids": ["evidence-A"],
                            "source_paragraph_ids": ["source-A"],
                            "text": a_text,
                        }],
                    },
                    {
                        "id": "section-B",
                        "title": "结果B",
                        "role": "result",
                        "figure_ids": [],
                        "source_paragraph_ids": ["source-B"],
                        "findings": [{
                            "id": "finding-B",
                            "evidence_ids": ["evidence-B"],
                            "anchors": ["80 %"],
                        }],
                        "story_beat": {"evidence_ids": ["evidence-B"]},
                        "blocks": [{
                            "id": "block-B",
                            "evidence_ids": ["evidence-B"],
                            "source_paragraph_ids": ["source-B"],
                            "text": b_text,
                        }],
                    },
                ],
                "story_evidence": {
                    "evidence-A": {
                        "figure_ids": [],
                        "anchors": ["80 %"],
                        "source_paragraph_ids": ["source-A"],
                    },
                    "evidence-B": {
                        "figure_ids": [],
                        "anchors": ["80 %"],
                        "source_paragraph_ids": ["source-B"],
                    },
                },
            }

        valid_markdown = (
            "# 标题\n\n## 结果A\n\nA结果为80 %。\n\n"
            "## 结果B\n\nB结果为80 %。"
        )
        _validate_paper_evidence_plan(
            make_plan("A结果为80 %。", "B结果为80 %。"),
            valid_markdown,
            {"source-A", "source-B"},
            evidence_registry=registry,
        )

        moved_markdown = (
            "# 标题\n\n## 结果A\n\nA结果待补充。\n\n"
            "## 结果B\n\nB结果包含80 %。"
        )
        with self.assertRaisesRegex(RuntimeError, "anchor missing from bound block"):
            _validate_paper_evidence_plan(
                make_plan("A结果待补充。", "B结果包含80 %。"),
                moved_markdown,
                {"source-A", "source-B"},
                evidence_registry=registry,
            )

    def test_paper_canonical_unique_anchor_without_bundle_metadata_has_no_unbound_local(self):
        registry = [{
            "evidence_id": "evidence-unique",
            "value": "80 %",
            "normalized_value": "80%",
            "source_paragraph_ids": ["source-A"],
            "source_sentence": "Evidence reports 80 %.",
            "scope": "section_context",
            "supported_figures": [],
            "anchors": ["80 %"],
        }]
        plan = {
            "sections": [{
                "id": "section-A",
                "title": "结果A",
                "role": "result",
                "figure_ids": [],
                "source_paragraph_ids": ["source-A"],
                "findings": [{
                    "id": "finding-A",
                    "evidence_ids": ["evidence-unique"],
                    "anchors": ["80 %"],
                }],
            }]
        }
        _validate_paper_evidence_plan(
            plan,
            "# 标题\n\n## 结果A\n\n结果为80 %。",
            {"source-A"},
            evidence_registry=registry,
        )

    def test_paper_canonical_section_context_anchor_uses_section_location(self):
        registry = [
            {
                "evidence_id": "evidence-context",
                "value": "80 %",
                "normalized_value": "80%",
                "source_paragraph_ids": ["source-A"],
                "source_sentence": "Evidence reports 80 %.",
                "scope": "section_context",
                "supported_figures": [],
                "anchors": ["80 %"],
            },
            {
                "evidence_id": "evidence-other",
                "value": "other",
                "normalized_value": "other",
                "source_paragraph_ids": ["source-B"],
                "source_sentence": "Other evidence.",
                "scope": "section_context",
                "supported_figures": [],
                "anchors": [],
            },
        ]
        bundles = [{
            "figure_id": "Fig. 1",
            "provenance": [{
                "normalized_value": "80%",
                "value": "80 %",
                "source_paragraph_ids": ["source-A"],
                "scope": "section_context",
                "supported_figures": [],
            }],
        }]

        def make_plan(context_index):
            evidence_by_section = (
                [["evidence-context"], ["evidence-other"]]
                if context_index == 0
                else [["evidence-other"], ["evidence-context"]]
            )
            sections = []
            for index, evidence_ids in enumerate(evidence_by_section):
                sections.append({
                    "id": f"section-{index + 1}",
                    "title": f"结果{index + 1}",
                    "role": "result",
                    "figure_ids": ["Fig. 1"],
                    "source_paragraph_ids": [
                        "source-A" if "evidence-context" in evidence_ids else "source-B"
                    ],
                    "findings": [{
                        "id": f"finding-{index + 1}",
                        "evidence_ids": evidence_ids,
                        "anchors": ["80 %"] if index == context_index else [],
                    }],
                })
            return {"sections": sections}

        markdown = "# 标题\n\n## 结果1\n\n结果为80 %。\n\n## 结果2\n\n其他证据。"
        _validate_paper_evidence_plan(
            make_plan(0), markdown, {"source-A", "source-B"}, bundles, registry
        )
        with self.assertRaisesRegex(RuntimeError, "evidence section mismatch"):
            _validate_paper_evidence_plan(
                make_plan(1), markdown, {"source-A", "source-B"}, bundles, registry
            )

    def test_paper_canonical_figure_anchor_keeps_block_and_figure_hard_failures(self):
        registry = [{
            "evidence_id": "evidence-figure",
            "value": "80 %",
            "normalized_value": "80%",
            "source_paragraph_ids": ["source-F"],
            "source_sentence": "Figure evidence reports 80 %.",
            "scope": "figure_specific",
            "supported_figures": ["Fig. 2"],
            "anchors": ["80 %"],
        }]
        bundles = [{"figure_id": "Fig. 2", "quantitative_anchors": ["80 %"]}]

        def make_plan(block_text, figure_id="Fig. 2"):
            return {
                "sections": [{
                    "id": "section-1",
                    "title": "结果",
                    "role": "result",
                    "figure_ids": ["Fig. 2"],
                    "source_paragraph_ids": ["source-F"],
                    "findings": [{
                        "id": "finding-1",
                        "figure_ids": ["Fig. 2"],
                        "evidence_ids": ["evidence-figure"],
                        "anchors": ["80 %"],
                    }],
                    "story_beat": {"evidence_ids": ["evidence-figure"]},
                    "blocks": [{
                        "id": "block-1",
                        "evidence_ids": ["evidence-figure"],
                        "figure_ids": [figure_id],
                        "source_paragraph_ids": ["source-F"],
                        "text": block_text,
                    }],
                }],
                "story_evidence": {
                    "evidence-figure": {
                        "figure_ids": ["Fig. 2"],
                        "anchors": ["80 %"],
                        "source_paragraph_ids": ["source-F"],
                    }
                },
            }

        markdown = "# 标题\n\n## 结果\n\n结果为80 %。"
        _validate_paper_evidence_plan(
            make_plan("结果为80 %。"), markdown, {"source-F"}, bundles, registry
        )
        with self.assertRaisesRegex(RuntimeError, "anchor missing from bound block"):
            _validate_paper_evidence_plan(
                make_plan("结果待补充。"),
                "# 标题\n\n## 结果\n\n结果待补充。",
                {"source-F"},
                bundles,
                registry,
            )
        with self.assertRaisesRegex(RuntimeError, "evidence block figure mismatch"):
            _validate_paper_evidence_plan(
                make_plan("结果为80 %。", "Fig. 3"),
                markdown,
                {"source-F"},
                bundles,
                registry,
            )

    def test_paper_repeated_figure_anchor_in_context_block_passes(self):
        registry = [
            {
                "evidence_id": "evidence-figure",
                "value": "1979–2024",
                "normalized_value": "1979–2024",
                "source_paragraph_ids": ["source-F"],
                "source_sentence": "Figure evidence reports 1979–2024.",
                "scope": "figure_specific",
                "supported_figures": ["Fig. 2"],
                "anchors": ["1979–2024"],
            },
            {
                "evidence_id": "evidence-context",
                "value": "1979–2024",
                "normalized_value": "1979–2024",
                "source_paragraph_ids": ["source-C"],
                "source_sentence": "Context evidence reports 1979–2024.",
                "scope": "section_context",
                "supported_figures": [],
                "anchors": ["1979–2024"],
            },
        ]
        bundles = [{"figure_id": "Fig. 2", "quantitative_anchors": ["1979–2024"]}]
        plan = {
            "sections": [{
                "id": "section-1",
                "title": "结果",
                "role": "result",
                "figure_ids": ["Fig. 2"],
                "source_paragraph_ids": ["source-F", "source-C"],
                "findings": [
                    {
                        "id": "finding-figure",
                        "figure_ids": ["Fig. 2"],
                        "evidence_ids": ["evidence-figure"],
                        "anchors": ["1979–2024"],
                    },
                    {
                        "id": "finding-context",
                        "evidence_ids": ["evidence-context"],
                        "anchors": ["1979–2024"],
                    },
                ],
                "story_beat": {
                    "evidence_ids": ["evidence-figure", "evidence-context"]
                },
                "blocks": [
                    {
                        "id": "block-figure",
                        "evidence_ids": ["evidence-figure"],
                        "figure_ids": ["Fig. 2"],
                        "source_paragraph_ids": ["source-F"],
                        "text": "图中时段为1979–2024。",
                    },
                    {
                        "id": "block-context",
                        "evidence_ids": ["evidence-context"],
                        "figure_ids": [],
                        "source_paragraph_ids": ["source-C"],
                        "text": "研究背景也覆盖1979–2024。",
                    },
                ],
            }],
            "story_evidence": {
                "evidence-figure": {
                    "figure_ids": ["Fig. 2"],
                    "anchors": ["1979–2024"],
                    "source_paragraph_ids": ["source-F"],
                },
                "evidence-context": {
                    "figure_ids": [],
                    "anchors": ["1979–2024"],
                    "source_paragraph_ids": ["source-C"],
                },
            },
        }
        markdown = "# 标题\n\n## 结果\n\n图中时段为1979–2024。\n\n研究背景也覆盖1979–2024。"
        _validate_paper_evidence_plan(
            plan,
            markdown,
            {"source-F", "source-C"},
            bundles,
            registry,
        )

    def test_paper_canonical_section_source_order_is_registry_order(self):
        registry = [
            {
                "evidence_id": "evidence-source-context",
                "value": "context",
                "normalized_value": "context",
                "source_paragraph_ids": ["source-80"],
                "source_sentence": "Context source.",
                "scope": "section_context",
                "supported_figures": [],
                "anchors": [],
            },
            {
                "evidence_id": "evidence-source-caption",
                "value": "caption",
                "normalized_value": "caption",
                "source_paragraph_ids": ["source-figure-2"],
                "source_sentence": "Caption source.",
                "scope": "figure_specific",
                "supported_figures": ["Fig. 2"],
                "anchors": [],
            },
        ]
        plan = {
            "sections": [{
                "id": "section-1",
                "title": "结果",
                "role": "result",
                "figure_ids": ["Fig. 2"],
                "findings": [
                    {"id": "E1", "figure_ids": ["Fig. 2"], "evidence_ids": ["evidence-source-caption"]},
                    {"id": "E2", "figure_ids": ["Fig. 2"], "evidence_ids": ["evidence-source-context"]},
                ],
            }]
        }
        sections = _validate_paper_plan_structure(
            plan,
            {"source-80", "source-figure-2"},
            {"Fig. 2"},
            registry,
        )
        self.assertEqual(
            sections[0]["source_paragraph_ids"],
            ["source-80", "source-figure-2"],
        )

    def test_paper_block_sources_are_recomputed_after_story_output(self):
        registry = [{
            "evidence_id": "evidence-anchor",
            "source_paragraph_ids": ["source-8"],
            "source_sentence": "The reconstruction reaches r = 0.73.",
            "scope": "section_context",
            "supported_figures": [],
            "anchors": ["r = 0.73"],
        }]
        sections = [{
            "id": "beat-1",
            "title": "结果",
            "source_paragraph_ids": ["source-8"],
            "story_beat": {"evidence_ids": ["evidence-anchor"]},
        }]
        evidence_map = {
            "evidence-anchor": (
                sections[0],
                {"figure_ids": [], "evidence_ids": ["evidence-anchor"]},
            )
        }
        _paper_apply_story_output(
            sections,
            [{
                "id": "beat-1",
                "title": "结果",
                "blocks": [{"block_id": "beat-1-block-1", "text": "r = 0.73。"}],
            }],
            evidence_map,
            registry,
            {
                "beat-1": [{
                    "block_id": "beat-1-block-1",
                    "evidence_ids": ("evidence-anchor",),
                    "figure_ids": (),
                    "source_paragraph_ids": ("source-8",),
                }],
            },
        )
        self.assertEqual(sections[0]["blocks"][0]["source_paragraph_ids"], ["source-8"])

    def test_paper_canonical_wrong_source_fails_hard(self):
        registry = [{
            "evidence_id": "evidence-anchor",
            "value": "r = 0.73",
            "normalized_value": "r=0.73",
            "source_paragraph_ids": ["source-8"],
            "source_sentence": "The reconstruction reaches r = 0.73.",
            "scope": "section_context",
            "supported_figures": [],
            "anchors": ["r = 0.73"],
        }]
        plan = {
            "sections": [{
                "id": "section-1",
                "title": "结果",
                "role": "result",
                "source_paragraph_ids": ["source-1"],
                "findings": [{"id": "E1", "evidence_ids": ["evidence-anchor"], "anchors": ["r = 0.73"]}],
            }]
        }
        markdown = "# 标题\n\n## 结果\n\nr = 0.73。"
        with self.assertRaisesRegex(RuntimeError, "source mismatch"):
            _validate_paper_evidence_plan(
                plan,
                markdown,
                {"source-1", "source-8"},
                [],
                registry,
            )

    def test_paper_canonical_source_duplicate_is_deduplicated(self):
        registry = [{
            "evidence_id": "evidence-anchor",
            "value": "r = 0.73",
            "normalized_value": "r=0.73",
            "source_paragraph_ids": ["source-8", "source-8"],
            "source_sentence": "The reconstruction reaches r = 0.73.",
            "scope": "section_context",
            "supported_figures": [],
            "anchors": ["r = 0.73"],
        }]
        self.assertEqual(
            _paper_clean_story_evidence(
                {"sections": [{"role": "result", "source_paragraph_ids": ["source-8"], "findings": [{"evidence": "r = 0.73", "anchors": ["r = 0.73"]}]}]},
                [{"id": "source-8", "text": "The reconstruction reaches r = 0.73."}],
            )[0][0]["source_paragraph_ids"],
            ["source-8"],
        )

    def test_paper_provenance_figure_specific_mismatch_is_rejected(self):
        images = [
            {"figure_number": 2, "caption": "Figure 2. Correlation coefficient R = 0.71."},
            {"figure_number": 3, "caption": "Figure 3. Forest correlation R = −0.77."},
        ]
        source_paragraphs = [
            {"id": "source-0", "text": "The reconstruction reports R = 0.71 in Figure 2."},
        ]
        bundles = _paper_figure_evidence_bundles(images, source_paragraphs)
        plan = {
            "sections": [
                {
                    "id": "section-1",
                    "title": "森林结果",
                    "figure_ids": ["Fig. 3"],
                    "source_paragraph_ids": ["source-0"],
                    "findings": [{"id": "E1", "evidence": "错误归属", "anchors": ["R = 0.71"]}],
                }
            ]
        }
        markdown = "# 标题\n\n## 森林结果\n\nR = 0.71。"
        with self.assertRaisesRegex(RuntimeError, "PAPER evidence figure mismatch"):
            _validate_paper_evidence_plan(plan, markdown, {"source-0"}, bundles)

    def test_paper_provenance_contextual_period_does_not_require_a_figure(self):
        images = [{"figure_number": 1, "caption": "Figure 1. Historical wind-speed patterns."}]
        source_paragraphs = [
            {"id": "source-0", "text": "The full-paper historical period is 1970–2014."},
        ]
        bundles = _paper_figure_evidence_bundles(images, source_paragraphs)
        provenance = [record for bundle in bundles for record in bundle["provenance"]]
        self.assertEqual(provenance[0]["scope"], "global_context")
        self.assertEqual(provenance[0]["supported_figures"], [])
        plan = {
            "sections": [
                {
                    "id": "section-1",
                    "title": "历史模拟",
                    "figure_ids": ["Fig. 1"],
                    "source_paragraph_ids": ["source-0"],
                    "findings": [{"id": "E1", "evidence": "研究时段", "anchors": ["1970–2014"]}],
                }
            ]
        }
        markdown = "# 标题\n\n## 历史模拟\n\n研究时段为1970—2014。"
        _validate_paper_evidence_plan(plan, markdown, {"source-0"}, bundles)

    def test_paper_provenance_projection_period_is_section_context(self):
        images = [{"figure_number": 4, "caption": "Figure 4. Future wind-speed projection spread."}]
        source_paragraphs = [
            {"id": "source-0", "text": "The abstract introduces the study."},
            {"id": "source-1", "text": "The projection periods are 2025–2054 and 2070–2099."},
        ]
        bundles = _paper_figure_evidence_bundles(images, source_paragraphs)
        records = [
            record
            for bundle in bundles
            for record in bundle["provenance"]
            if record["normalized_value"] == "2025-2054"
        ]
        self.assertEqual(records[0]["scope"], "section_context")
        self.assertEqual(records[0]["supported_figures"], [])

    def test_paper_provenance_same_sentence_keeps_anchor_with_nearest_figure(self):
        images = [
            {"figure_number": 2, "caption": "Figure 2. Reconstruction correlation."},
            {"figure_number": 3, "caption": "Figure 3. Forest correlation."},
        ]
        source_paragraphs = [{
            "id": "source-0",
            "text": "XGBoost reconstruction reports R = 0.71 (Figure 2a); forest correlation reports R = −0.77 (Figure 3b).",
        }]
        bundles = _paper_figure_evidence_bundles(images, source_paragraphs)
        by_id = {bundle["figure_id"]: bundle for bundle in bundles}
        self.assertEqual(by_id["Fig. 2"]["supported_figures_by_anchor"]["R = 0.71"], ["Fig. 2"])
        self.assertEqual(by_id["Fig. 3"]["supported_figures_by_anchor"]["R = −0.77"], ["Fig. 3"])
        self.assertNotIn("R = 0.71", by_id["Fig. 3"]["quantitative_anchors"])

    def test_paper_provenance_ambiguous_reference_stays_contextual(self):
        images = [
            {"figure_number": 2, "caption": "Figure 2. Reconstruction."},
            {"figure_number": 3, "caption": "Figure 3. Forest result."},
        ]
        source_paragraphs = [{
            "id": "source-0",
            "text": "The result R = 0.71 is reported across Figure 2 and Figure 3.",
        }]
        bundles = _paper_figure_evidence_bundles(images, source_paragraphs)
        records = [
            record
            for bundle in bundles
            for record in bundle["provenance"]
            if record["normalized_value"] == "R=0.71"
        ]
        self.assertTrue(records)
        self.assertTrue(all(record["scope"] != "figure_specific" for record in records))
        self.assertTrue(all(not record["supported_figures"] for record in records))

    def test_paper_provenance_supplementary_reference_is_not_main_figure_support(self):
        images = [{"figure_number": 1, "caption": "Figure 1. Main result."}]
        source_paragraphs = [{
            "id": "source-0",
            "text": "Supplementary Figure 2 reports R = 0.55.",
        }]
        bundles = _paper_figure_evidence_bundles(images, source_paragraphs)
        self.assertNotIn("R = 0.55", bundles[0]["quantitative_anchors"])
        self.assertTrue(all(
            "R=0.55" != record["normalized_value"]
            or not record["supported_figures"]
            for record in bundles[0]["provenance"]
        ))

    def test_paper_pruning_uses_actual_allocation_and_reviewer_decision(self):
        plan = {
            "sections": [
                {
                    "id": "section-1",
                    "title": "主图结果",
                    "role": "phenomenon",
                    "source_paragraph_ids": ["source-figure-1"],
                    "findings": [{"id": "E1", "evidence": "主图结果", "anchors": []}],
                },
                {
                    "id": "section-2",
                    "title": "悬空补充",
                    "role": "minor mechanism",
                    "source_paragraph_ids": ["source-0"],
                    "findings": [{"id": "E2", "evidence": "次要机制", "anchors": []}],
                },
            ]
        }
        allocation = {
            "sections": [
                {"section_index": 0, "section": "主图结果", "selected_figures": ["Fig. 1"]},
                {"section_index": 1, "section": "悬空补充", "selected_figures": []},
            ]
        }
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {"decisions": [{"section_id": "section-2", "action": "prune", "reason": "不承载不可替代的核心证据"}]},
                            ensure_ascii=False,
                        )
                    )
                )
            ]
        )
        settings = replace(load_settings(), model_base_url="https://model.example/v1", model_api_key="test-key", model_name="test-model")
        with tempfile.TemporaryDirectory() as tmp, patch("writer.llm.OpenAI", return_value=MagicMock()) as openai:
            client = openai.return_value
            client.chat.completions.create.return_value = response
            markdown_path = Path(tmp) / "article.md"
            markdown_path.write_text("# 标题\n\n摘要。\n\n## 主图结果\n\n结果正文。\n\n## 悬空补充\n\n补充正文。\n", encoding="utf-8")
            dossier = {"abstract": "摘要。", "text": "source", "images": [], "paper_evidence_plan": plan}
            prune_paper_sections_after_allocation(markdown_path, dossier, allocation, settings)
            final_markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual([section["id"] for section in plan["sections"]], ["section-1"])
        self.assertEqual(plan["pruned_sections"][0]["id"], "section-2")
        self.assertNotIn("## 悬空补充", final_markdown)
        self.assertEqual(allocation["sections"][0]["selected_figures"], ["Fig. 1"])

    def test_paper_pruning_keeps_reviewer_selected_bridge_without_figure(self):
        plan = {
            "sections": [
                {
                    "id": "section-1",
                    "title": "必要桥梁",
                    "role": "transition",
                    "source_paragraph_ids": ["source-0"],
                    "necessary_transition": False,
                    "findings": [{"id": "E1", "evidence": "桥梁", "anchors": []}],
                }
            ]
        }
        allocation = {"sections": [{"section_index": 0, "section": "必要桥梁", "selected_figures": []}]}
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps({"decisions": [{"section_id": "section-1", "action": "keep", "reason": "连接两个不可替代的科学结果"}]}, ensure_ascii=False)
                    )
                )
            ]
        )
        settings = replace(load_settings(), model_base_url="https://model.example/v1", model_api_key="test-key", model_name="test-model")
        with tempfile.TemporaryDirectory() as tmp, patch("writer.llm.OpenAI", return_value=MagicMock()) as openai:
            openai.return_value.chat.completions.create.return_value = response
            markdown_path = Path(tmp) / "article.md"
            markdown_path.write_text("# 标题\n\n摘要。\n\n## 必要桥梁\n\n桥梁正文。\n", encoding="utf-8")
            dossier = {"abstract": "摘要。", "text": "source", "images": [], "paper_evidence_plan": plan}
            prune_paper_sections_after_allocation(markdown_path, dossier, allocation, settings)
            final_markdown = markdown_path.read_text(encoding="utf-8")

        self.assertTrue(plan["sections"][0]["retained_without_figure"])
        self.assertEqual(plan["sections"][0]["retention_reason"], "连接两个不可替代的科学结果")
        self.assertEqual(allocation["sections"][0]["selected_figures"], [])

    def test_paper_pruning_falls_back_when_all_sections_would_be_removed(self):
        plan = {
            "sections": [
                {"id": "section-1", "title": "结果一", "role": "result", "source_paragraph_ids": ["source-0"], "findings": []},
                {"id": "section-2", "title": "结果二", "role": "result", "source_paragraph_ids": ["source-0"], "findings": []},
            ]
        }
        allocation = {
            "sections": [
                {"section_index": 0, "section": "结果一", "selected_figures": []},
                {"section_index": 1, "section": "结果二", "selected_figures": []},
            ]
        }
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps({"decisions": [{"section_id": "section-1", "action": "prune", "reason": "弱"}, {"section_id": "section-2", "action": "prune", "reason": "弱"}]}, ensure_ascii=False)
                    )
                )
            ]
        )
        settings = replace(load_settings(), model_base_url="https://model.example/v1", model_api_key="test-key", model_name="test-model")
        with tempfile.TemporaryDirectory() as tmp, patch("writer.llm.OpenAI", return_value=MagicMock()) as openai:
            openai.return_value.chat.completions.create.return_value = response
            markdown_path = Path(tmp) / "article.md"
            markdown_path.write_text("# 标题\n\n摘要。\n\n## 结果一\n\n正文一。\n\n## 结果二\n\n正文二。\n", encoding="utf-8")
            dossier = {"abstract": "摘要。", "text": "source", "images": [], "paper_evidence_plan": plan}
            prune_paper_sections_after_allocation(markdown_path, dossier, allocation, settings)
            final_markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual(plan["pruning_fallback_reason"], "all sections would be removed")
        self.assertEqual([section["id"] for section in plan["sections"]], ["section-1", "section-2"])
        self.assertIn("## 结果一", final_markdown)
        self.assertIn("## 结果二", final_markdown)

    def test_paper_ai_style_lint_detects_repeated_connectives(self):
        markdown = "并非A而是B。并非C而是D。进一步表明结果稳定。进一步表明趋势一致。"
        counts = _paper_ai_style_lint(markdown)
        self.assertEqual(counts["并非而是"], 2)
        self.assertEqual(counts["进一步表明"], 2)
        self.assertTrue(_paper_ai_style_lint_failed(counts))
        clean_counts = _paper_ai_style_lint("森林变化解释了模式差异。未来投影仍有不确定性。")
        self.assertFalse(_paper_ai_style_lint_failed(clean_counts))

    def test_paper_style_exemplar_loads_every_arbitrary_markdown_document(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            exemplar_dir = root / "writer" / "exemplars"
            exemplar_dir.mkdir(parents=True)
            (exemplar_dir / "STYLE_GUIDE.md").write_text(
                "规则标记：只学习语言，不迁移事实。", encoding="utf-8"
            )
            for index in range(1, 6):
                (exemplar_dir / f"exemplar_{index:02d}.md").write_text(
                    f"# 范文{index}\n\n完整正文标记{index}。\n\n结尾标记{index}。",
                    encoding="utf-8",
                )
            (exemplar_dir / "new_voice.md").write_text(
                "# 新增范文\n\n任意文件名也应自动进入语料库。\n\n新增结尾。",
                encoding="utf-8",
            )
            nested = exemplar_dir / "archive"
            nested.mkdir()
            (nested / "field_notes.md").write_text(
                "# 嵌套范文\n\n嵌套目录中的正文也应自动进入语料库。",
                encoding="utf-8",
            )
            (exemplar_dir / "README.md").write_text(
                "说明文件不属于正文语料。", encoding="utf-8"
            )
            historical = root / "articles" / "paper" / "old"
            historical.mkdir(parents=True)
            (historical / "article.md").write_text(
                "不应被读取的历史文章标记。", encoding="utf-8"
            )
            with patch("writer.llm.PROJECT_ROOT", root):
                package = _paper_style_exemplar()
        self.assertIn("规则标记", package)
        for index in range(1, 6):
            self.assertIn(f"范文文件：exemplar_{index:02d}.md", package)
            self.assertIn(f"完整正文标记{index}", package)
        self.assertIn("范文文件：new_voice.md", package)
        self.assertIn("任意文件名也应自动进入语料库", package)
        self.assertIn("嵌套范文", package)
        self.assertIn("嵌套目录中的正文也应自动进入语料库", package)
        self.assertNotIn("说明文件不属于正文语料", package)
        self.assertNotIn("不应被读取的历史文章标记", package)

    def test_paper_style_exemplar_passes_small_corpus_in_full(self):
        project_root = Path(__file__).resolve().parents[1]
        exemplar_dir = project_root / "writer" / "exemplars"
        package = _paper_style_exemplar()
        for path in sorted(exemplar_dir.glob("exemplar_*.md")):
            text = path.read_text(encoding="utf-8").strip()
            self.assertIn(text, package)
        self.assertIn("STYLE_GUIDE（辅助规则", package)
        self.assertIn("STYLE CORPUS（主要参考", package)

    def test_paper_style_exemplar_compresses_large_corpus_deterministically(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            exemplar_dir = root / "writer" / "exemplars"
            exemplar_dir.mkdir(parents=True)
            (exemplar_dir / "STYLE_GUIDE.md").write_text("辅助规则。", encoding="utf-8")
            for index in range(6):
                (exemplar_dir / f"voice_{index}.md").write_text(
                    f"# 标题{index}\n\n开头标记{index}。\n\n"
                    + (f"正文标记{index}。\n" * 20)
                    + f"\n结尾标记{index}。",
                    encoding="utf-8",
                )
            with patch("writer.llm.PROJECT_ROOT", root), patch(
                "writer.llm._PAPER_STYLE_CORPUS_CHAR_BUDGET", 1200
            ):
                package = _paper_style_exemplar()
        for index in range(6):
            self.assertIn(f"范文文件：voice_{index}.md", package)
            self.assertIn(f"标题{index}", package)
            self.assertIn(f"开头标记{index}", package)
            self.assertIn(f"结尾标记{index}", package)
        self.assertLessEqual(package.count("范文文件："), 6)

    def test_paper_style_payload_is_the_same_corpus_for_each_beat(self):
        package = _paper_style_exemplar()
        client = MagicMock()
        payloads = []
        for beat_id in ("beat-1", "beat-2"):
            story_plan = _story_plan_for_evidence(1)
            story_plan["story_beats"][0]["id"] = beat_id
            client.chat.completions.create.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                    "sections": [{
                        "id": beat_id,
                        "title": "科学结果",
                        "blocks": [{
                            "block_id": f"{beat_id}-block-1",
                            "text": "森林变化解释了模式差异。",
                        }],
                    }]
                }, ensure_ascii=False)))]
            )
            _paper_story_writer(
                client,
                story_plan,
                [{
                    "evidence_id": "evidence-1",
                    "evidence_group": "context",
                    "anchors": [],
                }],
                package,
                "test-model",
            )
            payloads.append(json.loads(
                client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
            )["style_exemplar"])
        self.assertEqual(payloads, [package, package])

    def test_paper_ai_style_lint_covers_requested_templates(self):
        phrases = (
            "研究发现", "结果表明", "进一步分析", "值得注意的是",
            "这意味着", "这一发现表明", "综上所述",
        )
        counts = _paper_ai_style_lint("。".join(phrases))
        for phrase in phrases:
            self.assertEqual(counts[phrase], 1)
        self.assertTrue(_paper_ai_style_lint_failed(counts))

    def test_paper_ai_style_lint_flags_author_voice_only_in_body(self):
        markdown = (
            "# 我们发现了新的结果\n\n"
            "## 科学结果\n\n我们展示了敏感性试验的结果。"
        )
        counts = _paper_ai_style_lint(markdown)
        self.assertEqual(counts["作者式第一人称"], 1)
        self.assertTrue(_paper_ai_style_lint_failed(counts))
        quoted = (
            "# 我们发现了新的结果\n\n"
            "## 科学结果\n\n英文原文为: We show the result."
        )
        self.assertEqual(_paper_ai_style_lint(quoted)["作者式第一人称"], 0)

    def test_paper_ai_style_lint_rejects_one_hit_antithesis_patterns(self):
        examples = (
            ("不是而是", "不是A，而是B。"),
            ("并不是而是", "并不是A，而是B。"),
            ("并非而是", "并非A，而是B。"),
            ("不在而在", "不在A，而在B。"),
            ("不只是更是", "不只是A，更是B。"),
            ("不仅更", "不仅A，更重要的是B。"),
            ("真正不是而是", "真正重要的不是A，而是B。"),
            ("与其说不如说", "与其说是A，不如说是B。"),
        )
        for key, text in examples:
            with self.subTest(key=key):
                counts = _paper_ai_style_lint(text)
                self.assertEqual(counts[key], 1)
                self.assertTrue(_paper_ai_style_lint_failed(counts))
        self.assertFalse(_paper_ai_style_lint_failed(_paper_ai_style_lint("但是结果仍然稳定，因此可以继续比较。")))

    def test_paper_ai_style_lint_covers_standalone_article_expressions_and_abstract(self):
        for key, text in (
            ("并非", "并非来自当地蒸发。"),
            ("并不是", "并不是来自当地蒸发。"),
            ("而不是", "主要来自远洋输送，而不是当地蒸发。"),
            ("roughly", "roughly 68% 的区域出现变化。"),
            ("作者式第一人称", "我们发现结果仍然稳定。"),
        ):
            with self.subTest(key=key):
                counts = _paper_ai_style_lint(text)
                self.assertEqual(counts[key], 1)
                self.assertTrue(_paper_ai_style_lint_failed(counts))
        full_article = "# 标题\\n\\n研究发现我们发现结果。\\n\\n## 结果\\n\\n并非来自当地蒸发，而不是别的过程。"
        counts = _paper_ai_style_lint(full_article, include_abstract=True)
        self.assertEqual(counts["作者式第一人称"], 1)
        self.assertEqual(counts["并非"], 1)
        self.assertEqual(counts["而不是"], 1)

    def test_paper_plain_language_cleanup_translates_english_qualifiers(self):
        cleaned = _paper_plain_language_cleanup("roughly 68% and about 9% of the area changed")
        self.assertNotIn("roughly", cleaned)
        self.assertNotIn("about", cleaned)
        self.assertIn("约68%", cleaned)
        self.assertIn("约9%", cleaned)

    def test_paper_abstract_deauthoring_preserves_scientific_content(self):
        source = "我们的结果表明变化可能来自远洋输送；我们发现约74%的区域受影响。"
        cleaned = _paper_deauthor_abstract(source)
        self.assertNotIn("我们", cleaned)
        self.assertIn("研究结果表明变化可能来自远洋输送", cleaned)
        self.assertIn("约74%的区域受影响", cleaned)

    def test_paper_ai_style_lint_flags_broad_body_author_voice(self):
        for phrase in ("我们可以看到", "我们看到", "我们注意到", "咱们"):
            with self.subTest(phrase=phrase):
                counts = _paper_ai_style_lint(f"{phrase}结果仍然稳定。")
                self.assertEqual(counts["作者式第一人称"], 1)
                self.assertTrue(_paper_ai_style_lint_failed(counts))

    def test_paper_article_editor_receives_full_article_and_corpus(self):
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                "sections": [
                    {
                        "section_id": "section-2",
                        "title": "机制如何接上结果",
                        "paragraphs": [{"block_ids": ["section-2-block-1"], "text": "第二段。"}],
                    },
                    {
                        "section_id": "section-1",
                        "title": "先看现象",
                        "paragraphs": [{"block_ids": ["section-1-block-1"], "text": "第一段。"}],
                    },
                ]
            }, ensure_ascii=False)))]
        )
        plan = {"sections": [
            {"id": "section-1", "title": "现象", "blocks": [{"id": "section-1-block-1", "text": "旧一。"}]},
            {"id": "section-2", "title": "机制", "blocks": [{"id": "section-2-block-1", "text": "旧二。"}]},
        ]}
        specs = {
            "section-1": [{"block_id": "section-1-block-1", "beat_id": "section-1", "evidence_ids": ("e1",), "anchors": ("80%",), "figure_ids": ("Fig. 1",)}],
            "section-2": [{"block_id": "section-2-block-1", "beat_id": "section-2", "evidence_ids": ("e2",), "anchors": (), "figure_ids": ()}],
        }
        corpus = "范文文件：exemplar_01.md\\n完整范文全文"
        result = _paper_article_editor(
            client, plan, "# 标题\\n\\n摘要\\n\\n## 现象\\n\\n旧一。\\n\\n## 机制\\n\\n旧二。",
            "摘要", corpus, "test-model", specs,
            {"article_level": "check cross-block repetition"},
        )
        self.assertEqual([item["id"] for item in result], ["section-2", "section-1"])
        payload = json.loads(client.chat.completions.create.call_args.kwargs["messages"][1]["content"])
        self.assertEqual(payload["style_exemplar"], corpus)
        self.assertEqual(payload["draft"].count("旧"), 2)
        self.assertEqual({item["section_id"] for item in payload["sections"]}, {"section-1", "section-2"})
        self.assertEqual({item["block_id"] for item in payload["writing_facts"]}, {"section-1-block-1", "section-2-block-1"})
        self.assertEqual(
            next(item for item in payload["writing_facts"] if item["block_id"] == "section-1-block-1")["required_facts"],
            ["80%"],
        )
        self.assertTrue(
            next(item for item in payload["writing_facts"] if item["block_id"] == "section-1-block-1")["has_figure"]
        )
        self.assertNotIn("anchors", json.dumps(payload, ensure_ascii=False))
        self.assertNotIn("evidence_ids", json.dumps(payload, ensure_ascii=False))
        self.assertNotIn("mandatory_anchors", json.dumps(payload, ensure_ascii=False))
        self.assertNotIn("source_paragraph_ids", json.dumps(payload, ensure_ascii=False))
        serialized_payload = json.dumps(payload, ensure_ascii=False)
        for forbidden in (
            "mandatory_anchors", "anchors", "source_paragraph_ids", "supported_figures",
            "provenance", "mandatory anchor", "required fact",
        ):
            self.assertNotIn(forbidden, serialized_payload)
        self.assertEqual(payload["article_feedback"]["article_level"], "check cross-block repetition")
        self.assertIn("五篇范文全文", PAPER_ARTICLE_EDITOR_PROMPT)

    def test_paper_article_editor_rejects_section_and_block_id_contract_breaks(self):
        plan = {"sections": [
            {"id": "section-1", "title": "一", "blocks": []},
            {"id": "section-2", "title": "二", "blocks": []},
        ]}
        specs = {
            "section-1": [{"block_id": "b1", "beat_id": "section-1"}],
            "section-2": [{"block_id": "b2", "beat_id": "section-2"}],
        }
        responses = [
            {"sections": [{"section_id": "section-1", "title": "一", "paragraphs": [{"block_ids": ["b1"], "text": "一"}]}]},
            {"sections": [
                {"section_id": "section-1", "title": "一", "paragraphs": [{"block_ids": ["b1"], "text": "一"}]},
                {"section_id": "section-1", "title": "重复", "paragraphs": [{"block_ids": ["b2"], "text": "二"}]},
            ]},
            {"sections": [
                {"section_id": "unknown", "title": "未知", "paragraphs": [{"block_ids": ["b1"], "text": "一"}]},
                {"section_id": "section-2", "title": "二", "paragraphs": [{"block_ids": ["b2"], "text": "二"}]},
            ]},
            {"sections": [
                {"section_id": "section-1", "title": "一", "paragraphs": [{"block_ids": ["b1"], "text": "一"}]},
                {"section_id": "section-2", "title": "二", "paragraphs": [{"block_ids": ["b1"], "text": "二"}]},
            ]},
        ]
        for response in responses:
            with self.subTest(response=response):
                with self.assertRaises(RuntimeError):
                    _paper_normalize_article_editor_output(response, plan, specs)

    def test_paper_article_editor_accepts_merged_paragraph_and_unions_bindings(self):
        section = {
            "id": "section-1",
            "title": "结果",
            "role": "result",
            "findings": [{"id": "finding-1", "evidence_ids": ["e1", "e2"]}],
            "story_beat": {"evidence_ids": ["e1", "e2"]},
        }
        plan = {"sections": [section]}
        specs = {
            "section-1": [
                {
                    "block_id": "b1",
                    "beat_id": "section-1",
                    "evidence_ids": ("e1",),
                    "anchors": ("68%",),
                    "figure_ids": ("Fig. 1",),
                    "source_paragraph_ids": ("source-1",),
                },
                {
                    "block_id": "b2",
                    "beat_id": "section-1",
                    "evidence_ids": ("e2",),
                    "anchors": ("49%",),
                    "figure_ids": ("Fig. 2",),
                    "source_paragraph_ids": ("source-2",),
                },
            ]
        }
        evidence_map = {
            "e1": (section, section["findings"][0]),
            "e2": (section, section["findings"][0]),
        }
        generated = [{
            "id": "section-1",
            "title": "结果如何连起来",
            "paragraphs": [{
                "block_ids": ["b1", "b2"],
                "text": "结果覆盖约68%的区域，气候平均贡献约49%。",
            }],
        }]
        registry = [
            {"evidence_id": "e1", "source_paragraph_ids": ["source-1"], "source_sentence": "约68%。", "anchors": ["68%"]},
            {"evidence_id": "e2", "source_paragraph_ids": ["source-2"], "source_sentence": "约49%。", "anchors": ["49%"]},
        ]
        rebuilt = _paper_article_editor_rebuild_sections(
            generated, plan, evidence_map, registry, specs, {"source-1", "source-2"}
        )
        paragraph = rebuilt[0]["paragraphs"][0]
        self.assertEqual(paragraph["block_ids"], ["b1", "b2"])
        self.assertEqual(paragraph["evidence_ids"], ["e1", "e2"])
        self.assertEqual(paragraph["source_paragraph_ids"], ["source-1", "source-2"])
        self.assertEqual(paragraph["figure_ids"], ["Fig. 1", "Fig. 2"])
        self.assertEqual(rebuilt[0]["blocks"][0]["evidence_ids"], ["e1"])
        self.assertEqual(rebuilt[0]["blocks"][1]["evidence_ids"], ["e2"])

    def test_paper_article_editor_rejects_duplicate_missing_and_unknown_block_ids(self):
        plan = {"sections": [{"id": "section-1", "title": "一", "blocks": []}]}
        specs = {"section-1": [
            {"block_id": "b1", "beat_id": "section-1"},
            {"block_id": "b2", "beat_id": "section-1"},
        ]}
        responses = (
            {"sections": [{"section_id": "section-1", "title": "一", "paragraphs": [
                {"block_ids": ["b1", "b1"], "text": "一。"},
                {"block_ids": ["b2"], "text": "二。"},
            ]}]},
            {"sections": [{"section_id": "section-1", "title": "一", "paragraphs": [
                {"block_ids": ["b1"], "text": "一。"},
            ]}]},
            {"sections": [{"section_id": "section-1", "title": "一", "paragraphs": [
                {"block_ids": ["b1", "unknown"], "text": "一。"},
                {"block_ids": ["b2"], "text": "二。"},
            ]}]},
            {"sections": [{"section_id": "section-1", "title": "一", "paragraphs": [
                {"block_ids": ["b1"], "text": "一。", "evidence_ids": ["e1"]},
                {"block_ids": ["b2"], "text": "二。"},
            ]}]},
        )
        for response in responses:
            with self.subTest(response=response):
                with self.assertRaises(RuntimeError):
                    _paper_normalize_article_editor_output(response, plan, specs)

    def test_paper_article_editor_merged_paragraph_preserves_and_requires_each_anchor(self):
        registry = [
            {"evidence_id": "e1", "source_paragraph_ids": ["source-1"], "source_sentence": "约68%。", "scope": "section_context", "supported_figures": [], "anchors": ["68%"]},
            {"evidence_id": "e2", "source_paragraph_ids": ["source-2"], "source_sentence": "约49%。", "scope": "section_context", "supported_figures": [], "anchors": ["49%"]},
        ]
        section = {
            "id": "section-1", "title": "结果", "role": "result",
            "source_paragraph_ids": ["source-1", "source-2"],
            "findings": [{"id": "finding-1", "evidence_ids": ["e1", "e2"], "figure_ids": []}],
            "story_beat": {"evidence_ids": ["e1", "e2"]},
            "blocks": [
                {"id": "b1", "evidence_ids": ["e1"], "source_paragraph_ids": ["source-1"], "figure_ids": [], "text": "结果覆盖约68%的区域，气候平均贡献约49%。"},
                {"id": "b2", "evidence_ids": ["e2"], "source_paragraph_ids": ["source-2"], "figure_ids": [], "text": "结果覆盖约68%的区域，气候平均贡献约49%。"},
            ],
            "paragraphs": [{"block_ids": ["b1", "b2"], "text": "结果覆盖约68%的区域，气候平均贡献约49%。"}],
        }
        plan = {"sections": [section], "story_evidence": {
            "e1": {"figure_ids": [], "anchors": ["68%"], "source_paragraph_ids": ["source-1"]},
            "e2": {"figure_ids": [], "anchors": ["49%"], "source_paragraph_ids": ["source-2"]},
        }}
        _validate_paper_evidence_plan(
            plan,
            "# 标题\n\n## 结果\n\n结果覆盖约68%的区域，气候平均贡献约49%。",
            {"source-1", "source-2"},
            evidence_registry=registry,
        )
        section["paragraphs"][0]["text"] = "结果覆盖约68%的区域。"
        section["blocks"][0]["text"] = section["paragraphs"][0]["text"]
        section["blocks"][1]["text"] = section["paragraphs"][0]["text"]
        with self.assertRaisesRegex(RuntimeError, "anchor missing from bound block"):
            _validate_paper_evidence_plan(
                plan,
                "# 标题\n\n## 结果\n\n结果覆盖约68%的区域。",
                {"source-1", "source-2"},
                evidence_registry=registry,
            )

    def test_paper_final_style_lint_hard_fails_after_fallback_rollback(self):
        counts = _paper_ai_style_lint("摘要中的我们仍然保留了metadata leakage。", include_abstract=True)
        with self.assertRaisesRegex(RuntimeError, "PAPER final style lint failed after local fallback"):
            _paper_require_clean_final_style_lint(counts)

    def test_paper_metadata_leakage_lint_is_deterministic(self):
        text = "证据锚点、锚点为、对应的锚点、mandatory anchor、required fact、evidence_id、block_id、provenance。"
        counts = _paper_metadata_leakage_lint(text)
        self.assertTrue(all(value > 0 for value in counts.values()))
        lint = _paper_ai_style_lint(text)
        self.assertGreater(lint["metadata_leakage"], 0)
        self.assertTrue(_paper_ai_style_lint_failed(lint))

    def test_paper_article_editor_rejects_metadata_in_paragraph_text(self):
        plan = {"sections": [{"id": "section-1", "title": "一", "blocks": []}]}
        specs = {"section-1": [{"block_id": "b1", "beat_id": "section-1"}]}
        response = {"sections": [{"section_id": "section-1", "title": "一", "paragraphs": [
            {"block_ids": ["b1"], "text": "对应的证据锚点为约68%。"},
        ]}]}
        with self.assertRaisesRegex(RuntimeError, "metadata text"):
            _paper_normalize_article_editor_output(response, plan, specs)

    def test_paper_article_editor_reorders_without_changing_python_bindings(self):
        registry = [
            {"evidence_id": "e1", "source_paragraph_ids": ["source-1"], "source_sentence": "结果为80%。", "scope": "section_context", "supported_figures": [], "anchors": ["80%"]},
            {"evidence_id": "e2", "source_paragraph_ids": ["source-2"], "source_sentence": "结果持续三年。", "scope": "section_context", "supported_figures": [], "anchors": []},
            {"evidence_id": "e3", "source_paragraph_ids": ["source-3"], "source_sentence": "机制来自水汽输送。", "scope": "section_context", "supported_figures": [], "anchors": []},
        ]
        section_one = {"id": "section-1", "title": "现象", "role": "result", "source_paragraph_ids": ["source-1", "source-2"], "findings": [{"id": "f1", "evidence_ids": ["e1", "e2"], "figure_ids": []}]}
        section_two = {"id": "section-2", "title": "机制", "role": "mechanism", "source_paragraph_ids": ["source-3"], "findings": [{"id": "f2", "evidence_ids": ["e3"], "figure_ids": []}]}
        plan = {
            "sections": [section_one, section_two],
            "story_evidence": {
                "e1": {"figure_ids": [], "anchors": ["80%"], "source_paragraph_ids": ["source-1"], "scope": "section_context", "supported_figures": []},
                "e2": {"figure_ids": [], "anchors": [], "source_paragraph_ids": ["source-2"], "scope": "section_context", "supported_figures": []},
                "e3": {"figure_ids": [], "anchors": [], "source_paragraph_ids": ["source-3"], "scope": "section_context", "supported_figures": []},
            },
        }
        evidence_map = {"e1": (section_one, section_one["findings"][0]), "e2": (section_one, section_one["findings"][0]), "e3": (section_two, section_two["findings"][0])}
        specs = {
            "section-1": [
                {"block_id": "section-1-block-1", "beat_id": "section-1", "evidence_ids": ("e1",), "anchors": ("80%",), "figure_ids": (), "source_paragraph_ids": ("source-1",)},
                {"block_id": "section-1-block-2", "beat_id": "section-1", "evidence_ids": ("e2",), "anchors": (), "figure_ids": (), "source_paragraph_ids": ("source-2",)},
            ],
            "section-2": [{"block_id": "section-2-block-1", "beat_id": "section-2", "evidence_ids": ("e3",), "anchors": (), "figure_ids": (), "source_paragraph_ids": ("source-3",)}],
        }
        generated = [
            {"id": "section-2", "title": "水汽如何接上增雪", "paragraphs": [{"block_ids": ["section-2-block-1"], "text": "机制来自水汽输送。"}]},
            {"id": "section-1", "title": "先看结果", "paragraphs": [
                {"block_ids": ["section-1-block-2"], "text": "结果持续三年。"},
                {"block_ids": ["section-1-block-1"], "text": "结果达到80%。"},
            ]},
        ]
        working = copy.deepcopy(plan)
        accepted, markdown = _paper_apply_article_editor_candidate(
            working, generated, evidence_map, registry, specs, "标题", "摘要", {"source-1", "source-2", "source-3"}, None, "旧稿", "test editor",
        )
        self.assertTrue(accepted)
        self.assertEqual([section["id"] for section in working["sections"]], ["section-2", "section-1"])
        self.assertEqual([block["id"] for block in working["sections"][1]["blocks"]], ["section-1-block-2", "section-1-block-1"])
        self.assertEqual(working["sections"][1]["blocks"][1]["evidence_ids"], ["e1"])
        self.assertEqual(working["sections"][1]["blocks"][1]["source_paragraph_ids"], ["source-1"])
        self.assertIn("结果达到80%", markdown)

    def test_paper_article_style_review_reports_cross_block_issue(self):
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                "issues": [{"block_ids": ["b1", "b2"], "issue": "机制重复", "instruction": "整篇只解释一次，后文改为承接。"}]
            }, ensure_ascii=False)))]
        )
        sections = [{"id": "s1", "title": "结果", "blocks": [{"id": "b1", "text": "第一段。"}, {"id": "b2", "text": "第二段。"}]}]
        result = _paper_style_review(client, "完整五篇范文", sections, "## 结果\\n\\n第一段。\\n\\n第二段。", "test-model", article_level=True, feedback={"style_lint": {"roughly": 1}})
        self.assertEqual(result["issues"][0]["block_ids"], ["b1", "b2"])
        payload = json.loads(client.chat.completions.create.call_args.kwargs["messages"][1]["content"])
        self.assertEqual(payload["article_feedback"]["style_lint"]["roughly"], 1)
        self.assertIn("第一段。", payload["article_body"])

    def test_paper_style_reviewer_accepts_only_known_block_issues(self):
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                "issues": [{
                    "block_id": "beat-1-block-1",
                    "issue": "句式像Results翻译",
                    "instruction": "保留事实，改成更自然的中文叙述。",
                }]
            }, ensure_ascii=False)))]
        )
        sections = [{
            "id": "beat-1",
            "title": "科学结果",
            "blocks": [{"id": "beat-1-block-1", "text": "结果表明变化。"}],
        }]
        result = _paper_style_review(client, "范文", sections, "## 科学结果\n\n结果表明变化。", "test-model")
        self.assertEqual(result["issues"][0]["block_id"], "beat-1-block-1")
        payload = json.loads(client.chat.completions.create.call_args.kwargs["messages"][1]["content"])
        self.assertEqual(payload["sections"][0]["blocks"][0], {"block_id": "beat-1-block-1", "text": "结果表明变化。"})
        self.assertNotIn("evidence_ids", json.dumps(payload, ensure_ascii=False))

    def test_paper_style_reviewer_rejects_unknown_duplicate_and_extra_fields(self):
        sections = [{"id": "beat-1", "title": "结果", "blocks": [{"id": "b1", "text": "正文。"}]}]
        responses = (
            {"issues": [{"block_id": "unknown", "issue": "问题", "instruction": "修改"}]},
            {"issues": [{"block_id": "b1", "issue": "问题", "instruction": "修改"}, {"block_id": "b1", "issue": "重复", "instruction": "修改"}]},
            {"issues": [{"block_id": "b1", "issue": "问题", "instruction": "修改", "text": "越权"}]},
            {"issues": [], "status": "pass"},
        )
        for response in responses:
            with self.subTest(response=response):
                client = MagicMock()
                client.chat.completions.create.return_value = SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(response, ensure_ascii=False)))]
                )
                with self.assertRaises(RuntimeError):
                    _paper_style_review(client, "范文", sections, "## 结果\n\n正文。", "test-model")

    def test_paper_humanizer_target_rewrites_only_selected_blocks(self):
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                "block": {"block_id": "b2", "text": "第二段已校对。"}
            }, ensure_ascii=False)))]
        )
        story_plan = {
            "editorial_brief": {},
            "story_beats": [{
                "id": "beat-1", "title": "结果", "reader_question": "问题？", "core_message": "核心。",
                "evidence_ids": ["e1", "e2"],
            }],
        }
        clean_evidence = [{"evidence_id": "e1", "anchors": []}, {"evidence_id": "e2", "anchors": []}]
        output = _paper_humanize_story(
            client, story_plan, clean_evidence,
            [{"beat_id": "beat-1", "blocks": [
                {"block_id": "b1", "text": "第一段保持不变。"},
                {"block_id": "b2", "text": "第二段原文。"},
            ]}], "test-model",
            block_specs={"beat-1": [
                {"block_id": "b1", "evidence_ids": ["e1"]},
                {"block_id": "b2", "evidence_ids": ["e2"]},
            ]}, target_block_ids={"b2"}, feedback_by_block={"b2": {"style_review": "问题"}},
        )
        self.assertEqual([block["text"] for block in output[0]["blocks"]], ["第一段保持不变。", "第二段已校对。"])
        self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_paper_stop_slop_audit_reports_author_voice(self):
        audit = _paper_stop_slop_audit("## 结果\n\n我们使用了敏感性试验。")
        self.assertIn("author_voice", {issue["type"] for issue in audit["issues"]})

    def test_paper_stop_slop_audit_catches_repeated_paragraph_openings(self):
        markdown = (
            "## 结果\n\n"
            "这一结果说明了空间差异。后续证据支持这一判断。\n\n"
            "这一结果说明了空间差异。另一组证据给出相同方向。"
        )
        audit = _paper_stop_slop_audit(markdown)
        issue_types = {issue["type"] for issue in audit["issues"]}
        self.assertIn("repeated_paragraph_opening", issue_types)

    def test_paper_story_prompts_prioritize_exemplars_without_fixed_arc(self):
        self.assertIn("结论和方法的先后以自然表达为准", PAPER_STORY_PLANNER_PROMPT)
        self.assertIn("style_exemplar中的范文原文是主要写作参考", PAPER_STORY_PLANNER_PROMPT)
        self.assertIn("style corpus自然组织", PAPER_STORY_WRITER_PROMPT)
        self.assertIn("quantitative evidence-anchor", PAPER_PLANNER_PROMPT)
        self.assertIn("仅作上下文", PAPER_PLANNER_PROMPT)
        self.assertIn("style_exemplar中的范文原文是主要写作参考", PAPER_STORY_WRITER_PROMPT)
        self.assertIn("style_exemplar中的范文原文是主要写作参考", PAPER_HUMANIZER_PROMPT)
        self.assertIn("非本领域专家", PAPER_STORY_PLANNER_PROMPT)
        self.assertIn("跨专业科研读者", PAPER_STORY_WRITER_PROMPT)
        self.assertIn("专业词在需要时顺手解释", PAPER_HUMANIZER_PROMPT)
        self.assertNotIn("问题/现象→核心发现→为什么→进一步证据→意义", PAPER_STORY_PLANNER_PROMPT)
        self.assertNotIn("先回答读者问题，再给最重要的发现", PAPER_STORY_WRITER_PROMPT)
        self.assertIn("不要写成论文Results", PAPER_STORY_WRITER_PROMPT)
        self.assertIn("Results转述", PAPER_HUMANIZER_PROMPT)
        for prompt in (PAPER_STORY_WRITER_PROMPT, PAPER_HUMANIZER_PROMPT):
            self.assertIn("第三方科学公众号编辑", prompt)
            self.assertIn("我们展示", prompt)
            self.assertIn("英文原文引用和论文题目保持原样", prompt)

    def test_paper_story_writer_audit_allows_one_retry(self):
        responses = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "sections": [
                                        {
                                            "id": "section-1",
                                            "title": "归因",
                                            "role": "attribution",
                                            "figure_ids": ["Fig. 1"],
                                            "findings": [{"id": "E1", "figure_ids": ["Fig. 1"], "evidence_ids": ["evidence-source-source-0"]}],
                                        }
                                    ]
                                },
                                ensure_ascii=False,
                            )
                        )
                    )
                ]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps({"abstract_cn": "摘要翻译"}, ensure_ascii=False)
                        )
                    )
                ]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=json.dumps({"body": "森林变化解释差异。"}, ensure_ascii=False))
                    )
                ]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps({"status": "pass", "corrections": []}, ensure_ascii=False)
                        )
                    )
                ]
            ),
        ]
        client = MagicMock()
        client.chat.completions.create.side_effect = responses
        story_plan = _story_plan_for_evidence(1)
        story_plan["story_beats"][0]["evidence_ids"] = ["evidence-source-source-0"]
        with tempfile.TemporaryDirectory() as tmp, patch("writer.llm.OpenAI", return_value=client), patch(
            "writer.llm._paper_story_planner", return_value=story_plan
        ), patch(
            "writer.llm._paper_story_writer",
            side_effect=[
                _story_output_for_evidence(1, ["并非A而是B。并非C而是D。"]),
                _story_output_for_evidence(1, ["森林变化解释差异。"]),
            ],
        ) as story_writer, patch(
            "writer.llm._paper_humanize_story", return_value=_story_output_for_evidence(1, ["森林变化解释差异。"])
        ):
            path, _ = generate_article_markdown(
                {
                    "content_type": PAPER_CONTENT,
                    "title": "Test paper",
                    "title_cn": "测试标题",
                    "text": "source",
                    "openalex": {"abstract": "Abstract"},
                    "images": [
                        {"image_role": "figure", "figure_number": 1, "caption": "Figure 1"}
                    ],
                },
                replace(load_settings(), model_base_url="https://model.example/v1", model_api_key="test-key", model_name="test-model"),
                Path(tmp) / "paper",
            )
            final_text = path.read_text(encoding="utf-8")
        self.assertEqual(story_writer.call_count, 2)
        self.assertNotIn("并非A而是B", final_text)

    def test_paper_abstract_translation_is_not_compressed(self):
        faithful = "这是完整的中文摘要翻译。研究结果和限定条件全部保留。" + "重要结果继续保留。" * 20
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"abstract_cn": faithful}, ensure_ascii=False)))]
        )
        settings = replace(load_settings(), model_base_url="https://model.example/v1", model_api_key="test-key", model_name="test-model")
        with patch("writer.llm.OpenAI", return_value=client):
            result = translate_paper_abstract("Original abstract", settings)
        self.assertEqual(result, faithful)
        self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_paper_body_length_and_readability_audits(self):
        markdown = (
            "# 标题\n\n"
            "## 结果\n\n"
            "CMIP6模式间离散度由XGBoost、SHAP和CCA共同分析，R=0.71，结果显示森林变化是主要因素。"
        )
        lengths = _paper_body_length_audit(markdown)
        self.assertEqual(lengths["sections"][0]["title"], "结果")
        self.assertGreater(lengths["total_characters"], 0)
        audit = _paper_readability_audit(markdown)
        self.assertGreater(audit["issue_count"], 0)
        self.assertTrue(any(issue["type"] == "unexplained_acronym" for issue in audit["issues"]))

    def test_paper_clean_story_evidence_hides_backend_figure_labels(self):
        plan = {
            "sections": [
                {
                    "id": "section-1",
                    "role": "attribution",
                    "figure_ids": ["Fig. 3"],
                    "source_paragraph_ids": ["source-0"],
                    "findings": [{"id": "E1", "evidence": "森林变化见Figure 3b", "anchors": ["R = −0.77"]}],
                }
            ]
        }
        evidence, _ = _paper_clean_story_evidence(
            plan,
            [{"id": "source-0", "text": "As shown in Figure 3b, forest trend is negatively correlated."}],
        )
        serialized = json.dumps(evidence, ensure_ascii=False)
        self.assertNotIn("Figure 3", serialized)
        self.assertNotIn("Fig. 3", serialized)
        self.assertNotIn("source-figure", serialized)
        self.assertIn("R = −0.77", serialized)

    def test_paper_story_writer_payload_contains_no_figure_identifiers(self):
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {"sections": [{"id": "beat-1", "title": "森林变化提供线索", "blocks": [{"block_id": "beat-1-block-1", "text": "森林变化与风速差异相关。"}]}]},
                            ensure_ascii=False,
                        )
                    )
                )
            ]
        )
        story_plan = _story_plan_for_evidence(1)
        clean_evidence = [
            {
                "evidence_id": "evidence-1",
                "role": "attribution",
                "core_finding": "森林变化提供线索。",
                "anchors": ["R = −0.77"],
                "source_evidence": ["Forest trend is negatively correlated."],
            }
        ]
        _paper_story_writer(client, story_plan, clean_evidence, "## 发现\n自然节奏。", "test-model")
        payload = json.loads(client.chat.completions.create.call_args.kwargs["messages"][1]["content"])
        serialized = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("Figure 1", serialized)
        self.assertNotIn("Fig. 1", serialized)
        self.assertNotIn("source-figure-1", serialized)
        self.assertNotIn("panel a", serialized)
        self.assertNotIn("abstract", payload)

    def test_paper_plain_language_cleanup_reduces_technical_shorthand(self):
        cleaned = _paper_plain_language_cleanup(
            "ENSO驱动，DJF海温与JJA降水相关；Equation 4和standard deviation用于分析。"
        )
        self.assertNotIn("Equation", cleaned)
        self.assertNotIn("standard deviation", cleaned)
        self.assertIn("厄尔尼诺—拉尼娜现象", cleaned)
        self.assertIn("冬季海温", cleaned)
        self.assertIn("夏季降水", cleaned)

    def test_paper_story_sections_can_merge_related_figures(self):
        section_one = {
            "id": "section-1",
            "role": "attribution",
            "figure_ids": ["Fig. 2"],
            "source_paragraph_ids": ["source-1"],
        }
        section_two = {
            "id": "section-2",
            "role": "attribution",
            "figure_ids": ["Fig. 3"],
            "source_paragraph_ids": ["source-2"],
        }
        evidence_map = {
            "evidence-1": (section_one, {"id": "E1", "figure_ids": ["Fig. 2"], "anchors": ["R = 0.71"]}),
            "evidence-2": (section_two, {"id": "E2", "figure_ids": ["Fig. 3"], "anchors": ["R = −0.77"]}),
        }
        beats = [{
            "id": "beat-1",
            "title": "森林变化提供关键线索",
            "reader_question": "为什么模式会不同？",
            "core_message": "森林变化连接两组结果。",
            "evidence_ids": ["evidence-1", "evidence-2"],
            "transition_to_next": "再看未来影响。",
        }]
        sections = _paper_story_sections(beats, evidence_map)
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0]["figure_ids"], ["Fig. 2", "Fig. 3"])
        self.assertEqual(sections[0]["source_paragraph_ids"], ["source-1", "source-2"])

    def test_paper_title_style_lint_flags_media_headlines(self):
        markdown = "# 标题\n\n## 同一片中国，模式为何不同\n\n正文。\n\n## 森林变化改写了结果\n\n正文。"
        lint = _paper_title_style_lint(markdown)
        self.assertEqual(lint["issue_count"], 3)
        self.assertTrue(all(item["term"] in {"同一片中国", "为何", "改写"} for item in lint["issues"]))

    def test_paper_story_normalizer_ignores_extra_metadata_and_restores_order(self):
        story_plan = {
            "story_beats": [{
                "id": "beat-1",
                "title": "结果",
                "reader_question": "问题？",
                "core_message": "核心。",
                "evidence_ids": ["e1", "e2"],
            }]
        }
        block_specs = {"beat-1": [
            {"block_id": "b1"},
            {"block_id": "b2"},
        ]}
        normalized = _paper_normalize_story_output(
            {"sections": [{
                "id": "beat-1",
                "blocks": [
                    {
                        "block_id": "b2",
                        "text": "第二段。",
                        "evidence_ids": ["rogue"],
                        "figure_ids": ["Fig. 9"],
                        "source_paragraph_ids": ["source-rogue"],
                    },
                    {"block_id": "b1", "text": "第一段。", "anchors": ["80 %"]},
                ],
            }]},
            story_plan,
            [],
            "story writer",
            block_specs,
        )
        self.assertEqual(
            [block["block_id"] for block in normalized[0]["blocks"]], ["b1", "b2"]
        )
        self.assertEqual(
            set(normalized[0]["blocks"][0]), {"block_id", "text"}
        )

    def test_paper_story_normalizer_reports_structural_diagnostics(self):
        story_plan = {
            "story_beats": [{
                "id": "beat-1", "title": "结果", "reader_question": "问题？",
                "core_message": "核心。", "evidence_ids": ["e1", "e2"],
            }]
        }
        block_specs = {"beat-1": [{"block_id": "b1"}, {"block_id": "b2"}]}
        cases = [
            ([{"block_id": "b1", "text": "第一段。"}], "missing=['b2']"),
            ([
                {"block_id": "b1", "text": "第一段。"},
                {"block_id": "foo", "text": "未知。"},
            ], "unknown=['foo']"),
            ([
                {"block_id": "b1", "text": "第一段。"},
                {"block_id": "b1", "text": "重复。"},
            ], "duplicate=['b1']"),
            ([
                {"block_id": "b1", "text": ""},
                {"block_id": "b2", "text": "第二段。"},
            ], "invalid_text=['b1']"),
        ]
        for raw_blocks, expected in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(RuntimeError, re.escape(expected)):
                    _paper_normalize_story_output(
                        {"sections": [{"id": "beat-1", "blocks": raw_blocks}]},
                        story_plan,
                        [],
                        "story writer",
                        block_specs,
                    )

    def test_paper_story_writer_uses_per_block_fallback_after_two_bad_beats(self):
        client = MagicMock()
        bad = {"sections": [{"id": "beat-1", "blocks": [
            {"block_id": "b1", "text": "第一段。"},
        ]}]}
        client.chat.completions.create.side_effect = [
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(bad, ensure_ascii=False)))]),
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(bad, ensure_ascii=False)))]),
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"text": "第一段回退。"}, ensure_ascii=False)))]),
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"text": "第二段回退。"}, ensure_ascii=False)))]),
        ]
        story_plan = {
            "editorial_brief": {"audience": "读者", "purpose": "解释", "tone": "清楚"},
            "story_beats": [{
                "id": "beat-1", "title": "结果", "reader_question": "问题？",
                "core_message": "核心。", "evidence_ids": ["e1", "e2"],
            }],
        }
        clean_evidence = [
            {"evidence_id": "e1", "evidence_group": "group-a", "anchors": []},
            {"evidence_id": "e2", "evidence_group": "group-b", "anchors": []},
        ]
        block_specs = {"beat-1": [
            {"block_id": "b1", "evidence_ids": ("e1",), "figure_ids": (), "source_paragraph_ids": ()},
            {"block_id": "b2", "evidence_ids": ("e2",), "figure_ids": (), "source_paragraph_ids": ()},
        ]}
        output = _paper_story_writer(
            client, story_plan, clean_evidence, "范文", "test-model", block_specs=block_specs
        )
        self.assertEqual(client.chat.completions.create.call_count, 4)
        self.assertEqual(
            [(block["block_id"], block["text"]) for block in output[0]["blocks"]],
            [("b1", "第一段回退。"), ("b2", "第二段回退。")],
        )
        first_payload = json.loads(client.chat.completions.create.call_args_list[0].kwargs["messages"][1]["content"])
        self.assertIn("blocks", first_payload)
        self.assertNotIn("clean_evidence", first_payload)
        fallback_payload = json.loads(client.chat.completions.create.call_args_list[2].kwargs["messages"][1]["content"])
        self.assertIn("block", fallback_payload)
        self.assertNotIn("blocks", fallback_payload)

    def test_paper_story_writer_fallback_still_hard_fails_scientific_anchor(self):
        client = MagicMock()
        bad = {"sections": [{"id": "beat-1", "blocks": []}]}
        fallback = {"text": "没有数字的回退正文。"}
        client.chat.completions.create.side_effect = [
            *[
                SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(bad, ensure_ascii=False)))])
                for _ in range(2)
            ],
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(fallback, ensure_ascii=False)))]),
        ]
        story_plan = {
            "editorial_brief": {"audience": "读者", "purpose": "解释", "tone": "清楚"},
            "story_beats": [{
                "id": "beat-1", "title": "结果", "reader_question": "问题？",
                "core_message": "核心。", "evidence_ids": ["e1"],
            }],
        }
        clean_evidence = [{"evidence_id": "e1", "evidence_group": "context", "anchors": ["80 %"]}]
        block_specs = {"beat-1": [{
            "block_id": "b1", "evidence_ids": ("e1",), "figure_ids": (), "source_paragraph_ids": ("source-1",),
        }]}
        output = _paper_story_writer(
            client, story_plan, clean_evidence, "范文", "test-model", block_specs=block_specs
        )
        registry = [{
            "evidence_id": "e1", "source_paragraph_ids": ["source-1"],
            "source_sentence": "Evidence reports 80 %.", "scope": "section_context",
            "supported_figures": [], "anchors": ["80 %"],
        }]
        plan = {
            "sections": [{
                "id": "beat-1", "title": "结果", "figure_ids": [],
                "source_paragraph_ids": ["source-1"],
                "findings": [{"id": "f1", "evidence_ids": ["e1"]}],
                "story_beat": {"evidence_ids": ["e1"]},
            }],
            "story_evidence": {"e1": {"figure_ids": [], "anchors": ["80 %"]}},
        }
        with self.assertRaisesRegex(RuntimeError, "anchor missing"):
            _paper_apply_story_candidate(
                plan, output, {"e1": (plan["sections"][0], plan["sections"][0]["findings"][0])},
                registry, block_specs, "标题", "摘要", {"source-1"}, None, "", "story writer", rollback_on_failure=False,
            )

    def test_paper_story_writer_pre_splits_different_figure_groups(self):
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                "sections": [{
                    "id": "beat-1",
                    "title": "专业结果",
                    "blocks": [
                        {"block_id": "beat-1-block-1", "text": "重构结果达到R = 0.71。"},
                        {"block_id": "beat-1-block-2", "text": "森林相关结果为R = -0.77。"},
                    ],
                }]
            }, ensure_ascii=False)))]
        )
        clean_evidence = [
            {"evidence_id": "evidence-1", "evidence_group": "evidence_group_A", "anchors": ["R = 0.71"]},
            {"evidence_id": "evidence-2", "evidence_group": "evidence_group_B", "anchors": ["R = -0.77"]},
        ]
        story_plan = _story_plan_for_evidence(1)
        story_plan["story_beats"][0]["evidence_ids"] = ["evidence-1", "evidence-2"]
        output = _paper_story_writer(client, story_plan, clean_evidence, "", "test-model")
        payload = json.loads(client.chat.completions.create.call_args.kwargs["messages"][1]["content"])
        self.assertEqual(
            [block["block_id"] for block in payload["blocks"]],
            ["beat-1-block-1", "beat-1-block-2"],
        )
        self.assertEqual(
            [block["block_id"] for block in output[0]["blocks"]],
            ["beat-1-block-1", "beat-1-block-2"],
        )

    def test_paper_python_restores_evidence_after_writer_text_output(self):
        story_plan = _story_plan_for_evidence(1)
        story_plan["story_beats"][0]["evidence_ids"] = ["evidence-1", "evidence-2"]
        clean_evidence = [
            {"evidence_id": "evidence-1", "evidence_group": "group-a", "anchors": ["R = 0.71"]},
            {"evidence_id": "evidence-2", "evidence_group": "group-b", "anchors": ["R = -0.77"]},
        ]
        sections = [{
            "id": "beat-1",
            "title": "结果",
            "figure_ids": ["Fig. 2", "Fig. 3"],
            "story_beat": {"evidence_ids": ["evidence-1", "evidence-2"]},
        }]
        evidence_map = {
            "evidence-1": (sections[0], {"figure_ids": ["Fig. 2"]}),
            "evidence-2": (sections[0], {"figure_ids": ["Fig. 3"]}),
        }
        specs = _paper_story_block_specs(story_plan, clean_evidence, evidence_map)
        _paper_apply_story_output(
            sections,
            [{
                "id": "beat-1",
                "title": "结果",
                "blocks": [
                    {"block_id": "beat-1-block-1", "text": "R = -0.77。"},
                    {"block_id": "beat-1-block-2", "text": "R = 0.71。"},
                ],
            }],
            evidence_map,
            block_specs=specs,
        )
        self.assertEqual(
            [block["evidence_ids"] for block in sections[0]["blocks"]],
            [["evidence-1"], ["evidence-2"]],
        )
        self.assertEqual(
            [block["figure_ids"] for block in sections[0]["blocks"]],
            [["Fig. 2"], ["Fig. 3"]],
        )

    def test_paper_story_writer_retries_missing_fixed_block_once(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                "sections": [{"id": "beat-1", "blocks": [{"block_id": "wrong", "text": "结果。"}]}]
            }, ensure_ascii=False)))]),
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                "sections": [{"id": "beat-1", "blocks": [{"block_id": "beat-1-block-1", "text": "结果。"}]}]
            }, ensure_ascii=False)))]),
        ]
        story_plan = _story_plan_for_evidence(1)
        clean_evidence = [{"evidence_id": "evidence-1", "evidence_group": "context", "anchors": []}]
        output = _paper_story_writer(client, story_plan, clean_evidence, "", "test-model")
        self.assertEqual(client.chat.completions.create.call_count, 2)
        self.assertEqual(output[0]["blocks"][0]["block_id"], "beat-1-block-1")

    def test_paper_story_block_validator_rejects_anchor_in_wrong_figure_block(self):
        bundles = [
            {"figure_id": "Fig. 2", "quantitative_anchors": ["R = 0.71"]},
            {"figure_id": "Fig. 3", "quantitative_anchors": ["R = -0.77"]},
        ]
        plan = {
            "sections": [{
                "id": "beat-1",
                "title": "专业结果",
                "figure_ids": ["Fig. 2", "Fig. 3"],
                "source_paragraph_ids": ["source-0"],
                "findings": [
                    {"id": "E1", "figure_ids": ["Fig. 2"], "evidence": "重构", "anchors": ["R = 0.71"]},
                    {"id": "E2", "figure_ids": ["Fig. 3"], "evidence": "森林", "anchors": ["R = -0.77"]},
                ],
                "story_beat": {"evidence_ids": ["evidence-1", "evidence-2"]},
                "story_evidence": {},
                "blocks": [
                    {"id": "block-a", "evidence_ids": ["evidence-1"], "figure_ids": ["Fig. 3"], "text": "R = 0.71。"},
                    {"id": "block-b", "evidence_ids": ["evidence-2"], "figure_ids": ["Fig. 2"], "text": "R = -0.77。"},
                ],
            }],
            "story_evidence": {
                "evidence-1": {"figure_ids": ["Fig. 2"], "anchors": ["R = 0.71"]},
                "evidence-2": {"figure_ids": ["Fig. 3"], "anchors": ["R = -0.77"]},
            },
        }
        markdown = "# 标题\n\n## 专业结果\n\nR = 0.71。\n\nR = -0.77。"
        with self.assertRaisesRegex(RuntimeError, "PAPER evidence block figure mismatch"):
            _validate_paper_evidence_plan(plan, markdown, {"source-0"}, bundles)

    def test_paper_story_block_validator_rejects_negative_anchor_in_wrong_figure_block(self):
        bundles = [
            {"figure_id": "Fig. 2", "quantitative_anchors": []},
            {"figure_id": "Fig. 3", "quantitative_anchors": ["R = -0.77"]},
        ]
        plan = {
            "sections": [{
                "id": "beat-1",
                "title": "专业结果",
                "figure_ids": ["Fig. 2", "Fig. 3"],
                "source_paragraph_ids": ["source-0"],
                "findings": [{"id": "E1", "figure_ids": ["Fig. 3"], "evidence": "森林", "anchors": ["R = -0.77"]}],
                "story_beat": {"evidence_ids": ["evidence-1"]},
                "blocks": [{"id": "block-a", "evidence_ids": ["evidence-1"], "figure_ids": ["Fig. 2"], "text": "R = -0.77。"}],
            }],
            "story_evidence": {"evidence-1": {"figure_ids": ["Fig. 3"], "anchors": ["R = -0.77"]}},
        }
        markdown = "# 标题\n\n## 专业结果\n\nR = -0.77。"
        with self.assertRaisesRegex(RuntimeError, "PAPER evidence block figure mismatch"):
            _validate_paper_evidence_plan(plan, markdown, {"source-0"}, bundles)

    def test_paper_humanizer_payload_is_block_scoped_without_abstract(self):
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                "block": {"block_id": "beat-1-block-1", "text": "R = 0.71。"}
            }, ensure_ascii=False)))]
        )
        story_plan = _story_plan_for_evidence(1)
        clean_evidence = [{"evidence_id": "evidence-1", "evidence_group": "evidence_group_A", "anchors": ["R = 0.71"]}]
        from writer.llm import _paper_humanize_story
        package = _paper_style_exemplar()
        _paper_humanize_story(
            client,
            story_plan,
            clean_evidence,
            [{"beat_id": "beat-1", "blocks": [{"block_id": "beat-1-block-1", "text": "R = 0.71。"}]}],
            "test-model",
            style_exemplar=package,
        )
        payload = json.loads(client.chat.completions.create.call_args.kwargs["messages"][1]["content"])
        self.assertIn("current_block", payload)
        self.assertIn("clean_evidence", payload)
        self.assertEqual(payload["style_exemplar"], package)
        self.assertNotIn("draft_blocks_by_beat", payload)
        self.assertNotIn("abstract", payload)
        self.assertNotIn("beat-2", json.dumps(payload, ensure_ascii=False))

    def test_paper_plain_language_cleanup_splits_semicolons_and_removes_abstract_term(self):
        cleaned = _paper_plain_language_cleanup("结果为R = 0.71；陆地—大气通量发生变化。")
        self.assertNotIn("；", cleaned)
        self.assertNotIn("陆地—大气通量", cleaned)
        self.assertIn("陆面与大气之间的交换", cleaned)
        self.assertNotIn("approximately", _paper_plain_language_cleanup("约approximately 74%的差异。"))

    def test_paper_stop_slop_audit_catches_numbered_figure_reportage(self):
        markdown = (
            "# 标题\n\n"
            "## 第一部分\n\n首先，图1显示变化。\n\n"
            "## 第二部分\n\n其次，图2显示差异。\n\n"
            "## 第三部分\n\n最后，图3显示未来。"
        )
        audit = _paper_stop_slop_audit(markdown)
        issue_types = {issue["type"] for issue in audit["issues"]}
        self.assertIn("figure_reportage", issue_types)
        self.assertIn("numbered_structure", issue_types)
        self.assertGreater(audit["metrics"]["template_risk"], 0)

    def test_paper_humanizer_prompt_preserves_fidelity_contract(self):
        self.assertIn("数字及其修饰对象", PAPER_HUMANIZER_PROMPT)
        self.assertIn("correlation不能写成causation", PAPER_HUMANIZER_PROMPT)
        self.assertIn("不新增事实", PAPER_HUMANIZER_PROMPT)

    def test_paper_popular_science_editor_prompt_prioritizes_plain_language(self):
        self.assertIn("Popular Science Editor", PAPER_POPULAR_SCIENCE_EDITOR_PROMPT)
        self.assertIn("普通中文", PAPER_POPULAR_SCIENCE_EDITOR_PROMPT)
        self.assertIn("直接的科学陈述", PAPER_POPULAR_SCIENCE_EDITOR_PROMPT)
        self.assertIn("350到500个中文字符", PAPER_POPULAR_SCIENCE_EDITOR_PROMPT)
        self.assertIn("Story Planner", PAPER_STORY_PLANNER_PROMPT)
        self.assertIn("Story Writer", PAPER_STORY_WRITER_PROMPT)
        self.assertIn("按证据需要保持紧凑", PAPER_STORY_WRITER_PROMPT)
        self.assertNotIn("每个beat固定", PAPER_STORY_WRITER_PROMPT)
        self.assertIn("第三方科学公众号编辑", PAPER_POPULAR_SCIENCE_EDITOR_PROMPT)
        self.assertIn("只读", PAPER_STYLE_REVIEWER_PROMPT)
        self.assertIn("style_exemplar", PAPER_STYLE_REVIEWER_PROMPT)
        self.assertIn("不能移动证据", PAPER_STYLE_REVIEWER_PROMPT)

    def test_paper_popular_editor_anchor_audit_preserves_section_mapping(self):
        plan = {
            "sections": [
                {
                    "id": "section-1",
                    "title": "归因",
                    "findings": [{"anchors": ["R = 0.71"]}],
                },
                {
                    "id": "section-2",
                    "title": "投影",
                    "findings": [{"anchors": ["74%"]}],
                },
            ]
        }
        before = "# 标题\n\n## 归因\n\n相关达到R = 0.71。\n\n## 投影\n\n森林变化解释74%。"
        after = "# 标题\n\n## 归因\n\n相关达到R=0.71。\n\n## 投影\n\n森林变化解释约74%。"
        result = _paper_editor_anchor_audit(before, after, plan)
        self.assertEqual(result["issue_count"], 0)

    def test_paper_scientific_review_triggers_one_structured_revision(self):
        plan = {
            "sections": [
                {
                    "id": "section-1",
                    "title": "归因",
                    "role": "attribution",
                    "source_paragraph_ids": ["source-0"],
                    "findings": [{"id": "E1", "evidence": "74%", "anchors": ["74%"]}],
                }
            ]
        }
        review_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "status": "needs_revision",
                                "corrections": [
                                    {
                                        "section_id": "section-1",
                                        "evidence": "74%",
                                        "issue": "数字缺少归属",
                                        "instruction": "明确74%是模式间差异的归因比例",
                                    }
                                ],
                            },
                            ensure_ascii=False,
                        )
                    )
                )
            ]
        )
        revision_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {"sections": [{"id": "section-1", "body": "森林变化约解释74%的模式差异。"}]},
                            ensure_ascii=False,
                        )
                    )
                )
            ]
        )
        client = MagicMock()
        client.chat.completions.create.side_effect = [review_response, revision_response]
        reviewed = _paper_review(
            client,
            "Abstract",
            "paper text",
            [{"id": "source-0", "text": "source"}],
            plan,
            "## 归因\n\n初稿。",
            "test-model",
        )
        bodies = _paper_revision(
            client,
            "Abstract",
            "paper text",
            [{"id": "source-0", "text": "source"}],
            plan,
            "## 归因\n\n初稿。",
            reviewed["corrections"],
            "test-model",
        )
        self.assertEqual(reviewed["status"], "needs_revision")
        self.assertEqual(bodies, ["森林变化约解释74%的模式差异。"])
        self.assertEqual(client.chat.completions.create.call_count, 2)

    def test_paper_scientific_review_allows_three_bounded_cycles(self):
        planner = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "sections": [
                                    {
                                        "id": "section-1",
                                        "title": "归因",
                                        "role": "attribution",
                                        "source_paragraph_ids": ["source-0"],
                                        "findings": [{"id": "E1", "evidence": "结果", "anchors": []}],
                                    }
                                ]
                            },
                            ensure_ascii=False,
                        )
                    )
                )
            ]
        )
        abstract = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"abstract_cn": "摘要翻译"}, ensure_ascii=False)))]
        )
        writer = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"body": "初稿正文。"}, ensure_ascii=False)))]
        )
        settings = replace(load_settings(), model_base_url="https://model.example/v1", model_api_key="test-key", model_name="test-model")
        client = MagicMock()
        client.chat.completions.create.side_effect = [planner, abstract, writer]
        reviews = [
            {"status": "needs_revision", "corrections": [{"section": "section-1", "evidence": "结果", "issue": "问题一", "correction": "修正一"}]},
            {"status": "needs_revision", "corrections": [{"section": "section-1", "evidence": "结果", "issue": "问题二", "correction": "修正二"}]},
            {"status": "needs_revision", "corrections": [{"section": "section-1", "evidence": "结果", "issue": "问题三", "correction": "修正三"}]},
        ]
        revisions = [["第一次修正。"], ["第二次修正。"]]
        with tempfile.TemporaryDirectory() as tmp, patch("writer.llm.OpenAI", return_value=client), patch(
            "writer.llm._paper_review", side_effect=reviews
        ) as review, patch("writer.llm._paper_revision", side_effect=revisions) as revision, patch(
            "writer.llm._paper_story_planner", return_value=_story_plan_for_evidence(1)
        ), patch(
            "writer.llm._paper_story_writer", return_value=_story_output_for_evidence(1, ["最终正文。"])
        ), patch("writer.llm._paper_editorial_rewrite", return_value=["最终正文。"]), patch(
            "writer.llm._paper_humanize_story", return_value=_story_output_for_evidence(1, ["最终正文。"])
        ):
            path, metadata_path = generate_article_markdown(
                {
                    "content_type": PAPER_CONTENT,
                    "title": "Test paper",
                    "title_cn": "测试标题",
                    "text": "source",
                    "openalex": {"abstract": "Abstract"},
                    "images": [],
                },
                settings,
                Path(tmp) / "paper",
            )
            final_text = path.read_text(encoding="utf-8")
            paper_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(review.call_count, 3)
        self.assertEqual(revision.call_count, 2)
        self.assertIn("最终正文。", final_text)
        self.assertEqual(
            paper_metadata["paper_evidence_plan"]["scientific_review"]["status"],
            "unresolved_after_max_cycles",
        )
        self.assertEqual(paper_metadata["paper_evidence_plan"]["scientific_review"]["cycles"], 3)

    def test_paper_scientific_review_passes_on_second_cycle(self):
        settings = replace(
            load_settings(),
            model_base_url="https://model.example/v1",
            model_api_key="test-key",
            model_name="test-model",
        )
        planner = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "sections": [
                                    {
                                        "id": "section-1",
                                        "title": "归因",
                                        "role": "attribution",
                                        "source_paragraph_ids": ["source-0"],
                                        "findings": [{"id": "E1", "evidence": "结果", "anchors": []}],
                                    }
                                ]
                            },
                            ensure_ascii=False,
                        )
                    )
                )
            ]
        )
        abstract = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"abstract_cn": "摘要"}, ensure_ascii=False)))]
        )
        writer = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"body": "初稿正文。"}, ensure_ascii=False)))]
        )
        client = MagicMock()
        client.chat.completions.create.side_effect = [planner, abstract, writer]
        reviews = [
            {"status": "needs_revision", "corrections": [{"section": "section-1", "evidence": "结果", "issue": "问题", "correction": "修正"}]},
            {"status": "pass", "corrections": []},
        ]
        with tempfile.TemporaryDirectory() as tmp, patch("writer.llm.OpenAI", return_value=client), patch(
            "writer.llm._paper_review", side_effect=reviews
        ) as review, patch("writer.llm._paper_revision", return_value=["修订正文。"]), patch(
            "writer.llm._paper_story_planner", return_value=_story_plan_for_evidence(1)
        ), patch(
            "writer.llm._paper_story_writer", return_value=_story_output_for_evidence(1, ["最终正文。"])
        ), patch("writer.llm._paper_editorial_rewrite", return_value=["最终正文。"]) as editor, patch(
            "writer.llm._paper_humanize_story", return_value=_story_output_for_evidence(1, ["最终正文。"])
        ):
            _, metadata_path = generate_article_markdown(
                {
                    "content_type": PAPER_CONTENT,
                    "title": "Test paper",
                    "title_cn": "测试标题",
                    "text": "source",
                    "openalex": {"abstract": "Abstract"},
                    "images": [],
                },
                settings,
                Path(tmp) / "paper",
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(review.call_count, 2)
        self.assertEqual(editor.call_count, 0)
        self.assertEqual(metadata["paper_evidence_plan"]["scientific_review"]["status"], "pass")
        self.assertEqual(metadata["paper_evidence_plan"]["scientific_review"]["cycles"], 2)

    def test_paper_revision_deterministic_anchor_mismatch_still_fails(self):
        settings = replace(
            load_settings(),
            model_base_url="https://model.example/v1",
            model_api_key="test-key",
            model_name="test-model",
        )
        planner = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "sections": [
                                    {
                                        "id": "section-1",
                                        "title": "归因",
                                        "role": "attribution",
                                        "figure_ids": ["Fig. 2"],
                                        "source_paragraph_ids": ["source-0"],
                                        "findings": [{"id": "E1", "figure_ids": ["Fig. 2"], "evidence": "相关", "anchors": ["R = 0.71"]}],
                                    }
                                ]
                            },
                            ensure_ascii=False,
                        )
                    )
                )
            ]
        )
        abstract = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"abstract_cn": "摘要"}, ensure_ascii=False)))]
        )
        writer = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"body": "相关结果为R = 0.71。"}, ensure_ascii=False)))]
        )
        client = MagicMock()
        client.chat.completions.create.side_effect = [planner, abstract, writer]
        with tempfile.TemporaryDirectory() as tmp, patch("writer.llm.OpenAI", return_value=client), patch(
            "writer.llm._paper_review",
            return_value={
                "status": "needs_revision",
                "corrections": [{"section": "section-1", "evidence": "R = 0.71", "issue": "问题", "correction": "修正"}],
            },
        ), patch("writer.llm._paper_revision", return_value=["修订后未保留数字。"]):
            with self.assertRaisesRegex(RuntimeError, "PAPER evidence anchor missing"):
                generate_article_markdown(
                    {
                        "content_type": PAPER_CONTENT,
                        "title": "Test paper",
                        "title_cn": "测试标题",
                        "text": "source",
                        "openalex": {"abstract": "Abstract"},
                        "images": [{"figure_number": 2, "caption": "Figure 2. Correlation R = 0.71."}],
                        "paper_selected_body_images": [{"figure_number": 2, "caption": "Figure 2. Correlation R = 0.71."}],
                    },
                    settings,
                    Path(tmp) / "paper",
                )

    def test_paper_popular_science_warning_does_not_fail_generation(self):
        settings = replace(
            load_settings(),
            model_base_url="https://model.example/v1",
            model_api_key="test-key",
            model_name="test-model",
        )
        planner = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "sections": [
                                    {
                                        "id": "section-1",
                                        "title": "归因",
                                        "role": "attribution",
                                        "source_paragraph_ids": ["source-0"],
                                        "findings": [{"id": "E1", "evidence": "结果", "anchors": []}],
                                    }
                                ]
                            },
                            ensure_ascii=False,
                        )
                    )
                )
            ]
        )
        abstract = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"abstract_cn": "摘要"}, ensure_ascii=False)))]
        )
        writer = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"body": "初稿正文。"}, ensure_ascii=False)))]
        )
        client = MagicMock()
        client.chat.completions.create.side_effect = [planner, abstract, writer]
        unreadable = "模式间离散度显示显著性水平与典型相关。"
        with tempfile.TemporaryDirectory() as tmp, patch("writer.llm.OpenAI", return_value=client), patch(
            "writer.llm._paper_review", return_value={"status": "pass", "corrections": []}
        ), patch("writer.llm._paper_story_planner", return_value=_story_plan_for_evidence(1)), patch(
            "writer.llm._paper_story_writer",
            side_effect=[_story_output_for_evidence(1, [unreadable]), _story_output_for_evidence(1, [unreadable])],
        ) as story_writer, patch(
            "writer.llm._paper_humanize_story",
            side_effect=[
                _story_output_for_evidence(1, [unreadable]),
                _story_output_for_evidence(1, [unreadable]),
                _story_output_for_evidence(1, [unreadable]),
            ],
        ):
            _, metadata_path = generate_article_markdown(
                {
                    "content_type": PAPER_CONTENT,
                    "title": "Test paper",
                    "title_cn": "测试标题",
                    "text": "source",
                    "openalex": {"abstract": "Abstract"},
                    "images": [],
                },
                settings,
                Path(tmp) / "paper",
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        audit = metadata["paper_evidence_plan"]["popular_science_audit"]
        self.assertEqual(story_writer.call_count, 2)
        self.assertEqual(audit["status"], "warning")
        self.assertEqual(audit["retry_count"], 1)
        self.assertTrue(audit["unresolved_issues"]["feedback"]["readability"]["issue_count"])

    def test_paper_popular_science_anchor_break_still_fails(self):
        settings = replace(
            load_settings(),
            model_base_url="https://model.example/v1",
            model_api_key="test-key",
            model_name="test-model",
        )
        planner = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "sections": [
                                    {
                                        "id": "section-1",
                                        "title": "归因",
                                        "role": "attribution",
                                        "figure_ids": ["Fig. 2"],
                                        "source_paragraph_ids": ["source-0"],
                                        "findings": [{"id": "E1", "figure_ids": ["Fig. 2"], "evidence": "相关", "anchors": ["R = 0.71"]}],
                                    }
                                ]
                            },
                            ensure_ascii=False,
                        )
                    )
                )
            ]
        )
        abstract = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"abstract_cn": "摘要"}, ensure_ascii=False)))]
        )
        writer = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"body": "相关结果为R = 0.71。"}, ensure_ascii=False)))]
        )
        client = MagicMock()
        client.chat.completions.create.side_effect = [planner, abstract, writer]
        with tempfile.TemporaryDirectory() as tmp, patch("writer.llm.OpenAI", return_value=client), patch(
            "writer.llm._paper_review", return_value={"status": "pass", "corrections": []}
        ), patch("writer.llm._paper_story_planner", return_value=_story_plan_for_evidence(1)), patch(
            "writer.llm._paper_story_writer", return_value=_story_output_for_evidence(1, ["相关结果为R = 0.71。"])
        ), patch(
            "writer.llm._paper_editorial_rewrite", side_effect=[["编辑后丢失数字。"], ["重试后仍丢失数字。"]]
        ), patch(
            "writer.llm._paper_humanize_story", return_value=_story_output_for_evidence(1, ["重试后仍丢失数字。"])
        ):
            with self.assertRaisesRegex(RuntimeError, "PAPER story block validation failed"):
                generate_article_markdown(
                    {
                        "content_type": PAPER_CONTENT,
                        "title": "Test paper",
                        "title_cn": "测试标题",
                        "text": "source",
                        "openalex": {"abstract": "Abstract"},
                        "images": [{"figure_number": 2, "caption": "Figure 2. Correlation R = 0.71."}],
                        "paper_selected_body_images": [{"figure_number": 2, "caption": "Figure 2. Correlation R = 0.71."}],
                    },
                    settings,
                    Path(tmp) / "paper",
                )

    def test_paper_staged_pipeline_isolated_and_reviewed(self):
        settings = replace(
            load_settings(),
            model_base_url="https://model.example/v1",
            model_api_key="test-key",
            model_name="test-model",
        )
        planner = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "sections": [
                                    {
                                        "id": "section-1",
                                        "title": "历史模式差异",
                                        "role": "phenomenon",
                                        "source_paragraph_ids": ["source-figure-1"],
                                        "findings": [{"id": "E1", "evidence": "历史差异", "anchors": []}],
                                    },
                                    {
                                        "id": "section-2",
                                        "title": "森林归因",
                                        "role": "attribution",
                                        "source_paragraph_ids": ["source-figure-2"],
                                        "findings": [{"id": "E2", "evidence": "74%归因", "anchors": ["74%"]}],
                                    },
                                    {
                                        "id": "section-3",
                                        "title": "未来投影",
                                        "role": "projection",
                                        "source_paragraph_ids": ["source-figure-3"],
                                        "findings": [{"id": "E3", "evidence": "未来情景", "anchors": ["SSP3-7.0"]}],
                                    },
                                ]
                            },
                            ensure_ascii=False,
                        )
                    )
                )
            ]
        )
        section_responses = [
            {"body": "历史模式差异位于中国北部。"},
            {"body": "森林覆盖变化约解释74%的模式差异。"},
            {"body": "SSP3-7.0（未来排放情景）下的未来投影仍存在不确定性。"},
        ]
        responses = [planner]
        responses.extend(
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(item, ensure_ascii=False)))])
            for item in section_responses
        )
        responses.insert(
            1,
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps({"abstract_cn": "忠实的中文摘要翻译"}, ensure_ascii=False)
                        )
                    )
                ]
            ),
        )
        responses.extend(
            [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content=json.dumps({"status": "pass", "corrections": []}, ensure_ascii=False)
                            )
                        )
                    ]
                ),
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content=json.dumps(
                                    {
                                        "sections": [
                                            {"id": "beat-1", "body": "历史模式差异位于中国北部。"},
                                            {"id": "beat-2", "body": "森林覆盖变化约解释74%的模式差异。"},
                                            {"id": "beat-3", "body": "SSP3-7.0（未来排放情景）下的未来投影仍存在不确定性。"},
                                        ]
                                    },
                                    ensure_ascii=False,
                                )
                            )
                        )
                    ]
                ),
            ]
        )
        news_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="# 测试新闻\n\n新闻正文。"))]
        )
        client = MagicMock()
        client.chat.completions.create.side_effect = responses
        with tempfile.TemporaryDirectory() as tmp, patch("writer.llm.OpenAI", return_value=client), patch(
            "writer.llm._paper_story_planner", return_value=_story_plan_for_evidence(3)
        ), patch(
            "writer.llm._paper_story_writer",
            return_value=_story_output_for_evidence(
                3,
                [
                    "历史模式差异位于中国北部。",
                    "森林覆盖变化约解释74%的模式差异。",
                    "SSP3-7.0（未来排放情景）下的未来投影仍存在不确定性。",
                ],
            ),
        ), patch(
            "writer.llm._paper_humanize_story",
            return_value=_story_output_for_evidence(
                3,
                [
                    "历史模式差异位于中国北部。",
                    "森林覆盖变化约解释74%的模式差异。",
                    "SSP3-7.0（未来排放情景）下的未来投影仍存在不确定性。",
                ],
            ),
        ):
            root = Path(tmp)
            paper_path, paper_metadata_path = generate_article_markdown(
                {
                    "content_type": PAPER_CONTENT,
                    "title": "Test paper",
                    "title_cn": "测试标题",
                    "text": "historical source\nattribution source\nprojection source",
                    "openalex": {"abstract": "Paper abstract"},
                    "images": [
                        {"image_role": "figure", "figure_number": 1, "caption": "Figure 1"},
                        {"image_role": "figure", "figure_number": 2, "caption": "Figure 2"},
                        {"image_role": "figure", "figure_number": 3, "caption": "Figure 3"},
                    ],
                },
                settings,
                root / "paper",
            )
            calls = client.chat.completions.create.call_args_list
            self.assertEqual(len(calls), 6)
            self.assertIn("Scientific Planner", calls[0].kwargs["messages"][0]["content"])
            abstract_prompt = calls[1].kwargs["messages"][0]["content"]
            self.assertIn("完整保留原文的重要背景", abstract_prompt)
            self.assertNotIn("80到120个汉字", abstract_prompt)
            reviewer_prompt = calls[5].kwargs["messages"][0]["content"]
            section_payloads = [json.loads(calls[index].kwargs["messages"][1]["content"]) for index in (2, 3, 4)]
            self.assertEqual([payload["section"]["id"] for payload in section_payloads], ["section-1", "section-2", "section-3"])
            self.assertNotIn("74%归因", json.dumps(section_payloads[0], ensure_ascii=False))
            self.assertNotIn("SSP3-7.0", json.dumps(section_payloads[1], ensure_ascii=False))
            paper_markdown = paper_path.read_text(encoding="utf-8")
            paper_metadata = json.loads(paper_metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(
                [section["role"] for section in paper_metadata["paper_evidence_plan"]["sections"]],
                ["phenomenon", "attribution", "projection"],
            )
            self.assertNotIn("PAPER_EVIDENCE_PLAN", paper_markdown)
            self.assertLess(paper_markdown.index("历史模式差异位于中国北部"), paper_markdown.index("森林覆盖变化约解释74%"))
            self.assertLess(paper_markdown.index("森林覆盖变化约解释74%"), paper_markdown.index("SSP3-7.0"))
            client.chat.completions.create.reset_mock()
            client.chat.completions.create.side_effect = [news_response]
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
            news_call = client.chat.completions.create.call_args
            news_prompt = news_call.kwargs["messages"][0]["content"]
            news_input = json.loads(news_call.kwargs["messages"][1]["content"])

        self.assertIn("只返回严格JSON对象", calls[0].kwargs["messages"][0]["content"])
        self.assertIn("只审核科学准确性和section结构", reviewer_prompt)
        self.assertIn('"section":"section-1"', reviewer_prompt)
        self.assertIn('"evidence":"..."', reviewer_prompt)
        self.assertIn('"correction":"..."', reviewer_prompt)
        self.assertIn("不要因句式、节奏、中文措辞", reviewer_prompt)
        self.assertIn("Story Writer", PAPER_STORY_WRITER_PROMPT)
        self.assertIn("问题—发现—为什么—意义/未来", PAPER_STORY_PLANNER_PROMPT)
        self.assertIn("中文母语科学编辑", PAPER_HUMANIZER_PROMPT)
        self.assertIn("约1000到2000中文字", news_prompt)
        self.assertNotIn("abstract", news_input)
        self.assertEqual(news_input["news_summary"], "News summary")

    def test_paper_figures_follow_their_evidence_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "images"
            images_dir.mkdir()
            figure_two = images_dir / "figure-02.png"
            figure_three = images_dir / "figure-03.png"
            figure_two.write_bytes(b"png")
            figure_three.write_bytes(b"png")
            markdown = root / "article.md"
            markdown.write_text(
                "## 植被变化\n\n机器学习重构得到R = 0.71。\n\n森林相关结果为R = -0.77。\n",
                encoding="utf-8",
            )
            images = [
                {"local_path": str(figure_two), "figure_number": 2, "caption": "Figure 2", "image_role": "figure"},
                {"local_path": str(figure_three), "figure_number": 3, "caption": "Figure 3", "image_role": "figure"},
            ]
            dossier = {
                "content_type": PAPER_CONTENT,
                "paper_evidence_plan": {
                    "sections": [{
                        "title": "植被变化",
                        "blocks": [
                            {"figure_ids": ["Fig. 2"], "text": "机器学习重构得到R = 0.71。"},
                            {"figure_ids": ["Fig. 3"], "text": "森林相关结果为R = -0.77。"},
                        ],
                    }],
                },
            }
            _insert_paper_figures(markdown, images, ["图2说明。", "图3说明。"], dossier)
            text = markdown.read_text(encoding="utf-8")
            self.assertLess(text.index("R = 0.71"), text.index("![Fig. 2]"))
            self.assertLess(text.index("![Fig. 2]"), text.index("R = -0.77"))
            self.assertLess(text.index("R = -0.77"), text.index("![Fig. 3]"))

    def test_paper_merged_paragraph_inserts_all_python_bound_figures_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "images"
            images_dir.mkdir()
            figure_one = images_dir / "figure-01.png"
            figure_two = images_dir / "figure-02.png"
            figure_one.write_bytes(b"png")
            figure_two.write_bytes(b"png")
            markdown = root / "article.md"
            markdown.write_text(
                "## 结果\n\n合并后的自然段同时解释约68%的区域和约49%的来源贡献。\n",
                encoding="utf-8",
            )
            images = [
                {"local_path": str(figure_one), "figure_number": 1, "caption": "Figure 1", "image_role": "figure"},
                {"local_path": str(figure_two), "figure_number": 2, "caption": "Figure 2", "image_role": "figure"},
            ]
            dossier = {
                "content_type": PAPER_CONTENT,
                "paper_evidence_plan": {
                    "sections": [{
                        "title": "结果",
                        "blocks": [
                            {"id": "b1", "figure_ids": ["Fig. 1"]},
                            {"id": "b2", "figure_ids": ["Fig. 2"]},
                        ],
                        "paragraphs": [{
                            "block_ids": ["b1", "b2"],
                            "text": "合并后的自然段同时解释约68%的区域和约49%的来源贡献。",
                        }],
                    }],
                },
            }
            _insert_paper_figures(markdown, images, ["图1说明。", "图2说明。"], dossier)
            text = markdown.read_text(encoding="utf-8")
            paragraph_index = text.index("合并后的自然段")
            self.assertLess(paragraph_index, text.index("![Fig. 1]"))
            self.assertLess(text.index("![Fig. 1]"), text.index("![Fig. 2]"))
            self.assertEqual(text.count("![Fig. 1]"), 1)
            self.assertEqual(text.count("![Fig. 2]"), 1)

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

                async def fake_details(_rank, _date, _content_type=None, **_kwargs):
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

    def test_paper_html_figure_download_enters_body_images(self):
        image_buffer = io.BytesIO()
        Image.new("RGB", (312, 116), color="white").save(image_buffer, format="PNG")
        content = image_buffer.getvalue()

        class FakeResponse:
            status_code = 200

            def __init__(self, payload):
                self.content = payload

            def raise_for_status(self):
                return None

        class FakeClient:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def get(self, _url, **_kwargs):
                return FakeResponse(content)

        with tempfile.TemporaryDirectory() as tmp:
            with patch("news.extract.httpx.Client", return_value=FakeClient()):
                downloaded = download_images(
                    [{
                        "url": "https://example.test/Fig1_HTML.png",
                        "local_path": "",
                        "caption": "Fig. 1: Hadley circulation.",
                        "metadata_title": "Fig. 1: Hadley circulation.",
                        "image_source": "html_figure",
                        "image_role": "figure",
                        "figure_number": 1,
                    }],
                    tmp,
                )

            local_path = Path(downloaded[0]["local_path"])
            self.assertTrue(local_path.is_file())
            _, body_images, _ = _select_article_images(
                downloaded,
                PAPER_CONTENT,
                "## Hadley circulation\\n\\nFigure 1 shows the Hadley circulation.",
            )
            self.assertEqual([image["figure_number"] for image in body_images], [1])

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
                    },
                    "images": [
                        {
                            "url": "https://example.test/html-figure.png",
                            "local_path": "",
                            "caption": "Complete Figure 1",
                            "image_source": "html_figure",
                            "image_role": "figure",
                        }
                    ],
                }

                async def fake_details(_rank, _date, _content_type=None, **_kwargs):
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
                        "image_source": "pdf_figure",
                        "image_role": "figure",
                        "figure_number": 1,
                    }
                    return [figure], {"matched_figures": 1, "figures": [figure]}

                pipeline.paper_details = fake_details
                with (
                    patch("news.pipeline.article_output_dir", return_value=root / "article"),
                    patch("news.pipeline.download_images", side_effect=fake_download),
                    patch(
                        "news.pipeline.discover_pdf_source",
                        return_value={
                            "pdf_url": "https://example.test/paper.pdf",
                            "landing_url": "https://example.test/paper",
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
                },
                {
                    "local_path": str(figure_two),
                    "url": "https://example.test/figure-2.png",
                    "image_role": "figure",
                    "figure_number": 2,
                    "caption": "Fig. 2 | Atmospheric circulation and surface wind.",
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

    def test_paper_figure_reference_parser_excludes_supplementary_figures(self):
        cases = {
            "Fig. 4": {4},
            "Figure 3": {3},
            "Fig. 4 and Fig. 5": {4, 5},
            "Fig. 4 and Supplementary Fig. 7": {4},
            "Supplementary Fig. 7": set(),
            "Supplementary Figs. 2–4": set(),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(_paper_figure_reference_numbers(text), expected)

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
                "paper_first_page": {"local_path": str(first_page)},
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
            self.assertEqual(_selected_cover_path(markdown), first_page)
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

    def test_paper_first_page_is_selected_as_default_cover(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            images.mkdir()
            first_page = images / "paper-first-page.png"
            first_page.write_bytes(b"png")
            figure = images / "figure-03.png"
            figure.write_bytes(b"png")
            markdown = root / "article.md"
            markdown.write_text("正文\n", encoding="utf-8")
            (root / "metadata.json").write_text(
                json.dumps(
                    {
                        "content_type": PAPER_CONTENT,
                        "paper_first_page": {"local_path": str(first_page)},
                        "body_images": [{"local_path": str(figure)}],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            self.assertEqual(_selected_cover_path(markdown), first_page)

            metadata_path = root / "metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "content_type": PAPER_CONTENT,
                        "body_images": [{"local_path": str(figure)}],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            self.assertEqual(_selected_cover_path(markdown), DEFAULT_COVER)

    def test_paper_final_html_styles_intro_and_hides_figure_source(self):
        html = (
            '<p><img alt="论文第一页" src="images/paper-first-page.png" /></p>'
            '<p>摘要导语内容。</p>'
            '<h2>第一节</h2><p>第一节正文不能进入摘要框。</p>'
            '<section data-role="img-wrapper"><p>Fig. 2 | 图注内容。</em><br />'
            '<em>图源：作者，CC BY-NC-ND</p></section>'
        )
        styled = _style_paper_intro(html)
        self.assertIn('data-role="paper-intro"', styled)
        self.assertIn("background:#f3f4f6", styled)
        intro = styled.split('data-role="paper-intro"', 1)[1].split("</section>", 1)[0]
        self.assertIn("摘要导语内容。", intro)
        self.assertNotIn("第一节正文不能进入摘要框。", intro)
        cleaned = _remove_paper_figure_attributions(styled)
        self.assertIn("Fig. 2 | 图注内容。", cleaned)
        self.assertNotIn("图源：", cleaned)

        quote_html = (
            '<p><img alt="论文第一页" src="images/paper-first-page.png" /></p>'
            '<section data-role="blockquote" style="quote-style">独立摘要。</section>'
            '<h2>第一节</h2><p>第一节正文。</p>'
        )
        quote_styled = _style_paper_intro(quote_html)
        self.assertEqual(quote_styled.count('data-role="paper-intro"'), 1)
        self.assertNotIn('data-role="blockquote"', quote_styled)
        self.assertIn("独立摘要。", quote_styled)

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

    def test_pdf_figure_crop_padding_is_clamped_to_page_bounds(self):
        rect = _expanded_crop_rect([2.0, 3.0, 120.0, 140.0], pymupdf.Rect(0.0, 0.0, 130.0, 150.0))
        self.assertGreater(rect.y1 - rect.y0, 137.0)
        self.assertGreaterEqual(rect.x0, 0.0)
        self.assertGreaterEqual(rect.y0, 0.0)
        self.assertLessEqual(rect.x1, 130.0)
        self.assertLessEqual(rect.y1, 150.0)

    def test_pdf_figure_crop_refines_native_content_without_page_text(self):
        class FakePage:
            rect = pymupdf.Rect(0.0, 0.0, 600.0, 800.0)

            def get_text(self, _kind):
                return [
                    (20.0, 15.0, 580.0, 28.0, "Journal of Climate 2026", 0, 0, 0),
                    (180.0, 82.0, 230.0, 96.0, "(a)", 0, 0, 0),
                    (100.0, 330.0, 500.0, 390.0, "A long unrelated body paragraph", 0, 0, 0),
                    (100.0, 310.0, 500.0, 325.0, "Figure caption text", 0, 0, 0),
                ]

            def get_drawings(self):
                return [{"rect": pymupdf.Rect(110.0, 95.0, 490.0, 300.0)}]

        refined = _refine_figure_crop_bounds(
            FakePage(),
            [110.0, 100.0, 490.0, 300.0],
            [[100.0, 310.0, 500.0, 325.0]],
        )
        self.assertLessEqual(refined.y0, 82.0)
        self.assertGreater(refined.y0, 28.0)
        self.assertLessEqual(refined.y1, 305.0)
        self.assertGreaterEqual(refined.x0, 105.0)
        self.assertLessEqual(refined.x1, 495.0)

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
                )

            self.assertEqual(metadata["layout_picture_regions"], 2)
            self.assertEqual(metadata["matched_figures"], 1)
            self.assertEqual(len(metadata["rejected_picture_regions"]), 1)
            self.assertEqual(figures[0]["figure_number"], 3)
            self.assertEqual(figures[0]["caption_boxclasses"], ["text", "text"])
            self.assertIn("Panels a-j", figures[0]["original_caption"])
            self.assertTrue(Path(figures[0]["local_path"]).is_file())

    def test_pdf_figure_mapping_merges_all_panels_for_one_figure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def fake_download(_url, destination):
                document = pymupdf.open()
                page = document.new_page(width=600, height=800)
                page.draw_rect(pymupdf.Rect(100, 100, 250, 250), color=(1, 0, 0), fill=(1, 0, 0))
                page.draw_rect(pymupdf.Rect(300, 100, 500, 250), color=(0, 0, 1), fill=(0, 0, 1))
                document.save(destination)
                document.close()

            def fake_layout(document, **kwargs):
                raw_dir = Path(kwargs["image_path"])
                raw_dir.mkdir(parents=True, exist_ok=True)
                for index in range(4):
                    (raw_dir / f"{Path(document).name}-0001-0{index}.png").write_bytes(b"png")
                return {
                    "page_count": 1,
                    "pages": [
                        {
                            "page_number": 1,
                            "boxes": [
                                {
                                    "x0": 100.0,
                                    "y0": 100.0,
                                    "x1": 250.0,
                                    "y1": 250.0,
                                    "boxclass": "picture",
                                    "image": None,
                                    "table": None,
                                    "textlines": [],
                                },
                                {
                                    "x0": 300.0,
                                    "y0": 100.0,
                                    "x1": 500.0,
                                    "y1": 250.0,
                                    "boxclass": "picture",
                                    "image": None,
                                    "table": None,
                                    "textlines": [],
                                },
                                {
                                    "x0": 100.0,
                                    "y0": 270.0,
                                    "x1": 250.0,
                                    "y1": 400.0,
                                    "boxclass": "picture",
                                    "image": None,
                                    "table": None,
                                    "textlines": [],
                                },
                                {
                                    "x0": 300.0,
                                    "y0": 270.0,
                                    "x1": 500.0,
                                    "y1": 400.0,
                                    "boxclass": "picture",
                                    "image": None,
                                    "table": None,
                                    "textlines": [],
                                },
                                {
                                    "x0": 100.0,
                                    "y0": 426.0,
                                    "x1": 500.0,
                                    "y1": 486.0,
                                    "boxclass": "caption",
                                    "image": None,
                                    "table": None,
                                    "textlines": [
                                        {"spans": [{"text": "Fig. 5 | A complete two-panel result."}]}
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
                )

            self.assertEqual(metadata["matched_figures"], 1)
            self.assertEqual(figures[0]["figure_number"], 5)
            self.assertEqual(len(figures[0]["picture_bboxes"]), 4)
            self.assertEqual(figures[0]["picture_bbox"], [100.0, 100.0, 500.0, 400.0])
            output = pymupdf.open(figures[0]["local_path"])
            try:
                self.assertGreater(output[0].rect.width, 390)
                self.assertGreater(output[0].rect.height, 300)
            finally:
                output.close()
            self.assertNotIn("paper-first-page-cover.png", figures[0]["local_path"])

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

    def test_paper_selection_ignores_legacy_publishable_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            images = []
            for number in range(1, 9):
                path = Path(tmp) / f"figure-{number}.png"
                path.write_bytes(b"cached figure")
                images.append(
                    {
                        "local_path": str(path),
                        "url": f"https://example.test/figure-{number}.png",
                        "image_source": "pdf_figure",
                        "image_role": "figure",
                        "figure_number": number,
                        "caption": f"Figure {number} atmospheric circulation result",
                        "publishable": False,
                        "license": "unknown",
                    }
                )
            cover, body, _ = _select_article_images(
                images,
                PAPER_CONTENT,
                "Atmospheric circulation and surface wind results",
            )

        self.assertIsNotNone(cover)
        self.assertEqual(len(body), 4)
        self.assertTrue(all(image["local_path"] for image in body))

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
                    PAPER_LOOKBACK_HOURS: seen_items
                    + [make_item(index) for index in range(2, 20)],
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
                        "news.pipeline.select_paper_ranked",
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

                self.assertEqual(calls, [PAPER_LOOKBACK_HOURS])
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
                    patch("news.pipeline.select_paper_ranked", side_effect=select_batch),
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
                        "news.pipeline.select_paper_ranked",
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

                self.assertEqual(calls, [PAPER_LOOKBACK_HOURS])
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
                        "news.pipeline.select_paper_ranked",
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
                        "news.pipeline.select_paper_ranked",
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
                        "news.pipeline.select_paper_ranked",
                        side_effect=fake_selection,
                    ),
                    patch(
                        "news.pipeline.translate_paper_titles",
                        return_value=(['', ''], False, "502 server_is_overloaded"),
                    ) as translate,
                ):
                    candidates = await pipeline.refresh("2026-08-26", PAPER_CONTENT)

                self.assertEqual(translate.call_count, 2)
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
                pipeline.db.replace_paper_candidate_pool(
                    "2026-08-26",
                    [
                        {
                            "article_id": article_ids[index],
                            "score": float(30 - index * 10),
                            "title_cn": "已有标题" if index == 0 else "",
                        }
                        for index in range(3)
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
                    patch("news.pipeline.select_paper_ranked") as select,
                ):
                    candidates = await pipeline.get_or_refresh("2026-08-26", PAPER_CONTENT)

                self.assertEqual(translate.call_count, 1)
                self.assertEqual(select.call_count, 0)
                self.assertEqual(
                    [(value["rank"], value["title_cn"]) for value in candidates],
                    [(1, "已有标题"), (2, "补充标题2"), (3, "补充标题3")],
                )
                pool = pipeline.db.get_paper_candidate_pool("2026-08-26", PAPER_CONTENT)
                self.assertEqual(
                    [value["title_cn"] for value in pool],
                    ["已有标题", "补充标题2", "补充标题3"],
                )
                restarted = NewsPipeline(settings)
                with patch("news.pipeline.translate_paper_titles") as retry_translate:
                    restarted_result = await restarted.get_or_refresh("2026-08-26", PAPER_CONTENT)
                retry_translate.assert_not_called()
                self.assertEqual(
                    [value["title_cn"] for value in restarted_result],
                    ["已有标题", "补充标题2", "补充标题3"],
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
        title_prompt = client.chat.completions.create.call_args.kwargs["messages"][0][
            "content"
        ]
        self.assertIn("适合微信公众号显示", title_prompt)
        self.assertIn("禁止使用省略号或半截标题", title_prompt)
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
                    return [dict(value) for value in values], True, ""

                def translate_batch(values, _settings):
                    return [str(value.get("title_cn") or "") for value in values], True, ""

                with (
                    patch("news.pipeline.fetch_all_feeds", return_value=(items, [], {"test": 24})),
                    patch("news.pipeline.select_paper_ranked", side_effect=select_batch),
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
                self.assertIn("本批次新增10篇，已展示20/24篇", second_text)
                self.assertIn("11.", second_text)
                with patch(
                    "news.pipeline.translate_paper_titles",
                    return_value=(['' for _ in range(24)], True, ""),
                ):
                    all_current = await pipeline.get_or_refresh("2026-08-26", PAPER_CONTENT)
                self.assertEqual(len(all_current), 24)
                self.assertIn("精选论文候选（近90天；已展示24/24篇）", pipeline.format_news(all_current))
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
                        "news.pipeline.select_paper_ranked",
                        return_value=([], False, "429 usage_limit_reached"),
                    ),
                    patch("news.pipeline.deduplicate", side_effect=lambda values: values),
                ):
                    failed = await pipeline.next_paper_batch("2026-08-26")
                self.assertEqual(
                    [item["title"] for item in failed],
                    [f"Paper {i}" for i in range(1, 25)],
                )
                self.assertIn("⚠ 已到最后一批，近90天候选池共 24 篇", pipeline.format_news(failed))
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

    def test_paper_uses_one_90_day_window_and_persists_full_pool(self):
        async def check():
            with tempfile.TemporaryDirectory() as tmp:
                settings = replace(
                    load_settings(),
                    database_path=Path(tmp) / "paper-90-day.db",
                    model_base_url="",
                    model_api_key="",
                    model_name="",
                    openalex_api_key="",
                )
                pipeline = NewsPipeline(settings)
                items = [
                    {
                        "source": "Journal of Climate",
                        "url": f"https://example.test/90-day-{index}",
                        "canonical_url": f"https://example.test/90-day-{index}",
                        "title": f"Distinct climate mechanism finding {index}",
                        "summary": "Near-surface wind climate mechanism",
                        "published_at": "2026-06-01T00:00:00+00:00",
                        "doi": f"10.1000/90-day-{index}",
                        "journal": "Journal of Climate",
                        "word_count": 800,
                        "status": "discovered",
                        "discovered_at": "2026-09-08T00:00:00+00:00",
                    }
                    for index in range(12)
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
                    patch("news.pipeline.deduplicate", side_effect=lambda values: values),
                ):
                    first_page = await pipeline.refresh("2026-09-08", PAPER_CONTENT)

                self.assertEqual(calls, [PAPER_LOOKBACK_HOURS])
                self.assertEqual(len(first_page), 10)
                self.assertEqual(
                    pipeline.db.get_paper_candidate_pool_count("2026-09-08"),
                    12,
                )
                self.assertEqual(
                    pipeline.last_paper_discovery_stats["lookback_days"],
                    90,
                )

        asyncio.run(check())

    def test_journal_first_broad_prefilter_excludes_obvious_off_topic(self):
        self.assertTrue(
            is_broad_journal_first_paper(
                {
                    "title": "Aerosol forcing in the Earth system",
                    "summary": "Atmospheric aerosol changes climate radiation",
                    "journal": "Nature",
                }
            )
        )
        self.assertFalse(
            is_broad_journal_first_paper(
                {
                    "title": "A new cancer treatment",
                    "summary": "Clinical patient outcomes",
                    "journal": "Nature",
                }
            )
        )
        self.assertEqual(
            paper_journal_tier({"journal": "PNAS"}),
            1,
        )

    def test_paper_next_reads_persisted_pool_without_network_or_model(self):
        async def check():
            with tempfile.TemporaryDirectory() as tmp:
                database_path = Path(tmp) / "paper-pages.db"
                settings = replace(load_settings(), database_path=database_path)
                pipeline = NewsPipeline(settings)
                pool = []
                for index in range(22):
                    item = {
                        "source": "Nature",
                        "url": f"https://example.test/page-{index}",
                        "canonical_url": f"https://example.test/page-{index}",
                        "title": f"Page paper {index}",
                        "summary": "Climate",
                        "doi": f"10.1000/page-{index}",
                        "journal": "Nature",
                    }
                    article_id = pipeline.db.upsert_article(item)
                    pool.append(
                        {
                            "article_id": article_id,
                            "title_cn": f"论文{index}",
                            "score": float(index),
                        }
                    )
                pipeline.db.replace_paper_candidate_pool("2026-09-08", pool)
                pipeline.db.replace_candidates(
                    "2026-09-08", pool[:10], PAPER_CONTENT
                )
                pipeline.openalex.discover_recent_papers = MagicMock(
                    side_effect=AssertionError("next must not discover")
                )
                with patch(
                    "news.pipeline.select_paper_ranked",
                    side_effect=AssertionError("next must not call AI"),
                ):
                    second = await pipeline.next_paper_batch("2026-09-08")
                    third = await pipeline.next_paper_batch("2026-09-08")

                self.assertEqual([item["rank"] for item in second], list(range(11, 21)))
                self.assertEqual([item["rank"] for item in third], [21, 22])
                restarted = NewsPipeline(settings)
                exhausted = await restarted.next_paper_batch("2026-09-08")
                self.assertIn("最后一批", restarted.last_paper_refresh_warning)
                self.assertEqual(len(exhausted), 22)
                pipeline.openalex.discover_recent_papers.assert_not_called()

        asyncio.run(check())

    def test_paper_ranked_selector_processes_all_candidates(self):
        settings = replace(
            load_settings(),
            model_base_url="https://model.example/v1",
            model_api_key="test-key",
            model_name="test-model",
        )
        candidates = [
            {
                "title": f"Climate mechanism paper {index}",
                "summary": "Climate mechanism",
                "paper_local_score": 2,
            }
            for index in range(35)
        ]
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "items": [
                                    {"index": index, "score": 2, "reason": "relevant"}
                                    for index in range(1, 36)
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
            selected, used_model, error = select_paper_ranked(candidates, settings)
        payload = json.loads(
            client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        )
        self.assertTrue(used_model)
        self.assertEqual(error, "")
        self.assertEqual(len(payload), 35)
        self.assertEqual(len(selected), 35)


    def test_paper_refresh_translates_only_first_ten_of_large_pool(self):
        async def check():
            with tempfile.TemporaryDirectory() as tmp:
                settings = replace(
                    load_settings(),
                    database_path=Path(tmp) / "large-title-page.db",
                    model_base_url="https://model.example/v1",
                    model_api_key="test-key",
                    model_name="test-model",
                    openalex_api_key="",
                )
                pipeline = NewsPipeline(settings)
                items = [
                    {
                        "source": "Nature",
                        "url": f"https://example.test/large-title-{index}",
                        "canonical_url": f"https://example.test/large-title-{index}",
                        "title": f"Large title paper {index} with distinct subject {index}",
                        "summary": "Near-surface wind climate mechanism",
                        "published_at": "2026-09-09T00:00:00+00:00",
                        "doi": f"10.1000/large-title-{index}",
                        "journal": "Nature",
                        "word_count": 800,
                        "status": "discovered",
                        "discovered_at": "2026-09-09T00:00:00+00:00",
                    }
                    for index in range(842)
                ]
                translation_sizes = []
                pipeline._extract_shortlist = lambda values: asyncio.sleep(0, result=copy.deepcopy(values))
                pipeline._published_papers = lambda values, _date: asyncio.sleep(
                    0, result=[dict(value, paper_local_score=2) for value in values]
                )
                with (
                    patch("news.pipeline.fetch_all_feeds", return_value=(items, [], {"test": 842})),
                    patch("news.pipeline.deduplicate", side_effect=lambda values: values),
                    patch(
                        "news.pipeline.select_paper_ranked",
                        side_effect=lambda values, _settings: (
                            [dict(value, paper_relevance_score=3) for value in values],
                            True,
                            "",
                        ),
                    ),
                    patch(
                        "news.pipeline.translate_paper_titles",
                        side_effect=lambda values, _settings: (
                            translation_sizes.append(len(values)) or [f"中文标题{index}" for index, _ in enumerate(values)],
                            True,
                            "",
                        ),
                    ),
                ):
                    first_page = await pipeline.refresh("2026-09-09", PAPER_CONTENT)
                pool = pipeline.db.get_paper_candidate_pool("2026-09-09", PAPER_CONTENT)
                self.assertEqual(len(first_page), 10)
                self.assertEqual(translation_sizes, [10])
                self.assertEqual(len(pool), 842)
                self.assertTrue(all(value["title_cn"] for value in pool[:10]))
                self.assertTrue(all(not value["title_cn"] for value in pool[10:]))

        asyncio.run(check())

    def test_paper_next_translates_page_and_keeps_pool_order(self):
        async def check():
            with tempfile.TemporaryDirectory() as tmp:
                settings = replace(
                    load_settings(),
                    database_path=Path(tmp) / "next-title-page.db",
                    model_base_url="https://model.example/v1",
                    model_api_key="test-key",
                    model_name="test-model",
                )
                pipeline = NewsPipeline(settings)
                pool = []
                for index in range(22):
                    article_id = pipeline.db.upsert_article(
                        {
                            "source": "Nature",
                            "url": f"https://example.test/page-title-{index}",
                            "canonical_url": f"https://example.test/page-title-{index}",
                            "title": f"Page title paper {index}",
                            "summary": "Climate",
                            "doi": f"10.1000/page-title-{index}",
                            "journal": "Nature",
                            "word_count": 800,
                        }
                    )
                    pool.append({"article_id": article_id, "score": float(100 - index), "title_cn": ""})
                pipeline.db.replace_paper_candidate_pool("2026-09-09", pool, PAPER_CONTENT)
                pipeline.db.replace_candidates("2026-09-09", pool[:10], PAPER_CONTENT)
                seen_pages = []

                def translate_page(values, _settings):
                    seen_pages.append([value["rank"] for value in values])
                    return [f"页标题{value['rank']}" for value in values], True, ""

                pipeline.openalex.discover_recent_papers = MagicMock(
                    side_effect=AssertionError("OpenAlex called")
                )
                with (
                    patch("news.pipeline.fetch_all_feeds", side_effect=AssertionError("feeds called")),
                    patch("news.pipeline.select_paper_ranked", side_effect=AssertionError("ranking called")),
                    patch("news.pipeline.translate_paper_titles", side_effect=translate_page),
                ):
                    page = await pipeline.next_paper_batch("2026-09-09")
                self.assertEqual([value["rank"] for value in page], list(range(11, 21)))
                self.assertEqual(seen_pages, [list(range(11, 21))])
                stored = pipeline.db.get_paper_candidate_pool("2026-09-09", PAPER_CONTENT)
                self.assertEqual([value["title_cn"] for value in stored[10:20]], [f"页标题{rank}" for rank in range(11, 21)])
                self.assertEqual([value["score"] for value in stored], [float(100 - index) for index in range(22)])
                pipeline.openalex.discover_recent_papers.assert_not_called()

        asyncio.run(check())

    def test_paper_page_translation_retries_timeout_and_falls_back_without_mutation(self):
        async def check():
            with tempfile.TemporaryDirectory() as tmp:
                settings = replace(
                    load_settings(),
                    database_path=Path(tmp) / "title-timeout.db",
                    model_base_url="https://model.example/v1",
                    model_api_key="test-key",
                    model_name="test-model",
                )
                pipeline = NewsPipeline(settings)
                article_id = pipeline.db.upsert_article(
                    {
                        "source": "Nature",
                        "url": "https://example.test/title-timeout",
                        "canonical_url": "https://example.test/title-timeout",
                        "title": "Timeout paper",
                        "summary": "Climate",
                        "doi": "10.1000/title-timeout",
                        "journal": "Nature",
                        "word_count": 800,
                    }
                )
                candidate = {"article_id": article_id, "score": 42.0, "title_cn": ""}
                pipeline.db.replace_paper_candidate_pool("2026-09-09", [candidate], PAPER_CONTENT)
                page = pipeline.db.get_paper_candidate_page("2026-09-09", 1, 10, PAPER_CONTENT)
                with patch(
                    "news.pipeline.translate_paper_titles",
                    side_effect=[
                        TimeoutError("Request timed out"),
                        (["重试成功"], True, ""),
                    ],
                ) as translate:
                    result = await pipeline._translate_paper_page("2026-09-09", page)
                self.assertEqual(translate.call_count, 2)
                self.assertEqual(result[0]["title_cn"], "重试成功")
                before = pipeline.db.get_paper_candidate_pool("2026-09-09", PAPER_CONTENT)
                with patch(
                    "news.pipeline.translate_paper_titles",
                    side_effect=[TimeoutError("Request timed out"), TimeoutError("Request timed out")],
                ) as translate_failed:
                    result = await pipeline._translate_paper_page("2026-09-09", page)
                self.assertEqual(translate_failed.call_count, 2)
                self.assertEqual(result[0]["title_cn"], "")
                after = pipeline.db.get_paper_candidate_pool("2026-09-09", PAPER_CONTENT)
                self.assertEqual([(value["rank"], value["score"]) for value in before], [(value["rank"], value["score"]) for value in after])

        asyncio.run(check())


    def test_paper_title_translation_retries_timeout_rate_limit_and_5xx(self):
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
                        content=json.dumps({"items": [{"index": 1, "title_cn": "中文标题"}]})
                    )
                )
            ]
        )

        class HttpError(Exception):
            def __init__(self, status_code):
                super().__init__(f"HTTP {status_code}")
                self.status_code = status_code

        class APITimeoutError(Exception):
            pass

        for failure in (HttpError(429), HttpError(504), APITimeoutError("timed out")):
            client = MagicMock()
            client.chat.completions.create.side_effect = [failure, response]
            with (
                patch("writer.llm.OpenAI", return_value=client),
                patch("writer.llm.time.sleep"),
            ):
                titles, used_model, error = translate_paper_titles(
                    [{"title": "English title", "title_cn": ""}],
                    settings,
                )
            self.assertTrue(used_model)
            self.assertEqual(error, "")
            self.assertEqual(titles, ["中文标题"])
            self.assertEqual(client.chat.completions.create.call_count, 2)


    def test_paper_partial_page_translation_keeps_english_fallback(self):
        async def check():
            with tempfile.TemporaryDirectory() as tmp:
                settings = replace(
                    load_settings(),
                    database_path=Path(tmp) / "partial-title-page.db",
                    model_base_url="https://model.example/v1",
                    model_api_key="test-key",
                    model_name="test-model",
                )
                pipeline = NewsPipeline(settings)
                pool = []
                for index in range(3):
                    article_id = pipeline.db.upsert_article(
                        {
                            "source": "Nature",
                            "url": f"https://example.test/partial-title-{index}",
                            "canonical_url": f"https://example.test/partial-title-{index}",
                            "title": f"Partial title paper {index}",
                            "summary": "Climate",
                            "doi": f"10.1000/partial-title-{index}",
                            "journal": "Nature",
                        }
                    )
                    pool.append({"article_id": article_id, "score": float(index), "title_cn": ""})
                pipeline.db.replace_paper_candidate_pool("2026-09-09", pool, PAPER_CONTENT)
                page = pipeline.db.get_paper_candidate_page("2026-09-09", 1, 10, PAPER_CONTENT)
                with patch(
                    "news.pipeline.translate_paper_titles",
                    return_value=(["中文标题", "", ""], True, ""),
                ):
                    result = await pipeline._translate_paper_page("2026-09-09", page)
                self.assertEqual([item["title_cn"] for item in result], ["中文标题", "", ""])
                self.assertIn("Partial title paper 1", pipeline.format_news(result))
                stored = pipeline.db.get_paper_candidate_pool("2026-09-09", PAPER_CONTENT)
                self.assertEqual(stored[0]["title_cn"], "中文标题")
                self.assertEqual(stored[1]["title_cn"], "")

        asyncio.run(check())


if __name__ == "__main__":
    unittest.main()
