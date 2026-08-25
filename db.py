"""Minimal SQLite persistence for news candidates and publishing state."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    url TEXT NOT NULL,
    canonical_url TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    published_at TEXT NOT NULL DEFAULT '',
    doi TEXT NOT NULL DEFAULT '',
    journal TEXT NOT NULL DEFAULT '',
    word_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'discovered',
    discovered_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_articles_canonical ON articles(canonical_url);
CREATE INDEX IF NOT EXISTS idx_articles_doi ON articles(doi);
CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at);

CREATE TABLE IF NOT EXISTS daily_candidates (
    date TEXT NOT NULL,
    content_type TEXT NOT NULL DEFAULT '',
    rank INTEGER NOT NULL,
    article_id INTEGER NOT NULL REFERENCES articles(id),
    title_cn TEXT NOT NULL DEFAULT '',
    score REAL NOT NULL,
    PRIMARY KEY (date, content_type, rank)
);

CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id INTEGER NOT NULL REFERENCES articles(id),
    url TEXT NOT NULL,
    local_path TEXT NOT NULL DEFAULT '',
    caption TEXT NOT NULL DEFAULT '',
    credit TEXT NOT NULL DEFAULT '',
    license TEXT NOT NULL DEFAULT '',
    publishable INTEGER NOT NULL DEFAULT 0,
    reason TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_images_article ON images(article_id);

CREATE TABLE IF NOT EXISTS generated_posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id INTEGER NOT NULL REFERENCES articles(id),
    markdown_path TEXT NOT NULL,
    model TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS publish_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id INTEGER NOT NULL REFERENCES articles(id),
    draft_media_id TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS daily_runs (
    date TEXT NOT NULL,
    content_type TEXT NOT NULL DEFAULT '',
    fetched_at TEXT NOT NULL DEFAULT '',
    pushed_at TEXT NOT NULL DEFAULT '',
    candidate_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (date, content_type)
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)

            run_info = connection.execute("PRAGMA table_info(daily_runs)").fetchall()
            run_pk = [
                row["name"]
                for row in sorted(run_info, key=lambda row: row["pk"])
                if row["pk"]
            ]
            if run_pk != ["date", "content_type"]:
                connection.executescript(
                    """
                    ALTER TABLE daily_runs RENAME TO daily_runs_legacy;
                    CREATE TABLE daily_runs (
                        date TEXT NOT NULL,
                        content_type TEXT NOT NULL DEFAULT '',
                        fetched_at TEXT NOT NULL DEFAULT '',
                        pushed_at TEXT NOT NULL DEFAULT '',
                        candidate_count INTEGER NOT NULL DEFAULT 0,
                        status TEXT NOT NULL DEFAULT '',
                        error TEXT NOT NULL DEFAULT '',
                        PRIMARY KEY (date, content_type)
                    );
                    INSERT INTO daily_runs
                    (date,content_type,fetched_at,pushed_at,candidate_count,status,error)
                    SELECT date,COALESCE(content_type,''),fetched_at,pushed_at,
                           candidate_count,status,error
                    FROM daily_runs_legacy;
                    DROP TABLE daily_runs_legacy;
                    """
                )

            candidate_info = connection.execute(
                "PRAGMA table_info(daily_candidates)"
            ).fetchall()
            candidate_pk = [
                row["name"]
                for row in sorted(candidate_info, key=lambda row: row["pk"])
                if row["pk"]
            ]
            if candidate_pk != ["date", "content_type", "rank"]:
                has_content_type = any(
                    row["name"] == "content_type" for row in candidate_info
                )
                content_expression = (
                    "COALESCE(c.content_type, r.content_type, '')"
                    if has_content_type
                    else "COALESCE(r.content_type, '')"
                )
                connection.execute(
                    "ALTER TABLE daily_candidates RENAME TO daily_candidates_legacy"
                )
                connection.execute(
                    """CREATE TABLE daily_candidates (
                    date TEXT NOT NULL,
                    content_type TEXT NOT NULL DEFAULT '',
                    rank INTEGER NOT NULL,
                    article_id INTEGER NOT NULL REFERENCES articles(id),
                    title_cn TEXT NOT NULL DEFAULT '',
                    score REAL NOT NULL,
                    PRIMARY KEY (date, content_type, rank)
                    )"""
                )
                connection.execute(
                    f"""INSERT INTO daily_candidates
                    (date,content_type,rank,article_id,title_cn,score)
                    SELECT c.date,{content_expression},c.rank,c.article_id,c.title_cn,c.score
                    FROM daily_candidates_legacy c
                    LEFT JOIN daily_runs r ON r.date=c.date"""
                )
                connection.execute("DROP TABLE daily_candidates_legacy")

    def upsert_article(self, article: dict[str, Any]) -> int:
        doi = str(article.get("doi") or "")
        canonical = str(article.get("canonical_url") or article.get("url") or "")
        with self.connect() as connection:
            row = None
            if doi:
                row = connection.execute(
                    "SELECT id FROM articles WHERE doi = ? ORDER BY id LIMIT 1", (doi,)
                ).fetchone()
            if row is None and canonical:
                row = connection.execute(
                    "SELECT id FROM articles WHERE canonical_url = ? ORDER BY id LIMIT 1",
                    (canonical,),
                ).fetchone()
            values = (
                str(article.get("source") or ""),
                str(article.get("url") or ""),
                canonical,
                str(article.get("title") or ""),
                str(article.get("summary") or ""),
                str(article.get("published_at") or ""),
                doi,
                str(article.get("journal") or ""),
                int(article.get("word_count") or 0),
                str(article.get("status") or "discovered"),
                str(article.get("discovered_at") or utc_now()),
            )
            if row:
                article_id = int(row["id"])
                connection.execute(
                    """UPDATE articles SET source=?, url=?, canonical_url=?, title=?, summary=?,
                    published_at=?, doi=?, journal=?, word_count=?, status=?, discovered_at=?
                    WHERE id=?""",
                    (*values, article_id),
                )
                return article_id
            cursor = connection.execute(
                """INSERT INTO articles
                (source,url,canonical_url,title,summary,published_at,doi,journal,word_count,status,discovered_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                values,
            )
            return int(cursor.lastrowid)

    def replace_candidates(
        self,
        date: str,
        candidates: Iterable[dict[str, Any]],
        content_type: str = "",
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                "DELETE FROM daily_candidates WHERE date=? AND content_type=?",
                (date, content_type),
            )
            for rank, candidate in enumerate(candidates, start=1):
                connection.execute(
                    """INSERT INTO daily_candidates
                    (date,content_type,rank,article_id,title_cn,score)
                    VALUES (?,?,?,?,?,?)""",
                    (
                        date,
                        content_type,
                        rank,
                        int(candidate["article_id"]),
                        str(candidate.get("title_cn") or ""),
                        float(candidate.get("score") or 0),
                    ),
                )

    def get_candidates(
        self,
        date: str,
        content_type: str | None = None,
    ) -> list[dict[str, Any]]:
        where = "c.date=?"
        params: tuple[Any, ...] = (date,)
        if content_type is not None:
            where += " AND c.content_type=?"
            params = (date, content_type)
        with self.connect() as connection:
            rows = connection.execute(
                f"""SELECT c.date,c.content_type,c.rank,c.title_cn,c.score,a.*
                FROM daily_candidates c JOIN articles a ON a.id=c.article_id
                WHERE {where} ORDER BY c.content_type,c.rank""",
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def get_candidate(
        self,
        date: str,
        rank: int,
        content_type: str | None = None,
    ) -> dict[str, Any] | None:
        where = "c.date=? AND c.rank=?"
        params: tuple[Any, ...] = (date, rank)
        if content_type is not None:
            where += " AND c.content_type=?"
            params = (date, rank, content_type)
        with self.connect() as connection:
            row = connection.execute(
                f"""SELECT c.date,c.content_type,c.rank,c.title_cn,c.score,a.*
                FROM daily_candidates c JOIN articles a ON a.id=c.article_id
                WHERE {where} ORDER BY c.content_type LIMIT 1""",
                params,
            ).fetchone()
        return dict(row) if row else None

    def replace_images(self, article_id: int, images: Iterable[dict[str, Any]]) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM images WHERE article_id=?", (article_id,))
            for image in images:
                connection.execute(
                    """INSERT INTO images
                    (article_id,url,local_path,caption,credit,license,publishable,reason)
                    VALUES (?,?,?,?,?,?,?,?)""",
                    (
                        article_id,
                        str(image.get("url") or ""),
                        str(image.get("local_path") or ""),
                        str(image.get("caption") or image.get("alt") or ""),
                        str(image.get("credit") or ""),
                        str(image.get("license") or ""),
                        1 if image.get("publishable") else 0,
                        str(image.get("reason") or ""),
                    ),
                )

    def get_images(self, article_id: int) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM images WHERE article_id=? ORDER BY id", (article_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def save_generated_post(self, article_id: int, path: str, model: str, status: str) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """INSERT INTO generated_posts(article_id,markdown_path,model,created_at,status)
                VALUES (?,?,?,?,?)""",
                (article_id, path, model, utc_now(), status),
            )
            return int(cursor.lastrowid)

    def latest_generated_post(
        self,
        article_id: int,
        content_type: str | None = None,
    ) -> dict[str, Any] | None:
        with self.connect() as connection:
            if content_type is None:
                row = connection.execute(
                    "SELECT * FROM generated_posts WHERE article_id=? ORDER BY id DESC LIMIT 1",
                    (article_id,),
                ).fetchone()
            else:
                row = connection.execute(
                    """SELECT * FROM generated_posts
                    WHERE article_id=? AND markdown_path LIKE ?
                    ORDER BY id DESC LIMIT 1""",
                    (article_id, f"%/articles/{content_type}/%"),
                ).fetchone()
        return dict(row) if row else None

    def save_publish_history(
        self,
        article_id: int,
        status: str,
        draft_media_id: str = "",
        error: str = "",
    ) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """INSERT INTO publish_history(article_id,draft_media_id,created_at,status,error)
                VALUES (?,?,?,?,?)""",
                (article_id, draft_media_id, utc_now(), status, error),
            )
            return int(cursor.lastrowid)

    def published_article_identifiers(self) -> dict[str, set[Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT DISTINCT a.id,a.doi,a.canonical_url
                FROM publish_history p JOIN articles a ON a.id=p.article_id
                WHERE p.status='drafted'"""
            ).fetchall()
        return {
            "article_ids": {int(row["id"]) for row in rows},
            "dois": {
                str(row["doi"] or "").strip().lower()
                for row in rows
                if str(row["doi"] or "").strip()
            },
            "canonical_urls": {
                str(row["canonical_url"] or "").strip()
                for row in rows
                if str(row["canonical_url"] or "").strip()
            },
        }

    def set_daily_run(
        self,
        date: str,
        *,
        fetched_at: str | None = None,
        pushed_at: str | None = None,
        candidate_count: int | None = None,
        content_type: str | None = None,
        status: str,
        error: str = "",
    ) -> None:
        current = self.get_daily_run(date, content_type) or {}
        run_type = (
            content_type if content_type is not None else current.get("content_type", "")
        )
        values = (
            date,
            run_type,
            fetched_at if fetched_at is not None else current.get("fetched_at", ""),
            pushed_at if pushed_at is not None else current.get("pushed_at", ""),
            candidate_count if candidate_count is not None else current.get("candidate_count", 0),
            status,
            error,
        )
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO daily_runs
                (date,content_type,fetched_at,pushed_at,candidate_count,status,error)
                VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(date,content_type) DO UPDATE SET
                fetched_at=excluded.fetched_at,pushed_at=excluded.pushed_at,
                candidate_count=excluded.candidate_count,status=excluded.status,
                error=excluded.error""",
                values,
            )

    def get_daily_run(
        self,
        date: str,
        content_type: str | None = None,
    ) -> dict[str, Any] | None:
        with self.connect() as connection:
            if content_type is None:
                row = connection.execute(
                    """SELECT * FROM daily_runs WHERE date=?
                    ORDER BY (pushed_at != '') DESC, fetched_at DESC LIMIT 1""",
                    (date,),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT * FROM daily_runs WHERE date=? AND content_type=?",
                    (date, content_type),
                ).fetchone()
        return dict(row) if row else None

    def status_snapshot(self, date: str) -> dict[str, Any]:
        run = self.get_daily_run(date) or {}
        with self.connect() as connection:
            if run.get("content_type"):
                candidate_count = connection.execute(
                    """SELECT COUNT(*) FROM daily_candidates
                    WHERE date=? AND content_type=?""",
                    (date, run["content_type"]),
                ).fetchone()[0]
            else:
                candidate_count = connection.execute(
                    "SELECT COUNT(*) FROM daily_candidates WHERE date=?",
                    (date,),
                ).fetchone()[0]
            last_error = connection.execute(
                """SELECT error FROM daily_runs WHERE error != '' ORDER BY date DESC LIMIT 1"""
            ).fetchone()
        return {
            "candidate_count": int(candidate_count),
            "fetched_at": run.get("fetched_at", ""),
            "pushed_at": run.get("pushed_at", ""),
            "status": run.get("status", "not_run"),
            "content_type": run.get("content_type", ""),
            "last_error": last_error[0] if last_error else "",
        }

    def history(self, limit: int = 10) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT 'generated' AS kind,g.id,g.article_id,g.created_at,g.status,'' AS error,
                a.title,g.markdown_path AS detail FROM generated_posts g
                JOIN articles a ON a.id=g.article_id
                UNION ALL
                SELECT 'draft' AS kind,p.id,p.article_id,p.created_at,p.status,p.error,
                a.title,p.draft_media_id AS detail FROM publish_history p
                JOIN articles a ON a.id=p.article_id
                ORDER BY created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
