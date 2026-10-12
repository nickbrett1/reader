"""The ONLY module that opens SQLite.

Every ``sqlite3`` connection in the codebase is created here, so the
record/index boundary is one reviewable place.  WAL mode is on.  All writes are
idempotent and resumable: re-running a stage updates rows in place rather than
duplicating them (§5.2, §5.5).

Vectors are *not* stored here (they live in Turbopuffer); SQLite stores the
per-row embedding **provenance** (model_id, dimensions, embed_config_version)
so a model swap can be scoped.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from . import schema


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


class SchemaMismatch(RuntimeError):
    """Raised when the on-disk schema version is not what we expect."""


def connect(path: Path | str) -> sqlite3.Connection:
    """Open the database with WAL mode and foreign keys enforced."""
    p = Path(path)
    if p.parent and str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def current_version(conn: sqlite3.Connection) -> int:
    try:
        row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    except sqlite3.OperationalError:
        return 0
    return int(row[0]) if row and row[0] is not None else 0


def apply_migrations(conn: sqlite3.Connection) -> int:
    """Apply pending migrations.  Returns the resulting schema version."""
    cur = current_version(conn)
    for version in sorted(schema.MIGRATIONS):
        if version <= cur:
            continue
        conn.executescript(schema.MIGRATIONS[version])
        if cur == 0:
            conn.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                (version, now_iso()),
            )
        else:
            conn.execute(
                "UPDATE schema_version SET version = ?, applied_at = ?",
                (version, now_iso()),
            )
        conn.commit()
        cur = version
    return cur


def check_schema(conn: sqlite3.Connection, expected: int = schema.SCHEMA_VERSION) -> None:
    found = current_version(conn)
    if found != expected:
        raise SchemaMismatch(
            f"database schema version {found} != expected {expected}; "
            "run a migration or rebuild"
        )


class Store:
    """A thin, explicit wrapper over one connection."""

    def __init__(self, path: Path | str, *, migrate: bool = True) -> None:
        self.path = Path(path)
        self.conn = connect(self.path)
        if migrate:
            apply_migrations(self.conn)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- embedding config (must be pinned before the first write) -----------

    def get_embedding_config(self) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT embedding_dimensions, embedding_model, embed_config_version "
            "FROM schema_version LIMIT 1"
        ).fetchone()

    def pin_embedding_config(
        self, dimensions: int, model: str, config_version: str = schema.EMBED_CONFIG_VERSION
    ) -> None:
        if dimensions not in schema.CANDIDATE_DIMENSIONS:
            raise ValueError(
                f"dimensions {dimensions} not in {schema.CANDIDATE_DIMENSIONS}"
            )
        self.conn.execute(
            "UPDATE schema_version SET embedding_dimensions = ?, embedding_model = ?, "
            "embed_config_version = ?",
            (dimensions, model, config_version),
        )
        self.conn.commit()

    # --- runs ---------------------------------------------------------------

    def start_run(self, verb: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO run (verb, started_at) VALUES (?, ?)", (verb, now_iso())
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def finish_run(self, run_id: int, status: str, log: str = "") -> None:
        self.conn.execute(
            "UPDATE run SET finished_at = ?, status = ?, log = ? WHERE id = ?",
            (now_iso(), status, log, run_id),
        )
        self.conn.commit()

    # --- books / pages / chunks --------------------------------------------

    def upsert_book(
        self,
        work_key: str,
        fmt: str,
        source_path: str,
        source_sha256: str,
        title: str | None = None,
        author: str | None = None,
    ) -> int:
        self.conn.execute(
            """
            INSERT INTO book (work_key, title, author, format, source_path,
                              source_sha256, added_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(work_key) DO UPDATE SET
                title = excluded.title, author = excluded.author,
                format = excluded.format, source_path = excluded.source_path,
                source_sha256 = excluded.source_sha256
            """,
            (work_key, title, author, fmt, source_path, source_sha256, now_iso()),
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT id FROM book WHERE work_key = ?", (work_key,)
        ).fetchone()
        return int(row["id"])

    def upsert_page(
        self,
        book_id: int,
        page_no: int,
        text: str,
        source_sha256: str,
        scanned: bool = False,
        extractor_version: str = schema.EXTRACTOR_VERSION,
    ) -> int:
        self.conn.execute(
            """
            INSERT INTO document_page (book_id, page_no, text, scanned,
                                       extractor_version, source_sha256)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(book_id, page_no) DO UPDATE SET
                text = excluded.text, scanned = excluded.scanned,
                extractor_version = excluded.extractor_version,
                source_sha256 = excluded.source_sha256
            """,
            (book_id, page_no, text, int(scanned), extractor_version, source_sha256),
        )
        row = self.conn.execute(
            "SELECT id FROM document_page WHERE book_id = ? AND page_no = ?",
            (book_id, page_no),
        ).fetchone()
        page_id = int(row["id"])
        # A re-extract invalidates any chunks cut from the old text.
        self.conn.execute("DELETE FROM chunk WHERE page_id = ?", (page_id,))
        self.conn.commit()
        return page_id

    def set_chunks(self, page_id: int, chunks: Sequence[tuple[int, int, int, str]]) -> None:
        """Replace a page's chunks.  ``chunks`` = (ordinal, start, end, text)."""
        self.conn.execute("DELETE FROM chunk WHERE page_id = ?", (page_id,))
        self.conn.executemany(
            "INSERT INTO chunk (page_id, ordinal, start_offset, end_offset, text) "
            "VALUES (?, ?, ?, ?, ?)",
            [(page_id, o, s, e, t) for (o, s, e, t) in chunks],
        )
        self.conn.commit()

    # --- images -------------------------------------------------------------

    def upsert_image(
        self,
        sha256: str,
        path: str,
        mime: str,
        nbytes: int,
        kind: str,
        source_sha256: str,
        *,
        phash: str | None = None,
        w: int | None = None,
        h: int | None = None,
        book_id: int | None = None,
        page_no: int | None = None,
        caption: str | None = None,
        thumb: bytes | None = None,
        extractor_version: str = schema.EXTRACTOR_VERSION,
    ) -> int:
        self.conn.execute(
            """
            INSERT INTO image (sha256, path, mime, bytes, kind, source_sha256,
                               phash, w, h, book_id, page_no, caption, thumb,
                               extractor_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(sha256) DO UPDATE SET
                path = excluded.path, mime = excluded.mime, bytes = excluded.bytes,
                phash = excluded.phash, w = excluded.w, h = excluded.h,
                book_id = excluded.book_id, page_no = excluded.page_no,
                caption = COALESCE(excluded.caption, image.caption),
                thumb = COALESCE(excluded.thumb, image.thumb)
            """,
            (
                sha256, path, mime, nbytes, kind, source_sha256, phash, w, h,
                book_id, page_no, caption, thumb, extractor_version,
            ),
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT id FROM image WHERE sha256 = ?", (sha256,)
        ).fetchone()
        return int(row["id"])

    # --- concepts -----------------------------------------------------------

    def ensure_concept(self, name: str, type_: str = "term") -> int:
        self.conn.execute(
            "INSERT INTO concept (name, type) VALUES (?, ?) "
            "ON CONFLICT(name, type) DO NOTHING",
            (name, type_),
        )
        row = self.conn.execute(
            "SELECT id FROM concept WHERE name = ? AND type = ?", (name, type_)
        ).fetchone()
        self.conn.commit()
        return int(row["id"])

    def add_surface(self, concept_id: int, surface: str) -> int:
        self.conn.execute(
            "INSERT INTO concept_surface (concept_id, surface) VALUES (?, ?) "
            "ON CONFLICT(concept_id, surface) DO NOTHING",
            (concept_id, surface),
        )
        row = self.conn.execute(
            "SELECT id FROM concept_surface WHERE concept_id = ? AND surface = ?",
            (concept_id, surface),
        ).fetchone()
        self.conn.commit()
        return int(row["id"])

    def add_mention(
        self, concept_id: int, surface_id: int, page_id: int, start: int, end: int
    ) -> None:
        self.conn.execute(
            "INSERT INTO concept_mention (concept_id, surface_id, page_id, "
            "start_offset, end_offset) VALUES (?, ?, ?, ?, ?)",
            (concept_id, surface_id, page_id, start, end),
        )
        self.conn.commit()

    def add_review(
        self,
        from_id: int,
        to_id: int,
        decision: str,
        decided_by: str,
        confidence: float | None = None,
    ) -> None:
        if decision not in ("merge", "reject"):
            raise ValueError(f"bad decision: {decision!r}")
        self.conn.execute(
            "INSERT INTO concept_review (from_id, to_id, decision, decided_by, "
            "confidence, decided_at) VALUES (?, ?, ?, ?, ?, ?)",
            (from_id, to_id, decision, decided_by, confidence, now_iso()),
        )
        self.conn.commit()

    def rebuild_derived(self) -> None:
        """Rebuild ``concept_merge`` / ``concept_resolution`` from the log.

        Only ``merge`` decisions contribute.  ``reject`` is recorded but never
        merges.  Later decisions win, so the log stays append-only while a
        reviewer can change their mind.
        """
        rows = self.conn.execute(
            "SELECT from_id, to_id, decision FROM concept_review ORDER BY id"
        ).fetchall()
        self.conn.execute("DELETE FROM concept_merge")
        self.conn.execute("DELETE FROM concept_resolution")

        parent: dict[int, int] = {}

        def find(x: int) -> int:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        merged_pairs: set[tuple[int, int]] = set()
        for r in rows:
            if r["decision"] != "merge":
                continue
            parent.setdefault(r["from_id"], r["from_id"])
            parent.setdefault(r["to_id"], r["to_id"])
            a, b = find(r["from_id"]), find(r["to_id"])
            if a != b:
                parent[a] = b
            merged_pairs.add((r["from_id"], r["to_id"]))

        self.conn.executemany(
            "INSERT OR IGNORE INTO concept_merge (from_id, to_id) VALUES (?, ?)",
            sorted(merged_pairs),
        )
        for concept_id in parent:
            self.conn.execute(
                "INSERT INTO concept_resolution (concept_id, resolved_id) VALUES (?, ?)",
                (concept_id, find(concept_id)),
            )
        self.conn.commit()

    # --- embedding provenance + work queues --------------------------------

    def stamp_chunk_embedding(
        self, chunk_id: int, model_id: str, dimensions: int, config_version: str
    ) -> None:
        self.conn.execute(
            "UPDATE chunk SET embed_model_id = ?, dimensions = ?, "
            "embed_config_version = ?, embedded_at = ? WHERE id = ?",
            (model_id, dimensions, config_version, now_iso(), chunk_id),
        )
        self.conn.commit()

    def stamp_image_embedding(
        self, image_id: int, model_id: str, dimensions: int, config_version: str
    ) -> None:
        self.conn.execute(
            "UPDATE image SET embed_model_id = ?, dimensions = ?, "
            "embed_config_version = ?, embedded_at = ? WHERE id = ?",
            (model_id, dimensions, config_version, now_iso(), image_id),
        )
        self.conn.commit()

    def stamp_concept_embedding(
        self, concept_id: int, model_id: str, dimensions: int, config_version: str
    ) -> None:
        self.conn.execute(
            "UPDATE concept SET embed_model_id = ?, dimensions = ?, "
            "embed_config_version = ?, embedded_at = ? WHERE id = ?",
            (model_id, dimensions, config_version, now_iso(), concept_id),
        )
        self.conn.commit()

    def chunks_to_embed(self, model_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT id, text FROM chunk "
            "WHERE embed_model_id IS NULL OR embed_model_id != ? ORDER BY id",
            (model_id,),
        ).fetchall()

    def images_to_embed(self, model_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT id, path, mime FROM image "
            "WHERE embed_model_id IS NULL OR embed_model_id != ? ORDER BY id",
            (model_id,),
        ).fetchall()

    def concepts_to_embed(self, model_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT id, name FROM concept "
            "WHERE embed_model_id IS NULL OR embed_model_id != ? ORDER BY id",
            (model_id,),
        ).fetchall()

    # --- reporting ----------------------------------------------------------

    def stats(self) -> dict[str, int]:
        tables: Iterable[str] = (
            "book", "document_page", "chunk", "image", "concept",
            "concept_surface", "concept_mention", "concept_review",
        )
        out: dict[str, int] = {}
        for t in tables:
            out[t] = int(self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
        return out
