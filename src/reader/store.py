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

    def books_matching(
        self, ids: set[int] | None = None, work_keys: set[str] | None = None
    ) -> list[sqlite3.Row]:
        """Return books matching either selector, ordered by id."""
        clauses: list[str] = []
        params: list[object] = []
        if ids:
            clauses.append(f"id IN ({','.join('?' * len(ids))})")
            params.extend(sorted(ids))
        if work_keys:
            clauses.append(f"work_key IN ({','.join('?' * len(work_keys))})")
            params.extend(sorted(work_keys))
        if not clauses:
            return []
        return self.conn.execute(
            f"SELECT id, work_key, title, source_path FROM book "
            f"WHERE {' OR '.join(clauses)} ORDER BY id",
            tuple(params),
        ).fetchall()

    def delete_books(self, book_ids: set[int]) -> dict[str, object]:
        """Delete books and everything derived from them.

        SQLite is the system of record, so this is the authoritative removal
        path; the vector index is reconciled afterwards by ``index --optimize``
        (or by :meth:`reader.__main__.cmd_prune`).  Concepts shared with a kept
        book survive; concepts left with no mentions are removed along with
        their surfaces, review decisions and derived rows.
        """
        if not book_ids:
            return {"books": 0, "pages": 0, "chunks": 0, "images": 0, "concepts": 0, "paths": []}
        ids = sorted(
            int(r["id"])
            for r in self.conn.execute(
                f"SELECT id FROM book WHERE id IN ({','.join('?' * len(book_ids))})",
                tuple(sorted(book_ids)),
            )
        )
        if not ids:
            return {"books": 0, "pages": 0, "chunks": 0, "images": 0, "concepts": 0, "paths": []}
        q = ",".join("?" * len(ids))
        params = tuple(ids)
        page_ids = [
            int(r["id"])
            for r in self.conn.execute(
                f"SELECT id FROM document_page WHERE book_id IN ({q})", params
            )
        ]
        chunk_count = 0
        if page_ids:
            pq = ",".join("?" * len(page_ids))
            chunk_count = int(
                self.conn.execute(
                    f"SELECT COUNT(*) FROM chunk WHERE page_id IN ({pq})",
                    tuple(page_ids),
                ).fetchone()[0]
            )
        image_paths = [
            str(r["path"])
            for r in self.conn.execute(
                f"SELECT path FROM image WHERE book_id IN ({q})", params
            )
        ]
        concept_ids: set[int] = set()
        if page_ids:
            pq = ",".join("?" * len(page_ids))
            concept_ids = {
                int(r["concept_id"])
                for r in self.conn.execute(
                    f"SELECT DISTINCT concept_id FROM concept_mention "
                    f"WHERE page_id IN ({pq})",
                    tuple(page_ids),
                )
            }
        with self.conn:  # one transaction: a crash leaves the corpus unchanged
            if page_ids:
                pq = ",".join("?" * len(page_ids))
                self.conn.execute(
                    f"DELETE FROM concept_mention WHERE page_id IN ({pq})",
                    tuple(page_ids),
                )
                self.conn.execute(
                    f"DELETE FROM chunk WHERE page_id IN ({pq})", tuple(page_ids)
                )
                self.conn.execute(
                    f"DELETE FROM document_page WHERE id IN ({pq})", tuple(page_ids)
                )
            self.conn.execute(f"DELETE FROM image WHERE book_id IN ({q})", params)
            self.conn.execute(f"DELETE FROM book WHERE id IN ({q})", params)

            deleted_concepts = 0
            for cid in sorted(concept_ids):
                still_used = self.conn.execute(
                    "SELECT 1 FROM concept_mention WHERE concept_id = ? LIMIT 1", (cid,)
                ).fetchone()
                if still_used:
                    continue
                self.conn.execute("DELETE FROM concept_surface WHERE concept_id = ?", (cid,))
                self.conn.execute(
                    "DELETE FROM concept_review WHERE from_id = ? OR to_id = ?", (cid, cid)
                )
                self.conn.execute("DELETE FROM concept_merge WHERE from_id = ? OR to_id = ?", (cid, cid))
                self.conn.execute(
                    "DELETE FROM concept_resolution WHERE concept_id = ? OR resolved_id = ?",
                    (cid, cid),
                )
                self.conn.execute("DELETE FROM concept WHERE id = ?", (cid,))
                deleted_concepts += 1

        counts = {
            "books": len(ids),
            "pages": len(page_ids),
            "chunks": chunk_count,
            "images": len(image_paths),
            "concepts": deleted_concepts,
            "paths": image_paths,
        }
        self.rebuild_derived()
        self.conn.execute("VACUUM")
        return counts

    def upsert_page(
        self,
        book_id: int,
        page_no: int,
        text: str,
        source_sha256: str,
        scanned: bool = False,
        extractor_version: str = schema.EXTRACTOR_VERSION,
    ) -> int:
        row = self.conn.execute(
            "SELECT id, text, extractor_version FROM document_page "
            "WHERE book_id = ? AND page_no = ?",
            (book_id, page_no),
        ).fetchone()
        if row is not None:
            page_id = int(row["id"])
            # Unchanged text at the same extractor version invalidates nothing:
            # keep the chunks (and therefore their embedding provenance).
            if row["text"] == text and row["extractor_version"] == extractor_version:
                return page_id
            self.conn.execute(
                "UPDATE document_page SET text = ?, scanned = ?, "
                "extractor_version = ?, source_sha256 = ? WHERE id = ?",
                (text, int(scanned), extractor_version, source_sha256, page_id),
            )
            # The text changed: chunks cut from the old text are stale.
            self.conn.execute("DELETE FROM chunk WHERE page_id = ?", (page_id,))
            self.conn.commit()
            return page_id

        cur = self.conn.execute(
            """
            INSERT INTO document_page (book_id, page_no, text, scanned,
                                       extractor_version, source_sha256)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (book_id, page_no, text, int(scanned), extractor_version, source_sha256),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def set_chunks(self, page_id: int, chunks: Sequence[tuple[int, int, int, str]]) -> None:
        """Replace a page's chunks.  ``chunks`` = (ordinal, start, end, text).

        Provenance is carried forward for a chunk whose (start, end, text) is
        identical to the one it replaces, so a re-chunk of unchanged text does
        not force a re-embed.  This is the expensive invariant: new ids for the
        same bytes must not look like new work.
        """
        old = {
            (r["start_offset"], r["end_offset"], r["text"]): (
                r["embed_model_id"], r["dimensions"],
                r["embed_config_version"], r["embedded_at"],
            )
            for r in self.conn.execute(
                "SELECT start_offset, end_offset, text, embed_model_id, dimensions, "
                "embed_config_version, embedded_at FROM chunk WHERE page_id = ?",
                (page_id,),
            )
        }
        self.conn.execute("DELETE FROM chunk WHERE page_id = ?", (page_id,))
        rows = []
        for ordinal, start, end, text in chunks:
            prov = old.get((start, end, text), (None, None, None, None))
            rows.append((page_id, ordinal, start, end, text, *prov))
        self.conn.executemany(
            "INSERT INTO chunk (page_id, ordinal, start_offset, end_offset, text, "
            "embed_model_id, dimensions, embed_config_version, embedded_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
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

    def ensure_concept(self, name: str, type_: str = "term", description: str | None = None) -> int:
        self.conn.execute(
            "INSERT INTO concept (name, type, description) VALUES (?, ?, ?) "
            "ON CONFLICT(name, type) DO UPDATE SET "
            "description = COALESCE(concept.description, excluded.description)",
            (name, type_, description),
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
            "INSERT OR IGNORE INTO concept_mention (concept_id, surface_id, page_id, "
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
        # Carry the page/book columns the index stores, so a query can scope by
        # book_id/page_id without a round-trip to SQLite.
        return self.conn.execute(
            "SELECT c.id, c.text, c.page_id, p.book_id FROM chunk c "
            "JOIN document_page p ON p.id = c.page_id "
            "WHERE c.embed_model_id IS NULL OR c.embed_model_id != ? ORDER BY c.id",
            (model_id,),
        ).fetchall()

    def images_to_embed(self, model_id: str) -> list[sqlite3.Row]:
        # image.page_no maps to a document_page, so resolve page_id when the
        # book is known; book_id/page_id stay NULL when the book is unknown.
        return self.conn.execute(
            "SELECT i.id, i.path, i.mime, i.w, i.h, i.book_id, p.id AS page_id "
            "FROM image i "
            "LEFT JOIN document_page p ON p.book_id = i.book_id AND p.page_no = i.page_no "
            "WHERE i.embed_model_id IS NULL OR i.embed_model_id != ? ORDER BY i.id",
            (model_id,),
        ).fetchall()

    def index_keep_ids(self) -> dict[str, set[int]]:
        """Record ids the index is allowed to hold, per kind.

        Fed to ``LanceIndex.prune`` so vectors for deleted records are
        dropped.  Every row counts here - not just embedded ones - because a
        record awaiting re-embedding still legitimately owns its index row.
        """
        return {
            "text": {int(r["id"]) for r in self.conn.execute("SELECT id FROM chunk")},
            "concept": {int(r["id"]) for r in self.conn.execute("SELECT id FROM concept")},
            "image": {int(r["id"]) for r in self.conn.execute("SELECT id FROM image")},
        }

    def concepts_to_embed(self, model_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT id, name, description FROM concept "
            "WHERE embed_model_id IS NULL OR embed_model_id != ? ORDER BY id",
            (model_id,),
        ).fetchall()

    # --- per-book stage checkpoints ----------------------------------------

    def _book_stamp(self, work_key: str, column: str) -> sqlite3.Row | None:
        return self.conn.execute(
            f"SELECT id, source_sha256, {column} FROM book WHERE work_key = ?", (work_key,)
        ).fetchone()

    def book_extraction_current(self, work_key: str, sha256: str) -> bool:
        """True when this source file was fully extracted by this extractor."""
        row = self._book_stamp(work_key, "extracted_source_sha256")
        return bool(
            row
            and row["extracted_source_sha256"] == sha256
            and self.conn.execute(
                "SELECT extractor_version FROM book WHERE id = ?", (row["id"],)
            ).fetchone()[0] == schema.EXTRACTOR_VERSION
        )

    def mark_book_extracted(self, book_id: int, sha256: str) -> None:
        self.conn.execute(
            "UPDATE book SET extracted_source_sha256 = ?, extractor_version = ?, "
            "extracted_at = ? WHERE id = ?",
            (sha256, schema.EXTRACTOR_VERSION, now_iso(), book_id),
        )
        self.conn.commit()

    def book_images_current(self, work_key: str, sha256: str) -> bool:
        row = self._book_stamp(work_key, "images_source_sha256")
        return bool(row and row["images_source_sha256"] == sha256)

    def mark_book_images(self, book_id: int, sha256: str) -> None:
        self.conn.execute(
            "UPDATE book SET images_source_sha256 = ?, images_extracted_at = ? WHERE id = ?",
            (sha256, now_iso(), book_id),
        )
        self.conn.commit()

    def book_concepts_current(self, work_key: str, sha256: str) -> bool:
        row = self._book_stamp(work_key, "concepts_source_sha256")
        return bool(row and row["concepts_source_sha256"] == sha256)

    def mark_book_concepts(self, book_id: int, sha256: str) -> None:
        self.conn.execute(
            "UPDATE book SET concepts_source_sha256 = ?, concepts_extracted_at = ? "
            "WHERE id = ?",
            (sha256, now_iso(), book_id),
        )
        self.conn.commit()

    # --- spend ledger -------------------------------------------------------

    def record_spend(
        self, run_id: int | None, kind: str, model: str, items: int,
        tokens: int, usd: float, measured: bool,
    ) -> None:
        self.conn.execute(
            "INSERT INTO spend (run_id, stage, kind, model, items, tokens, usd, "
            "measured, created_at) VALUES (?, 'embed', ?, ?, ?, ?, ?, ?, ?)",
            (run_id, kind, model, items, tokens, usd, int(measured), now_iso()),
        )
        self.conn.commit()

    def spend_for_run(self, run_id: int) -> float:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(usd), 0) FROM spend WHERE run_id = ?", (run_id,)
        ).fetchone()
        return float(row[0])

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
