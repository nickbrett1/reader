"""Schema constants shared by BOTH the ``reader`` and ``gallery`` repos.

This module has no imports from the rest of the package on purpose: the
``gallery`` repo reads the same constants (its own vendored copy) and both
repos check :data:`SCHEMA_VERSION` against the ``schema_version`` table before
they touch the database.  Bumping the version here is a deliberate act that
requires a migration in :mod:`reader.store`.
"""

from __future__ import annotations

# --- versions ---------------------------------------------------------------

#: Version of the SQLite schema.  ``gallery`` refuses to read a database whose
#: recorded version does not match the one it was built against.
SCHEMA_VERSION = 4

#: Version of the extraction logic.  Stamped onto every ``document_page`` and
#: ``image`` row so a re-extract can be scoped to rows produced by an older
#: extractor instead of re-processing the whole corpus.
EXTRACTOR_VERSION = "1"

#: Version of the quote-selection logic.  Stamped on every quote and on the
#: book checkpoint, so a prompt/filter change scopes itself to old rows.
QUOTES_VERSION = "1"

#: Version of the embedding *configuration* (model family + prompt/template
#: choices).  Stamped onto every vector's provenance so a model swap can be
#: scoped per row.  Bumped "1" -> "2" when client-side image resizing became
#: part of the config: rows stamped "1" were embedded at full size (Cohere
#: clamped anything over ~2.4 Mpx server-side), rows stamped "2" are embedded
#: at :data:`IMAGE_MAX_PIXELS` or below.
EMBED_CONFIG_VERSION = "2"

#: Client-side downscale cap for images before they are wrapped as a data URI.
#: Matches Cohere's own cap: image tokens are area-driven
#: (``ceil(w/32)*ceil(h/32)+2``) and Cohere downscales anything over ~2.4 Mpx
#: itself, so uploading more only inflates the payload without changing the
#: vector.  2,400,000 px = 2.4 Mpx.
IMAGE_MAX_PIXELS = 2_400_000

#: Turbopuffer namespace holding the derived index.  There is exactly one.
CHUNKS_NAMESPACE = "chunks"

#: The three kinds of row that live in the ``chunks`` namespace.  ``text`` and
#: ``concept`` contribute to BM25; ``image`` contributes on the vector side
#: only.
CHUNK_KINDS = ("text", "concept", "image")

#: Candidate dimensionalities.  ``eval`` decides which one is pinned; the
#: choice is recorded in ``schema_version`` *before the first write*.
CANDIDATE_DIMENSIONS = (2048, 1536)

#: Cohere list price for image tokens (USD per 1M tokens) used for off-gateway
#: budgeting.  The gateway under-bills image tokens and reports 0, so we never
#: trust its accounting for images.
COHERE_IMAGE_USD_PER_MTOK = 0.40

#: Text embedding list price, USD per 1M tokens. Used only for the local spend
#: estimate and the ``--max-usd`` guard; the gateway's own usage is preferred
#: when it reports tokens.
COHERE_TEXT_USD_PER_MTOK = 0.12

# --- DDL --------------------------------------------------------------------

# Applied in order; ``store.apply_migrations`` records each version it reaches.
MIGRATIONS: dict[int, str] = {
    1: """
    -- Recorded provenance of the database itself.  Both repos read the single
    -- row and compare ``version`` to their compiled-in SCHEMA_VERSION.
    CREATE TABLE IF NOT EXISTS schema_version (
        version              INTEGER NOT NULL,
        applied_at           TEXT    NOT NULL,
        -- Pinned by `eval` before the first vector is written; NULL until then.
        embedding_dimensions INTEGER,
        embedding_model      TEXT,
        embed_config_version TEXT
    );

    -- A work is one book; a work may have several source files but only one
    -- format (enforced upstream in manifest.py).
    CREATE TABLE IF NOT EXISTS book (
        id          INTEGER PRIMARY KEY,
        work_key    TEXT    NOT NULL UNIQUE,
        title       TEXT,
        author      TEXT,
        format      TEXT    NOT NULL,
        source_path TEXT    NOT NULL,
        source_sha256 TEXT  NOT NULL,
        added_at    TEXT    NOT NULL
    );

    -- Ingest state + log, one row per CLI invocation.
    CREATE TABLE IF NOT EXISTS run (
        id          INTEGER PRIMARY KEY,
        verb        TEXT    NOT NULL,
        started_at  TEXT    NOT NULL,
        finished_at TEXT,
        status      TEXT    NOT NULL DEFAULT 'running',
        log         TEXT    NOT NULL DEFAULT ''
    );

    -- Canonical extracted text.  Chunks are OFFSET RANGES into this text; the
    -- text is never duplicated into a chunk's own copy on disk.
    CREATE TABLE IF NOT EXISTS document_page (
        id                INTEGER PRIMARY KEY,
        book_id           INTEGER NOT NULL REFERENCES book(id),
        page_no           INTEGER NOT NULL,
        text              TEXT    NOT NULL DEFAULT '',
        scanned           INTEGER NOT NULL DEFAULT 0,
        extractor_version TEXT    NOT NULL,
        source_sha256     TEXT    NOT NULL,
        UNIQUE (book_id, page_no)
    );

    -- A contiguous, non-overlapping slice of document_page.text.  Because
    -- ranges tile the page, concatenating text[start:end] over a page
    -- reconstructs the page exactly.
    CREATE TABLE IF NOT EXISTS chunk (
        id                   INTEGER PRIMARY KEY,
        page_id              INTEGER NOT NULL REFERENCES document_page(id),
        ordinal              INTEGER NOT NULL,
        start_offset         INTEGER NOT NULL,
        end_offset           INTEGER NOT NULL,
        text                 TEXT    NOT NULL,
        -- Per-row embedding provenance.
        embed_model_id       TEXT,
        dimensions           INTEGER,
        embed_config_version TEXT,
        embedded_at          TEXT,
        UNIQUE (page_id, ordinal)
    );

    -- The image manifest.  Bytes live content-addressed on disk at
    -- images/<sha256[0:2]>/<sha256>.<ext>; only a small thumbnail is inline.
    CREATE TABLE IF NOT EXISTS image (
        id                INTEGER PRIMARY KEY,
        sha256            TEXT    NOT NULL UNIQUE,
        path              TEXT    NOT NULL,
        phash             TEXT,
        w                 INTEGER,
        h                 INTEGER,
        mime              TEXT    NOT NULL,
        bytes             INTEGER NOT NULL,
        book_id           INTEGER REFERENCES book(id),
        page_no           INTEGER,
        kind              TEXT    NOT NULL CHECK (kind IN ('embedded','rendered')),
        extractor_version TEXT    NOT NULL,
        source_sha256     TEXT    NOT NULL,
        caption           TEXT,
        thumb             BLOB,
        embed_model_id       TEXT,
        dimensions           INTEGER,
        embed_config_version TEXT,
        embedded_at          TEXT
    );

    CREATE TABLE IF NOT EXISTS concept (
        id       INTEGER PRIMARY KEY,
        name     TEXT    NOT NULL,
        type     TEXT    NOT NULL DEFAULT 'term',
        embed_model_id       TEXT,
        dimensions           INTEGER,
        embed_config_version TEXT,
        embedded_at          TEXT,
        UNIQUE (name, type)
    );

    -- Every surface form a concept was seen as.
    CREATE TABLE IF NOT EXISTS concept_surface (
        id         INTEGER PRIMARY KEY,
        concept_id INTEGER NOT NULL REFERENCES concept(id),
        surface    TEXT    NOT NULL,
        UNIQUE (concept_id, surface)
    );

    -- Where a surface form occurred in canonical text.
    CREATE TABLE IF NOT EXISTS concept_mention (
        id           INTEGER PRIMARY KEY,
        concept_id   INTEGER NOT NULL REFERENCES concept(id),
        surface_id   INTEGER NOT NULL REFERENCES concept_surface(id),
        page_id      INTEGER NOT NULL REFERENCES document_page(id),
        start_offset INTEGER NOT NULL,
        end_offset   INTEGER NOT NULL
    );

    -- APPEND-ONLY review log.  Nothing updates or deletes a row here; the
    -- derived concept_merge / concept_resolution tables are rebuilt from it.
    CREATE TABLE IF NOT EXISTS concept_review (
        id         INTEGER PRIMARY KEY,
        from_id    INTEGER NOT NULL,
        to_id      INTEGER NOT NULL,
        decision   TEXT    NOT NULL CHECK (decision IN ('merge','reject')),
        decided_by TEXT    NOT NULL,
        confidence REAL,
        decided_at TEXT    NOT NULL
    );

    -- DERIVED from concept_review (rebuildable, never hand-written).
    CREATE TABLE IF NOT EXISTS concept_merge (
        from_id INTEGER NOT NULL,
        to_id   INTEGER NOT NULL,
        PRIMARY KEY (from_id, to_id)
    );

    -- DERIVED: every concept id mapped to its surviving representative.
    CREATE TABLE IF NOT EXISTS concept_resolution (
        concept_id   INTEGER PRIMARY KEY,
        resolved_id  INTEGER NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_page_book ON document_page(book_id);
    CREATE INDEX IF NOT EXISTS idx_chunk_page ON chunk(page_id);
    CREATE INDEX IF NOT EXISTS idx_image_book ON image(book_id);
    CREATE INDEX IF NOT EXISTS idx_mention_concept ON concept_mention(concept_id);
    CREATE INDEX IF NOT EXISTS idx_mention_page ON concept_mention(page_id);
    CREATE INDEX IF NOT EXISTS idx_review_from ON concept_review(from_id);
    """,
    # v2 — a concept's optional description.  §6.2.2 embeds
    # ``canonical_name + description``, not the bare name: a single short token
    # is too flat a signal to place in the vector space (see the 2026-10-12
    # calibration dry-run).
    2: """
    ALTER TABLE concept ADD COLUMN description TEXT;
    """,
    # v3 — paid-work checkpoints.  A stage may skip a book only when the
    # columns below say the work was completed for the *current* source file.
    # deleted_source_sha256 is deliberately NOT backfilled: the first run after
    # this migration re-extracts once, but upsert_page now preserves chunks when
    # the text is unchanged, so that pass costs CPU and not an embedding call.
    3: """
    ALTER TABLE book ADD COLUMN extracted_source_sha256 TEXT;
    ALTER TABLE book ADD COLUMN extractor_version       TEXT;
    ALTER TABLE book ADD COLUMN extracted_at            TEXT;
    ALTER TABLE book ADD COLUMN images_source_sha256    TEXT;
    ALTER TABLE book ADD COLUMN images_extracted_at     TEXT;
    ALTER TABLE book ADD COLUMN concepts_source_sha256  TEXT;
    ALTER TABLE book ADD COLUMN concepts_extracted_at   TEXT;

    -- A concept mention is identified by (concept, page, offsets).  Collapse
    -- any duplicates left by pre-v3 reruns before enforcing that.
    DELETE FROM concept_mention WHERE id NOT IN (
        SELECT MIN(id) FROM concept_mention
        GROUP BY concept_id, page_id, start_offset, end_offset
    );
    CREATE UNIQUE INDEX IF NOT EXISTS uq_mention
        ON concept_mention (concept_id, page_id, start_offset, end_offset);

    -- Books already carrying mentions were concept-extracted at least once;
    -- mark them done so the first post-migration run does not re-pay for them.
    UPDATE book
       SET concepts_source_sha256 = source_sha256,
           concepts_extracted_at  = added_at
     WHERE EXISTS (
        SELECT 1 FROM concept_mention m
          JOIN document_page p ON p.id = m.page_id
         WHERE p.book_id = book.id
     );

    -- Out-of-band spend ledger.  The gateway reports 0 for image tokens, so
    -- ``measured`` says whether ``tokens`` came from the gateway or an estimate.
    CREATE TABLE IF NOT EXISTS spend (
        id         INTEGER PRIMARY KEY,
        run_id     INTEGER,
        stage      TEXT    NOT NULL,
        kind       TEXT    NOT NULL,
        model      TEXT    NOT NULL,
        items      INTEGER NOT NULL,
        tokens     INTEGER NOT NULL DEFAULT 0,
        usd        REAL    NOT NULL DEFAULT 0,
        measured   INTEGER NOT NULL DEFAULT 0,
        created_at TEXT    NOT NULL
    );
    """,
    # v4 — quotes: verbatim sentence spans, linked to concepts.  A quote is a
    # range into document_page.text exactly like a chunk; the copy is verified
    # against the page at insert, so it cannot drift.  A quote carries no
    # vector of its own: it is already embedded as part of a chunk.
    4: """
    ALTER TABLE book ADD COLUMN quotes_source_sha256   TEXT;
    ALTER TABLE book ADD COLUMN quotes_extractor_version TEXT;
    ALTER TABLE book ADD COLUMN quotes_extracted_at    TEXT;

    CREATE TABLE IF NOT EXISTS quote (
        id                INTEGER PRIMARY KEY,
        page_id           INTEGER NOT NULL REFERENCES document_page(id),
        start_offset      INTEGER NOT NULL,
        end_offset        INTEGER NOT NULL,
        text              TEXT    NOT NULL,
        score             REAL    NOT NULL,
        kind              TEXT    NOT NULL,
        rationale         TEXT,
        extractor_version TEXT    NOT NULL,
        model_id          TEXT    NOT NULL DEFAULT '',
        source_sha256     TEXT    NOT NULL,
        created_at        TEXT    NOT NULL DEFAULT (datetime('now')),
        UNIQUE (page_id, start_offset, end_offset)
    );
    CREATE INDEX IF NOT EXISTS quote_page ON quote(page_id);
    CREATE INDEX IF NOT EXISTS quote_rank ON quote(score DESC, id);

    CREATE TABLE IF NOT EXISTS quote_concept (
        quote_id    INTEGER NOT NULL REFERENCES quote(id),
        concept_id  INTEGER NOT NULL REFERENCES concept(id),
        confidence  REAL    NOT NULL,
        decided_by  TEXT    NOT NULL DEFAULT 'llm',
        created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
        PRIMARY KEY (quote_id, concept_id)
    );
    CREATE INDEX IF NOT EXISTS quote_concept_concept
        ON quote_concept(concept_id, confidence DESC);

    -- APPEND-ONLY review log.  Display is derived: a quote is shown when it
    -- has no 'reject' row and its score clears the configured threshold.
    CREATE TABLE IF NOT EXISTS quote_review (
        quote_id   INTEGER NOT NULL,
        decision   TEXT    NOT NULL CHECK (decision IN ('approve','reject','demote')),
        decided_by TEXT    NOT NULL,
        note       TEXT,
        request_id TEXT,
        decided_at TEXT    NOT NULL DEFAULT (datetime('now'))
    );
    """,
}
