"""The ONLY module that talks to Turbopuffer.

Turbopuffer stores **supplied** vectors and never embeds: every vector written
here was computed by :mod:`reader.embed` through the LiteLLM gateway.  One
``chunks`` namespace holds all three kinds (``text``, ``concept``, ``image``);
``text`` and ``concept`` carry the text attribute that BM25 reads, ``image``
contributes on the vector side only.

The index is always rebuildable from SQLite.  If the two disagree, SQLite wins:
:meth:`Index.rebuild` re-derives every row from the system of record.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .schema import CHUNK_KINDS, CHUNKS_NAMESPACE

DISTANCE = "cosine"

#: Backend names accepted by :func:`create_index`.  ``local``/``lancedb`` are
#: the same embedded backend; ``turbopuffer`` is the hosted fallback.
LOCAL_BACKENDS = frozenset({"local", "lancedb"})
TURBOPUFFER_BACKEND = "turbopuffer"
DEFAULT_BACKEND = "lancedb"

#: Above this many rows we build an ANN index; below it LanceDB's flat scan is
#: faster and avoids training a tiny index.
ANN_INDEX_THRESHOLD = 256


def text_id(chunk_id: int) -> str:
    return f"text:{chunk_id}"


def image_id(image_id: int) -> str:
    return f"image:{image_id}"


def concept_id(concept_id: int) -> str:
    return f"concept:{concept_id}"


def _row(id_: str, kind: str, vector: Sequence[float], **attrs: Any) -> dict[str, Any]:
    if kind not in CHUNK_KINDS:
        raise ValueError(f"bad kind: {kind!r}")
    row: dict[str, Any] = {"id": id_, "vector": list(vector), "kind": kind}
    row.update(attrs)
    return row


def text_row(chunk_id: int, vector: Sequence[float], text: str, **attrs: Any) -> dict[str, Any]:
    return _row(text_id(chunk_id), "text", vector, text=text, **attrs)


def image_row(image_id_: int, vector: Sequence[float], **attrs: Any) -> dict[str, Any]:
    # No text attribute: images do not participate in BM25.
    return _row(image_id(image_id_), "image", vector, **attrs)


def concept_row(concept_id_: int, vector: Sequence[float], text: str, **attrs: Any) -> dict[str, Any]:
    return _row(concept_id(concept_id_), "concept", vector, text=text, **attrs)


class DimensionalityNotPinned(RuntimeError):
    """Raised when a write is attempted before ``eval`` pinned a dimensionality."""


class Index:
    """Thin wrapper over the ``chunks`` namespace.

    The Turbopuffer client is injected (``client``) so tests never need the
    network or the SDK's exact surface; when omitted it is constructed lazily.
    """

    def __init__(
        self,
        api_key: str | None,
        dimensions: int | None = None,
        namespace: str = CHUNKS_NAMESPACE,
        client: Any | None = None,
    ) -> None:
        self.api_key = api_key
        self.dimensions = dimensions
        self.namespace = namespace
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            import turbopuffer  # lazy: only needed on the network path

            self._client = turbopuffer.Turbopuffer(api_key=self.api_key)
        return self._client

    @property
    def ns(self) -> Any:
        return self.client.namespace(self.namespace)

    def upsert(self, rows: Sequence[dict[str, Any]]) -> Any:
        """Upsert supplied-vector rows.  Dimensionality must be pinned first."""
        if self.dimensions is None:
            raise DimensionalityNotPinned(
                "pin embedding_dimensions (run `eval`) before the first write"
            )
        for row in rows:
            if len(row.get("vector", [])) != self.dimensions:
                raise ValueError(
                    f"row {row.get('id')!r} has {len(row.get('vector', []))} dims, "
                    f"expected {self.dimensions}"
                )
        if not rows:
            return None
        return self.ns.write(upsert_rows=list(rows), distance_metric=DISTANCE)

    def query(self, **kwargs: Any) -> Any:
        """Raw passthrough to the SDK's ``query``."""
        return self.ns.query(**kwargs)

    def hybrid_query(
        self,
        vector: Sequence[float] | None = None,
        bm25_text: str | None = None,
        top_k: int = 10,
        filters: Any | None = None,
        kinds: Sequence[str] | None = None,
    ) -> Any:
        """Hybrid vector + BM25 retrieval with server-side RRF.

        The vector side and the BM25 side are issued as one multi-query so the
        server fuses them; images contribute only on the vector side.
        """
        rank_bys: list[Any] = []
        if vector is not None:
            rank_bys.append(("vector", "ANN", list(vector)))
        if bm25_text is not None:
            rank_bys.append(("text", "BM25", bm25_text))
        flt: list[Any] = []
        if kinds:
            flt.append(("kind", "In", list(kinds)))
        if filters:
            flt.append(filters)
        return self.query(
            rank_by=rank_bys if len(rank_bys) != 1 else rank_bys[0],
            top_k=top_k,
            filters=flt or None,
            include_attributes=["kind", "text"],
            distance_metric=DISTANCE,
        )

    def rebuild(self, rows: Sequence[dict[str, Any]]) -> Any:
        """Rewrite the index from rows re-derived from SQLite (SQLite wins)."""
        return self.upsert(rows)


def rebuild_rows(
    store: Any,
    embed_text: Callable[[list[str]], list[list[float]]],
    embed_images: Callable[[list[tuple[str, bytes]]], list[list[float]]],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Re-derive every index row from the system of record.

    Text and concept rows carry their text; image rows carry only a vector.
    Vectors are re-computed through the supplied embedders (SQLite is the
    record; Turbopuffer holds nothing we cannot rebuild).
    """
    conn = store.conn
    rows: list[dict[str, Any]] = []

    chunks = conn.execute(
        "SELECT c.id, c.text, c.page_id, p.book_id FROM chunk c "
        "JOIN document_page p ON p.id = c.page_id ORDER BY c.id"
        + (f" LIMIT {int(limit)}" if limit else "")
    ).fetchall()
    if chunks:
        vecs = embed_text([c["text"] for c in chunks])
        for c, v in zip(chunks, vecs, strict=True):
            rows.append(text_row(c["id"], v, c["text"], book_id=c["book_id"], page_id=c["page_id"]))

    concepts = conn.execute(
        "SELECT id, name FROM concept ORDER BY id"
        + (f" LIMIT {int(limit)}" if limit else "")
    ).fetchall()
    if concepts:
        vecs = embed_text([c["name"] for c in concepts])
        for c, v in zip(concepts, vecs, strict=True):
            rows.append(concept_row(c["id"], v, c["name"]))

    images = conn.execute(
        "SELECT i.id, i.path, i.mime, i.book_id, p.id AS page_id FROM image i "
        "LEFT JOIN document_page p ON p.book_id = i.book_id AND p.page_no = i.page_no "
        "ORDER BY i.id" + (f" LIMIT {int(limit)}" if limit else "")
    ).fetchall()
    if images:
        payload: list[tuple[str, bytes]] = []
        for im in images:
            payload.append((im["mime"], _read_blob(im["path"])))
        vecs = embed_images(payload)
        for im, v in zip(images, vecs, strict=True):
            rows.append(image_row(im["id"], v, book_id=im["book_id"], page_id=im["page_id"]))

    return rows


def _read_blob(path: str) -> bytes:
    from pathlib import Path

    return Path(path).read_bytes()


# --------------------------------------------------------------------------- #
# LanceDB backend (the local index)
# --------------------------------------------------------------------------- #

#: The Lance table columns.  Fixed on purpose: a stable Arrow schema keeps
#: every ``upsert`` (and the create-then-merge path) consistent, and mirrors
#: v5's single-namespace design — one table, a ``kind`` column.
_LANCE_COLUMNS = ("id", "vector", "kind", "text", "book_id", "page_id")


class LanceIndex:
    """Embedded LanceDB index — the ``local`` backend.

    No service, no secret: the index is a directory next to the SQLite record.
    Vectors are *supplied* (LanceDB never embeds).  Hybrid retrieval is native
    (BM25 + ANN + RRF), and the whole index is rebuildable from SQLite.
    """

    def __init__(
        self,
        path: Path | str,
        dimensions: int | None = None,
        table: str = CHUNKS_NAMESPACE,
    ) -> None:
        self.path = Path(path)
        self.dimensions = dimensions
        self.table_name = table
        self._db: Any | None = None

    # --- connection / schema -------------------------------------------------

    @property
    def db(self) -> Any:
        if self._db is None:
            import lancedb

            self.path.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(str(self.path))
        return self._db

    def _arrow_schema(self) -> Any:
        import pyarrow as pa

        if self.dimensions is None:
            raise DimensionalityNotPinned(
                "pin embedding_dimensions (run `eval --pin`) before opening the index"
            )
        return pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self.dimensions)),
                pa.field("kind", pa.string()),
                pa.field("text", pa.string()),
                pa.field("book_id", pa.int64()),
                pa.field("page_id", pa.int64()),
            ]
        )

    def _table(self) -> Any | None:
        try:
            return self.db.open_table(self.table_name)
        except Exception:  # noqa: BLE001 - table simply does not exist yet
            return None

    def _records(self, rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        recs: list[dict[str, Any]] = []
        for r in rows:
            vec = r.get("vector", [])
            if len(vec) != self.dimensions:
                raise ValueError(
                    f"row {r.get('id')!r} has {len(vec)} dims, expected {self.dimensions}"
                )
            kind = r.get("kind")
            if kind not in CHUNK_KINDS:
                raise ValueError(f"bad kind: {kind!r}")
            recs.append(
                {
                    "id": str(r["id"]),
                    "vector": [float(x) for x in vec],
                    "kind": kind,
                    "text": r.get("text"),
                    "book_id": r.get("book_id"),
                    "page_id": r.get("page_id"),
                }
            )
        return recs

    def _ensure_indices(self, tbl: Any) -> None:
        names = [i.name for i in tbl.list_indices()]
        if "text_idx" not in names:
            from lancedb.index import FTS

            tbl.create_index("text", config=FTS(), replace=True, name="text_idx")
        if tbl.count_rows() >= ANN_INDEX_THRESHOLD and "vector_idx" not in names:
            tbl.create_index(metric=DISTANCE, vector_column_name="vector", name="vector_idx")

    # --- write path ----------------------------------------------------------

    def upsert(self, rows: Sequence[dict[str, Any]]) -> int:
        if self.dimensions is None:
            raise DimensionalityNotPinned("pin embedding_dimensions before the first write")
        recs = self._records(rows)
        if not recs:
            return 0
        import pyarrow as pa

        data = pa.Table.from_pylist(recs, schema=self._arrow_schema())
        tbl = self._table()
        if tbl is None:
            tbl = self.db.create_table(self.table_name, data=data)
        else:
            (
                tbl.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(data)
            )
        self._ensure_indices(tbl)
        return len(recs)

    def optimize(self, cleanup_older_than_days: int | None = None) -> None:
        """Compact versions and rebuild stale indices (index housekeeping)."""
        tbl = self._table()
        if tbl is None:
            return
        kwargs = {}
        if cleanup_older_than_days is not None:
            from datetime import timedelta

            kwargs["cleanup_older_than"] = timedelta(days=cleanup_older_than_days)
        tbl.optimize(**kwargs)

    def count(self) -> int:
        tbl = self._table()
        return tbl.count_rows() if tbl is not None else 0

    def drop(self) -> None:
        try:
            self.db.drop_table(self.table_name)
        except Exception:  # noqa: BLE001 - nothing to drop
            pass

    # --- read path -----------------------------------------------------------

    @staticmethod
    def _where(kinds: Sequence[str] | None, filters: Any | None) -> str:
        parts: list[str] = []
        if kinds:
            literals = ",".join("'" + k.replace("'", "''") + "'" for k in kinds)
            parts.append(f"kind IN ({literals})")
        if isinstance(filters, str) and filters:
            parts.append(f"({filters})")
        elif isinstance(filters, dict):
            for key, value in filters.items():
                parts.append(f"{key} = '{value}'" if isinstance(value, str) else f"{key} = {value}")
        return " AND ".join(parts)

    def query(
        self,
        *,
        vector: Sequence[float] | None = None,
        bm25_text: str | None = None,
        top_k: int = 10,
        kinds: Sequence[str] | None = None,
        filters: Any | None = None,
    ) -> list[dict[str, Any]]:
        return self.hybrid_query(
            vector=vector, bm25_text=bm25_text, top_k=top_k, kinds=kinds, filters=filters
        )

    def hybrid_query(
        self,
        vector: Sequence[float] | None = None,
        bm25_text: str | None = None,
        top_k: int = 10,
        filters: Any | None = None,
        kinds: Sequence[str] | None = None,
        *,
        rerank: bool = True,
    ) -> list[dict[str, Any]]:
        """BM25 + ANN with native RRF fusion, in one table via the `kind` column."""
        tbl = self._table()
        if tbl is None:
            return []
        if vector is not None and bm25_text:
            from lancedb.rerankers import RRFReranker

            q = (
                tbl.search(
                    vector_column_name="vector",
                    query_type="hybrid",
                    fts_columns="text",
                )
                .vector([float(x) for x in vector])
                .text(bm25_text)
            )
            if rerank:
                q = q.rerank(RRFReranker())
        elif vector is not None:
            q = tbl.search([float(x) for x in vector], vector_column_name="vector")
        elif bm25_text:
            q = tbl.search(bm25_text, query_type="fts", fts_columns="text")
        else:
            return []
        where = self._where(kinds, filters)
        if where:
            q = q.where(where)
        return q.limit(top_k).to_list()


def create_index(
    backend: str | None,
    *,
    dimensions: int | None = None,
    api_key: str | None = None,
    index_dir: Path | str | None = None,
    namespace: str = CHUNKS_NAMESPACE,
) -> LanceIndex | Index:
    """Open the configured backend behind one surface.

    ``local``/``lancedb`` -> :class:`LanceIndex` (embedded); ``turbopuffer`` ->
    :class:`Index` (hosted fallback).  The rest of ``reader`` never learns which.
    """
    name = (backend or DEFAULT_BACKEND).lower()
    if name in LOCAL_BACKENDS:
        if index_dir is None:
            raise ValueError("index_dir is required for the local (LanceDB) backend")
        return LanceIndex(index_dir, dimensions, table=namespace)
    if name == TURBOPUFFER_BACKEND:
        return Index(api_key, dimensions, namespace)
    raise ValueError(f"unknown INDEX_BACKEND: {backend!r}")
