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
from typing import Any

from .schema import CHUNK_KINDS, CHUNKS_NAMESPACE

DISTANCE = "cosine"


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
        "SELECT id, text FROM chunk ORDER BY id" + (f" LIMIT {int(limit)}" if limit else "")
    ).fetchall()
    if chunks:
        vecs = embed_text([c["text"] for c in chunks])
        for c, v in zip(chunks, vecs, strict=True):
            rows.append(text_row(c["id"], v, c["text"]))

    concepts = conn.execute(
        "SELECT id, name FROM concept ORDER BY id"
        + (f" LIMIT {int(limit)}" if limit else "")
    ).fetchall()
    if concepts:
        vecs = embed_text([c["name"] for c in concepts])
        for c, v in zip(concepts, vecs, strict=True):
            rows.append(concept_row(c["id"], v, c["name"]))

    images = conn.execute(
        "SELECT id, path, mime FROM image ORDER BY id"
        + (f" LIMIT {int(limit)}" if limit else "")
    ).fetchall()
    if images:
        payload: list[tuple[str, bytes]] = []
        for im in images:
            payload.append((im["mime"], _read_blob(im["path"])))
        vecs = embed_images(payload)
        for im, v in zip(images, vecs, strict=True):
            rows.append(image_row(im["id"], v))

    return rows


def _read_blob(path: str) -> bytes:
    from pathlib import Path

    return Path(path).read_bytes()
