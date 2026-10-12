"""LanceDB backend: upsert/merge, vector, BM25, hybrid RRF, kind filter."""

from __future__ import annotations

import pytest

from reader.index import DimensionalityNotPinned, Index, LanceIndex, create_index


def _rows():
    return [
        {"id": "text:1", "vector": [1.0, 0.0, 0.0, 0.0], "kind": "text",
         "text": "Emberfall is a region of magic swords", "book_id": 1, "page_id": 1},
        {"id": "text:2", "vector": [0.0, 1.0, 0.0, 0.0], "kind": "text",
         "text": "The inventory screen shows held items", "book_id": 1, "page_id": 2},
        {"id": "concept:1", "vector": [0.9, 0.1, 0.0, 0.0], "kind": "concept",
         "text": "Magic Sword", "book_id": None, "page_id": None},
        {"id": "image:1", "vector": [0.0, 0.0, 1.0, 0.0], "kind": "image", "text": None},
    ]


def test_upsert_then_reupsert_does_not_duplicate(tmp_path):
    idx = LanceIndex(tmp_path / "idx", dimensions=4)
    assert idx.upsert(_rows()) == 4
    assert idx.count() == 4
    idx.upsert(_rows())  # same ids -> merge, not duplicate
    assert idx.count() == 4


def test_vector_query(tmp_path):
    idx = LanceIndex(tmp_path / "idx", 4)
    idx.upsert(_rows())
    res = idx.hybrid_query(vector=[1.0, 0.0, 0.0, 0.0], top_k=2)
    assert res[0]["id"] == "text:1"


def test_bm25_query(tmp_path):
    idx = LanceIndex(tmp_path / "idx", 4)
    idx.upsert(_rows())
    res = idx.hybrid_query(bm25_text="inventory", top_k=3)
    assert any(r["id"] == "text:2" for r in res)


def test_hybrid_rrf_returns_both_bands(tmp_path):
    idx = LanceIndex(tmp_path / "idx", 4)
    idx.upsert(_rows())
    res = idx.hybrid_query(vector=[0.9, 0.1, 0.0, 0.0], bm25_text="magic sword", top_k=4)
    ids = [r["id"] for r in res]
    assert "concept:1" in ids  # most vector-similar
    assert "text:1" in ids  # BM25 match


def test_kind_filter_excludes_other_surfaces(tmp_path):
    idx = LanceIndex(tmp_path / "idx", 4)
    idx.upsert(_rows())
    res = idx.hybrid_query(vector=[1.0, 0.0, 0.0, 0.0], top_k=5, kinds=["image"])
    assert res and all(r["kind"] == "image" for r in res)


def test_dimensions_must_be_pinned(tmp_path):
    idx = LanceIndex(tmp_path / "idx")  # no dimensions
    with pytest.raises(DimensionalityNotPinned):
        idx.upsert(_rows())


def test_factory_selects_backend(tmp_path):
    assert isinstance(create_index("local", dimensions=4, index_dir=tmp_path / "i"), LanceIndex)
    assert isinstance(create_index("lancedb", dimensions=4, index_dir=tmp_path / "i"), LanceIndex)
    assert isinstance(create_index("turbopuffer", dimensions=4, api_key="x"), Index)
    with pytest.raises(ValueError):
        create_index("nope", dimensions=4, index_dir=tmp_path / "i")
