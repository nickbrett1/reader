"""End-to-end offline CLI: manifest -> extract -> chunk -> images -> concepts."""

from __future__ import annotations

import pymupdf
import pytest

from reader.__main__ import main
from reader.store import Store


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    ingest = tmp_path / "ingest"
    data = tmp_path / "data"
    ingest.mkdir()
    doc = pymupdf.open()
    for i in range(2):
        page = doc.new_page()
        page.insert_text(
            (72, 72),
            f"Page {i}. Emberfall holds a Magic Sword. "
            "The Magic Sword is a legendary item. Emberfall is a region.",
        )
    doc.save(str(ingest / "book.pdf"))
    doc.close()
    monkeypatch.setenv("READER_INGEST", str(ingest))
    monkeypatch.setenv("READER_DATA", str(data))
    return ingest, data


def test_manifest_json(corpus, capsys):
    assert main(["manifest", "--json"]) == 0
    out = capsys.readouterr().out
    assert "book.pdf" in out and '"format": "pdf"' in out


def test_full_offline_pipeline(corpus):
    _, data = corpus
    assert main(["extract"]) == 0
    assert main(["chunk"]) == 0
    assert main(["images"]) == 0
    assert main(["concepts"]) == 0

    store = Store(data / "reader.db")
    try:
        stats = store.stats()
        assert stats["book"] == 1
        assert stats["document_page"] == 2
        assert stats["chunk"] >= 2
        assert stats["concept"] >= 2  # Emberfall, Magic Sword
        # Chunks tile their page exactly (offset integrity at the DB level).
        for page in store.conn.execute("SELECT id, text FROM document_page"):
            rows = store.conn.execute(
                "SELECT start_offset, end_offset FROM chunk WHERE page_id = ? ORDER BY ordinal",
                (page["id"],),
            ).fetchall()
            rebuilt = "".join(page["text"][r["start_offset"] : r["end_offset"]] for r in rows)
            assert rebuilt == page["text"]
    finally:
        store.close()


def test_manifest_records_run(corpus, capsys):
    _, data = corpus
    assert main(["manifest", "--json", "--commit"]) == 0
    store = Store(data / "reader.db")
    try:
        assert store.stats()["book"] == 1
        run = store.conn.execute("SELECT * FROM run ORDER BY id DESC LIMIT 1").fetchone()
        assert run["verb"] == "manifest" and run["status"] == "ok"
    finally:
        store.close()


def test_index_id_normalises_eval_labels():
    from reader.__main__ import _index_id

    # A bare record id is a text chunk; a qualified id is kept as-is.
    assert _index_id(12) == "text:12"
    assert _index_id("12") == "text:12"
    assert _index_id("concept:3") == "concept:3"
    assert _index_id("image:7") == "image:7"


def test_book_id_filter_parses_csv():
    from reader.__main__ import _book_id_filter

    # No filter -> None (every book is processed).
    assert _book_id_filter(None) is None
    assert _book_id_filter("") is None
    # Comma-separated ids, tolerant of surrounding whitespace.
    assert _book_id_filter("42,43,44") == {42, 43, 44}
    assert _book_id_filter(" 7 , 9 ") == {7, 9}
