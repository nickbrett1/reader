"""Corpus pruning: ignore patterns and removing books without collateral loss."""

from __future__ import annotations

import pymupdf
import pytest

from reader import manifest
from reader.__main__ import main
from reader.store import Store


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    ingest = tmp_path / "ingest"
    data = tmp_path / "data"
    ingest.mkdir()
    for name, body in {
        "kept": "Emberfall holds a Magic Sword. Emberfall is a region.",
        "dropped": "Emberfall appears again. Dark Tower is a fortress.",
    }.items():
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), body)
        doc.save(str(ingest / f"{name}.pdf"))
        doc.close()
    monkeypatch.setenv("READER_INGEST", str(ingest))
    monkeypatch.setenv("READER_DATA", str(data))
    return ingest, data


def test_ingestignore_skips_matching_directory(tmp_path):
    (tmp_path / "Data Visualization").mkdir()
    (tmp_path / "Data Visualization" / "chart.epub").write_bytes(b"x")
    (tmp_path / "Machine Learning, AI and Bots").mkdir()
    (tmp_path / "Machine Learning, AI and Bots" / "llm.epub").write_bytes(b"x")
    (tmp_path / "game.epub").write_bytes(b"x")
    (tmp_path / ".ingestignore").write_text(
        "# off-topic tranches\nData Visualization\nMachine Learning, AI and Bots\n"
    )

    files = manifest.walk(tmp_path)
    assert [f.relpath for f in files] == ["game.epub"]


def test_prune_removes_books_and_only_orphaned_concepts(corpus):
    _, data = corpus
    assert main(["extract"]) == 0
    assert main(["chunk"]) == 0
    assert main(["concepts"]) == 0

    store = Store(data / "reader.db")
    try:
        books = {r["work_key"]: int(r["id"]) for r in store.conn.execute("SELECT id, work_key FROM book")}
        dropped = books["dropped"]
        assert store.stats()["concept"] >= 3  # Emberfall (shared), Magic Sword, Dark Tower
    finally:
        store.close()

    assert main(["prune", "--books", str(dropped), "--yes"]) == 0

    store = Store(data / "reader.db")
    try:
        assert [r["work_key"] for r in store.conn.execute("SELECT work_key FROM book")] == ["kept"]
        names = {r["name"] for r in store.conn.execute("SELECT name FROM concept")}
        assert "Emberfall" in names  # still mentioned by the kept book
        assert "Magic Sword" in names
        assert "Dark Tower" not in names
        assert store.stats()["concept_mention"] > 0
        # No page/chunk may survive without its book.
        assert store.conn.execute(
            "SELECT COUNT(*) FROM document_page WHERE book_id NOT IN (SELECT id FROM book)"
        ).fetchone()[0] == 0
        assert store.conn.execute(
            "SELECT COUNT(*) FROM chunk c LEFT JOIN document_page p ON p.id = c.page_id "
            "WHERE p.id IS NULL"
        ).fetchone()[0] == 0
    finally:
        store.close()


def test_prune_dry_run_changes_nothing(corpus):
    _, data = corpus
    assert main(["extract"]) == 0
    assert main(["chunk"]) == 0

    store = Store(data / "reader.db")
    try:
        before = store.stats()
    finally:
        store.close()

    assert main(["prune", "--books", "1-2", "--dry-run"]) == 0

    store = Store(data / "reader.db")
    try:
        assert store.stats() == before
    finally:
        store.close()
