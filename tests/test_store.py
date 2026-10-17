"""Resume / idempotency: re-running a stage must not duplicate rows."""

from __future__ import annotations

from reader.store import Store


def _store(tmp_path):
    return Store(tmp_path / "r.db")


def test_page_and_chunk_rerun_is_idempotent(tmp_path):
    s = _store(tmp_path)
    try:
        bid = s.upsert_book("w", "epub", "w.epub", "sha")
        pid = s.upsert_page(bid, 0, "hello world", "sha")
        s.set_chunks(pid, [(0, 0, 11, "hello world")])
        # Re-run both stages.
        pid2 = s.upsert_page(bid, 0, "hello world", "sha")
        s.set_chunks(pid2, [(0, 0, 11, "hello world")])
        assert pid2 == pid
        assert s.stats()["document_page"] == 1
        assert s.stats()["chunk"] == 1
    finally:
        s.close()


def test_re_page_upsert_drops_stale_chunks(tmp_path):
    s = _store(tmp_path)
    try:
        bid = s.upsert_book("w", "epub", "w.epub", "sha")
        pid = s.upsert_page(bid, 0, "abcdef", "sha")
        s.set_chunks(pid, [(0, 0, 6, "abcdef")])
        # Re-extract the same page with different text.
        s.upsert_page(bid, 0, "ABCDEF", "sha2")
        assert s.stats()["chunk"] == 0
    finally:
        s.close()


def test_image_upsert_dedupes_on_sha(tmp_path):
    s = _store(tmp_path)
    try:
        s.upsert_image("hash", "p1", "image/png", 10, "embedded", "src")
        s.upsert_image("hash", "p2", "image/png", 20, "embedded", "src")
        assert s.stats()["image"] == 1
        row = s.conn.execute("SELECT bytes FROM image WHERE sha256='hash'").fetchone()
        assert row["bytes"] == 20  # updated in place
    finally:
        s.close()


def test_embedding_provenance_scopes_the_queue(tmp_path):
    s = _store(tmp_path)
    try:
        bid = s.upsert_book("w", "epub", "w.epub", "sha")
        pid = s.upsert_page(bid, 0, "hello", "sha")
        s.set_chunks(pid, [(0, 0, 5, "hello")])
        cid = s.conn.execute("SELECT id FROM chunk").fetchone()["id"]

        # Before embedding it is in every model's queue.
        assert [r["id"] for r in s.chunks_to_embed("cohere/embed-v5.0")] == [cid]
        assert [r["id"] for r in s.chunks_to_embed("gemini/gemini-embedding-2")] == [cid]

        s.stamp_chunk_embedding(cid, "cohere/embed-v5.0", 2048, "1")
        assert s.chunks_to_embed("cohere/embed-v5.0") == []
        assert [r["id"] for r in s.chunks_to_embed("gemini/gemini-embedding-2")] == [cid]
        row = s.conn.execute("SELECT * FROM chunk WHERE id=?", (cid,)).fetchone()
        assert row["embed_model_id"] == "cohere/embed-v5.0"
        assert row["dimensions"] == 2048
    finally:
        s.close()


def test_embed_selectors_carry_book_and_page(tmp_path):
    """chunk/image embed queues must carry book_id/page_id for the index rows."""
    s = _store(tmp_path)
    try:
        bid = s.upsert_book("w", "epub", "w.epub", "sha")
        pid = s.upsert_page(bid, 3, "hello world", "sha")
        s.set_chunks(pid, [(0, 0, 11, "hello world")])
        [c] = s.chunks_to_embed("m")
        assert c["book_id"] == bid
        assert c["page_id"] == pid

        img = s.upsert_image(
            "imgsha", "/x.bin", "image/png", 3, "embedded", "sha",
            book_id=bid, page_no=3, w=1024, h=768,
        )
        [im] = s.images_to_embed("m")
        assert im["book_id"] == bid
        assert im["page_id"] == pid  # page_no 3 resolves to the page
        # w/h must be selected: cmd_embed's image cost estimate reads them.
        assert (im["w"], im["h"]) == (1024, 768)

        # A book-less image keeps NULL book_id/page_id rather than inventing them.
        s.upsert_image("imgsha2", "/y.bin", "image/png", 3, "embedded", "sha")
        orphan = [r for r in s.images_to_embed("m") if r["id"] != img][0]
        assert orphan["book_id"] is None
        assert orphan["page_id"] is None
    finally:
        s.close()


def test_rebuild_rows_carry_book_and_page(tmp_path):
    import reader.index as index_mod

    s = _store(tmp_path)
    try:
        bid = s.upsert_book("w", "epub", "w.epub", "sha")
        pid = s.upsert_page(bid, 0, "hello", "sha")
        s.set_chunks(pid, [(0, 0, 5, "hello")])
        rows = index_mod.rebuild_rows(
            s,
            embed_text=lambda texts: [[0.0, 1.0] for _ in texts],
            embed_images=lambda imgs: [[0.0, 1.0] for _ in imgs],
        )
        [text_row_] = [r for r in rows if r["kind"] == "text"]
        assert text_row_["book_id"] == bid
        assert text_row_["page_id"] == pid
    finally:
        s.close()


def test_concept_review_log_is_append_only_and_derived(tmp_path):
    s = _store(tmp_path)
    try:
        a = s.ensure_concept("A")
        b = s.ensure_concept("B")
        c = s.ensure_concept("C")
        assert s.ensure_concept("A") == a  # idempotent
        s.add_review(a, b, "merge", "nick", 0.9)
        s.add_review(b, c, "merge", "nick", 0.9)
        s.add_review(c, a, "reject", "nick")
        s.rebuild_derived()
        res = {
            r["concept_id"]: r["resolved_id"]
            for r in s.conn.execute("SELECT * FROM concept_resolution")
        }
        assert res[a] == res[c]  # a->b->c merged transitively
        assert s.stats()["concept_review"] == 3
    finally:
        s.close()


def test_runs_are_recorded(tmp_path):
    s = _store(tmp_path)
    try:
        rid = s.start_run("extract")
        s.finish_run(rid, "ok")
        row = s.conn.execute("SELECT * FROM run WHERE id=?", (rid,)).fetchone()
        assert row["verb"] == "extract"
        assert row["status"] == "ok"
        assert row["finished_at"]
    finally:
        s.close()
