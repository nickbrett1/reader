"""Quote selection: deterministic filter, verbatim safety, and store links."""

from __future__ import annotations

from reader import quotes
from reader.store import Store

PAGE = (
    "The ruin was never meant to be entered, and the wind knows it. "
    "A second sentence follows here so the page has a pair to work with. "
    "Fig. 12"
)


def test_candidate_filter_keeps_prose_and_drops_noise():
    page = (
        "THE TOWER\n\n"
        "The ruin was never meant to be entered, and the wind knows it. "
        "A second sentence follows here so the page has a pair to work with. "
        "Contents ........ 42\n"
        "Short.\n"
    )
    spans = quotes.candidate_spans(page)
    texts = [c.text for c in spans]
    assert any("never meant to be entered" in t for t in texts)
    assert all(not t.startswith("THE TOWER") for t in texts)
    assert all("...." not in t for t in texts)
    assert all(len(t) >= quotes.MIN_CHARS for t in texts)
    # Offsets are exact and each candidate is numbered from zero.
    for c in spans:
        assert page[c.start : c.end] == c.text
    assert [c.index for c in spans] == list(range(len(spans)))


def test_two_sentence_window_does_not_cross_a_paragraph():
    page = (
        "The ruin was never meant to be entered, and the wind knows it.\n\n"
        "A second sentence follows here so the page has a pair to work with."
    )
    spans = quotes.candidate_spans(page)
    assert not any("\n\n" in c.text for c in spans)


def test_verify_span_rejects_a_truncated_copy():
    text = PAGE[0 : PAGE.index(".") + 1]
    assert quotes.verify_span(PAGE, 0, len(text), text)
    assert not quotes.verify_span(PAGE, 0, len(text), text[:-1] + "…")


def test_parse_selection_uses_indices_and_canonical_concepts():
    candidates = quotes.candidate_spans(PAGE)
    payload = [
        {"index": 0, "score": 0.9, "kind": "statement",
         "concepts": ["emberfall", "Not A Concept"]},
        {"index": 99, "score": 1.0},  # no such candidate
    ]
    selected = quotes.parse_selection(payload, candidates, ["Emberfall"])
    assert len(selected) == 1
    assert selected[0].text == candidates[0].text
    assert selected[0].concepts == ["Emberfall"]


def test_select_quotes_verifies_before_returning():
    def fake_llm(_prompt: str) -> object:
        return [{"index": 0, "score": 0.8, "kind": "observation", "concepts": []}]

    selected = quotes.select_quotes(PAGE, fake_llm)
    assert len(selected) == 1
    assert PAGE[selected[0].start : selected[0].end] == selected[0].text


def test_no_model_means_no_quotes():
    assert quotes.select_quotes(PAGE, None) == []


def test_cli_writes_quotes_then_skips_unchanged_book(tmp_path, monkeypatch):
    import pymupdf

    from reader.__main__ import main

    ingest, data = tmp_path / "ingest", tmp_path / "data"
    ingest.mkdir()
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "The ruin was never meant to be entered, and the wind knows it. "
        "A second sentence follows here so the page has a pair to work with.",
    )
    doc.save(str(ingest / "book.pdf"))
    doc.close()
    monkeypatch.setenv("READER_INGEST", str(ingest))
    monkeypatch.setenv("READER_DATA", str(data))
    assert main(["extract"]) == 0
    assert main(["chunk"]) == 0

    calls: list[str] = []

    def fake_llm(*_args, **_kwargs):
        def _call(prompt: str) -> object:
            calls.append(prompt)
            return [{"index": 0, "score": 0.8, "kind": "statement", "concepts": []}]

        return _call

    monkeypatch.setattr(quotes, "gateway_quote_llm", fake_llm)
    assert main(["quotes", "--llm-model", "fake"]) == 0
    first_calls = len(calls)

    store = Store(data / "reader.db")
    try:
        assert store.stats()["quote"] >= 1
        before = store.stats()["quote"]
        assert store.book_quotes_current("book", store.conn.execute(
            "SELECT source_sha256 FROM book").fetchone()[0])
    finally:
        store.close()

    assert main(["quotes", "--llm-model", "fake"]) == 0
    assert len(calls) == first_calls  # unchanged book: zero new LLM calls
    store = Store(data / "reader.db")
    try:
        assert store.stats()["quote"] == before
    finally:
        store.close()


def test_store_links_reads_and_review(tmp_path):
    store = Store(tmp_path / "reader.db")
    try:
        book_id = store.upsert_book("book", "pdf", "book.pdf", "sha")
        page_id = store.upsert_page(book_id, 1, PAGE, "sha")
        concept_id = store.ensure_concept("Emberfall")
        start = PAGE.index("The ruin")
        end = PAGE.index("knows it.") + len("knows it.")
        quote_id = store.upsert_quote(
            page_id, start, end, PAGE[start:end], 0.9, "statement",
            "why", "model", "sha",
        )
        store.link_quote_concept(quote_id, concept_id, 0.8)

        rows = store.quotes_for_concept(concept_id)
        assert [r["id"] for r in rows] == [quote_id]
        assert store.quote_citations(quote_id)["page_no"] == 1
        assert store.quote_counts_by_book()[0]["quotes"] == 1

        store.add_quote_review(quote_id, "reject", "human")
        assert store.quotes_for_concept(concept_id) == []
    finally:
        store.close()


def test_changed_page_text_invalidates_its_quotes(tmp_path):
    store = Store(tmp_path / "reader.db")
    try:
        book_id = store.upsert_book("book", "pdf", "book.pdf", "sha")
        page_id = store.upsert_page(book_id, 1, PAGE, "sha")
        start = PAGE.index("The ruin")
        end = PAGE.index("knows it.") + len("knows it.")
        store.upsert_quote(page_id, start, end, PAGE[start:end], 0.9,
                           "statement", "", "model", "sha")
        store.mark_book_quotes(book_id, "sha")
        assert store.book_quotes_current("book", "sha")

        store.upsert_page(book_id, 1, "A wholly different page of text.", "sha")
        assert store.stats()["quote"] == 0
        assert store.stats()["quote_concept"] == 0
        assert not store.book_quotes_current("book", "sha")
    finally:
        store.close()
