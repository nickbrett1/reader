"""export-quotes: JSONL annotation items, and the Jev judge behind --jev."""

from __future__ import annotations

import json

import pytest

from reader import jev
from reader.store import Store

PAGE = (
    "The ruin was never meant to be entered, and the wind knows it. "
    "A second sentence follows here so the page has a pair to work with."
)


def _seed(data):
    store = Store(data / "reader.db")
    try:
        book_id = store.upsert_book("book", "pdf", "book.pdf", "sha")
        page_id = store.upsert_page(book_id, 7, PAGE, "sha")
        start = PAGE.index("The ruin")
        end = PAGE.index("knows it.") + len("knows it.")
        quote_id = store.upsert_quote(
            page_id, start, end, PAGE[start:end], 0.8, "statement", "why",
            "model", "sha",
        )
        store.link_quote_concept(quote_id, store.ensure_concept("Emberfall"), 0.9)
        store.upsert_quote(page_id, 0, 10, PAGE[0:10], 0.2, "caption", "",
                           "model", "sha")
    finally:
        store.close()


def test_export_writes_items_with_provenance_and_concepts(tmp_path, monkeypatch):
    from reader.__main__ import main

    data = tmp_path / "data"
    monkeypatch.setenv("READER_DATA", str(data))
    _seed(data)
    out = tmp_path / "quotes.jsonl"

    assert main(["export-quotes", "--out", str(out), "--include-context"]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 2
    top = rows[0]
    assert top["id"].startswith("q")
    assert top["text"] == PAGE[PAGE.index("The ruin"):PAGE.index("knows it.") + 9]
    assert top["book_title"] == "book" and top["page_no"] == 7
    assert top["concepts"] == ["Emberfall"]
    assert top["context"] == PAGE
    assert "jev" not in top  # no --jev, no judge output


def test_export_min_score_and_limit_filter(tmp_path, monkeypatch):
    from reader.__main__ import main

    data = tmp_path / "data"
    monkeypatch.setenv("READER_DATA", str(data))
    _seed(data)
    out = tmp_path / "q.jsonl"
    assert main(["export-quotes", "--out", str(out), "--min-score", "0.5"]) == 0
    rows = out.read_text().splitlines()
    assert len(rows) == 1 and json.loads(rows[0])["score"] == 0.8
    assert main(["export-quotes", "--out", str(out), "--limit", "1"]) == 0
    assert len(out.read_text().splitlines()) == 1


def test_jev_batches_and_aligns(monkeypatch):
    calls: list[list[str]] = []

    def fake_call(questions):
        calls.append(sorted(questions))
        return {
            qid: jev.Answer(0.5 + i / 100, False, {})
            for i, qid in enumerate(sorted(questions))
        }

    texts = [f"line {i}" for i in range(7)]
    scores = jev.display_confidence(texts, call=fake_call, batch=3)
    assert [len(c) for c in calls] == [3, 3, 1]
    assert len(scores) == 7 and all(s is not None for s in scores)


def test_jev_failure_is_none_not_zero():
    def fake_call(questions):
        return {qid: jev.Answer(None, True, {}) for qid in questions}

    assert jev.display_confidence(["a", "b"], call=fake_call, batch=5) == [None, None]


def test_export_jev_writes_potato_model_pred(tmp_path, monkeypatch):
    from reader import jev as jev_mod
    from reader.__main__ import main

    data = tmp_path / "data"
    monkeypatch.setenv("READER_DATA", str(data))
    monkeypatch.setenv("LITELLM_MASTER_KEY", "test-key")
    _seed(data)

    def fake_gateway_ask(base_url, api_key, model=jev_mod.DEFAULT_MODEL):
        def _call(questions):
            return {qid: jev_mod.Answer(0.42, False, {}) for qid in questions}

        return _call

    monkeypatch.setattr(jev_mod, "gateway_ask", fake_gateway_ask)
    out = tmp_path / "scored.jsonl"
    assert main(["export-quotes", "--out", str(out), "--jev"]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert all(r["jev"]["display"] == pytest.approx(0.42) for r in rows)
    assert all(r["model_pred"] == {"label": "plate", "confidence": pytest.approx(0.42)}
               for r in rows)


def test_export_jev_needs_a_gateway_key(tmp_path, monkeypatch):
    from reader.__main__ import main

    data = tmp_path / "data"
    monkeypatch.setenv("READER_DATA", str(data))
    monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)
    _seed(data)
    assert main(["export-quotes", "--out", str(tmp_path / "x.jsonl"), "--jev"]) == 2


def test_export_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    from reader.__main__ import main

    data = tmp_path / "data"
    monkeypatch.setenv("READER_DATA", str(data))
    _seed(data)
    out = tmp_path / "nothing.jsonl"
    assert main(["export-quotes", "--out", str(out), "--jev", "--dry-run"]) == 0
    assert not out.exists()
    assert "dry-run" in capsys.readouterr().out


def test_jev_ask_parses_answers_and_failed_items(monkeypatch):
    class FakeResponse:
        def __init__(self, payload):
            self._payload = json.dumps(payload).encode()

        def read(self):
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    payload = {
        "answers": {
            "q0": {"type": "noul", "noul": 0.69},
            "q1": {"type": "noul"},
        }
    }
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data)
        return FakeResponse(payload)

    monkeypatch.setattr(jev.urllib.request, "urlopen", fake_urlopen)
    answers = jev.ask(
        "http://gateway:4000",
        "key",
        {"q0": {"type": "noul", "instructions": "a"},
         "q1": {"type": "noul", "instructions": "b"}},
    )
    assert seen["url"] == "http://gateway:4000/typesafe/v1/system-one"
    assert seen["body"]["model"] == "jev-latest"
    assert answers["q0"].ok and answers["q0"].value == pytest.approx(0.69)
    assert answers["q1"].failed and answers["q1"].value is None
