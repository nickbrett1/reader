"""Concept extraction: LLM parsing, the gateway callable, and description plumbing."""

from __future__ import annotations

import json

import reader.concepts as concepts
from reader.store import Store


def test_parse_llm_concepts_reads_description():
    payload = [
        {"name": "Agro", "type": "character", "description": "Wander's horse", "confidence": 0.9},
        {"name": "Dormin", "type": "creature"},  # no description -> ""
        "Shadow of the Colossus",  # bare string -> name only
    ]
    out = concepts.parse_llm_concepts(payload)
    assert out[0].description == "Wander's horse"
    assert out[0].type == "character"
    assert out[1].description == ""
    assert out[2].name == "Shadow of the Colossus"


def test_candidate_embed_text_uses_name_and_description():
    c = concepts.ConceptCandidate(name="Agro", type="character", description="Wander's horse")
    assert c.embed_text() == "Agro. Wander's horse"
    # No description -> the bare name.
    assert concepts.ConceptCandidate(name="Agro").embed_text() == "Agro"


def test_gateway_concept_llm_posts_and_parses(monkeypatch):
    captured = {}

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {"message": {"content": "```json\n[{\"name\": \"Ico\", \"type\": \"franchise\"}]\n```"}}
                ]
            }

    def _post(url, *, headers=None, json=None, timeout=None):  # noqa: A002 - matches httpx
        captured.update(url=url, headers=headers, body=json, timeout=timeout)
        return _Resp()

    monkeypatch.setattr(concepts.httpx, "post", _post)
    llm = concepts.gateway_concept_llm("http://nas:4000/", "secret", "deepseek-v4-pro", known=["Agro"])
    result = llm("some book text")

    assert captured["url"] == "http://nas:4000/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["body"]["model"] == "deepseek-v4-pro"
    assert "Agro" in captured["body"]["messages"][0]["content"]  # known list injected
    assert result == [{"name": "Ico", "type": "franchise"}]


def test_extract_book_concepts_uses_llm_and_backfills_book_id():
    def llm(_text):
        return [{"name": "Mono", "type": "character", "description": "the maiden"}]

    out = concepts.extract_book_concepts("ignored", llm=llm, book_id=7)
    assert out[0].name == "Mono" and out[0].book_id == 7


def test_heuristic_drops_pronoun_and_filler_concepts():
    text = "You see him. But what about the Master Sword? She waits. All of them. Master Sword."
    names = {c.name for c in concepts.heuristic_concepts(text)}
    assert "Master Sword" in names
    assert names.isdisjoint({"You", "But", "What", "She", "All", "Him"})


def test_ensure_concept_stores_and_backfills_description(tmp_path):
    with Store(tmp_path / "reader.db") as store:
        cid = store.ensure_concept("Agro", "character", "Wander's horse")
        assert store.conn.execute(
            "SELECT description FROM concept WHERE id = ?", (cid,)
        ).fetchone()["description"] == "Wander's horse"
        # Idempotent, and a later mention without a description does not erase it.
        assert store.ensure_concept("Agro", "character") == cid
        assert store.conn.execute(
            "SELECT description FROM concept WHERE id = ?", (cid,)
        ).fetchone()["description"] == "Wander's horse"


def test_concepts_to_embed_selects_description(tmp_path):
    with Store(tmp_path / "reader.db") as store:
        store.ensure_concept("Dormin", "creature", "an ancient entity")
        row = store.concepts_to_embed("cohere/embed-v5.0")[0]
        assert row["name"] == "Dormin" and row["description"] == "an ancient entity"
        assert json.dumps(dict(row))  # row is mapping-like


def test_lexical_keeps_digits_so_sequels_are_not_identical():
    # "Resident Evil" must not token-match "Resident Evil 2" at 1.0 (digit was
    # dropped before the 2026-10-12 full-corpus calibration).
    assert concepts.lexical_similarity("Resident Evil", "Resident Evil 2") < 1.0
    assert concepts.lexical_similarity("The Witcher", "The Witcher 3") < 1.0
    # A true variant (same tokens) still scores 1.0.
    assert concepts.lexical_similarity("Resident Evil", "resident evil") == 1.0
