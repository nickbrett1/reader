"""Concept extraction: per-book pass, surface-form pass, cross-book merge.

Three moving parts:

* :func:`extract_book_concepts` — per-book extraction.  Uses the gateway LLM
  when a client is supplied (gated to DeepSeek off-peak, v5 §12.14); otherwise
  falls back to a deterministic capitalised-phrase heuristic so the stage and
  its tests run without network.
* :func:`surface_pass` — find every surface form of a concept in canonical
  text, as (surface, start, end) offsets.
* :func:`candidate_pairs` — the dedup detector: lexical + ANN candidates,
  always gated on matching ``type`` (v5 §5.5).

Cross-book merge collapses the same concept extracted independently from two
books into one node.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import httpx

#: A word token: alphanumeric, may carry apostrophes/hyphens.  Numeric tokens
#: are kept so numbered franchise sequels do not collapse onto their parent
#: ("Resident Evil" vs "Resident Evil 2" — the 2026-10-12 full-corpus run).
_WORD = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*")
_CAP_PHRASE = re.compile(r"\b([A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,}){0,3})\b")
_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "into", "when", "then",
    "they", "their", "there", "which", "while", "would", "could", "should",
    "chapter", "page", "figure", "table", "one", "two", "three",
    # Pronouns / determiners / conjunctions that the capitalised-phrase
    # heuristic otherwise mistakes for concepts (see the 2026-10-12 calibration
    # dry-run: "You", "She", "But", "What", "His", …).
    "he", "she", "it", "we", "you", "him", "her", "his", "its", "our",
    "your", "them", "us", "me", "my", "mine", "who", "whom", "whose",
    "what", "where", "why", "how", "but", "or", "so",
    "yet", "nor", "if", "as", "at", "by", "of", "on", "to", "in", "an", "a",
    "all", "any", "each", "both", "some", "such", "no", "not", "only", "own",
    "same", "than", "too", "very", "just", "after", "before", "again",
    "another", "these", "those", "here", "now", "off", "out", "over", "under",
}


@dataclass
class ConceptCandidate:
    name: str
    type: str = "term"
    description: str = ""
    surfaces: list[str] = field(default_factory=list)
    confidence: float = 0.0
    book_id: int | None = None

    def embed_text(self) -> str:
        """The text embedded for the dedup detector: ``name + description``.

        §6.2.2 embeds ``canonical_name + description``; a bare one-word name is
        too flat a signal (2026-10-12 calibration dry-run).
        """
        return f"{self.name}. {self.description}".strip(" .") if self.description else self.name


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def heuristic_concepts(text: str, limit: int = 40) -> list[ConceptCandidate]:
    """Deterministic fallback: rank capitalised phrases by frequency."""
    counts: dict[str, int] = {}
    for m in _CAP_PHRASE.finditer(text):
        phrase = m.group(1).strip()
        if normalize(phrase) in _STOP or len(phrase) < 3:
            continue
        counts[phrase] = counts.get(phrase, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
    total = sum(counts.values()) or 1
    return [
        ConceptCandidate(name=name, type="term", confidence=count / total)
        for name, count in ranked
    ]


def parse_llm_concepts(payload: object) -> list[ConceptCandidate]:
    """Parse the LLM's JSON into candidates, tolerating near-misses."""
    out: list[ConceptCandidate] = []
    if not isinstance(payload, list):
        return out
    for row in payload:
        if isinstance(row, str):
            out.append(ConceptCandidate(name=row))
        elif isinstance(row, dict) and row.get("name"):
            out.append(
                ConceptCandidate(
                    name=str(row["name"]),
                    type=str(row.get("type", "term")),
                    description=str(row.get("description", "") or ""),
                    confidence=float(row.get("confidence", 0.0) or 0.0),
                )
            )
    return out


#: The prompt asks for typed concepts with a one-line description; the type
#: vocabulary is open but seeded so the type gate (§6.2.2) has something to use.
_CONCEPT_PROMPT = (
    "Extract the notable concepts from this video-game book's text. A concept "
    "is a character, place, item, organisation, franchise, creature, event or "
    "recurring idea — not a pronoun, determiner or common word. For each, "
    "return an object with: name (short canonical form), type (one of "
    "character, place, item, organisation, franchise, creature, event, "
    "concept), description (one clause describing it in this book), "
    "confidence (0-1). Return a JSON array only. Where a thing is already in "
    "the known list below, reuse that exact name. Known so far: {known}\n\n"
    "TEXT:\n{text}"
)


def gateway_concept_llm(
    base_url: str,
    api_key: str | None,
    model: str,
    *,
    timeout: float = 300.0,
    known: Sequence[str] = (),
) -> Callable[[str], object]:
    """Build an ``llm`` callable that extracts concepts via the LiteLLM gateway.

    The returned callable POSTs a chat completion to ``{base_url}/v1/chat/completions``
    and returns the parsed JSON array.  Kept as a callable so
    :func:`extract_book_concepts` stays transport-free and testable.
    """
    def _call(text: str) -> object:
        prompt = _CONCEPT_PROMPT.format(known=", ".join(known) or "(none)", text=text)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        resp = httpx.post(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return _parse_json_array(content)

    return _call


def _parse_json_array(content: str) -> object:
    """Parse a JSON array out of an LLM reply, tolerating code fences/prose."""
    text = content.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []


def extract_book_concepts(
    text: str,
    llm: Callable[[str], object] | None = None,
    limit: int = 40,
    book_id: int | None = None,
) -> list[ConceptCandidate]:
    """Per-book extraction via ``llm`` (returning JSON) or the heuristic."""
    if llm is not None:
        candidates = parse_llm_concepts(llm(text))
    else:
        candidates = heuristic_concepts(text, limit=limit)
    for c in candidates:
        c.book_id = book_id
    return candidates


def surface_pass(candidate: ConceptCandidate, text: str) -> list[tuple[str, int, int]]:
    """Find every surface form of ``candidate`` in ``text`` as offsets."""
    forms = [candidate.name, *candidate.surfaces]
    seen: set[str] = set()
    found: list[tuple[str, int, int]] = []
    for form in forms:
        key = normalize(form)
        if not key or key in seen:
            continue
        seen.add(key)
        pattern = re.compile(r"\b" + re.escape(form) + r"\b", re.IGNORECASE)
        for m in pattern.finditer(text):
            found.append((m.group(0), m.start(), m.end()))
    return sorted(found, key=lambda t: t[1])


def cross_book_merge(
    candidates: Iterable[ConceptCandidate],
) -> list[ConceptCandidate]:
    """Collapse the same (normalized name, type) seen across books."""
    merged: dict[tuple[str, str], ConceptCandidate] = {}
    for c in candidates:
        key = (normalize(c.name), c.type)
        existing = merged.get(key)
        if existing is None:
            merged[key] = ConceptCandidate(
                name=c.name,
                type=c.type,
                surfaces=list(c.surfaces),
                confidence=c.confidence,
                book_id=c.book_id,
            )
        else:
            for s in c.surfaces:
                if s not in existing.surfaces:
                    existing.surfaces.append(s)
            existing.confidence = max(existing.confidence, c.confidence)
    return [merged[k] for k in sorted(merged)]


# --- dedup detector ---------------------------------------------------------


def _tokens(s: str) -> set[str]:
    return {w for w in _WORD.findall(normalize(s)) if w not in _STOP}


def lexical_similarity(a: str, b: str) -> float:
    """Token-set Jaccard, 1.0 for identical token sets."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


@dataclass
class CandidatePair:
    i: int
    j: int
    score: float
    reason: str  # 'lexical' | 'ann' | 'both'


def candidate_pairs(
    concepts: list[ConceptCandidate],
    embeddings: list[list[float]] | None = None,
    embed_fn: Callable[[str], list[float]] | None = None,
    lexical_threshold: float = 0.6,
    ann_threshold: float = 0.85,
) -> list[CandidatePair]:
    """Lexical + ANN dedup candidates, gated on matching ``type``.

    ``type`` is a hard gate: two concepts of different types are never a merge
    candidate regardless of how similar their names are.
    """
    if embeddings is None and embed_fn is not None:
        embeddings = [embed_fn(c.name) for c in concepts]

    pairs: dict[tuple[int, int], CandidatePair] = {}
    n = len(concepts)
    for i in range(n):
        for j in range(i + 1, n):
            if concepts[i].type != concepts[j].type:
                continue
            reasons = []
            score = 0.0
            lex = lexical_similarity(concepts[i].name, concepts[j].name)
            if lex >= lexical_threshold:
                reasons.append("lexical")
                score = max(score, lex)
            if embeddings is not None:
                ann = cosine(embeddings[i], embeddings[j])
                if ann >= ann_threshold:
                    reasons.append("ann")
                    score = max(score, ann)
            if reasons:
                reason = "both" if len(reasons) == 2 else reasons[0]
                pairs[(i, j)] = CandidatePair(i=i, j=j, score=score, reason=reason)
    return [pairs[k] for k in sorted(pairs)]
