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

import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]+")
_CAP_PHRASE = re.compile(r"\b([A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,}){0,3})\b")
_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "into", "when", "then",
    "they", "their", "there", "which", "while", "would", "could", "should",
    "chapter", "page", "figure", "table", "one", "two", "three",
}


@dataclass
class ConceptCandidate:
    name: str
    type: str = "term"
    surfaces: list[str] = field(default_factory=list)
    confidence: float = 0.0
    book_id: int | None = None


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
                    confidence=float(row.get("confidence", 0.0) or 0.0),
                )
            )
    return out


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
