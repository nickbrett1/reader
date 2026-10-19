"""Quote selection: verbatim sentence spans, linked to canonical concepts.

A quote is an **offset range into ``document_page.text``**, exactly like a
chunk.  The LLM never retypes the quote: it numbers the deterministic
candidates and selects by index, and the server reconstructs and verifies the
span.  That removes the "model paraphrased slightly" failure class entirely.

The paid part of this module is one chat call per page (:func:`select_quotes`).
The candidate filter (:func:`candidate_spans`) is free, deterministic and
unit-testable without an API key.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import httpx

#: Candidate bounds (characters).  Shorter is a fragment; longer stops looking
#: like a line worth putting on a plate.
MIN_CHARS = 25
MAX_CHARS = 400
#: At most this many candidates go to the model for one page.
MAX_CANDIDATES = 40

QUOTE_KINDS = ("statement", "caption", "author_voice", "observation")

_TERMINAL = re.compile(r'[.!?]+["\'”’)\]]*(?=\s|$)')
_PARAGRAPH = re.compile(r"\n[ \t]*\n")
_TOC = re.compile(r"\.{3,}\s*\d+")
_WORD = re.compile(r"[A-Za-z0-9]+")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


@dataclass(frozen=True)
class Candidate:
    """A deterministic, numbered candidate span for the LLM to select."""

    index: int
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class QuoteCandidate:
    """A verified span the model selected."""

    start: int
    end: int
    text: str
    score: float
    kind: str
    rationale: str = ""
    concepts: list[str] = field(default_factory=list)


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _TERMINAL.finditer(text):
        spans.append((start, m.end()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _noise(text: str) -> bool:
    if not text or len(text) < MIN_CHARS or len(text) > MAX_CHARS:
        return True
    if not text.endswith((".", "!", "?", '"', "'", "”", "’", ")")):
        return True  # bare heading / rail
    if _TOC.search(text) or re.search(r"\.{4,}", text):
        return True
    if _CONTROL.search(text):
        return True
    words = _WORD.findall(text)
    if len(words) < 4:
        return True
    if len(text) >= 4 and text.upper() == text and sum(c.isalpha() for c in text) > 8:
        return True  # all-caps rail
    non_word = sum(1 for c in text if not (c.isalnum() or c.isspace() or c in ".,!?'\"-—:;()’“”"))
    return non_word / len(text) > 0.2


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _PARAGRAPH.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def candidate_spans(page_text: str, cap: int = MAX_CANDIDATES) -> list[Candidate]:
    """Free shortlist: sentences and two-sentence windows, noise removed.

    Multi-sentence quotes are allowed (they often read better) but never span a
    paragraph break.  Headings on their own paragraph are dropped because they
    have no terminal punctuation.  Offsets are into ``page_text`` unchanged.
    """
    spans: list[tuple[int, int]] = []
    for p_start, p_end in _paragraph_spans(page_text):
        sentences = [
            (p_start + a, p_start + b)
            for a, b in _sentence_spans(page_text[p_start:p_end])
        ]
        for i, (start, end) in enumerate(sentences):
            spans.append((start, end))
            if i + 1 < len(sentences):
                nxt_end = sentences[i + 1][1]
                if (nxt_end - start) <= MAX_CHARS:
                    spans.append((start, nxt_end))
    out: list[Candidate] = []
    seen: set[str] = set()
    for start, end in spans:
        start, end = _trim(page_text, start, end)
        text = page_text[start:end]
        if _noise(text):
            continue
        key = " ".join(text.lower().split())
        if key in seen:
            continue
        seen.add(key)
        out.append(Candidate(len(out), start, end, text))
        if len(out) >= cap:
            break
    return out


def verify_span(page_text: str, start: int, end: int, text: str) -> bool:
    """The load-bearing check: the stored text is a byte-exact page substring."""
    if not (0 <= start < end <= len(page_text)):
        return False
    if page_text[start:end] != text:
        return False
    if _CONTROL.search(text):
        return False
    return not text.endswith(("…", "--"))


def _prompt(page_text: str, candidates: Sequence[Candidate], known: Sequence[str]) -> str:
    numbered = "\n".join(f"[{c.index}] {c.text}" for c in candidates)
    concepts = ", ".join(known) if known else "(none)"
    return (
        "You are selecting pull-quotes from a video-game book page. A good quote "
        "is a memorable, self-contained observation in the author's voice — not "
        "a heading, caption fragment, table row, or table-of-contents line.\n\n"
        "Choose only from the numbered candidates below. Return a JSON array; "
        "each item must be an object with:\n"
        "  index: the candidate number you selected\n"
        "  score: 0-1, how quotable it is\n"
        "  kind: one of statement, caption, author_voice, observation\n"
        "  rationale: one short line why (debug only)\n"
        "  concepts: array of concept names from the supplied list that the "
        "quote speaks to; use an empty array when it names none. Never invent "
        "a name that is not in the list.\n\n"
        "Return [] when nothing qualifies. Do not rewrite or paraphrase the "
        "text; select by index only.\n\n"
        f"CONCEPT LIST: {concepts}\n\n"
        f"CANDIDATES:\n{numbered}"
    )


def _parse_array(content: str) -> object:
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


def gateway_quote_llm(
    base_url: str,
    api_key: str | None,
    model: str,
    *,
    timeout: float = 300.0,
) -> Callable[[str], object]:
    """Build a callable that asks the gateway to select quotes from a prompt."""

    def _call(prompt: str) -> object:
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
        return _parse_array(resp.json()["choices"][0]["message"]["content"])

    return _call


def parse_selection(
    payload: object,
    candidates: Sequence[Candidate],
    known_concepts: Sequence[str],
) -> list[QuoteCandidate]:
    """Turn the model's index-based selection into verified quote candidates."""
    if not isinstance(payload, list):
        return []
    by_index = {c.index: c for c in candidates}
    canonical = {name.casefold(): name for name in known_concepts}
    out: list[QuoteCandidate] = []
    seen: set[tuple[int, int]] = set()
    for row in payload:
        if not isinstance(row, dict):
            continue
        try:
            idx = int(row.get("index"))
        except (TypeError, ValueError):
            continue
        cand = by_index.get(idx)
        if cand is None or (cand.start, cand.end) in seen:
            continue
        try:
            score = min(1.0, max(0.0, float(row.get("score", 0.0) or 0.0)))
        except (TypeError, ValueError):
            score = 0.0
        kind = str(row.get("kind", "statement"))
        if kind not in QUOTE_KINDS:
            kind = "statement"
        concepts: list[str] = []
        for raw in row.get("concepts") or []:
            name = canonical.get(str(raw).casefold())
            if name and name not in concepts:
                concepts.append(name)
        seen.add((cand.start, cand.end))
        out.append(
            QuoteCandidate(
                start=cand.start,
                end=cand.end,
                text=cand.text,
                score=score,
                kind=kind,
                rationale=str(row.get("rationale", "") or ""),
                concepts=concepts,
            )
        )
    return out


def select_quotes(
    page_text: str,
    llm: Callable[[str], object] | None,
    known_concepts: Sequence[str] = (),
    *,
    cap: int = MAX_CANDIDATES,
) -> list[QuoteCandidate]:
    """Filter -> LLM selection -> verbatim verification for one page.

    No model means no quotes: the stage is a judgement pass, and a deterministic
    fallback would put arbitrary sentences on the plate.  ``cmd_quotes`` refuses
    to run without a model for the same reason.
    """
    if llm is None:
        return []
    candidates = candidate_spans(page_text, cap=cap)
    if not candidates:
        return []
    selected = parse_selection(llm(_prompt(page_text, candidates, known_concepts)), candidates, known_concepts)
    return [
        q for q in selected if verify_span(page_text, q.start, q.end, q.text)
    ]
