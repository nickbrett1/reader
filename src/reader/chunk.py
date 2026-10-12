"""Text -> semantic chunks anchored to (book, page, offsets).

A chunk is an *offset range* into ``document_page.text`` (see §5.3).  The
chunker produces contiguous, non-overlapping ranges that tile the whole page,
so ``"".join(page.text[c.start:c.end] for c in chunks) == page.text`` — the
offset-integrity invariant tested in ``tests/test_chunk.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Soft target for a chunk, in characters.
DEFAULT_MAX_CHARS = 1200
#: A chunk is not closed before this many characters unless it is the tail of
#: the page (avoids crumbs when a break falls early).
DEFAULT_MIN_CHARS = 200

# Candidate break positions, in decreasing preference: paragraph, sentence,
# then any newline.  A long wall of text with no break still gets a hard split.
_PARAGRAPH = re.compile(r"\n[ \t]*\n")
_SENTENCE = re.compile(r"[.!?][\"')\]]?(?=\s)")
_NEWLINE = re.compile(r"\n")


@dataclass(frozen=True)
class Chunk:
    ordinal: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def _breakpoints(text: str) -> list[int]:
    pts: set[int] = set()
    for rx in (_PARAGRAPH, _SENTENCE, _NEWLINE):
        for m in rx.finditer(text):
            if 0 < m.end() < len(text):
                pts.add(m.end())
    return sorted(pts)


def chunk_text(
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[Chunk]:
    """Return contiguous chunk ranges covering ``text`` exactly."""
    n = len(text)
    if n == 0:
        return []
    if n <= max_chars:
        return [Chunk(ordinal=0, start=0, end=n)]

    pts = _breakpoints(text)
    chunks: list[Chunk] = []
    start = 0
    ordinal = 0
    while start < n:
        target = start + max_chars
        if target >= n:
            end = n
        else:
            # Prefer the latest break at or before the target, but do not close
            # a chunk that is still shorter than min_chars.
            floor = min(start + min_chars, target)
            candidates = [p for p in pts if floor <= p <= target]
            if candidates:
                end = candidates[-1]
            else:
                # No break inside the window: take the next break after the
                # target (a long paragraph), or the page end.
                after = [p for p in pts if p > target]
                end = after[0] if after else n
        # Guarantee forward progress even if all candidates collapsed.
        if end <= start:
            end = min(start + max_chars, n)
        chunks.append(Chunk(ordinal=ordinal, start=start, end=end))
        start = end
        ordinal += 1
    return chunks


def reconstruct(text: str, chunks: list[Chunk]) -> str:
    """Rebuild the page text from chunk ranges — the offset-integrity check."""
    return "".join(text[c.start : c.end] for c in chunks)
