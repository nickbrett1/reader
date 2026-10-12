"""Offset integrity: chunk ranges reconstruct document_page.text exactly."""

from __future__ import annotations

import pytest

from reader.chunk import chunk_text, reconstruct

TEXTS = [
    "",
    "short",
    "one two three. " * 40,
    ("Paragraph one, a few sentences long. It continues here.\n\n"
     "Paragraph two starts now and keeps going for a while. " * 20),
    "x" * 5000,  # a single unbroken wall of text
    "\n\n\n".join(f"Sentence {i}. More words follow here." for i in range(200)),
]


@pytest.mark.parametrize("text", TEXTS)
def test_ranges_tile_the_page(text):
    ranges = chunk_text(text)
    assert reconstruct(text, ranges) == text


@pytest.mark.parametrize("text", TEXTS)
def test_ranges_are_contiguous_and_ordered(text):
    ranges = chunk_text(text)
    cursor = 0
    for i, c in enumerate(ranges):
        assert c.ordinal == i
        assert c.start == cursor
        assert c.end > c.start
        cursor = c.end
    assert cursor == len(text)


def test_short_text_is_one_chunk():
    ranges = chunk_text("hello world")
    assert len(ranges) == 1
    assert (ranges[0].start, ranges[0].end) == (0, 11)


def test_empty_text_has_no_chunks():
    assert chunk_text("") == []
