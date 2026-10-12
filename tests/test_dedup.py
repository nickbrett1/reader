"""Image dedupe (perceptual hash) and the concept dedup detector."""

from __future__ import annotations

import io

from PIL import Image

from reader import images
from reader.concepts import ConceptCandidate, candidate_pairs, lexical_similarity
from reader.images import RawImage, dedupe, hamming


def _grad_png(offset: int, size: int = 64) -> bytes:
    img = Image.new("L", (size, size))
    img.putdata([(x * 4 + offset) % 256 for _ in range(size) for x in range(size)])
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _raw(data: bytes) -> RawImage:
    return RawImage(data=data, mime="image/png", w=64, h=64, kind="embedded")


def test_exact_duplicates_collapse():
    data = _grad_png(0)
    deduped = dedupe([_raw(data), _raw(data), _raw(data)])
    assert len(deduped) == 1
    assert deduped[0].duplicates == 2


def test_distinct_images_survive():
    deduped = dedupe([_raw(_grad_png(0)), _raw(_grad_png(97))])
    assert len(deduped) == 2


def test_hamming_distance():
    assert hamming("0" * 16, "0" * 16) == 0
    assert hamming("0" * 16, "f" * 16) == 64


def test_thumbnail_is_a_png_under_cap():
    blob = images.thumbnail(_grad_png(0))
    assert blob.startswith(b"\x89PNG")
    assert len(blob) <= 64 * 1024


def test_near_duplicates_collapse_by_phash():
    # One flipped low bit is within the perceptual threshold.
    a = "0000000000000000"
    b = "0000000000000001"
    assert hamming(a, b) <= images.PHASH_THRESHOLD


# --- concept dedup detector -------------------------------------------------


def test_lexical_similarity():
    assert lexical_similarity("Magic Sword", "magic sword") == 1.0
    assert lexical_similarity("Magic Sword", "the sun") == 0.0


def test_candidate_pairs_lexical_same_type():
    cs = [ConceptCandidate("Magic Sword", "item"), ConceptCandidate("magic sword", "item")]
    pairs = candidate_pairs(cs)
    assert len(pairs) == 1
    assert pairs[0].reason == "lexical"


def test_candidate_pairs_gated_on_type():
    cs = [ConceptCandidate("Emberfall", "place"), ConceptCandidate("Emberfall", "item")]
    assert candidate_pairs(cs) == []


def test_candidate_pairs_ann():
    cs = [ConceptCandidate("Aaa", "term"), ConceptCandidate("Bbb", "term")]
    embs = [[1.0, 0.0], [0.999, 0.001]]
    pairs = candidate_pairs(cs, embeddings=embs, ann_threshold=0.9)
    assert len(pairs) == 1
    assert pairs[0].reason == "ann"
