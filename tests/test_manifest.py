"""Manifest walk, work grouping, and the one-format-per-work guard."""

from __future__ import annotations

import pytest

from reader import manifest


def test_skips_eadir_dsstore_and_junk(tmp_path):
    (tmp_path / "@eaDir").mkdir()
    (tmp_path / "@eaDir" / "ghost.epub").write_bytes(b"x")
    (tmp_path / ".DS_Store").write_bytes(b"x")
    (tmp_path / "._resource.epub").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")
    (tmp_path / "a.epub").write_bytes(b"book")

    files = manifest.walk(tmp_path)
    assert [f.relpath for f in files] == ["a.epub"]
    assert files[0].fmt == "epub"
    assert len(files[0].sha256) == 64


def test_format_clash_keeps_preferred_and_warns(tmp_path):
    # A duplicate title in two formats must not abort the manifest; epub wins.
    (tmp_path / "b.epub").write_bytes(b"epub")
    (tmp_path / "b.pdf").write_bytes(b"pdf")
    with pytest.warns(UserWarning, match="mixed formats"):
        works = manifest.build(tmp_path)
    assert len(works) == 1
    assert works[0].format == "epub"
    assert [f.relpath for f in works[0].files] == ["b.epub"]


def test_works_grouped_by_dir_and_stem(tmp_path):
    sub = tmp_path / "series"
    sub.mkdir()
    (sub / "c.epub").write_bytes(b"e")
    (tmp_path / "d.pdf").write_bytes(b"p")
    works = manifest.build(tmp_path)
    assert [w.key for w in works] == ["d", "series/c"]
    assert {w.format for w in works} == {"epub", "pdf"}
