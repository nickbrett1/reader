"""Extraction: PDF via PyMuPDF, EPUB via ebooklib, HTML text helper."""

from __future__ import annotations

import pymupdf
import pytest
from ebooklib import epub

from reader import extract
from reader.extract import html_to_text


@pytest.fixture
def pdf_path(tmp_path):
    path = tmp_path / "book.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Emberfall is a land of magic swords and old maps.")
    doc.new_page()  # left blank -> scanned
    doc.save(str(path))
    doc.close()
    return path


def test_pdf_pages_text_and_scanned_flag(pdf_path):
    pages = extract.pdf_pages(pdf_path)
    assert [p.page_no for p in pages] == [0, 1]
    assert "Emberfall" in pages[0].text
    assert pages[0].scanned is False
    assert pages[1].scanned is True  # no text layer


def test_render_pdf_page_is_png(pdf_path):
    r = extract.render_pdf_page(pdf_path, 1, dpi=72)
    assert r.mime == "image/png"
    assert r.data.startswith(b"\x89PNG")
    assert r.w > 0 and r.h > 0


def test_extract_dispatch(pdf_path):
    assert len(extract.extract(pdf_path, "pdf")) == 2
    with pytest.raises(ValueError):
        extract.extract(pdf_path, "mobi")


def test_html_to_text():
    text = html_to_text("<h1>Chapter</h1><p>Magic swords.</p><p>Old maps.</p>")
    assert "Chapter" in text
    assert "Magic swords." in text
    assert "<p>" not in text


def test_epub_pages(tmp_path):
    path = tmp_path / "book.epub"
    book = epub.EpubBook()
    book.set_identifier("test-id")
    book.set_title("Test")
    book.set_language("en")
    ch = epub.EpubHtml(title="Ch1", file_name="ch1.xhtml", lang="en")
    ch.content = "<h1>Chapter One</h1><p>A magic sword appears.</p>"
    book.add_item(ch)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", ch]
    epub.write_epub(str(path), book)

    pages = extract.epub_pages(path)
    combined = "\n".join(p.text for p in pages)
    assert "magic sword appears" in combined.lower()
