"""Text extraction: EPUB -> ebooklib, PDF -> PyMuPDF.

Both extractors return a list of :class:`Page` objects.  A PDF page whose
extracted text is shorter than :data:`SCANNED_TEXT_THRESHOLD` is flagged
``scanned=True`` so the images stage knows to rasterise it.  Every page carries
``extractor_version`` and the source ``sha256`` for provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

from .schema import EXTRACTOR_VERSION

#: Below this many non-whitespace characters a PDF page is assumed to be a
#: scanned image with no text layer.
SCANNED_TEXT_THRESHOLD = 16


@dataclass
class Page:
    page_no: int
    text: str
    scanned: bool = False
    extractor_version: str = EXTRACTOR_VERSION


@dataclass
class Render:
    """A rasterised page image."""

    page_no: int
    mime: str
    data: bytes
    w: int
    h: int


class _TextExtractor(HTMLParser):
    """Minimal HTML -> text so we do not pull in a parser dependency."""

    _BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        lines = [ln.strip() for ln in raw.splitlines()]
        return "\n".join(ln for ln in lines if ln)


def html_to_text(html: str | bytes) -> str:
    if isinstance(html, bytes):
        html = html.decode("utf-8", errors="replace")
    p = _TextExtractor()
    p.feed(html)
    return p.text()


def epub_pages(path: Path) -> list[Page]:
    """One ``Page`` per spine document.  EPUB has no native pagination."""
    import ebooklib
    from ebooklib import epub

    book = epub.read_epub(str(path))
    pages: list[Page] = []
    n = 0
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        text = html_to_text(item.get_content())
        pages.append(Page(page_no=n, text=text))
        n += 1
    return pages


def pdf_pages(path: Path) -> list[Page]:
    import pymupdf

    pages: list[Page] = []
    with pymupdf.open(str(path)) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text")
            scanned = len(text.strip()) < SCANNED_TEXT_THRESHOLD
            pages.append(Page(page_no=i, text=text, scanned=scanned))
    return pages


def render_pdf_page(path: Path, page_no: int, dpi: int = 200) -> Render:
    """Rasterise a single PDF page to PNG (used for scanned pages)."""
    import pymupdf

    with pymupdf.open(str(path)) as doc:
        pix = doc[page_no].get_pixmap(dpi=dpi)
        return Render(
            page_no=page_no,
            mime="image/png",
            data=pix.tobytes("png"),
            w=pix.width,
            h=pix.height,
        )


def extract(path: Path, fmt: str) -> list[Page]:
    if fmt == "epub":
        return epub_pages(path)
    if fmt == "pdf":
        return pdf_pages(path)
    raise ValueError(f"unsupported format: {fmt!r}")
