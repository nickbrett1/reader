"""Image handling: embedded-art extraction, page rasterisation, pHash dedupe.

Two kinds of image are produced:

* ``embedded`` — artwork carried inside the book (EPUB ``<img>``, PDF
  ``get_images``);
* ``rendered`` — a rasterised page, produced for scanned PDF pages that have
  no usable text layer.

Bytes are content-addressed (keyed by sha256) and deduped perceptually: two
images whose dHash differ by fewer than :data:`PHASH_THRESHOLD` bits are the
same picture (the same artwork appears at several resolutions inside one book).
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

from .schema import EXTRACTOR_VERSION

#: Hamming distance (out of 64 bits) at or below which two dHashes collapse.
PHASH_THRESHOLD = 6


@dataclass
class RawImage:
    """An image extracted but not yet deduped or stored."""

    data: bytes
    mime: str
    w: int
    h: int
    kind: str  # 'embedded' | 'rendered'
    page_no: int | None = None
    caption: str | None = None
    extractor_version: str = EXTRACTOR_VERSION

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    @property
    def ext(self) -> str:
        return {"image/png": "png", "image/jpeg": "jpg", "image/gif": "gif"}.get(
            self.mime, "bin"
        )

    @property
    def size(self) -> int:
        return len(self.data)


def dhash(data: bytes, hash_size: int = 8) -> str:
    """64-bit difference hash as a hex string."""
    from PIL import Image

    img = Image.open(io.BytesIO(data)).convert("L").resize(
        (hash_size + 1, hash_size), Image.LANCZOS
    )
    # Mode "L" tobytes() is one byte per pixel, row-major — indexable directly.
    px = img.tobytes()
    bits = 0
    for row in range(hash_size):
        for col in range(hash_size):
            left = px[row * (hash_size + 1) + col]
            right = px[row * (hash_size + 1) + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return f"{bits:016x}"


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def thumbnail(data: bytes, max_px: int = 256, max_bytes: int = 64 * 1024) -> bytes:
    """Small PNG thumbnail for the inline ``image.thumb`` BLOB (<= 64 KB)."""
    from PIL import Image

    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail((max_px, max_px))
    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    blob = out.getvalue()
    if len(blob) > max_bytes:  # shrink further rather than blow the cap
        img.thumbnail((max_px // 2, max_px // 2))
        out = io.BytesIO()
        img.save(out, format="PNG", optimize=True)
        blob = out.getvalue()
    return blob


# --- extraction -------------------------------------------------------------


def extract_pdf_images(path: Path) -> list[RawImage]:
    import pymupdf

    out: list[RawImage] = []
    with pymupdf.open(str(path)) as doc:
        for page_no, page in enumerate(doc):
            for info in page.get_images(full=True):
                xref = info[0]
                base = doc.extract_image(xref)
                out.append(
                    RawImage(
                        data=base["image"],
                        mime=f"image/{base.get('ext', 'png')}",
                        w=base.get("width", 0),
                        h=base.get("height", 0),
                        kind="embedded",
                        page_no=page_no,
                    )
                )
    return out


def extract_epub_images(path: Path) -> list[RawImage]:
    import ebooklib
    from ebooklib import epub

    book = epub.read_epub(str(path))
    out: list[RawImage] = []
    idx = 0
    for item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
        data = item.get_content()
        mime = item.media_type or "image/jpeg"
        try:
            from PIL import Image

            with Image.open(io.BytesIO(data)) as im:
                w, h = im.size
        except Exception:  # pragma: no cover - malformed asset
            w = h = 0
        out.append(
            RawImage(data=data, mime=mime, w=w, h=h, kind="embedded", page_no=idx)
        )
        idx += 1
    return out


def extract_embedded(path: Path, fmt: str) -> list[RawImage]:
    if fmt == "pdf":
        return extract_pdf_images(path)
    if fmt == "epub":
        return extract_epub_images(path)
    return []


def render_scanned_pages(path: Path, page_nos: list[int], dpi: int = 200) -> list[RawImage]:
    """Rasterise the given PDF pages as ``rendered`` images."""
    from . import extract

    out: list[RawImage] = []
    for pno in page_nos:
        r = extract.render_pdf_page(path, pno, dpi=dpi)
        out.append(
            RawImage(
                data=r.data, mime=r.mime, w=r.w, h=r.h, kind="rendered", page_no=pno
            )
        )
    return out


# --- dedupe -----------------------------------------------------------------


@dataclass
class DedupedImage:
    image: RawImage
    phash: str
    duplicates: int = 0


def dedupe(images: list[RawImage], threshold: int = PHASH_THRESHOLD) -> list[DedupedImage]:
    """Collapse perceptually-identical images, keeping the first seen.

    Exact byte duplicates are collapsed first (cheap), then near-duplicates via
    dHash hamming distance.  The kept image records how many it absorbed.
    """
    seen_sha: dict[str, DedupedImage] = {}
    kept: list[DedupedImage] = []
    for img in images:
        sha = img.sha256
        if sha in seen_sha:
            seen_sha[sha].duplicates += 1
            continue
        try:
            ph = dhash(img.data)
        except Exception:  # pragma: no cover - undecodable asset
            ph = "0" * 16
        match = next((k for k in kept if hamming(k.phash, ph) <= threshold), None)
        if match is not None:
            match.duplicates += 1
            seen_sha[sha] = match
            continue
        ded = DedupedImage(image=img, phash=ph)
        kept.append(ded)
        seen_sha[sha] = ded
    return kept
