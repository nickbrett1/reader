"""The ONE embedding client.

Every embedding in the system goes through this module and the LiteLLM gateway
(``POST /v1/embeddings`` on ``nas:4000``).  Turbopuffer stores *supplied*
vectors and never embeds anything itself.

The four house rules (v5 §5.4) are enforced here because each is a silent
failure otherwise:

1. images are wrapped as ``data:<mime>;base64,…`` — bare base64 is embedded as
   *text* (wrong vector, no error);
2. text and image are never mixed in one ``input`` array — the gateway's Cohere
   handler is not version-branched and returns HTTP 422 when they are mixed;
   :meth:`EmbedClient.embed` rejects a mixed batch client-side, before the
   round-trip;
3. gateway-side image-token accounting reports 0, so cost must be measured
   out-of-band;
4. budget images at Cohere's list rate, not the gateway's (see
   :func:`image_cost_usd`).
"""

from __future__ import annotations

import base64
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .schema import (
    COHERE_IMAGE_USD_PER_MTOK,
    EMBED_CONFIG_VERSION,
    IMAGE_MAX_PIXELS,
)

#: Cohere / gateway hard limit is 50,000,000 decoded pixels per batch.
#: We cap at 40 MP to keep a comfortable safety margin.
MAX_IMAGE_PIXELS_PER_BATCH = 40_000_000

#: Cohere's tokenizer rejects *its own* special tokens (``<|endoftext|>``,
#: ``<|start_header_id|>``, …) when they appear as literal text; the gateway
#: surfaces this as HTTP 500 (observed live 2026-10-12 while embedding the
#: widened corpus — an LLM book quotes ``<|endoftext|>`` in prose).  Strip any
#: ``<|…|>`` marker from text before embedding.  Applied only to the payload
#: sent to the gateway, never to stored text, so chunk offsets stay intact.
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]{0,64}\|>")


def strip_special_tokens(text: str) -> str:
    """Remove Cohere-special-token-shaped markers from text before embedding."""
    return _SPECIAL_TOKEN_RE.sub("", text)


class MixedModalityError(ValueError):
    """Raised when one ``input`` array would mix text and image."""


class EmbedError(RuntimeError):
    """Raised when the gateway returns an error or an unparseable payload."""


@dataclass(frozen=True)
class TextInput:
    text: str


@dataclass(frozen=True)
class ImageInput:
    mime: str
    data: bytes


Input = TextInput | ImageInput


def data_uri(mime: str, data: bytes) -> str:
    """Rule 1: always a ``data:`` URI, never bare base64."""
    if not mime.startswith("image/"):
        raise ValueError(f"not an image mime: {mime!r}")
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def resize_image(mime: str, data: bytes, max_pixels: int = IMAGE_MAX_PIXELS) -> bytes:
    """Downscale an encoded image so ``w * h <= max_pixels``.

    * Aspect ratio is preserved; the resize uses PIL's deterministic LANCZOS
      filter so the same input always yields the same output (stable hashes).
    * An image already at or under the cap is returned **byte-for-byte
      unchanged** — no re-encode.  That keeps dedupe hashes and existing rows
      stable.
    * Only a resized image is re-encoded, back to its original format
      (JPEG -> JPEG, quality 90, preserving the source's chroma subsampling;
      PNG -> PNG).
    * Bytes that PIL cannot decode are passed through untouched, matching
      :func:`_item_pixels` which treats them as unmeasurable rather than fatal.
    """
    import io

    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.width, im.height
            if w * h <= max_pixels:
                return data
            scale = (max_pixels / (w * h)) ** 0.5
            # Derive the *shorter* edge from the scale, then derive the longer
            # edge from it so the aspect ratio survives integer rounding; a
            # naive round of both edges independently can drift by >1px.
            if w >= h:
                nh = max(1, int(h * scale))
                nw = max(1, round(w * nh / h))
            else:
                nw = max(1, int(w * scale))
                nh = max(1, round(h * nw / w))
            # Guard against rounding pushing the product back over the cap.
            while nw * nh > max_pixels and (nw > 1 or nh > 1):
                if nw >= nh:
                    nw -= 1
                else:
                    nh -= 1
            # Re-encode to the original mime; fall back to the decoded format
            # when the mime is something we don't map explicitly.
            fmt = {
                "image/png": "PNG",
                "image/jpeg": "JPEG",
                "image/jpg": "JPEG",
            }.get(mime.lower(), im.format)
            subsampling = None
            if fmt == "JPEG":
                from PIL import JpegImagePlugin

                subsampling = JpegImagePlugin.get_sampling(im)
            resized = im.resize((nw, nh), Image.LANCZOS)
            buf = io.BytesIO()
            if fmt == "JPEG":
                if resized.mode not in ("L", "RGB", "CMYK"):
                    resized = resized.convert("RGB")
                save_kwargs: dict[str, Any] = {"format": "JPEG", "quality": 90}
                if subsampling is not None:
                    save_kwargs["subsampling"] = subsampling
                resized.save(buf, **save_kwargs)
            elif fmt == "PNG":
                resized.save(buf, format="PNG")
            else:
                resized.save(buf, format=fmt)
            return buf.getvalue()
    except Exception:
        return data


def image_cost_usd(tokens: int, usd_per_mtok: float = COHERE_IMAGE_USD_PER_MTOK) -> float:
    """Rule 4: budget image tokens at Cohere's list rate."""
    return tokens / 1_000_000 * usd_per_mtok


@dataclass
class EmbedResult:
    vectors: list[list[float]]
    model: str
    dimensions: int
    usage: dict[str, Any] = field(default_factory=dict)

    def provenance(self) -> dict[str, Any]:
        """Per-row embedding provenance to stamp alongside each vector."""
        return {
            "model_id": self.model,
            "dimensions": self.dimensions,
            "embed_config_version": EMBED_CONFIG_VERSION,
        }


def _item_payload_size(item: Input) -> int:
    return len(item.text) if isinstance(item, TextInput) else len(item.data)


def _item_pixels(item: Input) -> int:
    """Decoded pixel count (w * h) for image inputs; 0 for text or invalid bytes."""
    if not isinstance(item, ImageInput):
        return 0
    import io

    from PIL import Image

    try:
        with Image.open(io.BytesIO(item.data)) as im:
            return int(im.width * im.height)
    except Exception:
        return 0


def split_batches(
    items: Sequence[Input],
    max_batch: int = 64,
    max_bytes: int = 16 * 1024 * 1024,
    max_pixels: int | None = None,
) -> list[list[Input]]:
    """Split into gateway-sized batches by count, payload bytes, and decoded pixels."""
    batches: list[list[Input]] = []
    cur: list[Input] = []
    cur_bytes = 0
    cur_pixels = 0
    for item in items:
        size = _item_payload_size(item)
        pixels = _item_pixels(item) if max_pixels is not None else 0
        if cur and (
            len(cur) >= max_batch
            or cur_bytes + size > max_bytes
            or (max_pixels is not None and cur_pixels + pixels > max_pixels)
        ):
            batches.append(cur)
            cur, cur_bytes, cur_pixels = [], 0, 0
        cur.append(item)
        cur_bytes += size
        cur_pixels += pixels
    if cur:
        batches.append(cur)
    return batches


class EmbedClient:
    """Batched, modality-uniform embedding client."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        dimensions: int | None = None,
        *,
        timeout: float = 60.0,
        max_batch: int = 64,
        max_bytes: int = 16 * 1024 * 1024,
        max_pixels: int | None = MAX_IMAGE_PIXELS_PER_BATCH,
        http: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.timeout = timeout
        self.max_batch = max_batch
        self.max_bytes = max_bytes
        self.max_pixels = max_pixels
        #: Injectable transport for tests; when None, httpx is used.
        self._http = http

    # --- public -------------------------------------------------------------

    def embed(self, items: Sequence[Input]) -> EmbedResult:
        """Embed a homogeneous batch, splitting into gateway batches."""
        if not items:
            return EmbedResult(vectors=[], model=self.model, dimensions=self.dimensions or 0)
        modalities = {type(i) for i in items}
        if len(modalities) > 1:
            # Rule 2, enforced client-side before the gateway can 422.
            raise MixedModalityError(
                "text and image inputs must be embedded in separate calls"
            )
        images = next(iter(modalities)) is ImageInput

        vectors: list[list[float]] = []
        usage: dict[str, Any] = {}
        dims = self.dimensions or 0
        pix_limit = self.max_pixels if images else None
        for batch in split_batches(
            items, self.max_batch, self.max_bytes, max_pixels=pix_limit
        ):
            payload = [self._to_input(i) for i in batch]
            resp = self._post({"model": self.model, "input": payload})
            data = resp.get("data")
            if not isinstance(data, list):
                raise EmbedError(f"gateway returned no data array: {resp!r}")
            ordered = sorted(data, key=lambda d: d.get("index", 0))
            for row in ordered:
                vec = row.get("embedding")
                if not isinstance(vec, list):
                    raise EmbedError(f"missing embedding in row: {row!r}")
                vectors.append(vec)
                dims = len(vec)
            self._merge_usage(usage, resp.get("usage", {}))
        if images:
            # Rule 3, confirmed live (2026-10-12): the gateway returns
            # prompt_tokens=0 / total_tokens=0 / image_tokens=null for an image
            # embed, so its accounting can never be trusted for images.
            usage["image_tokens_reliable"] = False
            usage["note"] = "gateway reports 0 image tokens; measure out-of-band"
        return EmbedResult(vectors=vectors, model=self.model, dimensions=dims, usage=usage)

    def embed_texts(self, texts: Sequence[str]) -> EmbedResult:
        return self.embed([TextInput(t) for t in texts])

    def embed_images(self, images: Sequence[tuple[str, bytes]]) -> EmbedResult:
        return self.embed([ImageInput(mime, data) for mime, data in images])

    # --- internals ----------------------------------------------------------

    def _to_input(self, item: Input) -> str:
        if isinstance(item, TextInput):
            return strip_special_tokens(item.text)
        return data_uri(item.mime, resize_image(item.mime, item.data))

    @staticmethod
    def _merge_usage(acc: dict[str, Any], new: dict[str, Any]) -> None:
        for key, value in (new or {}).items():
            if isinstance(value, (int, float)):
                acc[key] = acc.get(key, 0) + value
            else:
                acc[key] = value

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._http is not None:
            return self._http(payload)
        import httpx

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            r = httpx.post(
                f"{self.base_url}/v1/embeddings",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:  # pragma: no cover - network path
            raise EmbedError(f"gateway {exc.response.status_code}: {exc.response.text}") from exc
        except httpx.HTTPError as exc:  # pragma: no cover - network path
            raise EmbedError(f"gateway request failed: {exc}") from exc
        return r.json()
