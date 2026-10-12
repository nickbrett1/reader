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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .schema import COHERE_IMAGE_USD_PER_MTOK, EMBED_CONFIG_VERSION


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


def split_batches(
    items: Sequence[Input],
    max_batch: int = 64,
    max_bytes: int = 16 * 1024 * 1024,
) -> list[list[Input]]:
    """Split into gateway-sized batches by count and by payload bytes."""
    batches: list[list[Input]] = []
    cur: list[Input] = []
    cur_bytes = 0
    for item in items:
        size = _item_payload_size(item)
        if cur and (len(cur) >= max_batch or cur_bytes + size > max_bytes):
            batches.append(cur)
            cur, cur_bytes = [], 0
        cur.append(item)
        cur_bytes += size
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
        http: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.timeout = timeout
        self.max_batch = max_batch
        self.max_bytes = max_bytes
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
        for batch in split_batches(items, self.max_batch, self.max_bytes):
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
        if images and not usage:
            # Rule 3: the gateway reports nothing useful for image tokens.
            usage = {"image_tokens_reported": 0, "note": "measure out-of-band"}
        return EmbedResult(vectors=vectors, model=self.model, dimensions=dims, usage=usage)

    def embed_texts(self, texts: Sequence[str]) -> EmbedResult:
        return self.embed([TextInput(t) for t in texts])

    def embed_images(self, images: Sequence[tuple[str, bytes]]) -> EmbedResult:
        return self.embed([ImageInput(mime, data) for mime, data in images])

    # --- internals ----------------------------------------------------------

    def _to_input(self, item: Input) -> str:
        if isinstance(item, TextInput):
            return item.text
        return data_uri(item.mime, item.data)

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
