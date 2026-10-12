"""Environment + path configuration.

The app owns its entry point, so all configuration is resolved here from the
environment (populated by Doppler in production, by ``.env`` locally) and
handed to the other modules as a frozen object.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_GATEWAY_URL = "http://nas:4000"
DEFAULT_INGEST = "/ingest"
DEFAULT_DATA = "/data"


@dataclass(frozen=True)
class Config:
    """Resolved runtime configuration."""

    ingest_dir: Path
    data_dir: Path
    litellm_base_url: str
    litellm_master_key: str | None
    turbopuffer_api_key: str | None

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Config:
        e = os.environ if env is None else env
        return cls(
            ingest_dir=Path(e.get("READER_INGEST", DEFAULT_INGEST)),
            data_dir=Path(e.get("READER_DATA", DEFAULT_DATA)),
            litellm_base_url=e.get("LITELLM_BASE_URL", DEFAULT_GATEWAY_URL).rstrip("/"),
            litellm_master_key=e.get("LITELLM_MASTER_KEY") or None,
            turbopuffer_api_key=e.get("TURBOPUFFER_API_KEY") or None,
        )

    # --- derived paths ------------------------------------------------------

    @property
    def db_path(self) -> Path:
        return self.data_dir / "reader.db"

    @property
    def images_dir(self) -> Path:
        """Content-addressed blob store root."""
        return self.data_dir / "images"

    def image_path(self, sha256: str, ext: str) -> Path:
        """``images/<sha256[0:2]>/<sha256>.<ext>`` for a content hash."""
        return self.images_dir / sha256[:2] / f"{sha256}.{ext}"

    def embed_endpoint(self) -> str:
        return f"{self.litellm_base_url}/v1/embeddings"

    def chat_endpoint(self) -> str:
        return f"{self.litellm_base_url}/v1/chat/completions"
