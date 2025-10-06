"""Placeholder Sora 2 client to ease future API wiring."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Sora2Config:
    """Configuration for the future Sora 2 API client."""

    api_key: Optional[str] = None
    base_url: str = "https://api.openai.com/v1"
    model: str = "sora-2-preview"
    request_timeout: float = 30.0


class Sora2Client:
    """Lightweight client stub that keeps the UI future-proof."""

    def __init__(self, config: Sora2Config | None = None) -> None:
        self._config = config or Sora2Config()

    def config(self) -> Sora2Config:
        return self._config

    def is_configured(self) -> bool:
        """True if an API key has been provided."""
        return bool(self._config.api_key)

    # ------------------------------------------------------------------
    # Future API surface area
    def generate_video(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Placeholder for the streaming video generation call."""
        raise NotImplementedError(
            "Sora 2 public API is not yet available. "
            "Once OpenAI publishes the REST/streaming spec, implement this method."
        )

    def stream_frames(self, *_, **__) -> Any:
        """Reserved for future streaming frame retrieval."""
        raise NotImplementedError("Streaming is not available: awaiting official Sora 2 endpoints.")


__all__ = ["Sora2Client", "Sora2Config"]
