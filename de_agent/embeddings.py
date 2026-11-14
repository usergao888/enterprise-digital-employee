"""Embedding backends.

`hashing` is a deterministic, dependency-free embedder (hashing trick over the
CJK bigram tokeniser). It is not a semantic model, but it is a *real* vector
space with the same interface, which lets the retrieval stack be unit-tested and
demoed offline. For production quality swap in `openai_compat` (DashScope
text-embedding-v3 / BGE-M3 served by any OpenAI-compatible gateway).
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
import zlib
from collections import Counter
from typing import Protocol, Sequence

from .config import Settings
from .textutil import tokenize


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Hashing-trick embedder with sublinear term weighting + L2 norm."""

    name = "hashing"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        counts = Counter(tokenize(text))
        if not counts:
            return vector
        for token, tf in counts.items():
            digest = zlib.crc32(token.encode("utf-8"))
            index = digest % self.dim
            sign = 1.0 if (digest >> 16) & 1 else -1.0
            vector[index] += sign * (1.0 + math.log(tf))
        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0:
            vector = [value / norm for value in vector]
        return vector

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(text) for text in texts]


class OpenAICompatEmbedder:
    """Any OpenAI-compatible /embeddings endpoint."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.dim = settings.embed_dim
        self.name = f"openai_compat:{settings.embed_model}"
        self.base_url = settings.llm_base_url.rstrip("/")

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = json.dumps(
            {"model": self.settings.embed_model, "input": list(texts)},
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.settings.llm_api_key}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.settings.llm_timeout_s) as response:
                data = json.loads(response.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - network path
            raise RuntimeError(
                f"embedding HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:300]}"
            ) from exc
        except urllib.error.URLError as exc:  # pragma: no cover - network path
            raise RuntimeError(f"embedding 无法连接: {exc.reason}") from exc

        items = sorted(data.get("data", []), key=lambda item: item.get("index", 0))
        vectors = [item["embedding"] for item in items]
        if vectors:
            self.dim = len(vectors[0])
        return vectors


def build_embedder(settings: Settings) -> Embedder:
    if settings.embed_backend == "openai_compat":
        return OpenAICompatEmbedder(settings)
    return HashingEmbedder(settings.embed_dim)
