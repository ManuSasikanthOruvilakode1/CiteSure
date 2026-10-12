"""Offline test doubles: a hashing embedder and a scripted LLM (no model downloads, no API)."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator

import numpy as np
import pytest

from citesure.retrieval.bm25 import tokenize
from citesure.types import Chunk


class HashingEmbedder:
    """Bag-of-words hashed into a fixed-size, L2-normalised vector."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim))

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


class ScriptedLLM:
    """Returns queued responses; structured() returns queued pydantic objects by schema."""

    def __init__(self, text: str = "", structured: dict | None = None) -> None:
        self.text = text
        self.structured_responses = structured or {}
        self.calls: list[tuple[str, str]] = []

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        self.calls.append((system, prompt))
        yield from (w + " " for w in self.text.split(" "))

    def complete(self, system: str, prompt: str) -> str:
        return "".join(self.stream(system, prompt))

    def structured(self, system: str, prompt: str, schema):
        self.calls.append((system, prompt))
        value = self.structured_responses[schema.__name__]
        # A callable builds the response from the (possibly per-call) schema.
        return value(schema) if callable(value) else value


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def chunks() -> list[Chunk]:
    texts = {
        "auth": "API keys are rotated from the dashboard under Settings > Keys. Old keys stay valid for 24 hours.",
        "rate": "The default rate limit is 600 requests per minute per key. Exceeding it returns HTTP 429.",
        "webhook": "Webhooks are signed with HMAC-SHA256 using the endpoint secret in the X-Signature header.",
        "pagination": "List endpoints use cursor pagination via the next_cursor field and a limit of up to 100.",
        "retry": "Clients should retry 5xx errors with exponential backoff and honour the Retry-After header.",
    }
    return [Chunk(chunk_id=k, text=v, source=f"{k}.md", section=k.title()) for k, v in texts.items()]
