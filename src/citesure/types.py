"""Core data structures shared across the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Document:
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    section: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        return cls(**data)


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    # Per-stage scores/ranks for debugging and attribution (e.g. {"bm25_rank": 3}).
    signals: dict[str, float] = field(default_factory=dict)


@dataclass
class RAGAnswer:
    question: str
    answer: str
    contexts: list[ScoredChunk]
    latency_s: float = 0.0
