"""Cross-encoder reranking: jointly scores (query, passage) pairs for precise ordering."""

from __future__ import annotations

from typing import Protocol

from citesure.retrieval.dense import accelerator_kwargs, chunk_embedding_text
from citesure.types import ScoredChunk


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str, batch_size: int = 16, max_length: int = 512) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, max_length=max_length, **accelerator_kwargs())
        self.batch_size = batch_size

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        pairs = [(query, chunk_embedding_text(c.chunk)) for c in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        reranked = [
            ScoredChunk(c.chunk, float(s), {**c.signals, "rerank": float(s)})
            for c, s in zip(candidates, scores, strict=True)
        ]
        reranked.sort(key=lambda s: s.score, reverse=True)
        return reranked[:top_k]
