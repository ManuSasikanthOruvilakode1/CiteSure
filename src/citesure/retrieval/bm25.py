"""Sparse lexical retrieval with Okapi BM25."""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from citesure.types import Chunk, ScoredChunk

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, keeping identifiers like ``max_retries`` or ``v1.2`` whole."""
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        corpus = [tokenize(f"{c.section}\n{c.text}") for c in chunks]
        self._bm25 = BM25Okapi(corpus, k1=k1, b=b) if chunks else None

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        tokens = tokenize(query)
        if self._bm25 is None or not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        order = np.argsort(-scores)[:top_k]
        return [
            ScoredChunk(self.chunks[i], float(scores[i]), {"bm25": float(scores[i])})
            for i in order
            if scores[i] > 0
        ]
