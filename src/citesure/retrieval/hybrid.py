"""Hybrid retriever: BM25 + dense -> RRF -> optional cross-encoder rerank.

The ``mode`` switch reproduces each row of the benchmark table so ablations share code.
"""

from __future__ import annotations

from enum import Enum

from citesure.retrieval.bm25 import BM25Index
from citesure.retrieval.dense import Embedder, VectorStore
from citesure.retrieval.fusion import reciprocal_rank_fusion
from citesure.retrieval.reranker import Reranker
from citesure.types import ScoredChunk


class RetrievalMode(str, Enum):
    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


class HybridRetriever:
    def __init__(
        self,
        bm25: BM25Index,
        vector_store: VectorStore,
        embedder: Embedder,
        reranker: Reranker | None = None,
        top_k_dense: int = 30,
        top_k_sparse: int = 30,
        top_k_fused: int = 30,
        top_k_final: int = 5,
        rrf_k: int = 60,
        min_rerank_score: float = 0.0,
    ) -> None:
        self.bm25 = bm25
        self.vector_store = vector_store
        self.embedder = embedder
        self.reranker = reranker
        self.top_k_dense = top_k_dense
        self.top_k_sparse = top_k_sparse
        self.top_k_fused = top_k_fused
        self.top_k_final = top_k_final
        self.rrf_k = rrf_k
        self.min_rerank_score = min_rerank_score

    def retrieve(
        self, query: str, mode: RetrievalMode = RetrievalMode.HYBRID_RERANK
    ) -> list[ScoredChunk]:
        mode = RetrievalMode(mode)
        if mode is RetrievalMode.BM25:
            return self.bm25.search(query, self.top_k_final)
        dense = self.vector_store.search(self.embedder.embed_query(query), self.top_k_dense)
        if mode is RetrievalMode.DENSE:
            return dense[: self.top_k_final]

        sparse = self.bm25.search(query, self.top_k_sparse)
        fused = reciprocal_rank_fusion(
            {"bm25": sparse, "dense": dense}, k=self.rrf_k, top_k=self.top_k_fused
        )
        if mode is RetrievalMode.HYBRID or self.reranker is None:
            return fused[: self.top_k_final]
        reranked = self.reranker.rerank(query, fused, self.top_k_final)
        # Cross-encoder scores are calibrated probabilities, so a fixed floor separates
        # relevant passages from filler. An empty result makes the generator refuse
        # without calling the LLM.
        return [c for c in reranked if c.score >= self.min_rerank_score]
