import pytest

from citesure.retrieval.bm25 import BM25Index, tokenize
from citesure.retrieval.dense import NumpyVectorStore
from citesure.retrieval.fusion import reciprocal_rank_fusion
from citesure.retrieval.hybrid import HybridRetriever, RetrievalMode
from citesure.types import Chunk, ScoredChunk


def _ranking(ids):
    return [ScoredChunk(Chunk(i, i, "s"), 1.0) for i in ids]


def test_tokenize_keeps_identifiers():
    assert tokenize("Set max_retries=3 on v1.2 API") == ["set", "max_retries", "3", "on", "v1.2", "api"]


def test_rrf_rewards_agreement_across_rankers():
    fused = reciprocal_rank_fusion({"a": _ranking(["x", "y", "z"]), "b": _ranking(["y", "x", "w"])}, k=60)
    assert {fused[0].chunk.chunk_id, fused[1].chunk.chunk_id} == {"x", "y"}
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert fused[-1].chunk.chunk_id in {"z", "w"}
    assert fused[0].signals.keys() >= {"a_rank", "b_rank", "rrf"}


def test_rrf_top_k():
    assert len(reciprocal_rank_fusion({"a": _ranking("abcdef")}, top_k=3)) == 3


def test_bm25_finds_exact_terms(chunks):
    results = BM25Index(chunks).search("what does HTTP 429 mean", top_k=3)
    assert results[0].chunk.chunk_id == "rate"


def test_bm25_empty_query_returns_nothing(chunks):
    assert BM25Index(chunks).search("???", top_k=3) == []


def test_numpy_store_round_trip(tmp_path, chunks, embedder):
    store = NumpyVectorStore()
    store.add(chunks, embedder.embed_documents([c.text for c in chunks]))
    store.save(tmp_path / "e.npy")
    loaded = NumpyVectorStore.load(chunks, tmp_path / "e.npy")
    hit = loaded.search(embedder.embed_query("how are webhooks signed"), top_k=1)[0]
    assert hit.chunk.chunk_id == "webhook"


class ReverseReranker:
    """Reranker double that inverts the incoming order, to prove it was applied."""

    def rerank(self, query, candidates, top_k):
        return list(reversed(candidates))[:top_k]


@pytest.fixture
def retriever(chunks, embedder):
    store = NumpyVectorStore()
    store.add(chunks, embedder.embed_documents([c.text for c in chunks]))
    return HybridRetriever(
        BM25Index(chunks), store, embedder, reranker=ReverseReranker(),
        top_k_dense=5, top_k_sparse=5, top_k_fused=5, top_k_final=3,
    )


@pytest.mark.parametrize("mode", list(RetrievalMode))
def test_every_mode_returns_bounded_results(retriever, mode):
    results = retriever.retrieve("rotate API keys", mode)
    assert 0 < len(results) <= 3


def test_hybrid_ranks_relevant_chunk_first(retriever):
    assert retriever.retrieve("how do I rotate API keys", RetrievalMode.HYBRID)[0].chunk.chunk_id == "auth"


def test_rerank_mode_applies_reranker(retriever):
    hybrid = retriever.retrieve("rotate API keys", RetrievalMode.HYBRID)
    reranked = retriever.retrieve("rotate API keys", RetrievalMode.HYBRID_RERANK)
    assert reranked[0].chunk.chunk_id != hybrid[0].chunk.chunk_id


class FixedScoreReranker:
    """Reranker double that scores passages from a lookup, defaulting to irrelevant."""

    def __init__(self, scores):
        self.scores = scores

    def rerank(self, query, candidates, top_k):
        scored = [ScoredChunk(c.chunk, self.scores.get(c.chunk.chunk_id, 0.0)) for c in candidates]
        return sorted(scored, key=lambda s: s.score, reverse=True)[:top_k]


def _thresholded(chunks, embedder, scores, floor=0.02):
    store = NumpyVectorStore()
    store.add(chunks, embedder.embed_documents([c.text for c in chunks]))
    return HybridRetriever(
        BM25Index(chunks), store, embedder, reranker=FixedScoreReranker(scores),
        top_k_final=5, min_rerank_score=floor,
    )


def test_rerank_floor_drops_irrelevant_passages(chunks, embedder):
    retriever = _thresholded(chunks, embedder, {"auth": 0.97, "rate": 0.09, "retry": 0.004})
    ids = [c.chunk.chunk_id for c in retriever.retrieve("rotate API keys", RetrievalMode.HYBRID_RERANK)]
    assert ids == ["auth", "rate"]


def test_rerank_floor_can_return_nothing(chunks, embedder):
    retriever = _thresholded(chunks, embedder, {})
    assert retriever.retrieve("rotate API keys", RetrievalMode.HYBRID_RERANK) == []
    # The floor applies only to calibrated cross-encoder scores, not other modes.
    assert retriever.retrieve("rotate API keys", RetrievalMode.HYBRID)
