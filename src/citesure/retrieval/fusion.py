"""Reciprocal Rank Fusion (Cormack et al., 2009).

    RRF(d) = sum over rankers r of  1 / (k + rank_r(d))

Rank-based, so BM25 scores and cosine similarities need no calibration against each other.
"""

from __future__ import annotations

from citesure.types import ScoredChunk


def reciprocal_rank_fusion(
    rankings: dict[str, list[ScoredChunk]], k: int = 60, top_k: int | None = None
) -> list[ScoredChunk]:
    fused: dict[str, ScoredChunk] = {}
    for name, ranking in rankings.items():
        for rank, item in enumerate(ranking, start=1):
            cid = item.chunk.chunk_id
            entry = fused.get(cid)
            if entry is None:
                entry = fused[cid] = ScoredChunk(item.chunk, 0.0, {})
            entry.score += 1.0 / (k + rank)
            entry.signals.update(item.signals)
            entry.signals[f"{name}_rank"] = rank
    for entry in fused.values():
        entry.signals["rrf"] = entry.score
    ranked = sorted(fused.values(), key=lambda s: s.score, reverse=True)
    return ranked[:top_k] if top_k else ranked
