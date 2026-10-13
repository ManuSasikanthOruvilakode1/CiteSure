"""Sweep retrieval settings and score them with judge-free metrics (Hit@5, MRR).

Only retrieval settings are swept (chunk size, reranker candidate count, RRF k); the
models are unchanged. Each chunk size gets its own in-memory index, and a hit is
matched on the question's evidence text, so labels stay valid across chunkings.
Tune on the dev set only; confirm the chosen setting once on the held-out set.

Usage:
  python scripts/sweep_retrieval.py --dataset data/evaluation/qa_pairs.jsonl
  python scripts/sweep_retrieval.py --dataset data/evaluation/qa_heldout.jsonl \\
      --chunk-sizes 512 --candidates 30 --rrf-ks 60 --modes all
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from citesure.config import PROJECT_ROOT, load_settings
from citesure.evaluation.stats import bootstrap_ci, evidence_rank
from citesure.ingestion import RecursiveChunker, load_directory
from citesure.pipeline import make_embedder
from citesure.retrieval.bm25 import BM25Index
from citesure.retrieval.dense import NumpyVectorStore, chunk_embedding_text
from citesure.retrieval.hybrid import HybridRetriever, RetrievalMode
from citesure.retrieval.reranker import CrossEncoderReranker


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--chunk-sizes", type=_ints, default=[256, 384, 512])
    parser.add_argument("--candidates", type=_ints, default=[10, 20, 30], help="fused passages sent to the reranker")
    parser.add_argument("--rrf-ks", type=_ints, default=[20, 60])
    parser.add_argument("--modes", default="hybrid_rerank", help="comma list, or 'all'")
    parser.add_argument("--out", type=Path, default=None, help="optional JSON file for the results")
    args = parser.parse_args()

    settings = load_settings()
    modes = list(RetrievalMode) if args.modes == "all" else [RetrievalMode(m) for m in args.modes.split(",")]
    rows = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    docs = load_directory(settings.resolve(settings.paths.documents_dir))
    embedder = make_embedder(settings)
    reranker = CrossEncoderReranker(settings.models.reranker_model)
    r = settings.retrieval

    results = []
    print("| Chunk size | Candidates | RRF k | Mode | Hit@5 [95% CI] | MRR [95% CI] | Retrieval ms/query |")
    print("|---|---|---|---|---|---|---|")
    for size in args.chunk_sizes:
        chunks = RecursiveChunker(size, round(size * 0.1)).chunk_documents(docs)
        store = NumpyVectorStore()
        store.add(chunks, embedder.embed_documents([chunk_embedding_text(c) for c in chunks]))
        bm25 = BM25Index(chunks)
        for cand in args.candidates:
            for rrf_k in args.rrf_ks:
                retriever = HybridRetriever(
                    bm25, store, embedder, reranker,
                    top_k_dense=r.top_k_dense, top_k_sparse=r.top_k_sparse, top_k_fused=cand,
                    top_k_final=r.top_k_reranked, rrf_k=rrf_k, min_rerank_score=r.min_rerank_score,
                )
                for mode in modes:
                    hits, rr = [], []
                    start = time.perf_counter()
                    for row in rows:
                        texts = [c.chunk.text for c in retriever.retrieve(row["question"], mode)]
                        rank = evidence_rank(texts, row["evidence"])
                        hits.append(1.0 if rank else 0.0)
                        rr.append(1.0 / rank if rank else 0.0)
                    ms = (time.perf_counter() - start) / len(rows) * 1000
                    h, m = bootstrap_ci(hits), bootstrap_ci(rr)
                    results.append({"chunk_size": size, "candidates": cand, "rrf_k": rrf_k, "mode": mode.value,
                                    "chunks": len(chunks), "hit_at_5": h[0], "mrr": m[0], "ms_per_query": ms})
                    print(f"| {size} | {cand} | {rrf_k} | {mode.value} | {h[0]:.2f} [{h[1]:.2f}, {h[2]:.2f}] "
                          f"| {m[0]:.3f} [{m[1]:.2f}, {m[2]:.2f}] | {ms:.0f} |", flush=True)
    if args.out:
        args.out.write_text(json.dumps({"dataset": str(args.dataset.relative_to(PROJECT_ROOT) if args.dataset.is_absolute() else args.dataset),
                                        "results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
