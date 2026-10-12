# CiteSure: Architecture

> **Status: pipeline implemented (v0.1.0). No evaluation has been run yet, so no measurements exist.** Numbers in this file are configuration values or targets, and are labelled as such.

## 1. Goal

Answer questions about technical documentation (API references, manuals) with answers that cite the retrieved passages, and measure answer quality with reproducible metrics instead of manual spot checks. Each retrieval or prompt change is evaluated against the same held-out question set before it is accepted.

## 2. Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                 RAG PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│   [ Technical documents: Markdown / PDF ]                                   │
│                 │                                                           │
│                 ▼                                                           │
│   [ Ingestion & chunking ]                                                  │
│   • Markdown-header-aware splitting, then recursive splitting to a token    │
│     budget with overlap (sizes to be tuned on the evaluation set)           │
│   • Metadata per chunk: source file, section path                           │
│                 │                                                           │
│                 ▼                                                           │
│   [ Indexing ]                                                              │
│   ├── Sparse: BM25 (rank-bm25)                                              │
│   └── Dense: BAAI/bge-large-en-v1.5 (in-process numpy, or Qdrant)           │
│                 │                                                           │
│                 ▼                                                           │
│   [ Retrieval ]                                                             │
│   • Top-30 BM25 + top-30 dense, merged with Reciprocal Rank Fusion (k = 60) │
│   • Cross-encoder rerank (BAAI/bge-reranker-large) to the top 5 chunks      │
│                 │                                                           │
│                 ▼                                                           │
│   [ Generation ]                                                            │
│   • Local LLM (qwen2.5:7b via Ollama), prompt requires [n] citations and a  │
│     refusal when the retrieved context does not contain the answer          │
│                 │                                                           │
│                 ▼                                                           │
│   [ Evaluation: RAG Triad ]                                                 │
│   ├── Context relevance                                                     │
│   ├── Faithfulness (groundedness)                                           │
│   └── Answer relevance                                                      │
│                 │                                                           │
│                 ▼                                                           │
│   [ Serving & regression gate ]                                             │
│   • FastAPI endpoint, streamed answer + cited chunks                        │
│   • CI job runs the evaluation set and fails below the configured thresholds│
└─────────────────────────────────────────────────────────────────────────────┘
```

Retrieval and model settings live in [`configs/default.yaml`](configs/default.yaml). Because each synthetic question is written from one chunk, the evaluation also reports retrieval hit rate and MRR for that source chunk, which need no judge.

## 3. Metrics (RAG Triad)

**Context relevance.** The share of retrieved context that is needed to answer the question. A low score means the retriever is noisy; the levers are reranking, smaller chunks and fewer retrieved chunks. Implemented per passage, as in TruLens: the judge grades each retrieved passage 0–3 and the score is the mean grade divided by 3. A sentence-level ratio would be systematically low for 512-token chunks, because most sentences in a relevant chunk are not themselves needed.

$$\text{Context relevance} = \frac{1}{k}\sum_{i=1}^{k} \frac{\text{grade}(p_i)}{3}, \quad \text{grade} \in \{0,1,2,3\}$$

**Faithfulness (groundedness).** The share of claims in the answer that are supported by the retrieved context. A low score means the generator adds unsupported content; the levers are the citation requirement and the refusal instruction. Implemented in two judge calls: extract atomic claims, then verify each against the context. An answer with no claims (a refusal) scores 1.

$$\text{Faithfulness} = \frac{|\{c \in \text{answer claims} : c \text{ is supported by the context}\}|}{|\text{answer claims}|}$$

**Answer relevance.** How directly the answer addresses the question. Implemented as in Ragas: the judge writes three questions the answer would respond to, and the score is their mean cosine similarity to the original question under the BGE embedder, set to 0 when the answer is non-committal.

These metrics are computed by an LLM judge (the local model, with JSON-schema-constrained output), so they inherit its errors. The synthetic evaluation set is written by the same model, and a set generated and judged by the same model tends to overstate quality. The plan is to hand-label a subset of the evaluation set and report how well the judge agrees with the human labels, alongside the scores.

## 4. Experiments to run

1. **Retrieval ablation.** Dense only, BM25 only, hybrid with RRF, and hybrid with reranking, all on the same question set. Report the three metrics plus p95 latency for each.
2. **Chunking.** Two or three chunk sizes and overlaps with the best retrieval configuration.
3. **Regression gate.** Set the CI thresholds from the measured scores of the chosen configuration minus a tolerance. The values currently in the config (0.85 / 0.92 / 0.90) are placeholders until then.

## 5. Layout

```
configs/default.yaml         Model names, retrieval parameters, evaluation thresholds
src/citesure/ingestion/      Markdown/PDF/text loaders, header-aware recursive chunker
src/citesure/retrieval/      BM25, dense stores (numpy, Qdrant), RRF, cross-encoder, hybrid
src/citesure/generation/     Ollama client, citation prompt, generator
src/citesure/evaluation/     RAG Triad judge metrics, ablation runner, CI gate
src/citesure/serving/        FastAPI: POST /query, POST /query/stream (SSE), GET /health
src/citesure/pipeline.py     Index build/load and end-to-end question answering
scripts/                     build_index, generate_synthetic_data, evaluate_pipeline, ask
data/documents/              Sample corpus (fictional API docs), to be replaced
data/evaluation/             QA set, created by generate_synthetic_data.py
tests/                       Offline unit tests (hashing embedder, scripted LLM)
```
