# CiteSure

**Grounded document Q&A with continuous faithfulness evaluation.**

> **Status: pipeline implemented (v0.1.0); not yet evaluated.** Ingestion, hybrid retrieval, reranking, generation, RAG Triad evaluation, the CI gate and the API server are built and unit-tested. The indexed corpus is a small set of **sample** docs for a fictional API, and no evaluation run has been reported, so **no results exist yet**.

A Retrieval-Augmented Generation (RAG) system for question answering over technical documentation. It combines hybrid sparse/dense retrieval with cross-encoder reranking, generates cited answers with a local LLM, and evaluates every change against the **RAG Triad**: context relevance, faithfulness and answer relevance. Everything runs locally at no cost.

```
Query ──► Hybrid retrieval (BM25 + dense, RRF) ──► Cross-encoder rerank ──► LLM generation ──► RAG Triad evaluation
```

## Components

| Stage | Implementation | Status |
|---|---|---|
| Ingestion | Markdown/PDF/text loaders; header-aware recursive chunking (512 tokens, ~10% overlap) with section-path metadata | Done |
| Sparse retrieval | BM25 Okapi (`rank-bm25`), identifier-preserving tokenizer | Done |
| Dense retrieval | `BAAI/bge-large-en-v1.5`, in-process numpy store (default) or Qdrant | Done |
| Fusion + rerank | Reciprocal Rank Fusion (k = 60) over top-30 + top-30, then `BAAI/bge-reranker-large` to top 5; passages scoring below 0.02 are dropped | Done |
| Generation | Local LLM via Ollama (`qwen2.5:7b`); numbered-passage citations; refuses, without calling the LLM, when no passage is relevant | Done |
| Evaluation | LLM-judge RAG Triad, plus retrieval hit rate/MRR; ablation across four retrieval modes | Done |
| Regression gate | `scripts/evaluate_pipeline.py` exits non-zero below thresholds; wired into GitHub Actions | Done |
| Serving | FastAPI `POST /query`, `POST /query/stream` (SSE with sources, tokens, citations) | Done |
| Real corpus + reviewed eval set | | Not started |

## Project structure

```
citesure/
├── .github/
│   └── workflows/
│       └── ci.yml                   # Lint + tests on every PR; opt-in RAG Triad gate
├── configs/
│   └── default.yaml                 # Models, chunking, retrieval, thresholds, server
├── data/
│   ├── documents/                   # Corpus to index (Markdown, PDF, plain text)
│   │   ├── sample_api_authentication.md
│   │   ├── sample_api_reliability.md
│   │   └── sample_webhooks.md
│   └── evaluation/                  # qa_pairs.jsonl, created by generate_synthetic_data.py
├── scripts/
│   ├── build_index.py               # Load, chunk, embed and save the corpus
│   ├── ask.py                       # Ask one question from the command line
│   ├── generate_synthetic_data.py   # Build the QA evaluation set from indexed chunks
│   └── evaluate_pipeline.py         # Run the RAG Triad evaluation and quality gate
├── src/
│   └── citesure/
│       ├── __init__.py
│       ├── config.py                # Typed settings loaded from YAML
│       ├── types.py                 # Document, Chunk, ScoredChunk, RAGAnswer
│       ├── pipeline.py              # Index build/load and end-to-end answering
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── loaders.py           # Markdown / PDF / text loaders
│       │   └── chunker.py           # Header-aware recursive chunker with overlap
│       ├── retrieval/
│       │   ├── __init__.py
│       │   ├── bm25.py              # Sparse keyword index (BM25 Okapi)
│       │   ├── dense.py             # BGE embedder; numpy and Qdrant vector stores
│       │   ├── fusion.py            # Reciprocal Rank Fusion
│       │   ├── reranker.py          # Cross-encoder reranker
│       │   └── hybrid.py            # Retrieval modes: dense, bm25, hybrid, hybrid_rerank
│       ├── generation/
│       │   ├── __init__.py
│       │   ├── llm.py               # LLM interface and Ollama client
│       │   ├── prompts.py           # Citation-enforcing prompt templates
│       │   └── generator.py         # Answer generation and citation extraction
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py           # Context relevance, faithfulness, answer relevance
│       │   └── runner.py            # Per-mode evaluation, reports, quality gate
│       └── serving/
│           ├── __init__.py
│           ├── app.py               # FastAPI app: demo page, /query, /query/stream, /health
│           └── static/
│               └── index.html       # Browser demo: streamed answers with linked citations
├── tests/                           # Offline unit tests (no model downloads, no LLM)
│   ├── conftest.py                  # Test doubles: hashing embedder, scripted LLM
│   ├── test_chunker.py
│   ├── test_retrieval.py
│   ├── test_llm.py
│   ├── test_evaluation.py
│   └── test_serving.py
├── .gitignore
├── Dockerfile                       # API server image (index and models mounted)
├── pyproject.toml                   # Package metadata, dependencies, tool settings
├── uv.lock                          # Locked dependency versions
├── PROJECT_EXPLANATION_AND_ARCHITECTURE.md   # Architecture and metric definitions
└── README.md
```

Generated at runtime and excluded from git:

```
artifacts/
├── index/                           # chunks.jsonl, embeddings.npy, manifest.json
└── reports/                         # Evaluation results: eval_<timestamp>.json, latest.md
.venv/                               # Python environment created by `uv sync`
```

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com). The first run downloads about 3.5 GB of BGE models plus the 4.7 GB LLM. An NVIDIA GPU is optional but much faster: on an RTX 4060 laptop GPU, retrieval with reranking takes about 0.2 s and a full answer 0.5–3 s, compared with roughly 7 s for reranking alone on CPU. On Windows, `uv sync` installs CUDA-enabled PyTorch; on Linux it installs the CPU build.

```bash
ollama pull qwen2.5:7b                                   # local LLM (generator + judge)
uv sync                                                  # Python 3.12 env + PyTorch
uv run pytest -q                                         # offline unit tests

uv run python scripts/build_index.py                     # chunk + embed data/documents/
uv run python scripts/ask.py "How do I verify a webhook signature?"
uv run python -m citesure.serving.app                    # demo + API on http://localhost:8001

uv run python scripts/generate_synthetic_data.py --n 200 # QA set -> data/evaluation/
uv run python scripts/evaluate_pipeline.py --modes all   # ablation table + gate
```

Reports are written to `artifacts/reports/` (per-sample JSON plus `latest.md`). If Ollama runs on another machine or the API runs in Docker, set `CITESURE_OLLAMA_HOST` (for example `http://host.docker.internal:11434`).

## Live demo

Start the server and open <http://localhost:8001> in a browser. Ask a question, pick a retrieval mode, and watch the answer stream in; each `[n]` citation links to the retrieved passage it came from. Interactive API docs are at <http://localhost:8001/docs>.

## Evaluation plan

The headline output is an **ablation table**, with every cell measured on the same question set (`scripts/evaluate_pipeline.py --modes all`):

| Configuration | Context relevance | Faithfulness | Answer relevance | p95 latency |
|---|---|---|---|---|
| Dense only | – | – | – | – |
| BM25 only | – | – | – | – |
| Hybrid (BM25 + dense, RRF) | – | – | – | – |
| Hybrid + cross-encoder rerank | – | – | – | – |

The pass thresholds in [`configs/default.yaml`](configs/default.yaml) (context relevance ≥ 0.85, faithfulness ≥ 0.92, answer relevance ≥ 0.90) are **targets for the CI gate, not achieved results**.

Decisions taken:
- **Generator and judge LLM:** `qwen2.5:7b` served locally by Ollama, so the project runs at no cost. Other Ollama models can be swapped in through `configs/default.yaml`. A 7B judge is noticeably less reliable than a large hosted model, so its scores should be read as rough signals until checked against human labels.
- **Vector store:** in-process numpy by default (exact search, no service to run); Qdrant via `retrieval.vector_store: qdrant` for larger corpora.

Still open:
- **Corpus:** which public technical documentation to index (it must be licensed for reuse). `data/documents/` currently holds three sample files.
- **Evaluation set:** the generator script produces LLM-written questions. A purely synthetic set judged by the same model that generated it overstates quality, so a hand-reviewed subset and judge-vs-human agreement are still needed.

See [`PROJECT_EXPLANATION_AND_ARCHITECTURE.md`](PROJECT_EXPLANATION_AND_ARCHITECTURE.md) for the architecture and the metric definitions.
