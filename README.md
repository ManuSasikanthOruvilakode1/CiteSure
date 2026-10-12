# CiteSure

**Grounded document Q&A with continuous faithfulness evaluation.**

> **Status: v0.1.0, evaluated on real documentation.** The pipeline indexes the official FastAPI docs (122 pages, 1,083 chunks) and was evaluated on 100 questions across four retrieval configurations. Results and their limitations are below.

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
| Real corpus + evaluation | FastAPI docs (122 pages), 100-question evaluation across 4 configurations | Done |
| Human-reviewed eval subset | | Not started |

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
│   │   └── fastapi/                 # FastAPI docs, 122 pages (MIT); see SOURCE and LICENSE
│   └── evaluation/
│       └── qa_pairs.jsonl           # 100 QA pairs, created by generate_synthetic_data.py
├── scripts/
│   ├── prepare_fastapi_docs.py      # Convert FastAPI's docs into the corpus above
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
├── LICENSE                          # MIT License
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
uv run python scripts/ask.py "How do I declare an optional query parameter?"
uv run python -m citesure.serving.app                    # demo + API on http://localhost:8001

uv run python scripts/generate_synthetic_data.py --n 100 # QA set -> data/evaluation/
uv run python scripts/evaluate_pipeline.py --modes all   # ablation table + gate
```

Reports are written to `artifacts/reports/` (per-sample JSON plus `latest.md`). If Ollama runs on another machine or the API runs in Docker, set `CITESURE_OLLAMA_HOST` (for example `http://host.docker.internal:11434`).

## Live demo

Start the server and open <http://localhost:8001> in a browser. Ask a question, pick a retrieval mode, and watch the answer stream in; each `[n]` citation links to the retrieved passage it came from. Interactive API docs are at <http://localhost:8001/docs>.

## Results

Measured on 2026-10-12 against the FastAPI corpus: 100 questions, each run through all four retrieval configurations, on a laptop RTX 4060 with `qwen2.5:7b` as both generator and judge. Raw per-question data: [`results/fastapi_eval_2026-10-12.json`](results/fastapi_eval_2026-10-12.json).

| Configuration | Hit@5 | MRR | Context relevance | Faithfulness | Answer relevance | Latency p50 / p95 |
|---|---|---|---|---|---|---|
| Dense only (BGE-large) | 0.92 | 0.73 | 0.52 | 0.97 | 0.87 | 2.8 s / 7.7 s |
| BM25 only | 0.87 | 0.70 | 0.41 | 0.94 | 0.84 | 2.8 s / 6.1 s |
| Hybrid (BM25 + dense, RRF) | 0.92 | 0.78 | 0.46 | 0.94 | 0.87 | 2.8 s / 7.0 s |
| **Hybrid + cross-encoder rerank** | **0.99** | **0.85** | 0.52 | 0.94 | 0.86 | 3.1 s / 7.4 s |

**What the numbers show**
- **Reranking is the clear retrieval win.** The passage each question was written from appears in the top 5 for 99 of 100 questions (vs 92 for dense or hybrid alone), and ranks higher on average (MRR 0.85 vs 0.73 for dense). Hit@5 and MRR need no LLM judge, so they are the most trustworthy numbers here.
- **Answers stay grounded in every configuration (faithfulness 0.94–0.97).** In the reranked configuration the judge rejected 26 of 427 answer claims. A manual check of 10 of those found about 7 genuine problems (general-knowledge padding such as "FastAPI is built on ASGI", or embellishment) and about 3 judge mistakes, so true faithfulness is likely slightly higher than reported.
- **Context relevance (~0.5) is the mean grade over all five retrieved passages.** Usually one or two of the five answer the question and the rest are related background, so ~0.5 is expected at k = 5 and does not mean retrieval failed.
- **The judge-scored metrics barely separate the configurations.** Differences of ±0.03 are within the noise of a 7B judge on 100 questions; the retrieval metrics separate them clearly.
- **Latency:** a typical answer takes about 3 s end to end; the p95 of about 7 s comes from long answers that include code. Retrieval with reranking takes about 0.2 s of that.

**Limitations**
- The questions are synthetic: `qwen2.5:7b` wrote one question per sampled passage, and the same model judged the answers. Same-model generation and judging tends to overstate quality.
- A 7B judge is noisy (see the manual check above). Hand-labelling a subset to measure judge agreement is still to do.
- Questions written from a single passage favour lookup-style questions; multi-page reasoning is under-tested.

To reproduce: `uv run python scripts/evaluate_pipeline.py --modes all` (about 75 minutes on the hardware above).

The CI gate thresholds in [`configs/default.yaml`](configs/default.yaml) are set from these measurements minus a small tolerance, so a change that makes answers less grounded or retrieval worse fails the gate.

Decisions taken:
- **Generator and judge LLM:** `qwen2.5:7b` served locally by Ollama, so the project runs at no cost. Other Ollama models can be swapped in through `configs/default.yaml`. A 7B judge is noticeably less reliable than a large hosted model, so its scores should be read as rough signals until checked against human labels.
- **Corpus:** the English [FastAPI documentation](https://github.com/fastapi/fastapi/tree/master/docs/en/docs) (MIT license, © Sebastián Ramírez), 122 pages with their code examples inlined by `scripts/prepare_fastapi_docs.py`. Release notes, contributor/translation admin pages and generated API-reference stubs are excluded. The exact upstream commit is recorded in `data/documents/fastapi/SOURCE`.
- **Vector store:** in-process numpy by default (exact search, no service to run); Qdrant via `retrieval.vector_store: qdrant` for larger corpora.

Still open:
- **Human-reviewed evaluation subset** and judge-vs-human agreement, to put a number on judge reliability.

See [`PROJECT_EXPLANATION_AND_ARCHITECTURE.md`](PROJECT_EXPLANATION_AND_ARCHITECTURE.md) for the architecture and the metric definitions.

## License

CiteSure is released under the [MIT License](LICENSE).

The indexed corpus in `data/documents/fastapi/` is the [FastAPI documentation](https://github.com/fastapi/fastapi), © Sebastián Ramírez, also under the MIT License; its licence text is kept alongside it in `data/documents/fastapi/LICENSE`.
