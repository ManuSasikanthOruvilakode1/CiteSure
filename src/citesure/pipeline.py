"""End-to-end pipeline: index build/load and question answering."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path

from citesure.config import Settings
from citesure.generation import Generator, make_llm
from citesure.ingestion import RecursiveChunker, load_directory
from citesure.retrieval.bm25 import BM25Index
from citesure.retrieval.dense import (
    Embedder,
    NumpyVectorStore,
    QdrantVectorStore,
    SentenceTransformerEmbedder,
    VectorStore,
    chunk_embedding_text,
)
from citesure.retrieval.hybrid import HybridRetriever, RetrievalMode
from citesure.retrieval.reranker import CrossEncoderReranker, Reranker
from citesure.types import Chunk, RAGAnswer, ScoredChunk

CHUNKS_FILE = "chunks.jsonl"
EMBEDDINGS_FILE = "embeddings.npy"
MANIFEST_FILE = "manifest.json"
QDRANT_DIR = "qdrant"


def make_embedder(settings: Settings) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(
        settings.models.embedding_model, settings.models.embedding_query_instruction
    )


def build_index(settings: Settings, embedder: Embedder) -> list[Chunk]:
    """Load, chunk, embed and persist the document corpus."""
    docs_dir = settings.resolve(settings.paths.documents_dir)
    index_dir = settings.resolve(settings.paths.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    docs = load_directory(docs_dir)
    if not docs:
        raise FileNotFoundError(f"No supported documents found in {docs_dir}")
    chunker = RecursiveChunker(
        settings.chunking.chunk_size_tokens, settings.chunking.chunk_overlap_tokens
    )
    chunks = chunker.chunk_documents(docs)
    embeddings = embedder.embed_documents([chunk_embedding_text(c) for c in chunks])

    with (index_dir / CHUNKS_FILE).open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    if settings.retrieval.vector_store == "qdrant":
        store = QdrantVectorStore(dim=embeddings.shape[1], path=str(index_dir / QDRANT_DIR))
        store.client.delete_collection(store.collection)
        store = QdrantVectorStore(dim=embeddings.shape[1], path=str(index_dir / QDRANT_DIR))
        store.add(chunks, embeddings)
    else:
        store = NumpyVectorStore()
        store.add(chunks, embeddings)
        store.save(index_dir / EMBEDDINGS_FILE)

    manifest = {
        "documents": len(docs),
        "chunks": len(chunks),
        "embedding_model": settings.models.embedding_model,
        "embedding_dim": int(embeddings.shape[1]),
        "vector_store": settings.retrieval.vector_store,
        "chunking": settings.chunking.model_dump(),
    }
    (index_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return chunks


def load_chunks(index_dir: Path) -> list[Chunk]:
    path = index_dir / CHUNKS_FILE
    if not path.exists():
        raise FileNotFoundError(f"No index at {index_dir}; run scripts/build_index.py first")
    with path.open(encoding="utf-8") as f:
        return [Chunk.from_dict(json.loads(line)) for line in f if line.strip()]


def load_retriever(
    settings: Settings, embedder: Embedder, reranker: Reranker | None
) -> HybridRetriever:
    index_dir = settings.resolve(settings.paths.index_dir)
    chunks = load_chunks(index_dir)
    manifest = json.loads((index_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    if manifest["embedding_model"] != settings.models.embedding_model:
        raise ValueError(
            f"Index was built with {manifest['embedding_model']} but config uses "
            f"{settings.models.embedding_model}; rebuild the index"
        )

    store: VectorStore
    if manifest["vector_store"] == "qdrant":
        store = QdrantVectorStore(dim=manifest["embedding_dim"], path=str(index_dir / QDRANT_DIR))
    else:
        store = NumpyVectorStore.load(chunks, index_dir / EMBEDDINGS_FILE)

    r = settings.retrieval
    return HybridRetriever(
        bm25=BM25Index(chunks),
        vector_store=store,
        embedder=embedder,
        reranker=reranker,
        top_k_dense=r.top_k_dense,
        top_k_sparse=r.top_k_sparse,
        top_k_fused=r.top_k_fused,
        top_k_final=r.top_k_reranked,
        rrf_k=r.rrf_k,
        min_rerank_score=r.min_rerank_score,
    )


class RAGPipeline:
    def __init__(self, retriever: HybridRetriever, generator: Generator) -> None:
        self.retriever = retriever
        self.generator = generator

    @classmethod
    def from_settings(cls, settings: Settings) -> RAGPipeline:
        embedder = make_embedder(settings)
        reranker = CrossEncoderReranker(settings.models.reranker_model)
        llm = make_llm(settings.models, "generator")
        return cls(load_retriever(settings, embedder, reranker), Generator(llm))

    def retrieve(
        self, question: str, mode: RetrievalMode = RetrievalMode.HYBRID_RERANK
    ) -> list[ScoredChunk]:
        return self.retriever.retrieve(question, mode)

    def stream(
        self, question: str, mode: RetrievalMode = RetrievalMode.HYBRID_RERANK
    ) -> tuple[list[ScoredChunk], Iterator[str]]:
        contexts = self.retrieve(question, mode)
        return contexts, self.generator.stream(question, contexts)

    def answer(
        self, question: str, mode: RetrievalMode = RetrievalMode.HYBRID_RERANK
    ) -> RAGAnswer:
        start = time.perf_counter()
        contexts = self.retrieve(question, mode)
        answer = self.generator.generate(question, contexts)
        return RAGAnswer(question, answer, contexts, time.perf_counter() - start)
