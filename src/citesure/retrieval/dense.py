"""Dense vector retrieval: embedders and vector stores."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from citesure.types import Chunk, ScoredChunk


def accelerator_kwargs() -> dict:
    """Half precision on CUDA: halves VRAM so the models fit beside a local LLM."""
    import torch

    if torch.cuda.is_available():
        return {"device": "cuda", "model_kwargs": {"torch_dtype": torch.float16}}
    return {"device": "cpu"}


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """BGE-style bi-encoder. Embeddings are L2-normalised so dot product == cosine."""

    def __init__(self, model_name: str, query_instruction: str = "", batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, **accelerator_kwargs())
        self.query_instruction = query_instruction
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 64,
            convert_to_numpy=True,
        ).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.model.encode(
            self.query_instruction + text, normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)


def chunk_embedding_text(chunk: Chunk) -> str:
    """Prefix the section path so headers contribute to the passage embedding."""
    return f"{chunk.section}\n{chunk.text}" if chunk.section else chunk.text


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None: ...
    def search(self, query_vector: np.ndarray, top_k: int) -> list[ScoredChunk]: ...


class NumpyVectorStore:
    """Exact in-process cosine search. Fast enough for corpora up to ~1e5 chunks."""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.matrix = np.zeros((0, 0), dtype=np.float32)

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        self.matrix = embeddings if not self.chunks else np.vstack([self.matrix, embeddings])
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[ScoredChunk]:
        if not self.chunks:
            return []
        scores = self.matrix @ np.asarray(query_vector, dtype=np.float32)
        order = np.argsort(-scores)[:top_k]
        return [
            ScoredChunk(self.chunks[i], float(scores[i]), {"dense": float(scores[i])})
            for i in order
        ]

    def save(self, path: Path) -> None:
        np.save(path, self.matrix)

    @classmethod
    def load(cls, chunks: list[Chunk], path: Path) -> NumpyVectorStore:
        store = cls()
        store.add(chunks, np.load(path))
        return store


class QdrantVectorStore:
    """Qdrant-backed store (embedded local mode by default, or a server via ``url``)."""

    def __init__(
        self,
        dim: int,
        collection: str = "citesure",
        path: str | None = None,
        url: str | None = None,
    ) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.client = QdrantClient(url=url) if url else QdrantClient(path=path or ":memory:")
        self.collection = collection
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        from qdrant_client.models import PointStruct

        start = self.client.count(self.collection).count
        points = [
            PointStruct(id=start + i, vector=vec.tolist(), payload=chunk.to_dict())
            for i, (chunk, vec) in enumerate(zip(chunks, embeddings, strict=True))
        ]
        self.client.upsert(self.collection, points=points)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[ScoredChunk]:
        hits = self.client.query_points(
            self.collection, query=np.asarray(query_vector).tolist(), limit=top_k
        ).points
        return [
            ScoredChunk(Chunk.from_dict(h.payload), float(h.score), {"dense": float(h.score)})
            for h in hits
        ]
