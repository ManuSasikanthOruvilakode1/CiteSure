"""Typed configuration loaded from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


class PathsConfig(BaseModel):
    documents_dir: Path = Path("data/documents")
    index_dir: Path = Path("artifacts/index")
    eval_dataset: Path = Path("data/evaluation/qa_pairs.jsonl")
    reports_dir: Path = Path("artifacts/reports")


class ChunkingConfig(BaseModel):
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 51


class ModelsConfig(BaseModel):
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_query_instruction: str = ""
    reranker_model: str = "BAAI/bge-reranker-large"
    llm_generator: str = "qwen2.5:7b"
    llm_judge: str = "qwen2.5:7b"
    ollama_host: str = "http://localhost:11434"
    ollama_num_ctx: int = 8192
    generator_temperature: float = 0.1


class RetrievalConfig(BaseModel):
    vector_store: Literal["numpy", "qdrant"] = "numpy"
    top_k_dense: int = 30
    top_k_sparse: int = 30
    top_k_fused: int = 30
    top_k_reranked: int = 5
    rrf_k: int = 60
    min_rerank_score: float = 0.02


class EvaluationConfig(BaseModel):
    min_context_relevance: float = 0.85
    min_faithfulness: float = 0.92
    min_answer_relevance: float = 0.90
    test_dataset_size: int = 200


class ServingConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8001


class Settings(BaseModel):
    paths: PathsConfig = PathsConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    models: ModelsConfig = ModelsConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    serving: ServingConfig = ServingConfig()

    def resolve(self, path: Path) -> Path:
        """Resolve a config-relative path against the project root."""
        return path if path.is_absolute() else PROJECT_ROOT / path


def load_settings(path: str | Path | None = None) -> Settings:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    raw.pop("project", None)
    return Settings.model_validate(raw)
