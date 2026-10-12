"""FastAPI service: JSON and Server-Sent-Events endpoints with chunk attribution."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from citesure.config import load_settings
from citesure.generation.generator import extract_citations
from citesure.pipeline import RAGPipeline
from citesure.retrieval.hybrid import RetrievalMode
from citesure.types import ScoredChunk

DEMO_PAGE = Path(__file__).parent / "static" / "index.html"


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    mode: RetrievalMode = RetrievalMode.HYBRID_RERANK


class SourceOut(BaseModel):
    index: int
    chunk_id: str
    source: str
    section: str
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    cited: list[int]
    latency_s: float


def _sources(contexts: list[ScoredChunk]) -> list[SourceOut]:
    return [
        SourceOut(
            index=i,
            chunk_id=c.chunk.chunk_id,
            source=c.chunk.source,
            section=c.chunk.section,
            score=c.score,
            text=c.chunk.text,
        )
        for i, c in enumerate(contexts, start=1)
    ]


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(pipeline: RAGPipeline | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = pipeline or RAGPipeline.from_settings(load_settings())
        yield

    app = FastAPI(title="CiteSure", version="0.1.0", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def demo() -> str:
        return DEMO_PAGE.read_text(encoding="utf-8")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    # Sync handlers run in FastAPI's threadpool, so blocking model/LLM calls don't stall the loop.
    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest, request: Request) -> QueryResponse:
        result = request.app.state.pipeline.answer(req.question, req.mode)
        return QueryResponse(
            answer=result.answer,
            sources=_sources(result.contexts),
            cited=extract_citations(result.answer, len(result.contexts)),
            latency_s=result.latency_s,
        )

    @app.post("/query/stream")
    def query_stream(req: QueryRequest, request: Request) -> StreamingResponse:
        try:
            contexts, tokens = request.app.state.pipeline.stream(req.question, req.mode)
        except Exception as exc:  # retrieval failure before streaming starts
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        def events() -> Iterator[str]:
            yield _sse("sources", [s.model_dump() for s in _sources(contexts)])
            parts: list[str] = []
            try:
                for token in tokens:
                    parts.append(token)
                    yield _sse("token", token)
            except Exception as exc:  # noqa: BLE001 - surface errors in-band once streaming
                yield _sse("error", f"{type(exc).__name__}: {exc}")
                return
            yield _sse("done", {"cited": extract_citations("".join(parts), len(contexts))})

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def main() -> None:
    import uvicorn

    settings = load_settings()
    uvicorn.run(create_app(), host=settings.serving.host, port=settings.serving.port)


if __name__ == "__main__":
    main()
