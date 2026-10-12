import json

from conftest import ScriptedLLM
from fastapi.testclient import TestClient

from citesure.generation.generator import Generator
from citesure.pipeline import RAGPipeline
from citesure.retrieval.bm25 import BM25Index
from citesure.retrieval.dense import NumpyVectorStore
from citesure.retrieval.hybrid import HybridRetriever
from citesure.serving.app import create_app


def _client(chunks, embedder):
    store = NumpyVectorStore()
    store.add(chunks, embedder.embed_documents([c.text for c in chunks]))
    retriever = HybridRetriever(BM25Index(chunks), store, embedder, top_k_final=2)
    pipeline = RAGPipeline(retriever, Generator(ScriptedLLM(text="Use the dashboard [1].")))
    return TestClient(create_app(pipeline))


def test_query_returns_answer_sources_and_citations(chunks, embedder):
    with _client(chunks, embedder) as client:
        resp = client.post("/query", json={"question": "rotate API keys"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Use the dashboard [1]."
    assert body["cited"] == [1]
    assert body["sources"][0]["chunk_id"] == "auth"


def test_stream_emits_sources_tokens_done(chunks, embedder):
    with _client(chunks, embedder) as client:
        resp = client.post("/query/stream", json={"question": "rotate API keys"})
    events = [
        (block.split("\n")[0].removeprefix("event: "), json.loads(block.split("\n")[1].removeprefix("data: ")))
        for block in resp.text.strip().split("\n\n")
    ]
    names = [e for e, _ in events]
    assert names[0] == "sources" and names[-1] == "done"
    assert "".join(d for e, d in events if e == "token").strip() == "Use the dashboard [1]."
    assert events[-1][1] == {"cited": [1]}


def test_rejects_empty_question(chunks, embedder):
    with _client(chunks, embedder) as client:
        assert client.post("/query", json={"question": ""}).status_code == 422


def test_demo_page_is_served(chunks, embedder):
    with _client(chunks, embedder) as client:
        resp = client.get("/")
    assert resp.status_code == 200 and "text/html" in resp.headers["content-type"]
    assert "/query/stream" in resp.text


def test_no_relevant_passages_refuses_without_llm(chunks, embedder):
    from citesure.generation.prompts import REFUSAL_MESSAGE

    class RejectAll:
        def rerank(self, query, candidates, top_k):
            return []

    store = NumpyVectorStore()
    store.add(chunks, embedder.embed_documents([c.text for c in chunks]))
    llm = ScriptedLLM(text="should not be called")
    retriever = HybridRetriever(BM25Index(chunks), store, embedder, reranker=RejectAll())
    with TestClient(create_app(RAGPipeline(retriever, Generator(llm)))) as client:
        body = client.post("/query", json={"question": "What is a computer?"}).json()
    assert body["answer"] == REFUSAL_MESSAGE and body["sources"] == []
    assert llm.calls == []
