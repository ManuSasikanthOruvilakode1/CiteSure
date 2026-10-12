import json

import httpx
import pytest
from pydantic import BaseModel

from citesure.config import ModelsConfig
from citesure.generation.llm import (
    LLMOutputError,
    OllamaClient,
    make_llm,
)


class Answer(BaseModel):
    value: int


def _client(handler) -> OllamaClient:
    http = httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(handler))
    return OllamaClient(model="m", http=http)


def test_stream_yields_ndjson_content_and_sends_context_size():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        lines = [
            {"message": {"content": "Hello"}, "done": False},
            {"message": {"content": " world"}, "done": False},
            {"message": {"content": ""}, "done": True},
        ]
        return httpx.Response(200, text="\n".join(json.dumps(x) for x in lines))

    assert _client(handler).complete("sys", "hi") == "Hello world"
    assert seen["stream"] is True
    assert seen["options"]["num_ctx"] == 8192
    assert seen["messages"][0] == {"role": "system", "content": "sys"}


def test_structured_sends_schema_and_parses():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["format"]["properties"]["value"]["type"] == "integer"
        return httpx.Response(200, json={"message": {"content": '{"value": 3}'}})

    assert _client(handler).structured("s", "p", Answer).value == 3


def test_structured_retries_then_fails_on_invalid_json():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, json={"message": {"content": "not json"}})

    with pytest.raises(LLMOutputError):
        _client(handler).structured("s", "p", Answer, retries=2)
    assert len(calls) == 3


def test_stream_surfaces_ollama_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=json.dumps({"error": "model 'm' not found"}))

    with pytest.raises(RuntimeError, match="not found"):
        _client(handler).complete("s", "p")


def test_make_llm_ollama_roles():
    assert make_llm(ModelsConfig(), "generator").options["temperature"] == 0.1
    assert make_llm(ModelsConfig(), "judge").options["temperature"] == 0.0


def test_make_llm_host_env_override(monkeypatch):
    monkeypatch.setenv("CITESURE_OLLAMA_HOST", "http://host.docker.internal:11434")
    llm = make_llm(ModelsConfig(llm_judge="llama3.1:8b"), "judge")
    assert isinstance(llm, OllamaClient) and llm.model == "llama3.1:8b"
    assert str(llm.http.base_url).startswith("http://host.docker.internal:11434")


def test_http_error_reports_ollama_message():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'm' not found"})

    with pytest.raises(RuntimeError, match=r"not found.*ollama pull m"):
        _client(handler).complete("s", "p")
    with pytest.raises(RuntimeError, match="404"):
        _client(handler).structured("s", "p", Answer)
