"""LLM client abstraction with a local Ollama backend."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMOutputError(RuntimeError):
    """The model did not return a usable response."""


class LLMClient(Protocol):
    def complete(self, system: str, prompt: str) -> str: ...
    def stream(self, system: str, prompt: str) -> Iterator[str]: ...
    def structured(self, system: str, prompt: str, schema: type[T]) -> T: ...


class OllamaClient:
    """Local model served by Ollama (https://ollama.com) via its native REST API."""

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
        num_ctx: int = 8192,
        timeout: float = 300.0,
        http: httpx.Client | None = None,
    ) -> None:
        self.model = model
        # Ollama's default context (2-4k tokens) would silently truncate 5 x 512-token passages.
        self.options = {"temperature": temperature, "num_ctx": num_ctx}
        self.http = http or httpx.Client(base_url=host, timeout=timeout)

    def _body(self, system: str, prompt: str, **extra) -> dict:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "options": self.options,
            **extra,
        }

    def _raise(self, r: httpx.Response) -> None:
        try:
            detail = r.json().get("error", r.text)
        except ValueError:
            detail = r.text
        hint = f" (try `ollama pull {self.model}`)" if "not found" in detail else ""
        raise RuntimeError(
            f"Ollama at {self.http.base_url} returned {r.status_code}: {detail}{hint}"
        )

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        try:
            with self.http.stream("POST", "/api/chat", json=self._body(system, prompt, stream=True)) as r:
                if r.status_code != 200:
                    r.read()
                    self._raise(r)
                for line in r.iter_lines():
                    if not line:
                        continue
                    event = json.loads(line)
                    if "error" in event:
                        raise RuntimeError(f"Ollama error: {event['error']}")
                    if text := event.get("message", {}).get("content"):
                        yield text
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.http.base_url}; start it with `ollama serve`"
            ) from exc

    def complete(self, system: str, prompt: str) -> str:
        return "".join(self.stream(system, prompt))

    def structured(self, system: str, prompt: str, schema: type[T], retries: int = 2) -> T:
        # Ollama constrains decoding to the JSON schema; small models can still return
        # semantically invalid objects, so validate and retry a couple of times.
        body = self._body(system, prompt, stream=False, format=schema.model_json_schema())
        last_error: Exception | None = None
        for _ in range(retries + 1):
            r = self.http.post("/api/chat", json=body)
            if r.status_code != 200:
                self._raise(r)
            try:
                return schema.model_validate_json(r.json()["message"]["content"])
            except (ValidationError, KeyError) as exc:
                last_error = exc
        raise LLMOutputError(f"Invalid structured output after {retries + 1} tries: {last_error}")


def make_llm(models, role: str = "generator") -> LLMClient:
    """Build the configured LLM for ``role`` ("generator" or "judge") from ModelsConfig."""
    if role not in ("generator", "judge"):
        raise ValueError(f"unknown role {role!r}")
    return OllamaClient(
        model=models.llm_generator if role == "generator" else models.llm_judge,
        # CITESURE_OLLAMA_HOST overrides the config, e.g. for Docker (host.docker.internal).
        host=os.environ.get("CITESURE_OLLAMA_HOST", models.ollama_host),
        temperature=models.generator_temperature if role == "generator" else 0.0,
        num_ctx=models.ollama_num_ctx,
    )
