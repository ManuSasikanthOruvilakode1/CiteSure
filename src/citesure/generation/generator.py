"""Answer generation over retrieved context."""

from __future__ import annotations

import re
from collections.abc import Iterator

from citesure.generation.llm import LLMClient
from citesure.generation.prompts import (
    GENERATOR_SYSTEM,
    REFUSAL_MESSAGE,
    build_generation_prompt,
)
from citesure.types import ScoredChunk

_CITATION_RE = re.compile(r"\[(\d+)\]")


def extract_citations(answer: str, num_contexts: int) -> list[int]:
    """Return the sorted, valid 1-based passage numbers cited in ``answer``."""
    cited = {int(m) for m in _CITATION_RE.findall(answer)}
    return sorted(n for n in cited if 1 <= n <= num_contexts)


class Generator:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def stream(self, question: str, contexts: list[ScoredChunk]) -> Iterator[str]:
        if not contexts:
            yield REFUSAL_MESSAGE
            return
        yield from self.llm.stream(GENERATOR_SYSTEM, build_generation_prompt(question, contexts))

    def generate(self, question: str, contexts: list[ScoredChunk]) -> str:
        return "".join(self.stream(question, contexts)).strip()
