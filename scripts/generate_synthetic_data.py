"""Generate a synthetic QA evaluation set from indexed chunks with the configured judge LLM.

Each question is written from a single source chunk, so its chunk_id doubles as a gold
label for retrieval hit-rate / MRR. Chunks are sampled round-robin across documents so
large files don't dominate the set.

Usage: python scripts/generate_synthetic_data.py [--n 200] [--seed 7]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import track

from citesure.config import load_settings
from citesure.generation.llm import make_llm
from citesure.pipeline import load_chunks
from citesure.types import Chunk

SYSTEM = (
    "You write evaluation questions for a documentation question-answering system. "
    "Questions should sound like a real engineer asking about the product, not like a "
    "reading-comprehension test, and must be answerable from the given passage alone."
)


class SyntheticQA(BaseModel):
    usable: bool = Field(description="False if the passage has no substantive, answerable facts")
    question: str
    ground_truth: str = Field(description="Concise answer drawn only from the passage")
    question_type: str = Field(description="One of: factual, procedural, comparative, troubleshooting")


def sample_chunks(chunks: list[Chunk], n: int, seed: int, min_words: int = 40) -> list[Chunk]:
    rng = random.Random(seed)
    by_source: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        if len(c.text.split()) >= min_words:
            by_source[c.source].append(c)
    for group in by_source.values():
        rng.shuffle(group)
    picked: list[Chunk] = []
    while len(picked) < n and any(by_source.values()):
        for source in sorted(by_source):
            if by_source[source] and len(picked) < n:
                picked.append(by_source[source].pop())
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--n", type=int, default=None, help="defaults to evaluation.test_dataset_size")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    console = Console()
    settings = load_settings(args.config)
    n = args.n or settings.evaluation.test_dataset_size
    chunks = load_chunks(settings.resolve(settings.paths.index_dir))
    # Oversample to leave room for chunks the model marks as unusable.
    candidates = sample_chunks(chunks, int(n * 1.3) + 5, args.seed)
    if len(candidates) < n:
        console.print(f"[yellow]Only {len(candidates)} eligible chunks; dataset will be smaller than {n}.[/yellow]")

    llm = make_llm(settings.models, "judge")

    def generate(chunk: Chunk) -> dict | None:
        prompt = (
            f"<passage source=\"{chunk.source}\" section=\"{chunk.section}\">\n"
            f"{chunk.text}\n</passage>\n\n"
            "Write one question answerable from this passage and its concise answer."
        )
        try:
            qa = llm.structured(SYSTEM, prompt, SyntheticQA)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]skip {chunk.chunk_id}: {exc}[/red]")
            return None
        if not qa.usable:
            return None
        return {
            "question": qa.question,
            "ground_truth": qa.ground_truth,
            "question_type": qa.question_type,
            "source_chunk_id": chunk.chunk_id,
            "source": chunk.source,
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(track(pool.map(generate, candidates), total=len(candidates), description="Generating"))
    rows = [r for r in results if r][:n]

    out = settings.resolve(settings.paths.eval_dataset)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    console.print(f"[green]Wrote {len(rows)} QA pairs -> {out}[/green]")


if __name__ == "__main__":
    main()
