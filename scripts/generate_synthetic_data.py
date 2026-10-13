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
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import track

from citesure.config import load_settings
from citesure.generation.llm import make_llm
from citesure.pipeline import load_chunks, make_embedder
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
    parser.add_argument("--out", type=Path, default=None, help="defaults to paths.eval_dataset")
    parser.add_argument("--exclude", type=Path, action="append", default=[],
                        help="existing dataset(s): their passages are skipped and near-duplicate "
                             "questions are dropped, so a held-out set cannot leak into dev")
    parser.add_argument("--dup-threshold", type=float, default=0.92,
                        help="cosine similarity above which a question counts as a paraphrase")
    args = parser.parse_args()

    console = Console()
    settings = load_settings(args.config)
    n = args.n or settings.evaluation.test_dataset_size
    chunks = load_chunks(settings.resolve(settings.paths.index_dir))
    excluded = [json.loads(line) for path in args.exclude
                for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    used_chunks = {r.get("source_chunk_id") for r in excluded}
    chunks = [c for c in chunks if c.chunk_id not in used_chunks]
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
            # The passage text itself: retrieval hits are matched on text overlap, so the
            # label stays valid when the corpus is re-chunked with different settings.
            "evidence": chunk.text,
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(track(pool.map(generate, candidates), total=len(candidates), description="Generating"))
    rows = [r for r in results if r]
    if excluded and rows:
        # Drop paraphrases of excluded questions (and of each other) across the split.
        embedder = make_embedder(settings)
        old = embedder.embed_documents([r["question"] for r in excluded])
        kept, kept_vecs = [], []
        for row, vec in zip(rows, embedder.embed_documents([r["question"] for r in rows]), strict=True):
            sims = [float(np.max(old @ vec))] + [float(v @ vec) for v in kept_vecs]
            if max(sims) < args.dup_threshold:
                kept.append(row)
                kept_vecs.append(vec)
        console.print(f"Dropped {len(rows) - len(kept)} near-duplicate questions")
        rows = kept
    rows = rows[:n]

    out = args.out or settings.resolve(settings.paths.eval_dataset)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    console.print(f"[green]Wrote {len(rows)} QA pairs -> {out}[/green]")


if __name__ == "__main__":
    main()
