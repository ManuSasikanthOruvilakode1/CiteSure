"""Ask a question from the command line and stream the cited answer.

Usage: python scripts/ask.py "How do I rotate an API key?" [--mode hybrid_rerank]
"""

from __future__ import annotations

import argparse

from rich.console import Console

from citesure.config import load_settings
from citesure.pipeline import RAGPipeline
from citesure.retrieval.hybrid import RetrievalMode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--mode", default=RetrievalMode.HYBRID_RERANK.value,
                        choices=[m.value for m in RetrievalMode])
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    console = Console()
    with console.status("Loading models..."):
        pipeline = RAGPipeline.from_settings(load_settings(args.config))
    contexts, tokens = pipeline.stream(args.question, RetrievalMode(args.mode))
    for token in tokens:
        console.print(token, end="", soft_wrap=True, markup=False, highlight=False)
    console.print("\n\n[bold]Sources[/bold]")
    for i, c in enumerate(contexts, start=1):
        section = f" > {c.chunk.section}" if c.chunk.section else ""
        console.print(f"[{i}] {c.chunk.source}{section}  (score {c.score:.3f})", markup=False)


if __name__ == "__main__":
    main()
