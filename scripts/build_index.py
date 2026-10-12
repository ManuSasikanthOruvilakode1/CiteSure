"""Chunk, embed, and persist the document corpus.

Usage: python scripts/build_index.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import time

from rich.console import Console

from citesure.config import load_settings
from citesure.pipeline import build_index, make_embedder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    console = Console()
    settings = load_settings(args.config)
    start = time.perf_counter()
    with console.status(f"Loading {settings.models.embedding_model}..."):
        embedder = make_embedder(settings)
    with console.status("Chunking and embedding documents..."):
        chunks = build_index(settings, embedder)
    sources = len({c.source for c in chunks})
    console.print(
        f"[green]Indexed {len(chunks)} chunks from {sources} documents "
        f"in {time.perf_counter() - start:.1f}s[/green] -> {settings.paths.index_dir}"
    )


if __name__ == "__main__":
    main()
