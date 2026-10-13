"""Run the RAG Triad evaluation, write reports, and enforce the CI quality gate.

Usage:
  python scripts/evaluate_pipeline.py                          # full system only, gate enforced
  python scripts/evaluate_pipeline.py --modes all --limit 50   # ablation table on 50 questions
  python scripts/evaluate_pipeline.py --no-gate

Exit code is 1 when the gated mode (hybrid_rerank) misses any configured threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

from citesure.config import load_settings
from citesure.evaluation.metrics import TriadEvaluator
from citesure.evaluation.runner import check_gate, evaluate_mode, load_dataset, markdown_table
from citesure.generation.llm import make_llm
from citesure.pipeline import RAGPipeline
from citesure.retrieval.hybrid import RetrievalMode

GATED_MODE = RetrievalMode.HYBRID_RERANK


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--modes", default=GATED_MODE.value, help="comma list of modes, or 'all'")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dataset", type=Path, default=None, help="defaults to paths.eval_dataset")
    parser.add_argument("--tag", default="eval", help="prefix for the saved report file")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-gate", action="store_true")
    args = parser.parse_args()

    console = Console()
    settings = load_settings(args.config)
    modes = (
        list(RetrievalMode)
        if args.modes == "all"
        else [RetrievalMode(m.strip()) for m in args.modes.split(",")]
    )
    samples = load_dataset(args.dataset or settings.resolve(settings.paths.eval_dataset), args.limit)
    console.print(f"Evaluating {len(samples)} questions across modes: {[m.value for m in modes]}")

    with console.status("Loading models..."):
        pipeline = RAGPipeline.from_settings(settings)
    judge = make_llm(settings.models, "judge")
    evaluator = TriadEvaluator(judge, pipeline.retriever.embedder)

    reports = []
    for mode in modes:
        with console.status(f"Running {mode.value}..."):
            reports.append(evaluate_mode(pipeline, evaluator, samples, mode, args.workers))

    table = markdown_table(reports)
    console.print(table)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    reports_dir = settings.resolve(settings.paths.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{args.tag}_{stamp}.json").write_text(
        json.dumps(
            {
                "timestamp": stamp,
                "generator": settings.models.llm_generator,
                "judge": settings.models.llm_judge,
                "n": len(samples),
                "reports": [asdict(r) for r in reports],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (reports_dir / "latest.md").write_text(table + "\n", encoding="utf-8")

    gated = next((r for r in reports if r.mode == GATED_MODE.value), None)
    if args.no_gate or gated is None:
        return 0
    failures = check_gate(gated, settings.evaluation)
    if failures:
        console.print("[bold red]Quality gate FAILED:[/bold red] " + "; ".join(failures))
        return 1
    console.print("[bold green]Quality gate passed.[/bold green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
