"""Measure refusal behaviour: does CiteSure decline when the corpus has no answer?

Runs the hand-written unanswerable set (data/evaluation/unanswerable.jsonl) through each
retrieval mode and counts correct refusals, and reads the answerable-question results to
count false refusals. No LLM judge is involved: a refusal is the exact refusal message or
an explicit "the documents do not cover..." statement (see citesure.evaluation.stats).

Usage: python scripts/evaluate_refusals.py [--modes all] [--answerable results/fastapi_eval_2026-10-12.json]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

from citesure.config import PROJECT_ROOT, load_settings
from citesure.evaluation.stats import is_refusal, wilson_interval
from citesure.pipeline import RAGPipeline
from citesure.retrieval.hybrid import RetrievalMode


def _rate(k: int, n: int) -> str:
    p, lo, hi = wilson_interval(k, n)
    return f"{k}/{n} ({p:.0%}, 95% CI {lo:.0%}–{hi:.0%})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--modes", default="all", help="comma list of modes, or 'all'")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data/evaluation/unanswerable.jsonl")
    parser.add_argument("--answerable", type=Path, default=PROJECT_ROOT / "results/fastapi_eval_2026-10-12.json",
                        help="results file from evaluate_pipeline.py, for false-refusal counts")
    args = parser.parse_args()

    console = Console()
    modes = list(RetrievalMode) if args.modes == "all" else [RetrievalMode(m) for m in args.modes.split(",")]
    rows = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    categories = sorted({r["category"] for r in rows})
    answerable = {}
    if args.answerable.exists():
        for rep in json.loads(args.answerable.read_text(encoding="utf-8"))["reports"]:
            answerable[rep["mode"]] = [is_refusal(s["answer"]) for s in rep["samples"]]

    with console.status("Loading models..."):
        pipeline = RAGPipeline.from_settings(load_settings(args.config))

    report = {"timestamp": datetime.now(timezone.utc).isoformat(), "modes": {}}
    header = "| Mode | " + " | ".join(f"Correct refusals: {c}" for c in categories) + " | False refusals (answerable) |"
    lines = [header, "|" + "---|" * (len(categories) + 2)]
    for mode in modes:
        results = []
        with console.status(f"Running {mode.value}..."):
            for r in rows:
                ans = pipeline.answer(r["question"], mode)
                results.append({**r, "answer": ans.answer, "refused": is_refusal(ans.answer),
                                "passages": len(ans.contexts), "latency_s": round(ans.latency_s, 3)})
        cells = []
        for c in categories:
            group = [x for x in results if x["category"] == c]
            cells.append(_rate(sum(x["refused"] for x in group), len(group)))
        fr = answerable.get(mode.value, [])
        cells.append(_rate(sum(fr), len(fr)) if fr else "-")
        lines.append(f"| {mode.value} | " + " | ".join(cells) + " |")
        report["modes"][mode.value] = {"samples": results, "false_refusals_answerable": sum(fr), "answerable_n": len(fr)}

    table = "\n".join(lines)
    console.print(table)
    out = PROJECT_ROOT / f"results/refusal_eval_{datetime.now().strftime('%Y-%m-%d')}.json"
    out.write_text(json.dumps({**report, "table": table}, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"Saved {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
