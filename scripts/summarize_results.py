"""Add 95% bootstrap confidence intervals to a saved evaluation results file.

No models are run: per-question scores are read from the results JSON written by
evaluate_pipeline.py, and Hit@k / reciprocal rank are recomputed per question from the
gold source chunk recorded in the QA dataset.

Usage: python scripts/summarize_results.py [results/fastapi_eval_2026-10-12.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from citesure.config import PROJECT_ROOT
from citesure.evaluation.stats import bootstrap_ci


def _fmt(values: list[float]) -> str:
    mean, lo, hi = bootstrap_ci(values)
    return f"{mean:.2f} [{lo:.2f}, {hi:.2f}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, nargs="?", default=PROJECT_ROOT / "results/fastapi_eval_2026-10-12.json")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data/evaluation/qa_pairs.jsonl")
    args = parser.parse_args()

    gold = {}
    for line in args.dataset.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            gold[row["question"]] = row.get("source_chunk_id", "")

    reports = json.loads(args.results.read_text(encoding="utf-8"))["reports"]
    print("| Mode | Hit@5 | MRR | Context relevance | Faithfulness | Answer relevance |")
    print("|---|---|---|---|---|---|")
    for rep in reports:
        ok = [s for s in rep["samples"] if "error" not in s]
        hits, rr = [], []
        for s in ok:
            g = gold.get(s["question"], "")
            rank = s["contexts"].index(g) + 1 if g in s["contexts"] else 0
            hits.append(1.0 if rank else 0.0)
            rr.append(1.0 / rank if rank else 0.0)
        cols = [hits, rr] + [[s[k] for s in ok] for k in ("context_relevance", "faithfulness", "answer_relevance")]
        print(f"| {rep['mode']} | " + " | ".join(_fmt(c) for c in cols) + " |")
    print("\nValues are mean [95% bootstrap CI] over questions; overlapping intervals mean the "
          "difference is not established by this sample.")


if __name__ == "__main__":
    main()
