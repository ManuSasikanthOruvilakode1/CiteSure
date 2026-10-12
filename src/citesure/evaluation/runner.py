"""Run the pipeline over an evaluation set and aggregate RAG Triad scores per mode."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from citesure.config import EvaluationConfig
from citesure.evaluation.metrics import TriadEvaluator, retrieval_hit_metrics
from citesure.pipeline import RAGPipeline
from citesure.retrieval.hybrid import RetrievalMode


@dataclass
class EvalSample:
    question: str
    ground_truth: str = ""
    source_chunk_id: str = ""


@dataclass
class ModeReport:
    mode: str
    n: int
    context_relevance: float
    faithfulness: float
    answer_relevance: float
    latency_p50_s: float
    latency_p95_s: float
    hit_rate: float | None = None
    mrr: float | None = None
    errors: int = 0
    samples: list[dict] = field(default_factory=list)


def load_dataset(path: Path, limit: int | None = None) -> list[EvalSample]:
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    samples = [
        EvalSample(r["question"], r.get("ground_truth", ""), r.get("source_chunk_id", ""))
        for r in rows
    ]
    return samples[:limit] if limit else samples


def evaluate_mode(
    pipeline: RAGPipeline,
    evaluator: TriadEvaluator,
    samples: list[EvalSample],
    mode: RetrievalMode,
    workers: int = 4,
) -> ModeReport:
    # Answer sequentially so latency reflects a single request, then judge in parallel.
    answers = [pipeline.answer(s.question, mode) for s in samples]

    def judge(i: int) -> dict:
        a = answers[i]
        try:
            scores = asdict(evaluator.evaluate(a.question, a.answer, a.contexts))
        except Exception as exc:  # noqa: BLE001 - one bad judgement must not sink the run
            scores = {"error": f"{type(exc).__name__}: {exc}"}
        return {
            "question": a.question,
            "answer": a.answer,
            "contexts": [c.chunk.chunk_id for c in a.contexts],
            "latency_s": a.latency_s,
            **scores,
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(judge, range(len(answers))))

    ok = [r for r in rows if "error" not in r]
    latencies = [a.latency_s for a in answers]

    def mean(key: str) -> float:
        return float(np.mean([r[key] for r in ok])) if ok else 0.0

    report = ModeReport(
        mode=mode.value,
        n=len(samples),
        context_relevance=mean("context_relevance"),
        faithfulness=mean("faithfulness"),
        answer_relevance=mean("answer_relevance"),
        latency_p50_s=float(np.percentile(latencies, 50)) if latencies else 0.0,
        latency_p95_s=float(np.percentile(latencies, 95)) if latencies else 0.0,
        errors=len(rows) - len(ok),
        samples=rows,
    )
    gold = [(s.source_chunk_id, r["contexts"]) for s, r in zip(samples, rows) if s.source_chunk_id]
    if gold:
        hits = retrieval_hit_metrics([ids for _, ids in gold], [g for g, _ in gold])
        report.hit_rate, report.mrr = hits["hit_rate"], hits["mrr"]
    return report


def check_gate(report: ModeReport, thresholds: EvaluationConfig) -> list[str]:
    """Return human-readable failures; an empty list means the quality gate passed."""
    failures = []
    checks = [
        ("context_relevance", report.context_relevance, thresholds.min_context_relevance),
        ("faithfulness", report.faithfulness, thresholds.min_faithfulness),
        ("answer_relevance", report.answer_relevance, thresholds.min_answer_relevance),
    ]
    for name, value, minimum in checks:
        if value < minimum:
            failures.append(f"{name} {value:.3f} < {minimum:.2f}")
    if report.errors:
        failures.append(f"{report.errors} samples failed to evaluate")
    return failures


def markdown_table(reports: list[ModeReport]) -> str:
    lines = [
        "| Mode | Context Relevance | Faithfulness | Answer Relevance | Hit@k | MRR | Latency p95 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        hit = f"{r.hit_rate:.2f}" if r.hit_rate is not None else "-"
        mrr = f"{r.mrr:.2f}" if r.mrr is not None else "-"
        lines.append(
            f"| {r.mode} | {r.context_relevance:.2f} | {r.faithfulness:.2f} | "
            f"{r.answer_relevance:.2f} | {hit} | {mrr} | {r.latency_p95_s:.2f} s |"
        )
    return "\n".join(lines)
