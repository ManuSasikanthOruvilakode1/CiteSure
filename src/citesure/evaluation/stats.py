"""Uncertainty estimates and refusal detection for evaluation results.

With 100 questions, a difference of a few points between configurations can be noise.
Bootstrap intervals (for means of per-question scores) and Wilson intervals (for rates)
make that visible instead of leaving it to the reader.
"""

from __future__ import annotations

import math
import re

import numpy as np

from citesure.generation.prompts import REFUSAL_MESSAGE

# Phrases a grounded answer uses to say the documents do not cover the question. The
# generator is told to use REFUSAL_MESSAGE exactly, but it may also decline in its own words.
_DECLINE_PATTERNS = re.compile(
    r"(do(?:es)?n['’]?t|do(?:es)? not)\s+(?:cover|mention|contain|include|provide|describe|specify)"
    r"|not (?:covered|mentioned|described|specified) in"
    r"|no (?:information|mention) (?:about|of|on)",
    re.IGNORECASE,
)


def is_refusal(answer: str) -> bool:
    """True if the answer declines rather than answering (exact refusal or explicit decline)."""
    return REFUSAL_MESSAGE.lower() in answer.lower() or bool(_DECLINE_PATTERNS.search(answer))


def _shingles(text: str, n: int = 5) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i : i + n]) for i in range(max(len(words) - n + 1, 1))}


def evidence_rank(retrieved: list[str], evidence: str, min_overlap: float = 0.5) -> int | None:
    """1-based rank of the first retrieved passage that overlaps the gold evidence text.

    Overlap is the share of word 5-grams the two texts have in common, relative to the
    shorter one, so the label stays valid when the corpus is re-chunked at a different size.
    """
    gold = _shingles(evidence)
    for rank, text in enumerate(retrieved, start=1):
        cand = _shingles(text)
        if gold and cand and len(gold & cand) / min(len(gold), len(cand)) >= min_overlap:
            return rank
    return None


def bootstrap_ci(
    values: list[float], confidence: float = 0.95, n_resamples: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of ``values`` with a percentile-bootstrap confidence interval: (mean, low, high)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_resamples, arr.size), replace=True).mean(axis=1)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(means, [alpha, 1 - alpha])
    return float(arr.mean()), float(low), float(high)


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """Rate with a Wilson score interval: (rate, low, high). Well-behaved near 0 and 1."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    z = {0.90: 1.6449, 0.95: 1.96, 0.99: 2.5758}.get(confidence, 1.96)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)
