"""
Statistics primitives for the eval harness (bootstrap confidence intervals)
and judge calibration (Spearman correlation, weighted Cohen's kappa).

The build prompt asked to reuse SpandanNagale/Assay's `lib/stats` rather than
reimplementing these from scratch. Assay (github.com/SpandanNagale/Assay) is
a TypeScript/Next.js web app, not a Python package, so there is nothing to
`pip install` -- this module ports the same well-established formulas
(bootstrap CI, paired bootstrap, Spearman rank correlation, quadratic-weighted
Cohen's kappa) Assay's `lib/stats/bootstrap.ts` / `paired-bootstrap.ts`
implement, rather than inventing different statistical approaches. Stdlib
only (`random`/`statistics`), no new dependency.
"""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple


@dataclass
class ConfidenceInterval:
    point_estimate: float
    lower: float
    upper: float
    confidence: float


def bootstrap_ci(
    samples: Sequence[float],
    statistic: Callable[[Sequence[float]], float] = statistics.mean,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> ConfidenceInterval:
    """
    Percentile bootstrap confidence interval for `statistic` over `samples`.
    Deterministic across runs (fixed seed) so two harness runs on identical
    data produce an identical interval -- important for CI regression checks.
    """
    if not samples:
        raise ValueError("bootstrap_ci requires at least one sample")

    point_estimate = statistic(samples)
    if len(samples) == 1:
        return ConfidenceInterval(point_estimate, point_estimate, point_estimate, confidence)

    rng = random.Random(seed)
    n = len(samples)
    resampled_stats = []
    for _ in range(n_resamples):
        resample = [samples[rng.randrange(n)] for _ in range(n)]
        resampled_stats.append(statistic(resample))
    resampled_stats.sort()

    alpha = (1 - confidence) / 2
    lower = resampled_stats[int(alpha * n_resamples)]
    upper = resampled_stats[min(n_resamples - 1, int((1 - alpha) * n_resamples))]
    return ConfidenceInterval(point_estimate, lower, upper, confidence)


def paired_bootstrap_ci(
    baseline: Sequence[float],
    candidate: Sequence[float],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> ConfidenceInterval:
    """
    Bootstrap CI over the paired difference (candidate - baseline), for
    comparing two runs of the same golden set (e.g. this run vs. the
    committed baseline). An interval that excludes 0 is a real regression
    or improvement, not sampling noise.
    """
    if len(baseline) != len(candidate):
        raise ValueError("paired_bootstrap_ci requires equal-length paired samples")
    diffs = [c - b for b, c in zip(baseline, candidate)]
    return bootstrap_ci(diffs, statistics.mean, n_resamples, confidence, seed)


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation coefficient, -1..1. Ties get averaged ranks."""
    if len(x) != len(y):
        raise ValueError("spearman_correlation requires equal-length sequences")
    if len(x) < 2:
        raise ValueError("spearman_correlation requires at least 2 points")

    rx = _rank(x)
    ry = _rank(y)
    return _pearson(rx, ry)


def _rank(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = average_rank
        i = j + 1
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    mean_x, mean_y = statistics.mean(x), statistics.mean(y)
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    var_x = sum((a - mean_x) ** 2 for a in x)
    var_y = sum((b - mean_y) ** 2 for b in y)
    denom = (var_x * var_y) ** 0.5
    if denom == 0:
        return 0.0
    return cov / denom


def weighted_cohens_kappa(
    ratings_a: Sequence[int], ratings_b: Sequence[int], categories: Sequence[int]
) -> float:
    """
    Quadratic-weighted Cohen's kappa between two raters over ordinal
    `categories` (e.g. a 1-5 scale) -- rewards near-misses (a 4 vs a 5) more
    than plain unweighted kappa, appropriate for judge-vs-human agreement on
    a Likert-style score rather than a nominal label.
    """
    if len(ratings_a) != len(ratings_b):
        raise ValueError("weighted_cohens_kappa requires equal-length rating sequences")
    n = len(ratings_a)
    if n == 0:
        raise ValueError("weighted_cohens_kappa requires at least one rating pair")

    categories = list(categories)
    k = len(categories)
    index = {c: i for i, c in enumerate(categories)}

    weights = [[1 - ((i - j) ** 2) / ((k - 1) ** 2) for j in range(k)] for i in range(k)] if k > 1 else [[1.0]]

    observed = [[0] * k for _ in range(k)]
    for a, b in zip(ratings_a, ratings_b):
        observed[index[a]][index[b]] += 1

    row_totals = [sum(row) for row in observed]
    col_totals = [sum(observed[i][j] for i in range(k)) for j in range(k)]

    po = sum(weights[i][j] * observed[i][j] for i in range(k) for j in range(k)) / n
    pe = sum(weights[i][j] * row_totals[i] * col_totals[j] for i in range(k) for j in range(k)) / (n * n)

    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile (p in [0, 100])."""
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((p / 100) * (len(ordered) - 1))))
    return ordered[index]
