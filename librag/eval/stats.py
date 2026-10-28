"""Statistical tests and intervals used in the paper.

* Paired randomisation (permutation) test for IR metrics — recommended by
  Smucker, Allan & Carterette (2007).
* Holm–Bonferroni correction for multiple comparisons.
* Percentile bootstrap CI for mean metric values.
* Wilson score interval for proportions (hallucination, abstention rates).
* Exact McNemar test for paired binary outcomes (same query, two conditions).
"""
from __future__ import annotations

import numpy as np
from scipy import stats as st


def randomization_test(a, b, n_permutations: int = 10000, seed: int = 42) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if np.allclose(d, 0):
        return 1.0
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, d.size))
    perm = np.abs((signs * d).mean(axis=1))
    return float((np.sum(perm >= obs - 1e-12) + 1) / (n_permutations + 1))


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    items = sorted(pvalues.items(), key=lambda x: x[1])
    m = len(items)
    adjusted, running = {}, 0.0
    for i, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[key] = running
    return adjusted


def bootstrap_ci(values, n_bootstrap: int = 10000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    v = np.asarray(values, float)
    if v.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_bootstrap, v.size), replace=True).mean(axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z = st.norm.ppf(1 - alpha / 2)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def mcnemar_exact(x, y) -> dict:
    """x, y: boolean outcomes for the same items under two conditions."""
    x, y = np.asarray(x, bool), np.asarray(y, bool)
    b = int(np.sum(x & ~y))
    c = int(np.sum(~x & y))
    p = 1.0 if b + c == 0 else float(st.binomtest(min(b, c), b + c, 0.5).pvalue)
    return {"only_first": b, "only_second": c, "p_value": p}


def paired_summary(a, b, n_permutations: int, n_bootstrap: int, seed: int) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b
    lo, hi = bootstrap_ci(diff, n_bootstrap, seed=seed)
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "diff": float(diff.mean()),
            "diff_ci_low": lo, "diff_ci_high": hi,
            "p_value": randomization_test(a, b, n_permutations, seed),
            "wins": int(np.sum(diff > 0)), "losses": int(np.sum(diff < 0)), "ties": int(np.sum(diff == 0))}
