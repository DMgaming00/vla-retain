"""Interval estimates for success rates and paired comparisons."""

from __future__ import annotations

import math

import numpy as np

_Z_95 = 1.959963984540054


def wilson_interval(n_success: int, n: int, z: float = _Z_95) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    if not 0 <= n_success <= n:
        raise ValueError(f"n_success={n_success} is outside [0, {n}]")
    p = n_success / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    # The exact endpoints at p=0 and p=1 are 0 and 1; clamp out floating-point error.
    lo = 0.0 if n_success == 0 else max(0.0, centre - half)
    hi = 1.0 if n_success == n else min(1.0, centre + half)
    return lo, hi


def paired_bootstrap(
    a: np.ndarray | list[float], b: np.ndarray | list[float], n_boot: int = 10_000, seed: int = 0, level: float = 0.95
) -> dict[str, float]:
    """Bootstrap CI of mean(a - b), resampling pairs. Tasks are the natural pairing unit."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or len(a) == 0:
        raise ValueError("a and b must be non-empty 1-D arrays of equal length")
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    tail = (1 - level) / 2
    return {
        "mean_diff": float(diff.mean()),
        "ci_lo": float(np.quantile(means, tail)),
        "ci_hi": float(np.quantile(means, 1 - tail)),
    }
