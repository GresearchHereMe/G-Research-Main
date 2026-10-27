"""Synthetic price pairs with known ground truth, for testing estimators and strategies."""

from __future__ import annotations

import numpy as np


def cointegrated_pair(
    n: int,
    beta: float = 1.0,
    alpha: float = 0.0,
    theta: float = 0.05,
    sigma_s: float = 0.01,
    sigma_x: float = 0.01,
    x0: float = 100.0,
    seed: int = 0,
    break_at: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Prices (y, x) with ln y = alpha + beta ln x + s, s an AR(1)/OU spread.

    After `break_at` the spread becomes a random walk (cointegration break).
    """
    rng = np.random.default_rng(seed)
    log_x = np.log(x0) + np.cumsum(rng.normal(0.0, sigma_x, n))
    shocks = rng.normal(0.0, sigma_s, n)
    spread = np.zeros(n)
    for t in range(1, n):
        keep = 1.0 if break_at is not None and t >= break_at else 1.0 - theta
        spread[t] = keep * spread[t - 1] + shocks[t]
    log_y = alpha + beta * log_x + spread
    return np.exp(log_y), np.exp(log_x)


def random_walk_pair(
    n: int, sigma: float = 0.01, x0: float = 100.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Two independent geometric random walks (no cointegration)."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, sigma, (2, n))
    paths = np.log(x0) + np.cumsum(steps, axis=1)
    return np.exp(paths[0]), np.exp(paths[1])
