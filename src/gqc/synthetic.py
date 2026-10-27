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


def book_with_bots(
    n: int = 5000, seed: int = 0, tick: float = 0.01
) -> tuple[list[dict], list[dict]]:
    """Synthetic L1 book + trades with planted behaviours, for rehearsing diagnostics.

    Planted: displayed mid overreacts around fair (mean reversion); book imbalance leads the
    next fair move; `noise_bot` trades randomly (loses); `smart_bot` trades the next 10-step
    fair move (wins); `mm_bot` is the passive counterparty.
    """
    rng = np.random.default_rng(seed)
    moves = rng.normal(0, 2 * tick, n)
    fair = 100 + np.cumsum(moves)
    over = np.zeros(n)
    for t in range(1, n):
        over[t] = 0.5 * over[t - 1] + rng.normal(0, 2 * tick)
    mid = np.round((fair + over) / tick) * tick
    nxt = np.append(moves[1:], 0.0)
    imb = np.clip(nxt / (4 * tick) + rng.normal(0, 0.3, n), -0.9, 0.9)
    book = [{"time": t, "symbol": "XYZ", "bid": round(mid[t] - tick, 4),
             "ask": round(mid[t] + tick, 4), "bid_size": round(50 * (1 + imb[t])),
             "ask_size": round(50 * (1 - imb[t]))} for t in range(n)]
    trades = []
    for t in range(0, n - 10, 3):
        smart = t % 2 == 0
        side = np.sign(fair[t + 10] - fair[t]) if smart else rng.choice([-1, 1])
        side = side or 1
        bot = "smart_bot" if smart else "noise_bot"
        buyer, seller = (bot, "mm_bot") if side > 0 else ("mm_bot", bot)
        price = book[t]["ask"] if side > 0 else book[t]["bid"]
        trades.append({"time": t, "symbol": "XYZ", "price": price,
                       "quantity": 5 if smart else int(rng.integers(1, 4)),
                       "buyer": buyer, "seller": seller})
    return book, trades


def random_walk_pair(
    n: int, sigma: float = 0.01, x0: float = 100.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Two independent geometric random walks (no cointegration)."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, sigma, (2, n))
    paths = np.log(x0) + np.cumsum(steps, axis=1)
    return np.exp(paths[0]), np.exp(paths[1])
