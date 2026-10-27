"""Replay a paste-ready submission (`step(book, positions) -> orders`) over a price history.

Purpose: prove the code we paste behaves like the research version (same PnL ballpark, no
errors, limits respected) before it goes live. Orders decided at t fill at the price of t+lag
with a linear cost; PnL is marked to the price series.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

StepFn = Callable[[dict, dict], dict]


@dataclass(frozen=True)
class ReplayResult:
    pnl: np.ndarray
    trades: int
    final_positions: dict = field(default_factory=dict)
    max_abs_position: dict = field(default_factory=dict)


def load_submission(path: str | Path) -> dict:
    """exec() the file into a fresh namespace exactly like a paste box would."""
    ns: dict = {"__name__": "submission"}
    source = Path(path).read_text()
    exec(compile(source, str(path), "exec"), ns)  # noqa: S102 - intentional, mimics platform
    return ns


def replay(
    step: StepFn, prices: Mapping[str, np.ndarray], lag: int = 1, cost_bps: float = 0.0
) -> ReplayResult:
    symbols = list(prices)
    n = len(prices[symbols[0]])
    positions = {s: 0.0 for s in symbols}
    pending: list[tuple[int, dict]] = []
    pnl = np.zeros(n)
    stats = {"trades": 0}
    peak = {s: 0.0 for s in symbols}
    last_marks: dict[str, float] = {}

    def fill(orders: dict, px: dict, t: int) -> None:
        for s, qty in orders.items():
            if s in px and qty:
                positions[s] += qty
                pnl[t] -= abs(qty) * px[s] * cost_bps / 1e4
                stats["trades"] += 1
                peak[s] = max(peak[s], abs(positions[s]))

    for t in range(n):
        px = {s: float(prices[s][t]) for s in symbols if np.isfinite(prices[s][t])}
        pnl[t] += sum(positions[s] * (p - last_marks[s])
                      for s, p in px.items() if s in last_marks)
        last_marks.update(px)  # a missing quote does not erase the previous inventory mark
        for _, orders in [p for p in pending if p[0] == t]:
            fill(orders, px, t)
        pending = [p for p in pending if p[0] > t]
        orders = step(dict(px), {s: q for s, q in positions.items() if q}) or {}
        for s, qty in orders.items():
            if not np.isfinite(qty):
                raise ValueError(f"non-finite order for {s} at step {t}: {qty}")
        if orders and lag == 0:
            fill(orders, px, t)  # same-step fill at this step's price
        elif orders:
            pending.append((t + lag, dict(orders)))
    final = {s: q for s, q in positions.items() if q}
    return ReplayResult(pnl, stats["trades"], final, peak)
