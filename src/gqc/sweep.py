"""Parameter sweeps with train/test split and plateau scoring (median of grid neighbours)."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from gqc import backtest, core, stats


@dataclass(frozen=True)
class SweepRow:
    params: core.Params
    train_pnl: float
    test_pnl: float
    train_sharpe: float
    test_sharpe: float
    trips: int
    plateau: float  # median train PnL over this config and its grid neighbours


def _sharpe(pnl: np.ndarray) -> float:
    sd = float(np.std(pnl))
    return float(np.mean(pnl) / sd * np.sqrt(len(pnl))) if sd > 0 else 0.0


def _static_fit(py: np.ndarray, px: np.ndarray, split: int, use_log: bool) -> tuple[float, float]:
    y, x = (np.log(py[:split]), np.log(px[:split])) if use_log else (py[:split], px[:split])
    alpha, beta, _ = stats.ols(y, x)
    return beta, alpha


def sweep_pair(
    py: np.ndarray,
    px: np.ndarray,
    train_frac: float = 0.6,
    cost_bps: float = 1.0,
    lag: int = 1,
    base: core.Params | None = None,
    entries: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5),
    halflives: tuple[float, ...] = (20, 50, 100, 200),
    exits: tuple[float, ...] = (0.0, 0.5),
    hedges: tuple[str, ...] = ("kalman", "ewm", "static"),
) -> list[SweepRow]:
    """Grid-search on the train window only; report out-of-sample test PnL alongside."""
    base = base or core.Params()
    split = int(len(py) * train_frac)
    beta, alpha = _static_fit(py, px, split, base.use_log)
    raw: dict[tuple, tuple] = {}
    for hedge, entry, hl, ex in itertools.product(hedges, entries, halflives, exits):
        params = base._replace(hedge=hedge, entry_z=entry, halflife=hl, exit_z=ex,
                               static_beta=beta, static_alpha=alpha,
                               stop_z=max(base.stop_z, entry + 1.5))
        res = backtest.run_pair(py, px, params, lag=lag, cost_bps=cost_bps)
        train, test = res.step_pnl[:split], res.step_pnl[split:]
        raw[(hedge, entry, hl, ex)] = (params, train, test, res.metrics["round_trips"])
    return _rank(raw, entries, halflives)


def _rank(raw: dict, entries: tuple, halflives: tuple) -> list[SweepRow]:
    rows = []
    for (hedge, entry, hl, ex), (params, train, test, trips) in raw.items():
        ei, hi = entries.index(entry), halflives.index(hl)
        neighbours = [
            float(np.sum(raw[(hedge, entries[i], halflives[j], ex)][1]))
            for i in range(max(ei - 1, 0), min(ei + 2, len(entries)))
            for j in range(max(hi - 1, 0), min(hi + 2, len(halflives)))
        ]
        rows.append(SweepRow(params, float(np.sum(train)), float(np.sum(test)),
                             _sharpe(train), _sharpe(test), trips, float(np.median(neighbours))))
    return sorted(rows, key=lambda r: r.plateau, reverse=True)
