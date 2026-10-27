"""Pair parameter sweeps: formation -> development (selection) -> untouched final test.

Windows (chronological):
  [0, formation)        static hedge fit + estimator warm-up; never scored
  [formation, split)    development: grid scored here, plateau selection, overfit evidence
  [split, n)            final test: reported once for the selected config, never used to choose
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from gqc import backtest, core, research, stats


@dataclass(frozen=True)
class SweepRow:
    params: core.Params
    train_pnl: float
    test_pnl: float
    train_sharpe: float
    test_sharpe: float
    trips: int
    plateau: float  # median DEVELOPMENT Sharpe over this config and its grid neighbours
    pnl: np.ndarray = field(repr=False, compare=False, default_factory=lambda: np.zeros(0))
    formation: int = 0
    split: int = 0


def _sharpe(pnl: np.ndarray) -> float:
    sd = float(np.std(pnl))
    return float(np.mean(pnl) / sd * np.sqrt(len(pnl))) if sd > 0 else 0.0


def _static_fit(py: np.ndarray, px: np.ndarray, end: int, use_log: bool) -> tuple[float, float]:
    y, x = (np.log(py[:end]), np.log(px[:end])) if use_log else (py[:end], px[:end])
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
    formation_frac: float = 0.2,
) -> list[SweepRow]:
    """Grid-search scored on the development window only; rows sorted by plateau (best first)."""
    base = base or core.Params()
    n = len(py)
    formation, split = int(n * formation_frac), int(n * train_frac)
    if not 0 < formation < split < n:
        raise ValueError("need 0 < formation_frac < train_frac < 1 with data in every window")
    beta, alpha = _static_fit(py, px, formation, base.use_log)
    raw: dict[tuple, tuple] = {}
    for hedge, entry, hl, ex in itertools.product(hedges, entries, halflives, exits):
        params = base._replace(hedge=hedge, entry_z=entry, halflife=hl, exit_z=ex,
                               static_beta=beta, static_alpha=alpha,
                               stop_z=max(base.stop_z, entry + 1.5))
        res = backtest.run_pair(py, px, params, lag=lag, cost_bps=cost_bps)
        raw[(hedge, entry, hl, ex)] = (params, res.step_pnl, res.metrics["round_trips"])
    return _rank(raw, entries, halflives, formation, split)


def _rank(raw: dict, entries: tuple, halflives: tuple, formation: int, split: int) -> list[SweepRow]:
    """Objective = development Sharpe (scale-free); size is set afterwards from limits/risk."""
    dev_sharpe = {key: _sharpe(pnl[formation:split]) for key, (_, pnl, _) in raw.items()}
    rows = []
    for (hedge, entry, hl, ex), (params, pnl, trips) in raw.items():
        ei, hi = entries.index(entry), halflives.index(hl)
        neighbours = [
            dev_sharpe[(hedge, entries[i], halflives[j], ex)]
            for i in range(max(ei - 1, 0), min(ei + 2, len(entries)))
            for j in range(max(hi - 1, 0), min(hi + 2, len(halflives)))
        ]
        dev, test = pnl[formation:split], pnl[split:]
        rows.append(SweepRow(params, float(np.sum(dev)), float(np.sum(test)), _sharpe(dev),
                             _sharpe(test), trips, float(np.median(neighbours)), pnl,
                             formation, split))
    return sorted(rows, key=lambda r: r.plateau, reverse=True)


def evidence_report(rows: list[SweepRow], prior_trials: int = 0) -> list[str]:
    """Overfitting evidence from DEVELOPMENT data only, for the selected (top-plateau) config."""
    configs = [r.params._asdict() for r in rows]
    dev = np.column_stack([r.pnl[r.formation : r.split] for r in rows])
    return research.overfit_report(configs, dev, selected=0, prior_trials=prior_trials)


def holdout_line(rows: list[SweepRow]) -> str:
    r = rows[0]
    return (f"FINAL TEST (look once; never tune on it): selected config PnL {r.test_pnl:.2f}, "
            f"Sharpe {r.test_sharpe:.2f} over {len(r.pnl) - r.split} steps")
