"""Generic research harness for strategies written live at the event.

Write a strategy as  f(prices: {sym: array}, **params) -> {sym: position array}  (positions
decided with data up to and including t). `grid_returns` runs it over a parameter grid and
`overfit_report` tells you whether the best configuration is real or a product of the search.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from gqc import overfit

Strategy = Callable[..., Mapping[str, np.ndarray]]


def _ffill(prices: np.ndarray) -> np.ndarray:
    p = np.asarray(prices, float).copy()
    for t in range(1, len(p)):
        if not np.isfinite(p[t]):
            p[t] = p[t - 1]
    return p


def positions_pnl(
    prices: Mapping[str, np.ndarray],
    positions: Mapping[str, np.ndarray],
    lag: int = 1,
    cost_bps: float = 0.0,
) -> np.ndarray:
    """Per-step PnL: position decided at t is held from t+lag; linear cost on traded notional."""
    n = len(next(iter(prices.values())))
    pnl = np.zeros(n)
    for sym, pos in positions.items():
        p = _ffill(prices[sym])
        target = np.nan_to_num(np.asarray(pos, float))
        held = np.zeros(n)
        held[lag:] = target[: n - lag] if lag else target
        pnl[1:] += held[:-1] * np.nan_to_num(np.diff(p))
        trades = np.abs(np.diff(held, prepend=0.0))
        pnl -= trades * np.nan_to_num(p) * cost_bps / 1e4
    return pnl


def grid_returns(
    strategy: Strategy,
    prices: Mapping[str, np.ndarray],
    grid: Mapping[str, Sequence],
    lag: int = 1,
    cost_bps: float = 0.0,
) -> tuple[list[dict], np.ndarray]:
    """Run every parameter combination; return (configs, T x N matrix of per-step PnL)."""
    keys = list(grid)
    configs = [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]
    columns = [positions_pnl(prices, strategy(prices, **cfg), lag, cost_bps) for cfg in configs]
    return configs, np.column_stack(columns)


def choose_blocks(n_rows: int, min_rows_per_block: int = 20) -> int:
    """Largest even S in {16, 14, ..., 4} leaving at least min_rows_per_block rows per block."""
    for s in range(16, 3, -2):
        if n_rows // s >= min_rows_per_block:
            return s
    return 4


def verdict(pbo: float, dsr: float, n_rows: int, min_btl: int) -> str:
    if pbo > 0.5 or dsr < 0.5:
        return ("REJECT the tuning: the search is mostly fitting noise. Use the simplest/default "
                "parameters, fewer knobs, smaller size.")
    if pbo > 0.2 or dsr < 0.95 or n_rows < min_btl:
        return ("CAUTION: material overfitting risk. Prefer plateau parameters, shrink the grid, "
                "trade smaller.")
    return "OK: the edge survives PBO and Deflated Sharpe. Size up to limits."


def overfit_report(
    configs: Sequence[Mapping], matrix: np.ndarray, n_blocks: int | None = None
) -> list[str]:
    t, n = matrix.shape
    s = n_blocks or choose_blocks(t)
    srs = np.atleast_1d(overfit.sharpe(matrix, axis=0))
    best = int(np.argmax(srs))
    res = overfit.cscv_pbo(matrix, n_blocks=s)
    sr_var = float(np.var(srs, ddof=1)) if n > 1 else 1.0 / t
    dsr = overfit.deflated_sharpe(matrix[:, best], n, sr_var)
    min_btl = overfit.min_backtest_length(n, float(srs[best]))
    med_is = float(np.median(res.is_selected))
    decay = float(np.median(res.oos_selected)) / med_is if med_is > 0 else 0.0
    return [
        f"== Overfitting check: N={n} configs tried, T={t} steps, CSCV S={s} ({res.n_combos} splits)",
        f"PBO  P(in-sample winner below out-of-sample median): {res.pbo:.2f}",
        (f"P(in-sample winner loses out of sample): {res.prob_oos_loss:.2f}; "
        f"median OOS/IS Sharpe of the winner: {decay:.2f} (1 = no decay, <=0 = luck)"),
        (f"Best by Sharpe: {dict(configs[best])}  Sharpe/step {srs[best]:.4f}, "
        f"PnL {matrix[:, best].sum():.2f}"),
        f"Deflated Sharpe P(true Sharpe > best-of-{n}-noise): {dsr:.3f}",
        f"MinBTL at this Sharpe with {n} trials: {min_btl} steps (have {t})",
        f"VERDICT: {verdict(res.pbo, dsr, t, min_btl)}",
    ]
