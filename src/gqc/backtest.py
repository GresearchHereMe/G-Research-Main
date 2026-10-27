"""Bar-level pairs backtester: execution lag, linear costs, PnL accounting and metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from gqc import core


@dataclass(frozen=True)
class BacktestResult:
    equity: np.ndarray  # cumulative PnL after costs, per tick
    positions: np.ndarray  # executed shares (n, 2): y, x
    step_pnl: np.ndarray
    costs: np.ndarray
    metrics: dict


def _executed(targets: np.ndarray, lag: int) -> np.ndarray:
    exec_pos = np.zeros_like(targets)
    exec_pos[lag:] = targets[: len(targets) - lag] if lag else targets
    return exec_pos


def run_targets(
    py: np.ndarray,
    px: np.ndarray,
    targets: np.ndarray,
    lag: int = 1,
    cost_bps: float = 0.0,
    half_spread_y: np.ndarray | float = 0.0,
    half_spread_x: np.ndarray | float = 0.0,
) -> BacktestResult:
    """Targets decided at t are held from t+lag, trading at the price of t+lag."""
    prices = np.column_stack([py, px]).astype(float)
    exec_pos = _executed(np.asarray(targets, float), lag)
    held = np.vstack([np.zeros((1, 2)), exec_pos[:-1]])
    pnl = np.zeros(len(prices))
    pnl[1:] = np.sum(held[1:] * np.diff(prices, axis=0), axis=1)
    trades = np.abs(exec_pos - held)
    half_spreads = np.column_stack([np.broadcast_to(half_spread_y, len(py)),
                                    np.broadcast_to(half_spread_x, len(px))])
    costs = np.sum(trades * (prices * cost_bps / 1e4 + half_spreads), axis=1)
    equity = np.cumsum(pnl - costs)
    metrics = _metrics(pnl - costs, costs, exec_pos)
    return BacktestResult(equity, exec_pos, pnl - costs, costs, metrics)


def run_pair(
    py: np.ndarray,
    px: np.ndarray,
    params: core.Params,
    lag: int = 1,
    cost_bps: float = 0.0,
    half_spread_y: np.ndarray | float = 0.0,
    half_spread_x: np.ndarray | float = 0.0,
) -> BacktestResult:
    trader = core.PairsTrader(params)
    targets = np.zeros((len(py), 2))
    for t, (y, x) in enumerate(zip(py, px)):
        signal = trader.step(float(y), float(x))
        targets[t] = core.leg_targets(signal, float(y), float(x), params)
    return run_targets(py, px, targets, lag, cost_bps, half_spread_y, half_spread_x)


def _round_trips(net: np.ndarray, exec_pos: np.ndarray) -> list[float]:
    """PnL per holding episode (contiguous same-side y position)."""
    side = np.sign(exec_pos[:, 0])
    trips, current, active = [], 0.0, 0
    for t in range(len(net)):
        prev = side[t - 1] if t else 0
        current += net[t]
        if side[t] != prev:
            if active:
                trips.append(current - (net[t] if side[t] else 0.0))
            current = net[t] if side[t] else 0.0
            active = int(side[t] != 0)
    return trips  # completed episodes only; an episode still open at the end is excluded


def _metrics(net: np.ndarray, costs: np.ndarray, exec_pos: np.ndarray) -> dict:
    """Open inventory at the end is marked at the last price, NOT closed (check the organiser's
    terminal rule); `open_at_end` flags it. Drawdown is measured from the zero starting equity."""
    equity = np.concatenate([[0.0], np.cumsum(net)])
    sd = float(np.std(net))
    trips = _round_trips(net, exec_pos)
    return {
        "total_pnl": float(equity[-1]),
        "costs": float(np.sum(costs)),
        "sharpe": float(np.mean(net) / sd * math.sqrt(len(net))) if sd > 0 else 0.0,
        "max_drawdown": float(np.max(np.maximum.accumulate(equity) - equity)),
        "open_at_end": bool(len(exec_pos) and np.any(exec_pos[-1] != 0)),
        "round_trips": len(trips),
        "win_rate": float(np.mean([p > 0 for p in trips])) if trips else 0.0,
        "avg_trip_pnl": float(np.mean(trips)) if trips else 0.0,
        "time_in_market": float(np.mean(exec_pos[:, 0] != 0)),
    }
