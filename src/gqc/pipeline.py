"""The gate sequence every algo passes before it trades (AGENTS.md "Algo pipeline").

G0 integrity   no lookahead, deterministic, finite positions                (hard)
G1 overfit     PBO (CSCV) + Deflated Sharpe + MinBTL over ALL configs tried   (hard if REJECT)
G2 robustness  best config still profitable at 2x costs and +1 step lag      (hard)
G3 stability   positive PnL in >= 3 of 4 time slices                         (soft)
G4 risk        total PnL >= 2 x max drawdown                                  (soft)

Trading frequency is part of the search: every strategy is also run rebalancing only every
k steps and with a no-trade band, and those variants count as trials.
Decision: any hard fail -> NO-GO (size 0); any soft fail or CAUTION -> GO-REDUCED (half the
limits); else GO (full limits).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from gqc import overfit, research

GO_SIZE, REDUCED_SIZE = 1.0, 0.5


@dataclass(frozen=True)
class Gate:
    name: str
    passed: bool
    hard: bool
    detail: str


@dataclass(frozen=True)
class Evaluation:
    decision: str  # GO | GO-REDUCED | NO-GO
    size_fraction: float  # fraction of position limits to use
    best_config: dict
    gates: tuple[Gate, ...]
    lines: tuple[str, ...]


def throttle(positions: np.ndarray, every: int = 1, band: float = 0.0) -> np.ndarray:
    """Re-decide only every `every` steps, and only if the change exceeds `band`."""
    out = np.zeros(len(positions))
    held = 0.0
    for t, target in enumerate(np.asarray(positions, float)):
        if t % max(every, 1) == 0 and np.isfinite(target) and abs(target - held) > band:
            held = float(target)
        out[t] = held
    return out


def _throttled(strategy: research.Strategy) -> research.Strategy:
    def run(prices, every=1, band=0.0, **cfg):
        return {s: throttle(p, every, band) for s, p in strategy(prices, **cfg).items()}
    return run


def _peeks(strategy, prices, cfg, base, seed: int) -> bool:
    """True if positions before a cut change when only prices from the cut onward change."""
    n = len(next(iter(prices.values())))
    rng = np.random.default_rng(seed)
    for cut in (n // 3, n // 2, 2 * n // 3):
        for direction in (1.0, -1.0):
            bumped = {}
            for s, p in prices.items():
                big = 10 * float(np.nanstd(np.diff(p))) + 1.0
                tail = p[cut:] + direction * big * (1 + rng.uniform(0, 1, n - cut))
                bumped[s] = np.concatenate([p[:cut], np.maximum(tail, 1e-6)])
            moved = strategy(bumped, **cfg)
            if any(not np.allclose(base[s][:cut], moved[s][:cut]) for s in base):
                return True
    return False


def _integrity(strategy, prices, cfg, seed: int) -> Gate:
    first, second = strategy(prices, **cfg), strategy(prices, **cfg)
    problems = []
    if any(not np.array_equal(first[s], second[s]) for s in first):
        problems.append("non-deterministic")
    if any(not np.all(np.isfinite(first[s])) for s in first):
        problems.append("non-finite positions")
    if _peeks(strategy, prices, cfg, first, seed):
        problems.append("LOOKAHEAD: past positions change when future prices change")
    return Gate("G0 integrity", not problems, True, "; ".join(problems) or "ok")


def _slices_positive(pnl: np.ndarray, k: int = 4) -> int:
    return sum(float(chunk.sum()) > 0 for chunk in np.array_split(pnl, k))


def _max_drawdown(pnl: np.ndarray) -> float:
    equity = np.cumsum(pnl)
    return float(np.max(np.maximum.accumulate(equity) - equity)) if len(equity) else 0.0


def _decide(gates: Sequence[Gate], caution: bool) -> tuple[str, float]:
    if any(g.hard and not g.passed for g in gates):
        return "NO-GO", 0.0
    if caution or any(not g.passed for g in gates):
        return "GO-REDUCED", REDUCED_SIZE
    return "GO", GO_SIZE


def _frequency_table(strategy, prices, base_cfg, frequencies, lag, cost_bps) -> list[str]:
    lines = [f"Frequency check (best params, 2x costs): {'every':>5} {'Sharpe/step':>12} {'PnL':>10}"]
    for k in frequencies:
        pnl = research.positions_pnl(prices, strategy(prices, every=k, **base_cfg), lag, 2 * cost_bps)
        lines.append(f"{'':41}{k:5d} {overfit.sharpe(pnl):12.4f} {pnl.sum():10.2f}")
    return lines


def evaluate(
    strategy: research.Strategy,
    prices: Mapping[str, np.ndarray],
    grid: Mapping[str, Sequence],
    cost_bps: float = 1.0,
    lag: int = 1,
    frequencies: Sequence[int] = (1, 2, 5, 10),
    bands: Sequence[float] = (0.0,),
    seed: int = 0,
) -> Evaluation:
    run = _throttled(strategy)
    full_grid = {**grid, "every": list(frequencies), "band": list(bands)}
    configs, matrix = research.grid_returns(run, prices, full_grid, lag, cost_bps)
    srs = np.atleast_1d(overfit.sharpe(matrix, axis=0))
    best_idx = int(np.argmax(srs))
    best, pnl = configs[best_idx], matrix[:, best_idx]

    report = research.overfit_report(configs, matrix)
    verdict_line = report[-1]
    rejected, caution = "REJECT" in verdict_line, "CAUTION" in verdict_line
    positions = run(prices, **best)
    stressed = [research.positions_pnl(prices, positions, lag, 2 * cost_bps),
                research.positions_pnl(prices, positions, lag + 1, cost_bps)]
    base_sr = float(srs[best_idx])
    robust = pnl.sum() > 0 and all(overfit.sharpe(s) > max(0.0, 0.5 * base_sr) for s in stressed)
    dd = _max_drawdown(pnl)
    gates = (
        _integrity(run, prices, best, seed),
        Gate("G1 overfit", not rejected, True, verdict_line.removeprefix("VERDICT: ")),
        Gate("G2 robustness", robust, True,
             f"Sharpe base {base_sr:.4f}, 2x cost {overfit.sharpe(stressed[0]):.4f}, "
             f"lag+1 {overfit.sharpe(stressed[1]):.4f}"),
        Gate("G3 stability", _slices_positive(pnl) >= 3, False,
             f"{_slices_positive(pnl)}/4 time slices profitable"),
        Gate("G4 risk", pnl.sum() >= 2 * dd, False, f"PnL {pnl.sum():.2f} vs max DD {dd:.2f}"),
    )
    decision, size = _decide(gates, caution)
    base_cfg = {k: v for k, v in best.items() if k != "every"}
    lines = (
        *report[:-1],
        *(f"{g.name:14} {'PASS' if g.passed else 'FAIL'}{' (hard)' if g.hard else ''}: {g.detail}"
          for g in gates),
        *_frequency_table(run, prices, base_cfg, frequencies, lag, cost_bps),
        f"DECISION: {decision}  size = {size:.0%} of position limits  config = {best}",
    )
    return Evaluation(decision, size, best, gates, tuple(lines))
