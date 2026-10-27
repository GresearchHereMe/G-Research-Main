"""The gate sequence every algo passes before it trades (AGENTS.md "Algo pipeline").

G0 integrity   no lookahead, deterministic, finite positions                (hard)
G1 overfit     current-grid PBO + DSR/MinBTL with recorded trial counts       (hard if REJECT)
G2 robustness  best config still profitable at 2x costs and +1 step lag      (hard)
G3 stability   positive PnL in >= 3 of 4 development slices                  (soft)
G4 risk        development PnL >= 2 x max drawdown                            (soft)
G5 holdout     selected config profitable on the untouched final window      (soft)

Selection, PBO/DSR and G1-G4 use only the development window [0, train_frac); G0 checks the
RAW strategy output (before throttling) for causality and finiteness.

Trading frequency is part of the search: every strategy is also run rebalancing only every
k steps and with a no-trade band, and those variants count as trials. Selection ranks the
median Sharpe of neighbouring numeric grid values on development data at doubled costs;
categorical choices are compared separately. Ties prefer the candidate's own stressed Sharpe.
Decision: any hard fail -> NO-GO (size 0); any soft fail or CAUTION -> GO-REDUCED (half the
limits); else GO (full limits).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

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
    return research.max_drawdown(pnl)


def _select_plateau(configs: Sequence[Mapping], grid: Mapping[str, Sequence],
                    scores: np.ndarray) -> int:
    """Median over immediate numeric neighbours; categorical settings stay fixed."""
    plateaus = []
    for cfg in configs:
        near = {}
        for key, values in grid.items():
            if all(isinstance(v, Real) and not isinstance(v, bool) for v in values):
                ordered = sorted(set(values))
                i = ordered.index(cfg[key])
                near[key] = ordered[max(0, i - 1) : i + 2]
            else:
                near[key] = [cfg[key]]
        neighbours = [i for i, other in enumerate(configs)
                      if all(other[k] in near[k] for k in grid)]
        plateaus.append(float(np.median(scores[neighbours])))
    return max(range(len(configs)), key=lambda i: (plateaus[i], scores[i], -i))


def _decide(gates: Sequence[Gate], caution: bool) -> tuple[str, float]:
    if any(g.hard and not g.passed for g in gates):
        return "NO-GO", 0.0
    if caution or any(not g.passed for g in gates):
        return "GO-REDUCED", REDUCED_SIZE
    return "GO", GO_SIZE


def _frequency_table(strategy, prices, base_cfg, frequencies, lag, cost_bps, split) -> list[str]:
    lines = [f"Frequency check (dev window, 2x costs):   {'every':>5} {'Sharpe/step':>12} {'PnL':>10}"]
    for k in frequencies:
        pos = strategy(prices, every=k, **base_cfg)
        pnl = research.positions_pnl(prices, pos, lag, 2 * cost_bps)[:split]
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
    train_frac: float = 0.7,
    prior_trials: int = 0,
) -> Evaluation:
    """Select and gate on the development window [0, split); report the holdout once (G5)."""
    run = _throttled(strategy)
    full_grid = {**grid, "every": list(frequencies), "band": list(bands)}
    configs, matrix = research.grid_returns(run, prices, full_grid, lag, cost_bps)
    split = int(len(matrix) * train_frac)
    dev = matrix[:split]
    _, stressed = research.grid_returns(run, prices, full_grid, lag, 2 * cost_bps)
    srs = np.atleast_1d(overfit.sharpe(stressed[:split], axis=0))
    best_idx = _select_plateau(configs, full_grid, srs)
    best = configs[best_idx]
    report = research.overfit_report(configs, dev, selected=best_idx, prior_trials=prior_trials)
    raw_cfg = {k: v for k, v in best.items() if k not in ("every", "band")}
    gates = _gates(run, strategy, prices, best, raw_cfg, matrix[:, best_idx], split,
                   report[-1], lag, cost_bps, seed)
    decision, size = _decide(gates, "CAUTION" in report[-1])
    base_cfg = {k: v for k, v in best.items() if k != "every"}
    lines = (
        f"Development window [0, {split}) used for selection; holdout [{split}, {len(matrix)})",
        "Selection: neighbouring-grid median Sharpe at 2x costs (categorical settings fixed)",
        *report,
        *(f"{g.name:14} {'PASS' if g.passed else 'FAIL'}{' (hard)' if g.hard else ''}: {g.detail}"
          for g in gates),
        *_frequency_table(run, prices, base_cfg, frequencies, lag, cost_bps, split),
        f"DECISION: {decision}  size = {size:.0%} of position limits  config = {best}",
    )
    return Evaluation(decision, size, best, gates, tuple(lines))


def _gates(run, strategy, prices, best, raw_cfg, pnl_full, split, verdict_line, lag, cost_bps,
           seed) -> tuple[Gate, ...]:
    pnl, holdout = pnl_full[:split], pnl_full[split:]
    positions = run(prices, **best)
    stressed = [research.positions_pnl(prices, positions, lag, 2 * cost_bps)[:split],
                research.positions_pnl(prices, positions, lag + 1, cost_bps)[:split]]
    base_sr = float(overfit.sharpe(pnl))
    robust = pnl.sum() > 0 and all(overfit.sharpe(s) > max(0.0, 0.5 * base_sr) for s in stressed)
    dd = _max_drawdown(pnl)
    return (
        _integrity(strategy, prices, raw_cfg, seed),  # RAW targets: throttling hides bad values
        Gate("G1 overfit", "REJECT" not in verdict_line, True,
             verdict_line.removeprefix("VERDICT: ")),
        Gate("G2 robustness", robust, True,
             f"Sharpe base {base_sr:.4f}, 2x cost {overfit.sharpe(stressed[0]):.4f}, "
             f"lag+1 {overfit.sharpe(stressed[1]):.4f}"),
        Gate("G3 stability", _slices_positive(pnl) >= 3, False,
             f"{_slices_positive(pnl)}/4 development slices profitable"),
        Gate("G4 risk", pnl.sum() >= 2 * dd, False, f"PnL {pnl.sum():.2f} vs max DD {dd:.2f}"),
        Gate("G5 holdout", len(holdout) == 0 or holdout.sum() > 0, False,
             f"holdout PnL {holdout.sum():.2f}, Sharpe/step {overfit.sharpe(holdout):.4f} "
             f"(final check, look once, never tune on it)"),
    )
