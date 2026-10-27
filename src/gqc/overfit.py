"""Backtest-overfitting statistics.

* PBO via CSCV — Bailey, Borwein, López de Prado & Zhu (2017), "The Probability of Backtest
  Overfitting", J. Computational Finance. Probability that the configuration that is best
  in-sample ranks below the median out-of-sample, over all symmetric half/half block splits.
* PSR / DSR — Bailey & López de Prado (2014), "The Deflated Sharpe Ratio". Probability the true
  Sharpe exceeds a benchmark given skew/kurtosis and sample length; DSR uses the expected max
  Sharpe of N unskilled trials as the benchmark.
* MinBTL — Bailey, Borwein, López de Prado & Zhu (2014), "Pseudo-Mathematics and Financial
  Charlatanism": sample length needed before the best of N noise trials stops looking this good.

All Sharpe ratios here are per period (per row), not annualised.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

EULER_GAMMA = 0.5772156649015329
_NORMAL = NormalDist()


def sharpe(returns: np.ndarray, axis: int = 0) -> np.ndarray | float:
    r = np.asarray(returns, float)
    mean, sd = r.mean(axis=axis), r.std(axis=axis, ddof=1)
    out = np.divide(mean, sd, out=np.zeros_like(np.asarray(mean, float)), where=sd > 0)
    return float(out) if np.ndim(out) == 0 else out


@dataclass(frozen=True)
class PBOResult:
    pbo: float  # P(best in-sample config ranks <= median out-of-sample)
    logits: np.ndarray  # lambda per split; > 0 means IS winner beat OOS median
    n_combos: int
    prob_oos_loss: float  # P(IS winner has negative OOS performance)
    degradation_slope: float  # OLS slope of OOS on IS performance of the IS winner
    is_selected: np.ndarray
    oos_selected: np.ndarray


def _block_sums(matrix: np.ndarray, n_blocks: int) -> tuple[np.ndarray, np.ndarray, int]:
    t, n = matrix.shape
    per = t // n_blocks
    if per < 2:
        raise ValueError(f"need at least {2 * n_blocks} rows for {n_blocks} blocks, got {t}")
    blocks = matrix[t - per * n_blocks :].reshape(n_blocks, per, n)  # drop oldest remainder
    return blocks.sum(axis=1), (blocks**2).sum(axis=1), per


def _perf(sums: np.ndarray, sqs: np.ndarray, count: int, metric: str) -> np.ndarray:
    if metric == "pnl":
        return sums
    mean = sums / count
    var = np.maximum(sqs / count - mean**2, 0.0) * count / (count - 1)
    sd = np.sqrt(var)
    return np.divide(mean, sd, out=np.zeros_like(mean), where=sd > 0)


def _split_masks(n_blocks: int, max_combos: int, seed: int) -> np.ndarray:
    total = math.comb(n_blocks, n_blocks // 2)
    if total <= max_combos:
        combos = itertools.combinations(range(n_blocks), n_blocks // 2)
        rows = [np.isin(np.arange(n_blocks), c) for c in combos]
        return np.array(rows, dtype=float)
    rng = np.random.default_rng(seed)
    rows = [np.isin(np.arange(n_blocks), rng.choice(n_blocks, n_blocks // 2, replace=False))
            for _ in range(max_combos)]
    return np.array(rows, dtype=float)


def cscv_pbo(
    matrix: np.ndarray,
    n_blocks: int = 16,
    metric: str = "sharpe",
    max_combos: int = 20_000,
    seed: int = 0,
) -> PBOResult:
    """matrix: (T steps x N configs) per-step returns/PnL of every configuration tried."""
    if n_blocks % 2 or n_blocks < 2:
        raise ValueError("n_blocks must be an even number >= 2")
    m = np.nan_to_num(np.asarray(matrix, float))
    sums, sqs, per = _block_sums(m, n_blocks)
    masks = _split_masks(n_blocks, max_combos, seed)
    half = per * (n_blocks // 2)
    is_perf = _perf(masks @ sums, masks @ sqs, half, metric)
    oos_perf = _perf((1 - masks) @ sums, (1 - masks) @ sqs, half, metric)
    best = np.argmax(is_perf, axis=1)
    rows = np.arange(len(masks))
    chosen_oos = oos_perf[rows, best]
    below = (oos_perf < chosen_oos[:, None]).sum(axis=1)
    ties = (oos_perf == chosen_oos[:, None]).sum(axis=1)
    n = m.shape[1]
    omega = (below + 0.5 * (ties - 1) + 1) / (n + 1)  # relative OOS rank in (0, 1)
    logits = np.log(omega / (1 - omega))
    chosen_is = is_perf[rows, best]
    slope = float(np.polyfit(chosen_is, chosen_oos, 1)[0]) if np.std(chosen_is) > 0 else 0.0
    return PBOResult(float(np.mean(logits <= 0)), logits, len(masks),
                     float(np.mean(chosen_oos < 0)), slope, chosen_is, chosen_oos)


def expected_max_sharpe(n_trials: int, sr_var: float) -> float:
    """E[max Sharpe] among n_trials unskilled strategies whose Sharpe estimates have variance sr_var."""
    if n_trials <= 1:
        return 0.0
    z = _NORMAL.inv_cdf
    tail = (1 - EULER_GAMMA) * z(1 - 1 / n_trials) + EULER_GAMMA * z(1 - 1 / (n_trials * math.e))
    return math.sqrt(sr_var) * tail


def probabilistic_sharpe(returns: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """P(true per-period Sharpe > sr_benchmark), adjusting for skewness, kurtosis and length."""
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    if len(r) < 3 or np.std(r) == 0:
        return 0.0
    sr = sharpe(r)
    z = (r - r.mean()) / r.std()
    skew, kurt = float(np.mean(z**3)), float(np.mean(z**4))
    denom = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr**2, 1e-12))
    return _NORMAL.cdf((sr - sr_benchmark) * math.sqrt(len(r) - 1) / denom)


def deflated_sharpe(returns: np.ndarray, n_trials: int, sr_var: float) -> float:
    """PSR against the Sharpe that the best of n_trials pure-noise strategies would show."""
    return probabilistic_sharpe(returns, expected_max_sharpe(n_trials, sr_var))


def min_backtest_length(n_trials: int, sr_per_period: float) -> int:
    """Rows needed so the best of n_trials noise strategies is unlikely to reach this Sharpe."""
    if n_trials <= 1 or sr_per_period <= 0:
        return 1
    return math.ceil(2 * math.log(n_trials) / sr_per_period**2)
