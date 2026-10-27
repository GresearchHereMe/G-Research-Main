"""Cointegration statistics: OLS, ADF, Engle-Granger with MacKinnon critical values, screening."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

# MacKinnon (2010) response surface, Engle-Granger with constant, N=2 variables:
# crit(T) = tau_inf + b1 / T + b2 / T**2
_MACKINNON_N2 = {
    0.01: (-3.89644, -10.9519, -22.527),
    0.05: (-3.33613, -6.1101, -6.823),
    0.10: (-3.04445, -4.2412, -2.720),
}


def mackinnon_crit(n: int) -> dict[float, float]:
    return {lvl: t + b1 / n + b2 / n**2 for lvl, (t, b1, b2) in _MACKINNON_N2.items()}


def ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float, np.ndarray]:
    """y = alpha + beta * x + resid."""
    design = np.column_stack([np.ones_like(x), x])
    (alpha, beta), *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(alpha), float(beta), y - alpha - beta * x


def adf_tstat(series: np.ndarray, lags: int = 1) -> float:
    """t-statistic of b in  d s_t = a + b s_{t-1} + sum_i c_i d s_{t-i} + e."""
    diff = np.diff(series)
    rows = len(diff) - lags
    cols = [np.ones(rows), series[lags:-1]]
    cols += [diff[lags - i : len(diff) - i] for i in range(1, lags + 1)]
    design = np.column_stack(cols)
    target = diff[lags:]
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    resid = target - design @ coef
    dof = max(rows - design.shape[1], 1)
    sigma2 = resid @ resid / dof
    cov = sigma2 * np.linalg.pinv(design.T @ design)
    return float(coef[1] / math.sqrt(cov[1, 1]))


def half_life(spread: np.ndarray) -> float:
    """OU half-life in ticks from d s_t = a + b s_{t-1}; inf if not mean-reverting."""
    _, b, _ = ols(np.diff(spread), spread[:-1])
    return -math.log(2.0) / math.log1p(b) if -1.0 < b < 0.0 else math.inf


@dataclass(frozen=True)
class CointResult:
    alpha: float
    beta: float
    tstat: float
    crit: Mapping[float, float]
    half_life: float
    n: int

    def is_coint(self, level: float = 0.05) -> bool:
        return self.tstat < self.crit[level]


def eg_coint(y: np.ndarray, x: np.ndarray, lags: int = 1) -> CointResult:
    """Engle-Granger two-step test of y on x (pass log prices for log spreads)."""
    alpha, beta, resid = ols(y, x)
    return CointResult(
        alpha=alpha,
        beta=beta,
        tstat=adf_tstat(resid, lags),
        crit=mackinnon_crit(len(y)),
        half_life=half_life(resid),
        n=len(y),
    )


@dataclass(frozen=True)
class PairRow:
    y: str
    x: str
    beta: float
    alpha: float
    tstat: float
    crit5: float
    half_life: float
    ret_corr: float
    n: int


def schwert_lags(n: int) -> int:
    """ADF lag rule of thumb 4*(n/100)^(1/4); more lags curb over-rejection on noisy walks."""
    return max(1, int(4 * (n / 100) ** 0.25))


def screen(
    prices: Mapping[str, np.ndarray], use_log: bool = True, lags: int | None = None
) -> list[PairRow]:
    """Test every pair in both directions; best direction per pair; most cointegrated first."""
    rows = []
    for a, b in itertools.combinations(sorted(prices), 2):
        pa, pb = np.asarray(prices[a], float), np.asarray(prices[b], float)
        ok = np.isfinite(pa) & np.isfinite(pb) & (pa > 0) & (pb > 0)
        if ok.sum() < 30:
            continue
        sa, sb = (np.log(pa[ok]), np.log(pb[ok])) if use_log else (pa[ok], pb[ok])
        lags = lags if lags is not None else schwert_lags(len(sa))
        corr = float(np.corrcoef(np.diff(sa), np.diff(sb))[0, 1])
        best = min(((a, b, eg_coint(sa, sb, lags)), (b, a, eg_coint(sb, sa, lags))),
                   key=lambda item: item[2].tstat)
        y, x, res = best
        rows.append(PairRow(y, x, res.beta, res.alpha, res.tstat, res.crit[0.05],
                            res.half_life, corr, res.n))
    return sorted(rows, key=lambda r: r.tstat)
