"""Market-behaviour diagnostics: find out what kind of edge the data offers BEFORE writing logic.

Each probe maps to a playbook action (PLAYBOOK.md §4-§5):
  variance ratio / autocorr  -> mean reversion (fade) vs momentum (follow)
  lead-lag                   -> trade the laggard on the leader's move
  cointegration screen       -> spread trading
  order-book imbalance       -> microprice as fair value, skew quotes
  quote staleness / spread   -> pick off stale quotes, market-make wide spreads
  markouts by counterparty   -> who is dumb (trade against / quote to) vs informed (follow)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from gqc import stats

MEAN_REVERTING_VR, TRENDING_VR = 0.8, 1.2


def log_returns(prices: np.ndarray) -> np.ndarray:
    """Returns over valid observations only (gaps removed) — for single-series statistics."""
    r = aligned_log_returns(prices)
    return r[np.isfinite(r)]


def aligned_log_returns(prices: np.ndarray) -> np.ndarray:
    """Returns on the original time grid; NaN wherever either endpoint is missing/invalid."""
    p = np.asarray(prices, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.diff(np.log(np.where(p > 0, p, np.nan)))


def autocorr(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, float)
    if len(x) <= lag + 2 or np.std(x) == 0:
        return 0.0
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])


def variance_ratio(returns: np.ndarray, k: int) -> float:
    """Var of k-step returns / (k * var of 1-step). <1 mean-reverting, >1 trending."""
    r = np.asarray(returns, float)
    if len(r) < 2 * k or np.var(r) == 0:
        return 1.0
    agg = np.convolve(r, np.ones(k), mode="valid")
    return float(np.var(agg) / (k * np.var(r)))


def lead_lag(a: np.ndarray, b: np.ndarray, max_lag: int = 5) -> dict[int, float]:
    """corr(a_t, b_{t+lag}); a large value at lag > 0 means a leads b."""
    out = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x, y = a[: len(a) - lag], b[lag:]
        else:
            x, y = a[-lag:], b[: len(b) + lag]
        n = min(len(x), len(y))
        x, y = np.asarray(x[:n], float), np.asarray(y[:n], float)
        ok = np.isfinite(x) & np.isfinite(y)
        good = ok.sum() > 3 and np.std(x[ok]) > 0 and np.std(y[ok]) > 0
        out[lag] = float(np.corrcoef(x[ok], y[ok])[0, 1]) if good else 0.0
    return out


def tick_size(prices: np.ndarray) -> float:
    uniq = np.unique(np.round(np.asarray(prices, float)[np.isfinite(prices)], 8))
    diffs = np.round(np.diff(uniq), 8)
    diffs = diffs[diffs > 0]
    return float(diffs.min()) if len(diffs) else 0.0


def microprice(bid, ask, bid_size, ask_size):
    """Size-weighted mid: leans toward the side with LESS size (it is about to be eaten)."""
    total = np.asarray(bid_size, float) + np.asarray(ask_size, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        micro = (np.asarray(bid) * ask_size + np.asarray(ask) * bid_size) / total
    return np.where(total > 0, micro, (np.asarray(bid) + np.asarray(ask)) / 2)


def imbalance_signal(bid, ask, bid_size, ask_size, horizon: int = 1) -> float:
    """corr(book imbalance_t, mid_{t+h} - mid_t). > ~0.1 = imbalance predicts the next move."""
    bs, as_ = np.asarray(bid_size, float), np.asarray(ask_size, float)
    imb = (bs - as_) / np.where(bs + as_ > 0, bs + as_, 1)
    mid = (np.asarray(bid, float) + np.asarray(ask, float)) / 2
    move = mid[horizon:] - mid[:-horizon]
    x = imb[:-horizon]
    if np.std(x) == 0 or np.std(move) == 0:
        return 0.0
    return float(np.corrcoef(x, move)[0, 1])


def staleness(bid: np.ndarray, ask: np.ndarray) -> float:
    """Fraction of steps where neither best bid nor best ask changed."""
    same = (np.diff(bid) == 0) & (np.diff(ask) == 0)
    return float(np.mean(same)) if len(same) else 0.0


def markouts_by_group(
    idx: np.ndarray,
    price: np.ndarray,
    sign: np.ndarray,
    mid: np.ndarray,
    horizon: int,
    groups: Sequence[str] | np.ndarray,
) -> dict[str, dict[str, float]]:
    """Per group: mean of sign * (mid[t+h] - price). >0: group is informed; <0: group loses."""
    idx, price, sign = np.asarray(idx), np.asarray(price, float), np.asarray(sign, float)
    groups = np.asarray(groups)
    ok = idx + horizon < len(mid)
    marks = sign[ok] * (np.asarray(mid)[idx[ok] + horizon] - price[ok])
    table = {}
    for g in np.unique(groups[ok]):
        m = marks[groups[ok] == g]
        sd = float(np.std(m))
        table[str(g)] = {"mean": float(np.mean(m)), "n": len(m),
                         "t": float(np.mean(m) / sd * np.sqrt(len(m))) if sd > 0 else 0.0}
    return table


def _behaviour(vr: float) -> str:
    if vr < MEAN_REVERTING_VR:
        return "mean-reverting -> fade moves"
    if vr > TRENDING_VR:
        return "trending -> follow moves"
    return "random-walk-like"


def price_report(
    prices: Mapping[str, np.ndarray],
    max_lag: int = 3,
    *,
    observed_prices: Mapping[str, np.ndarray] | None = None,
) -> list[str]:
    """Use the common `prices` grid for cross-asset tests; optional observations for each series."""
    lines = ["== Per-series behaviour (log returns)",
             f"{'symbol':>10} {'n':>6} {'vol':>9} {'ac1':>6} {'VR5':>6} {'VR20':>6}  read"]
    aligned = {s: aligned_log_returns(p) for s, p in prices.items()}
    per_series = prices if observed_prices is None else observed_prices
    rets = {s: log_returns(p) for s, p in per_series.items()}
    for sym, r in rets.items():
        vr5, vr20 = variance_ratio(r, 5), variance_ratio(r, 20)
        lines.append(f"{sym:>10} {len(r):6d} {np.std(r):9.5f} {autocorr(r, 1):6.2f} "
                     f"{vr5:6.2f} {vr20:6.2f}  {_behaviour(vr20)}")
    lines += ["", "== Lead-lag (|corr| >= 0.1 at a non-zero lag)"]
    names = sorted(aligned)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            n = min(len(aligned[a]), len(aligned[b]))  # same time grid; NaNs masked pairwise
            corr = lead_lag(aligned[a][:n], aligned[b][:n], max_lag)
            lag = max((k for k in corr if k), key=lambda k: abs(corr[k]))
            if abs(corr[lag]) >= 0.1:
                leader, follower = (a, b) if lag > 0 else (b, a)
                lines.append(f"  {leader} leads {follower} by {abs(lag)} step(s): corr {corr[lag]:.2f}")
    if len(prices) >= 2:
        lines += ["", "== Cointegration (top 5, full sample; confirm with screen_pairs.py)"]
        for row in stats.screen(prices)[:5]:
            sig = "*" if row.tstat < row.crit5 else " "
            lines.append(f"  {row.y}/{row.x} beta {row.beta:.3f} t {row.tstat:.2f}{sig} "
                         f"half-life {row.half_life:.1f}")
    return lines


def book_report(sym: str, bid, ask, bid_size=None, ask_size=None) -> list[str]:
    bid, ask = np.asarray(bid, float), np.asarray(ask, float)
    tick = tick_size(np.concatenate([bid, ask]))
    spread = ask - bid
    ticks = np.median(spread) / tick if tick else float("nan")
    header = (
        f"== Book {sym}: tick {tick:g}, spread median {np.median(spread):g} ({ticks:.1f} ticks), "
        f"crossed/locked {np.mean(spread <= 0):.1%}, stale {staleness(bid, ask):.1%}"
    )
    lines = [header]
    if bid_size is not None and ask_size is not None:
        for h in (1, 5):
            s = imbalance_signal(bid, ask, bid_size, ask_size, h)
            hint = " -> use microprice / skew quotes" if abs(s) >= 0.1 else ""
            lines.append(f"   imbalance -> mid move corr (h={h}): {s:.2f}{hint}")
    return lines
