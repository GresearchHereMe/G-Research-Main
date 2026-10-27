# v5: learned optimal stopping + automatic mean-reversion / momentum handling.
#
# Accepting at round t locks us out for `lockout` rounds.
# Case 1, rewards independent (default): backwards dynamic programming
#   c[t] = V[t+1] - V[min(t+L+1, T+1)],  V[t] = V[t+1] + E[(r - c[t])^+]
#   accept iff reward >= KAPPA * c[t]; E[(r-c)^+] from the empirical distribution of all rewards
#   observed so far (we see the reward every round, including while locked).
# Case 2, rewards autocorrelated (lag-1 autocorrelation statistically AND practically
#   significant): fit AR(1) r_t = mu + phi (r_{t-1} - mu) + e and solve the DP on a reward grid,
#   where the current reward is the state:
#   W_t(r) = max(r + E[W_{t+L+1} | r], E[W_{t+1} | r]); accept iff r + E[W_{t+L+1}|r] >= E[W_{t+1}|r].
#   Mean reversion (phi < 0): after a low reward a better one is likely soon -> wait.
#   Momentum (phi > 0): good rewards cluster -> timing adapts to the current level.
# First WARMUP rounds: observe only. Never accept a non-positive reward. Never raises.

import bisect
import math

import numpy as np

KAPPA = 1.05  # rounds 1-2 rank on BEST score; use 1.0 for round 3 (latest score counts)
KNOWN_MU = None  # None = learn the distribution online
KNOWN_NORMAL = (0.0, 1.0)  # round 2: N(0,1) confirmed from earlier runs; None = learn. Auto-off if data disagrees.
NORMAL_PRIOR_N = 300  # pseudo-observations behind KNOWN_NORMAL (shrinks in-game estimates towards it)
WARMUP = 10  # rounds of pure observation before accepting anything
RECENT = 400  # window used when a level shift is detected
REFRESH_EARLY, REFRESH = 5, 20  # rounds between threshold-table recomputes (iid mode)
AR_MIN_N, AR_REFRESH, AR_MIN_PHI, AR_Z = 200, 100, 0.10, 3.0
GRID = 81

_S = {}


def _reset(T, L):
    _S.clear()
    _S.update(last_t=0, T=T, L=L, xs=[], c=None, next_refresh=0, ar=None, ar_next=AR_MIN_N)


# ----------------------------------------------------------------------------- iid mode
def _sample():
    xs = _S["xs"]
    if len(xs) >= 2 * RECENT:
        old, new = xs[:-RECENT], xs[-RECENT:]
        mo, mn = sum(old) / len(old), sum(new) / len(new)
        vo = sum((x - mo) ** 2 for x in old) / len(old)
        vn = sum((x - mn) ** 2 for x in new) / len(new)
        se = math.sqrt(vo / len(old) + vn / len(new)) or 1e-12
        if abs(mn - mo) / se > 4.0:
            return new
    return xs


def _normal_from_prior():
    """Shrunk (mean, sd) towards KNOWN_NORMAL, or None if the data clearly disagrees."""
    if KNOWN_NORMAL is None:
        return None
    m0, s0 = KNOWN_NORMAL
    xs = _S["xs"]
    n = len(xs)
    if n == 0:
        return m0, s0
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / n
    if n >= 50:
        sd = math.sqrt(var)
        if abs(m - m0) > 4 * s0 / math.sqrt(n) or not 0.8 <= sd / s0 <= 1.25:
            return None
        skew = sum((x - m) ** 3 for x in xs) / n / max(sd, 1e-12) ** 3
        if abs(skew) > 0.5:
            return None
    w = NORMAL_PRIOR_N
    mean = (m * n + m0 * w) / (n + w)
    var_s = (var * n + s0 * s0 * w) / (n + w)
    return mean, math.sqrt(max(var_s, 1e-18))


def _excess_fn():
    if KNOWN_MU is not None:
        mu = KNOWN_MU
        return lambda cut: mu * math.exp(-max(cut, 0.0) / mu)
    nf = _normal_from_prior()
    if nf is not None:
        m, sd = nf

        def gaussian(cut):
            cut = max(cut, 0.0)
            z = (cut - m) / sd
            return (sd * math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
                    + (m - cut) * 0.5 * math.erfc(z / math.sqrt(2)))

        return gaussian
    srt = sorted(_sample())
    n = len(srt)
    pre = [0.0]
    for x in srt:
        pre.append(pre[-1] + x)
    total = pre[-1]

    def empirical(cut):
        cut = max(cut, 0.0)
        i = bisect.bisect_right(srt, cut)
        return ((total - pre[i]) - cut * (n - i)) / n

    return empirical


def _table(T, L, excess):
    V = [0.0] * (T + L + 3)
    c = [0.0] * (T + 2)
    for t in range(T, 0, -1):
        ct = V[t + 1] - V[min(t + L + 1, T + 1)]
        c[t] = ct
        V[t] = V[t + 1] + excess(ct)
    return c


# ----------------------------------------------------------------------------- AR(1) mode
def _erf(x):
    # Abramowitz-Stegun 7.1.26, |error| < 1.5e-7, vectorised
    s = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
               + 0.254829592) * t * np.exp(-a * a)
    return s * y


def _ar_fit():
    x = np.asarray(_S["xs"][-1500:], dtype=float)
    n = len(x)
    mu = float(x.mean())
    d = x - mu
    den = float((d[:-1] ** 2).sum())
    if n < AR_MIN_N or den <= 0:
        return None
    phi = float((d[1:] * d[:-1]).sum() / den)
    if abs(phi) < max(AR_MIN_PHI, AR_Z / math.sqrt(n)) or abs(phi) >= 0.995:
        return None
    sig = float((d[1:] - phi * d[:-1]).std())
    return mu, phi, max(sig, 1e-9)


def _ar_tables(T, L, mu, phi, sig):
    s_stat = sig / math.sqrt(1.0 - phi * phi)
    grid = mu + s_stat * np.linspace(-4.5, 4.5, GRID)
    edges = np.concatenate(([-np.inf], (grid[1:] + grid[:-1]) / 2, [np.inf]))

    def trans(k):
        m = mu + (phi ** k) * (grid - mu)
        sd = math.sqrt(max(sig * sig * (1 - phi ** (2 * k)) / (1 - phi * phi), 1e-18))
        z = (edges[None, :] - m[:, None]) / (sd * math.sqrt(2.0))
        cdf = 0.5 * (1.0 + _erf(np.clip(z, -40, 40)))
        return np.diff(cdf, axis=1)

    p1, pl = trans(1), trans(L + 1)
    gain = np.where(grid > 0, grid, -np.inf)  # never take non-positive rewards
    W = np.zeros((T + L + 3, GRID))
    cont = np.zeros((T + 2, GRID))
    after = np.zeros((T + 2, GRID))
    for t in range(T, 0, -1):
        cont[t] = p1 @ W[t + 1]
        after[t] = pl @ W[min(t + L + 1, T + 1)]
        W[t] = np.maximum(gain + after[t], cont[t])
    return grid, cont, after


# ----------------------------------------------------------------------------- entry point
def play(reward: float, lockout: int, t: int, T: int, your_score: float,
         other_scores: list[tuple[float, bool]], is_locked: bool) -> bool:
    try:
        if not _S or t <= _S["last_t"] or _S["T"] != T or _S["L"] != lockout:
            _reset(T, lockout)
        _S["last_t"] = t
        r = float(reward)
        if math.isfinite(r):
            _S["xs"].append(r)
        if t >= _S["ar_next"] and T >= 200:
            fit = _ar_fit()
            _S["ar"] = (fit, _ar_tables(T, lockout, *fit)) if fit else None
            _S["ar_next"] = t + AR_REFRESH
        if is_locked or not r > 0:
            return False
        if t >= T:
            return True
        if len(_S["xs"]) < WARMUP and KNOWN_NORMAL is None and KNOWN_MU is None:
            return False
        if _S["ar"] is not None:
            grid, cont, after = _S["ar"][1]
            take = r + float(np.interp(r, grid, after[t]))
            wait = float(np.interp(r, grid, cont[t]))
            return take >= wait
        if _S["c"] is None or t >= _S["next_refresh"]:
            _S["c"] = _table(T, lockout, _excess_fn())
            _S["next_refresh"] = t + (REFRESH_EARLY if t < 150 else REFRESH)
        return r >= KAPPA * _S["c"][t]
    except Exception:
        return False
