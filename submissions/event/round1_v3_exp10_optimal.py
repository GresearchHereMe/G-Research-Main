# v2: exact finite-horizon optimal stopping with online distribution learning.
#
# Accepting at round t locks us out for `lockout` rounds. Dynamic programming backwards from T:
#   c[t] = V[t+1] - V[min(t+L+1, T+1)]        (value of staying free minus value after lockout)
#   V[t] = V[t+1] + E[(r - c[t])^+]
# Accept iff reward >= KAPPA * c[t]. E[(r-c)^+] comes from the reward distribution, learned
# online: exponential with a running mean if rewards look exponential (CV ~ 1, non-negative),
# otherwise the empirical distribution of all rewards seen this game (observed every round).
# Every team gets the full reward on the same stream, so other teams do not change our payoff.

import bisect
import math

KAPPA = 1.05  # rounds 1-2 rank on BEST score: slightly pickier = fatter upper tail; use 1.0 for round 3
KNOWN_MU = 10.0  # round 1 published p(x) = (1/10) exp(-x/10). None = learn online. Auto-ignored if data disagrees.
PRIOR_MEAN = 10.0  # round 1 announced Exponential(mean=10); only a prior, data takes over fast
PRIOR_WEIGHT = 30  # pseudo-observations behind the prior
REFRESH = 25  # recompute the threshold table every REFRESH rounds

_S = {}


def _reset(T, L):
    _S.clear()
    _S.update(last_t=0, T=T, L=L, xs=[], sum=0.0, sumsq=0.0, c=None, next_refresh=0)


def _table(T, L, excess):
    V = [0.0] * (T + L + 3)
    c = [0.0] * (T + 2)
    for t in range(T, 0, -1):
        ct = V[t + 1] - V[min(t + L + 1, T + 1)]
        c[t] = ct
        V[t] = V[t + 1] + excess(ct)
    return c


def _known_ok():
    """Use the published distribution unless >= 50 rewards clearly contradict it."""
    xs, n = _S["xs"], len(_S["xs"])
    if KNOWN_MU is None:
        return False
    if n < 50:
        return True
    m = _S["sum"] / n
    sd = math.sqrt(max(_S["sumsq"] / n - m * m, 0.0))
    se = KNOWN_MU / math.sqrt(n)
    return abs(m - KNOWN_MU) < 4 * se and 0.75 <= sd / KNOWN_MU <= 1.3 and min(xs) >= 0.0


def _excess_fn():
    if _known_ok():
        mu = KNOWN_MU
        return lambda cut: mu * math.exp(-max(cut, 0.0) / mu) - min(cut, 0.0)
    xs, n = _S["xs"], len(_S["xs"])
    w = PRIOR_WEIGHT
    if n >= 15 and not 0.5 <= (_S["sum"] / n) / PRIOR_MEAN <= 2.0:
        w = 0  # data clearly disagrees with the round-1 prior: trust the data only
    mean = (_S["sum"] + PRIOR_MEAN * w) / (n + w)
    if n >= 30:
        m = _S["sum"] / n
        var = max(_S["sumsq"] / n - m * m, 0.0)
        cv = math.sqrt(var) / m if m > 0 else 0.0
        if not (0.8 <= cv <= 1.25 and min(xs) >= 0.0):  # not exponential-like: use empirical
            srt = sorted(xs)
            pre = [0.0]
            for x in srt:
                pre.append(pre[-1] + x)
            total = pre[-1]

            def empirical(cut):
                i = bisect.bisect_right(srt, cut)
                return ((total - pre[i]) - cut * (n - i)) / n

            return empirical
    mu = max(mean, 1e-9)
    return lambda cut: mu * math.exp(-max(cut, 0.0) / mu) - min(cut, 0.0)


def play(reward: float, lockout: int, t: int, T: int, your_score: float,
         other_scores: list[tuple[float, bool]], is_locked: bool) -> bool:
    try:
        if not _S or t <= _S["last_t"] or _S["T"] != T or _S["L"] != lockout:
            _reset(T, lockout)
        _S["last_t"] = t
        r = float(reward)
        _S["xs"].append(r)
        _S["sum"] += r
        _S["sumsq"] += r * r
        if is_locked or r <= 0:
            return False
        if t >= T:
            return True
        if _S["c"] is None or t >= _S["next_refresh"]:
            _S["c"] = _table(T, lockout, _excess_fn())
            _S["next_refresh"] = t + (5 if t < 100 else REFRESH)
        return r >= KAPPA * _S["c"][t]
    except Exception:
        return False
