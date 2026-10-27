# v8 (round 3, exclusive claims): order-independent competition model.
#
# Rules: one accepter on a turn gets the reward; two or more accepters all get zero; every
# accepter is locked out for `lockout` turns. Rewards ~ Exponential(mean 10).
# We never rely on the order of `other_scores`; only on aggregates:
#   locked_count(s) = number of rivals flagged locked at s
#   new claims at s from the change in locked_count plus lockouts that just expired
#   (lock-flag timing, same turn or next, is chosen as the one consistent with rival payouts).
# From history we learn P(no rival claims | reward size, number of free rivals). Expected
# payoff of accepting r = r * P(win). Lockout opportunity cost: finite-horizon DP on the
# empirical distribution of those expected payoffs. Accept iff r * P(win) >= c[t].

import bisect
import math

MU = 10.0  # published: exponential rewards with mean 10
WARMUP = 10
REFRESH_EARLY, REFRESH = 5, 20
R_EDGES = (4.0, 8.0, 13.0, 20.0, 30.0)  # reward bins
F_EDGES = (0, 1, 3, 6, 10)  # free-rival-count bins (lower edges)

_S = {}


def _reset(T, L):
    _S.clear()
    _S.update(last_t=0, T=T, L=L, xs=[], lc=[], tot=[], nfree=[], new=[[], []], viol=[0, 0],
              c=None, next_refresh=0, eff=[], wins={}, n_riv=0)


def _rbin(r):
    return bisect.bisect_right(R_EDGES, r)


def _fbin(nf):
    return bisect.bisect_right(F_EDGES, nf) - 1


def _p_win(r, nf):
    stats = _S["wins"]
    rb, fb = _rbin(r), _fbin(nf)
    w, n = stats.get((rb, fb), (0.0, 0.0))
    wr, nr = 0.0, 0.0
    for (b, _), (ww, nn) in stats.items():  # same reward bin, any free count: prior
        if b == rb:
            wr += ww
            nr += nn
    prior = (wr + 0.5) / (nr + 1.0)
    return (w + 2.0 * prior) / (n + 2.0)


def _observe(t, other_scores):
    L = _S["L"]
    flags = [bool(f) for _, f in other_scores]
    total = sum(float(s) for s, _ in other_scores)
    s = t - 1  # observed state is for turn s
    lc_now = sum(flags)
    _S["n_riv"] = len(flags)
    lc, tot, new = _S["lc"], _S["tot"], _S["new"]
    prev_lc = lc[-1] if lc else 0
    prev_tot = tot[-1] if tot else total
    lc.append(lc_now)
    tot.append(total)
    _S["nfree"].append(len(flags) - lc_now)
    d = lc_now - prev_lc
    # hypothesis 0: flag set in the claim turn s, stays L+1 turns
    exp0 = new[0][s - L - 2] if s - L - 2 >= 0 else 0
    new[0].append(max(d + exp0, 0))  # new[0][s-1] = claims at turn s
    # hypothesis 1: flag set the turn after the claim, stays L turns -> d reveals claims at s-1
    exp1 = new[1][s - L - 2] if s - L - 2 >= 0 else 0
    if s >= 2:
        new[1].append(max(d + exp1, 0))  # new[1][s-2] = claims at turn s-1
    paid = total - prev_tot > 1e-9
    if paid:
        if new[0] and new[0][-1] == 0:
            _S["viol"][0] += 1
        if len(new[1]) >= 1 and new[1][-1] == 0:
            _S["viol"][1] += 1  # rough: compares with previous turn's claims


def _learn(t):
    """Record whether a lone claim would have won at the latest fully observed turn."""
    h = 0 if _S["viol"][0] <= _S["viol"][1] else 1
    claims = _S["new"][h]
    k = len(claims)  # claims known for turns 1..k
    if k == 0:
        return
    turn = k
    if turn > len(_S["xs"]) or turn - 1 >= len(_S["nfree"]):
        return
    r = _S["xs"][turn - 1]
    nf_idx = turn - 2 if turn >= 2 else 0
    nf = _S["nfree"][nf_idx] if _S["nfree"] else 0
    won = 1.0 if claims[turn - 1] == 0 else 0.0
    key = (_rbin(r), _fbin(nf))
    w, n = _S["wins"].get(key, (0.0, 0.0))
    if _S.get("last_learned", 0) < turn:
        _S["wins"][key] = (w + won, n + 1.0)
        _S["last_learned"] = turn


def _excess_fn():
    srt = sorted(_S["eff"])
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


def play(reward: float, lockout: int, t: int, T: int, your_score: float,
         other_scores: list[tuple[float, bool]], is_locked: bool) -> bool:
    try:
        if not _S or t <= _S["last_t"] or _S["T"] != T or _S["L"] != lockout:
            _reset(T, lockout)
        _S["last_t"] = t
        r = float(reward)
        if not math.isfinite(r):
            return False
        _S["xs"].append(r)
        if other_scores:
            _observe(t, other_scores)
            _learn(t)
        nf = (_S["n_riv"] - _S["lc"][-1]) if _S["lc"] else 0
        pw = _p_win(r, nf) if other_scores else 1.0
        _S["eff"].append(max(r, 0.0) * pw)
        if is_locked or not r > 0:
            return False
        if t >= T:
            return pw > 0.3
        if len(_S["xs"]) < WARMUP:
            return False
        if _S["c"] is None or t >= _S["next_refresh"]:
            _S["c"] = _table(T, lockout, _excess_fn())
            _S["next_refresh"] = t + (REFRESH_EARLY if t < 150 else REFRESH)
        return r * pw >= _S["c"][t]
    except Exception:
        return False
