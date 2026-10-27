# v7 (round 3, exclusive claims): optimal stopping on the EXPECTED payoff given rival competition.
#
# Each round at most one team is paid. We see every rival's (score, is_locked) at t-1, so:
#   paid at s     -> score jump            -> rival claimed (and won) at s, locked s+1..s+L
#   newly locked  -> flag False -> True    -> rival claimed at s (won or lost a collision)
# From claims we learn each rival's threshold (lowest reward it claims) and the collision rule
# Published round-3 rule: two or more accepters on a turn all score zero, all are locked out.
# Lock-flag timing (set in the claim round or the next) is calibrated from rivals' payouts.
# Expected payoff of claiming r now = r * P(we win | free rivals now, their thresholds).
# Lockout opportunity cost: finite-horizon DP on the empirical distribution of those expected
# payoffs (computed every round for the reward actually offered). Accept iff payoff >= KAPPA*c[t].
# Never raises; never claims non-positive rewards.

import bisect
import math

KAPPA = 1.0  # round 3: latest score counts -> maximise the expected score
WARMUP = 10
REFRESH_EARLY, REFRESH = 5, 20
SOFT = 0.15  # softness (fraction of threshold) of rival claim decisions
MIN_CLAIMS = 3  # claims needed before trusting a rival's own threshold estimate

_S = {}


def _reset(T, L):
    _S.clear()
    _S.update(last_t=0, T=T, L=L, xs=[], eff=[], c=None, next_refresh=0, rivals=None,
              prev_scores=None, prev_flags=None, void=0, winner=0, my_claim=None, my_score=0.0,
              flag_same=0, flag_next=0)


def _rival_threshold(rv, default):
    claims = rv["claims"]
    if len(claims) >= MIN_CLAIMS:
        return min(claims)
    return default


def _win_prob(r, t, default_tau):
    """P(we are paid if we claim r at round t)."""
    probs = []
    for rv in _S["rivals"] or []:
        if rv["lock_until"] >= t or rv["inactive"]:
            continue
        tau = _rival_threshold(rv, default_tau)
        z = (r - tau) / max(SOFT * max(tau, 1e-9), 1e-9)
        probs.append(1.0 / (1.0 + math.exp(-max(min(z, 40.0), -40.0))))
    dist = [1.0]  # Poisson-binomial: number of rival claimants
    for p in probs:
        nxt = [0.0] * (len(dist) + 1)
        for k, q in enumerate(dist):
            nxt[k] += q * (1 - p)
            nxt[k + 1] += q * p
        dist = nxt
    return dist[0]  # published rule: two or more accepters -> all score zero


def _observe(t, other_scores, your_score):
    """Update the rival model from the state at t-1 (and our own claim outcome)."""
    L = _S["L"]
    xs = _S["xs"]
    scores = [float(s) for s, _ in other_scores]
    flags = [bool(f) for _, f in other_scores]
    if _S["rivals"] is None or len(_S["rivals"]) != len(scores):
        _S["rivals"] = [{"claims": [], "lock_until": 0, "inactive": False, "last_claim": 0}
                        for _ in scores]
        _S["prev_scores"], _S["prev_flags"] = scores, flags
        _S["my_score"] = float(your_score)
        return
    s_prev = t - 1  # round whose outcome these scores describe
    paid_now, claims_by_round = set(), {}
    for i, rv in enumerate(_S["rivals"]):
        paid = scores[i] - _S["prev_scores"][i] > 1e-9
        newly_locked = flags[i] and not _S["prev_flags"][i]
        if paid:
            paid_now.add(i)
            if flags[i]:
                _S["flag_same"] += 1  # locked flag already set in the claim round
            else:
                _S["flag_next"] += 1  # locked flag appears the round after
        lag = 1 if _S["flag_next"] > _S["flag_same"] else 0
        # a locked flag after the known lockout ended = a new claim (also catches immediate
        # re-claims, where the flag never visibly switches off)
        relocked = flags[i] and (s_prev - lag) > rv["lock_until"]
        claim_round = s_prev if paid else (s_prev - lag if (newly_locked or relocked) else None)
        if claim_round is not None and claim_round > rv["last_claim"] and 1 <= claim_round <= len(xs):
            rv["claims"].append(xs[claim_round - 1])
            rv["lock_until"] = claim_round + L
            rv["last_claim"] = claim_round
            claims_by_round.setdefault(claim_round, []).append(i)
    # collisions among rivals that we can now see completely (claims made at s_prev - 1)
    for cr, who in claims_by_round.items():
        if cr == s_prev - 1 and len(who) >= 2:
            if any(_S["rivals"][i]["claims"] and i in _S.get("paid_prev", set()) for i in who):
                _S["winner"] += 1
            else:
                _S["void"] += 1
    # our own claim at s_prev: did a rival also claim, and was anyone paid?
    mc = _S["my_claim"]
    if mc is not None and mc == s_prev - 1:
        rivals_too = [i for i in claims_by_round.get(mc, [])]
        if rivals_too:
            if _S.get("my_paid_prev") or any(i in _S.get("paid_prev", set()) for i in rivals_too):
                _S["winner"] += 1
            else:
                _S["void"] += 1
        _S["my_claim"] = None
    _S["my_paid_prev"] = float(your_score) - _S["my_score"] > 1e-9
    _S["paid_prev"] = paid_now
    for rv in _S["rivals"]:
        rv["inactive"] = t > 300 and not rv["claims"]
    _S["prev_scores"], _S["prev_flags"] = scores, flags
    _S["my_score"] = float(your_score)


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
            _observe(t, other_scores, your_score)
        if _S["c"] is not None:
            default_tau = _S["c"][t] / 0.8  # rivals assumed slightly pickier than our cut
        else:
            default_tau = max(_S["xs"])
        pw = _win_prob(r, t, default_tau) if _S["rivals"] else 1.0
        _S["eff"].append(max(r, 0.0) * pw)
        if is_locked or not r > 0:
            return False
        if t >= T:
            _S["my_claim"] = t
            return pw > 0.0
        if len(_S["xs"]) < WARMUP:
            return False
        if _S["c"] is None or t >= _S["next_refresh"]:
            _S["c"] = _table(T, lockout, _excess_fn())
            _S["next_refresh"] = t + (REFRESH_EARLY if t < 150 else REFRESH)
        if r * pw >= KAPPA * _S["c"][t]:
            _S["my_claim"] = t
            return True
        return False
    except Exception:
        return False
