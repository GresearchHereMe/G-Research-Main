"""G-Research trading game: take-or-skip a reward each turn, taking locks us out for `lockout` turns.

Strategy: finite-horizon optimal stopping. From every reward seen so far (we observe the reward
even while locked) we estimate the reward distribution and solve, by backward induction,

    V[s] = E[max(X + V[s + L + 1], V[s + 1])],      V[s > T] = 0

so a reward is worth taking at turn s iff it beats c[s] = V[s + 1] - V[s + L + 1], the value of
the turns the lockout costs us. c[s] shrinks to zero near the end of the game.

If score changes show that teams accepting the same turn SHARE the reward, the value of a
reward becomes reward * E[1 / (1 + K)], with K the other teams expected to accept: each team's
lockout is tracked from its score jumps and its acceptance rate is learned per reward level.
That is how we exploit greedy/random bots: take rewards while they are locked out.

v2: when the reward distribution is announced (round 1: exponential, mean 10) the thresholds
come from the exact formula E[(X - c)+] = mean * exp(-c / mean), so there is no learning
period. If observed rewards contradict the announced mean, we fall back to learning it.

v3: if rewards are autocorrelated (persistent high/low spells), the distribution of future
rewards depends on the current one. We then bin rewards by level, estimate the turn-to-turn
transition matrix P (smoothed toward the marginal), and solve the same stopping problem over
(turn, level): taking at level j is worth reward + E[G(s + L + 1) | level after L more turns],
using P^L. With i.i.d. rewards this reduces to the plain rule, and it is only switched on when
the rank autocorrelation at the lockout lag is clearly non-zero.

v4 (round 3): the lockout may change from turn to turn, possibly with the reward. We learn
the joint distribution of (reward, lockout) and price each decision with the CURRENT lockout:
take iff value > V[t + 1] - V[t + lockout + 1]. A changing lockout no longer resets learning.

Standard library only; play() never raises and stays well under the 15 ms per call budget.
"""

import math
from bisect import bisect_right

KNOWN_MEAN = None  # announced exponential mean (round 1); None = learn the distribution
WINDOW = 600  # most recent rewards defining the distribution
MIN_SAMPLES = 8  # skip until this many rewards have been seen
RECOMPUTE = 20  # turns between threshold-table rebuilds
TEAM_MEMORY = 300  # eligible-turn records kept per other team
EVIDENCE = 5  # multi-taker turns needed before trusting the sharing test
N_BINS = 8  # reward-level bins for other teams' acceptance rates
TOL = 1e-6
MARKOV = "auto"  # "auto": level-dependent rule only if autocorrelated; "never"; "always"
MARKOV_BINS = 10  # reward levels for the transition matrix
MARKOV_SMOOTH = 3.0  # pseudo-counts pulling each transition row toward the marginal
MARKOV_EVERY = 100  # turns between level-dependent table rebuilds
MARKOV_MIN = 100  # rewards needed before testing autocorrelation


def exp_value_table(mean, t0, T, L):
    """V[s] for exponential rewards and a fixed lockout: E[(X - c)+] = mean * exp(-c / mean)."""
    value = [0.0] * (T + L + 3)
    for s in range(T, t0 - 1, -1):
        cut = value[s + 1] - value[s + L + 1]
        value[s] = value[s + 1] + mean * math.exp(-cut / mean)
    return value


def value_table(pairs, t0, T):
    """V[s]: expected future score from turn s (eligible, reward unseen); pairs = (reward, lockout)."""
    groups = {}
    for x, lk in pairs:
        groups.setdefault(max(int(lk), 0), []).append(x)
    n = len(pairs)
    parts = []
    for lk, xs in groups.items():
        xs.sort()
        suf = [0.0] * (len(xs) + 1)
        for i in range(len(xs) - 1, -1, -1):
            suf[i] = suf[i + 1] + xs[i]
        parts.append((lk, xs, suf))
    value = [0.0] * (T + max(groups, default=0) + 3)
    for s in range(T, t0 - 1, -1):
        nxt = value[s + 1]
        total = 0.0
        for lk, xs, suf in parts:
            cut = nxt - value[s + lk + 1]
            i = bisect_right(xs, cut)
            total += suf[i] - cut * (len(xs) - i)
        value[s] = nxt + total / n
    return value


def rank_autocorr(xs, lag=1):
    """Autocorrelation of ranks at `lag` (robust to heavy tails)."""
    n = len(xs)
    rank = [0] * n
    for r, i in enumerate(sorted(range(n), key=xs.__getitem__)):
        rank[i] = r
    mid = (n - 1) / 2.0
    den = sum((r - mid) ** 2 for r in rank)
    num = sum((rank[i] - mid) * (rank[i + lag] - mid) for i in range(n - lag))
    return num / den if den > 0 else 0.0


def transitions(levels, lag, k, marg):
    """Level-to-level transition probabilities over `lag` turns, smoothed toward the marginal."""
    counts = [[0.0] * k for _ in range(k)]
    for i in range(len(levels) - lag):
        counts[levels[i]][levels[i + lag]] += 1.0
    return [[(counts[a][b] + MARKOV_SMOOTH * marg[b]) / (sum(counts[a]) + MARKOV_SMOOTH)
             for b in range(k)] for a in range(k)]


def markov_table(xs, t0, T, L, bins):
    """Level edges and c[s][level] for s in [t0, T] under a Markov model of reward levels."""
    srt = sorted(xs)
    n = len(srt)
    edges = sorted({srt[n * b // bins] for b in range(1, bins)})
    k = len(edges) + 1
    levels = [bisect_right(edges, x) for x in xs]
    marg = [0.0] * k
    for lv in levels:
        marg[lv] += 1.0 / n
    P = transitions(levels, 1, k, marg)
    PL = transitions(levels, L, k, marg)  # direct L-step estimate keeps the persistence
    groups = [[] for _ in range(k)]
    for x, lv in zip(xs, levels):
        groups[lv].append(x)
    sufs = []
    for g in groups:
        g.sort()
        suf = [0.0] * (len(g) + 1)
        for i in range(len(g) - 1, -1, -1):
            suf[i] = suf[i + 1] + g[i]
        sufs.append(suf)
    zero = [0.0] * k
    G = {}
    thr = {}
    for s in range(T, t0 - 1, -1):
        g1, gl = G.get(s + 1, zero), G.get(s + L + 1, zero)
        cuts, e = [], []
        for j in range(k):
            cut = g1[j] - sum(PL[j][m] * gl[m] for m in range(k))
            grp, suf = groups[j], sufs[j]
            i = bisect_right(grp, cut)
            gain = (suf[i] - cut * (len(grp) - i)) / len(grp) if grp else 0.0
            cuts.append(cut)
            e.append(g1[j] + gain)
        G[s] = [sum(P[a][j] * e[j] for j in range(k)) for a in range(k)]
        if s + L + 2 in G:
            del G[s + L + 2]
        thr[s] = cuts
    return edges, thr


def share_expectation(probs):
    """E[1 / (1 + K)] for K = sum of independent Bernoulli(p)."""
    dist = [1.0]
    for p in probs:
        nxt = [0.0] * (len(dist) + 1)
        for j, mass in enumerate(dist):
            nxt[j] += mass * (1.0 - p)
            nxt[j + 1] += mass * p
        dist = nxt
    return sum(mass / (1.0 + j) for j, mass in enumerate(dist))


def close(a, b):
    return abs(a - b) <= TOL * max(1.0, abs(a), abs(b))


class Solution:
    def __init__(self):
        self.reset(0, 0)

    def reset(self, T, lockout):
        self.T, self.L, self.t_last = T, lockout, 0
        self.rewards = []  # every reward seen, including locked turns
        self.locks = []  # lockout offered with each reward
        self.takers = []  # other teams that took that turn's reward (None = unknown)
        self.V, self.thr_at, self.thr_mode, self.every = None, -(10**9), None, RECOMPUTE
        self.prev_scores, self.prev_own, self.took_last = None, None, False
        self.split_err = self.indep_err = 0.0
        self.evidence = 0
        self.stable = self.unstable = 0
        self.lock_end, self.elig, self.records = [], [], []
        self.edges, self.rates = [], []
        self.pooled = []  # (reward, takers, eligible) when team identity is unreliable
        self.mk_edges, self.mk_thr, self.mk_at, self.mk_on = None, None, -(10**9), False

    # ---- learning from the turn that just finished ----------------------------------------
    def observe(self, t, other_scores, your_score):
        cur = [float(s) for s, _ in other_scores]
        n = len(cur)
        if self.prev_scores is None and t == 2:
            self.prev_scores = [0.0] * n
        if n != len(self.lock_end):
            self.lock_end, self.elig = [-1] * n, [True] * n
            self.records = [[] for _ in range(n)]
            self.rates = []
        if not self.rewards or self.prev_scores is None or len(self.prev_scores) != n:
            return
        r, lock_prev = self.rewards[-1], self.locks[-1]
        changed = [i for i in range(n) if not close(cur[i], self.prev_scores[i])]
        k = self.count_changed(cur)
        self.takers[-1] = k
        stable = k == len(changed)
        self.stable += stable
        self.unstable += not stable
        incs = [cur[i] - self.prev_scores[i] for i in changed] if stable else []
        if self.took_last and self.prev_own is not None:
            incs.append(float(your_score) - self.prev_own)
        m = k + (1 if self.took_last else 0)
        if m >= 2 and abs(r) > TOL and incs:
            for inc in incs:
                ratio = inc / r
                self.indep_err += (ratio - 1.0) ** 2
                self.split_err += (ratio - 1.0 / m) ** 2
            self.evidence += 1
        if stable:
            for i in range(n):
                took = i in changed
                if self.elig[i]:
                    rec = self.records[i]
                    rec.append((r, took))
                    if len(rec) > TEAM_MEMORY:
                        del rec[0]
                if took:
                    self.lock_end[i] = t - 1 + lock_prev
        else:
            free = sum(1 for _, locked in other_scores if not locked)
            self.pooled.append((r, k, max(free, k)))
            if len(self.pooled) > WINDOW:
                del self.pooled[0]

    def count_changed(self, cur):
        old = sorted(self.prev_scores)
        used = [False] * len(old)
        k = 0
        for v in cur:
            i = bisect_right(old, v)
            hit = False
            for j in (i - 1, i, i - 2):
                if 0 <= j < len(old) and not used[j] and close(old[j], v):
                    used[j] = hit = True
                    break
            k += not hit
        return k

    def predict_eligible(self, t, other_scores):
        self.elig = [
            (t > end) if end >= 0 else not locked
            for end, (_, locked) in zip(self.lock_end, other_scores)
        ]

    # ---- mode, thresholds, acceptance model -----------------------------------------------
    def sharing(self):
        return self.evidence >= EVIDENCE and self.split_err < 0.5 * self.indep_err

    def known_ok(self):
        """Announced distribution still plausible: rewards >= 0, mean within 4 standard errors."""
        if KNOWN_MEAN is None:
            return False
        xs = self.rewards
        if not xs or min(xs) < 0:
            return False
        n = len(xs)
        return n < 50 or abs(sum(xs) / n - KNOWN_MEAN) <= 4 * KNOWN_MEAN / math.sqrt(n)

    def markov_on(self, t):
        """Rebuild the level-dependent rule every MARKOV_EVERY turns; True if it applies now."""
        if MARKOV == "never" or len(self.rewards) < MARKOV_MIN or self.L < 1:
            return False
        if any(lk != self.L for lk in self.locks[-WINDOW:]):
            return False  # level rule assumes one fixed lockout
        if t - self.mk_at >= MARKOV_EVERY or t not in (self.mk_thr or {}):
            xs = self.rewards[-WINDOW:]
            rho = rank_autocorr(xs, self.L)  # must persist across a lockout to matter
            self.mk_on = MARKOV == "always" or abs(rho) > 3.0 / math.sqrt(len(xs))
            if self.mk_on:
                end = min(self.T, t + MARKOV_EVERY - 1)
                self.mk_edges, full = markov_table(xs, t, self.T, self.L, MARKOV_BINS)
                self.mk_thr = {s: full[s] for s in range(t, end + 1)}
            self.mk_at = t
        return self.mk_on

    def net_pairs(self, split):
        rows = zip(self.rewards[-WINDOW:], self.locks[-WINDOW:], self.takers[-WINDOW:])
        if not split:
            return [(r, lk) for r, lk, _ in rows]
        return [(r / (1.0 + k), lk) for r, lk, k in rows if k is not None]

    def rebuild(self, t, split, known=False):
        if known:
            self.V = exp_value_table(KNOWN_MEAN, t, self.T, self.L)
            self.thr_at, self.thr_mode = t, (split, known)
            return
        pairs = self.net_pairs(split)
        if len(pairs) < MIN_SAMPLES:
            pairs = self.net_pairs(False)
        self.V = value_table(pairs, t, self.T)
        n_locks = len({lk for _, lk in pairs})
        self.every = RECOMPUTE if n_locks <= 5 else 50  # many lockout values: rebuild less often
        self.thr_at, self.thr_mode = t, (split, known)
        if split:
            xs = sorted(self.rewards[-WINDOW:])
            self.edges = [xs[len(xs) * b // N_BINS] for b in range(1, N_BINS)]
            self.rates = [self.team_rates(rec) for rec in self.records]

    def team_rates(self, rec):
        acc, tot = [0] * N_BINS, [0] * N_BINS
        for r, took in rec:
            b = bisect_right(self.edges, r)
            acc[b] += took
            tot[b] += 1
        prior = (sum(acc) + 0.5) / (sum(tot) + 1.0)
        return [(a + 2.0 * prior) / (n + 2.0) for a, n in zip(acc, tot)]

    def expected_share(self, reward, other_scores):
        b = bisect_right(self.edges, reward)
        if self.stable >= self.unstable and len(self.rates) == len(self.elig):
            probs = [rates[b] for rates, ok in zip(self.rates, self.elig) if ok]
            return share_expectation(probs)
        near = [(k, e) for r, k, e in self.pooled if bisect_right(self.edges, r) == b]
        q = (sum(k for k, _ in near) + 0.5) / (sum(e for _, e in near) + 1.0)
        free = sum(1 for _, locked in other_scores if not locked)
        return share_expectation([min(q, 1.0)] * free)

    def value_at(self, s):
        return self.V[s] if s < len(self.V) else 0.0

    # ---- entry point ------------------------------------------------------------------------
    def play(self, reward, lockout, t, T, your_score, other_scores, is_locked):
        try:
            return bool(self.decide(reward, lockout, t, T, your_score, other_scores, is_locked))
        except Exception:  # noqa: BLE001 - never crash the game; a skipped turn is cheap
            return False

    def decide(self, reward, lockout, t, T, your_score, other_scores, is_locked):
        reward, lockout, t, T = float(reward), int(lockout), int(t), int(T)
        if t <= self.t_last or T != self.T:  # a new game; a changed lockout is not
            self.reset(T, lockout)
        self.L = max(lockout, 0)
        if not math.isfinite(reward):
            reward = 0.0
        other_scores = list(other_scores or [])
        self.observe(t, other_scores, your_score)
        self.predict_eligible(t, other_scores)
        self.prev_scores = [float(s) for s, _ in other_scores] or self.prev_scores
        self.prev_own = float(your_score)
        self.rewards.append(reward)
        self.locks.append(self.L)
        self.takers.append(None)
        self.t_last = t
        self.took_last = False
        if is_locked or reward <= 0:
            return False
        split = self.sharing()
        if not split and self.markov_on(t):
            lv = bisect_right(self.mk_edges, reward)
            take = reward > self.mk_thr[t][lv]
            self.took_last = take
            return take
        known = not split and self.known_ok()
        if not known and len(self.rewards) < MIN_SAMPLES:
            return False
        if self.V is None or t - self.thr_at >= self.every or (split, known) != self.thr_mode:
            self.rebuild(t, split, known)
        value = reward * self.expected_share(reward, other_scores) if split else reward
        take = value > self.value_at(t + 1) - self.value_at(t + self.L + 1)
        self.took_last = take
        return take


play = Solution().play
