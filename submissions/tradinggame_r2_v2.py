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


def threshold_table(sample, t0, T, L):
    """c[s] for s in [t0, T]: take a reward worth more than c[s] at turn s."""
    xs = sorted(sample)
    n = len(xs)
    suffix = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + xs[i]
    value = [0.0] * (T + L + 3)
    thr = {}
    for s in range(T, t0 - 1, -1):
        cut = value[s + 1] - value[s + L + 1]
        i = bisect_right(xs, cut)
        value[s] = value[s + 1] + (suffix[i] - cut * (n - i)) / n
        thr[s] = cut
    return thr


def exp_threshold_table(mean, t0, T, L):
    """Exact c[s] for exponential rewards: E[(X - c)+] = mean * exp(-c / mean)."""
    value = [0.0] * (T + L + 3)
    thr = {}
    for s in range(T, t0 - 1, -1):
        cut = value[s + 1] - value[s + L + 1]
        value[s] = value[s + 1] + mean * math.exp(-cut / mean)
        thr[s] = cut
    return thr


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
        self.takers = []  # other teams that took that turn's reward (None = unknown)
        self.thr, self.thr_at, self.thr_mode = None, -(10**9), None
        self.prev_scores, self.prev_own, self.took_last = None, None, False
        self.split_err = self.indep_err = 0.0
        self.evidence = 0
        self.stable = self.unstable = 0
        self.lock_end, self.elig, self.records = [], [], []
        self.edges, self.rates = [], []
        self.pooled = []  # (reward, takers, eligible) when team identity is unreliable

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
        r = self.rewards[-1]
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
                    self.lock_end[i] = t - 1 + self.L
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

    def net_sample(self, split):
        if not split:
            return self.rewards[-WINDOW:]
        pairs = zip(self.rewards[-WINDOW:], self.takers[-WINDOW:])
        return [r / (1.0 + k) for r, k in pairs if k is not None]

    def rebuild(self, t, split, known=False):
        if known:
            self.thr = exp_threshold_table(KNOWN_MEAN, t, self.T, self.L)
            self.thr_at, self.thr_mode = t, (split, known)
            return
        sample = self.net_sample(split)
        if len(sample) < MIN_SAMPLES:
            sample = self.rewards[-WINDOW:]
        self.thr = threshold_table(sample, t, self.T, self.L)
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

    # ---- entry point ------------------------------------------------------------------------
    def play(self, reward, lockout, t, T, your_score, other_scores, is_locked):
        try:
            return bool(self.decide(reward, lockout, t, T, your_score, other_scores, is_locked))
        except Exception:  # noqa: BLE001 - never crash the game; a skipped turn is cheap
            return False

    def decide(self, reward, lockout, t, T, your_score, other_scores, is_locked):
        reward, lockout, t, T = float(reward), int(lockout), int(t), int(T)
        if t <= self.t_last or T != self.T or lockout != self.L or t != self.t_last + 1:
            self.reset(T, lockout)
        if not math.isfinite(reward):
            reward = 0.0
        other_scores = list(other_scores or [])
        self.observe(t, other_scores, your_score)
        self.predict_eligible(t, other_scores)
        self.prev_scores = [float(s) for s, _ in other_scores] or self.prev_scores
        self.prev_own = float(your_score)
        self.rewards.append(reward)
        self.takers.append(None)
        self.t_last = t
        self.took_last = False
        if is_locked or reward <= 0:
            return False
        split = self.sharing()
        known = not split and self.known_ok()
        if not known and len(self.rewards) < MIN_SAMPLES:
            return False
        if self.thr is None or t - self.thr_at >= RECOMPUTE or (split, known) != self.thr_mode:
            self.rebuild(t, split, known)
        value = reward * self.expected_share(reward, other_scores) if split else reward
        take = value > self.thr.get(t, 0.0)
        self.took_last = take
        return take


play = Solution().play
