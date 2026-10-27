"""Round 3 of the G-Research trading game: competing for rewards.

Rules (briefing): rewards are exponential with mean 10, T = 2000 turns. If exactly one team
accepts on a turn it receives the reward; if two or more accept, all of them score zero.
Every accepting team is locked out for 15 turns. No bots; the latest score counts.

Strategy: a reward is worth r * P(no other team accepts this turn). Other teams reveal every
acceptance through their lock flag in other_scores (collision or not, they get locked), so we
learn per team (a) when it is eligible and (b) how likely it accepts a reward of each size.
We then solve the optimal-stopping problem on the empirical distribution of what we would
have collected alone (r if nobody else accepted, else 0): take iff
r * P(alone) > V[t + 1] - V[t + L + 1]. In practice: take mid-size rewards the other teams
skip, and big rewards while the teams that would take them are locked out.

Symmetry breaking: two teams running the same logic would collide every time. We see our own
collisions (we accepted but our score did not rise); when they pile up we accept good
opportunities only with probability 0.5 (0.3 if it persists), using an unseeded RNG so copies
decorrelate, until collisions become rare again.

Standard library only; play() never raises.
"""

import math
import random
from bisect import bisect_right

MEAN = 10.0  # announced exponential mean of the rewards
WINDOW = 800  # recent turns defining the alone-value distribution
MIN_NET = 40  # observed turn outcomes before trusting the empirical distribution
RECOMPUTE = 20  # turns between value-table rebuilds
EDGES = [2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 25.0, 30.0, 40.0]  # reward bins
PRIOR_W = 3.0  # pseudo-counts pulling a team's rate toward the pooled rate
TOL = 1e-6
OUTCOMES = 12  # our recent accepts used to measure our own collision rate
RNG = random.Random()  # deliberately unseeded: identical strategies must not stay in sync


def prior_rate(b):
    """Before any data, assume teams play the round-1 rule (take rewards above ~20)."""
    return 0.85 if b >= EDGES.index(20.0) + 1 else 0.05


def exp_value_table(mean, t0, T, L):
    value = [0.0] * (T + L + 3)
    for s in range(T, t0 - 1, -1):
        cut = value[s + 1] - value[s + L + 1]
        value[s] = value[s + 1] + mean * math.exp(-cut / mean)
    return value


def value_table(sample, t0, T, L):
    """V[s] for i.i.d. values `sample` with a fixed lockout L."""
    xs = sorted(sample)
    n = len(xs)
    suf = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suf[i] = suf[i + 1] + xs[i]
    value = [0.0] * (T + L + 3)
    for s in range(T, t0 - 1, -1):
        cut = value[s + 1] - value[s + L + 1]
        i = bisect_right(xs, cut)
        value[s] = value[s + 1] + (suf[i] - cut * (n - i)) / n
    return value


class Solution:
    def __init__(self):
        self.reset(0, 15)

    def reset(self, T, lockout):
        self.T, self.L, self.t_last = T, max(int(lockout), 0), 0
        self.rewards = []  # reward of turn a at index a - 1
        self.takers = {}  # turn a -> number of other teams that accepted on it
        self.V, self.v_at, self.v_kind = None, -(10**9), None
        self.prev_scores, self.prev_flags = None, None
        self.pre_votes = self.post_votes = 0  # flag semantics, from solo score jumps
        self.checks = self.bad = 0  # team-order consistency
        self.new_locks = {}  # flag round -> number of teams that became locked
        self.pool_acc = [PRIOR_W * prior_rate(b) for b in range(len(EDGES) + 1)]
        self.pool_n = [PRIOR_W] * (len(EDGES) + 1)
        self.n_teams = -1
        self.took_last, self.prev_own = False, 0.0
        self.outcomes = []  # True = our accept collided
        self.mixing = False

    def init_teams(self, n):
        nb = len(EDGES) + 1
        self.n_teams = n
        self.acc = [[0.0] * nb for _ in range(n)]
        self.cnt = [[0.0] * nb for _ in range(n)]
        self.first_true = [-(10**9)] * n
        self.prev_scores, self.prev_flags = None, None

    # ---- what other teams did ------------------------------------------------------------------
    def pre(self):
        return self.pre_votes >= self.post_votes

    def stable(self):
        return self.checks < 10 or self.bad <= 0.1 * self.checks

    def rate(self, i, b):
        pool = self.pool_acc[b] / self.pool_n[b]
        return (self.acc[i][b] + PRIOR_W * pool) / (self.cnt[i][b] + PRIOR_W)

    def observe(self, t, other_scores):
        n = len(other_scores)
        if n == 0:
            return
        if n != self.n_teams:
            self.init_teams(n)
        scores = [float(s) for s, _ in other_scores]
        flags = [bool(f) for _, f in other_scores]
        if self.prev_flags is None:
            self.prev_scores, self.prev_flags = scores, flags
            if t == 2:  # round 1 flags: anyone locked now accepted on turn 1
                self.record(t, [False] * n, flags)
            return
        self.calibrate(t, scores, flags)
        self.record(t, self.prev_flags, flags)
        self.prev_scores, self.prev_flags = scores, flags

    def calibrate(self, t, scores, flags):
        r = self.rewards[-1] if self.rewards else 0.0
        for i, (cur, old) in enumerate(zip(scores, self.prev_scores)):
            jump = cur - old
            if abs(jump) <= TOL * max(1.0, abs(cur)):
                continue
            if abs(jump - r) <= 1e-6 * max(1.0, abs(r)):
                if flags[i]:
                    self.post_votes += 1
                else:
                    self.pre_votes += 1
            else:
                self.bad += 1
        self.checks += 1

    def record(self, t, before, after):
        """Flags of round t-2 (`before`) and t-1 (`after`): who accepted, on which turn."""
        n = len(after)
        flips = [i for i in range(n) if after[i] and not before[i]]
        ends = sum(1 for i in range(n) if before[i] and not after[i])
        expected_ends = self.new_locks.get(t - 1 - self.L, 0)
        if ends != expected_ends and self.new_locks.get(t - 1 - self.L) is not None:
            self.bad += 1
        stable = self.stable()
        k = len(flips) if stable else max(0, sum(after) - sum(before) + expected_ends)
        self.new_locks[t - 1] = k
        for i in flips:
            self.first_true[i] = t - 1
        turn = t - 1 - (1 if self.pre() else 0)
        if turn < 1 or turn > len(self.rewards):
            return
        self.takers[turn] = k
        r = self.rewards[turn - 1]
        b = bisect_right(EDGES, r)
        flipped = set(flips)
        for i in range(n):
            if before[i]:
                continue
            took = i in flipped
            if stable:
                self.acc[i][b] += took
                self.cnt[i][b] += 1
            self.pool_acc[b] += took
            self.pool_n[b] += 1

    # ---- chance that nobody else takes this reward ----------------------------------------------
    def p_alone(self, t, reward):
        if self.prev_flags is None:  # turn 1: nothing known about the others yet
            return 0.0 if t == 1 else 1.0
        b = bisect_right(EDGES, reward)
        b_prev = bisect_right(EDGES, self.rewards[-2]) if len(self.rewards) >= 2 else b
        pre, stable = self.pre(), self.stable()
        alone = 1.0
        for i, locked in enumerate(self.prev_flags):
            p_now = self.rate(i, b) if stable else self.pool_acc[b] / self.pool_n[b]
            if not locked:
                p_prev = self.rate(i, b_prev) if stable else self.pool_acc[b_prev] / self.pool_n[b_prev]
                eligible = 1.0 - p_prev if pre else 1.0  # may have accepted last turn unseen
            elif pre and stable and t >= self.first_true[i] + self.L:
                eligible = 1.0  # its lockout ended after last turn
            else:
                eligible = 0.0
            alone *= 1.0 - eligible * p_now
        if pre and not stable:  # teams whose lockouts end now, by count
            ending = self.new_locks.get(t - self.L, 0)
            alone *= (1.0 - self.pool_acc[b] / self.pool_n[b]) ** ending
        return alone

    # ---- our decision -------------------------------------------------------------------------
    def net_sample(self):
        lo = max(1, len(self.rewards) - WINDOW)
        return [self.rewards[a - 1] if k == 0 else 0.0
                for a, k in self.takers.items() if a >= lo]

    def threshold(self, t):
        sample = self.net_sample()
        kind = "empirical" if len(sample) >= MIN_NET else "prior"
        if self.V is None or t - self.v_at >= RECOMPUTE or kind != self.v_kind:
            if kind == "empirical":
                self.V = value_table(sample, t, self.T, self.L)
            else:
                self.V = exp_value_table(MEAN, t, self.T, self.L)
            self.v_at, self.v_kind = t, kind
        hi = len(self.V)
        return (self.V[t + 1] if t + 1 < hi else 0.0) - (self.V[t + self.L + 1] if t + self.L + 1 < hi else 0.0)

    def decide(self, reward, lockout, t, T, your_score, other_scores, is_locked):
        reward, t, T = float(reward), int(t), int(T)
        if t <= self.t_last or T != self.T:
            self.reset(T, lockout)
        self.L = max(int(lockout), 0)
        if not math.isfinite(reward):
            reward = 0.0
        self.own_outcome(float(your_score))
        self.observe(t, list(other_scores or []))
        self.rewards.append(reward)
        self.t_last = t
        self.took_last = False
        if is_locked or reward <= 0:
            return False
        if reward * self.p_alone(t, reward) <= self.threshold(t):
            return False
        take = not self.mixing or RNG.random() < self.accept_prob()
        self.took_last = take
        return take

    def own_outcome(self, score):
        """Did our accept last turn pay? If not, someone accepted the same turn."""
        if self.took_last and self.rewards:
            r = self.rewards[-1]
            self.outcomes.append(abs(score - self.prev_own - r) > 1e-6 * max(1.0, abs(r)))
            del self.outcomes[:-OUTCOMES]
            n = len(self.outcomes)
            rate = sum(self.outcomes) / n
            if n >= 3 and rate >= 0.25:
                self.mixing = True
            elif n >= 6 and rate < 0.1:
                self.mixing = False
        self.prev_own = score

    def accept_prob(self):
        rate = sum(self.outcomes) / max(len(self.outcomes), 1)
        return 0.5 if rate < 0.6 else 0.3

    def play(self, reward, lockout, t, T, your_score, other_scores, is_locked):
        try:
            return bool(self.decide(reward, lockout, t, T, your_score, other_scores, is_locked))
        except Exception:  # noqa: BLE001 - never crash the game; a skipped turn is cheap
            return False


play = Solution().play
