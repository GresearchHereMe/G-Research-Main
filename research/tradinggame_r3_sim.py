#!/usr/bin/env python3
"""Round-3 simulator: collisions pay nobody (our model of the briefing slide).

Rewards ~ exponential(mean 10), T = 2000, lockout 15. If exactly one team accepts it scores
the reward; two or more accepting teams all score zero. Every accepting team is locked out.
Fields of 23 opponents are guesses at what other teams submit.

  uv run python research/tradinggame_r3_sim.py submissions/tradinggame_r3_v5.py [--games 6]
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

T, L = 2000, 15


class Fixed:
    def __init__(self, c):
        self.c = c

    def play(self, reward, lockout, t, TT, score, others, locked):
        return reward > self.c


def load(path: Path):
    ns: dict = {}
    exec(compile(path.read_text(), str(path), "exec"), ns)  # noqa: S102 - fresh state per game
    return ns["play"]


def fields(rng: random.Random, smart: Path):
    return {
        "all round-1 rule (c=20)": [Fixed(20.1).play for _ in range(23)],
        "random thresholds 8-30": [Fixed(rng.uniform(8, 30)).play for _ in range(23)],
        "mixed": [Fixed(20.1).play for _ in range(10)] + [Fixed(rng.uniform(8, 30)).play for _ in range(6)]
        + [Fixed(2.0).play for _ in range(4)] + [load(smart) for _ in range(3)],
        "few teams (5x c=20)": [Fixed(20.1).play for _ in range(5)],
        "all smart (23 copies)": [load(smart) for _ in range(23)],
    }


def run_game(players, seed, post_lock):
    rng = random.Random(seed)
    n = len(players)
    scores, until, prev_locked = [0.0] * n, [0] * n, [False] * n
    for t in range(1, T + 1):
        reward = rng.expovariate(0.1)
        snap = [(scores[i], prev_locked[i]) for i in range(n)]
        takers = []
        for i, play in enumerate(players):
            locked = t <= until[i]
            others = [] if t == 1 else snap[:i] + snap[i + 1:]
            if play(reward, L, t, T, scores[i], others, locked) and not locked:
                takers.append(i)
        if len(takers) == 1:
            scores[takers[0]] += reward
        for i in takers:
            until[i] = t + L
        prev_locked = [(t + 1 if post_lock else t) <= until[i] for i in range(n)]
    return scores


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", type=Path, nargs="+")
    parser.add_argument("--games", type=int, default=6)
    parser.add_argument("--smart", type=Path, default=Path("submissions/tradinggame_r3_v5.py"))
    args = parser.parse_args()
    names = [c.stem.replace("tradinggame_", "") for c in args.candidates] + ["fixed c=20", "fixed c=12"]
    for field_name in fields(random.Random(0), args.smart):
        totals = {n: [0.0, 0.0] for n in names}  # mean score, mean rank
        for g in range(args.games):
            for name in names:
                rng = random.Random(100 + g)
                opp = fields(rng, args.smart)[field_name]
                if name == "fixed c=20":
                    me = Fixed(20.1).play
                elif name == "fixed c=12":
                    me = Fixed(12.0).play
                else:
                    me = load(args.candidates[names.index(name)])
                scores = run_game([me, *opp], seed=g, post_lock=g % 2 == 1)
                rank = 1 + sum(s > scores[0] for s in scores[1:])
                totals[name][0] += scores[0] / args.games
                totals[name][1] += rank / args.games
        print(f"{field_name:26}" + "  ".join(f"{n}: {s:6.0f} (rank {r:4.1f})" for n, (s, r) in totals.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
