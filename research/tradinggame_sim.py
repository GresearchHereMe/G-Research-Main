#!/usr/bin/env python3
"""Local simulator for the take-or-skip trading game (our model of it, not the organiser's).

Each turn every team sees the same reward; an unlocked team that takes it scores it and is
locked for `lockout` turns. Two rule variants because the briefing did not say: `indep` (every
taker gets the full reward) and `split` (takers share it equally). Bots are our guesses of
the reference bots: greedy, random, novice, pro.

  uv run python research/tradinggame_sim.py submissions/tradinggame_v1.py [--games 20]
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path


def draw_factory(kind: str, rng: random.Random):
    if kind == "exp":
        return lambda t: rng.expovariate(0.1)
    if kind == "lognormal":
        return lambda t: rng.lognormvariate(2.0, 0.8)
    if kind == "uniform":
        return lambda t: rng.uniform(0, 30)
    if kind == "regime":  # mean switches between 5 and 20 every ~300 turns
        state = {"mean": 10.0}

        def draw(t):
            if rng.random() < 1 / 300:
                state["mean"] = rng.choice([5.0, 20.0])
            return rng.expovariate(1 / state["mean"])
        return draw
    if kind == "drift":  # mean rises from 5 to 20 over the game
        return lambda t: rng.expovariate(1 / (5 + 15 * t / 2000))
    raise ValueError(kind)


class Greedy:
    def play(self, reward, lockout, t, T, score, others, locked):
        return reward > 0


class RandomBot:
    def __init__(self, rng):
        self.rng = rng

    def play(self, reward, lockout, t, T, score, others, locked):
        return self.rng.random() < 0.5


class Novice:
    """Takes anything above the running mean reward."""

    def __init__(self):
        self.total, self.n = 0.0, 0

    def play(self, reward, lockout, t, T, score, others, locked):
        self.total, self.n = self.total + reward, self.n + 1
        return reward > self.total / self.n


class Pro:
    """Stationary optimal threshold c = L * g from the observed rewards (ignores sharing)."""

    def __init__(self):
        self.seen, self.c = [], None

    def play(self, reward, lockout, t, T, score, others, locked):
        self.seen.append(reward)
        if len(self.seen) % 25 == 1 or self.c is None:
            xs, c = self.seen, 0.0
            for _ in range(30):
                above = [x for x in xs if x > c]
                p = len(above) / len(xs)
                g = (sum(above) / len(xs)) / (1 + lockout * p) if above else 0.0
                c = lockout * g
            self.c = c
        return reward > self.c


class FixedThreshold:
    def __init__(self, c):
        self.c = c

    def play(self, reward, lockout, t, T, score, others, locked):
        return reward > self.c


def load_ours(path: Path):
    ns: dict = {}
    exec(compile(path.read_text(), str(path), "exec"), ns)  # noqa: S102 - fresh state per game
    return ns["play"]


def run_game(players, names, rule, kind, seed, T=2000, lockout=15, post_lock=False,
             shuffle=False):
    rng = random.Random(seed)
    draw = draw_factory(kind, rng)
    n = len(players)
    scores, locked_until = [0.0] * n, [0] * n
    prev_locked = [False] * n
    spent = [0.0] * n
    for t in range(1, T + 1):
        reward = draw(t)
        snapshot = [(scores[i], prev_locked[i]) for i in range(n)]
        takers = []
        for i, play in enumerate(players):
            others = [] if t == 1 else snapshot[:i] + snapshot[i + 1:]
            if shuffle:  # e.g. a list sorted by score: team identity not stable
                others = sorted(others, reverse=True)
            locked = t <= locked_until[i]
            tic = time.perf_counter()
            took = play(reward, lockout, t, T, scores[i], others, locked)
            spent[i] += time.perf_counter() - tic
            if took and not locked:
                takers.append(i)
        for i in takers:
            scores[i] += reward if rule == "indep" else reward / len(takers)
            locked_until[i] = t + lockout
        prev_locked = [(t + 1 if post_lock else t) <= locked_until[i] for i in range(n)]
    return dict(zip(names, scores)), dict(zip(names, spent))


def field(ours_path: Path, rng: random.Random, with_bots: bool, crowd: int):
    players, names = [load_ours(ours_path)], ["OURS"]
    if with_bots:
        players += [Greedy().play, RandomBot(rng).play, Novice().play, Pro().play]
        names += ["greedy", "random", "novice", "pro"]
    for j in range(crowd):
        players.append(FixedThreshold(rng.uniform(8, 30)).play)
        names.append(f"team{j}")
    return players, names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("solution", type=Path)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--crowd", type=int, default=19, help="other fixed-threshold teams")
    parser.add_argument("--dists", default="exp,lognormal,uniform,regime,drift")
    parser.add_argument("--rules", default="indep,split")
    parser.add_argument("--sorted-others", action="store_true",
                        help="pass other_scores sorted by score (unstable team order)")
    args = parser.parse_args()
    for rule in args.rules.split(","):
        for with_bots in (True, False):
            for kind in args.dists.split(","):
                totals: dict[str, list[float]] = {}
                worst_ms = 0.0
                for g in range(args.games):
                    rng = random.Random(1000 + g)
                    players, names = field(args.solution, rng, with_bots, args.crowd)
                    res, spent = run_game(players, names, rule, kind, seed=g, post_lock=g % 2 == 1,
                                           shuffle=args.sorted_others)
                    for k, v in res.items():
                        totals.setdefault(k, []).append(v)
                    worst_ms = max(worst_ms, spent["OURS"] / 2000 * 1e3)
                means = {k: sum(v) / len(v) for k, v in totals.items()}
                rank = sorted(means, key=means.get, reverse=True)
                best_other = max(v for k, v in means.items() if k != "OURS")
                sd = math.sqrt(sum((x - means["OURS"]) ** 2 for x in totals["OURS"]) / len(totals["OURS"]))
                bots = " ".join(f"{b}={means[b]:.0f}" for b in ("greedy", "random", "novice", "pro")
                                if b in means)
                print(f"{rule:5} bots={'Y' if with_bots else 'N'} {kind:9} OURS={means['OURS']:7.0f} "
                      f"(sd {sd:5.0f}) rank {rank.index('OURS') + 1}/{len(rank)} "
                      f"best_other={best_other:7.0f} {bots}  {worst_ms:.2f} ms/call")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
