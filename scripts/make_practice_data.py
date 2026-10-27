#!/usr/bin/env python3
"""Write a synthetic practice CSV (wide layout) with known structure, for rehearsals.

Series: AAA/BBB cointegrated (fast), CCC/DDD cointegrated (slow, beta 1.5),
EEE/FFF cointegrated until 60% then broken, GGG and HHH independent random walks.

Usage: uv run python scripts/make_practice_data.py data/practice_synth.csv [--n 5000] [--seed 7]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gqc.synthetic import cointegrated_pair, random_walk_pair


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    n, s = args.n, args.seed
    a, b = cointegrated_pair(n, beta=1.0, theta=0.08, sigma_s=0.006, seed=s)
    c, d = cointegrated_pair(n, beta=1.5, alpha=-1.0, theta=0.01, sigma_s=0.004, seed=s + 1)
    e, f = cointegrated_pair(n, theta=0.05, sigma_s=0.005, seed=s + 2, break_at=int(0.6 * n))
    g, h = random_walk_pair(n, seed=s + 3)
    cols = {"AAA": a, "BBB": b, "CCC": c, "DDD": d, "EEE": e, "FFF": f, "GGG": g, "HHH": h}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", *cols])
        for t in range(n):
            writer.writerow([t, *(f"{cols[k][t]:.4f}" for k in cols)])
    print(f"wrote {args.out}: {n} rows x {len(cols)} series")
    return 0


if __name__ == "__main__":
    sys.exit(main())
