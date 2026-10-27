#!/usr/bin/env python3
"""Grid-search one pair: formation -> development (selection) -> look-once final test.

Usage:
  uv run python scripts/sweep_pair.py data/prices.csv --y A --x B \
      [--formation-frac 0.2] [--train-frac 0.6] [--cost-bps 1] [--lag 1] [--notional 10000]

The static hedge is fitted on the formation window only. Ranking = plateau of DEVELOPMENT
Sharpe. The overfitting block uses development data only, scores the SELECTED config, and
counts every trial previously run on this dataset (runs/trials.jsonl). The final-test line is
printed once for the selected config; never go back and re-select after reading it.
A REJECT verdict suppresses the exported parameters.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gqc import core, io, registry, sweep


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--y", required=True)
    parser.add_argument("--x", required=True)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--cost-bps", type=float, default=1.0)
    parser.add_argument("--lag", type=int, default=1)
    parser.add_argument("--notional", type=float, default=10_000.0)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--formation-frac", type=float, default=0.2)
    parser.add_argument("--runs-dir", type=Path, default=REPO / "runs")
    args = parser.parse_args()

    _, prices = io.load_prices(args.csv)
    py, px = prices[args.y], prices[args.x]
    ok = np.isfinite(py) & np.isfinite(px) & (py > 0) & (px > 0)
    if not ok.all():
        print(f"WARNING: dropping {int((~ok).sum())} rows with missing prices (time is compressed)")
    py, px = py[ok], px[ok]
    base = core.Params(use_log=not args.no_log, unit_notional=args.notional, warmup=args.warmup)
    rows = sweep.sweep_pair(py, px, args.train_frac, args.cost_bps, args.lag, base,
                            formation_frac=args.formation_frac)
    r0 = rows[0]
    print(f"{args.y}/{args.x}: {len(py)} rows; formation [0,{r0.formation}) dev "
          f"[{r0.formation},{r0.split}) final test [{r0.split},{len(py)}); "
          f"cost {args.cost_bps} bps, lag {args.lag}\n")
    head = (f"{'hedge':>7} {'entry':>5} {'hl':>5} {'exit':>4} {'plat_SR':>8} {'SR_dev':>7} "
            f"{'pnl_dev':>10} {'trips':>5}")
    print(head + "\n" + "-" * len(head))
    for r in rows[: args.top]:
        p = r.params
        print(f"{p.hedge:>7} {p.entry_z:5.1f} {p.halflife:5.0f} {p.exit_z:4.1f} "
              f"{r.plateau:8.2f} {r.train_sharpe:7.2f} {r.train_pnl:10.1f} {r.trips:5d}")
    print("(dev window only; SR = per-step Sharpe x sqrt(steps); plat_SR = median of neighbours)\n")
    fp = registry.fingerprint(args.csv)
    prior = registry.total_trials(args.runs_dir, fp)
    lines = sweep.evidence_report(rows, prior_trials=prior)
    for line in lines:
        print(line)
    registry.record(args.runs_dir, fp, f"sweep {args.y}/{args.x}", len(rows), lines[-1])
    print(sweep.holdout_line(rows))
    if "REJECT" in lines[-1]:
        print("\nREJECTED: do not export these parameters (RULES.md §5.9).")
        return 0
    print("\nSelected (top-plateau) params for the submission config:")
    print(json.dumps(r0.params._asdict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
