#!/usr/bin/env python3
"""Grid-search one pair on the train window; rank by plateau score; show out-of-sample PnL.

Usage:
  uv run python scripts/sweep_pair.py data/prices.csv --y A --x B \
      [--train-frac 0.6] [--cost-bps 1] [--lag 1] [--notional 10000] [--no-log] [--top 15]

Ranking = plateau of TRAIN Sharpe (median over grid neighbours). Pick near the top whose test
result is also positive; never pick by test PnL — that turns the test window into training data.
The overfitting block (PBO via CSCV over ALL configs tried + Deflated Sharpe) says whether the
search itself can be trusted; obey its VERDICT line.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gqc import core, io, research, sweep


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
    args = parser.parse_args()

    _, prices = io.load_prices(args.csv)
    py, px = prices[args.y], prices[args.x]
    ok = np.isfinite(py) & np.isfinite(px) & (py > 0) & (px > 0)
    py, px = py[ok], px[ok]
    base = core.Params(use_log=not args.no_log, unit_notional=args.notional, warmup=args.warmup)
    rows = sweep.sweep_pair(py, px, args.train_frac, args.cost_bps, args.lag, base)
    print(f"{args.y}/{args.x}: {len(py)} rows, cost {args.cost_bps} bps, lag {args.lag}\n")
    head = (f"{'hedge':>7} {'entry':>5} {'hl':>5} {'exit':>4} {'plat_SR':>8} {'SR_tr':>6} "
            f"{'SR_te':>6} {'pnl_tr':>10} {'pnl_te':>10} {'trips':>5}")
    print(head + "\n" + "-" * len(head))
    for r in rows[: args.top]:
        p = r.params
        print(f"{p.hedge:>7} {p.entry_z:5.1f} {p.halflife:5.0f} {p.exit_z:4.1f} "
              f"{r.plateau:8.2f} {r.train_sharpe:6.2f} {r.test_sharpe:6.2f} "
              f"{r.train_pnl:10.1f} {r.test_pnl:10.1f} {r.trips:5d}")
    print("(SR = per-step Sharpe x sqrt(steps in window); plat_SR = median train SR of neighbours)\n")
    configs = [r.params._asdict() for r in rows]
    for line in research.overfit_report(configs, np.column_stack([r.pnl for r in rows])):
        print(line)
    best = rows[0].params
    print("\nTop-plateau params (paste into submission config):")
    print(json.dumps(best._asdict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
