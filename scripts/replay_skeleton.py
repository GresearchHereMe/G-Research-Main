#!/usr/bin/env python3
"""Replay the file you are about to paste over the data; check it matches the research PnL.

  uv run python scripts/replay_skeleton.py submissions/round1-v1.py data/round1.csv \
      [--cost-bps 1] [--lag 1]

Loads the file with bare exec() (like a paste box), feeds {symbol: price} each step with the
current positions, fills orders at the price `lag` steps later, and reports PnL, trades, max
position per symbol and error count. Compare with evaluate_algo.py before pasting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gqc import io, overfit, replay


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("submission", type=Path)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--cost-bps", type=float, default=1.0)
    parser.add_argument("--lag", type=int, default=1)
    args = parser.parse_args()
    ns = replay.load_submission(args.submission)
    _, prices = io.load_prices(args.csv)
    res = replay.replay(ns["step"], prices, args.lag, args.cost_bps)
    equity = np.cumsum(res.pnl)
    max_dd = float(np.max(np.maximum.accumulate(equity) - equity))
    state = ns.get("STATE") or ns.get("BOOK")
    print(f"PnL {res.pnl.sum():.2f}  Sharpe/step {overfit.sharpe(res.pnl):.4f}  max DD {max_dd:.2f}")
    print(f"trades {res.trades}  max |position| {res.max_abs_position}")
    print(f"final positions {res.final_positions}  step errors {getattr(state, 'errors', 'n/a')}")
    if res.trades == 0:
        print("WARNING: zero trades. Check decide()/signals, warm-up, and that target sizes are "
              ">= one lot (e.g. pairs unit_notional large enough for integer shares).")
    limits = ns.get("LIMITS") or {}
    breaches = {s: q for s, q in res.max_abs_position.items() if s in limits and q > limits[s]}
    if breaches:
        print(f"ERROR: position limits exceeded {breaches}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
