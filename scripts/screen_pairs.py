#!/usr/bin/env python3
"""Rank all pairs in a price CSV by Engle-Granger cointegration strength.

Usage:
  uv run python scripts/screen_pairs.py data/prices.csv [--train-frac 0.6] [--top 20] [--no-log]

Screens on the TRAIN window only, then re-tests the same pair/direction on the test window so
you can see which relationships survive out of sample. `*` marks 5% significance.
Trust a pair only if: significant in BOTH windows, sensible beta, half-life short relative to
the data length (and to how long you can hold), and an economic reason to be related.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gqc import io, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--no-log", action="store_true", help="use price levels, not log prices")
    args = parser.parse_args()

    index, prices = io.load_prices(args.csv)
    split = int(len(index) * args.train_frac)
    use_log = not args.no_log
    print(f"{len(prices)} series x {len(index)} rows; train = first {split} rows\n")
    train = {k: v[:split] for k, v in prices.items()}
    rows = stats.screen(train, use_log=use_log)[: args.top]
    header = (f"{'y':>10} {'x':>10} {'beta':>7} {'t_train':>8} {'t_test':>8} {'hl':>7} "
              f"{'corr':>6}  verdict")
    print(header + "\n" + "-" * len(header))
    for r in rows:
        y, x = prices[r.y][split:], prices[r.x][split:]
        ok = np.isfinite(y) & np.isfinite(x) & (y > 0) & (x > 0)
        ty, tx = (np.log(y[ok]), np.log(x[ok])) if use_log else (y[ok], x[ok])
        test = stats.eg_coint(ty, tx, stats.schwert_lags(len(ty))) if ok.sum() > 30 else None
        t_test = f"{test.tstat:7.2f}{'*' if test.is_coint() else ' '}" if test else "    n/a "
        mark = "*" if r.tstat < r.crit5 else " "
        both = r.tstat < r.crit5 and test is not None and test.is_coint()
        verdict = "TRADE" if both and r.beta > 0 and r.ret_corr > 0.3 else "skip"
        print(f"{r.y:>10} {r.x:>10} {r.beta:7.3f} {r.tstat:7.2f}{mark} {t_test} "
              f"{r.half_life:7.1f} {r.ret_corr:6.2f}  {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
