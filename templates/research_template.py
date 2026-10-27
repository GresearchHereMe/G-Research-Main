#!/usr/bin/env python3
"""Live-research template: write a strategy for what diagnose.py found, then check overfitting.

Copy to research/<idea>.py, edit `strategy` and GRID, run:
  uv run python research/<idea>.py data/round1.csv [--cost-bps 1] [--lag 1]

Rules (RULES.md §5): every config you evaluate counts as a trial (N in the Deflated Sharpe);
keep GRID coarse (<= 3 knobs, 3-4 values each); obey the VERDICT line; select by Sharpe, then
size to the position limits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gqc import io, overfit, research

# ---- EDIT: parameter grid -------------------------------------------------------------------
GRID = {"halflife": [20, 50, 100], "entry": [1.0, 1.5, 2.0]}


# ---- EDIT: strategy -------------------------------------------------------------------------
def strategy(prices: dict[str, np.ndarray], halflife: float, entry: float) -> dict[str, np.ndarray]:
    """Return target positions per symbol; position[t] may only use prices[: t + 1].

    Example: fade deviations of each series from its EWM mean (mean reversion).
    """
    out = {}
    alpha = 1 - 0.5 ** (1 / halflife)
    for sym, p in prices.items():
        mean, var = np.zeros_like(p), np.zeros_like(p)
        mean[0] = p[0]
        for t in range(1, len(p)):
            d = p[t] - mean[t - 1]
            mean[t] = mean[t - 1] + alpha * d
            var[t] = (1 - alpha) * (var[t - 1] + alpha * d * d)
        z = np.divide(p - mean, np.sqrt(var), out=np.zeros_like(p), where=var > 0)
        out[sym] = np.where(z > entry, -1.0, np.where(z < -entry, 1.0, 0.0))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--cost-bps", type=float, default=1.0)
    parser.add_argument("--lag", type=int, default=1)
    args = parser.parse_args()
    _, prices = io.load_prices(args.csv)
    configs, matrix = research.grid_returns(strategy, prices, GRID, args.lag, args.cost_bps)
    srs = np.atleast_1d(overfit.sharpe(matrix, axis=0))
    for cfg, sr, col in sorted(zip(configs, srs, matrix.T), key=lambda x: -x[1])[:10]:
        print(f"{cfg}  Sharpe/step {sr:+.4f}  PnL {col.sum():+.2f}")
    print()
    for line in research.overfit_report(configs, matrix):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
