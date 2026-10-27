#!/usr/bin/env python3
"""Run an algo through the gate pipeline (G0 integrity .. G4 risk) and get GO / NO-GO.

  uv run python scripts/evaluate_algo.py research/idea1.py data/round1.csv \
      [--cost-bps 1] [--lag 1] [--frequencies 1,2,5,10] [--bands 0]

The research file must define `strategy(prices, **params) -> {sym: positions}` and `GRID`
(see templates/research_template.py). Output ends with:
  DECISION: GO | GO-REDUCED | NO-GO, size as a fraction of position limits, best config
(incl. rebalance interval `every` and no-trade `band`). Put size into SIZE_FRACTION in the
skeleton and the config into decide().
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gqc import io, pipeline


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("algo", type=Path)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--cost-bps", type=float, default=1.0)
    parser.add_argument("--lag", type=int, default=1)
    parser.add_argument("--frequencies", default="1,2,5,10")
    parser.add_argument("--bands", default="0")
    args = parser.parse_args()
    algo = load_module(args.algo)
    _, prices = io.load_prices(args.csv)
    ev = pipeline.evaluate(
        algo.strategy, prices, algo.GRID, cost_bps=args.cost_bps, lag=args.lag,
        frequencies=[int(k) for k in args.frequencies.split(",")],
        bands=[float(b) for b in args.bands.split(",")],
    )
    print(f"{args.algo.name} on {args.csv.name}: cost {args.cost_bps} bps, lag {args.lag}\n")
    for line in ev.lines:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
