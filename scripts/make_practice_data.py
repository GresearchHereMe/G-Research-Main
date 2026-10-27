#!/usr/bin/env python3
"""Write synthetic practice data with known, planted behaviour, for rehearsals.

Prices (wide CSV): AAA/BBB cointegrated (fast), CCC/DDD cointegrated (slow, beta 1.5),
EEE/FFF cointegrated until 60% then broken, GGG and HHH independent random walks.

Book mode (--book): writes <out> as an L1 order book and <out stem>_trades.csv with bots:
overreacting mid (mean reversion), imbalance that leads the next move, a losing noise bot,
an informed smart bot. Rehearse: can you find all four with diagnose.py?

Usage:
  uv run python scripts/make_practice_data.py data/practice_synth.csv [--n 5000] [--seed 7]
  uv run python scripts/make_practice_data.py data/practice_book.csv --book
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gqc.synthetic import book_with_bots, cointegrated_pair, random_walk_pair


def write_dicts(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_prices(path: Path, n: int, s: int) -> None:
    a, b = cointegrated_pair(n, beta=1.0, theta=0.08, sigma_s=0.006, seed=s)
    c, d = cointegrated_pair(n, beta=1.5, alpha=-1.0, theta=0.01, sigma_s=0.004, seed=s + 1)
    e, f = cointegrated_pair(n, theta=0.05, sigma_s=0.005, seed=s + 2, break_at=int(0.6 * n))
    g, h = random_walk_pair(n, seed=s + 3)
    cols = {"AAA": a, "BBB": b, "CCC": c, "DDD": d, "EEE": e, "FFF": f, "GGG": g, "HHH": h}
    rows = [{"t": t, **{k: f"{v[t]:.4f}" for k, v in cols.items()}} for t in range(n)]
    write_dicts(path, rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--book", action="store_true", help="write order book + trades instead")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.book:
        book, trades = book_with_bots(args.n, args.seed)
        trades_path = args.out.with_name(args.out.stem + "_trades.csv")
        write_dicts(args.out, book)
        write_dicts(trades_path, trades)
        print(f"wrote {args.out} ({len(book)} rows) and {trades_path} ({len(trades)} trades)")
    else:
        write_prices(args.out, args.n, args.seed)
        print(f"wrote {args.out}: {args.n} rows x 8 series")
    return 0


if __name__ == "__main__":
    sys.exit(main())
