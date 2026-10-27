#!/usr/bin/env python3
"""First thing to run on ANY data they give us: what behaviour is there to exploit?

  uv run python scripts/diagnose.py data/prices.csv                       # price series only
  uv run python scripts/diagnose.py data/book.csv                         # order book (auto-detect)
  uv run python scripts/diagnose.py data/book.csv --trades data/trades.csv  # + markouts

Book CSV: one row per (time[, symbol]) with best bid/ask (+ sizes). Column names are
auto-detected (bid, ask, bid_size/bid_qty/bid_volume, ...); override with --bid/--ask/...
Trades CSV: time[, symbol], price[, quantity][, buyer, seller]. Trades are matched to the latest
book row at or before their time; sign = above mid -> buy, below -> sell.
Markout = PnL per unit of that side after H book rows. Negative for a party = that party
loses = trade against it / quote to it. Positive = informed = follow it, don't fade it.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gqc import diagnostics as dg
from gqc import io

PATTERNS = {
    "time": r"^(time|timestamp|ts|date|datetime|t|step|tick)$",
    "symbol": r"^(symbol|ticker|asset|instrument|product|name)$",
    "bid": r"^(best_?)?bid(_?price)?(_?[01])?$",
    "ask": r"^(best_?)?(ask|offer)(_?price)?(_?[01])?$",
    "bid_size": r"^bid_?(size|qty|quantity|vol|volume|amount)(_?[01])?$",
    "ask_size": r"^(ask|offer)_?(size|qty|quantity|vol|volume|amount)(_?[01])?$",
    "price": r"^(price|px|trade_?price)$",
    "qty": r"^(qty|quantity|size|volume|amount)$",
    "buyer": r"^(buyer|buy_?trader|bidder)$",
    "seller": r"^(seller|sell_?trader|offerer)$",
}


def detect(header: list[str], overrides: dict[str, str | None]) -> dict[str, str]:
    found = {k: v for k, v in overrides.items() if v}
    for key, pattern in PATTERNS.items():
        if key not in found:
            match = next(
                (h for h in header if re.match(pattern, h.strip(), re.IGNORECASE)), None
            )
            if match:
                found[key] = match
    return found


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def by_symbol(rows: list[dict], cols: dict[str, str]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row[cols["symbol"]] if "symbol" in cols else "ALL"].append(row)
    return groups


def time_keys(values: list[str]) -> np.ndarray:
    """Numeric times sort numerically; otherwise fall back to strings (ISO sorts correctly)."""
    try:
        return np.array([float(v) for v in values])
    except ValueError:
        return np.array(values)


def column(rows: list[dict], name: str) -> np.ndarray:
    return np.array([float(r[name]) if r[name] not in ("", None) else np.nan for r in rows])


def book_section(groups: dict[str, list[dict]], cols: dict[str, str]) -> dict[str, dict]:
    books = {}
    for sym, rows in groups.items():
        if "time" in cols:  # trade matching uses searchsorted: book rows must be time-sorted
            keys = time_keys([r[cols["time"]] for r in rows])
            rows = [rows[i] for i in np.argsort(keys, kind="stable")]
        bid, ask = column(rows, cols["bid"]), column(rows, cols["ask"])
        sizes = [column(rows, cols[k]) if k in cols else None for k in ("bid_size", "ask_size")]
        for line in dg.book_report(sym, bid, ask, *sizes):
            print(line)
        times = time_keys([r[cols["time"]] for r in rows]) if "time" in cols else None
        books[sym] = {"mid": (bid + ask) / 2, "time": times}
    return books


def trade_section(path: Path, books: dict[str, dict], horizons: tuple[int, ...]) -> None:
    header, rows = read_rows(path)
    cols = detect(header, {})
    print(f"\n== Trades ({len(rows)} rows; columns {cols})")
    for sym, trows in by_symbol(rows, cols).items():
        book = books.get(sym) or (books.get("ALL") if len(books) == 1 else None)
        if book is None or book["time"] is None or "time" not in cols:
            print(f"  {sym}: no matching book/time column, skipped")
            continue
        price = column(trows, cols["price"])
        stamps = time_keys([r[cols["time"]] for r in trows])
        idx = np.searchsorted(book["time"], stamps, side="right") - 1
        ok = idx >= 0
        idx, price, trows = idx[ok], price[ok], [r for r, k in zip(trows, ok) if k]
        sign = np.sign(price - book["mid"][idx])
        if "qty" in cols:
            sizes, counts = np.unique(column(trows, cols["qty"]), return_counts=True)
            top = sorted(zip(counts, sizes), reverse=True)[:6]
            print(f"  {sym}: most common sizes " + ", ".join(f"{s:g}x{c}" for c, s in top))
        for h in horizons:
            groups, g_idx, g_px, g_sign = _party_rows(trows, cols, idx, price, sign)
            table = dg.markouts_by_group(g_idx, g_px, g_sign, book["mid"], h, groups)
            print(f"  {sym} markouts h={h}: " + "; ".join(
                f"{g} {v['mean']:+.4f} (t {v['t']:+.1f}, n {v['n']})" for g, v in table.items()))


def _party_rows(trows, cols, idx, price, sign):
    """One row per (trade, party side). Without party columns, group by aggressor side."""
    if "buyer" in cols and "seller" in cols:
        parties = [r[cols["buyer"]] for r in trows] + [r[cols["seller"]] for r in trows]
        signs = np.concatenate([np.ones(len(trows)), -np.ones(len(trows))])
        return np.array(parties), np.tile(idx, 2), np.tile(price, 2), signs
    labels = np.where(sign > 0, "aggressive_buys", np.where(sign < 0, "aggressive_sells", "at_mid"))
    return labels, idx, price, sign


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--trades", type=Path)
    for key in ("bid", "ask", "bid-size", "ask-size", "symbol-col", "time-col"):
        parser.add_argument(f"--{key}")
    parser.add_argument("--horizons", default="1,5,20")
    args = parser.parse_args()

    header, rows = read_rows(args.csv)
    cols = detect(header, {"bid": args.bid, "ask": args.ask, "bid_size": args.bid_size,
                           "ask_size": args.ask_size, "symbol": args.symbol_col,
                           "time": args.time_col})
    print(f"{args.csv}: {len(rows)} rows; detected columns {cols}\n")
    if "bid" in cols and "ask" in cols:
        groups = by_symbol(rows, cols)
        books = book_section(groups, cols)
        print()
        for line in dg.price_report({s: b["mid"] for s, b in books.items()}):
            print(line)
        if args.trades:
            trade_section(args.trades, books, tuple(int(h) for h in args.horizons.split(",")))
    else:
        _, prices = io.load_prices(args.csv)
        for line in dg.price_report(prices):
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
