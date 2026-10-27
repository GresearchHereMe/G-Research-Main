"""Load price CSVs in wide (time, A, B, ...) or long (time, symbol, price) layout."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

LONG_SYMBOL_COLS = ("symbol", "ticker", "asset", "instrument", "name")
LONG_PRICE_COLS = ("price", "close", "mid", "value", "px")


def _num(text: str) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def _is_numeric_column(rows: list[list[str]], col: int) -> bool:
    sample = [r[col] for r in rows[:50] if col < len(r) and r[col].strip()]
    return bool(sample) and all(not math.isnan(_num(v)) for v in sample)


def _find(header: list[str], names: tuple[str, ...]) -> int | None:
    lowered = [h.strip().lower() for h in header]
    return next((lowered.index(n) for n in names if n in lowered), None)


def _load_long(rows, t_col, s_col, p_col) -> tuple[list[str], dict[str, np.ndarray]]:
    stamps: dict[str, None] = {}  # insertion-ordered set of timestamps
    table: dict[str, dict[str, float]] = {}
    for i, row in enumerate(rows):
        stamp = row[t_col] if t_col is not None else str(i)
        stamps.setdefault(stamp, None)
        table.setdefault(row[s_col], {})[stamp] = _num(row[p_col])
    index = list(stamps)
    prices = {sym: np.array([vals.get(t, math.nan) for t in index]) for sym, vals in table.items()}
    return index, prices


def load_prices(path: str | Path) -> tuple[list[str], dict[str, np.ndarray]]:
    """Return (time index as strings, {symbol: float array}); missing values are NaN."""
    with open(path, newline="") as handle:
        header, *rows = list(csv.reader(handle))
    s_col, p_col = _find(header, LONG_SYMBOL_COLS), _find(header, LONG_PRICE_COLS)
    first_numeric = _is_numeric_column(rows, 0)
    if s_col is not None and p_col is not None:
        t_col = 0 if 0 not in (s_col, p_col) else None
        return _load_long(rows, t_col, s_col, p_col)
    start = 0 if first_numeric and header[0].strip().lower() not in ("time", "date", "t") else 1
    index = [r[0] for r in rows] if start else [str(i) for i in range(len(rows))]
    prices = {
        header[c].strip(): np.array([_num(r[c]) if c < len(r) else math.nan for r in rows])
        for c in range(start, len(header))
    }
    return index, prices
