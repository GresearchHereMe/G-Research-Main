#!/usr/bin/env python3
"""Build a single self-contained submission file from src/gqc/core.py + a JSON config.

Config JSON:
  {"pairs": {"name": {"y": "SYM_Y", "x": "SYM_X", "params": {...Params fields...}}},
   "limits": {"SYM": max_abs_shares},
   "size_fraction": 1.0,      # optional: GO 1.0, GO-REDUCED 0.5 (scales every target)
   "flatten_after": null}     # optional: tick count from which all targets are zero

Usage:
  python scripts/make_submission.py --config cfg.json --out submissions/round1-v1.py
The output uses only the standard library and is verified to import in an isolated interpreter.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

CORE = Path(__file__).resolve().parents[1] / "src" / "gqc" / "core.py"

ADAPTER = '''

# =====================================================================================
# GENERATED CONFIG — produced by scripts/make_submission.py
# =====================================================================================
PAIRS = {pairs}
LIMITS = {limits}
SIZE_FRACTION = {size_fraction}  # from evaluate_algo.py: GO 1.0, GO-REDUCED 0.5
FLATTEN_AFTER = {flatten_after}  # tick count (1 = first call) from which targets are flat
BOOK = PairBook(PAIRS, LIMITS)
CLOCK = {{"ticks": 0}}


def on_tick(prices, positions=None):
    """prices: {{symbol: price}} -> {{symbol: target position}} scaled by SIZE_FRACTION.

    On an internal error returns the CURRENT positions (hold: no new risk) when given, else None.
    Never feed stale targets or zero targets after a failure. From tick FLATTEN_AFTER on,
    every known symbol targets zero (end-of-round flatten).
    """
    CLOCK["ticks"] += 1
    if FLATTEN_AFTER is not None and CLOCK["ticks"] >= FLATTEN_AFTER:
        return {{sym: 0.0 for sym in set(positions or {{}}) | set(BOOK.prices)}}
    targets = BOOK.update(prices)
    if targets is None:
        return dict(positions) if positions is not None else None
    return {{sym: qty * SIZE_FRACTION for sym, qty in targets.items()}}


def to_orders(targets, positions, lot=1, outstanding=None):
    """Lot-rounded orders towards `targets` that can never breach LIMITS.

    Rounds toward zero (never overshoots the target), then caps assuming every resting order
    fills: outstanding = {{symbol: (resting_buy_qty, resting_sell_qty)}}. None targets -> {{}}.
    """
    if targets is None:
        return {{}}
    outstanding = outstanding or {{}}
    orders = {{}}
    for sym, target in targets.items():
        pos = positions.get(sym, 0)
        lots = int((target - pos) / lot)  # toward zero
        limit = LIMITS.get(sym)
        if limit is not None:
            buys, sells = outstanding.get(sym, (0, 0))
            max_buy = math.floor((limit - pos - buys) / lot + 1e-9)
            max_sell = math.floor((limit + pos - sells) / lot + 1e-9)
            lots = min(lots, max(max_buy, 0)) if lots > 0 else max(lots, -max(max_sell, 0))
        if lots:
            orders[sym] = lots * lot
    return orders


def step(prices, positions=None, outstanding=None, lot=1):
    """Orders-style entry point; also what scripts/replay_skeleton.py replays."""
    positions = positions or {{}}
    return to_orders(on_tick(prices, positions), positions, lot, outstanding)


# =====================================================================================
# ADAPTER — edit ONE of these to match the platform's interface (see PLAYBOOK.md §3)
# =====================================================================================
# A) Per-tick callback returning target positions (pass positions so errors hold them):
#     def strategy(prices, positions): return on_tick(prices, positions)
#
# B) Per-tick callback returning orders, given current positions (+ resting orders if known):
#     def strategy(prices, positions): return to_orders(on_tick(prices, positions), positions)
#
# C) Whole series at once (offline signal): rows = list of {{symbol: price}} dicts in time order
#     def positions_for_series(rows): return [on_tick(row) for row in rows]
'''


def _params_literal(params: dict) -> str:
    return "Params(" + ", ".join(f"{k}={v!r}" for k, v in params.items()) + ")"


def _sizing(config: dict) -> tuple[float, int | None]:
    """Validate the optional size_fraction (0, 1] and flatten_after (tick count >= 1)."""
    fraction = config.get("size_fraction", 1.0)
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) \
            or not 0 < fraction <= 1:
        raise SystemExit(f"size_fraction must be in (0, 1], got {fraction!r}")
    end = config.get("flatten_after")
    if end is not None and (isinstance(end, bool) or not isinstance(end, int) or end < 1):
        raise SystemExit(f"flatten_after must be a tick count >= 1 or null, got {end!r}")
    return float(fraction), end


def render(config: dict) -> str:
    fraction, end = _sizing(config)
    pairs = ",\n    ".join(
        f"{name!r}: PairSpec({spec['y']!r}, {spec['x']!r}, {_params_literal(spec.get('params', {}))})"
        for name, spec in config["pairs"].items()
    )
    limits = {k: float(v) for k, v in config.get("limits", {}).items()}
    source = CORE.read_text()
    return source + ADAPTER.format(pairs="{\n    " + pairs + ",\n}", limits=repr(limits),
                                   size_fraction=repr(fraction), flatten_after=repr(end))


def verify(path: Path) -> None:
    # Bare exec() into a fresh namespace mimics paste-box platforms most strictly.
    code = (
        "import sys\n"
        f"src = open({str(path)!r}).read()\n"
        "ns = {'__name__': 'submission'}\n"
        "exec(compile(src, 'submission', 'exec'), ns)\n"
        "ns['on_tick']({'__probe__': 1.0})\n"
        "bad = [k for k in sys.modules if k == 'numpy' or k.startswith('gqc')]\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-I", "-c", code], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(config))
    verify(args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes), verified stdlib-only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
