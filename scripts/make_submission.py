#!/usr/bin/env python3
"""Build a single self-contained submission file from src/gqc/core.py + a JSON config.

Config JSON:
  {"pairs": {"name": {"y": "SYM_Y", "x": "SYM_X", "params": {...Params fields...}}},
   "limits": {"SYM": max_abs_shares}}

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
BOOK = PairBook(PAIRS, LIMITS)


def on_tick(prices):
    """prices: {{symbol: price}} for this step -> {{symbol: target position in shares}}."""
    return BOOK.update(prices)


def to_orders(targets, positions, lot=1):
    """Order quantities (rounded to `lot`) that move current positions to the targets."""
    orders = {{}}
    for sym, target in targets.items():
        qty = round((target - positions.get(sym, 0)) / lot) * lot
        if qty:
            orders[sym] = qty
    return orders


# =====================================================================================
# ADAPTER — edit ONE of these to match the platform's interface (see PLAYBOOK.md §3)
# =====================================================================================
# A) Per-tick callback returning target positions:
#     def strategy(prices): return on_tick(prices)
#
# B) Per-tick callback returning orders, given current positions:
#     def strategy(prices, positions): return to_orders(on_tick(prices), positions)
#
# C) Whole series at once (offline signal): rows = list of {{symbol: price}} dicts in time order
#     def positions_for_series(rows): return [on_tick(row) for row in rows]
'''


def _params_literal(params: dict) -> str:
    return "Params(" + ", ".join(f"{k}={v!r}" for k, v in params.items()) + ")"


def render(config: dict) -> str:
    pairs = ",\n    ".join(
        f"{name!r}: PairSpec({spec['y']!r}, {spec['x']!r}, {_params_literal(spec.get('params', {}))})"
        for name, spec in config["pairs"].items()
    )
    limits = {k: float(v) for k, v in config.get("limits", {}).items()}
    source = CORE.read_text()
    return source + ADAPTER.format(pairs="{\n    " + pairs + ",\n}", limits=repr(limits))


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
