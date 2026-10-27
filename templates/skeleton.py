"""Paste-ready strategy skeleton — infrastructure only, NO trading logic.

Write the logic in `decide()` at the event, after running the pipeline (AGENTS.md §5).
Standard library only; survives bare exec(); never raises out of `step()`.

    book: {symbol: price}  or  {symbol: {"bid":..,"ask":..,"bid_size":..,"ask_size":..}}
    positions: {symbol: current position}
    outstanding (optional): {symbol: (resting_buy_qty, resting_sell_qty)}
    account_pnl (optional): platform-reported PnL; preferred over our mark-to-mid estimate
    step(...) -> {symbol: order quantity (+buy / -sell)}; every order (including risk exits)
                 passes MAX_ORDER, LOTS and LIMITS assuming all resting orders fill
    step_targets(...) -> {symbol: target position}; same pipeline, for platforms that want
                 positions instead of orders (on error: current positions, i.e. hold)
"""

import math
import sys
from collections import deque

LIMITS = {}  # symbol -> max absolute position          (fill in from the briefing)
LOTS = {}  # symbol -> lot size; orders round toward zero to multiples of it
MAX_ORDER = {}  # symbol -> max absolute order per step  (optional)
SIZE_FRACTION = 1.0  # from evaluate_algo.py: GO 1.0, GO-REDUCED 0.5; applied by target_orders()
RISK = {
    "max_drawdown": None,  # loss from peak equity that triggers flatten + pause
    "halt_steps": 50,  # steps to stay out after a drawdown stop (flattening continues until flat)
    "max_gross": None,  # worst-case gross with resting/new fills; no added risk above the cap
    "flatten_after": None,  # step count (1 = first call) from which to flatten and stay flat
}
HISTORY = 500  # ticks of mid history kept per symbol
EWM_HALFLIFE = 50.0


class Ewm:
    def __init__(self, halflife):
        self.alpha = 1.0 - 0.5 ** (1.0 / halflife)
        self.n, self.mean, self.var = 0, 0.0, 0.0

    def update(self, x):
        if self.n == 0:
            self.mean = x
        else:
            d = x - self.mean
            self.mean += self.alpha * d
            self.var = (1.0 - self.alpha) * (self.var + self.alpha * d * d)
        self.n += 1

    def z(self, x):
        sd = math.sqrt(self.var)
        return (x - self.mean) / sd if sd > 0 else 0.0


class State:
    def __init__(self):
        self.t = 0
        self.errors = 0
        self.mids = {}  # symbol -> deque of mids
        self.ewm = {}  # symbol -> Ewm of mid
        self.books = {}  # symbol -> latest normalised book
        self.memory = {}  # free-form scratch space for decide()
        self.equity = 0.0  # platform PnL if given, else mark-to-mid estimate
        self.peak = 0.0
        self.last_positions = {}  # positions held over the last interval
        self.halted_until = -1
        self.flattening = False


STATE = State()


def normalise(entry):
    """Return dict with bid/ask/bid_size/ask_size/mid/micro from a price or a book dict."""
    if isinstance(entry, (int, float)):
        p = float(entry)
        return {"bid": p, "ask": p, "bid_size": 0.0, "ask_size": 0.0, "mid": p, "micro": p}
    bid, ask = float(entry["bid"]), float(entry["ask"])
    bs, az = float(entry.get("bid_size", 0) or 0), float(entry.get("ask_size", 0) or 0)
    mid = (bid + ask) / 2.0
    micro = (bid * az + ask * bs) / (bs + az) if bs + az > 0 else mid
    return {"bid": bid, "ask": ask, "bid_size": bs, "ask_size": az, "mid": mid, "micro": micro}


def update_state(book, positions, state, account_pnl=None):
    """Mark the positions held over the LAST interval; new fills earn nothing until next step."""
    for sym, entry in book.items():
        nb = normalise(entry)
        if not (math.isfinite(nb["mid"]) and nb["mid"] > 0):
            continue
        if sym in state.books:
            held = state.last_positions.get(sym, 0)
            state.equity += held * (nb["mid"] - state.books[sym]["mid"])
        state.books[sym] = nb
        state.mids.setdefault(sym, deque(maxlen=HISTORY)).append(nb["mid"])
        state.ewm.setdefault(sym, Ewm(EWM_HALFLIFE)).update(nb["mid"])
    state.last_positions = dict(positions)
    if account_pnl is not None:
        state.equity = float(account_pnl)
    state.t += 1
    state.peak = max(state.peak, state.equity)


def risk_gate(positions, state, outstanding=None):
    """Forced orders while a drawdown stop or end-of-round flatten is active ({} = stay out)."""
    end = RISK.get("flatten_after")
    if end is not None and state.t >= end:
        if any(positions.values()):
            return clip_orders({s: -q for s, q in positions.items() if q}, positions, outstanding)
        return {}
    limit = RISK.get("max_drawdown")
    if limit is not None and state.peak - state.equity > limit:
        state.halted_until = state.t + RISK.get("halt_steps", 0)
        state.flattening = True
        state.peak = state.equity
        print(f"risk: drawdown stop at t={state.t}, equity {state.equity:.2f}", file=sys.stderr)
    if state.flattening:
        if any(positions.values()):
            return clip_orders({s: -q for s, q in positions.items() if q}, positions, outstanding)
        state.flattening = False
    if state.t <= state.halted_until:
        return {}
    return None


def cap_gross(orders: dict, positions: dict, state: State,
              outstanding: "dict | None" = None) -> dict:
    """Reserve gross for any subset of resting/new fills; reductions cannot fund new risk."""
    cap = RISK.get("max_gross")
    if cap is None or not orders:
        return orders
    outstanding = outstanding or {}
    mids = {s: b["mid"] for s, b in state.books.items()}
    gross, added = 0.0, {}
    for sym in set(positions) | set(outstanding) | set(orders):
        pos, qty = positions.get(sym, 0), orders.get(sym, 0)
        buys, sells = outstanding.get(sym, (0, 0))
        if not all(math.isfinite(v) for v in (pos, buys, sells)) or min(buys, sells) < 0:
            raise ValueError("invalid position or outstanding quantity for " + sym)
        if not any((pos, buys, sells, qty)):
            continue
        if sym not in mids:
            raise ValueError("no price for gross exposure in " + sym)
        high, low = pos + buys, pos - sells
        before = max(abs(high), abs(low))
        after = max(abs(high + max(qty, 0)), abs(low + min(qty, 0)))
        gross += before * mids[sym]
        added[sym] = (after - before) * mids[sym]
    headroom, increase = max(cap - gross, 0.0), sum(added.values())
    if increase <= headroom:
        return orders
    scale = headroom / increase
    return {s: q if added.get(s, 0) == 0 else q * scale for s, q in orders.items()
            if q and (added.get(s, 0) == 0 or scale > 0)}


def target_orders(targets: dict, positions: dict) -> dict:
    """Orders that move to SIZE_FRACTION x target, capped at LIMITS. Use as decide()'s return."""
    out = {}
    for sym, target in targets.items():
        want = target * SIZE_FRACTION
        limit = LIMITS.get(sym)
        if limit is not None:
            want = max(-limit, min(limit, want))
        qty = want - positions.get(sym, 0)
        if qty:
            out[sym] = qty
    return out


def decide(book, positions, state):
    """EDIT AT THE EVENT. Return {symbol: order quantity}; empty dict = do nothing.

    Prefer `return target_orders({symbol: target_position}, positions)` so SIZE_FRACTION and
    LIMITS apply; orders returned directly are not scaled by SIZE_FRACTION.
    """
    return {}


def clip_orders(orders: dict, positions: dict, outstanding: "dict | None" = None) -> dict:
    """Worst case: all resting orders fill. Apply MAX_ORDER, LOTS (toward zero), then LIMITS."""
    outstanding = outstanding or {}
    out = {}
    for sym, qty in orders.items():
        if not math.isfinite(qty):
            raise ValueError("non-finite order quantity for " + sym)
        pos = positions.get(sym, 0)
        cap = MAX_ORDER.get(sym)
        if cap is not None:
            qty = max(-cap, min(cap, qty))
        lot = LOTS.get(sym)
        if lot:
            qty = int(qty / lot) * lot
        limit = LIMITS.get(sym)
        if limit is not None:
            buys, sells = outstanding.get(sym, (0, 0))
            max_buy, max_sell = limit - pos - buys, limit + pos - sells
            if lot:
                max_buy = math.floor(max_buy / lot + 1e-9) * lot
                max_sell = math.floor(max_sell / lot + 1e-9) * lot
            qty = min(qty, max(max_buy, 0)) if qty > 0 else max(qty, -max(max_sell, 0))
        if qty:
            out[sym] = qty
    return out


def step(book: dict, positions: "dict | None" = None, outstanding: "dict | None" = None,
         account_pnl: "float | None" = None) -> dict:
    """Platform entry point. Never raises; on error returns no orders and counts it."""
    positions = positions or {}
    try:
        update_state(book, positions, STATE, account_pnl)
        forced = risk_gate(positions, STATE, outstanding)
        orders = forced if forced is not None else decide(book, positions, STATE) or {}
        orders = clip_orders(orders, positions, outstanding)
        return clip_orders(cap_gross(orders, positions, STATE, outstanding), positions, outstanding)
    except Exception as exc:  # noqa: BLE001 - a crash would cost the round
        STATE.errors += 1
        if STATE.errors <= 3:
            print(f"step error #{STATE.errors}: {exc!r}", file=sys.stderr)
        return {}


def step_targets(book: dict, positions: "dict | None" = None,
                 outstanding: "dict | None" = None, account_pnl: "float | None" = None) -> dict:
    """Target-position entry point: current positions plus step()'s guarded orders."""
    positions = positions or {}
    orders = step(book, positions, outstanding, account_pnl)
    return {s: positions.get(s, 0) + orders.get(s, 0) for s in set(positions) | set(orders)}


# =====================================================================================
# ADAPTER — wire ONE of these to the platform's interface (see PLAYBOOK.md §3)
# =====================================================================================
# A) Per-tick callback returning orders:
#     def on_data(book, positions): return step(book, positions)
#
# B) Per-tick callback returning target positions (errors hold current positions):
#     def on_data(book, positions): return step_targets(book, positions)
