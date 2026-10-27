"""Paste-ready strategy skeleton — infrastructure only, NO trading logic.

Write the logic in `decide()` at the event, after running the pipeline (AGENTS.md §5).
Standard library only; survives bare exec(); never raises out of `step()`.

    book: {symbol: price}  or  {symbol: {"bid":..,"ask":..,"bid_size":..,"ask_size":..}}
    positions: {symbol: current position}
    outstanding (optional): {symbol: (resting_buy_qty, resting_sell_qty)}
    account_pnl (optional): platform-reported PnL; preferred over our mark-to-mid estimate
    step(...) -> {symbol: order quantity (+buy / -sell)}; every order (including risk exits)
                 passes MAX_ORDER, LOTS and LIMITS assuming all resting orders fill
"""

import math
import sys
from collections import deque

LIMITS = {}  # symbol -> max absolute position          (fill in from the briefing)
LOTS = {}  # symbol -> lot size; orders round toward zero to multiples of it
MAX_ORDER = {}  # symbol -> max absolute order per step  (optional)
SIZE_FRACTION = 1.0  # from evaluate_algo.py: GO 1.0, GO-REDUCED 0.5 (scale your targets by it)
RISK = {
    "max_drawdown": None,  # loss from peak equity that triggers flatten + pause
    "halt_steps": 50,  # steps to stay out after a drawdown stop (flattening continues until flat)
    "max_gross": None,  # cap on sum(|position| * mid); risk-reducing orders always allowed
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
    """Forced orders while a drawdown stop is active ({} = stay out), else None."""
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


def cap_gross(orders, positions, state):
    """Scale only exposure-INCREASING orders so gross after fills stays within max_gross."""
    cap = RISK.get("max_gross")
    if cap is None or not orders:
        return orders
    mids = {s: b["mid"] for s, b in state.books.items()}
    reducing = {s: q for s, q in orders.items()
                if abs(positions.get(s, 0) + q) <= abs(positions.get(s, 0))}
    increasing = {s: q for s, q in orders.items() if s not in reducing}
    after = dict(positions)
    for s, q in reducing.items():
        after[s] = after.get(s, 0) + q
    headroom = cap - sum(abs(q) * mids.get(s, 0.0) for s, q in after.items())
    added = sum((abs(after.get(s, 0) + q) - abs(after.get(s, 0))) * mids.get(s, 0.0)
                for s, q in increasing.items())
    if added <= headroom:
        return orders
    scale = max(headroom, 0.0) / added if added > 0 else 0.0
    scaled = {s: q * scale for s, q in increasing.items() if q * scale}
    return {**reducing, **scaled}


def decide(book, positions, state):
    """EDIT AT THE EVENT. Return {symbol: desired order quantity}. Empty dict = do nothing."""
    return {}


def clip_orders(orders, positions, outstanding=None):
    """Worst case: all resting orders fill. Apply MAX_ORDER, LOTS (toward zero), then LIMITS."""
    outstanding = outstanding or {}
    out = {}
    for sym, qty in orders.items():
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


def step(book, positions=None, outstanding=None, account_pnl=None):
    """Platform entry point. Never raises; on error returns no orders and counts it."""
    positions = positions or {}
    try:
        update_state(book, positions, STATE, account_pnl)
        forced = risk_gate(positions, STATE, outstanding)
        if forced is not None:
            return forced
        orders = clip_orders(decide(book, positions, STATE) or {}, positions, outstanding)
        return clip_orders(cap_gross(orders, positions, STATE), positions, outstanding)
    except Exception as exc:  # noqa: BLE001 - a crash would cost the round
        STATE.errors += 1
        if STATE.errors <= 3:
            print(f"step error #{STATE.errors}: {exc!r}", file=sys.stderr)
        return {}
