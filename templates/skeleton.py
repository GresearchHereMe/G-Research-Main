"""Paste-ready strategy skeleton — infrastructure only, NO trading logic.

Write the logic in `decide()` at the event, after running scripts/diagnose.py on their data.
Standard library only; survives bare exec(); never raises out of `step()`.

    book: {symbol: price}  or  {symbol: {"bid":..,"ask":..,"bid_size":..,"ask_size":..}}
    positions: {symbol: current position}
    step(book, positions) -> {symbol: order quantity (+buy / -sell)}, clipped to limits
"""

import math
import sys
from collections import deque

LIMITS = {}  # symbol -> max absolute position          (fill in from the briefing)
MAX_ORDER = {}  # symbol -> max absolute order per step  (optional)
SIZE_FRACTION = 1.0  # from evaluate_algo.py: GO 1.0, GO-REDUCED 0.5 (scale your targets by it)
RISK = {
    "max_drawdown": None,  # mark-to-mid loss from peak that triggers flatten + pause
    "halt_steps": 50,  # steps to stay flat after a drawdown stop
    "max_gross": None,  # cap on sum(|position| * mid) across symbols
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
        self.equity = 0.0  # mark-to-mid PnL estimate
        self.peak = 0.0
        self.halted_until = -1


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


def update_state(book, positions, state):
    for sym, entry in book.items():
        nb = normalise(entry)
        if not (math.isfinite(nb["mid"]) and nb["mid"] > 0):
            continue
        if sym in state.books:  # positions held since the last step earn the mid change
            state.equity += positions.get(sym, 0) * (nb["mid"] - state.books[sym]["mid"])
        state.books[sym] = nb
        state.mids.setdefault(sym, deque(maxlen=HISTORY)).append(nb["mid"])
        state.ewm.setdefault(sym, Ewm(EWM_HALFLIFE)).update(nb["mid"])
    state.t += 1
    state.peak = max(state.peak, state.equity)


def risk_gate(positions, state):
    """Return flatten orders while a drawdown stop is active, else None."""
    limit = RISK.get("max_drawdown")
    if limit is not None and state.peak - state.equity > limit:
        state.halted_until = state.t + RISK.get("halt_steps", 0)
        state.peak = state.equity
        print(f"risk: drawdown stop at t={state.t}, equity {state.equity:.2f}", file=sys.stderr)
    if state.t <= state.halted_until or state.halted_until == state.t:
        return {s: -q for s, q in positions.items() if q}
    return None


def cap_gross(orders, positions, state):
    """Scale all orders down so gross exposure after fills stays within RISK['max_gross']."""
    cap = RISK.get("max_gross")
    if cap is None or not orders:
        return orders
    mids = {s: b["mid"] for s, b in state.books.items()}
    after = sum(abs(positions.get(s, 0) + orders.get(s, 0)) * mids.get(s, 0.0)
                for s in set(positions) | set(orders))
    if after <= cap:
        return orders
    now = sum(abs(q) * mids.get(s, 0.0) for s, q in positions.items())
    scale = max(0.0, (cap - now) / (after - now)) if after > now else 0.0
    return {s: q * scale for s, q in orders.items() if q * scale}


def decide(book, positions, state):
    """EDIT AT THE EVENT. Return {symbol: desired order quantity}. Empty dict = do nothing."""
    return {}


def clip_orders(orders, positions):
    """Assume every order fills: keep resulting positions within LIMITS and orders within MAX_ORDER."""
    out = {}
    for sym, qty in orders.items():
        pos = positions.get(sym, 0)
        cap = MAX_ORDER.get(sym)
        if cap is not None:
            qty = max(-cap, min(cap, qty))
        limit = LIMITS.get(sym)
        if limit is not None:
            qty = max(-limit - pos, min(limit - pos, qty))
        if qty:
            out[sym] = qty
    return out


def step(book, positions=None):
    """Platform entry point. Never raises; on error returns no orders and counts it."""
    positions = positions or {}
    try:
        update_state(book, positions, STATE)
        flatten = risk_gate(positions, STATE)
        if flatten is not None:
            return flatten
        orders = clip_orders(decide(book, positions, STATE) or {}, positions)
        return cap_gross(orders, positions, STATE)
    except Exception as exc:  # noqa: BLE001 - a crash would cost the round
        STATE.errors += 1
        if STATE.errors <= 3:
            print(f"step error #{STATE.errors}: {exc!r}", file=sys.stderr)
        return {}
