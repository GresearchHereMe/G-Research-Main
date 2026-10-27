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


def update_state(book, state):
    for sym, entry in book.items():
        nb = normalise(entry)
        if not (math.isfinite(nb["mid"]) and nb["mid"] > 0):
            continue
        state.books[sym] = nb
        state.mids.setdefault(sym, deque(maxlen=HISTORY)).append(nb["mid"])
        state.ewm.setdefault(sym, Ewm(EWM_HALFLIFE)).update(nb["mid"])
    state.t += 1


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
        update_state(book, STATE)
        return clip_orders(decide(book, positions, STATE) or {}, positions)
    except Exception as exc:  # noqa: BLE001 - a crash would cost the round
        STATE.errors += 1
        if STATE.errors <= 3:
            print(f"step error #{STATE.errors}: {exc!r}", file=sys.stderr)
        return {}
