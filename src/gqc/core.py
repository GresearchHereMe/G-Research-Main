"""Stdlib-only pairs-trading core.

This file is copied verbatim into single-file submissions (scripts/make_submission.py), so it
must not import numpy or anything from the gqc package. Estimators are small mutable objects
on purpose: they run once per tick and must be O(1) in time and memory.

Spread convention (log mode): s = ln(y) - beta * ln(x) - alpha.
Position +1 = long spread = long y, short beta * x (notional-hedged in log mode).

NamedTuple (not dataclass) on purpose: paste-box platforms often exec() code without
registering a module, which breaks dataclasses. Works on Python 3.8+.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from typing import NamedTuple

HEDGES = ("kalman", "ewm", "static")


class Params(NamedTuple):
    use_log: bool = True
    hedge: str = "kalman"
    static_beta: float = 1.0
    static_alpha: float = 0.0
    warmup: int = 100
    entry_z: float = 2.0
    exit_z: float = 0.0
    stop_z: float = 4.0
    max_hold: int = 0  # ticks; 0 disables the time stop
    cooldown: int = 0  # ticks to stay flat after a stop
    halflife: float = 100.0  # EWM half-life (ticks) for spread statistics
    kalman_delta: float = 1e-6
    kalman_ve: float = 1e-3
    unit_notional: float = 1.0  # log mode: y-leg notional; price mode: y-leg shares


def validate(params: Params) -> Params:
    if params.hedge not in HEDGES:
        raise ValueError(f"hedge must be one of {HEDGES}, got {params.hedge!r}")
    if not 0 <= params.exit_z < params.entry_z < params.stop_z:
        raise ValueError("need 0 <= exit_z < entry_z < stop_z")
    return params


class Signal(NamedTuple):
    position: int  # -1 short spread, 0 flat, +1 long spread
    z: float
    beta: float
    alpha: float
    ready: bool


class Ewm:
    """Exponentially weighted mean and variance (incremental, O(1))."""

    def __init__(self, halflife: float) -> None:
        self.alpha = 1.0 - 0.5 ** (1.0 / max(halflife, 1e-9))
        self.n = 0
        self.mean = 0.0
        self.var = 0.0

    def update(self, value: float) -> None:
        if self.n == 0:
            self.mean, self.var = value, 0.0
        else:
            diff = value - self.mean
            self.mean += self.alpha * diff
            self.var = (1.0 - self.alpha) * (self.var + self.alpha * diff * diff)
        self.n += 1

    @property
    def std(self) -> float:
        return math.sqrt(max(self.var, 0.0))

    def zscore(self, value: float) -> float:
        sd = self.std
        return (value - self.mean) / sd if sd > 0 else 0.0


class KalmanHedge:
    """Dynamic hedge ratio: state [beta, alpha] follows a random walk (Chan, 2013)."""

    def __init__(self, delta: float, ve: float) -> None:
        self.vw = delta / (1.0 - delta)
        self.ve = ve
        self.beta, self.alpha = 0.0, 0.0
        self.p = [[1.0, 0.0], [0.0, 1.0]]

    def update(self, y: float, x: float) -> float:
        """Return the one-step-ahead forecast error (prior state), then update the state."""
        r00, r01 = self.p[0][0] + self.vw, self.p[0][1]
        r10, r11 = self.p[1][0], self.p[1][1] + self.vw
        err = y - (self.beta * x + self.alpha)
        rf0, rf1 = r00 * x + r01, r10 * x + r11  # R @ F
        q = x * rf0 + rf1 + self.ve
        k0, k1 = rf0 / q, rf1 / q
        self.beta += k0 * err
        self.alpha += k1 * err
        fr0, fr1 = x * r00 + r10, x * r01 + r11  # F @ R
        self.p = [[r00 - k0 * fr0, r01 - k0 * fr1], [r10 - k1 * fr0, r11 - k1 * fr1]]
        return err


class SpreadModel:
    """Turns (y, x) into a z-score using only past data, then learns from the new point."""

    def __init__(self, params: Params) -> None:
        self.params = params
        self.spread = Ewm(params.halflife)
        self.kalman = KalmanHedge(params.kalman_delta, params.kalman_ve)
        self.moments = {k: Ewm(params.halflife) for k in ("x", "y", "xx", "xy")}
        self.beta, self.alpha = params.static_beta, params.static_alpha

    def update(self, y: float, x: float) -> tuple[float, float, float]:
        hedge = self.params.hedge
        if hedge == "kalman":
            resid = self.kalman.update(y, x)
            self.beta, self.alpha = self.kalman.beta, self.kalman.alpha
        else:
            resid = y - self.beta * x - self.alpha
        z = self.spread.zscore(resid)
        self.spread.update(resid)
        if hedge == "ewm":
            self._update_ewm_hedge(y, x)
        return z, self.beta, self.alpha

    def _update_ewm_hedge(self, y: float, x: float) -> None:
        m = self.moments
        for key, value in (("x", x), ("y", y), ("xx", x * x), ("xy", x * y)):
            m[key].update(value)
        var_x = m["xx"].mean - m["x"].mean ** 2
        if var_x > 1e-12 and m["x"].n > 2:
            self.beta = (m["xy"].mean - m["x"].mean * m["y"].mean) / var_x
            self.alpha = m["y"].mean - self.beta * m["x"].mean


def _valid(price: float) -> bool:
    return isinstance(price, (int, float)) and math.isfinite(price) and price > 0


class PairsTrader:
    """Online z-score pairs trader with entry/exit/stop/time-stop and cooldown."""

    def __init__(self, params: Params) -> None:
        self.params = validate(params)
        self.model = SpreadModel(params)
        self.t = 0
        self.position = 0
        self.held = 0
        self.cooldown_left = 0
        self.last = Signal(0, 0.0, params.static_beta, params.static_alpha, False)

    def step(self, price_y: float, price_x: float) -> Signal:
        if not (_valid(price_y) and _valid(price_x)):
            return self.last
        log = self.params.use_log
        y, x = (math.log(price_y), math.log(price_x)) if log else (price_y, price_x)
        z, beta, alpha = self.model.update(y, x)
        self.t += 1
        ready = self.t > self.params.warmup and math.isfinite(z)
        position = self.decide(z) if ready else self._go_flat()
        self.last = Signal(position, z, beta, alpha, ready)
        return self.last

    def decide(self, z: float) -> int:
        p = self.params
        if self.position == 0:
            if self.cooldown_left > 0:
                self.cooldown_left -= 1
            elif p.entry_z < abs(z) < p.stop_z:
                self.position, self.held = (-1 if z > 0 else 1), 0
            return self.position
        self.held += 1
        signed = z * self.position  # negative while the trade is still "open"
        if signed < -p.stop_z or (p.max_hold and self.held >= p.max_hold):
            self.cooldown_left = p.cooldown
            return self._go_flat()
        if signed >= -p.exit_z:
            return self._go_flat()
        return self.position

    def _go_flat(self) -> int:
        self.position, self.held = 0, 0
        return 0


def leg_targets(signal: Signal, price_y: float, price_x: float, params: Params) -> tuple[float, float]:
    """Target shares (y, x) for a signal."""
    if signal.position == 0:
        return 0.0, 0.0
    unit = params.unit_notional * signal.position
    if params.use_log:
        return unit / price_y, -signal.beta * unit / price_x
    return float(unit), -signal.beta * unit


class PairSpec(NamedTuple):
    y: str
    x: str
    params: Params


class PairBook:
    """Runs several pair traders, nets their legs per symbol and scales to position limits."""

    def __init__(self, pairs: Mapping[str, PairSpec], limits: Mapping[str, float] | None = None):
        self.pairs = dict(pairs)
        self.limits = dict(limits or {})
        self.traders = {name: PairsTrader(spec.params) for name, spec in self.pairs.items()}
        self.signals: dict[str, Signal] = {}
        self.prices: dict[str, float] = {}
        self.last_targets: dict[str, float] = {}
        self.errors = 0

    def update(self, prices: Mapping[str, float]) -> dict[str, float]:
        """Feed one tick of prices; return net target shares per symbol. Never raises."""
        try:
            self.prices.update({k: v for k, v in prices.items() if _valid(v)})
            for name, spec in self.pairs.items():
                if spec.y in prices and spec.x in prices:
                    self.signals[name] = self.traders[name].step(prices[spec.y], prices[spec.x])
            self.last_targets = self.combine(self.signals, self.prices)
        except Exception as exc:  # noqa: BLE001 - submission must never crash; hold last targets
            self.errors += 1
            if self.errors == 1:
                print(f"PairBook error (holding last targets): {exc!r}", file=sys.stderr)
        return dict(self.last_targets)

    def combine(self, signals: Mapping[str, Signal], prices: Mapping[str, float]) -> dict[str, float]:
        targets: dict[str, float] = {}
        for name, signal in signals.items():
            spec = self.pairs[name]
            if spec.y not in prices or spec.x not in prices:
                continue
            ty, tx = leg_targets(signal, prices[spec.y], prices[spec.x], spec.params)
            targets[spec.y] = targets.get(spec.y, 0.0) + ty
            targets[spec.x] = targets.get(spec.x, 0.0) + tx
        scale = 1.0
        for sym, qty in targets.items():
            limit = self.limits.get(sym)
            if limit is not None and abs(qty) > limit:
                scale = min(scale, limit / abs(qty))
        return {sym: qty * scale for sym, qty in targets.items()}
