"""Tests for gqc.pipeline: the gate sequence every algo must pass before it trades."""

import numpy as np

from gqc import pipeline


def _ou_prices(n: int = 3000, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.9 * x[t - 1] + rng.normal(0, 0.5)
    return {"A": 100 + x}


def _walk(n: int = 3000, seed: int = 1) -> dict[str, np.ndarray]:
    return {"A": 100 + np.cumsum(np.random.default_rng(seed).normal(0, 0.5, n))}


def fade(prices, window, entry):
    p = prices["A"]
    ma = np.convolve(p, np.ones(window) / window, mode="full")[: len(p)]
    sd = np.sqrt(np.convolve((p - ma) ** 2, np.ones(window) / window, mode="full")[: len(p)])
    z = np.divide(p - ma, sd, out=np.zeros_like(p), where=sd > 0)
    return {"A": np.where(z > entry, -1.0, np.where(z < -entry, 1.0, 0.0))}


def cheat(prices, window, entry):
    p = prices["A"]
    future = np.append(p[1:], p[-1])
    return {"A": np.sign(future - p)}


GRID = {"window": [10, 30], "entry": [0.5, 1.0]}


def test_throttle_holds_between_rebalances_and_respects_band():
    pos = np.array([1.0, -1.0, -1.0, 1.0, 1.0, 0.2])
    assert pipeline.throttle(pos, every=3, band=0.0).tolist() == [1, 1, 1, 1, 1, 1]
    assert pipeline.throttle(pos, every=1, band=0.9).tolist() == [1, -1, -1, 1, 1, 1]


def test_mean_reversion_on_ou_passes():
    ev = pipeline.evaluate(fade, _ou_prices(), GRID, cost_bps=0.5, frequencies=(1, 5))
    assert ev.decision in ("GO", "GO-REDUCED"), "\n".join(ev.lines)
    assert ev.size_fraction > 0
    assert all(g.passed for g in ev.gates if g.name == "G0 integrity")


def test_fade_on_random_walk_is_rejected():
    ev = pipeline.evaluate(fade, _walk(), GRID, cost_bps=0.5, frequencies=(1, 5))
    assert ev.decision == "NO-GO", "\n".join(ev.lines)
    assert ev.size_fraction == 0.0


def test_lookahead_strategy_fails_integrity():
    ev = pipeline.evaluate(cheat, _walk(), GRID, cost_bps=0.0, frequencies=(1,))
    integrity = next(g for g in ev.gates if g.name == "G0 integrity")
    assert not integrity.passed
    assert ev.decision == "NO-GO"


def test_report_lists_all_gates_and_frequency_table():
    ev = pipeline.evaluate(fade, _ou_prices(), GRID, cost_bps=0.5, frequencies=(1, 5))
    text = "\n".join(ev.lines)
    for name in ("G0", "G1", "G2", "G3", "G4", "DECISION", "every"):
        assert name in text
