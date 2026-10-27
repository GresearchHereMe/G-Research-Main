"""Tests for gqc.core — the stdlib-only pairs trader that gets pasted into the platform."""

import math
import random

import pytest

from gqc import core
from gqc.synthetic import cointegrated_pair


def test_ewm_matches_closed_form_mean():
    ewm = core.Ewm(halflife=10)
    for value in [1.0] * 200:
        ewm.update(value)
    assert ewm.mean == pytest.approx(1.0)
    assert ewm.std == pytest.approx(0.0, abs=1e-12)


def test_ewm_tracks_variance_of_noise():
    rng = random.Random(0)
    ewm = core.Ewm(halflife=500)
    for _ in range(20000):
        ewm.update(rng.gauss(0.0, 2.0))
    assert ewm.std == pytest.approx(2.0, rel=0.1)


def test_kalman_recovers_hedge_ratio():
    py, px = cointegrated_pair(n=3000, beta=1.5, alpha=0.2, seed=1)
    kf = core.KalmanHedge(delta=1e-5, ve=1e-4)
    for y, x in zip(py, px):
        kf.update(math.log(y), math.log(x))
    assert kf.beta == pytest.approx(1.5, rel=0.05)


@pytest.mark.parametrize("hedge", ["kalman", "ewm", "static"])
def test_trader_flat_during_warmup(hedge):
    params = core.Params(hedge=hedge, warmup=100, static_beta=1.0)
    trader = core.PairsTrader(params)
    py, px = cointegrated_pair(n=100, seed=2)
    signals = [trader.step(y, x) for y, x in zip(py, px)]
    assert all(s.position == 0 for s in signals)


@pytest.mark.parametrize("hedge", ["kalman", "ewm", "static"])
def test_trader_has_no_lookahead(hedge):
    params = core.Params(hedge=hedge, warmup=50, static_beta=1.0)
    py, px = cointegrated_pair(n=600, seed=3)
    base = [core.PairsTrader(params)]
    first = [base[0].step(y, x) for y, x in zip(py, px)]
    tweaked_y = list(py[:400]) + [v * 1.5 for v in py[400:]]
    other = core.PairsTrader(params)
    second = [other.step(y, x) for y, x in zip(tweaked_y, px)]
    assert first[:400] == second[:400]


def _force_z(trader: core.PairsTrader, zs: list[float]) -> list[int]:
    """Drive the state machine directly with a z sequence."""
    return [trader.decide(z) for z in zs]


def test_state_machine_enters_and_exits_at_mean():
    trader = core.PairsTrader(core.Params(entry_z=2.0, exit_z=0.0, stop_z=5.0))
    assert _force_z(trader, [0.0, -2.5, -1.0, 0.1, 2.5, 1.0, -0.1]) == [0, 1, 1, 0, -1, -1, 0]


def test_state_machine_stop_loss_and_cooldown():
    params = core.Params(entry_z=2.0, stop_z=4.0, cooldown=2)
    trader = core.PairsTrader(params)
    assert _force_z(trader, [-2.5, -4.5, -2.5, -2.5, -2.5]) == [1, 0, 0, 0, 1]


def test_state_machine_time_stop():
    trader = core.PairsTrader(core.Params(entry_z=2.0, max_hold=2))
    assert _force_z(trader, [-2.5, -2.4, -2.4, -2.4]) == [1, 1, 0, 1]


def test_bad_prices_hold_previous_signal():
    trader = core.PairsTrader(core.Params(warmup=5))
    py, px = cointegrated_pair(n=50, seed=4)
    for y, x in zip(py, px):
        last = trader.step(y, x)
    assert trader.step(float("nan"), 100.0) == last
    assert trader.step(-1.0, 100.0) == last


def test_leg_targets_log_mode_is_notional_hedged():
    signal = core.Signal(position=1, z=-2.5, beta=2.0, alpha=0.0, ready=True)
    params = core.Params(use_log=True, unit_notional=1000.0)
    shares_y, shares_x = core.leg_targets(signal, 50.0, 100.0, params)
    assert shares_y == pytest.approx(20.0)
    assert shares_x == pytest.approx(-20.0)


def test_leg_targets_price_mode_uses_beta_shares():
    signal = core.Signal(position=-1, z=2.5, beta=1.5, alpha=0.0, ready=True)
    params = core.Params(use_log=False, unit_notional=10.0)
    assert core.leg_targets(signal, 50.0, 100.0, params) == (-10.0, 15.0)


def test_pair_book_nets_and_scales_to_limits():
    pairs = {
        "ab": core.PairSpec("A", "B", core.Params(use_log=False, unit_notional=10.0)),
        "cb": core.PairSpec("C", "B", core.Params(use_log=False, unit_notional=10.0)),
    }
    book = core.PairBook(pairs, limits={"B": 5.0})
    targets = book.combine(
        {
            "ab": core.Signal(1, -3.0, 1.0, 0.0, True),
            "cb": core.Signal(1, -3.0, 1.0, 0.0, True),
        },
        {"A": 10.0, "B": 10.0, "C": 10.0},
    )
    assert targets["B"] == pytest.approx(-5.0)
    assert targets["A"] == pytest.approx(2.5)
    assert targets["C"] == pytest.approx(2.5)


def test_pair_book_counts_errors_and_holds_targets(capsys):
    pairs = {"ab": core.PairSpec("A", "B", core.Params(warmup=1))}
    book = core.PairBook(pairs)
    book.update({"A": 10.0, "B": 10.0})
    book.combine = None  # force a failure inside update
    assert book.update({"A": 11.0, "B": 10.0}) == book.last_targets
    assert book.errors == 1
    assert "PairBook error" in capsys.readouterr().err


def test_params_validation_rejects_bad_config():
    with pytest.raises(ValueError):
        core.PairsTrader(core.Params(hedge="magic"))
    with pytest.raises(ValueError):
        core.PairsTrader(core.Params(entry_z=5.0, stop_z=4.0))


def test_pair_book_update_never_raises_on_missing_symbol():
    pairs = {"ab": core.PairSpec("A", "B", core.Params())}
    book = core.PairBook(pairs)
    assert book.update({"A": 10.0}) == {}
