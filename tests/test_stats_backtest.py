"""Tests for gqc.stats (cointegration screening) and gqc.backtest (PnL accounting)."""

import numpy as np
import pytest

from gqc import backtest, core, stats
from gqc.synthetic import cointegrated_pair, random_walk_pair


def test_ols_recovers_line():
    x = np.linspace(0, 10, 50)
    alpha, beta, resid = stats.ols(3.0 + 2.0 * x, x)
    assert (alpha, beta) == pytest.approx((3.0, 2.0))
    assert np.allclose(resid, 0.0)


def test_eg_detects_cointegration():
    py, px = cointegrated_pair(n=2000, beta=1.2, theta=0.05, seed=10)
    result = stats.eg_coint(np.log(py), np.log(px))
    assert result.tstat < result.crit[0.05]
    assert result.beta == pytest.approx(1.2, rel=0.05)
    assert 8 < result.half_life < 20  # true ln2/0.05 ≈ 13.9


def test_eg_rarely_flags_independent_walks():
    flagged = 0
    for seed in range(40):
        py, px = random_walk_pair(n=1000, seed=seed)
        flagged += stats.eg_coint(np.log(py), np.log(px)).is_coint(0.05)
    assert flagged <= 6


def test_mackinnon_crit_finite_sample_is_more_negative():
    assert stats.mackinnon_crit(100)[0.05] < stats.mackinnon_crit(10_000)[0.05]


def test_screen_ranks_cointegrated_pair_first():
    py, px = cointegrated_pair(n=1500, seed=11)
    wy, wx = random_walk_pair(n=1500, seed=12)
    rows = stats.screen({"Y": py, "X": px, "W1": wy, "W2": wx})
    assert {rows[0].y, rows[0].x} == {"Y", "X"}


def test_backtest_hand_computed_pnl_and_costs():
    py = np.array([10.0, 10.0, 11.0, 12.0])
    px = np.array([10.0, 10.0, 10.0, 10.0])
    targets = np.array([[1.0, -1.0], [1.0, -1.0], [0.0, 0.0], [0.0, 0.0]])
    res = backtest.run_targets(py, px, targets, lag=1, cost_bps=0.0)
    # executed positions: t0 flat, t1..t2 long 1y/short 1x, t3 flat -> PnL (11-10) + (12-11)
    assert res.equity[-1] == pytest.approx(2.0)
    costly = backtest.run_targets(py, px, targets, lag=1, cost_bps=100.0)
    traded_notional = 10 + 10 + 12 + 10  # in at t1, out at t3
    assert res.equity[-1] - costly.equity[-1] == pytest.approx(traded_notional * 0.01)


def test_backtest_lag_zero_is_more_optimistic_on_mean_reversion():
    py, px = cointegrated_pair(n=3000, theta=0.1, seed=13)
    params = core.Params(hedge="kalman", entry_z=1.5, warmup=200)
    fast = backtest.run_pair(py, px, params, lag=0)
    slow = backtest.run_pair(py, px, params, lag=1)
    assert fast.metrics["total_pnl"] > slow.metrics["total_pnl"]


def test_backtest_profitable_on_cointegrated_with_small_costs():
    py, px = cointegrated_pair(n=4000, theta=0.05, sigma_s=0.01, seed=14)
    params = core.Params(hedge="kalman", entry_z=2.0, warmup=200, unit_notional=10_000)
    res = backtest.run_pair(py, px, params, lag=1, cost_bps=1.0)
    assert res.metrics["total_pnl"] > 0
    assert res.metrics["round_trips"] >= 10
    assert set(res.metrics) >= {"sharpe", "max_drawdown", "win_rate", "costs"}
