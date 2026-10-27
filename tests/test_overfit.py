"""Tests for gqc.overfit (PBO via CSCV, PSR/DSR, MinBTL) and gqc.research (generic grids)."""

import math

import numpy as np
import pytest

from gqc import overfit, research


def _noise(t: int, n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 1.0, (t, n))


def test_cscv_uses_all_half_splits():
    res = overfit.cscv_pbo(_noise(80, 5, 0), n_blocks=8)
    assert res.n_combos == math.comb(8, 4)


def test_pbo_averages_near_half_for_pure_noise():
    # A single noise sample can give PBO far from 0.5 (one column is lucky in both halves);
    # the expectation over samples is 0.5.
    pbos = [overfit.cscv_pbo(_noise(1000, 30, seed), n_blocks=10).pbo for seed in range(30)]
    assert 0.35 < float(np.mean(pbos)) < 0.65


def test_pbo_low_when_one_config_has_real_edge():
    m = _noise(2000, 30, 2)
    m[:, 0] += 0.15
    assert overfit.cscv_pbo(m, n_blocks=16).pbo < 0.1


def test_pbo_high_when_in_sample_winners_are_out_of_sample_losers():
    s, per_block, n = 8, 50, 40
    rng = np.random.default_rng(3)
    m = rng.normal(0.0, 0.1, (s * per_block, n))
    for j in range(n):
        good = rng.choice(s, s // 2, replace=False)
        for b in range(s):
            m[b * per_block : (b + 1) * per_block, j] += 1.0 if b in good else -1.0
    assert overfit.cscv_pbo(m, n_blocks=s).pbo > 0.9


def test_expected_max_sharpe_of_noise():
    assert overfit.expected_max_sharpe(1, 1.0) == 0.0
    assert overfit.expected_max_sharpe(1000, 1.0) == pytest.approx(3.25, abs=0.1)
    assert overfit.expected_max_sharpe(10, 1.0) < overfit.expected_max_sharpe(100, 1.0)


def test_psr_is_half_at_its_own_sharpe():
    r = np.random.default_rng(4).normal(0.05, 1.0, 3000)
    assert overfit.probabilistic_sharpe(r, overfit.sharpe(r)) == pytest.approx(0.5)


def test_deflated_sharpe_separates_edge_from_noise():
    t = 5000
    strong = np.random.default_rng(5).normal(0.1, 1.0, t)
    noise = np.random.default_rng(6).normal(0.0, 1.0, t)
    assert overfit.deflated_sharpe(strong, n_trials=100, sr_var=1.0 / t) > 0.95
    assert overfit.deflated_sharpe(noise, n_trials=100, sr_var=1.0 / t) < 0.5


def test_min_backtest_length():
    assert overfit.min_backtest_length(100, 0.1) == math.ceil(2 * math.log(100) / 0.01)
    assert overfit.min_backtest_length(1, 0.1) == 1


def test_positions_pnl_hand_computed():
    prices = {"A": np.array([10.0, 11.0, 12.0])}
    pos = {"A": np.array([1.0, 1.0, 0.0])}
    assert research.positions_pnl(prices, pos, lag=1).tolist() == [0.0, 0.0, 1.0]
    costly = research.positions_pnl(prices, pos, lag=1, cost_bps=100.0)
    assert costly.sum() == pytest.approx(1.0 - 11.0 * 0.01 - 0.0)  # entry at t1; exit after end


def test_research_template_runs_end_to_end(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    csv = tmp_path / "p.csv"
    rng = np.random.default_rng(8)
    rows = ["t,A,B"] + [f"{t},{100 + rng.normal():.4f},{100 + t * 0.01:.4f}" for t in range(600)]
    csv.write_text("\n".join(rows))
    out = subprocess.run([sys.executable, str(repo / "templates" / "research_template.py"),
                          str(csv), "--runs-dir", str(tmp_path / "runs")],
                         check=True, capture_output=True, text=True).stdout
    assert "VERDICT" in out


def test_grid_returns_shape_and_report():
    rng = np.random.default_rng(7)
    prices = {"A": 100 + np.cumsum(rng.normal(0, 1, 1500))}

    def fade(px, window, k):
        p = px["A"]
        ma = np.convolve(p, np.ones(window) / window, mode="full")[: len(p)]
        return {"A": np.where(p > ma + k, -1.0, np.where(p < ma - k, 1.0, 0.0))}

    configs, matrix = research.grid_returns(fade, prices, {"window": [5, 20], "k": [0.0, 1.0]})
    assert len(configs) == 4 and matrix.shape == (1500, 4)
    text = "\n".join(research.overfit_report(configs, matrix, n_blocks=8))
    assert "PBO" in text and "Deflated Sharpe" in text
