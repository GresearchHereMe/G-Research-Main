"""Tests for gqc.diagnostics: each probe must detect a behaviour we planted in synthetic data."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gqc import diagnostics as dg

REPO = Path(__file__).resolve().parents[1]


def test_diagnose_cli_finds_planted_book_behaviours(tmp_path):
    book = tmp_path / "book.csv"
    scripts = REPO / "scripts"
    subprocess.run([sys.executable, str(scripts / "make_practice_data.py"), str(book),
                    "--book", "--n", "3000"], check=True, capture_output=True)
    out = subprocess.run([sys.executable, str(scripts / "diagnose.py"), str(book),
                          "--trades", str(tmp_path / "book_trades.csv")],
                         check=True, capture_output=True, text=True).stdout
    assert "use microprice" in out  # planted imbalance signal
    assert "mean-reverting" in out  # planted overreaction
    lines = [ln for ln in out.splitlines() if "markouts h=20" in ln]
    assert lines, out
    assert "smart_bot +" in lines[0] and "noise_bot -" in lines[0]


def _ar1_returns(phi: float, n: int = 20000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, 1, n)
    out = np.zeros(n)
    for t in range(1, n):
        out[t] = phi * out[t - 1] + eps[t]
    return out


def test_variance_ratio_flags_mean_reversion_and_momentum():
    assert dg.variance_ratio(_ar1_returns(-0.4), 10) < 0.8
    assert dg.variance_ratio(_ar1_returns(0.4), 10) > 1.2
    assert dg.variance_ratio(_ar1_returns(0.0), 10) == pytest.approx(1.0, abs=0.1)


def test_autocorr_matches_ar1_coefficient():
    assert dg.autocorr(_ar1_returns(-0.3), 1) == pytest.approx(-0.3, abs=0.03)


def test_lead_lag_finds_leader():
    rng = np.random.default_rng(1)
    lead = rng.normal(0, 1, 5000)
    follow = np.roll(lead, 2) * 0.8 + rng.normal(0, 0.6, 5000)
    corr = dg.lead_lag(lead, follow, max_lag=4)
    assert max(corr, key=lambda k: abs(corr[k])) == 2  # lead at t predicts follow at t+2


def test_tick_size_detects_grid():
    assert dg.tick_size(np.array([100.00, 100.05, 100.10, 100.30, 99.95])) == pytest.approx(0.05)


def test_imbalance_predicts_mid_move_when_planted():
    rng = np.random.default_rng(2)
    n = 5000
    imb = rng.uniform(-1, 1, n)
    mid = 100 + np.cumsum(np.concatenate([[0], 0.05 * imb[:-1]]) + rng.normal(0, 0.01, n))
    bid, ask = mid - 0.05, mid + 0.05
    bid_sz, ask_sz = 10 * (1 + imb), 10 * (1 - imb)
    assert dg.imbalance_signal(bid, ask, bid_sz, ask_sz, horizon=1) > 0.5


def test_microprice_leans_toward_thin_side():
    assert dg.microprice(99.0, 101.0, 90.0, 10.0) > 100.0


def test_markouts_separate_dumb_and_informed_counterparties():
    rng = np.random.default_rng(3)
    n = 4000
    mid = 100 + np.cumsum(rng.normal(0, 0.02, n))
    idx = np.arange(10, n - 20, 7)
    signs = rng.choice([-1, 1], len(idx))
    informed = np.sign(mid[idx + 10] - mid[idx])  # trades in the future direction
    parties = np.where(np.arange(len(idx)) % 2 == 0, "smart", "dumb")
    trade_sign = np.where(parties == "smart", informed, signs)
    price = mid[idx] + trade_sign * 0.01
    table = dg.markouts_by_group(idx, price, trade_sign, mid, horizon=10, groups=parties)
    assert table["smart"]["mean"] > 0 > table["dumb"]["mean"]


def test_staleness_counts_unchanged_quotes():
    bid = np.array([1, 1, 1, 2, 2], float)
    ask = np.array([3, 3, 3, 3, 4], float)
    assert dg.staleness(bid, ask) == pytest.approx(2 / 4)


def test_price_report_mentions_mean_reversion():
    rng = np.random.default_rng(4)
    noise = rng.normal(0, 0.5, 3000)
    mean_rev = 100 + noise  # i.i.d. around a level -> strong negative lag-1 autocorr
    walk = 100 + np.cumsum(rng.normal(0, 0.5, 3000))
    lines = dg.price_report({"MR": mean_rev, "RW": walk})
    text = "\n".join(lines)
    assert "MR" in text and "mean-reverting" in text
