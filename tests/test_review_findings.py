"""Regression tests for WINNING.md §8 findings A-H. Each test reproduces the reported defect."""

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gqc import backtest, core, io, overfit, pipeline, registry, replay, research, sweep
from gqc.synthetic import cointegrated_pair

REPO = Path(__file__).resolve().parents[1]
SKELETON = REPO / "templates" / "skeleton.py"


def _skeleton() -> dict:
    return replay.load_submission(SKELETON)


def _submission(tmp_path, limits: dict, params: dict | None = None) -> dict:
    cfg = tmp_path / "cfg.json"
    params = params or {"warmup": 5}
    cfg.write_text(json.dumps({"pairs": {"ab": {"y": "A", "x": "B", "params": params}},
                               "limits": limits}))
    out = tmp_path / "sub.py"
    subprocess.run([sys.executable, str(REPO / "scripts" / "make_submission.py"),
                    "--config", str(cfg), "--out", str(out)], check=True, capture_output=True)
    return replay.load_submission(out)


# --- C: final order safety -------------------------------------------------------------------


def test_c_to_orders_lot_rounding_never_breaches_limit(tmp_path):
    ns = _submission(tmp_path, {"A": 10})
    assert ns["to_orders"]({"A": 10}, {"A": 8}, lot=3) == {}  # +3 would reach 11
    assert ns["to_orders"]({"A": -10}, {"A": 8}, lot=3) == {"A": -18}
    assert ns["to_orders"]({"A": -10}, {"A": 8}, lot=4) == {"A": -16}


def test_c_to_orders_counts_resting_orders(tmp_path):
    ns = _submission(tmp_path, {"A": 10})
    assert ns["to_orders"]({"A": 10}, {"A": 8}, outstanding={"A": (2, 0)}) == {}
    assert ns["to_orders"]({"A": -10}, {"A": 0}, outstanding={"A": (0, 7)}) == {"A": -3}


def test_c_skeleton_clip_lots_and_resting_orders():
    ns = _skeleton()
    ns["LIMITS"].update({"A": 10})
    ns["LOTS"].update({"A": 3})
    assert ns["clip_orders"]({"A": 2}, {"A": 8}) == {}
    assert ns["clip_orders"]({"A": 50}, {"A": 0}) == {"A": 9}
    ns["LOTS"].clear()
    assert ns["clip_orders"]({"A": 5}, {"A": 0}, {"A": (8, 0)}) == {"A": 2}
    assert ns["clip_orders"]({"A": -5}, {"A": 0}, {"A": (0, 8)}) == {"A": -2}


def test_e_generated_bundle_replays_within_limits(tmp_path):
    # unit_notional must make legs >= 1 lot, else integer orders round to zero (silent no-trade)
    ns = _submission(tmp_path, {"A": 50, "B": 50}, {"warmup": 50, "unit_notional": 2000})
    py, px = cointegrated_pair(n=800, theta=0.1, seed=33)
    res = replay.replay(ns["step"], {"A": py, "B": px}, lag=1, cost_bps=1.0)
    assert res.trades > 0
    assert all(v <= 50 for v in res.max_abs_position.values())
    assert ns["BOOK"].errors == 0


# --- D: an error must not create new risk ----------------------------------------------------


def test_d_pairbook_failure_returns_none():
    book = core.PairBook({"ab": core.PairSpec("A", "B", core.Params(warmup=1))})
    book.update({"A": 10.0, "B": 10.0})
    book.combine = None
    assert book.update({"A": 11.0, "B": 10.0}) is None
    assert book.errors == 1


def test_d_generated_on_tick_holds_positions_on_failure(tmp_path):
    ns = _submission(tmp_path, {"A": 100, "B": 100})
    ns["BOOK"].combine = None
    positions = {"A": 5, "B": -5}
    targets = ns["on_tick"]({"A": 10.0, "B": 10.0}, positions)
    assert targets == positions
    assert ns["to_orders"](targets, positions) == {}
    assert ns["on_tick"]({"A": 10.0, "B": 10.0}) is None
    assert ns["to_orders"](None, positions) == {}


# --- H: pipeline / replay / skeleton risk ----------------------------------------------------


def test_h_replay_lag_zero_fills_immediately():
    calls = {"n": 0}

    def step(book, positions):
        calls["n"] += 1
        return {"A": 1} if calls["n"] == 1 else {}

    res = replay.replay(step, {"A": np.array([100.0, 110.0, 120.0])}, lag=0)
    assert res.trades == 1
    assert res.pnl.sum() == pytest.approx(20.0)


def test_h_skeleton_equity_does_not_credit_fresh_fills():
    ns = _skeleton()
    ns["step"]({"A": 100.0}, {})
    ns["step"]({"A": 110.0}, {"A": 1})  # filled since last step, price unknown -> no phantom PnL
    assert ns["STATE"].equity == pytest.approx(0.0)
    ns["step"]({"A": 120.0}, {"A": 1})
    assert ns["STATE"].equity == pytest.approx(10.0)


def test_h_skeleton_uses_platform_pnl_when_given():
    ns = _skeleton()
    ns["RISK"].update({"max_drawdown": 5.0, "halt_steps": 0})
    ns["step"]({"A": 100.0}, {"A": 2}, account_pnl=10.0)
    assert ns["step"]({"A": 100.0}, {"A": 2}, account_pnl=3.0) == {"A": -2}


def test_h_gross_cap_allows_risk_reduction():
    ns = _skeleton()
    ns["RISK"].update({"max_gross": 100.0})
    ns["decide"] = lambda book, positions, state: {"A": -1}
    assert ns["step"]({"A": 10.0}, {"A": 20}) == {"A": -1}
    ns["decide"] = lambda book, positions, state: {"A": 1}
    assert ns["step"]({"A": 10.0}, {"A": 20}) == {}


def test_h_flatten_respects_max_order_until_flat():
    ns = _skeleton()
    ns["RISK"].update({"max_drawdown": 5.0, "halt_steps": 0})
    ns["MAX_ORDER"].update({"A": 2})
    ns["decide"] = lambda book, positions, state: {}
    ns["step"]({"A": 100.0}, {"A": 20})
    assert ns["step"]({"A": 99.0}, {"A": 20}) == {"A": -2}
    assert ns["step"]({"A": 99.0}, {"A": 18}) == {"A": -2}  # still flattening after halt_steps


def test_h_integrity_checks_raw_targets_not_throttled():
    def bad(prices, k):
        return {"A": np.full(len(prices["A"]), np.inf)}

    prices = {"A": 100 + np.cumsum(np.random.default_rng(0).normal(0, 1, 400))}
    ev = pipeline.evaluate(bad, prices, {"k": [1]}, frequencies=(1,), train_frac=0.7)
    gate = next(g for g in ev.gates if g.name == "G0 integrity")
    assert not gate.passed and "non-finite" in gate.detail


def test_h_pipeline_selects_on_development_and_reports_holdout():
    rng = np.random.default_rng(1)
    x = np.zeros(2000)
    for t in range(1, 2000):
        x[t] = 0.8 * x[t - 1] + rng.normal(0, 0.5)
    prices = {"A": 100 + x}

    def fade(p, entry):
        z = p["A"] - 100
        return {"A": np.where(z > entry, -1.0, np.where(z < -entry, 1.0, 0.0))}

    base = pipeline.evaluate(fade, prices, {"entry": [0.5, 1.0]}, frequencies=(1,), train_frac=0.7)
    tail = {"A": np.concatenate([prices["A"][:1400], prices["A"][1400:] * 1.5])}
    other = pipeline.evaluate(fade, tail, {"entry": [0.5, 1.0]}, frequencies=(1,), train_frac=0.7)
    assert base.best_config == other.best_config  # selection never saw the holdout
    assert any("holdout" in line.lower() for line in base.lines)


# --- A / B: sweep evidence -------------------------------------------------------------------


def test_a_sweep_decisions_before_a_change_do_not_depend_on_it():
    py, px = cointegrated_pair(n=1500, seed=31)
    kw = {"train_frac": 0.7, "formation_frac": 0.2, "entries": (1.5, 2.0), "halflives": (50,),
          "exits": (0.0,), "hedges": ("static", "kalman")}
    rows = sweep.sweep_pair(py, px, **kw)
    k = 900  # inside the train window, after formation
    py2 = py.copy()
    py2[k:] *= 1.3
    rows2 = sweep.sweep_pair(py2, px, **kw)
    by = {(r.params.hedge, r.params.entry_z): r.pnl for r in rows}
    for r in rows2:
        assert np.allclose(by[(r.params.hedge, r.params.entry_z)][:k], r.pnl[:k])


def test_b_sweep_evidence_ignores_test_window_and_scores_selected():
    py, px = cointegrated_pair(n=1500, seed=32)
    rows = sweep.sweep_pair(py, px, train_frac=0.7, formation_frac=0.2, entries=(1.5, 2.0),
                            halflives=(50, 100), exits=(0.0,), hedges=("kalman",))
    lines = sweep.evidence_report(rows)
    tampered = [r.__class__(**{**r.__dict__, "pnl": np.concatenate(
        [r.pnl[: r.split], r.pnl[r.split:] * -5])}) for r in rows]
    assert sweep.evidence_report(tampered) == lines
    assert any(str(rows[0].params.entry_z) in ln and "Selected" in ln for ln in lines)


# --- E: metrics ------------------------------------------------------------------------------


def test_e_open_inventory_drawdown_and_round_trips():
    py = np.array([100.0, 100.0, 100.0])
    targets = np.array([[1.0, 0.0]] * 3)
    res = backtest.run_targets(py, py, targets, lag=0, cost_bps=100.0)
    assert res.metrics["total_pnl"] == pytest.approx(-1.0)
    assert res.metrics["round_trips"] == 0
    assert res.metrics["open_at_end"] is True
    assert res.metrics["max_drawdown"] == pytest.approx(1.0)


# --- F: data alignment -----------------------------------------------------------------------


def test_f_timestamp_header_is_time_and_rows_are_sorted(tmp_path):
    path = tmp_path / "p.csv"
    path.write_text("timestamp,A,B\n3,12,22\n1,10,20\n2,11,21\n")
    index, prices = io.load_prices(path)
    assert set(prices) == {"A", "B"}
    assert index == ["1", "2", "3"]
    assert prices["A"].tolist() == [10.0, 11.0, 12.0]


def test_f_lead_lag_alignment_survives_gaps():
    from gqc import diagnostics as dg

    rng = np.random.default_rng(5)
    a = 100 + np.cumsum(rng.normal(0, 1, 3000))
    b = np.concatenate([[a[0]], a[:-1]])  # b follows a by one step
    a[500] = np.nan
    text = "\n".join(dg.price_report({"A": a, "B": b}))
    assert "A leads B by 1" in text


# --- G: trial accounting ---------------------------------------------------------------------


def test_g_single_candidate_is_not_auto_rejected():
    r = np.random.default_rng(6).normal(0.2, 1.0, (3000, 1))
    text = "\n".join(research.overfit_report([{"a": 1}], r))
    assert "PBO" in text and "n/a" in text
    assert "REJECT" not in text


def test_g_prior_trials_make_dsr_stricter():
    m = np.random.default_rng(7).normal(0.03, 1.0, (2000, 5))
    loose = research.overfit_report([{"i": i} for i in range(5)], m)
    strict = research.overfit_report([{"i": i} for i in range(5)], m, prior_trials=500)
    dsr = [float(next(ln for ln in rep if "Deflated" in ln).split(":")[-1]) for rep in (loose, strict)]
    assert dsr[1] < dsr[0]


def test_g_registry_accumulates_trials_per_dataset(tmp_path):
    data = tmp_path / "d.csv"
    data.write_text("t,A\n0,1\n")
    fp = registry.fingerprint(data)
    runs = tmp_path / "runs"
    assert registry.total_trials(runs, fp) == 0
    registry.record(runs, fp, "idea1", 12, "NO-GO")
    registry.record(runs, fp, "idea2", 8, "GO")
    assert registry.total_trials(runs, fp) == 20
    assert registry.total_trials(runs, "other") == 0


def test_kalman_spread_stats_not_poisoned_by_burn_in():
    # Found while fixing E: Kalman starts at beta=0, its early residuals (~4.6 in log units)
    # inflated the spread volatility so |z| never exceeded ~1.4 on 800 rows -> no trades.
    py, px = cointegrated_pair(n=800, theta=0.1, seed=33)
    trader = core.PairsTrader(core.Params(hedge="kalman", warmup=50))
    signals = [trader.step(float(y), float(x)) for y, x in zip(py, px)]
    assert max(abs(s.z) for s in signals[60:]) > 2.5
    assert any(s.position != 0 for s in signals)


def test_expected_max_sharpe_monotone_still_holds():
    assert overfit.expected_max_sharpe(520, 1.0) > overfit.expected_max_sharpe(20, 1.0)
    assert math.isfinite(overfit.expected_max_sharpe(10_000, 1e-4))
