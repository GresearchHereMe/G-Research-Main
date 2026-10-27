"""Replay of the paste-ready skeleton and the evaluate_algo CLI."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gqc import replay

REPO = Path(__file__).resolve().parents[1]


def test_replay_buy_and_hold_pnl_with_lag_and_cost():
    prices = {"A": np.array([10.0, 11.0, 12.0, 15.0])}
    calls = {"n": 0}

    def step(book, positions):
        calls["n"] += 1
        return {"A": 1} if positions.get("A", 0) == 0 and calls["n"] == 1 else {}

    res = replay.replay(step, prices, lag=1, cost_bps=100.0)
    # order at t0 fills at t1 price 11 (cost 0.11); held to 15 -> +4
    assert res.pnl.sum() == pytest.approx(4.0 - 0.11)
    assert res.trades == 1
    assert res.final_positions == {"A": 1}


def test_replay_runs_the_skeleton_file():
    ns = replay.load_submission(REPO / "templates" / "skeleton.py")
    res = replay.replay(ns["step"], {"A": np.linspace(100, 110, 50)}, lag=1)
    assert res.trades == 0 and res.pnl.sum() == 0.0


def test_evaluate_algo_cli_on_research_template(tmp_path):
    rng = np.random.default_rng(0)
    x = np.zeros(1500)
    for t in range(1, 1500):
        x[t] = 0.85 * x[t - 1] + rng.normal(0, 0.5)
    csv = tmp_path / "ou.csv"
    csv.write_text("t,A\n" + "\n".join(f"{t},{100 + v:.4f}" for t, v in enumerate(x)))
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "evaluate_algo.py"),
         str(REPO / "templates" / "research_template.py"), str(csv), "--frequencies", "1,5"],
        check=True, capture_output=True, text=True,
    ).stdout
    assert "DECISION:" in out and "G0 integrity" in out
