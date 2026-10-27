"""Tests for CSV loading, parameter sweeps and single-file submission generation."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from gqc import core, io, sweep
from gqc.synthetic import cointegrated_pair

REPO = Path(__file__).resolve().parents[1]


def test_load_wide_csv(tmp_path):
    path = tmp_path / "wide.csv"
    path.write_text("time,A,B\n2026-01-01,10,20\n2026-01-02,11,\n2026-01-03,12,22\n")
    index, prices = io.load_prices(path)
    assert index == ["2026-01-01", "2026-01-02", "2026-01-03"]
    assert prices["A"].tolist() == [10.0, 11.0, 12.0]
    assert np.isnan(prices["B"][1])


def test_load_long_csv(tmp_path):
    path = tmp_path / "long.csv"
    path.write_text("time,symbol,price\n1,A,10\n1,B,20\n2,A,11\n2,B,21\n")
    index, prices = io.load_prices(path)
    assert index == ["1", "2"]
    assert prices["B"].tolist() == [20.0, 21.0]


def test_load_wide_csv_without_time_column(tmp_path):
    path = tmp_path / "bare.csv"
    path.write_text("A,B\n10,20\n11,21\n")
    index, prices = io.load_prices(path)
    assert index == ["0", "1"]
    assert set(prices) == {"A", "B"}


def test_sweep_returns_ranked_rows_with_test_scores():
    py, px = cointegrated_pair(n=3000, theta=0.05, seed=21)
    rows = sweep.sweep_pair(py, px, train_frac=0.6, cost_bps=1.0,
                            entries=(1.5, 2.0), halflives=(50, 100), exits=(0.0,),
                            hedges=("kalman", "static"))
    assert len(rows) == 8
    assert rows[0].plateau >= rows[-1].plateau
    assert all(np.isfinite(r.test_pnl) for r in rows)
    assert isinstance(rows[0].params, core.Params)


def test_make_submission_is_self_contained(tmp_path):
    out = tmp_path / "submission.py"
    config = {"pairs": {"ab": {"y": "A", "x": "B", "params": {"warmup": 20}}},
              "limits": {"A": 100, "B": 100}}
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(config))
    subprocess.run([sys.executable, str(REPO / "scripts" / "make_submission.py"),
                    "--config", str(cfg), "--out", str(out)], check=True)
    py, px = cointegrated_pair(n=200, seed=22)
    driver = (
        "import sys\n"
        f"ns = {{}}\nexec(open({str(out)!r}).read(), ns)\n"
        "assert 'numpy' not in sys.modules and 'gqc' not in sys.modules\n"
        f"for y, x in zip({py.tolist()!r}, {px.tolist()!r}):\n"
        "    t = ns['on_tick']({'A': y, 'B': x})\n"
        "print(sorted(t))\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", driver], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "A" in result.stdout or result.stdout.strip() == "[]"
