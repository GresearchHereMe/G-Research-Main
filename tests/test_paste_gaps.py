"""Pasted-file gaps: SIZE_FRACTION scales targets, end-of-round flatten, both interface styles."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from gqc import replay
from gqc.synthetic import cointegrated_pair

REPO = Path(__file__).resolve().parents[1]
SKELETON = REPO / "templates" / "skeleton.py"
BUILDER = REPO / "scripts" / "make_submission.py"


def _skeleton() -> dict:
    return replay.load_submission(SKELETON)


def _build(tmp_path: Path, **extra) -> subprocess.CompletedProcess:
    config = {"pairs": {"ab": {"y": "A", "x": "B", "params": {"warmup": 20, "entry_z": 1.0,
                                                              "unit_notional": 1000}}},
              "limits": {"A": 1000, "B": 1000}, **extra}
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(config))
    out = tmp_path / "sub.py"
    return subprocess.run([sys.executable, str(BUILDER), "--config", str(cfg), "--out", str(out)],
                          check=False, capture_output=True, text=True)


def _submission(tmp_path: Path, **extra) -> dict:
    result = _build(tmp_path, **extra)
    assert result.returncode == 0, result.stderr
    return replay.load_submission(tmp_path / "sub.py")


# --- skeleton: SIZE_FRACTION -----------------------------------------------------------------


def test_target_orders_scales_targets_by_size_fraction():
    ns = _skeleton()
    ns["SIZE_FRACTION"] = 0.5
    assert ns["target_orders"]({"A": 40}, {"A": 5}) == {"A": 15}


def test_target_orders_caps_scaled_target_at_limit():
    ns = _skeleton()
    ns["LIMITS"]["A"] = 10
    assert ns["target_orders"]({"A": 50}, {"A": -2}) == {"A": 12}
    assert ns["target_orders"]({"A": -50}, {"A": 3}) == {"A": -13}


def test_target_orders_skips_symbols_already_at_target():
    ns = _skeleton()
    assert ns["target_orders"]({"A": 4, "B": 0}, {"A": 4}) == {}


def test_size_fraction_applies_through_step_when_decide_returns_targets():
    ns = _skeleton()
    ns["SIZE_FRACTION"] = 0.5
    ns["decide"] = lambda book, positions, state: ns["target_orders"]({"A": 10}, positions)
    assert ns["step"]({"A": 10.0}) == {"A": 5}


# --- skeleton: end-of-round flatten ----------------------------------------------------------


def test_flatten_after_closes_positions_and_stays_out():
    ns = _skeleton()
    ns["RISK"]["flatten_after"] = 3
    ns["decide"] = lambda book, positions, state: {"A": 1}
    assert ns["step"]({"A": 10.0}) == {"A": 1}
    assert ns["step"]({"A": 10.0}, {"A": 1}) == {"A": 1}
    assert ns["step"]({"A": 10.0}, {"A": 2}) == {"A": -2}
    assert ns["step"]({"A": 10.0}, {}) == {}


def test_flatten_after_respects_max_order_until_flat():
    ns = _skeleton()
    ns["RISK"]["flatten_after"] = 1
    ns["MAX_ORDER"]["A"] = 2
    assert ns["step"]({"A": 10.0}, {"A": 5}) == {"A": -2}
    assert ns["step"]({"A": 10.0}, {"A": 3}) == {"A": -2}


# --- skeleton: target-position interface -----------------------------------------------------


def test_step_targets_returns_positions_plus_orders():
    ns = _skeleton()
    ns["decide"] = lambda book, positions, state: {"A": 3}
    assert ns["step_targets"]({"A": 10.0, "B": 5.0}, {"A": 2, "B": 1}) == {"A": 5, "B": 1}


def test_step_targets_holds_positions_on_error():
    ns = _skeleton()

    def broken(book, positions, state):
        raise RuntimeError("boom")

    ns["decide"] = broken
    assert ns["step_targets"]({"A": 10.0}, {"A": 4}) == {"A": 4}
    assert ns["STATE"].errors == 1


# --- generated pairs file --------------------------------------------------------------------


def test_generated_size_fraction_scales_pair_targets(tmp_path):
    full = _submission(tmp_path / "full")
    half = _submission(tmp_path / "half", size_fraction=0.5)
    py, px = cointegrated_pair(n=300, seed=22)
    traded = False
    for y, x in zip(py.tolist(), px.tolist()):
        a, b = full["on_tick"]({"A": y, "B": x}), half["on_tick"]({"A": y, "B": x})
        assert set(a) == set(b)
        assert all(b[s] == pytest.approx(0.5 * a[s]) for s in a)
        traded = traded or any(a.values())
    assert traded


def test_generated_flatten_after_targets_zero_from_that_tick(tmp_path):
    py, px = cointegrated_pair(n=300, seed=22)
    rows = [{"A": y, "B": x} for y, x in zip(py.tolist(), px.tolist())]
    base = _submission(tmp_path / "base")
    targets = [base["on_tick"](row) for row in rows]
    k = next(i for i, t in enumerate(targets, 1) if any(t.values()))  # first tick holding risk
    ns = _submission(tmp_path / "flat", flatten_after=k)
    for row, expected in zip(rows[: k - 1], targets):
        assert ns["on_tick"](row) == expected
    positions = {"A": 3.0, "B": -3.0}
    assert ns["on_tick"](rows[k - 1], positions) == {"A": 0.0, "B": 0.0}
    assert ns["step"](rows[k], positions) == {"A": -3, "B": 3}


@pytest.mark.parametrize("extra", [{"size_fraction": 1.5}, {"size_fraction": 0},
                                   {"flatten_after": 0}, {"flatten_after": 2.5}])
def test_builder_rejects_invalid_sizing_and_flatten_settings(tmp_path, extra):
    result = _build(tmp_path, **extra)
    assert result.returncode != 0
    assert next(iter(extra)) in result.stderr
