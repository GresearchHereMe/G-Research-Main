"""Submission checks must reject invalid output and preserve marked inventory PnL."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gqc import replay

REPO = Path(__file__).resolve().parents[1]


def _run_cli(tmp_path: Path, source: str, *args: str) -> subprocess.CompletedProcess:
    submission = tmp_path / "submission.py"
    submission.write_text(source)
    prices = tmp_path / "prices.csv"
    prices.write_text("timestamp,A\n0,100\n1,100\n2,100\n")
    return subprocess.run(
        [sys.executable, str(REPO / "scripts/replay_skeleton.py"), str(submission), str(prices),
         *args], check=False, capture_output=True, text=True,
    )


def test_replay_marks_held_inventory_across_missing_quotes() -> None:
    def step(book: dict, positions: dict) -> dict:
        return {"A": 1} if not positions and "A" in book else {}

    res = replay.replay(step, {"A": np.array([100.0, np.nan, 110.0])}, lag=0)
    assert res.pnl.tolist() == [0.0, 0.0, 10.0]
    assert res.trades == 1


@pytest.mark.parametrize("qty", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("lag", [0, 1])
def test_replay_rejects_nonfinite_order_before_filling(qty: float, lag: int) -> None:
    with pytest.raises(ValueError, match="non-finite order"):
        replay.replay(lambda book, positions: {"A": qty}, {"A": np.full(3, 100.0)}, lag=lag)


@pytest.mark.parametrize("state_name", ["STATE", "BOOK"])
def test_cli_fails_on_recorded_submission_errors(tmp_path: Path, state_name: str) -> None:
    result = _run_cli(tmp_path, f"""
class State:
    errors = 0
{state_name} = State()
LIMITS = {{"A": 10}}
def step(book, positions):
    {state_name}.errors += 1
    return {{}}
""")
    assert result.returncode == 1
    assert "ERROR" in result.stdout and "3" in result.stdout


def test_cli_rejects_nonfinite_orders(tmp_path: Path) -> None:
    result = _run_cli(tmp_path, """
LIMITS = {"A": 10}
def step(book, positions):
    return {"A": float("nan")}
""")
    assert result.returncode == 1
    assert "ERROR" in result.stdout and "non-finite order" in result.stdout


def test_cli_includes_initial_cost_in_drawdown(tmp_path: Path) -> None:
    result = _run_cli(tmp_path, """
LIMITS = {"A": 10}
def step(book, positions):
    return {"A": 1} if not positions else {}
""", "--lag", "0", "--cost-bps", "100")
    assert result.returncode == 0, result.stderr
    assert "PnL -1.00" in result.stdout
    assert "max DD 1.00" in result.stdout


def test_cli_rejects_nonfinite_pnl_from_overflow(tmp_path: Path) -> None:
    result = _run_cli(tmp_path, """
def step(book, positions):
    return {"A": 1e308} if not positions else {}
""", "--lag", "0", "--cost-bps", "100")
    assert result.returncode == 1
    assert "ERROR" in result.stdout and "non-finite" in result.stdout
