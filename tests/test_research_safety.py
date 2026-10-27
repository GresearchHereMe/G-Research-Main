"""Regression checks for research selection and its event-time entry points."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gqc import pipeline, research


@pytest.mark.parametrize("pnl, expected", [([], 0), ([-1, 0, 0], 1),
                                          ([-1, -1, -1], 3), ([2, -3, 1], 3)])
def test_drawdown_includes_zero_starting_equity(pnl: list[float], expected: float) -> None:
    assert research.max_drawdown(np.array(pnl, dtype=float)) == expected


def test_pipeline_drawdown_includes_initial_loss() -> None:
    assert pipeline._max_drawdown(np.array([-1.0, 0.0, 0.0])) == 1.0


def test_pipeline_prefers_supported_plateau_over_isolated_sharpe_peak() -> None:
    rng = np.random.default_rng(81)
    prices = {"A": 100 + np.cumsum(rng.normal(0.1, 0.05, 1000))}

    def strategy(prices: dict[str, np.ndarray], mode: int) -> dict[str, np.ndarray]:
        n = len(prices["A"])
        if mode == 2:
            return {"A": np.ones(n)}  # strongest isolated development result
        if mode >= 4:
            return {"A": (np.arange(n) % 2).astype(float)}
        return {"A": -np.ones(n)}

    ev = pipeline.evaluate(strategy, prices, {"mode": list(range(7))},
                           cost_bps=0, frequencies=(1,))
    assert ev.best_config["mode"] in (4, 5, 6)


def test_pipeline_selects_using_doubled_costs_not_base_cost_peak() -> None:
    t = np.arange(1000)
    prices = {"A": 100 + 0.01 * t + 0.15 * (t % 2)}

    def strategy(prices: dict[str, np.ndarray], style: str) -> dict[str, np.ndarray]:
        t = np.arange(len(prices["A"]))
        return {"A": np.where(t % 2, 1.0, -1.0) if style == "fast" else np.ones(len(t))}

    ev = pipeline.evaluate(strategy, prices, {"style": ["fast", "slow"]},
                           cost_bps=5, frequencies=(1,))
    assert ev.best_config["style"] == "slow"


def test_direct_research_template_uses_gates_holdout_and_trial_ledger(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    csv = tmp_path / "prices.csv"
    p = 100 + np.cumsum(np.random.default_rng(82).normal(0, 0.1, 500))
    csv.write_text("timestamp,A\n" + "".join(f"{t},{v}\n" for t, v in enumerate(p)))
    runs = tmp_path / "runs"
    result = subprocess.run(
        [sys.executable, str(root / "templates/research_template.py"), str(csv),
         "--runs-dir", str(runs)], check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "G0 integrity" in result.stdout and "G5 holdout" in result.stdout
    assert "DECISION:" in result.stdout
    records = [json.loads(line) for line in (runs / "trials.jsonl").read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["n"] == 36  # 3 half-lives x 3 entries x 4 frequencies
