"""Final order guards reject invalid batches and bound independently filling orders."""

from pathlib import Path

import pytest

SKELETON = Path(__file__).resolve().parents[1] / "templates" / "skeleton.py"


def _load() -> dict:
    ns: dict = {"__name__": "submission"}
    exec(compile(SKELETON.read_text(), "submission", "exec"), ns)  # noqa: S102
    return ns


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("with_caps", [False, True])
def test_nonfinite_order_rejects_the_whole_batch_before_clamping(bad, with_caps):
    ns = _load()
    if with_caps:
        ns["MAX_ORDER"]["A"] = 2
        ns["LIMITS"]["A"] = 10
    ns["decide"] = lambda book, positions, state: {"B": 1, "A": bad}
    assert ns["step"]({"A": 10, "B": 10}) == {}
    assert ns["STATE"].errors == 1


def test_gross_cap_reserves_capacity_for_resting_buys():
    ns = _load()
    ns["RISK"]["max_gross"] = 100
    ns["LIMITS"]["A"] = 100
    ns["decide"] = lambda book, positions, state: {"A": 10}
    assert ns["step"]({"A": 10}, {}, {"A": (10, 0)}) == {}


def test_gross_cap_does_not_net_opposite_resting_orders():
    ns = _load()
    ns["RISK"]["max_gross"] = 100
    ns["decide"] = lambda book, positions, state: {"A": 10}
    assert ns["step"]({"A": 10}, {}, {"A": (8, 8)}) == {"A": 2}


def test_gross_cap_does_not_fund_one_leg_with_an_unfilled_reduction():
    ns = _load()
    ns["RISK"]["max_gross"] = 100
    ns["decide"] = lambda book, positions, state: {"A": -10, "B": 10}
    # Selling A may not fill; buying B must therefore not consume A's capacity yet.
    assert ns["step"]({"A": 10, "B": 10}, {"A": 10}) == {"A": -10}


def test_gross_cap_keeps_reductions_when_existing_exposure_exceeds_cap():
    ns = _load()
    ns["RISK"]["max_gross"] = 100
    ns["decide"] = lambda book, positions, state: {"A": -5, "B": 1}
    assert ns["step"]({"A": 10, "B": 10}, {"A": 20}) == {"A": -5}


def test_gross_cap_counts_resting_orders_on_other_symbols():
    ns = _load()
    ns["RISK"]["max_gross"] = 100
    ns["decide"] = lambda book, positions, state: {"B": 10}
    assert ns["step"]({"A": 10, "B": 10}, {}, {"A": (8, 0)}) == {"B": 2}


def test_drawdown_exit_also_reserves_gross_for_resting_orders():
    ns = _load()
    ns["RISK"].update({"max_gross": 100, "max_drawdown": 5})
    ns["step"]({"A": 10}, {"A": 10}, account_pnl=10)
    # Existing sells can already take us to -10. Another flattening sell would reach -20.
    assert ns["step"]({"A": 10}, {"A": 10}, {"A": (0, 20)}, account_pnl=0) == {}
