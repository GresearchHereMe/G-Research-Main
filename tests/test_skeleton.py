"""The paste-ready skeleton must exec standalone, never raise, and enforce limits."""

import sys
from pathlib import Path

SKELETON = Path(__file__).resolve().parents[1] / "templates" / "skeleton.py"


def _load() -> dict:
    ns: dict = {"__name__": "submission"}
    exec(compile(SKELETON.read_text(), "submission", "exec"), ns)  # noqa: S102 - mimics paste box
    return ns


def test_skeleton_execs_without_third_party_imports():
    before = set(sys.modules)
    _load()
    new = set(sys.modules) - before
    assert not [m for m in new if m.startswith(("numpy", "pandas", "gqc"))]


def test_step_accepts_flat_prices_and_books():
    ns = _load()
    assert ns["step"]({"A": 100.0, "B": 50.0}) == {}
    book = {"A": {"bid": 99.0, "ask": 101.0, "bid_size": 5, "ask_size": 7}}
    assert ns["step"](book) == {}
    assert ns["STATE"].mids["A"][-1] == 100.0


def test_clip_respects_position_limits_and_order_caps():
    ns = _load()
    ns["LIMITS"].update({"A": 10})
    ns["MAX_ORDER"].update({"A": 4})
    assert ns["clip_orders"]({"A": 50}, {"A": 8}) == {"A": 2}
    assert ns["clip_orders"]({"A": -50}, {"A": 8}) == {"A": -4}
    assert ns["clip_orders"]({"A": 3}, {"A": 10}) == {}


def test_drawdown_kill_switch_flattens_then_pauses():
    ns = _load()
    ns["RISK"].update({"max_drawdown": 5.0, "halt_steps": 2})
    ns["decide"] = lambda book, positions, state: {"A": 1}
    ns["step"]({"A": 100.0}, {"A": 10})
    orders = ns["step"]({"A": 99.0}, {"A": 10})  # loss 10 > 5 -> flatten
    assert orders == {"A": -10}
    assert ns["step"]({"A": 99.0}, {"A": 0}) == {}  # still halted
    assert ns["step"]({"A": 99.0}, {"A": 0}) == {}
    assert ns["step"]({"A": 99.0}, {"A": 0}) == {"A": 1}  # resumed


def test_gross_cap_scales_orders():
    ns = _load()
    ns["RISK"].update({"max_gross": 1000.0})
    ns["decide"] = lambda book, positions, state: {"A": 20}
    assert ns["step"]({"A": 100.0}, {"A": 0}) == {"A": 10}


def test_step_survives_exception_in_decide():
    ns = _load()

    def broken(book, positions, state):
        raise RuntimeError("boom")

    ns["decide"] = broken
    assert ns["step"]({"A": 1.0}) == {}
    assert ns["STATE"].errors == 1
