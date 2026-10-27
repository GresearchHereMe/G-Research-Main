"""Book diagnostics must compare actual timestamps, never per-symbol row numbers."""

import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np

from gqc import diagnostics as dg

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose.py"


def _diagnose(path, monkeypatch):
    reports = []
    price_report = dg.price_report

    def capture_report(prices, **kwargs):
        reports.append({s: p.copy() for s, p in prices.items()})
        return price_report(prices, **kwargs)

    monkeypatch.setattr(dg, "price_report", capture_report)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), str(path)])
    ns = runpy.run_path(str(SCRIPT))
    assert ns["main"]() == 0
    return reports


def test_book_diagnostics_align_asynchronous_prices_on_actual_time(tmp_path, monkeypatch):
    path = tmp_path / "book.csv"
    path.write_text(
        "time,symbol,bid,ask\n"
        "10,A,109,111\n3,B,200,202\n1,A,99,101\n"
        "10,B,209,211\n2,A,100,102\n2,B,199,201\n"
    )
    reports = _diagnose(path, monkeypatch)
    assert len(reports) == 1
    np.testing.assert_allclose(reports[0]["A"], [100, 101, np.nan, 110])
    np.testing.assert_allclose(reports[0]["B"], [np.nan, 200, 201, 210])


def test_book_without_timestamps_keeps_only_single_series_reports(
    tmp_path, monkeypatch, capsys
):
    path = tmp_path / "book.csv"
    path.write_text(
        "symbol,bid,ask\nA,99,101\nB,199,201\nA,100,102\n"
        "B,201,203\nA,102,104\nB,204,206\n"
    )
    reports = _diagnose(path, monkeypatch)
    assert [set(prices) for prices in reports] == [{"A"}, {"B"}]
    text = capsys.readouterr().out.lower()
    assert "skipping cross-asset" in text and "timestamp" in text


def test_interleaved_quotes_keep_observed_returns_without_fabricating_lead_lag(tmp_path):
    rng = np.random.default_rng(2026)
    a = 100 + np.cumsum(rng.normal(0, 0.4, 40))
    b = 200 + np.cumsum(rng.normal(0, 0.7, 40))
    path = tmp_path / "interleaved.csv"
    rows = ["time,symbol,bid,ask"]
    for t, (pa, pb) in enumerate(zip(a, b)):
        rows.extend([f"{2*t},A,{pa-0.1},{pa+0.1}", f"{2*t+1},B,{pb-0.1},{pb+0.1}"])
    path.write_text("\n".join(rows) + "\n")
    result = subprocess.run([sys.executable, str(SCRIPT), str(path)],
                            check=True, capture_output=True, text=True)
    behaviour = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 7 and fields[0] in {"A", "B"} and fields[1].isdigit():
            behaviour[fields[0]] = fields
    assert set(behaviour) == {"A", "B"}
    for fields in behaviour.values():
        assert int(fields[1]) == 39
        assert np.isfinite(float(fields[2]))
    assert " leads " not in result.stdout
    assert not result.stderr
