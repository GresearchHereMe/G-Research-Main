"""Append-only trial ledger in runs/ (gitignored): every evaluation on a dataset counts.

The Deflated Sharpe needs the number of trials ever run on the same data, not just the
current grid. Each record stores the dataset fingerprint, a label, the number of configs
evaluated and the decision; `total_trials` sums them per fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

LEDGER = "trials.jsonl"


def fingerprint(path: str | Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:16]


def record(runs_dir: str | Path, fp: str, label: str, n_configs: int, decision: str) -> None:
    runs = Path(runs_dir)
    runs.mkdir(parents=True, exist_ok=True)
    entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "data": fp, "label": label,
             "n": int(n_configs), "decision": decision}
    with open(runs / LEDGER, "a") as handle:
        handle.write(json.dumps(entry) + "\n")


def total_trials(runs_dir: str | Path, fp: str) -> int:
    path = Path(runs_dir) / LEDGER
    if not path.exists():
        return 0
    with open(path) as handle:
        return sum(e["n"] for e in map(json.loads, filter(str.strip, handle)) if e["data"] == fp)
