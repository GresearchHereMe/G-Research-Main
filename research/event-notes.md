# Event notes — briefing answers (PLAYBOOK §9)

Fill during the briefing, in our own words. Do not paste organiser documents or code here
(RULES.md §4). Each answer maps to a setting; set it before the first submission.

| # | Question | Answer | Sets |
|---|---|---|---|
| 1 | Interface: paste box? function name/signature? what comes in, what goes out (orders or target positions)? | | skeleton `step()` vs `step_targets()`; pairs `step()` vs `on_tick()`; ADAPTER block |
| 2 | Allowed imports (numpy? pandas?), time/memory limit per step | | stdlib-only paste? XGBoost export path (AGENTS §5.1 point 6) |
| 3 | Instruments, which are related, tick size, lot size, position limits, fees | | `LIMITS`, `LOTS`, `--cost-bps` |
| 4 | Matching: when do our orders execute? Do resting orders persist across steps? | | `--lag`, `outstanding=` |
| 5 | Scoring: PnL marked to what? Positions force-closed? Per round or cumulative? Round length in steps? | | `flatten_after` = steps − lag − margin, or `None` |
| 6 | Can we see bot IDs / trade history / our fills? Practice round with logs? | | `diagnose.py --trades` |
| 7 | Submissions allowed; resubmit mid-round? | | cadence, last-known-good plan |
| 8 | What is explicitly disallowed? | | RULES.md §7 |

## Derived settings (fill before the first paste)

```text
Interface style:        orders | target positions        entry point:
LIMITS:                 {}
LOTS:                   {}
MAX_ORDER:              {}
cost bps / lag:         /
SIZE_FRACTION:          (from evaluate_algo.py DECISION)
flatten_after:          (steps − lag − margin, or None)
RISK.max_drawdown:      (≈ 2 × backtest max DD × SIZE_FRACTION)
```

## Rounds

| Round | Data file | Steps | Notes |
|---|---|---|---|
| | | | |
