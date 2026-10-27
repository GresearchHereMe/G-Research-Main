# Research log — trial ledger

Every idea evaluated on event data gets a row. `N` = configs evaluated in that run (grid ×
frequencies × bands); the running total is what any Deflated Sharpe claim must account for
(AGENTS.md §10). Paste the `DECISION` line verbatim.

| Time | Round | File | Behaviour + mechanism | N | Running total | PBO | DSR | DECISION | Submitted as |
|---|---|---|---|---|---|---|---|---|---|
| 17:45 UTC | 1 | submissions/tradinggame_v1.py | Optimal stopping: finite-horizon DP threshold on the empirical reward distribution; share-aware value if takers split rewards (inferred from score jumps), so we take while greedy/random bots are locked | 1 | 1 | n/a | n/a | not the price pipeline: simulator (research/tradinggame_sim.py) + client test pass; ~pro bot when rewards are independent, rank 1/24 when shared | pending team submit |
