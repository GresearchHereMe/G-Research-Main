# Research log — trial ledger

Every idea evaluated on event data gets a row. `N` = configs evaluated in that run (grid ×
frequencies × bands); the running total is what any Deflated Sharpe claim must account for
(AGENTS.md §10). Paste the `DECISION` line verbatim.

| Time | Round | File | Behaviour + mechanism | N | Running total | PBO | DSR | DECISION | Submitted as |
|---|---|---|---|---|---|---|---|---|---|
| 17:45 UTC | 1 | submissions/tradinggame_v1.py | Optimal stopping: finite-horizon DP threshold on the empirical reward distribution; share-aware value if takers split rewards (inferred from score jumps), so we take while greedy/random bots are locked | 1 | 1 | n/a | n/a | not the price pipeline: simulator (research/tradinggame_sim.py) + client test pass; ~pro bot when rewards are independent, rank 1/24 when shared | pending team submit |
| 18:05 UTC | 1 | submissions/tradinggame_r1_v2.py | Round 1 rewards announced as exponential(mean 10): exact DP thresholds (c ~ 20.1, optimum 2690/game, sd 130); learning fallback if the mean is contradicted. Rejected: uniform threshold scaling (+13 at best for best-of-200) and target-seeking DP (no gain below a 3050 target) | 10 | 11 | n/a | n/a | sim: 2698 vs v1 2667 same seeds, rank 1/24 independent; unchanged when split | pending team submit |
| 18:05 UTC | 2 | submissions/tradinggame_r2_v2.py | Same code, KNOWN_MEAN = None (round 2 distribution not announced): learns the distribution online | 0 | 11 | n/a | n/a | client test pass | for round 2 |
