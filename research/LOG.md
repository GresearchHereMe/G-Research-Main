# Research log — trial ledger

Every idea evaluated on event data gets a row. `N` = configs evaluated in that run (grid ×
frequencies × bands); the running total is what any Deflated Sharpe claim must account for
(AGENTS.md §10). Paste the `DECISION` line verbatim.

| Time | Round | File | Behaviour + mechanism | N | Running total | PBO | DSR | DECISION | Submitted as |
|---|---|---|---|---|---|---|---|---|---|
| 18:45 | 3 | client `solution.py` (band 8-16) | Exclusive claims: best response to the field fitted from the 18:31 server game (33 teams, 589 sole wins). P(exactly one claimant) ~0.37-0.42 for r in 2-14, falls to ~0.14 above 20; claim iff r*P(sole) > ~3.3 per 16-turn cycle -> band 8-16. Expected ~400/game (scenario range 320-530), plateau 8-14..9-16 within 2%; endgame DP tweak +0.5, dropped | 16 (12 bands + DP x 4 scenarios) | 16 | n/a | n/a | not a G0-G5 pipeline algo (game, not prices) | not submitted |
| 17:36 | 1 | client `solution.py` (empirical DP learner) | Finite-horizon optimal stopping on the empirical distribution of all observed rewards; 0.989-0.999 of oracle on 6 simulated distributions | 1 | 17 | n/a | n/a | not a G0-G5 pipeline algo (game, not prices) | submitted 17:36:06; same run 2522.38 vs 2561.52 for the exact-DP teams |
| 17:42 | 1 | client `solution.py` (exact Exp(10) DP) | Announced distribution plugged in: E[(R-k)+] = 10 exp(-k/10); sim +0.7% vs learner (200 seeds, better in 65%) | 1 | 18 | n/a | n/a | not a G0-G5 pipeline algo | submitted 17:42:36; tied the leaders on every run (2613.11, 2566.63, 2564.99); best stayed 2964.44 |
| 17:44 | 1 | scratch `sim_best.py` | Best-of-runs variants: fixed top-125 cut 27.7 (with/without DP endgame), DP thresholds x0.9 / 1.1 / 1.2; target P(score > 2976.91) | 5 | 23 | n/a | n/a | rejected: top-125 mean 2442 vs 2692, P(>target) 0.10% vs 1.65%; no scaling beat the optimum | not submitted |
| 17:55 | 2 | client `solution.py` (empirical DP learner) | Hidden distribution learned online | 1 | 24 | n/a | n/a | not a G0-G5 pipeline algo | submitted 17:55:55; 8 same-run ranks 9, 16, 1, 17, 3, 6, 11, 7 |
| 18:04 | 2 | client `solution.py` (normal, per-run mean) | Leaderboard bot jumps (1213 samples, 5 runs): N(-0.19, 0.97) with run-varying mean; conjugate update of the mean only, prior sd 0.15; sim +0.42 vs learner | 10 | 34 | n/a | n/a | rejected live: ranks 17, 16, 16 of 31; normal tail too heavy vs data (99%: 2.06 vs 1.96) | submitted 18:04:11, reverted to learner 18:10:27 |
| 18:13 | 3 | client `solution.py` (round-2 learner) | No collision logic | 0 | 34 | n/a | n/a | live 41.8, 51.2 | submitted 18:13:35 |
| 18:24 | 3 | client `solution.py` (contrarian v1) | Per-opponent claim model from lock flips in `other_scores`; claim iff r * P(sole) beats DP threshold; claim prob 0.75; sim 1827-1919 vs best crowd 955-987; 3 deterministic copies -> 0 | 4 | 38 | n/a | n/a | live 187.8 (14/31) | submitted 18:24:23 and 18:26:19 (overwritten once by a teammate) |
| 18:33 | 3 | client `solution.py` (hybrid) | Pooled opponent prior + JLDSnoAI's inferred rule (claim above running mean) gated by P(sole) >= 0.3; sim 612-776 vs exact copy 0 when JLDSnoAI present | 9 | 47 | n/a | n/a | live 0.0, 0.0 (lp_bot flood), 63.0 (26/30); P(sole) gate never opens in a crowded field | submitted 18:34:22 |
| 18:40 | 3 | client `solution.py` (band 8-16, row above) | Fixed band | 0 | 47 | n/a | n/a | live 0.0 (28/30): field had moved into the band | submitted 18:40:09 |
| 18:43 | 3 | client `solution.py` (crowd-absence table) | P(nobody claims given reward bin and free-team bin) learned in game from lock flips; DP on r * P | 0 | 47 | n/a | n/a | untested; live 0.0 most likely ours (18:43:51) | submitted 18:43:16, overwritten 18:44:08 |
