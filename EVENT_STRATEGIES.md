# Event strategies — G-Research Trading Game (ETH Zürich, 27 Oct 2026)

## Round 3: submit exactly this file, once, from ONE person

```bash
cp submissions/event/round3_v9_band_RECOMMENDED.py ../gr_tradinggame_client/solution.py   # adjust path
cd ../gr_tradinggame_client && uv run python test.py && uv run python submit.py
```

Round 3 ranks the **latest** submission. Every resubmission by anyone on the team replaces the
previous one. Agree on a single submitter.

### Why the band strategy (`claim iff 9 <= reward <= 16`, never while locked)

Rules: exactly one accepter gets the reward; two or more accepters all score zero; all accepters
are locked out for 15 turns; rewards ~ Exponential(mean 10).

Observed field on the server (round-3 runs, 20-28 teams):
- About a dozen teams still use the round-1 optimal cut (~20). They collide on large rewards
  and score almost nothing (0-100).
- Near-greedy teams take small rewards (median payout < 4).
- Winners take mid-size rewards (Crack(ed)Traders median 12, JLDSnoAI median 14.5).

Against a replica of that field the 9-16 band averages ~500 per game, rank 1-2 of 28, and the
result is flat across lower bound 8-12 / upper bound 14-20 (not a fragile optimum).

### What did NOT work on the server (kept for reference only)

- `round3_v7_rival_model_REFERENCE.py` and `round3_v8_aggregate_REFERENCE.py` model rivals
  from `other_scores` (who is locked, who claimed). In simulation they rank #1, but on the server
  they scored 6-51: the lock information in `other_scores` was not usable as assumed (most
  likely the list order changes between turns). Do not submit them.

## Rounds 1-2 (finished; best score counted)

- `round1_v3_exp10_optimal.py`: exact finite-horizon optimal stopping for Exponential(10)
  (accept iff reward >= threshold from backward DP; mid-game threshold ~20). Mean 2691 =
  theoretical optimum. Reached #3-4 by best score.
- `round2_v6_normal_prior.py`: hidden distribution learned online (empirical DP), plus an
  N(0,1) prior confirmed from earlier runs. Finished #1 with best 142.90.

## Lessons

1. Read the scoring rule first: best-of rounds reward variance; latest-score rounds reward safety.
2. Validate against the real server early; a simulator that matches the rules can still miss
   data-format details (order of `other_scores`).
3. One submitter per team.
