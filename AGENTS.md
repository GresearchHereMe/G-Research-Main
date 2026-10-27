# AGENTS.md — operating manual for every agent in this repo

Any AI agent (Claude Code, Codex, Cursor, Copilot, …) working here follows this file.
Claude-specific extras: [CLAUDE.md](CLAUDE.md). Binding team rules: [RULES.md](RULES.md).
Background and math: [PLAYBOOK.md](PLAYBOOK.md). Roadmap: [PLAN.md](PLAN.md).

---

## 1. Mission: what the event rewards and how we exploit it

**Event:** G-Research Quant Challenge, ETH Zürich, Tue 27 Oct 2026, 17:30–21:30. A
"cointegration challenge" on **unseen, real-world data**, Python, rounds explained at the briefing,
roughly 2 h of actual play. Interface, scoring, limits and fees are **unknown until the briefing**
(questions: PLAYBOOK §9).

**Objective:** highest final PnL. We **select on Sharpe, then size to the limits**
(max PnL per unit of risk; selecting on raw PnL rewards risk-taking and overfits faster).

| Event structure | Lever | How / tool |
|---|---|---|
| Unseen data, short time | Speed of the observe→validate loop | `diagnose.py` → research template → `evaluate_algo.py` (§5) |
| One dataset, many ideas | Don't fool ourselves | PBO + Deflated Sharpe gate over **all** trials (§6) |
| Per-product position limits | **Use all capacity**: run every validated edge on every product at once; uncorrelated edges add PnL linearly, Sharpe ≈ √k | §10 |
| Limits bind, not risk appetite | Validated edge → trade at **full limits** | sizing policy §8 |
| Costs / spread / fill rules | Trading frequency + no-trade band chosen on 2× costs | §9 |
| Fills at mid or zero fees (confirm!) | Trade more often, smaller thresholds — large PnL lever | §9 |
| Cointegration theme | Spread trading, online hedge ratio, break detection | `src/gqc/core.py`, PLAYBOOK §5 |
| Bots in a book (if any) | Markouts: trade against losers, follow winners | `diagnose.py --trades`, PLAYBOOK §4 |
| Rounds / regimes change | Re-diagnose every round; never carry parameters blindly | §5 S0 |
| Rank-based prizes (tournament) | Leading → cut size; behind late → full size on validated edges only | §8 |
| End-of-round marking | Flatten if marking/liquidation is punitive | ask at briefing |

---

## 2. Setup and commands

```bash
uv sync && uv run pytest -q && uv run ruff check src scripts tests templates .claude/hooks
git config core.hooksPath .githooks                                  # once per clone

uv run python scripts/diagnose.py DATA.csv [--trades TRADES.csv]     # S0 observe
cp templates/research_template.py research/<idea>.py                 # S1 prototype
uv run python scripts/evaluate_algo.py research/<idea>.py DATA.csv \
    --cost-bps C --lag L --frequencies 1,2,5,10 --bands 0,0.5        # S2 gates -> DECISION
uv run python scripts/replay_skeleton.py submissions/rN-vK.py DATA.csv --cost-bps C   # S3 check port
# pairs-specific
uv run python scripts/screen_pairs.py DATA.csv
uv run python scripts/sweep_pair.py DATA.csv --y A --x B
uv run python scripts/make_submission.py --config cfg.json --out submissions/rN-vK.py
# rehearsal data
uv run python scripts/make_practice_data.py data/practice_synth.csv
uv run python scripts/make_practice_data.py data/practice_book.csv --book
```

## 3. Repository map

```
templates/skeleton.py           paste-ready shell: state, LIMITS, SIZE_FRACTION, RISK gate, never-raise
templates/research_template.py  strategy(prices, **params) + GRID; prints ranking + overfit verdict
src/gqc/pipeline.py             gates G0-G4, frequency/band search, GO / GO-REDUCED / NO-GO
src/gqc/overfit.py              PBO (CSCV), PSR, Deflated Sharpe, expected max Sharpe, MinBTL
src/gqc/research.py             grid runner + positions_pnl + overfit_report
src/gqc/replay.py               replays a pasted file (bare exec) over prices
src/gqc/diagnostics.py          variance ratio, lead-lag, imbalance, staleness, markouts
src/gqc/core.py                 stdlib pairs trader (Kalman/EWM/static hedge) + PairBook
src/gqc/stats.py backtest.py sweep.py io.py synthetic.py
scripts/                        CLIs listed in §2
research/                       one file per idea + LOG.md trial ledger (create at the event)
submissions/                    frozen pasted files, one-line note each
```

---

## 4. Event-time roles and rhythm

**Driver** owns the pasted file and submissions. **Analyst** runs S0–S2 on new data/ideas.
**Risk/ops** owns S3/S5: limits, `SIZE_FRACTION`, `RISK`, last-known-good, kill decisions.

Rhythm per round: briefing answers → baseline submitted within 10 min (flat or tiny size if
nothing validated) → S0–S4 loop → **one change per resubmission** → log every trial.

---

## 5. Algo pipeline (every algo, every time)

| Stage | Do | Exit criterion |
|---|---|---|
| **S0 Observe** | `diagnose.py` on the round's data; write the hypothesis + *mechanism* in `research/LOG.md` | A named behaviour (PLAYBOOK §9 cheat sheet) |
| **S1 Prototype** | Copy `research_template.py`; `strategy()` causal (pos[t] uses data ≤ t); coarse `GRID` (≤ 3 knobs × ≤ 4 values); derive what you can (half-life, cost-based entry) instead of tuning | Runs; numbers plausible |
| **S2 Evaluate** | `evaluate_algo.py` with the briefing's costs/lag (unknown → assume crossing half the spread per trade, lag 1) | `DECISION` ≠ NO-GO |
| **S3 Port** | Move the rule into `templates/skeleton.py` `decide()`; set `LIMITS`, `SIZE_FRACTION`, `RISK`; `replay_skeleton.py` | Replay PnL same sign and order of magnitude as S2; 0 errors; positions within limits |
| **S4 Submit** | Copy to `submissions/round<N>-v<K>.py` + one-line note; paste | Confirmed accepted by platform |
| **S5 Monitor** | Risk/ops watches live PnL vs backtest; skeleton auto-flattens on `RISK.max_drawdown` | Revert to last-known-good if 2 checks in a row underperform |
| **S6 Iterate** | Change one thing; back to S2. Every new variant adds trials (N) | — |

Agents never skip S2 or S3. A strategy that has not passed G0 **never** trades.

## 6. Gates and decisions (`src/gqc/pipeline.py`)

| Gate | Test | Type |
|---|---|---|
| G0 integrity | No lookahead (future prices perturbed at 3 cut points, ± directions → past positions unchanged), deterministic, finite positions | hard |
| G1 overfit | PBO via CSCV, Deflated Sharpe, MinBTL over all configs × frequencies × bands → OK / CAUTION / REJECT | hard if REJECT |
| G2 robustness | Best config still Sharpe > 0 and ≥ 50 % of base at **2× costs** and at **lag + 1**; total PnL > 0 | hard |
| G3 stability | Profitable in ≥ 3 of 4 time slices | soft |
| G4 risk | Development PnL ≥ 2 × max drawdown | soft |
| G5 holdout | Selected config profitable on the untouched last `1 − train_frac` (default 30 %) — look once, never re-select after reading it | soft |

Selection, PBO/DSR and G1–G4 use only the development window. G0 inspects the **raw**
strategy output. Every run of `evaluate_algo.py` / `sweep_pair.py` appends to
`runs/trials.jsonl`; later runs on the same data file add those trials to N automatically.

**Decision → size:** hard fail → **NO-GO, 0 %**. CAUTION or a soft fail → **GO-REDUCED, 50 %
of limits**. All pass → **GO, 100 %**. Thresholds: PBO ≤ 0.2 & DSR ≥ 0.95 & T ≥ MinBTL = OK;
PBO ≤ 0.5 & DSR ≥ 0.5 = CAUTION; else REJECT. PBO on one sample is noisy — it never decides
alone (PLAYBOOK §7.1).

## 7. Risk management

1. **Pre-trade:** only GO / GO-REDUCED algos trade, at `SIZE_FRACTION` from S2.
2. **In code (skeleton):** `LIMITS` per symbol checked worst-case (current position **plus all
   resting orders** via `step(..., outstanding={sym: (buys, sells)})`), `LOTS` (orders round
   toward zero), `MAX_ORDER` per step, `RISK.max_gross` (risk-reducing orders always allowed),
   `RISK.max_drawdown` (flatten through the same order guard until flat, then pause
   `halt_steps`); pass `account_pnl=` when the platform reports PnL. `step()` never raises;
   on error it sends **no orders** (never stale or zero targets); errors in `STATE.errors`.
   Pairs bundles: `on_tick(prices, positions)` holds positions on error; `to_orders()` is
   lot- and resting-order-aware; `step()` exists so `replay_skeleton.py` can replay them.
   Size check: legs must be ≥ 1 lot (pairs `unit_notional` large enough) or nothing trades —
   `replay_skeleton.py` warns on zero trades and fails on limit breaches.
3. **Setting the drawdown stop:** `max_drawdown ≈ 2 × backtest max DD × SIZE_FRACTION` (in the
   platform's PnL units). Tighter stops whipsaw mean-reversion strategies.
4. **Live:** risk/ops can veto any submission; a kill means flat + last-known-good resubmitted.
5. **End of round:** if positions are marked at bid/ask or force-closed, flatten before the end.

## 8. Sizing and leverage policy

- **Unit position limits (most likely):** "leverage" = fraction of the limit used. GO → 100 %;
  GO-REDUCED → 50 %; optionally scale by signal strength `min(1, |z| / z_full)`.
- **Cash / margin constraint instead:** half-Kelly on **out-of-sample** estimates,
  `f = 0.5 · μ / σ²` per unit capital, never above limits. In-sample μ is inflated by selection
  (that's what DSR measures) — never full Kelly.
- **Several edges:** equal risk per uncorrelated edge (PnL correlation < 0.5); correlated edges
  count as one and share one risk budget.
- **Tournament:** leading → GO-REDUCED sizing on everything; behind in the last round → full
  size on GO edges only. Never add variance with unvalidated ideas.

## 9. Trading-frequency policy

- Frequency (`every`) and no-trade `band` are searched inside S2; pick from the frequency table
  at 2× costs, preferring a plateau over the single best value.
- Only trade when expected move > round-trip cost: `entry_z · σ_signal > 2 · (half_spread + fee)`.
- Fills at mid / zero fees → lower thresholds, `every = 1`. Crossing a wide spread → fewer, larger
  moves; post passively on the wide leg, hedge on the liquid leg (PLAYBOOK §5.1).
- Rate limits or order caps → `MAX_ORDER`, spread big changes over several steps.

## 10. Testing many algos at once

- One file per idea: `research/<behaviour>-<variant>.py`; record each run's `DECISION` line and
  N configs in `research/LOG.md`. **Total trials today = sum of N** — report it with any claim.
- Different products → run all GO algos together. Same product → combine target positions in one
  `decide()` (sum, then clip); check PnL correlation first.
- Independent ideas can be evaluated in parallel by separate agents/worktrees (§11), but only the
  Driver edits the file that gets pasted.

---

## 11. Skills, agents and tools — when to use what

Skills are Claude Code names; other agents follow the same steps manually.

| Situation | Use | Notes |
|---|---|---|
| New idea or feature, before coding | `superpowers:brainstorming` | At the event: 2-minute version — name behaviour, mechanism, test |
| Multi-step build (pre-event) | `superpowers:writing-plans` → `superpowers:executing-plans` or `superpowers:subagent-driven-development` | Not during a round |
| Writing shared code (`src/`, `templates/`, `scripts/`) | `superpowers:test-driven-development`, `tdd-workflow`, `python-patterns`, `python-testing` | Research scripts are exempt from TDD, **not** from G0 |
| Strange result / bug | `superpowers:systematic-debugging` | Before changing parameters |
| Several independent ideas | `superpowers:dispatching-parallel-agents` | One idea per agent, one research file each |
| Experiments that touch shared code | `superpowers:using-git-worktrees` | Keeps `main` and the Driver's file clean |
| Strategy / backtest checklist | `antigravity-quant-analyst` (listed as quant-analyst) | Costs, slippage, OOS, risk metrics |
| Before saying "done" / before a PR | `superpowers:verification-before-completion`, `verification-loop` | Tests + lint + gate output pasted |
| Code review | `superpowers:requesting-code-review`, `python-reviewer` agent, `code-review` | Reviewer ≠ author |
| Responding to review | `superpowers:receiving-code-review` | Verify before agreeing |
| Wrapping up a branch | `superpowers:finishing-a-development-branch` | Choose "open PR"; **never merge** |
| Charts in research | `dataviz` | Price/spread/equity plots |
| Library/API questions | `documentation-lookup` (context7) | statsmodels, numpy, scipy |
| Web research | `deep-research`, `exa-search`, `search-first` | Cite in PLAYBOOK |
| Real market data for practice | Alpaca MCP **read-only** (`get_stock_bars`, `get_stock_quotes`, `feed: "iex"`) | Trading tools are denied |
| Options/vol instruments appear | `lseg:option-vol-analysis` (needs LSEG auth) else PLAYBOOK math | Unlikely tonight |

**Do not use for this project:** `finance:*` (accounting: journal entries, reconciliation, SOX),
`academic-research-skills:*` (paper writing), `sp-global:*` / `bigdata-com:*` (fundamentals),
brokerage order tools of any MCP.

---

## 12. Git rules (hard)

- **Never commit to or push `main`.** Branch first: `git switch -c <type>/<topic>`
  (`feat/ fix/ research/ exp/ docs/ chore/`).
- Push only your own branch; open a PR. **Never merge, never approve**, never force-push
  (`--force-with-lease` on your own branch only), never `--no-verify`, never weaken hooks.
- Conventional commits: `feat|fix|refactor|docs|test|chore|perf|ci: <description>`.

## 13. Code rules

- Python ≥ 3.11 for `src/` and scripts. **Files that get pasted** (`templates/skeleton.py`,
  generated submissions, `src/gqc/core.py`) are stdlib-only, Python 3.8+, `NamedTuple` not
  `@dataclass`, and must survive bare `exec()`.
- Causality: position at t uses data ≤ t (G0 checks it). Seed randomness. No wall-clock logic.
- Type hints, functions < 50 lines, files < 800 lines; `uv run pytest` and ruff green before a PR.

## 14. Research rules

- Select on **plateau of Sharpe**, never on test PnL; then size to limits (§8).
- Report PBO, DSR, MinBTL and total trials with every claim; obey the verdict.
- Headline numbers use realistic costs and lag ≥ 1. Correlation ≠ cointegration.
- Never commit data (`data/` is gitignored) or organiser material.

## 15. Never

- Trade a NO-GO algo, or anything that skipped G0.
- Place orders through any brokerage API or MCP.
- Commit secrets.
- Paste a file that has not been replayed (`replay_skeleton.py`) or built by `make_submission.py`.
