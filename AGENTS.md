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
| One dataset, many ideas | Don't fool ourselves | Current-grid PBO; DSR/MinBTL count recorded trials; disclose historical-return gaps (§6) |
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

## 1b. Strategy policy — what we trade, and what we don't

**Principle:** one explainable mechanism per algo, few parameters, every candidate through the
§5 pipeline. Nothing is pre-committed; `diagnose.py` decides which playbook below applies.
The edge in a cointegration game is the **spread** returning to equilibrium, not forecasting
single prices (close to a random walk).

### Strategies we use (in priority order)

| # | Strategy | Trigger from `diagnose.py` / `screen_pairs.py` | Implementation | Key rules |
|---|---|---|---|---|
| 1 | **Spread mean reversion** (pairs, baskets, dual listings) | `TRADE` verdict: cointegrated in dev and later window, β > 0, return corr > 0.3, sensible half-life | `src/gqc/core.py` (`PairsTrader`, `PairBook`), `sweep_pair.py`, `make_submission.py` | Log spread; hedge = Kalman (default) / EWM / static fitted on formation window; enter \|z\| > entry, exit at mean, stop \|z\| > 4, time stop ≈ 3 half-lives; trade on executable prices; hedge on **filled** qty |
| 2 | **Single-asset mean reversion** | VR20 < 0.8 and negative lag-1 autocorr on aligned mids | `templates/research_template.py` (EWM fade) | z vs EWM mean; check bid/ask bounce is not the cause |
| 3 | **Lead–lag** | `A leads B by k` with \|corr\| ≥ 0.1 | research template: position in B from A's last k-step move | Must survive lag + 1 (G2); dies fast if the lag is execution-only |
| 4 | **Microstructure / market making** (only if there is a book) | imbalance → mid corr ≥ 0.1, wide spread, stale quotes | skeleton `decide()` with `normalise()` microprice | Fair = microprice/wall mid; skew quotes by inventory; take stale quotes only beyond cost |
| 5 | **Counterparty exploitation** (only if trades/IDs exist) | markout < 0 for a party (they lose) or > 0 (informed) | skeleton `decide()` | Quote to / trade against losers; follow informed; check several horizons |
| 6 | **Signal combination** | ≥ 2 of the above are individually weak but present | ridge / lasso on ≤ 5 causal features fitted on the dev window; export coefficients as constants | Time-ordered fit only; counts as trials; still through G0–G5 |

Run all GO edges at the same time across products (capacity is per product); one shared
risk gate, net positions per symbol.

### Model allow-list

| Model family | Status tonight | Why |
|---|---|---|
| Kalman / state-space hedge, OU half-life, EWM mean/vol | **Use** (built) | Directly models the cointegrating spread; O(1), stdlib, pasteable |
| OLS / Engle–Granger / Johansen | **Use** for screening and formation fits | Standard, fast, interpretable |
| Ridge / Lasso | **Allowed** for combining ≤ 5 signals | Closed-form, regularised, exports as plain coefficients |
| EWM volatility (instead of GARCH) | **Use** | Same job, no fitting risk |
| ARIMA / GARCH full fits | Skip | Add parameters without adding edge here |
| HMM regime models | Skip | Break detectors + stops already cover regime changes |
| Random forest / LightGBM / XGBoost | **Not tonight** | Small data, overfit, not pasteable without libraries |
| FNN / LSTM / Transformers / any deep learning | **Never tonight** | Thousands of rows at low signal-to-noise; hundreds of trials sink the Deflated Sharpe; paste box is likely stdlib/numpy; no time to validate |

Reconsider ML only if the briefing reveals an **offline prediction task with a large,
feature-rich dataset and installable libraries**: ridge first, LightGBM second, time-ordered
cross-validation, and the full gate pipeline. The trial budget is the reason: with N trials the
data needed to beat luck grows as ≈ 2 ln N / SR² (300 trials at Sharpe/step 0.05 ⇒ ~4,600 steps).

### Never (strategy level)

- Trade a pair on correlation alone, or without an economic reason to be related.
- Average down / add to a losing spread because it is "more extreme"; no martingale sizing.
- Full Kelly, or sizing from in-sample returns.
- Hold a single unhedged leg after a partial fill — hedge the filled quantity or unwind.
- Exploit platform bugs or timing artefacts without organiser approval.
- Put an unvalidated "hunch" trade on at size, or tune on the holdout.
- Forecast single prices and call it a strategy without a mechanism and a passing G0–G5.

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
src/gqc/pipeline.py             gates G0-G5, plateau/frequency/band search, GO / GO-REDUCED / NO-GO
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

### 5.1 Developing algorithms: baselines first, optional ML second

Start with a mechanism found in S0 and a working baseline. Model complexity earns its place
through chronological, costed trading results. Price-level R² or directional accuracy alone
does not establish an edge. Four event hours include briefing, integration and networking;
do not allocate the whole window to model development.

| Candidate | Event-time priority | Conditions |
|---|---|---|
| Existing static/EWM/Kalman pairs strategy | First baseline when the data supports a spread relationship | Fit/screen pairs on past data only; check stability and costs |
| Ridge or a small autoregression | First supervised challenger | Few causal features; training-only preprocessing; export coefficients |
| Shallow XGBoost regression | Feasible optional nonlinear challenger | Offline CPU training, honest time splits, bounded search and verified submission export |
| Tiny feed-forward network, e.g. hidden layers 16/8 | Optional prepared experiment | Fixed small architecture and seeded training; compare against ridge and the baseline |
| LSTM/Transformer or a large ensemble | Defer during the event | Reconsider only with an already tested pipeline, suitable history and enough validation time |

Three trained models or a deep-learning component are not requirements. ARIMA, GARCH and
HMMs need a specific forecasting, volatility or regime hypothesis; do not add them merely
to increase the model count. Correlated models are not independent evidence, and an ensemble
is another configuration to validate and count.

**XGBoost starting budget (team heuristic, not winning parameters):** CPU `gbtree`,
`tree_method="hist"`, squared-error regression; try depth 2 or 3 with one fixed learning
rate and regularization setting. Cap boosting rounds around 100–200, using an earlier
chronological validation block for early stopping. Do not turn every parameter into a grid
axis. Start with roughly 5–15 interpretable features; measure actual training/inference time.
Allocate at most one 20–30 minute ML comparison after a baseline is accepted, unless the
round schedule leaves less time. Drop the experiment if fitting, export or validation
threatens the submission deadline. These models are research directions, not installed
tooling; their names do not relax dependency or pasted-file restrictions.

**Training and evaluation contract:**

1. Record the available history, timestamp spacing, allowed libraries, execution lag, fees
   and target horizon before fitting. Many adjacent ticks or overlapping labels do not
   imply many independent observations. With insufficient history for chronological
   training and evaluation, keep the simple/online baseline; there is no universal row
   count that makes deep learning safe.
2. Predict an executable-horizon return or spread change. For a spread label, freeze the
   hedge ratio at the prediction time rather than fitting it on the future label window.
   Start with spread z-score, recent spread/leg returns and rolling volatility; add book
   imbalance only if available and correctly timestamped. Select features/pairs and fit
   scalers, imputers or clipping thresholds using the training prefix only.
3. Declare formation/training, development folds and a final look-once holdout. Train on
   earlier observations; validate on later observations. Purge training examples whose
   label endpoints reach the validation period, including execution lag and target horizon.
   Use timestamps rather than a guessed row gap for irregular data. Random train/test
   splits and shuffled cross-validation are unsuitable here. Early stopping must never
   observe the final holdout; do not re-select after reading its result. Early stopping and
   iteration selection must finish before each scored prediction block begins; the block
   used to choose the stopping iteration is not independent performance evidence.
4. Implement the fit/predict boundary explicitly: `evaluate_algo.py` splits PnL, but does
   **not** construct estimator training folds. `strategy(prices, **params)` must generate
   out-of-sample scored positions using earlier completed labels, with neutral positions
   during formation. Freeze each fold's fitted model, or declare deterministic retraining
   steps using only labels available then. Fitting on the full array and predicting that
   array is leakage. Test the entire fitting/preprocessing path with prefix/future
   perturbations; G0 on already frozen weights cannot prove where those weights came from.
5. Compare candidates with the same information, horizon, execution and cost assumptions.
   Select on development plateaus; inspect net PnL, Sharpe, drawdown, turnover and stability,
   then apply S2 and the final holdout. Log model families, feature sets, horizons, seeds,
   retraining schedules and manual changes as trials. Trade only when predicted gain in
   the portfolio's units exceeds round-trip costs plus a declared uncertainty buffer.
   Translate predictions through the shared order/risk guards; a prediction is not an order.
6. Prove deployment before expanding the search. Research may train with separately
   approved libraries, but the pasted file follows §13: embed coefficients/weights/tree
   constants and stdlib inference unless the briefing explicitly permits more. Native
   XGBoost JSON or a pickle is not a stdlib predictor, and `make_submission.py` does not
   bundle ML models. A custom tree predictor must match native predictions for feature
   order/dtype, thresholds, missing-value branches, fitted base score, objective and selected
   boosting rounds. Early-stopped sklearn XGBoost predictions use the best iteration;
   native/default full-model prediction can differ. Check prediction and position parity,
   including boundary/missing inputs, then bare-exec and replay the exact final adapter (S3).

Primary references: [XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html),
[early stopping and prediction](https://xgboost.readthedocs.io/en/stable/python/python_api.html),
[model IO versus dumps](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html),
[chronological splits with a gap](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).

## 6. Gates and decisions (`src/gqc/pipeline.py`)

| Gate | Test | Type |
|---|---|---|
| G0 integrity | No lookahead (future prices perturbed at 3 cut points, ± directions → past positions unchanged), deterministic, finite positions | hard |
| G1 overfit | PBO over the current grid × frequencies × bands; DSR/MinBTL include recorded prior trial counts → OK / CAUTION / REJECT | hard if REJECT |
| G2 robustness | Best config still Sharpe > 0 and ≥ 50 % of base at **2× costs** and at **lag + 1**; total PnL > 0 | hard |
| G3 stability | Profitable in ≥ 3 of 4 time slices | soft |
| G4 risk | Development PnL ≥ 2 × max drawdown | soft |
| G5 holdout | Selected config profitable on the untouched last `1 − train_frac` (default 30 %) — look once, never re-select after reading it | soft |

Selection, PBO/DSR and G1–G4 use only the development window. G0 inspects the **raw**
strategy output. Every run of `evaluate_algo.py` / `sweep_pair.py` appends to
`runs/trials.jsonl`; later runs on the same data file add those trials to N automatically.
Selection uses median development Sharpe across immediate numeric-grid neighbours at **2×
costs**; categorical settings stay fixed, and ties prefer the candidate's own stressed Sharpe.
The direct research-template command delegates to the same evaluator. PBO covers only the
current return matrix: the ledger stores counts, not historical returns. DSR uses the current
grid's Sharpe dispersion with the accumulated count. Disclose these limits; a count alone
does not establish an all-history PBO. Manual trials and overlapping/reformatted datasets
still require the research log.

**Decision → size:** hard fail → **NO-GO, 0 %**. CAUTION or a soft fail → **GO-REDUCED, 50 %
of limits**. All pass → **GO, 100 %**. Thresholds: PBO ≤ 0.2 & DSR ≥ 0.95 & T ≥ MinBTL = OK;
PBO ≤ 0.5 & DSR ≥ 0.5 = CAUTION; else REJECT. PBO on one sample is noisy — it never decides
alone (PLAYBOOK §7.1).

## 7. Risk management

1. **Pre-trade:** only GO / GO-REDUCED algos trade, at `SIZE_FRACTION` from S2.
2. **In code (skeleton):** `LIMITS` per symbol checked worst-case (current position **plus all
   resting orders** via `step(..., outstanding={sym: (buys, sells)})`), `LOTS` (orders round
   toward zero), `MAX_ORDER` per step, `RISK.max_gross` (reserves capacity for independently
   filling resting/new orders; an unfilled reduction cannot fund another leg),
   `RISK.max_drawdown` (flatten through the same order guard until flat, then pause
   `halt_steps`); pass `account_pnl=` when the platform reports PnL. `step()` never raises;
   on error it sends **no orders** (never stale or zero targets); errors in `STATE.errors`.
   Pairs bundles: `on_tick(prices, positions)` holds positions on error; `to_orders()` is
   lot- and resting-order-aware; `step()` exists so `replay_skeleton.py` can replay them.
   Size check: legs must be ≥ 1 lot (pairs `unit_notional` large enough) or nothing trades —
   `replay_skeleton.py` warns on zero trades and fails on limit breaches, recorded strategy
   errors, or non-finite orders/PnL/positions. `SIZE_FRACTION` remains a manual target-sizing
   instruction; setting the constant alone does not scale orders.
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
