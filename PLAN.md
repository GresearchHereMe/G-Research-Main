# PLAN — G-Research Quant Challenge

Goal: maximise PnL in the G-Research cointegration (pairs trading) challenge on unseen real data,
with a backtester rigorous enough that what wins locally also wins on the night.
Strategy background lives in [PLAYBOOK.md](PLAYBOOK.md); process rules in [RULES.md](RULES.md).

**Event date (assumed, confirm):** London, Wed 4 Nov 2026, 17:30 → T-0. Today 27 Oct = T-8.
If the date differs, shift the T-minus labels; the order of phases stays.

---

## 1. Principles

1. **Pairs on real data first** (variant A in PLAYBOOK §1.4). Order-book/bot engine (variant B) is
   second and timeboxed.
2. **One strategy interface** used by research, both backtest engines and the pasted submission.
3. **Pessimistic by default**: costs, one-step execution delay, conservative fills. Report bands.
4. **Ground truth before real data**: every estimator is first tested on synthetic series with
   known parameters.
5. **Everything reproducible**: seeded, config-driven, every run logged to `runs/` (gitignored).
6. **Research ≠ production**: notebooks in `research/`, tested code in `src/`, frozen submissions in
   `submissions/`.

---

## 2. Repository layout (target)

```
src/gqc/
  data/        schema.py (frozen dataclasses), alpaca_io.py, synthetic.py, align.py
  stats/       coint.py (Engle–Granger both ways, Johansen), ou.py (half-life, OU fit),
               kalman.py (dynamic hedge ratio), ewma.py (Welford/EWMA), regime.py (break detectors)
  backtest/    bars_engine.py (variant A), costs.py, metrics.py, experiments.py (grid, walk-forward,
               Monte Carlo, plateau scoring), registry.py (runs/ log)
  book/        matching.py (price-time FIFO), fills.py (conservative/queue/optimistic),
               bots.py (synthetic bot agents), replay.py (variant B)
  strategies/  base.py (protocol), naive_z.py, distance.py, eg_static.py, kalman_pairs.py,
               ou_optimal.py, market_maker.py
  submit/      bundle.py (flatten to one file), templates/ (per platform archetype), dryrun.py
tests/         mirrors src/; tests/hooks/ already exists
research/      notebooks, event-notes.md
submissions/   round-N-vK.py frozen files + one-line notes
scripts/       fetch_data.py, run_backtest.py, leaderboard.py
```

---

## 3. Core interfaces (sketch)

```python
@dataclass(frozen=True)
class Observation:
    t: int                                  # step index; no wall-clock dependence
    mid: Mapping[str, float]
    bid: Mapping[str, float] | None = None  # optional top of book
    ask: Mapping[str, float] | None = None

@dataclass(frozen=True)
class Decision:
    target: Mapping[str, float]             # desired positions after this step (variant A)
    orders: tuple[OrderIntent, ...] = ()    # explicit orders (variant B)

class Strategy(Protocol):
    def init(self, params: Mapping[str, float]) -> State: ...
    def step(self, obs: Observation, state: State) -> tuple[Decision, State]: ...
```

State is an immutable value (frozen dataclass / tuple of deques snapshot) so runs are replayable
and the submission can serialise it to a string if the platform requires.

---

## 4. Phases

### P0 — Guardrails (T-8) ✅ this PR
Branch protection docs, Claude guard hook, git hooks, CI, PR template, playbook, plan.

### P1 — Foundations (T-8 → T-7)
- `pyproject.toml` (uv), `src/gqc` skeleton, ruff + pytest + coverage in CI.
- `data/synthetic.py`: cointegrated pairs `x` random walk, `s` OU(θ, μ, σ), `y = α + βx + s`;
  options for β drift, regime breaks (θ→0), jumps, heteroskedastic σ, bid–ask spreads.
- `data/alpaca_io.py` + `scripts/fetch_data.py`: real bars/quotes → `data/` (gitignored).
  Practice universe: share classes (GOOG/GOOGL, FOX/FOXA, NWS/NWSA, LBRDA/LBRDK, Z/ZG),
  ETF twins (SPY/IVV/VOO, GLD/IAU, QQQ/QQQM), economic pairs (KO/PEP, XOM/CVX, V/MA,
  EWA/EWC, GLD/GDX). Daily + 1-minute.

**Done when:** `uv run pytest` green in CI; synthetic generator unit-tested (sample θ, β recover
their true values within tolerance); ≥10 pairs × ≥2 timeframes cached locally.

### P2 — Stats core (T-7 → T-6)
EG test both orderings with MacKinnon p-values, Johansen, AR(1) half-life, OU MLE, Kalman hedge
ratio, EWMA z-score, break detectors (rolling ADF p-value, variance-ratio, CUSUM on spread).

**Done when:** each estimator recovers synthetic ground truth (β within 2%, half-life within 20%
for n ≥ 2000); break detector fires within N steps of an injected break with < 5% false alarms;
pure functions, O(1) per update for the online versions.

### P3 — Backtest engine A: bar/quote level (T-6 → T-4) — **critical path**
- Event loop over aligned observations; strategy sees data ≤ t; executes at t+1 (configurable
  0/1/2 lag to mirror the platform once known).
- Costs: per-unit half-spread (from quotes, else assumed bps) + fee + optional impact
  `k·|Δq|^1.5`; position limits (clip / reject / cancel-all policies); cash, positions, mark-to-mid.
- Metrics: PnL, Sharpe, Sortino, max drawdown, Calmar, VaR95 of step PnL, turnover, cost share,
  # round trips, holding time, hit rate, per-pair attribution, time in market.
- Experiments: parameter grids with **plateau score** (median PnL of the neighbourhood, not the
  max), walk-forward (formation → trading windows), Monte Carlo over synthetic seeds,
  bootstrap over pairs/days, deflated Sharpe across all tried configs; every run logged.
- Vectorised fast path (numpy) for screening + event-driven path for final numbers; a test
  asserts both agree on frictionless runs.

**Done when:** invariants hold in property tests (cash + Σ pos·price conserved minus costs;
position = Σ fills; limits never exceeded); golden scenarios with hand-computed PnL pass;
1 year of 1-min data for 10 pairs runs in < 30 s.

### P4 — Strategy library + leaderboard (T-5 → T-3)
1. `naive_z` — replica of the expected organiser example (static OLS β, full-sample z ±2): baseline.
2. `distance` — Gatev normalised-price distance method: robust fallback.
3. `eg_static` — EG-screened, log spread, EWMA z, exit at mean, time stop, break stop.
4. `kalman_pairs` — dynamic β, trade on normalised forecast error.
5. `ou_optimal` — Bertram-style thresholds from fitted OU + cost estimate.
6. Ensemble / sizing by confidence (p-value, half-life, recent fit quality).

`scripts/leaderboard.py` ranks strategies on: synthetic suite, real practice universe
(out-of-sample windows only), worst-case pair, cost sensitivity (×2 costs).

**Done when:** at least two strategies beat `naive_z` out-of-sample on real data at 2× costs
and on ≥80% of synthetic seeds; parameter plateaus documented.

### P5 — Backtest engine B: order book + bots (T-4 → T-2, timeboxed)
Price-time FIFO matching (limit/IOC/cancel, partial fills); fill models conservative / queue
(hftbacktest-style) / optimistic; latency; synthetic bots from PLAYBOOK §4.2 (noise taker,
stale quoter, momentum chaser, naive pairs bot, informed bot); replay mode for any captured
book data; markout analysis tooling.

**Done when:** matching engine passes golden order sequences; a dual-listing MM strategy is
profitable vs noise bots under the conservative model and unprofitable when bots are informed
(sanity check that the engine punishes adverse selection).

### P6 — Submission tooling + rehearsal (T-3 → T-1)
- `submit/bundle.py`: inline a strategy + its deps into **one file**, stdlib(+numpy) only; CI
  imports the bundle in a clean interpreter.
- Templates for archetypes A/B/C (PLAYBOOK §3) with try/except, limit guard, state serialisation.
- `submit/dryrun.py`: runs a bundle under time/memory budgets on a held-out dataset.
- **Rehearsal:** teammate prepares 3 unseen pairs (one non-cointegrated, one with a break);
  team runs a 2-hour mock event end-to-end using only the runbook.

**Done when:** rehearsal completed; post-mortem items fixed; `submissions/baseline.py` and
`submissions/best.py` frozen and dry-run clean.

### P7 — Event day (T-0)
Follow PLAYBOOK §9. Baseline submitted in the first 10 minutes; one change per iteration.

### P8 — Retro (T+1)
What the platform actually was, what transferred, update PLAYBOOK.

---

## 5. Tooling we use (skills, MCPs, agents)

| Need | Tool |
|---|---|
| Real market data (bars, quotes, trades) for practice sets | **Alpaca MCP** read-only data tools (`get_stock_bars`, `get_stock_quotes`, `get_stock_trades`, `get_crypto_bars`). Trading tools are denied in `.claude/settings.json`. Verified working on IEX feed 27 Oct 2026. |
| Library/API docs (statsmodels, numpy, scipy) | context7 MCP / `docs-lookup` agent |
| Web research | WebSearch, Exa MCP |
| Strategy/backtest checklists | `quant-analyst` skill |
| Code quality | `python-patterns`, `python-testing`, `tdd-workflow` skills; `python-reviewer` agent |
| Charts in research | `dataviz` skill |
| Before-merge checks | `verification-loop` skill, CI |

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| Platform differs from our assumptions | Archetype templates A/B/C; pure `step()` core; recon checklist at briefing |
| Overfitting to practice data | Plateau selection, OOS windows, synthetic suite, deflated Sharpe, few params |
| Cointegration breaks on unseen data | Online estimation + break detectors + time stops |
| Costs underestimated | Leaderboard at 2× costs; one-step delay default |
| Crash / limit breach on the night | Exception guard, hard limit guard, dry-run harness |
| Time pressure (≈2 h of play) | Frozen baseline, rehearsal, roles, one change per iteration |
| Engine B scope creep | Timeboxed; variant A is the critical path |

---

## 7. Open questions (answer at the briefing → `research/event-notes.md`)

See PLAYBOOK §9. Plus: exact event date and team size for our registration.
