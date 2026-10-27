# WINNING — a practical guide for our first quant challenge

Our best chance is to find a repeatable reason for someone to trade at a bad price, measure
what we can actually earn after execution costs, and submit a reliable implementation. A
complicated model earns its place only when it improves that evidence.

This guide teaches the decisions behind the commands in [PLAYBOOK.md](PLAYBOOK.md). Read
[RULES.md](RULES.md) for binding team policy. The repository review below covers the working
tree on **27 October 2026**, reviewed against `main` at `660e7c5`, including the new
pipeline, replay, and skeleton risk controls. Findings are bounded to this inspected version;
they are not fixes implemented by this document.

The [official event page](https://www.gresearch.com/events/eth-zurich-quant-challenge-2026/)
confirms a cointegration challenge on unseen real-world data, 17:30–21:30. It does not establish
the API, fees, matching rules, or scoring details. Bot and order-book tactics below are
conditional preparation, not confirmed features of this event.

**If preparation time is short:** read sections 1, 4, 8, and 9. Rehearse one complete submission
before adding another strategy.

## 1. Learn the game before choosing the trade

Write down the answers at the briefing. A wrong answer here can invalidate every backtest.

| Question | Why the answer changes our decisions |
|---|---|
| What exactly is scored: final PnL, risk-adjusted return, ranking, or several rounds combined? | This determines the objective and whether late-round risk helps or hurts. |
| Do we return desired positions or new orders? | Returning `10` repeatedly can mean “hold ten” or “buy ten again.” |
| Which prices can we see, and when can an order execute? | Seeing a closing price does not imply we can trade at it. |
| Do unfilled orders persist, expire, or get replaced each callback? | Persistent orders consume risk capacity even before they fill. |
| What are the position, order, gross exposure, tick, and lot limits? | A mathematically valid hedge may be an invalid order. |
| Are legs filled independently? Is there a queue or finite depth? | A two-leg signal can leave us with a one-leg position. |
| How are fees, borrowing, conversions, and final inventory treated? | Paper convergence may be smaller than the full trading bill. |
| Does Python state survive callbacks and rounds? What are runtime/import limits? | Global objects may reset; state may need explicit serialization. |
| What data/results can we inspect, and how many tests or submissions are allowed? | Repeated feedback is both a scarce resource and another way to overfit. |

Use permitted practice facilities to test a tiny buy, a tiny sell, an unfilled order, and a
restart. Predict the fills, cash, and positions first; compare those predictions with the
platform. Test limit rejection only where the organisers permit it. Keep the answers and
observed semantics in a local event note without committing organiser material.

## 2. Understand where PnL comes from

### Cash, inventory, and the spread

A buy of `q` units at price `p` changes cash by `-q*p-fees` and inventory by `+q`. Marked wealth
is `cash + sum(position * mark_price)`; PnL is its change from the start. Keep units explicit:
shares, currency, basis points, and log-price deviations are different quantities.

Suppose bid/ask is **99/101** and mid is 100. Buying at 101 then selling at 99 loses 2 per unit
even though mid never moved. A midpoint forecast of a 0.5 rise does not pay for that round
trip. One basis point is `0.0001` of traded notional; verify whether an advertised fee is
per side or per round trip, and avoid charging a quoted spread twice.

A passive bid at 99 might earn the spread, but only if it fills. If the next fair value is
97, the fill was bad: we bought at 99 just before the market fell. This is **adverse selection**.
Measure `side * (future_mid - fill_price)` after our fills, where side is +1 for a buy and -1
for a sell. That is a markout; subtract fees and eventual exit costs before calling it profit.
Use several horizons relevant to our holding time. Nearby markouts overlap, so their count
overstates the amount of independent evidence.

For a passive quote, ask both **“How often does it fill?”** and **“What happens conditional on
filling?”** A wide quoted spread, a good unconditional forecast, or a high fill rate alone
does not answer whether the quote makes money.

### Risk-adjusted selection versus the leaderboard

Our team policy selects on **train Sharpe plateaus**, then chooses size. This helps compare
strategies without rewarding larger exposure. The repository's displayed backtest Sharpe is
`mean(step_PnL) / std(step_PnL) * sqrt(number_of_steps)`: it is sample-length scaled, not an
annualized Sharpe. Compare equal windows and remember that serial dependence affects certainty.

Higher Sharpe does not mathematically guarantee higher final PnL under position limits,
different trading capacity, or different exposure patterns. Check the actual scoring rule,
net PnL, drawdown, and feasible size together. A statistical “OK” is not an instruction to
fill every available position slot. Increase size only after checking depth, rounding,
portfolio exposure, and the loss if the relationship breaks.

## 3. Start with a mechanism we can explain

For each idea, finish this sentence before searching parameters:

> We expect ___ to trade at ___ because ___. We can observe ___ before placing our order.
> We earn money if ___. We abandon the idea if ___ happens on later data or practice fills.

Example: two claims on the same payoff temporarily diverge because one book updates slowly.
We observe both executable quotes, trade only when convergence exceeds both legs' costs,
and reject the idea if the lag vanishes before we can execute. Two unrelated stocks that
happened to move together yesterday need a much stronger argument.

| What we observe | Small first hypothesis | What would disprove its usefulness? |
|---|---|---|
| A related pair has a stable spread | Fade unusually wide deviations | The spread drifts, convergence is too slow, or costs consume it |
| One instrument moves before another | Predict the follower using an already observed move | Alignment or an extra step of delay removes the profit |
| Imbalance precedes price moves | Use imbalance as a short-horizon fair-value feature | The relationship reverses later or is smaller than execution costs |
| Frequent apparently random takers | Quote around a fair value with inventory control | Filled quotes have adverse markouts larger than spread income |
| A counterparty has persistent signed markouts | Investigate whether its activity predicts later moves | Identity is unavailable until too late, or the effect fails later |

The diagnostic thresholds in `diagnose.py` are prompts to investigate. Correlation 0.1 or
variance ratio 0.8 is not a profitability boundary or an automatic trade instruction. Negative
autocorrelation in transaction prices can also come from trades alternating bid and ask.
Use aligned mids and executable prices before interpreting it as an opportunity.

Our `microprice()` is the top-of-book size-weighted midpoint, leaning toward the thin side.
Treat it as a candidate feature. Its predictive value and horizon need measurement; the
formula alone does not establish an executable fair value.

## 4. Pairs trading: what we are actually betting on

**Correlation** says two things move together. **Cointegration** asks whether a particular
combination of their price levels is stationary: its distribution does not keep wandering
without bound. Even a statistically convincing combination can be unprofitable to trade.
Screening many pairs also finds chance relationships: 100 null tests at a nominal 5% level
can produce about five false flags on average. Searching both directions adds selection.
Require an economic explanation and later evidence; the smallest test statistic is not a prize.

For positive prices our log-spread convention is:

```text
s = log(Y) - beta * log(X) - alpha
z = (s - estimated_mean) / estimated_standard_deviation
```

If z is sufficiently positive, Y is rich relative to the fitted relationship: a short-spread
trade sells Y and buys X when beta is positive. A negative z gives the opposite trade.
Estimate the relationship and spread statistics using only observations available then.
“Two standard deviations” describes the fitted distribution; it does not promise a reversal.

**Translate the model into money.** In log mode, a long-spread trade with Y-leg notional `N`
uses approximately `qY=N/Y` and `qX=-beta*N/X`. If Y=50, X=100, beta=1.5, and N=1,000, that
is **+20 Y, -15 X**: gross notional 2,500 and net notional -500. Beta is not a share count;
this hedge is neither automatically dollar neutral nor free of market risk. In price-level
mode `Y-alpha-beta*X`, the hedge instead uses `qX=-beta*qY`.

Before entering, calculate the cost of opening and closing **both** actual legs. For a fixed
hedge over a small move, expected log-spread convergence times `N` approximates gross PnL;
compare it with costs in currency. Use ask(Y)/bid(X) to open a long spread and bid(Y)/ask(X)
to open a short spread. Repricing the midpoint spread does not make those prices available.

**A usable relationship must fit the round's horizon.** A half-life of 100 observations means
an estimated deviation halves in about 100 observations under the fitted model. With only
20 observations left, full convergence is a weak assumption. Thousands of rows spanning a
few slow reversions are still little evidence. Missing rows also change what a “step” means.

**Adaptive beta is a hypothesis, too.** Kalman/EWM estimation can track change, but a very
flexible model can absorb a divergence by changing the hedge instead of earning money from
convergence. Changing beta or maintaining fixed notional also causes real rebalancing trades.
Compare static and adaptive models at the same costs and timestamps; inspect turnover.

**Define how we lose before entering.** Specify maximum holding time, maximum monetary loss,
maximum unhedged exposure, and a pause/re-entry rule. A z-score stop alone is insufficient:
as estimated volatility rises, z can shrink while our currency loss grows. `max_hold=0` and
`cooldown=0` in the current defaults disable those particular protections. A stop only requests
an exit; liquidity and delay determine the loss actually realized.

**Pairs are not automatically arbitrage.** Related companies can diverge permanently. Exact
dual claims can still have currency, conversion, settlement, borrowing, or execution costs.
Do not average down merely because a spread has become “more statistically extreme.”

## 5. Produce evidence that survives our own scrutiny

### Separate development from the final check

Use chronological **formation → development/validation → final test** windows. Formation
estimates the initial relationship. Development chooses a small, declared grid and a robust
region. Freeze the choice before inspecting the final test. Online updates during later
windows are legitimate if their rule was fixed and uses only available observations.

If a later window influenced pair choice, a threshold, a stop, or a go/no-go decision, call it
validation. Repeatedly reading “test PnL” while changing ideas turns it into development data.
When data is too short for three useful windows, acknowledge weak evidence and use fewer
choices and smaller exposure. A split cannot manufacture independent observations.
`diagnose.py` explores its full input, so give it the development portion when reserving a
holdout; learning which pairs look promising from the entire file also consumes information.

Use a formation warm-up and explicit evaluation start for each replay. At a boundary, document
whether positions/state carry forward and attribute the costs consistently. For overlapping
trades or labels, leave enough separation to avoid counting the same outcome on both sides.

### Make every candidate face the same questions

1. Does it beat doing nothing and the permitted organiser baseline **after costs**?
2. Does it still work at the required delay, and with one additional step of delay?
3. What happens at twice the plausible costs, and under worse permitted fills?
4. Do nearby train parameters work, or is only one isolated setting profitable?
5. Is PnL spread across episodes/windows, or dominated by one jump or one pair?
6. Does it stop taking risk during a planted break, missing quotes, and a restart?
7. Do the exact submitted artifact and research implementation make the same decisions?

These are stress tests, not new opportunities to tune on the final test. Record failed ideas
as well as successes. Independent synthetic random walks are a useful negative control;
recovering the deliberately planted synthetic edge proves a tool works on that mechanism,
not that the event contains it.

A **plateau** means nearby choices work too: for example, entries 1.5, 2.0, and 2.5 across
several reasonable history lengths. Prefer a stable region under comparable risk and costs
to a single spectacular result surrounded by losses. Check that the actual selected setting
works; a good neighborhood median can coexist with a poor individual setting.

### Treat overfitting statistics as conditional evidence

**PBO** asks how often the selected in-sample winner ranks below the out-of-sample median
under CSCV partitions. It is not the probability of losing money. A strategy can rank first
among losing alternatives. CSCV mixes earlier and later blocks; it does not replace a final
chronological evaluation. The [original PBO paper](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)
discusses hidden trials and dependence limitations.

**Deflated Sharpe** adjusts the evidence for a Sharpe ratio for selection across trials and
non-normal returns under its assumptions. DSR 0.95 is not a 95% chance of winning the event.
It cannot fix lookahead, optimistic fills, missing trial history, or a new market regime.
See the [original DSR paper](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf).

The repo's PBO/DSR thresholds are **team policy**, not universal constants. Follow its
reject/caution requirements; passing them is only one prerequisite. `min_backtest_length()`
uses the shortcut `2*log(N)/SR²`; do not interpret its output as an exact required sample size
for dependent trading PnL. Count all legitimate trials, including manual edits and abandoned
families; disclose how dependence between candidates and across time is handled.

Keep a local append-only record in gitignored `runs/`: data fingerprint, windows, strategy
revision, parameters, seed, lag, costs/fill model, net PnL, drawdown, turnover, exposure, and
decision. The current tools do not maintain this complete registry for us.

## 6. If the event has an order book

**Taking liquidity:** buy at available asks and sell at available bids. Check finite depth;
large orders can walk through several prices. A stale-looking quote may disappear before
our order arrives. Reject an apparent cross-book profit if either leg's executable price or
quantity eliminates it.

**Making liquidity:** a price touching our limit does not prove a fill. Existing orders may
be ahead in the queue; a cancellation might be behind us. Historical replay cannot show how
our inserted order would change other participants' behavior. Official
[HftBacktest documentation](https://hftbacktest.readthedocs.io/en/latest/order_fill.html)
explains these replay and queue limitations. Its assumptions are not automatically this
competition's matching rules.

**Inventory:** quote around a defensible fair value and skew away from unwanted inventory.
When already long, reduce the appetite to buy and improve the incentive to sell. Use actual
fills to hedge a passive leg; hedging intended quantities can create the opposite exposure.
Define what to do if the hedge cannot fill. A market maker's many small gains can be erased
by one persistent move while it accumulates inventory.

**Worst-case limits:** do not rely on buy and sell orders netting. With position +8, limit
10, resting buys 2, and a proposed buy 2, the possible long position is 12. Separately check
`position + all_possible_buys <= limit` and `position - all_possible_sells >= -limit`, including
new orders. Apply lot/tick rounding before the final risk check, and net strategies through
one shared risk gate. Respect additional order/gross limits from the briefing.

**Errors:** “no new orders,” “cancel resting orders,” “hold inventory,” and “flatten inventory”
are distinct actions. An empty order list usually does not flatten positions or cancel old
orders. Zero *targets* can request liquidation. Define error behavior using the actual API.

## 7. What we already have

| Component | Useful capability today | Evidence it does not provide |
|---|---|---|
| [diagnose.py](scripts/diagnose.py), [diagnostics.py](src/gqc/diagnostics.py) | Price behavior, lead–lag, L1 imbalance/staleness, counterparty markouts | Significance after all searches, executable profits, reliable arbitrary CSV alignment |
| [stats.py](src/gqc/stats.py), [screen_pairs.py](scripts/screen_pairs.py) | OLS, residual tests with cointegration critical values, half-life, both pair directions | A `TRADE` label is not a profitable strategy or proof the old hedge transfers |
| [core.py](src/gqc/core.py) | Causal online static/EWM/Kalman signals, warm-up, exit/stop logic, portfolio target netting | Complete exchange order management or a monetary-loss kill switch |
| [backtest.py](src/gqc/backtest.py) | Bar targets, execution delay, linear spread/fee costs, marked PnL | Depth, resting orders, queues, partial fills, impact, or independent leg fills |
| [sweep.py](src/gqc/sweep.py) | Formation-only static fitting, development plateau ranking, selected-config holdout report | Final portfolio sizing or exchange execution |
| [research.py](src/gqc/research.py), [overfit.py](src/gqc/overfit.py) | Current-grid PBO; DSR/MinBTL with recorded trial counts; initial-equity drawdown | Historical return matrices or automatic causality enforcement when called directly |
| [pipeline.py](src/gqc/pipeline.py) | Raw integrity probes, development plateau selection at 2× costs, stress/risk gates and a look-once holdout | Proof of causality for every candidate or PBO over historical searches |
| [skeleton.py](templates/skeleton.py) | Empty `decide()`, bounded history, finite-order checks, worst-case position/gross guards including supplied resting orders | A trading strategy, persistent-order manager, or fill-accurate account PnL |
| [replay.py](src/gqc/replay.py) | Delayed full fills, linear costs, marks retained across quote gaps; CLI rejects invalid outputs/errors | An order-book matching model or confirmed organiser semantics |
| [make_submission.py](scripts/make_submission.py) | Bundles the pairs core/config; isolated stdlib/bare-`exec()` smoke check | Bundling a custom skeleton or validating the final platform adapter |
| [synthetic.py](src/gqc/synthetic.py) | Seeded pairs, relationship breaks, independent walks, book/bot diagnostic exercises | A reactive multi-agent exchange or realistic passive-fill backtest |

**There is no implemented order-book backtest engine in this snapshot.** The book CSV
generator creates observations and trades; it does not match our submitted orders. The book
engine in [PLAN.md](PLAN.md) is a roadmap. RULES §6 names `replay_skeleton.py` and the pairs
builder's isolated smoke check. Packaging and local bar replay do not establish platform
compatibility.

Preserve the existing strengths: causal core prefix tests, seeded synthetic ground truth,
hand-computed PnL tests, constant-sized pair estimators, and netting before continuous target
scaling. The expanded checkout passed **165 tests** and `ruff check .` during this review. Those
checks do not cover all of the defects below.

## 8. Concrete findings and the additions worth making

The findings below record the original review at `660e7c5`; read the status notes before
treating a historical probe as a current defect. Fixes have targeted regression tests.

> **Status (fix/review-findings PR):** each finding below has a regression test in
> `tests/test_review_findings.py`. Fixed: **A** (static hedge fitted on a formation window
> only), **B** (evidence from development data only, scored for the selected config; final test
> printed once; REJECT suppresses export), **C** (lot-aware, resting-order-aware limit checks in
> `to_orders()` and the skeleton), **D** (failure ⇒ `None` / hold positions ⇒ no new orders),
> **E** (drawdown from zero equity, completed round trips only, `open_at_end` flag; generated
> pairs bundles expose `step()` so `replay_skeleton.py` replays the exact portfolio adapter),
> **F** (`timestamp`-style headers, numeric/ISO time sorting, gap-aligned lead–lag, sorted book
> rows for trade matching), **G** (single-candidate PBO = n/a; `runs/trials.jsonl` ledger adds
> prior trials on the same data to N), **H** (lag-0 replay fills, skeleton equity marks only
> positions held over the interval or uses platform PnL, gross cap allows reductions, risk
> exits pass the order guard until flat, G0 checks raw targets, pipeline selects on a
> development window with a G5 holdout gate).
> **Further pre-event fixes:** the pipeline now selects a numeric-grid plateau at 2× costs;
> the direct research template uses the same gates and trial ledger. Pipeline/replay drawdown
> includes initial equity; replay retains inventory marks across quote gaps and rejects
> non-finite output or recorded errors. The skeleton rejects non-finite orders and reserves
> gross capacity for resting orders, including during risk exits. Cross-asset book diagnostics
> align actual timestamps, preserving missing quotes rather than inventing fills.
>
> **Still open:** `screen_pairs.py` refits on the later window (treat it as validation);
> `sweep_pair.py` drops missing paired rows (warns); manual trials need `research/LOG.md`.
> Historical trial counts affect DSR/MinBTL, but PBO uses only the current return matrix.
> The ledger cannot reconstruct historical returns or their Sharpe dispersion. Platform
> adapter, fill/fee rules, terminal marking and account-PnL integration await the briefing.

### A. Static sweep training scores use hindsight — confirmed

In [sweep.py](src/gqc/sweep.py#L50), `sweep_pair()` fits alpha/beta using the entire training
window, then simulates the static model from its beginning. Later training observations
therefore influence earlier training decisions. The online core itself is causal when its
initial parameters are genuinely known at the time.

**Probe:** on a seeded 300-row price-level pair, changing only rows 120–179 changed fitted
beta from 1.99086 to 4.16268 and PnL from step 11. This is hindsight inside training, not by
itself leakage from the final test. Add a separate formation period or past-only refits;
changing future observations must not change earlier decisions in the complete sweep pipeline.

### B. The sweep verdict consumes test data and describes another selection — confirmed

[sweep_pair.py](scripts/sweep_pair.py#L58) sends full train-plus-test PnL columns to
`research.overfit_report()`. That report assesses the full-sample maximum-Sharpe candidate;
the exported parameters are the train-plateau winner. Its verdict therefore uses the test
and need not describe the candidate we export. The script also prints export parameters
after a negative verdict; the operator must still obey the rejection policy.

Compute selection diagnostics on development data, identify the actual selected candidate,
and reserve a later untouched evaluation. Likewise, selecting pairs using `screen_pairs.py`'s
later-window `TRADE` labels makes that window validation; the screen refits a new relationship
there rather than demonstrating that the original hedge transfers unchanged.

### C. Final order safety is incomplete — confirmed

The generated [`to_orders()`](scripts/make_submission.py#L38) rounds after `PairBook.combine()` scales continuous targets.
With limit 10, position 8, target 10, and lot 3, it emits +3 and reaches **11**. The final order
adapter needs a lot-aware limit check against actual inventory and outstanding orders.
The skeleton's `clip_orders()` only sees current inventory and the new quantities; that is
insufficient if previous orders can remain live. Test both directions and awkward lot sizes.

### D. An error can create new orders from an old target — confirmed

`PairBook.update()` in [core.py](src/gqc/core.py#L219) catches an exception and returns old targets.
If those targets are +10/-10 but actual positions are flat, feeding them to `to_orders()`
opens +10/-10 **after the failure**. This does not implement RULES §6's no-orders fallback.
Propagate failure status through the final adapter and test explicit no-new-risk behavior.
Returning zero targets is not an equivalent fix: that can force liquidation.

### E. Research PnL and submitted PnL can differ — confirmed limitations

`backtest.run_pair()` evaluates unconstrained fractional pair targets. Submitted `PairBook`
targets can be scaled by portfolio limits, netted across pairs, then rounded into orders.
Backtest the final adapter/portfolio too; isolated pair results do not establish portfolio PnL.

The bar engine leaves final inventory open and marks it at the supplied final price. It also
counts an unfinished holding episode in `round_trips`. Flat prices of 100, one long unit,
lag 0, and 100 bps cost produce PnL **-1**, final inventory **+1**, and `round_trips=1`.
Forced closure with the same cost would produce **-2**. Match the organiser's terminal rule
and distinguish completed trades from open episodes. In that example maximum drawdown is
also reported as zero because the initial zero-equity peak is omitted; include it in risk metrics.

### F. Data alignment needs an explicit check — confirmed

[io.py](src/gqc/io.py) reads a wide CSV headed `timestamp,A,B` with numeric timestamps as
three price series, including `timestamp`; `time,A,B` is recognized correctly. It also
preserves input order rather than sorting it. `diagnose.py` uses `searchsorted()` for trades,
which requires sorted book timestamps. Inspect detected columns and timestamp order first.

`diagnostics.log_returns()` drops invalid observations separately per asset. Different gaps
can make the lead–lag report correlate returns from different time intervals. `sweep_pair.py`
also drops missing paired rows, compressing time. Normalize schemas, retain an aligned time
index, and define stale/missing-data behavior before interpreting lags or half-lives.

### G. The statistical report cannot yet account for the whole search — confirmed

`research.overfit_report()` uses the columns from the current call as its trial count.
Earlier grids, other explored ideas, and manual tweaks are absent unless collected explicitly.
Add the local experiment registry described in section 5. A one-candidate report also returns
PBO=1 and rejects even a strongly positive seeded example; cross-candidate PBO is unavailable
when there is no selection to compare. Repair that case rather than treating the message as
evidence of either an edge or its absence.

The generic research harness passes full arrays to arbitrary strategy functions. It trusts
the strategy to be causal. Each new strategy needs a prefix/future-perturbation test; adding
execution lag cannot cure a function that reads future prices. The new pipeline adds sampled
integrity checks; it does not establish causality for the whole matrix of trials.

### H. The new pipeline/replay/risk additions need specific follow-up

These tools add useful checks, but the following probes found remaining gaps; do not treat a
`GO` label as certification of the complete submission path.

- [`pipeline.evaluate()`](src/gqc/pipeline.py#L125) selects the full-input maximum Sharpe,
  without a chronological split or plateau. Supplying the final test there consumes it.
  Its integrity check sees throttled output: a strategy returning `inf` everywhere can pass
  G0 because throttling silently discards invalid values. Validate raw targets first.
- [`replay.replay()`](src/gqc/replay.py#L50) processes fills before appending new orders.
  An initial +1 order on prices 100/110/120 produces **zero trades at lag 0**. Support that
  timing explicitly or reject it; a zero-PnL replay can otherwise look harmless while doing nothing.
- The new skeleton credits price changes to current inventory. A new fill of +1 at 110 after
  the previous mid of 100 records **+10** estimated equity although the trade has earned zero.
  Reconcile with actual fills/cash/costs before using that estimate as a drawdown gate.
- The new gross cap blocks partial reductions when already over the cap: cap 100, mid 10,
  position 20, order -1 returns no order. Conversely, stop flattening bypasses `MAX_ORDER`:
  a position of 20 with order cap 2 can emit -20. Risk exits must pass the final order guard too.

The new frequency and no-trade-band variants are worth investigating: they can reduce costly
rebalancing. Count them as trials, and preserve exits and risk reductions when throttling.

## 9. Use the remaining preparation time well

**Recommended order:** rehearse the final interface; repair order/error safety; repair the
research evidence path; then consider another signal. If time runs out, reduce scope to the
tested path. An untested simulator rewrite shortly before submission creates another unknown.

| Preparation block | Concrete output |
|---|---|
| First 10 minutes | Everyone runs tests; one driver owns submissions, one analyst owns experiments, one risk/ops owner checks limits and rollback |
| Next 25 minutes | Blind rehearsal: one person chooses an undisclosed seed/break scenario; others diagnose, name a mechanism, and replay one rule |
| Next 20 minutes | A tiny final-adapter replay covering position/order semantics, limits after rounding, missing input, an exception, and state restart |
| Next 15 minutes | Formation/development/test boundaries and a complete local experiment record; compare one baseline with one candidate |
| Final 10 minutes | Freeze a known-good artifact, note/hash its exact contents, and rehearse restoring it |

The timings are a proposed preparation budget, not a claim about the event's round lengths.
Have a teammate review the final file and check the diff; follow the branch/PR rules.

### Commands we can rehearse now

```bash
uv sync
uv run pytest -q
uv run ruff check .
mkdir -p data runs research submissions

# Diagnostic practice. Use another seed for a blind rehearsal.
uv run python scripts/make_practice_data.py data/practice_book.csv --book --seed 23
uv run python scripts/diagnose.py data/practice_book.csv --trades data/practice_book_trades.csv

# Pairs practice. These commands run; section 8 explains why their scores need repairs.
uv run python scripts/make_practice_data.py data/practice_synth.csv --seed 23
uv run python scripts/screen_pairs.py data/practice_synth.csv
uv run python scripts/sweep_pair.py data/practice_synth.csv --y AAA --x BBB --lag 1 --cost-bps 1

# Packaging rehearsal only: example symbols/parameters are not event recommendations.
uv run python scripts/make_submission.py --config submissions/example_config.json --out submissions/practice-v1.py
```

For a new idea, copy `templates/research_template.py` into `research/` and test its causality
before interpreting results. Its example is a single-asset EWM fade, not a ready-made pairs
strategy. The present builder packages the pairs core, so use the new skeleton replay path
to exercise custom `step()` files. Replaying the untouched skeleton produces no orders by
design; zero errors alone does not establish a working trading rule. Verify the generated
pairs file's final platform adapter after editing, too.
The new `evaluate_algo.py` and `replay_skeleton.py` offer additional checks for custom research
and `step()` files; use development data and account for section 8H's current limitations.

During this review, both data generators, book diagnostics, pair screening/sweep, the research
template, and submission generation ran successfully on temporary practice files (1,200 rows,
seed 23). This establishes that the commands execute, not that their output wins on event data.

### At the event: one complete loop at a time

Establish a valid baseline early, using the confirmed interface and allowed test facilities.
Once a safe artifact works, change one hypothesis at a time. Keep the last known-good file
and one-line reason for every version. Stop adding features when there is insufficient time
to replay, inspect risk, and restore the previous artifact before the deadline.

For each candidate, keep this short decision record in `runs/`:

```text
Mechanism and information available at decision time:
Data fingerprint; formation/development/final-test boundaries:
All trials so far; chosen config; reason for selection:
Baseline vs candidate: net PnL, drawdown, costs, turnover, maximum exposure:
Delay/cost/fill stress results; PBO/DSR scope and verdict:
Known failure conditions; monetary and unhedged-risk limits:
Final artifact hash; platform smoke result; rollback version:
Decision: reject / gather evidence / submit at stated size:
```

## 10. Sources worth reading

- [G-Research: ETH Zürich 2026](https://www.gresearch.com/events/eth-zurich-quant-challenge-2026/)
  establishes this event's public description; confirm operational rules at the briefing.
- [Frankfurt Hedgehogs: Prosperity 3, second-place team writeup](https://github.com/TimoDiehm/imc-prosperity-3#how-to-properly-backtest)
  explains why some bot-dependent fills required the official simulator and why repeated
  website-score optimization was dangerous. Transfer the method, not their game's mechanics.
- [HftBacktest: Order Fill](https://hftbacktest.readthedocs.io/en/latest/order_fill.html)
  explains queue assumptions and why historical replay cannot model our own market impact.
- [Bailey et al.: Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)
  defines CSCV/PBO, distinct from the probability of loss, and discusses limitations.
- [Bailey and López de Prado: Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
  explains selection bias, non-normal returns, and the role of independent trial counts.
- [Gatev, Goetzmann, and Rouwenhorst: Pairs Trading](https://www.nber.org/papers/w7032)
  provides an original example of separating formation and later trading periods. Its distance
  method is not proof that our chosen pair is cointegrated or profitable at event costs.
