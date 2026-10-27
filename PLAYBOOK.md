# PLAYBOOK — G-Research Quant Challenge (algorithmic trading, Python)

Intelligence and tactics for the event. Facts carry a source and a confidence tag;
inferences are labelled **[inference]**. Last researched: 2026-10-27.

---

## 0. TL;DR

1. **The G-Research Quant Challenge is short.** Event windows are 17:30–22:00 including the rules
   briefing and networking, so expect roughly 2 h of actual trading/coding in 2–3 rounds. Iteration
   speed and paste-ready, pre-tested templates beat clever ideas invented on the night.
2. **Our base case: since 2024 the G-Research event is a "cointegration challenge" — a pairs
   trading game run on "unseen, real world data sets".** Python-based, teams of up to 3. Core
   weapon: **spread / stat-arb trading** (pairs, baskets), estimated **online** because the data
   is unseen. §5.1 and §5.0 are the heart of this document.
3. Expect a naive example strategy (static hedge ratio + fixed z-score bands). It is the baseline
   to beat, not the ceiling. Beat it with: log-price spreads, online/Kalman hedge ratios,
   cost-aware thresholds, regime-break detection, sizing by confidence, and execution that
   doesn't pay the spread twice.
4. **Secondary case — bot-populated order book** (our prior Optiver Quant Challenge experience:
   bots traded badly, a dual-listing example strategy, one paste-in tab). If G-Research's game
   also has bots, they become a PnL source: profile them and pick the exploit in §4.
5. **Never blow up:** hard position-limit guard, `try/except` that returns no orders, bounded
   state. One crash or limit breach can zero a round (see Prosperity 3 cautionary tale, §2.1).
6. **Backtest pessimistically.** Only ship strategies that are positive under the conservative
   fill model; pick parameters from plateaus, not peaks.
7. **Confirm the interface in the first 5 minutes** (§9 checklist). We do not know it yet.

---

## 1. What is publicly known about the G-Research event

### 1.1 Format history

| Period | Format (as G-Research describes it) | Team | Source |
|---|---|---|---|
| 2020 (virtual) | Live trading game "based on continuous blackjack", multiple rounds vs a dealer; browser-based | 2–3 | [London virtual 2020](https://www.gresearch.com/news/virtual-quant-finance-algorithmic-challenge-london/) |
| 2021–2023 | "An algorithmic trading game" that "will put your Python skills to the test"; QRs explain "each round and what you will need to do"; bring a charged laptop | 2–3 | [Cambridge 2022](https://www.gresearch.com/news/quant-challenge-the-university-of-cambridge/), [KTH 2023](https://www.gresearch.com/events/kth-quant-challenge/), [EPFL 2023](https://siam.epfl.ch/2023/08/31/g-research-quant-finance-challenge/) |
| 2024 → now | "Brand new cointegration challenge … test your strategies on unseen, real world data sets"; "new cointegration game" | ≤3 (EPFL) | [Paris](https://www.gresearch.com/events/paris-quant-challenge/), [Tübingen](https://www.gresearch.com/events/tubingen-quant-challenge/), [Barcelona tweet](https://x.com/GRESEARCHjobs/status/1803731262529556681), [EPFL 2024](https://www.gresearch.com/events/epfl-quant-challenge-2/), [ETH 2025](https://www.gresearch.com/events/eth-zurich-quant-challenge-3/), [Oxford 2026](https://www.gresearch.com/events/oxford-quant-challenge-2026/), [Berkeley 2026](https://www.gresearch.com/events/berkeley-quant-challenge-2026/) |

Confidence: **high** that the current format is cointegration-themed, Python, small teams, a few
hours long. **Unknown** (not published anywhere we could find): the interface, the API, the
scoring rule, position limits, fees, tick size, whether bots are visible, number of products.

### 1.2 Upcoming / recent dates seen

- **Our event: ETH Zürich — Tue 27 Oct 2026, 17:30–21:30, Hauptgebäude, Rämistrasse 101**
  ([page](https://www.gresearch.com/events/eth-zurich-quant-challenge-2026/)). "Cointegration
  challenge … unseen, real world data sets", data provided at the event. Team size and laptop
  requirements not stated — bring charged laptops + chargers anyway.
- London — Wed 4 Nov 2026, 17:30–22:00, 1 Soho Place ([page](https://www.gresearch.com/events/london-quant-challenge-2026/)).
- Oxford 15 Oct 2026, Berkeley 5 Oct 2026 (past). EPFL 2024 drew 50+ PhD students
  ([SIAM EPFL](https://siam.epfl.ch/2024/10/23/g-research-quant-finance-workshop-wednesday-16th-october-2/)).

### 1.3 Our team's prior experience (Optiver Quant Challenge — not G-Research)

- Bots traded on a shared order book and **traded badly**.
- Organisers supplied an example strategy around a **dual listing** (same asset, two books).
- One tab to paste code into, then the algo traded.

These are **Optiver** facts. They shape our tooling (single-file strategies, bot-aware
backtester) but the G-Research base case is the pairs game in §1.4.

### 1.4 Working hypothesis of the G-Research game **[inference]**

A pairs/cointegration trading game: several instruments with real historical prices, some pairs
cointegrated (some deliberately not); we write a Python strategy that decides positions/orders
on each new observation; it is scored on PnL over **unseen** data, round by round. Two variants
to prepare for:

| Variant | What we submit | Implication |
|---|---|---|
| **A. Signal on price series** (most likely) | positions per timestep from bars/quotes | PnL depends on estimation quality, costs, risk; no queue dynamics |
| **B. Simulated exchange replaying real data** (possibly with bots) | orders against a book | execution and bot behaviour matter too (§4, §7) |

Things that decide the winner in variant A: (1) not trading pairs that aren't cointegrated,
(2) noticing when a pair stops being cointegrated, (3) correct hedge ratio and log-spread,
(4) thresholds that cover costs, (5) sizing. Every claim here must be re-checked at the briefing.

---

## 2. Comparable competitions — what transfers

### 2.1 IMC Prosperity (closest public analogue: Python, order books, bots, single-file upload)

Mechanics: upload a `Trader` class whose `run(state)` returns orders per product (+ conversions,
+ a `traderData` string for state persistence). Bots quote and trade around a hidden fair value;
position limits per product; exceeding a limit cancels **all** your orders for that product
([jmerle backtester](https://github.com/jmerle/imc-prosperity-3-backtester)).

Lessons from top teams:

| Lesson | Evidence |
|---|---|
| Fair value = mid of the **big, stable market maker** ("wall mid"), not the raw mid | [Linear Utility, #2 P2](https://github.com/ericcccsliu/imc-prosperity-2), [Frankfurt Hedgehogs, P3](https://github.com/TimoDiehm/imc-prosperity-3) |
| Validate the marking convention: buy 1 unit, hold, compare platform PnL to your model | Linear Utility, [chrispyroberts 1st USA P3](https://github.com/chrispyroberts/imc-prosperity-3) |
| Take mispriced orders first, then quote inside the spread, then **flatten inventory at fair** (zero-EV trades) — added ~3% PnL by freeing limit capacity | Linear Utility |
| Profile bots: one bot sized 20–30 (MM), another 1–5 crossing fair; a hidden taker filled anything at a specific price → systematic edge | Linear Utility, Frankfurt Hedgehogs |
| An **informed bot** ("Olivia") bought daily lows / sold daily highs → follow it | Frankfurt Hedgehogs, chrispyroberts |
| Basket vs synthetic spread was mean-reverting; fixed thresholds with a **hard-coded mean** and short rolling std worked; exit at mean, not opposite threshold | Linear Utility (~111–135k), Frankfurt Hedgehogs |
| Hedging costs spread; sometimes half-hedge or skip | Frankfurt Hedgehogs (50% hedge), chrispyroberts (hedging cost >40k) |
| **Prefer parameter plateaus**, justify every edge from first principles; don't optimise the website score | Frankfurt Hedgehogs |
| Backtest PnL overstates live (135k → 111k) | Linear Utility |
| A memory blow-up wiped rolling state → random trading → rank 7 → 241 | chrispyroberts |
| Exploiting deterministic platform timing was **ruled unfair** — behaviour edges OK, platform bugs not | chrispyroberts |

### 2.2 Optiver Optibook / Quant Challenge (our prior experience; Python client to a simulated exchange)

Classic case: **dual-listed stock** (e.g. `PHILIPS_A` liquid, `PHILIPS_B` illiquid) — make markets
on the illiquid listing, hedge fills on the liquid one; later baskets/ETFs and cointegrated pairs.
A public student package re-implements an Optibook-compatible simulator and an Engle–Granger pairs
trader with **bid/ask-aware executable spreads**, depth-capped sizing, ±100 position limits and a
hedge-ratio drift kill-switch ([Amir-Shokat/pairs-trading-cointegration](https://github.com/Amir-Shokat/pairs-trading-cointegration)).
Its bug log is a checklist: zero threshold during warm-up, hedging the full size after a partial
first-leg fill, flattening everything at extreme prices, refitting every loop.

Client calls seen in student repos (**unverified**): `Exchange().connect()`,
`get_last_price_book(id)`, `insert_order(id, price=, volume=, side='bid'|'ask', order_type='limit'|'ioc')`,
`get_positions()`, `delete_orders(id)`, `poll_new_trades(id)`.

### 2.3 Others

- **AlgoTrade 2026 (Zagreb)** — 10 venues, rate limits that disconnect, hard resets per segment;
  team modelled fills **only against visible top-of-book**, calling reported edge "a strict lower
  bound" ([repo](https://github.com/pitfa19/algotrade2026)).
- **Jane Street ETC** — live bots vs exchange, invitation-only; same skills (bonds = fixed fair,
  ADR/ETF conversion arbitrage).
- **Algothon (Susquehanna/UNSW)** — offline positions on price paths; overfitting is the killer.

---

## 3. Platform archetypes → how we write code

Write the **decision logic as a pure function** and wrap it with a thin adapter per archetype:

```python
def decide(snapshot: dict, state: dict, params: dict) -> tuple[list[dict], dict]:
    """snapshot -> (orders, new_state). No I/O, no globals, no randomness without a seed."""
```

| Archetype | Shape | Gotchas |
|---|---|---|
| A. Per-tick callback (Prosperity-style "paste tab") | `class Trader: def run(self, state) -> orders` | State may not survive restarts → serialise to the provided string; memory/time limits; limit breach may cancel all orders |
| B. Client loop to live exchange (Optibook-style) | `while True: book = get_book(); ...; sleep(dt)` | Rate limits, partial fills, stale books, reconnects, outstanding-order hygiene |
| C. Offline signal on historical data | return positions/weights per timestamp | Lookahead, scoring metric (PnL vs Sharpe), turnover costs |

Rules for the pasted file: **one file**, stdlib + `numpy` at most (have a stdlib-only fallback),
no file/network I/O, O(1) memory per tick (deques with `maxlen`, Welford running stats), every
`decide` wrapped in `try/except` that logs once and returns `[]`.

---

## 4. Exploiting bad bots (variant B / Optiver-style books)

Applies when the game has a shared order book with other participants. In a pure
signal-on-prices game (variant A) skip to §5; the autocorrelation and lead–lag probes below
still apply to the price series.

### 4.1 Detection toolkit (run on the first data we see)

| Probe | How | Reveals |
|---|---|---|
| Size/price fingerprint | histogram trade sizes and prices relative to fair, per counterparty if IDs exist | which "bots" exist, MM vs taker |
| **Markouts** | for each trade: `sign × (mid[t+k] − price)`, k = 1, 5, 20 steps, grouped by counterparty/size | who is dumb (we earn trading against them) vs informed (avoid/follow) |
| Return autocorrelation | ACF of mid returns lags 1–10; variance ratio (Lo–MacKinlay) | negative → overshoot/mean reversion; positive → momentum |
| Lead–lag | cross-correlation of returns between related instruments | which leg is stale → trade the laggard |
| Quote staleness | does a bot's quote move after fair moves? lag in steps | pick-off opportunities |
| Response probing | post small quotes at varying distances; record fill probability | the taker's acceptance boundary → quote right at it |
| Timing | trades by step index mod N, time-of-round | scheduled flow, end-of-round flattening |

### 4.2 Bot types → exploit

| Bot behaviour | Exploit |
|---|---|
| Noise taker (random market orders) | Passive market making: capture half-spread, skew for inventory |
| Fixed-fair / stale quoter | Take its quote whenever it crosses our fair by > costs |
| Overreacting / panic trader | Fade: mean reversion on price or spread with z-score entries |
| Momentum chaser | Provide liquidity into its bursts at extended prices; exit on reversion |
| Naive pairs bot (wrong β, slow) | Trade its imbalance; it pushes the spread, we take the other side |
| Hidden taker with fixed acceptance price | Quote at the edge of that price with max allowed size |
| Informed bot | Follow its direction; pull quotes when it is active |
| Inventory-constrained bot | Anticipate forced flattening near round end |

Boundary: exploiting **market behaviour** is the game. Exploiting **platform bugs or timing
artefacts** → ask organisers first (Prosperity precedent: ruled unfair).

---

## 5. Strategy math

### 5.0 Pairs trading on real data — what actually goes wrong

- **Correlation ≠ cointegration.** Highly correlated returns can have a non-stationary spread.
  Test the spread, not the returns.
- **Multiple testing.** Screening N assets gives N(N−1)/2 pairs; at 5% significance ~5% pass by
  luck. Require economic rationale (same company/sector/ETF twins), a stricter p-value, and an
  out-of-sample re-test before trading.
- **Use log prices**: `s = log y − β log x − α` (scale-free, spread in return units).
- **β drifts and breaks happen.** Even share classes drift: GOOGL−GOOG closes went 4.18 → 2.58
  between 19 and 26 Oct 2026 (Alpaca IEX daily bars). Estimate online; stop trading a pair whose
  spread variance or half-life blows up.
- **Heteroskedastic spreads**: scale z by an EWMA σ, not a full-sample σ.
- **Costs dominate at high frequency**: required move per round trip ≈ 2 × (half-spread_y +
  β·half-spread_x) + fees. Only enter when the expected reversion (`|s − μ|`) exceeds that by a margin.
- **Robust baseline**: distance method (Gatev et al. 2006) — normalised prices, enter at 2σ of the
  formation-period distance, exit at crossing. Hard to overfit; good fallback submission.
- **Data hygiene**: adjusted prices (splits/dividends), aligned timestamps, gaps/halts → skip.

### 5.1 Cointegration / pairs / dual listing (primary)

1. **Test.** Engle–Granger: regress `y = α + βx + ε`, ADF on residuals using MacKinnon
   cointegration critical values (`statsmodels.tsa.stattools.coint`), test both orderings.
   Johansen for ≥3 legs. Dual listing = special case β=1, α≈0 (verify, don't assume).
2. **Spread.** `s_t = y_t − β x_t − α`. Hedge ratio options: static OLS on warm-up; rolling OLS;
   **Kalman filter** with state `[β, α]` as a random walk (Chan, *Algorithmic Trading*, 2013),
   trade on the normalised forecast error `e_t / sqrt(Q_t)`.
3. **Speed.** OU model `ds = θ(μ − s)dt + σ dW`. From `Δs_t = a + b s_{t−1} + e`:
   `θ ≈ −b`, **half-life = −ln 2 / ln(1 + b)** (≈ ln 2/θ). Stationary std = `σ / sqrt(2θ)`.
   Use half-life to set the z-score window (≈ 2–5 half-lives) and a time stop (≈ 3 half-lives).
4. **Signal.** `z = (s − μ̂) / σ̂`. Enter at |z| > z_in, exit at z ≈ 0 (or |z| < z_out), stop at
   |z| > z_stop or regime break. Optimal thresholds with costs for OU: Bertram (2010).
5. **Executable spread.** Compute entry with the prices you'd actually hit: short-spread uses
   bid(y) and ask(x); long-spread uses ask(y) and bid(x). Signal must survive the bid–ask.
6. **Unseen data ⇒ estimate online.** Warm-up window (no trading or tiny size), then EWMA/Kalman
   updates. Detect breaks: rolling ADF p-value, spread variance blow-up, β drift → flatten.
7. **Execution.** Prefer posting the leg with the wider spread passively and hedging the fill
   aggressively on the liquid leg (Optibook dual-listing pattern). Size each order ≤ visible
   depth on the hedge leg; hedge on **filled** quantity, not intended quantity.

### 5.2 Single-asset mean reversion (secondary)

Works only if the data says so (negative lag-1 ACF, variance ratio < 1, Hurst < 0.5). Signals:
deviation from EMA / microprice in σ units; Bollinger-style bands. Robust choice from
Prosperity: fixed thresholds on a fast EMA, no vol scaling — then verify on plateaus.

### 5.3 Market making

- Fair value: wall mid (deep-liquidity levels) > microprice > mid for noisy books.
- Microprice: `(P_bid·Q_ask + P_ask·Q_bid) / (Q_bid + Q_ask)`; imbalance
  `I = (Q_bid − Q_ask)/(Q_bid + Q_ask)` predicts the next mid move (Stoikov, 2018).
- Avellaneda–Stoikov (2008): reservation price `r = s − q·γ·σ²·(T−t)`; optimal total spread
  `γσ²(T−t) + (2/γ)·ln(1 + γ/k)`. In practice: linear inventory skew `r = fair − κ·q`.
- Order of operations each tick: **take** mispriced → **clear** inventory at fair → **make**
  inside the spread with remaining capacity.

### 5.4 Combining

Use spare position capacity for market making on the legs (Prosperity teams: +~5k/day). Net
the inventory across strategies before sending orders; one risk gate for all.

---

## 6. Risk and execution checklist

- Hard guard: projected position after **all** resting + new orders within limits (assume all fill).
- Leg risk: max unhedged notional; if the second leg can't fill within N steps, unwind the first.
- Partial fills: hedge what filled; track outstanding orders; cancel stale ones each tick.
- Kill-switch: drawdown > X or spread z beyond stop → flatten and pause.
- Never crash: catch everything, return `[]`, log once (log spam can exceed limits).
- Determinism: seed any randomness; no reliance on wall-clock.
- End of round: know whether positions are marked or force-closed; flatten if marking is punitive.

---

## 7. Backtesting rigour (see PLAN.md for the engine)

- Three fill models, report all: **conservative** (take only visible touch; passive fill only when
  trades print strictly through our price), **queue-based** (track queue ahead; hftbacktest-style
  risk-averse / probabilistic queue models), **optimistic** (Prosperity "match all"). Ship only
  what is positive under conservative.
- Latency: decide on state at t, execute at t+L.
- Reproduce platform semantics: limit rule (cancel-all vs clip), tick/lot size, fees, marking.
- Validate the simulator against the platform before optimising (hftbacktest's principle:
  "accurate backtesting is the foundation").
- Agent-based mode: synthetic bots from §4.2 so our orders move the market and bots react.
- Overfitting defence: few parameters, plateau selection, walk-forward / out-of-sample days,
  Monte Carlo over seeds, PBO + Deflated Sharpe (§7.1).

### 7.1 Objective and overfitting control (binding — RULES.md §5)

**Objective.** We want maximum PnL, but we *select* on **Sharpe** and then **size to the
limits**. PnL = size × edge; once size is capped by position limits, the config with the best
risk-adjusted return gives the most PnL per unit of risk. Selecting on raw PnL rewards configs
that simply took more risk (more time in market, larger effective size) and is noisier, so it
overfits faster. The competition is a tournament: when leading, cut variance; when behind late,
more size on a *validated* edge is rational — never add variance by adding parameters.

**Why.** Try N configs on one dataset and the best one looks good by luck: the expected maximum
Sharpe of N skill-less strategies is ≈ σ_SR·[(1−γ)Φ⁻¹(1−1/N) + γΦ⁻¹(1−1/(Ne))] (≈ 3.25σ_SR for
N = 1000). Tonight we tune on one short sample, which is exactly where this bites.

**Tools** (`src/gqc/overfit.py`, `research.overfit_report`, printed by `sweep_pair.py` and
`templates/research_template.py`):

| Measure | What it answers | Source |
|---|---|---|
| **PBO** via CSCV | Split the PnL matrix (T steps × N configs) into S blocks; for every half/half split pick the in-sample winner and find its out-of-sample rank. PBO = P(winner ranks below the OOS median). | Bailey, Borwein, López de Prado & Zhu (2017) |
| **Deflated Sharpe** | P(true Sharpe > what the best of N noise trials would show), adjusted for skew, kurtosis, length | Bailey & López de Prado (2014) |
| **MinBTL** | Steps needed so N trials can't produce this Sharpe by luck: `2 ln N / SR²` | Bailey et al. (2014) |
| OOS/IS decay | Median OOS Sharpe ÷ median IS Sharpe of the CSCV winner | — |

**Gates** (`research.verdict`): PBO ≤ 0.2 **and** DSR ≥ 0.95 **and** T ≥ MinBTL → use the
tuned params, full size. PBO ≤ 0.5 and DSR ≥ 0.5 → plateau params, reduced size. Otherwise →
reject the tuning; run the simplest default rule small or not at all.

**Practice check** (synthetic data): real pair → PBO 0.00, DSR 1.000, decay 0.89 → OK.
Two independent random walks → P(OOS loss) 0.78, DSR 0.25, decay −0.41 → REJECT.

**Caveats we know about.**
- PBO from **one** sample is itself noisy: on a single pure-noise sample we measured 0.11 even
  though the expectation is 0.5. Never gate on PBO alone; require DSR and a plateau too.
- **Count every trial.** N must include manual tweaks and abandoned ideas tried on today's data;
  undercounting N inflates DSR. Correlated configs make the effective N smaller, so raw N is
  conservative — that is the side we want to err on.
- Positions held over many steps make PnL autocorrelated; per-step Sharpe then overstates
  significance (Lo, 2002). Treat a borderline DSR as CAUTION.
- CSCV assumes the blocks are comparable (stationarity). A regime change inside the data makes
  every in-sample choice look overfit; check the price plot before blaming the strategy.
- Keep grids coarse (≤ 3 knobs × 3–4 values). Derive what you can from data (half-life from an
  OU fit, entry threshold from round-trip cost) instead of tuning it.

---

## 8. Python speed tips

`collections.deque(maxlen=n)` for windows; Welford/EWMA for running mean/var; avoid pandas in the
tick loop; precompute constants; `numpy` only for batch maths; profile with `cProfile` on a full
replay. Target: decision < 1 ms per tick.

---

## 9. Event-day runbook

**Before leaving:** laptops charged + chargers; repo cloned; `templates/` printed in an editor;
offline copies of numpy/statsmodels docs; this playbook open.

**Briefing — ask/confirm (write answers into `research/event-notes.md`):**

1. Exact interface: paste box? function signature? what object comes in, what goes out?
2. Allowed imports (numpy? pandas? statsmodels?), time/memory limits per step.
3. Instruments, which are related, tick size, lot size, position limits, fees.
4. Matching rules: when do our orders execute? Do resting orders persist across steps?
5. Scoring: PnL marked to what? Positions force-closed? Per round or cumulative?
6. Can we see bot IDs / trade history / our fills? Is there a practice round with logs?
7. Number of submissions allowed; can we resubmit mid-round?
8. What is explicitly disallowed?

**Roles (team of 3):** *Driver* — owns the pasted file and submissions. *Analyst* — runs §4.1
probes and parameter checks on fresh data. *Risk/ops* — checks limits, logs, timing, keeps the
last-known-good version, calls the kill decision.

**Cadence:** submit the safe baseline within the first 10 minutes → read results → change **one**
thing → resubmit. Keep `submissions/round-N-vK.py` with a one-line note on each.

**Principle: no trading logic is fixed before the event.** We observe the data and the book,
diagnose which behaviour is exploitable, then write `decide()` for that behaviour.

**Flow when data arrives** (save as `data/round1.csv`; book/trades as CSV if offered):

```bash
# 1. OBSERVE — what behaviour exists? (prices, order book, trades/counterparties)
uv run python scripts/diagnose.py data/round1.csv [--trades data/round1_trades.csv]
# 2. If pairs look cointegrated, confirm out of sample
uv run python scripts/screen_pairs.py data/round1.csv
# 3. Prototype the idea + a coarse grid, check overfitting, obey the VERDICT (§7.1)
cp templates/research_template.py research/idea1.py   # edit strategy() and GRID
uv run python research/idea1.py data/round1.csv
# 4. Port the surviving rule into templates/skeleton.py decide(); size to limits
# 5. Pairs only: tune by plateau (prints PBO/DSR), or build from the ready pairs module
uv run python scripts/sweep_pair.py data/round1.csv --y A --x B
uv run python scripts/make_submission.py --config cfg.json --out submissions/round1-v1.py
```

| Diagnose says | Write in `decide()` | Playbook |
|---|---|---|
| VR < 0.8, negative ac1 | fade deviations from EWM mean (z-score) | §5.2 |
| VR > 1.2 | follow moves / don't fade | §5.2 |
| `A leads B` | trade B on A's last move | §4.1 |
| pair `TRADE` in screen | spread z-score (`src/gqc/core.py` ready-made) | §5.1 |
| imbalance corr ≥ 0.1 | fair = microprice; skew quotes toward it | §5.3 |
| wide spread, stale quotes | quote inside spread / take stale quotes | §4.2 |
| party markout < 0 | trade against that party / quote to it | §4.2 |
| party markout > 0 | follow that party; never be on the other side | §4.2 |

If their example strategy uses a static hedge ratio and fixed bands, the cheap upgrades are
online hedge (Kalman/EWM), exit at mean, stop at |z| > 4.

---

## 10. Sources

G-Research event pages linked in §1. Competition write-ups:
[Linear Utility (P2 #2)](https://github.com/ericcccsliu/imc-prosperity-2) ·
[Frankfurt Hedgehogs (P3)](https://github.com/TimoDiehm/imc-prosperity-3) ·
[chrispyroberts (P3 7th)](https://github.com/chrispyroberts/imc-prosperity-3) ·
[Alpha Animals (P3 9th)](https://github.com/CarterT27/imc-prosperity-3) ·
[jmerle P3 backtester](https://github.com/jmerle/imc-prosperity-3-backtester) ·
[Optibook pairs package](https://github.com/Amir-Shokat/pairs-trading-cointegration) ·
[AlgoTrade 2026](https://github.com/pitfa19/algotrade2026) ·
[hftbacktest docs](https://hftbacktest.readthedocs.io/en/latest/).

Literature: Engle & Granger (1987) *Econometrica*; Johansen (1991) *Econometrica*;
Gatev, Goetzmann & Rouwenhorst (2006) *RFS* "Pairs Trading"; Bertram (2010) *Physica A*
"Analytic solutions for optimal statistical arbitrage trading"; Avellaneda & Stoikov (2008)
*Quantitative Finance*; Stoikov (2018) *Quantitative Finance* "The micro-price";
Lo & MacKinlay (1988) *RFS* variance ratio; Bailey & López de Prado (2014) "The Deflated Sharpe
Ratio", *J. Portfolio Management*; Bailey, Borwein, López de Prado & Zhu (2017) "The Probability
of Backtest Overfitting", *J. Computational Finance*; Bailey, Borwein, López de Prado & Zhu
(2014) "Pseudo-Mathematics and Financial Charlatanism", *Notices of the AMS* (MinBTL);
Lo (2002) "The Statistics of Sharpe Ratios", *FAJ*; Chan (2013) *Algorithmic Trading*.
