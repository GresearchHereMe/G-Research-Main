# AGENTS.md

Instructions for any AI coding agent (Claude Code, Codex, Cursor, Copilot, …) working in this
repository. Tool-specific extras live in [CLAUDE.md](CLAUDE.md). Binding team rules:
[RULES.md](RULES.md).

## Project

Team repo for the **G-Research Quant Challenge** (ETH Zürich, 27 Oct 2026, 17:30–21:30): a
Python **cointegration / pairs-trading** game on unseen real market data. Goal: highest PnL.

- [PLAYBOOK.md](PLAYBOOK.md) — competition intel, strategy math, event-day runbook
- [PLAN.md](PLAN.md) — what we build and in which order

## Setup and commands

```bash
uv sync                                                    # create .venv with numpy + dev tools
uv run pytest -q                                           # all tests (must pass before any PR)
uv run ruff check .                                        # lint
git config core.hooksPath .githooks                        # once per clone: blocks commits to main

uv run python scripts/make_practice_data.py data/practice_synth.csv   # rehearsal prices
uv run python scripts/make_practice_data.py data/practice_book.csv --book   # rehearsal book + bots
uv run python scripts/diagnose.py data/prices_or_book.csv [--trades t.csv]  # OBSERVE first
uv run python scripts/screen_pairs.py data/prices.csv                  # rank pairs (train/test)
uv run python scripts/sweep_pair.py data/prices.csv --y A --x B        # params by plateau
uv run python scripts/make_submission.py --config cfg.json --out submissions/r1-v1.py
```

## Layout

Trading logic is **not** fixed in advance: diagnose the event data first, then write
`decide()` in `templates/skeleton.py` (or use the pairs module if the data calls for it).

```
templates/skeleton.py  paste-ready shell: state, limits, never-crash wrapper; decide() is empty
templates/research_template.py  write a strategy + grid live, get Sharpe ranking + overfit verdict
src/gqc/diagnostics.py behaviour probes: variance ratio, lead-lag, imbalance, staleness, markouts
src/gqc/overfit.py     PBO (CSCV), PSR / Deflated Sharpe, expected max Sharpe, MinBTL
src/gqc/research.py    generic grid runner for any strategy + overfit report and verdict
src/gqc/core.py        stdlib-only pairs trader (one option, copied verbatim into submissions)
src/gqc/stats.py       OLS, ADF, Engle–Granger (MacKinnon crit), half-life, screening
src/gqc/backtest.py    bar-level backtester: lag, costs, metrics
src/gqc/sweep.py       train/test grid search with plateau scoring
src/gqc/io.py          wide/long CSV loader
src/gqc/synthetic.py   cointegrated / random-walk generators with known truth
scripts/               CLIs above
submissions/           frozen single-file submissions (one-line note each)
tests/                 pytest; tests/hooks/ covers the repo guards
```

## Git rules (hard)

- **Never commit to or push `main`.** Create a branch first: `git switch -c <type>/<topic>`
  (`feat/ fix/ research/ exp/ docs/ chore/`).
- Push only your own branch; open a PR. **Never merge a PR, never approve a PR**, never
  force-push (use `--force-with-lease` on your own branch only), never use `--no-verify`, never
  edit branch protection or the guard hooks to get past them.
- Conventional commits: `feat|fix|refactor|docs|test|chore|perf|ci: <description>`.

## Code rules

- Python ≥ 3.11 for `src/` and scripts; **`src/gqc/core.py` must stay importable on Python 3.8+
  with only the standard library**, use `NamedTuple` (not `@dataclass`) and survive bare
  `exec()` — `scripts/make_submission.py` verifies this.
- Type hints on all signatures; functions < 50 lines; files < 800 lines.
- No lookahead: a decision at step t uses data ≤ t. A test must prove it for new strategies.
- Seed all randomness. No wall-clock logic in strategies.
- Write the failing test first; keep `uv run pytest` green.

## Research rules

- Choose parameters by **plateau of train Sharpe**, never by test PnL; then size to limits.
- Any tuning must print `research.overfit_report` (PBO via CSCV, Deflated Sharpe, MinBTL) over
  **all** configs tried, and the agent must follow its VERDICT. Count every trial.
- Report the conservative setting (lag 1, realistic costs) as the headline number.
- Correlation is not cointegration; many pairs screened ⇒ expect false positives.
- Never commit data (`data/` is gitignored) or organiser material.

## Never

- Place orders through any brokerage/trading API or MCP — market data reads only.
- Commit secrets. Use environment variables.
- Paste a submission that has not gone through `make_submission.py` (it checks stdlib-only + exec).
