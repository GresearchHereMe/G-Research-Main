# CLAUDE.md

Instructions for Claude Code (and other agents) in this repository.

## Project

Team repo for the **G-Research Quant Challenge**: a Python pairs-trading / cointegration game on
unseen real market data. We build (1) a rigorous backtest engine, (2) a strategy library, and
(3) paste-ready single-file submissions. Read before working:

- [PLAYBOOK.md](PLAYBOOK.md) — competition intelligence and strategy math
- [PLAN.md](PLAN.md) — phases, interfaces, acceptance criteria
- [RULES.md](RULES.md) — process rules (binding)

## Git rules (hard — enforced by `.claude/hooks/guard_main.py`)

- **Never work on `main`.** Before the first edit: `git switch main && git pull --ff-only && git switch -c <type>/<topic>`.
- Commit and push **only your feature branch**; open PRs with `gh pr create`.
- **Never** `gh pr merge`, never approve a PR, never push to `main`, never force-push
  (use `--force-with-lease` on your own branch if unavoidable), never `--no-verify`,
  never touch branch protection or rulesets.
- If the guard blocks you, stop and tell the user. Do not rephrase the command to slip past it.
- Conventional commits: `feat|fix|refactor|docs|test|chore|perf|ci: <description>`.

## Code conventions

- Python ≥ 3.11, type hints on every signature, PEP 8, ruff.
- Frozen dataclasses for data; strategies are pure `step(obs, state) -> (decision, state)`.
- No lookahead: code at step t reads only data ≤ t. Seed all randomness.
- Functions < 50 lines, files < 800 lines, organise by domain (`data/`, `stats/`, `backtest/`, `book/`, `strategies/`, `submit/`).
- Submission code: single file, stdlib (+ numpy if allowed), limit guard, exception guard, bounded memory.
- TDD: write the failing test first. Target 80% coverage for `src/`.

## Commands

```bash
uv run --no-project --with pytest pytest tests -q        # tests (until pyproject lands in P1)
git config core.hooksPath .githooks                      # enable local git hooks once per clone
```

## Tools to use for this project

- **Alpaca MCP — market data only** (`get_stock_bars`, `get_stock_quotes`, `get_stock_trades`,
  `get_crypto_bars`; use `feed: "iex"`). Trading/account tools are denied; never request them.
  Save fetched data under `data/` (gitignored).
- **context7 / docs-lookup** for statsmodels, numpy, scipy API details before coding against them.
- **WebSearch / Exa** for research; cite sources in PLAYBOOK.md.
- **Skills:** `quant-analyst` (strategy and backtest checklist), `python-patterns`,
  `python-testing`, `tdd-workflow`, `dataviz` (research charts), `verification-loop` (before PRs).
- **Agents:** `python-reviewer` after writing Python; `tdd-guide` for new features.

## Research hygiene

- Headline numbers use the conservative cost/fill model; report baseline vs change.
- Log runs to `runs/` (gitignored); choose parameters from plateaus; never tune on test windows.
- Put findings about the competition into PLAYBOOK.md, not into chat only.
