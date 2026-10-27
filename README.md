# G-Research-Main

Team repository for the **G-Research Quant Challenge** — a Python algorithmic-trading game built
around **cointegration / pairs trading on unseen real market data**. Goal: highest PnL.

## Documents

| File | What it is |
|---|---|
| [PLAYBOOK.md](PLAYBOOK.md) | Intelligence on the event and comparable competitions, strategy math, bot exploits, event-day runbook |
| [PLAN.md](PLAN.md) | Build plan: backtest engines, strategy library, submission tooling, timeline to the event |
| [RULES.md](RULES.md) | Binding process rules: protected `main`, PR review, research integrity, submission safety |
| [AGENTS.md](AGENTS.md) | Instructions for any AI coding agent: setup, commands, layout, git/code rules |
| [CLAUDE.md](CLAUDE.md) | Claude Code specifics (imports AGENTS.md): guard hook, MCPs, skills |

## Getting started

```bash
git clone https://github.com/GresearchHereMe/G-Research-Main.git
cd G-Research-Main
git config core.hooksPath .githooks        # blocks local commits/pushes to main
uv sync && uv run pytest -q
```

Requirements: Python ≥ 3.11, [uv](https://docs.astral.sh/uv/), `gh` CLI for PRs.

## Event-day kit

```bash
uv run python scripts/make_practice_data.py data/practice_synth.csv     # rehearsal data
uv run python scripts/screen_pairs.py data/practice_synth.csv           # rank pairs; verdict column
uv run python scripts/sweep_pair.py data/practice_synth.csv --y AAA --x BBB   # plateau params + OOS
uv run python scripts/make_submission.py --config submissions/example_config.json \
    --out submissions/round1-v1.py                                      # stdlib-only single file
```

The generated file exposes `on_tick(prices) -> target positions` and `to_orders(targets,
positions)`; adapt only its bottom ADAPTER block to the platform's interface.

## Workflow (short version of RULES.md)

1. `git switch main && git pull --ff-only && git switch -c feat/<topic>`
2. Write a failing test, make it pass, keep commits conventional (`feat: …`, `fix: …`).
3. `git push -u origin feat/<topic>` and `gh pr create` using the PR template.
4. A teammate who is **not the author** reviews and merges. Never merge your own PR.

## Status

| Phase | State |
|---|---|
| P0 Guardrails, playbook, plan | PR #1 |
| Event-day kit: core trader, stats, backtester, sweep, submission builder | PR #2 |
| Rehearsal (today, before 17:30) | todo |
| Order-book engine B, richer strategy library | after the event |

## Layout

```
.claude/        settings.json + hooks/guard_main.py (agent guard)
.githooks/      pre-commit, pre-push (human guard)
.github/        CI workflow, PR template
tests/          pytest suite
src/gqc/        engine + strategies (from P1)
research/       notebooks, event notes
submissions/    frozen single-file submissions
```
