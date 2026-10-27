# G-Research-Main

Team repository for the **G-Research Quant Challenge** — a Python algorithmic-trading game built
around **cointegration / pairs trading on unseen real market data**. Goal: highest PnL.

## Documents

| File | What it is |
|---|---|
| [PLAYBOOK.md](PLAYBOOK.md) | Intelligence on the event and comparable competitions, strategy math, bot exploits, event-day runbook |
| [PLAN.md](PLAN.md) | Build plan: backtest engines, strategy library, submission tooling, timeline to the event |
| [RULES.md](RULES.md) | Binding process rules: protected `main`, PR review, research integrity, submission safety |
| [CLAUDE.md](CLAUDE.md) | Instructions for Claude Code / AI agents working in this repo |

## Getting started

```bash
git clone https://github.com/GresearchHereMe/G-Research-Main.git
cd G-Research-Main
git config core.hooksPath .githooks        # blocks local commits/pushes to main
uv run --no-project --with pytest pytest tests -q
```

Requirements: Python ≥ 3.11, [uv](https://docs.astral.sh/uv/), `gh` CLI for PRs.

## Workflow (short version of RULES.md)

1. `git switch main && git pull --ff-only && git switch -c feat/<topic>`
2. Write a failing test, make it pass, keep commits conventional (`feat: …`, `fix: …`).
3. `git push -u origin feat/<topic>` and `gh pr create` using the PR template.
4. A teammate who is **not the author** reviews and merges. Never merge your own PR.

## Status

| Phase | State |
|---|---|
| P0 Guardrails, playbook, plan | in review |
| P1 Foundations (package, data, synthetic generator) | next |
| P2 Stats core · P3 Backtest engine A · P4 Strategies | planned |
| P5 Order-book engine B · P6 Submission tooling + rehearsal | planned |

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
