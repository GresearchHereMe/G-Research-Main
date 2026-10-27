# CLAUDE.md

Instructions for Claude Code in this repository. Shared agent rules (setup, commands, layout,
git and code rules) live in AGENTS.md and are imported here:

@AGENTS.md

## Claude-specific

### Guard hook

`.claude/hooks/guard_main.py` runs before every Bash call and blocks commits/merges on `main`,
pushes to `main`, force pushes, `--no-verify`, `gh pr merge`, `gh pr review --approve` and
branch-protection changes. If it blocks you, stop and tell the user. Never rephrase a command
to slip past it, and never edit the hook or `.claude/settings.json` to weaken it.

### Tools to use for this project

- **Alpaca MCP — market data only** (`get_stock_bars`, `get_stock_quotes`, `get_stock_trades`,
  `get_crypto_bars`; use `feed: "iex"`). Trading/account tools are denied in
  `.claude/settings.json`; never request them. Large pulls flood context: prefer small samples.
- **context7 / docs-lookup** for statsmodels, numpy, scipy API details before coding against them.
- **WebSearch / Exa** for research; cite sources in PLAYBOOK.md.
- **Skills:** `quant-analyst` (strategy and backtest checklist), `python-patterns`,
  `python-testing`, `tdd-workflow`, `dataviz` (research charts), `verification-loop` (before PRs).
- **Agents:** `python-reviewer` after writing Python; `tdd-guide` for new features.

### Working style

- Put findings about the competition into PLAYBOOK.md, not only into chat.
- When the event data arrives, run `screen_pairs.py` → `sweep_pair.py` → `make_submission.py`
  (AGENTS.md commands) before writing any new code.
