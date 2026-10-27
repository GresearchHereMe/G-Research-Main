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

### Skills and tools

Use the skill map in AGENTS.md §11 — invoke the listed skill via the Skill tool when its
situation applies. Claude-only details:

- **Alpaca MCP:** market data only, `feed: "iex"`; trading/account tools are denied in
  `.claude/settings.json`. Large pulls flood context — fetch small samples or write a script.
- During a live round, prefer speed: 2-minute brainstorming, no plan documents, but never skip
  `evaluate_algo.py` (G0–G4) or `replay_skeleton.py`.

### Working style

- Put findings about the competition into PLAYBOOK.md and trial results into
  `research/LOG.md`, not only into chat.
- When event data arrives, follow the pipeline in AGENTS.md §5 before writing new code.
