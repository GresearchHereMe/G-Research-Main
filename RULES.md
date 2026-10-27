# RULES

Non-negotiable rules for every contributor — humans and AI agents. If a rule blocks you, stop
and ask the team; do not work around it.

---

## 1. `main` is protected

1. **Nobody commits or pushes to `main` directly.** Every change lands through a pull request.
2. Branch from an up-to-date `main`: `git switch main && git pull --ff-only && git switch -c <type>/<topic>`.
   Types: `feat/`, `fix/`, `research/`, `exp/`, `docs/`, `chore/`. Topic in kebab-case.
3. One PR = one concern. Keep it reviewable (< ~400 changed lines when possible).
4. **Merge requires:** CI green **and** at least one approval from a teammate who is **not the
   author**. **Never merge your own PR.** The reviewer merges (squash merge), then deletes the branch.
5. No force-push to `main`, ever. On your own branch use `--force-with-lease` only.
6. Never bypass hooks (`--no-verify`) or weaken any enforcement layer (§8).

## 2. AI agents (Claude Code and others)

Agents follow §1 plus:

- Create a feature branch **before the first edit**. Never edit on `main`.
- May: commit to their branch, push their branch, open PRs (`gh pr create`), comment.
- **Must never:** merge a PR, approve a PR, push to `main`, force-push, change branch
  protection/rulesets, disable hooks, or delete `main`.
- If the guard hook blocks a command, report it — do not rephrase the command to get around it.
- **Brokerage MCPs are read-only.** Market-data calls (e.g. Alpaca bars/quotes) are fine; order,
  position and account-changing tools are denied in `.claude/settings.json` and must stay denied.
- Run the tests before claiming work is done; report failures verbatim.

## 3. Commits and pull requests

- Conventional commits: `<type>: <description>` with `feat|fix|refactor|docs|test|chore|perf|ci`.
- Fill in the PR template. Any strategy or engine change includes the **evidence table**
  (baseline on `main` vs this PR, conservative fill model as headline, seeds/datasets listed).
- Tests accompany code (TDD preferred; 80% coverage target for `src/`).

## 4. Data, secrets, confidentiality

- Never commit market data, organiser-provided code/data, or credentials. `data/` is gitignored.
- Treat organiser materials as confidential unless told otherwise. Keep the repo **private**.
- Secrets only via environment variables (e.g. `ALPACA_API_KEY`); never in code or notebooks.
- Strip notebook outputs that contain data before committing.

## 5. Research integrity

1. **No lookahead.** A decision at step t uses only data available at t.
2. Declare train / validation / test windows **before** looking at results; never tune on test.
3. Log every configuration tried (`runs/` registry), not just the winner.
4. Pick parameters from **plateaus**, not peaks. Fewer parameters beat more.
5. Every edge needs a written *why* (mechanism), not only a backtest number.
6. The **conservative** fill/cost model is the headline number; optimistic is context only.
7. Correlation is not cointegration; screening many pairs needs multiple-testing discipline.
8. **Objective: select on Sharpe, then size to limits** (maximum PnL per unit of risk).
9. **Every tuned strategy reports PBO (CSCV) and the Deflated Sharpe** over *all* configs tried
   on that data, plus MinBTL, and obeys the verdict: PBO ≤ 0.2 and DSR ≥ 0.95 → use tuned
   params; PBO ≤ 0.5 and DSR ≥ 0.5 → plateau params, smaller size; else reject the tuning.
   Count every trial, including manual tweaks (PLAYBOOK §7.1).

## 6. Submission safety

Every file pasted into the competition platform must:

- be a **single file**, standard library (+ `numpy` only if confirmed allowed);
- contain a hard position-limit guard and a top-level `try/except` that returns no orders;
- use bounded memory (fixed-length windows), no file/network I/O, no wall-clock logic;
- pass `submit/dryrun.py` on held-out data;
- be frozen in `submissions/round-N-vK.py` with a one-line note **before** it is pasted.

## 7. Competition conduct

- Follow the organisers' rules; when unsure, ask them.
- Exploiting **market behaviour** (mispricings, predictable bots) is the game. Exploiting
  **platform bugs, timing artefacts or infrastructure** requires organiser approval first.
- No sharing code or signals with other teams during the event.

## 8. Enforcement layers

| Layer | Protects against | Where | Status |
|---|---|---|---|
| GitHub branch protection / ruleset | direct pushes, self-merge, unreviewed merges | repo settings | **Unavailable**: org is on the free plan and the repo is private (API returns 403 "Upgrade to GitHub Pro"). Decision 2026-10-27: no upgrade; rely on the layers below. |
| `.githooks/pre-commit`, `.githooks/pre-push` | human commits on / pushes to `main` | local, enable with `git config core.hooksPath .githooks` | active once enabled per clone |
| `.claude/hooks/guard_main.py` | agent commits/pushes/merges/approvals on `main` | `.claude/settings.json` PreToolUse hook | active |
| Alpaca trading-tool deny list | agents placing real/paper orders | `.claude/settings.json` permissions | active |
| CI (`pytest`) | broken code reaching review | `.github/workflows/ci.yml` | active on PRs |

Until server-side protection exists, §1.4 ("never merge your own PR") is enforced by
discipline and review. Anyone who sees a self-merge reverts it and raises it with the team.
