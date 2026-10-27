## What

<!-- One paragraph: what changed and why. -->

## Type

- [ ] feat (engine / strategy / tooling)
- [ ] fix
- [ ] research (notebook / analysis, no production code)
- [ ] docs / chore

## Evidence (required for any strategy or engine change)

| Metric | Baseline (`main`) | This PR | Fill model |
|---|---|---|---|
| Total PnL | | | conservative |
| Total PnL | | | queue |
| Max drawdown | | | conservative |
| # fills / avg edge per fill | | | conservative |

- Seeds / days / datasets used:
- Parameter plateau screenshot or table (not a single best point):
- Out-of-sample split used:
- Configs tried on this data (N, all of them):
- PBO (CSCV, S=…): … · Deflated Sharpe: … · MinBTL vs T: … · VERDICT line:

## Checklist (RULES.md)

- [ ] Branch is not `main`; PR targets `main`
- [ ] Tests added/updated and `pytest` passes locally
- [ ] No lookahead (features use only data available at decision time)
- [ ] Hard position-limit guard + exception guard present in any submission code
- [ ] No organizer-provided data or secrets committed
- [ ] Reviewer is someone other than the author (agents never approve or merge)
