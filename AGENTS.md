# Stocker agent guide

## Repository purpose

Stocker is a personal algorithmic-trading platform.

Its intended controlled lifecycle is:

1. Historical research and backtesting.
2. Prospective market-data recording.
3. Idea and strategy evaluation.
4. Shadow trading.
5. Paper trading.
6. Controlled live trading.

These are distinct operating modes, not interchangeable labels. The repository does
not yet implement every mode safely. Do not treat a planned capability as authorised
or complete.

Read:

- `docs/architecture/stocker-platform-purpose.md`
- `docs/agent-workflow.md`
- the active implementation plan under `docs/plans/`, when one exists

## Active market-data direction

IBKR is the intended active prospective, paper, and live broker/data source.

Do not reintroduce EODHD, cross-vendor matching, source-transfer gates, or
provider-equality requirements into the active runtime unless the user explicitly
reverses this decision. Historical datasets may continue to exist separately for
research.

## Operating modes

### Research

Research may process historical or live data and produce observations, signals, and
proposed trades. Research must not transmit broker orders.

### Shadow

Shadow mode may process live data and record virtual positions and outcomes. Shadow
mode must not transmit broker orders.

### Paper

Paper mode may transmit approved orders only to an explicitly configured paper account
through the dedicated execution layer. Paper configuration must never fall through to
live trading.

### Live

Live mode is a legitimate future Stocker capability, but it must remain disabled by
default. Live trading requires:

- explicit operator authorisation
- an allowlisted live account
- server-side risk approval
- idempotent order intents
- broker-order reconciliation
- broker-position reconciliation
- emergency-stop controls
- conspicuous LIVE mode identification
- tests proving paper/live separation

A missing or invalid live configuration must fail closed.

## Architectural boundary

Preserve this direction:

Market data
→ idea plugin
→ signal or proposed trade
→ portfolio and risk approval
→ approved order intent
→ paper or live execution adapter
→ broker acknowledgement, fills, and reconciliation

Idea plugins must never:

- possess an IBKR client
- submit, modify, or cancel broker orders
- read broker credentials
- choose the broker account
- change operating mode
- bypass portfolio or risk approval
- treat a proposal as an approved order

Only the dedicated execution subsystem may possess order-capable broker access. The web
application is an operator interface, not the authoritative owner of order, fill,
position, or risk state.

## Trading-sensitive changes

Do not change any of the following unless the user explicitly requests that specific
change:

- paper or live enablement
- broker credentials
- account allowlists
- order-transmission settings
- maximum order value
- maximum position size
- gross or net exposure limits
- daily-loss limits
- emergency-stop behaviour
- paper/live environment selection

Never commit populated credentials or account identifiers. Automated tests must never
transmit live orders.

## Idea architecture

New ideas should use a generic plugin contract. Do not add idea-specific primary UI
tabs, core API routes, core database tables, broker integrations, execution paths, risk
bypasses, or frontend renderers.

A plugin may emit observations, signals, proposed positions, or proposed trades. It may
not approve or execute them. Plugin failures should remain isolated from ingestion and
other plugins.

## Order and reconciliation rules

Every executable order must originate from a durable, approved, uniquely identified
order intent. Retries must not create duplicate broker orders.

Never infer a fill merely because an order was submitted. Never infer a broker position
solely from a local intended position. Reconcile an uncertain order state before
submitting another potentially duplicate order.

Persist enough information to explain:

- which idea and version proposed the trade
- which data produced it
- which risk checks approved or rejected it
- which account and mode were selected
- which broker order identity was assigned
- which acknowledgements and fills occurred
- how the resulting position was reconciled

## Protected data

Do not use protected prospective, shadow, paper, or live results for retrospective
fitting, feature selection, threshold selection, or parameter selection without an
explicit authorised research protocol.

Keep development, retrospective assessment, stress, prospective, paper, and live data
boundaries explicit. Do not silently open a protected period.

## Storage

Maintain one authoritative writer for each active operational database. Permanent
storage and backups must remain bounded. Backups must be integrity checked, compressed,
rotated, and size limited.

Do not commit databases, backups, raw market evidence, generated reports, exports,
broker credentials, or account identifiers. Do not modify a legacy operational
database in place during migration.

## Simplicity

Prefer the smallest dependable implementation. Do not introduce infrastructure for
hypothetical requirements.

Avoid unnecessary services, generalised workflow engines, message buses without a
demonstrated requirement, duplicated safety projections, compatibility wrappers that
preserve dead architecture, giant central modules, and idea-specific wiring through
every layer.

Delete superseded runtime code when an accepted implementation plan requires it.

## Agent workflow

For substantial or trading-sensitive work:

1. Use an Architect first.
2. Save the accepted plan under `docs/plans/`.
3. Implement one bounded phase.
4. Run focused tests.
5. Use a separate read-only Reviewer.
6. Resolve blocking findings.
7. Run failure-oriented tests.
8. Perform a final review.

Do not run multiple write-capable agents simultaneously against the same operational
schema, risk engine, execution state machine, broker adapter, reconciliation path, or
migration cutover. Read-heavy exploration may be parallelised.

## Before editing

- Inspect the actual implementation and tests.
- Read the active plan.
- Identify affected operating modes.
- Identify whether market data, risk, execution, or reconciliation is affected.
- State whether the change can influence paper or live trading.
- Keep work within the assigned phase.

## Testing

When market-data code changes, test ordering, staleness, gaps, duplicate callbacks, and
reconnection.

When strategy code changes, test determinism, leakage, parameter identity, mode
restrictions, and plugin isolation.

When risk code changes, test every affected rejection condition, boundary values, stale
state, duplicate intents, and emergency stop.

When execution code changes, test:

- paper/live account separation
- idempotent submission
- duplicate prevention
- submission timeout with uncertain broker outcome
- partial fills
- rejection
- cancellation races
- reconnection
- process restart
- open-order reconciliation
- position reconciliation

Use fake, replay, simulation, or explicitly isolated paper adapters. Never make a real
live trade during automated testing.

Repository-wide checks are `bash scripts/check.sh`; tests alone are
`bash scripts/test.sh`. Prefer focused `uv run pytest <test-path>` commands while
iterating.

## Completion report

Report files changed, operating modes affected, tests actually run, tests not run,
paper/live implications, migration implications, and known limitations.

Do not claim completion while a plugin or frontend path can bypass risk, paper mode can
reach a live account, retries can duplicate orders, or local broker state can silently
diverge.
