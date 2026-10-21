# Stocker Development Guide

## Development philosophy

- Prefer the smallest correct implementation and boring, explicit code.
- Generalise only when a real second use-case requires it; do not build for hypothetical needs.
- Do not introduce microservices, dynamic plugin frameworks, dependency-injection frameworks,
  event buses, generic workflow engines, or elaborate orchestration unless explicitly required.
- Do not add fallback paths for hypothetical failures.
- Never silently repair, estimate, interpolate, or substitute trading-critical input data.
- Keep changes local to the requested stage. Do not refactor unrelated working code.
- Reuse working components where practical and prefer the standard library where adequate.
- Add dependencies only for a clear current requirement.
- Do not optimise for theoretical scale Stocker does not need.

## Reliability

Trading-critical invalid state must prevent a new trade. Examples include a disconnected broker,
stale or invalid required market data, incomplete required history, an unknown broker position, an
incorrect account or environment, and duplicate-order risk.

Non-critical failures should be isolated and logged where safe. A problem with one instrument,
dashboard, chart, export, analytics task, or optional metric should not normally terminate all runs.
A single-symbol failure should normally skip that symbol and continue.

Prefer `invalid input -> skip/reject -> log reason`. Do not build complicated retry, fallback,
approximation, or override trees.

## Testing

Tests protect meaningful behaviour and observed regressions. Prioritise PAPER/LIVE routing,
duplicate-order prevention, risk and position sizing, fill/position state, known strategy examples,
and historically observed failures.

Do not generate large combinatorial suites for remote hypothetical conditions. Tests never belong
in the runtime trading hot path.

## Architecture rules

- One repository and one application/codebase with clear modules.
- One frozen paper record (docs/PAPER-RECORD-PROTOCOL.md and its amendments): the first-candle rule
  on two Saxo index CFDs and QQQ options, Connors on QQQ options, VWAP crossings on the Nasdaq CFD
  (amendment 2), the Connors and Turnaround Tuesday dip rules on the Nasdaq CFD (amendment 3),
  the signals' context, the books, the playbook and the fear rule (amendment 4), and the stock
  dips on US stocks from Saxo's daily bars after the close (amendment 5).
  It reads Saxo prices and writes the record; it
  never places, changes or cancels an order and never changes the Saxo session. This repository
  holds only the runtime, its dashboard, their tests and operations material; retired futures code,
  providers and research live in git history, never in active packages.
- The authenticated dashboard is a consumer of runtime state; its failure must not stop the record.
- Pure frozen calculations (`methods.py`) receive parsed Saxo values and never request broker data.
- Research prices and models must never substitute for Saxo quotes in the record; a missing or
  delayed input is recorded as the reason no trade was made.
- Paper records are separate by data environment; never relabel history. The retired futures
  ledgers stay on the server as they are.
- Calculate shared data and features once where sensible.
- Keep enough local audit information to explain each decision without prematurely building an
  event-sourcing framework.

## Scope discipline

For every task:

1. Identify the requested stage or feature.
2. Make the smallest required change.
3. Run focused relevant checks.
4. Stop when the acceptance criteria are met.
5. Report later considerations instead of implementing them.

Never implement future stages automatically.
