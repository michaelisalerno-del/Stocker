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
- Multiple independent runs may execute concurrently. The user selects Market -> Method -> Run.
  A run saves its method version/specification, method-produced universe, environment and state.
- PAPER and LIVE execution are explicitly separated.
- The future dashboard is not part of the trading engine; dashboard failure must not affect trading.
- A Method owns how it finds, qualifies, vetoes, enters, manages and exits trades.
  Method packages supply universe/search and data services plus pure decision logic.
  New Session HARD candidate selection is Range5 HIGH250 -> RV10 HIGH50 -> RV15 HIGH30.
  Universe acquisition is separate; use saved broad eligible populations and explicit capacity
  diagnostics, never a silent legacy activity/scanner shortlist fallback. See docs/candidate-discovery.md.
  Legacy discovery profiles and audits remain intact for original saved runs.
  Candidate scores, scanner ranks and Session HARD trading scores must stay distinct.
  Pure calculations receive data; shared IBKR execution submits method-produced intentions.
- Account exposure, permissions, reconciliation and emergency controls remain shared.
- Add a method through the catalogue and explicit composition seam described in docs/ARCHITECTURE.md;
  do not scatter method-name conditionals across UI and runtime.
- PRE calculations receive bars as input and do not communicate with IBKR.
- Production and paper PRE history originates exclusively from IBKR. A local cache may store only
  IBKR-originated history and is not an alternative source.
- Never substitute another vendor's history for production or paper PRE calculations.
- Calculate shared data and features once where sensible; do not duplicate them for PAPER/LIVE or
  multiple strategies using the same snapshot.
- The broker's actual position is authoritative for broker positions.
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
