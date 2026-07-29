# Dense Five-Minute Signed-Pressure Reconstruction V0

## Outcome

Phase 1 decision: `blocked_dense_pressure_upstream_dependency`.

The exact sparse formula lineage was found. The archived cross-sectional signed-progress
normalization first filtered each session/checkpoint stock slate using availability of three
later bars. Removing those rows, or their being absent, changes membership and therefore the
peer median; mutating later OHLC values while retaining the rows does not. A current-bar-causal
reconstruction cannot preserve the archived values within the binding `1e-12` tolerance. No
alternative pressure definition was created.

Phase 2 was not authorized and the frozen directional experiment was not rerun.

## Lineage and causality

- Formula: equal mean of development-scaled `signed_progress`, `signed_efficiency`,
  `mean_close_location`, and `boundary_slope`.
- Activity field: `historical_relative_activity`, retained as an activity proxy and not
  described as exchange-verified volume.
- Direct order flow measured: no.
- Interpolation, forward fill, and backfill: none.
- Future-dependent dependency: cross-sectional signed-progress slate membership.
- Affected full sparse rows: 428.
- Maximum causal-versus-sparse difference: 0.115501985827478.
- Directional Branch-C affected rows: 103.
- Directional Branch-C maximum difference:
  0.0903177031301969.

## Dense grid and support

- Stocks: 20.
- Sessions: 412.
- Stock-sessions: 8,238.
- Expected checkpoint rows: 280,092.
- Materialised completed-bar rows: 279,906.
- Missing bars: 186.
- Misaligned timestamps: 33.
- Valid exact dense pressure rows: 0.
- Development fresh episodes: 285; complete five-bar windows:
  0.
- Assessment fresh episodes: 253; complete five-bar windows:
  0.

Coverage by stock and month is retained in `dense_pressure_episode_coverage.csv`; no stock or
month has a valid exact five-bar pressure window because the binding upstream dependency
failed before pressure materialization.

## Audit

- Independent audit: passed_fail_closed_blocker_verified.
- Determinism: passed_fail_closed_reconstruction.
- Phase 2 authorization: false.

This is retrospective, research-only feature-lineage work. It is not institutional
accumulation observation, direct order-flow measurement, option P&L, prospective validation,
paper readiness, live readiness, or a deployable strategy.
