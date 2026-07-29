# Frozen M1C Prospective Signal and Bid/Ask Microstructure Recorder V0

This directory freezes the scientific and operational contract for Stocker’s first causal movement recorder.

The runtime is prospective, research-only, record-only, and broker-mutation-free. M1C is the primary frozen movement gate at `0.488333710794033`. A1 is an unvalidated prospective hypothesis; C1 and R1 are unvalidated comparisons. Microstructure scores are descriptive and have no action threshold.

The quiet-state extension preserves the original decision `blocked_insufficient_low_tail_support`. It records the exact frozen M1C bottom-5, bottom-10, and bottom-20 classifications, deterministic neutral controls, and conservative long-premium and defined-risk short-premium shadow outcomes. It neither relaxes the retrospective 35% gate nor enables paper or live orders.

## Chronology

The first successful live activation writes an immutable activation record with UTC/New York timestamps, Git SHA, configuration hash, model hashes, official IBKR API version, and Gateway/TWS version. Raw and derived prospective writes reject timestamps before that activation. Historical evidence is admitted only for frozen preprocessing, parity, D-1 context, fixtures, replay, and tests.

## Runtime

The existing `stocker_prospective` service is extended rather than replaced. With `paths.frozen_m1c_artifact_root` configured, `stocker-prospective recorder run` starts the frozen live path. It uses the official TWS socket API, exact qualified contracts, protected Level I/five-minute bars for the 20-stock cohort, VTI, and seven frozen sector ETFs, ranked BidAsk/Last/depth promotion, bounded option top-of-book capture, append-only Parquet, the existing SQLite metadata store, and the existing read-only FastAPI/web application.

Startup is intentionally fail-closed when a frozen artifact is not one of the three exact preregistered M1C hashes, or when a parity report, bar-compatibility report, D-1 Group O package, historical activity baseline, Gateway/TWS version, or required exact stock/proxy contract is absent.

## Evidence

- `m1c_live_parity_report.json`: 250-row frozen inference parity.
- `direction_live_parity_report.json`: 417 assessment episodes, including at least 200 per archetype.
- `independent_audit.json`: 100 M1C probabilities, 100 rows per archetype, 100 microstructure windows, 50 shadow option outcomes, and two identical 200-event replays.
- `quiet_state_independent_audit.json`: 100 quiet-state predictions, 50 iron butterflies, 50 iron condors, deterministic controls, option legs, conservative fills, phase boundaries, and broker/route safety.
- `quiet_state_determinism_check.json`: two identical quiet-state fixture replays with zero mismatches and maximum floating difference `0`.

No live market data, account identifiers, credentials, positions, balances, orders, or broker logs belong in this directory.
