# Scanner-assisted acquisition: completion and Gateway follow-up

The scanner-assisted acquisition implementation is retained. This follow-up fixes the actual
Gateway blocker found during the PAPER access check: opening movement now resolves only to
capability-advertised `TOP_OPEN_PERC_GAIN` and `TOP_OPEN_PERC_LOSE`, never overnight gap scans.
The saved Gateway regression failed on US/LSE/ASX before the fix and passes afterward.

Current method remains `SESSION_HARD_CAUSAL_Q1_ACQUISITION_V9`; acquisition recipe remains
`SESSION_HARD_IBKR_ACQUISITION_EXPERIMENT_V1`. This is a resolver bug fix to the declared
since-open families, not a change to the recipe, an optimization using misses/P&L, or a new
evaluation period. Recipe fields, capacities, sweeps and method specification hashes are unchanged.
Legacy V7/V8 specifications and behavior remain intact.

## Completion report

| Requirement | Implemented / observed result |
|---|---|
| 1. Actual capabilities | PAPER Gateway API 178 advertised 527 codes, 156 locations and 1,153 filter-field codes. Relevant exact codes: TOP_TRADE_RATE, TOP_VOLUME_RATE, HOT_BY_VOLUME, TOP_OPEN_PERC_GAIN, TOP_OPEN_PERC_LOSE. Raw XML, retrieval time, version and digest persist. |
| 2. Components | Trade rate, volume rate, unusual volume, positive since-open percentage movement and negative since-open percentage movement. Only advertised location/instrument/filter/code combinations may execute. |
| 3. Coverage | Every family has UNCAPPED, BELOW_MICRO (<$50m), MICRO, SMALL, MID, LARGE and MEGA. Canonical cap boundaries and existing broker FX conversion; no price/volume floors or cap admission rule. |
| 4. Sweeps | OPEN+60/+180/+240 active seconds from the canonical exchange session. Union accumulates across sweeps and seals before OPEN+5. |
| 5. Concurrency | Two component requests per sweep; the existing broker ten-scanner ceiling and message limits remain. The actual diagnostic used at most two. |
| 6. Pool | Append-only conId union, saved broad-membership/eligibility checks, exact scanner provenance, no default pool cap or symbol truncation. Scanner ranks never become Range/RV/HARD ranks. |
| 7. Population | No promised eligible pool count. After-hours US diagnostic returned 1,058 unique raw contracts before membership/eligibility, not a validated opening population. LSE 669; ASX zero at observation time. |
| 8. OPEN+5 requests | Production requests only the acquired identities, minus shared/cache hits; then only 250 and 50 survivors. No opening-history requests were made in the access diagnostic. Actual opening request count remains unmeasured. Broad count is persisted as the all-market workload comparator. |
| 9. Gateway throughput | Access check: 355 actual scanner requests, 70 shared US-profile requests, 8,492 raw hits, 107.167 seconds total. Scanner p50/p90/p95/p99: 581.9/752.6/860.2/1,029.3 ms. All 355 had scannerDataEnd and cancellation; no scanner request failed. These are not historical-bar throughput measurements. |
| 10. Range5 completion | No real opening-session completion time measured. Intended cutoff and actual retrieval/calculation timestamps are separate persisted facts. |
| 11. Before RV10 | Still unproven operationally. Exact +5 bars must finish selection before +10 under the inherited V8 transport window. Deadline failure degrades the run rather than changing the list. |
| 12. Background oracle | After regular close, one broad identity per step on the existing bounded history ingress; yields/cancels for enabled market windows and foreground history, outside scheduler locks. |
| 13. Full-market truth | Saved broad membership, exact IBKR conId/TRADES/RTH opening 15-minute prefixes, same frozen Range250→RV50→RV30 calculations and deterministic ties. Missingness and denominator retained. Separate AUDIT_ONLY persistence. |
| 14. Recall | Range250/RV50/RV30 captured/available ratios, mean/median/worst day/exact days, missed identities/ranks and Range250 rank buckets implemented. No live-session oracle recall measured yet. |
| 15. Contributions | Per-target component/cap/sweep/rank provenance and unique component contribution implemented. Five predefined shadow masks reuse observations; no extra scanner requests or P&L optimization. Real oracle-target contribution counts remain unmeasured. |
| 16. Persistence/UI/API | Existing additive acquisition tables store capabilities, broad membership, components, hits, union, request metrics and oracle ranks. SQL summaries remain compact; paginated `/api/runs/{run_id}/acquisition` provides details. No new schema/UI change was needed for this resolver fix. |
| 17. Failure | Unsupported codes/locations/filters, FX, late/interrupted sweeps, missing prefixes or deadline failures remain explicit. Warning 492 stays request-scoped. No old shortlist, fabricated score, later repair, replenishment or same-day oracle feedback. |
| 18. Validation | Full pytest: 1,031 passed, 14 opt-in Gateway tests skipped, 10 existing warnings. Acquisition/diagnostic subset: 24 passed. Mypy: 52 source files pass. Changed-file Ruff passes; repository Ruff retains 47 pre-existing findings in two untouched research files. JS syntax and mocked-browser summary/acquisition regression pass. |
| 19. Candidate formulas | Unchanged; exact saved US scores, missingness, ordering and 250/50/30 memberships pass on three sessions. |
| 20. HARD rules | Qualification, MODEL_T0, Q1, MID/cohort, entry, exits and shared execution/account rules unchanged. Frozen artifacts and method definitions byte-identical to release 64b6e89. |
| 21. Orders | No broker orders placed. All order tests use fakes. The real access check used execution-disabled PAPER client 292 and isolated audit files. |
| 22. Remaining research | Prospective opening sessions must measure eligible union size, qualification and one-minute history throughput/finality, deadlines, full-oracle recall and component contributions. Non-US permissions/FX/location issues also remain. Acquisition is not validated or frozen. |

The corrected matrix was compared offline against the complete saved real Gateway access
report. All 425 supported market/component instances match successful observed request
parameters exactly, including 70 shared US instances (355 actual requests). This verifies
request construction, not opening recall. Korea had 30 cap components blocked by an unusable
USD/KRW quote; JSE's configured location was not advertised. Nine non-US markets received
precision warning 492; the capability matrix cannot establish paid account subscriptions.
See [the timestamped access report](scanner-access-check-20260910.md) for the full market table.

## Data and strategy boundaries

First-stage feature remains `(max(H0..H4)-min(L0..L4))/O0`; the RV features remain
`sqrt(log(C0/O0)^2 + sum(log(Cj/Cj-1)^2))` over the exact first 10/15 active minutes.
Only final TOP30 receives expensive required prior-history/PRE preparation and enters the
existing stateful Session HARD engine. Stock-local prior history is preserved. The real
runtime fake-broker integration still verifies 533 broad references → 350 acquired → 250 →
50 → 30, with only those final 30 entering history, signals and cohort state.

No new broad tick-by-tick subscriptions, history stack, data manager or strategy engine is
introduced. Discarded identities receive no later opening requests; shared broker resources
and the existing entry-feed budget remain unchanged. Calendar tests retain US/LSE/ASX DST
and active-minute behavior across all 14 existing profiles.

The optional five-minute transport parity experiment remains delayed/audit-only. Production
continues exact one-minute transport. No real transport parity sample was measured.

## Files and deployment boundary

This follow-up changes `stocker_execution/scanner_acquisition.py`, its existing tests, the
saved Gateway projection fixture, and architecture/candidate documentation. The acquisition,
persistence, oracle, benchmark and dashboard implementation from the preceding changes is
reused; see [the original implementation report](scanner-acquisition-implementation.md).

No saved run is migrated or activated by the patch. Any deployment uses the existing backup
procedure and preserves configuration, risk settings, historical data and PAPER-only controls.
A code deployment does not establish prospective acquisition feasibility.

Scanner-assisted acquisition is not a trading filter. It is a high-recall data-acquisition mechanism upstream of the frozen Range250 → RV50 → RV30 chain.

The full-market oracle is delayed audit-only information and can never affect the same day's strategy decisions.

RANGE5 HIGH TOP250 → RV10 HIGH TOP50 → RV15 HIGH TOP30 remains unchanged.
