# Loop payoff phase and path V2

Date: 2026-07-13

Decision: **`phase_hazard_features_not_supported_as_payoff_admission_discriminator`**

Scientific status: post-inspection causal retrospective development whose purpose is to specify a prospective logging contract. This is not validation.

Safety:

- `research_only: true`
- `live_ordering_enabled: false`
- `order_placement: disabled`
- no trading app, broker, paper/demo account, deployment, position, or order path was changed or used
- provider volume remains labelled `historical_volume_activity_proxy`; it was not used here

## Direct answer

Regime age and a 2024-frozen probability that the current admission regime exits before the frozen anchor+24 close do **not** distinguish profitable from losing occurrences reliably. Negative-hazard payoff AUC was 0.495–0.545 across the four candidate-period cells; every 95% interval crossed 0.5 and all four Holm-adjusted endpoints failed.

The path result is more useful: **80.0%–93.6% of final losers nevertheless achieved a favorable excursion greater than the 10 bps round-trip cost before finishing negative.** Their median favorable peak occurred 2–5 bars after admission, while their median adverse extreme occurred 15–18 bars after admission. This makes exit timing the next research priority, but MFE is hindsight and cannot itself become an exit rule.

## Data-seal decision

No genuinely unseen, still-sealed set was available. The common provider files extend through 2026-06-29, repository research outputs already span 2026, and no seal predating these two post-inspection hypotheses exists. Therefore:

- 2023 and 2025 remain opened retrospective development evidence;
- no prospective validation or economic-edge claim is made;
- the deliverable is an immutable prospective logging contract, not a strategy.

## Frozen population and correction trail

Primary V2 exactly reused the parent's 190-session score surface after excluding each period's first 60 completed provider sessions.

| Candidate | 2023 rows / mean net | 2025 rows / mean net |
|---|---:|---:|
| `cycle_04|state4` | 132 / +26.54 bps | 96 / +30.75 bps |
| `cycle_07|state5` | 722 / +7.02 bps | 713 / +17.04 bps |

An earlier V1 implementation mistakenly included all 250 sessions. It was preserved as an exploratory all-session diagnostic and is not the primary result. V2 changed only the 60-session population filter after V1 results had been opened; hypotheses, features, costs, bootstrap, bins, and decision gates were unchanged. V2 reproduced all four parent counts and means exactly.

Two no-output V1 pre-score attempts also exposed coordinate conventions before any result artifact was emitted: state-run positions are global within symbol, while provider execution advances from the timestamp-matched anchor row. Both corrections are documented in the V1 contract and failed-attempt records.

## Phase and hazard result

Hazard is the 2024 state-specific empirical probability that a regime exits before the frozen close, conditional on surviving to the admission age. Session-terminal runs were excluded from fitting. Realized total duration and future state were forbidden as admission features.

| Candidate | Period | AUC: lower hazard predicts positive payoff | 95% block interval | Holm-adjusted p | Orientation-survival net difference |
|---|---:|---:|---:|---:|---:|
| `cycle_04|state4` | 2023 | 0.545 | 0.461–0.631 | 0.590 | +7.94 bps |
| `cycle_04|state4` | 2025 | 0.497 | 0.429–0.564 | 1.000 | -58.76 bps |
| `cycle_07|state5` | 2023 | 0.497 | 0.463–0.532 | 1.000 | -54.63 bps |
| `cycle_07|state5` | 2025 | 0.495 | 0.459–0.532 | 1.000 | +11.23 bps |

Only the support gate passed. Association direction, multiplicity, orientation-survival sign, and leave-one-stock-out gates failed. Mean hazard was also extremely high for both profitable and losing rows (roughly 0.96–0.99), so a scalar “will the current regime survive until anchor+24?” variable is nearly saturated and does not describe the intervening loop path.

Matched opposite orientations were informative only for `cycle_07`: state 6 averaged -35.52 bps in 2023 and -33.71 bps in 2025, versus +7.02 and +17.04 bps for state 5. The `cycle_04` controls had only 8 and 6 rows, so they cannot support a comparison. These are descriptive opened-data observations, not validation.

## Payoff anatomy

“Timing failure” means final net payoff was nonpositive but gross MFE exceeded the 10 bps round-trip cost. “No usable move” means the final payoff was nonpositive and MFE never cleared 10 bps.

| Candidate | Period | Final positive | Timing failure | No usable move | Timing failures among losers |
|---|---:|---:|---:|---:|---:|
| `cycle_04|state4` | 2023 | 50.8% | 39.4% | 9.8% | 80.0% |
| `cycle_04|state4` | 2025 | 55.2% | 40.6% | 4.2% | 90.7% |
| `cycle_07|state5` | 2023 | 48.3% | 47.9% | 3.7% | 92.8% |
| `cycle_07|state5` | 2025 | 49.4% | 47.4% | 3.2% | 93.6% |

For timing failures:

| Candidate | Period | Mean MFE | Mean MAE | Median bars to MFE | Median bars to MAE |
|---|---:|---:|---:|---:|---:|
| `cycle_04|state4` | 2023 | +112.15 bps / +1.29 ATR | -237.06 bps / -2.84 ATR | 3 | 18 |
| `cycle_04|state4` | 2025 | +114.60 bps / +1.39 ATR | -253.86 bps / -3.08 ATR | 5 | 18 |
| `cycle_07|state5` | 2023 | +153.60 bps / +1.87 ATR | -360.33 bps / -4.25 ATR | 2 | 15 |
| `cycle_07|state5` | 2025 | +142.12 bps / +1.54 ATR | -331.73 bps / -3.46 ATR | 2 | 16 |

Final winners reached MFE much later, at a median 16–18 bars. This is a bifurcated path, not evidence for a universal early exit: early peaks characterize hindsight losers, while winners often need most of the window. A fixed “exit after 2–5 bars” rule would therefore be a post-inspection error.

The entry-bar OHLC path has unknown intrabar ordering. Conservative post-entry-bar excursions are retained at signal level; no conclusion relies on order-book, quote, or tick data.

## Interpretation for “the right loop at the right time”

The three questions remain separate:

1. Regime forecasting remains strong and was not retested.
2. Parent-loop occurrence/orientation remained frozen and was not promoted from payoff evidence.
3. Payoff timing remains unresolved. Regime age/hazard did not solve it, but the path shows that most losses contain a usable move that is later surrendered.

The next experiment should therefore test a predeclared, causally observable **path-state exit family**, not another admission dictionary. Candidate mechanisms to freeze before a new seal are:

- detectable parent-loop completion at completed-bar close, with a next-open research exit;
- running favorable excursion and retracement expressed in prior ATR, logged causally at each completed bar;
- current predicted-loop probability decay or orientation invalidation after admission;
- positive / negative / unknown exit-state classification with uncertainty and abstention.

Do not choose retracement thresholds, completion exceptions, or early-exit bars from these opened MFE results. First log the whole causal path prospectively under one immutable contract.

## Integrity and reproducibility

- V2 pre-score manifest SHA-256: `f9ac94849cbfad4765c19ad9cbd9120e910545b8d60dd910fa4dd13ffd2f67d1`.
- V2 contract SHA-256: `2a50f45637b9d4263b8dcbe4856b23303f480379c785d8a393dd7717464994a2`.
- V2 runner SHA-256: `db57a88401156cd63439eac44daff372d20d9da6a7b4d1bdcf37d8470d864b3c`.
- V2 independent auditor SHA-256: `b84651d4a382fbe42fbae17a7a26d3b90f3d6e780b4b5c4084562baff8c721cd`.
- Complete primary artifact-manifest SHA-256: `e76f47deba4bca555f7cdc5bc281217cc7c26ebeae8d61d7fda5fde973c77360`.
- Independent audit: 10/10 checks passed.
- Candidate population count error: 0.
- Maximum net-return, hazard, duration-percentile, and admission-state replay errors: 0.
- Maximum aggregate metric error: `5.55e-17`.
- Exact rerun: all 14 files matched byte-for-byte.
- Focused tests: 8/8 passed.
- The pre-existing dirty `StockerLocal` worktree was not modified; its final status matched the starting status.

Primary artifacts:

`work/artifacts/20260713-loop-payoff-phase-path-v2/primary`

Exact rerun:

`work/artifacts/20260713-loop-payoff-phase-path-v2/exact_rerun`
