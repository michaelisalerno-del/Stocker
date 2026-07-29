# M1C Asymmetric Downside Residual V1 — source audit

## Scope and provenance

- Audited tracked source at Git commit `1d32190883533a7c0a3d089185335482c4696e19` (`feat: add M1C tail phase V1`) on 2026-07-28.
- This is a source-only audit. Research runner and test source were inspected, but no research command or test suite was executed, and no report, artifact, outcome table, or protected row was opened.
- The audit covers the frozen M1C contract, exact checkpoint and fresh-episode mechanics, Tail Phase V1, current IV-scaled outcomes, A1 and the older D0/D1/D2 policy lineage, chronology guards, causal VWAP/volume support, prospective recording, order boundaries, and research packaging conventions.
- Concurrent untracked files for the proposed study were not treated as authoritative source. At the audited commit, Git contains no tracked `m1c_asymmetric_downside_residual_v1` module, study directory, runner, contract, auditor, or target-specific test.

## Conclusion

The repository has a strong reusable foundation for an M1C-conditioned, research-only downside study, but **Asymmetric Downside Residual V1 is not yet implemented or fully specified**.

The ready pieces are the frozen M1C score and threshold, causal checkpoint population, exact fresh-episode rule, Tail Phase V1, movement-consumed predictor, previous-close IV scaling, completed-bar outcome machinery, frozen A1 comparison, provenance conventions, and market-data-only recorder boundary. This source audit confirms the validation logic and pinned identities, not the excluded serialized artifact contents or their empirical provenance.

The blocking gap is the target contract itself. Current M1C and Tail Phase code defines a **symmetric absolute-movement residual**, not an asymmetric downside residual. Before implementation, the study must freeze what “downside,” “residual,” entry, horizon, IV benchmark, ties, missingness, and analysis population mean. It must also add a universally filter-pushed-down source loader, independent auditor, target tests, and—only if prospective validation is in scope—a bounded future-outcome recorder.

## Readiness map

| Concern | Authoritative source | Readiness |
|---|---|---|
| Frozen M1C identity | `packages/stocker_prospective/src/stocker_prospective/contract.py:9-20`; `packages/stocker_prospective/src/stocker_prospective/frozen_m1c.py:118-203` | Ready; reuse without fitting or changing hashes. |
| Exact M1C threshold | `contract.py:9`; `frozen_m1c.py:264-274` | Ready; membership is `probability >= 0.488333710794033`. |
| M1C target lineage | `packages/stocker_research/src/stocker_research/minimal_intraday_iv_excess_holdout_v0.py:42-51`; `research/directional-readiness/20260726-stock-local-directional-archetypes-v0/run_screen_v0.py:603-650` | Existing target is symmetric absolute 15-minute movement exceeding expected absolute IV movement. It is not the proposed target. |
| Causal checkpoint grid | `packages/stocker_prospective/src/stocker_prospective/m1c_features.py:42-48,307-387` | Ready; even checkpoints 6 through 34, complete contiguous bars only. |
| Fresh episodes | `frozen_m1c.py:298-418`; `packages/stocker_research/src/stocker_research/stock_local_directional_archetypes_v0.py:232-314` | Ready; stock/session crossing plus fixed 30-minute spacing. |
| Tail phase | `packages/stocker_prospective/src/stocker_prospective/tail_phase_v1.py:28-54,239-509` | Ready; exhaustive causal phase state with explicit incomplete state. |
| Movement consumed | `tail_phase_v1.py:527-641` | Ready; trailing three completed five-minute bars divided by previous-close expected absolute 15-minute movement. |
| Current outcome engine | `packages/stocker_research/src/stocker_research/m1c_low_movement_v0.py:242-257,267-313,329-460` | Reusable mechanics, but target extension required. |
| Frozen A1 comparison | `packages/stocker_prospective/src/stocker_prospective/direction.py:16-24,112-207,259-351`; `packages/stocker_research/src/stocker_research/m1c_tail_phase_v1.py:257-356` | Ready as an unchanged comparison only. |
| Prospective Tail logging | `packages/stocker_prospective/src/stocker_prospective/recorder_v0.py:183-324`; `packages/stocker_prospective/src/stocker_prospective/recorder_repository.py:950-1014,1126-1203` | Predictor/state persistence is ready; future downside outcomes are not recorded. |
| Order safety | `contract.py:49-95`; `packages/stocker_prospective/src/stocker_prospective/ibkr.py:255-315` | Ready; execution is disabled and order-capable wrappers are rejected. |

## Canonical M1C mechanics to preserve

### Model and target lineage

M1C is a frozen logistic model using unchanged Group O plus the causal Group I feature suffix. Its artifact loader verifies the model identity, exact threshold, feature order, stock controls, finite preprocessing parameters, and removal of `signed_pressure` and `tension`; it also rejects replacement features (`frozen_m1c.py:118-203`). Scoring performs only frozen imputation, scaling, category encoding, and logistic inference (`frozen_m1c.py:205-274`).

The original M1C fit used the binary target `movement_exceeds_prior_close_iv_15m` (`minimal_intraday_iv_excess_holdout_v0.py:42-51`). The target calculation is:

1. enter at the next bar open;
2. calculate `abs(log(close_15m / entry))`;
3. calculate previous-close ATM-IV expected absolute movement as
   `atm_iv * sqrt(15 / (252 * 390)) * sqrt(2 / pi)`;
4. label one only when absolute movement is strictly greater than that expectation.

That implementation is explicit at `20260726-stock-local-directional-archetypes-v0/run_screen_v0.py:603-650`. M1C was fit on target-valid 2024 rows, while its 95th-percentile threshold used all causal 2024 checkpoint rows after previous-close context, without target-validity filtering (`run_screen_v0.py:924-958,1033-1054,3403-3441`).

Therefore:

- the M1C probability remains a frozen **movement-magnitude gate**;
- an asymmetric downside residual can be attached as a new outcome or second-stage target;
- refitting M1C on a downside label would create a new model and must not be described as unchanged M1C.

### Checkpoints and activity history

The prospective feature builder fixes checkpoints at `6, 8, ..., 34`, requires exactly `checkpoint` completed and finalised bars, rejects non-contiguous ordinals and mixed symbols/sessions, and requires valid historical relative activity (`m1c_features.py:42-48,307-343`). Historical activity is prior-session-only, append-chronological, rejects same/future-session use, and requires its configured minimum history before returning a value (`m1c_features.py:98-179`).

### Fresh episodes

The live tracker freezes threshold and 30-minute spacing, detects a crossing only when the current eligible score is at/above threshold and the previous eligible score is absent/below, emits stable stock/session/checkpoint episode IDs, and advances previous-score state only for eligible checkpoints (`frozen_m1c.py:298-315,354-418`).

The retrospective helper independently applies the same stock/session crossing and spacing rule and records marker/trigger identities (`stock_local_directional_archetypes_v0.py:232-314`). Tail Phase V1 runs both implementations over the same panel and asserts identical episode keys before attaching the live ID (`packages/stocker_research/src/stocker_research/m1c_tail_phase_v1.py:536-599`).

The new study should retain both registered analysis levels already frozen by Tail Phase:

- every high-tail checkpoint; and
- fresh high-tail episodes.

These levels are explicit in `research/directional-readiness/20260728-m1c-tail-phase-v1/contract.json:2-5`.

## Tail Phase V1 and movement consumed

Tail Phase V1 is directly reusable:

- exact states are `FIRST_ENTRY`, `PERSISTENT`, `RE_ENTRY`, `OUTSIDE_TAIL`, and `UNKNOWN_INCOMPLETE`;
- exact consumed buckets are `LOW_OR_EQUAL`, `HIGH`, and `UNKNOWN_INCOMPLETE`;
- `signed_pressure`, `tension`, future-filtered peers, peer normalisation, future-dependent membership, and sequential weights are explicitly forbidden.

Those bindings are at `packages/stocker_prospective/src/stocker_prospective/tail_phase_v1.py:28-54`. The tracker does not bridge missing or invalid scheduled checkpoints, validates ten-minute checkpoint spacing, treats threshold equality as in-tail, and carries explicit incomplete reasons (`tail_phase_v1.py:232-430`). It also derives run length, run age, entry count, and time since the last exit only from the stock-local causal history (`tail_phase_v1.py:431-509`).

`movement_consumed_v1` uses exactly the three finalised five-minute bars ending at the trigger, rejects duplicates, gaps, invalid bars, cross-session contamination, and non-contiguous timestamps, then computes:

`log(max(pre-trigger highs) / min(pre-trigger lows)) / previous_close_implied_movement_15m`

The full calculation and missing-state contract are at `tail_phase_v1.py:527-620`; the frozen-median bucket uses inclusive equality on the low side at `tail_phase_v1.py:623-641`.

The retrospective adapter obtains the denominator from the existing previous-close IV helper (`packages/stocker_research/src/stocker_research/m1c_tail_phase_v1.py:359-477`), freezes the consumed median from complete 2024 predictor values only, and applies that same split to all partitions (`m1c_tail_phase_v1.py:480-533`).

## Why the proposed target is new

The canonical outcome engine already constructs:

- signed terminal return;
- absolute terminal return;
- IV sigma and expected absolute movement;
- symmetric terminal IV residual;
- maximum up/down/absolute excursion; and
- realised path range.

It enters exactly when checkpoint features become available and validates unique stock/session/bar identities before constructing future outcomes (`m1c_low_movement_v0.py:267-353`). The present residual is:

`abs(signed terminal return) - IV expected absolute movement`

as shown at `m1c_low_movement_v0.py:407-460`. Tail Phase merely aliases that symmetric value to `future_15m_iv_residual_v1` (`m1c_tail_phase_v1.py:602-651`).

A controlled search of the tracked package source, root tests, and allowlisted directional-readiness runner/contract/README files found no `asymmetric_downside`, `downside_residual`, or `M1C Asymmetric Downside Residual` implementation at the audited commit.

The new contract must choose, rather than imply, all of the following:

1. **Downside observation:** negative terminal return, terminal loss clipped at zero, maximum downside path excursion, or another fixed quantity.
2. **Residual benchmark:** expected absolute move, IV sigma, unconditional Gaussian downside expectation, or another preregistered value. These are not numerically interchangeable.
3. **Output type:** continuous residual, binary exceedance, exhaustive categorical side/residual state, or a fixed combination.
4. **Entry and horizon:** the existing next-bar-open entry and exactly 15 completed minutes are the strongest reusable defaults, but must be binding.
5. **Signs and ties:** define flat return, exact benchmark equality, zero/negative residual, and unavailable horizon behavior.
6. **Completeness:** define missing reasons for entry, future bars, ATM IV, session boundary, and invalid OHLC without silently dropping rows.
7. **Population:** report checkpoint and fresh-episode panels separately; do not let target availability alter M1C membership, stock/session weights, phase, or episode selection.

## Direction-policy lineage

The older D0/D1/D2 stack is not a drop-in policy for this study. It gates on legacy M1 threshold `0.49588519865576763`, not M1C (`packages/stocker_research/src/stocker_research/movement_qualified_direction_v0.py:31-32`; `research/directional-readiness/20260726-movement-qualified-direction-screen-v0/contract.json:44-47`).

- D0 is a causal price/market baseline (`movement_qualified_direction_v0.py:34-57,357-528`).
- D1 contains `signed_pressure`, `pressure_x_tension`, and related signed fields (`movement_qualified_direction_v0.py:58-69,531-617`), which conflict with M1C/Tail Phase’s explicit deny-list.
- D2 contains route-orientation features (`movement_qualified_direction_v0.py:70-85`) and was optional; when its audited source was unavailable, the runner did not fit it and only exposed a compatibility alias to D1 (`20260726-movement-qualified-direction-screen-v0/run_screen_v0.py:758-776,813-935`).

If directional comparison is desired, frozen A1 is the existing compatible seam. It remains labelled a prospective hypothesis rather than a validated strategy (`packages/stocker_prospective/src/stocker_prospective/direction.py:16-24`), is loaded and scored without fitting, and applies a symmetric `CALL` / `PUT` / `ABSTAIN` boundary (`direction.py:112-207,259-351`). Tail Phase attaches A1 unchanged and records explicit completeness/missing reasons (`m1c_tail_phase_v1.py:257-356`).

No downside-target study should silently retrain A1, import D1/D2, or convert a structural outcome diagnostic into an execution policy.

## Chronology and protected-data boundary

Tail Phase freezes:

- development: 2024-01-01 through 2024-12-31;
- assessment: 2025-01-01 through 2025-08-22;
- stress: 2025-09-01 through 2025-12-31; and
- protected: 2026-01-01 onward.

The dates are bound in `20260728-m1c-tail-phase-v1/contract.json:6-13,42,48-51` and in the frozen-config validator at `tail_phase_v1.py:148-211`. The runner rejects identities outside the three opened windows and the deliberate August-to-September gap (`run_tail_phase_v1.py:265-293`). Predictor construction precedes outcome attachment, and only the frozen high-tail subset receives outcomes (`run_tail_phase_v1.py:351-415`).

The reusable in-memory guard fails when any supplied session reaches 2026 (`tail_phase_v1.py:644-655`), and retrospective scoring/outcome helpers call it before their calculations (`m1c_tail_phase_v1.py:166-205,359-374,480-490,536-549,602-620`).

There is, however, a loader-level gap against the stronger requirement “protected rows never enter memory”:

- Tail Phase calls the predecessor `load_inputs()` and validates the returned frames afterward (`run_tail_phase_v1.py:151-184`).
- The predecessor verifies exact source hashes before reads (`20260726-stock-local-directional-archetypes-v0/run_screen_v0.py:709-737`).
- Its completed-bar state read pushes `session <= 2025-12-31` into `read_parquet` (`run_screen_v0.py:738-766`).
- Its dense checkpoint, historical-options, and stress-options reads do not all push a date filter into the read; chronology is checked after those frames or joined surfaces are materialised (`run_screen_v0.py:767-845`).

No protected access was performed in this audit, and the frozen hashes materially constrain source drift. Still, a new V1 runner should apply a protected-boundary predicate in **every** tabular read, select only required columns, and then repeat the in-memory assertions. This is required to make the source code itself support the “never materialised” claim without depending on known file contents.

## Causal VWAP and volume support

The frozen A1 feature builder has a causal volume-aware VWAP implementation. It requires exact contiguous completed bars, ends its direction prefix at marker `T-1`, excludes trigger bar `T`, and calculates VWAP from cumulative positive observed volume only (`packages/stocker_prospective/src/stocker_prospective/direction_features.py:269-378`). Its output records the maximum feature timestamp and trigger exclusion (`direction_features.py:618-640`).

M1C’s activity feature is also chronology-safe in source: its baseline uses only sessions strictly before the scored session (`m1c_features.py:98-179`).

Live parity remains blocked. The diagnostic recorder explicitly says IBKR realtime-bar trade volume has not been shown equivalent to the historical EODHD activity proxy and records `scoring_allowed: False` (`packages/stocker_prospective/src/stocker_prospective/recorder.py:1-6,183-222`). The bar gate also rejects a source-semantic mismatch (`packages/stocker_prospective/src/stocker_prospective/bars.py:127-176,179-227`). Consequently, the new study can reuse historical causal volume/VWAP code, but prospective M1C scoring must stay blocked until feature-source parity is demonstrated and recorded.

## Prospective recorder and order boundary

The current live engine can compute and persist the frozen M1C score, Tail Phase state, movement consumed, bucket, episode state, and frozen directional classifications. Eligibility is fail-closed on capability, parity, Group O completeness, live data type, clock drift, quote freshness, bar finality/gaps, and storage health (`packages/stocker_prospective/src/stocker_prospective/recorder_v0.py:183-324`); eligible fresh episodes can then receive frozen directional classifications (`recorder_v0.py:410-480`). Repository checks bind tail membership to the exact M1C threshold and the consumed bucket to the frozen median (`packages/stocker_prospective/src/stocker_prospective/recorder_repository.py:950-987`), while persistence stores explicit phase/consumed fields and source provenance (`recorder_repository.py:1126-1203`).

There is no corresponding recorder state machine or persistence surface for a future 15-minute asymmetric downside residual. Adding one would require:

- a pending outcome keyed to immutable checkpoint/episode identity;
- collection of exactly the preregistered completed post-entry bars;
- complete/missing/final status and idempotent restart behavior;
- target formula/version, entry, horizon, denominator source, and source timestamps;
- prevention of the future outcome from feeding back into M1C, Tail Phase, A1, eligibility, or episode state.

That work is needed only if prospective outcome validation is explicitly in scope. It is not needed for the first retrospective structural assessment.

The recorder remains market-data-only: the claims boundary disables paper/live orders and execution (`packages/stocker_prospective/src/stocker_prospective/contract.py:49-95`), and the IBKR adapter rejects any attached wrapper exposing order methods (`packages/stocker_prospective/src/stocker_prospective/ibkr.py:255-315`). “Order book” support in this package is bounded market-depth data, not trade-order routing (`ibkr.py:551-590`).

## Research command and artifact convention

The closest authoritative study layout uses:

- a dated directory;
- `README.md` and machine-readable `contract.json`;
- one deterministic runner;
- an independent auditor;
- exact input hashes/roles;
- primary artifacts and a concise report.

The separate runner/auditor commands and hash-failure behavior are documented at `research/directional-readiness/20260726-stock-local-directional-archetypes-v0/README.md:21-40`. Tail Phase follows the dated runner/contract layout but currently documents only one runner (`research/directional-readiness/20260728-m1c-tail-phase-v1/README.md:35-43`).

Tail Phase’s provenance builder records branch, commit, dirty status, configuration hashes, input identities, row/exclusion/missingness counts, exact command, date boundaries, protected-data confirmations, causality confirmations, execution confirmations, and output hashes (`run_tail_phase_v1.py:2060-2218`). Its runner writes frozen config, canonical implementation map, source manifest, checkpoint and episode tables, fixed summaries, report, provenance, and operational status (`run_tail_phase_v1.py:2221-2336`).

The new study should follow that convention and add a genuinely independent `audit_*.py`; reusing the runner’s target helper inside the auditor would not independently verify the formula.

## Missing implementation pieces

The tracked source needs all of the following before this study is runnable:

1. A versioned, pure research module defining the asymmetric downside observation, IV residual, completeness state, and exhaustive category assignment.
2. A binding contract freezing target formula, entry, horizon, benchmark, equality/zero rules, populations, chronology, thresholds or fixed descriptive bins, seeds, minimum support, and safety flags.
3. A runner that reuses frozen M1C/Tail/A1 code, constructs all causal predictors and populations before outcomes, and never refits M1C or A1.
4. Universal column projection and protected-date filter pushdown at every source read, followed by explicit in-memory assertions.
5. An independent auditor that reconstructs the target from primitive prices/IV rather than calling the runner’s target function.
6. Artifact schemas for checkpoint and fresh-episode rows with immutable identity, target version, entry timestamp/price, horizon, downside observation, IV benchmark, residual, complete flag, missing reason, population, partition, source provenance, and protected-access flag.
7. Exact tests for terminal-down, terminal-up, flat, path-down, equality, unavailable bars, session boundary, invalid OHLC/IV, duplicate identities, and target-category exhaustiveness.
8. Mutation tests proving post-horizon bars, peer stocks, later checkpoints, outcomes, and 2025 values cannot change causal membership, phase, movement consumed, frozen 2024 splits, or fresh-episode identity.
9. Tests proving target availability does not change checkpoint membership or stock/session weighting and that checkpoint and episode analyses remain separate.
10. A source-manifest/provenance test binding exact hashes, commands, chronology, missingness, and zero protected access.
11. If prospective validation is requested later, a bounded idempotent outcome recorder and repository migration; otherwise keep this out of the first implementation.

## Smallest defensible implementation surface

A minimal retrospective V1 should add only:

- `packages/stocker_research/src/stocker_research/m1c_asymmetric_downside_residual_v1.py`;
- `tests/test_m1c_asymmetric_downside_residual_v1.py`;
- `research/directional-readiness/20260728-m1c-asymmetric-downside-residual-v1/contract.json`;
- `run_asymmetric_downside_residual_v1.py`;
- `audit_asymmetric_downside_residual_v1.py`;
- `README.md`; and
- this source audit.

It should reuse the existing frozen prospective runtimes and Tail Phase module rather than duplicating them. It should not modify broker code, enable order methods, refit M1C/A1, import D1/D2, or add prospective persistence until the retrospective target contract and independent audit are complete.
