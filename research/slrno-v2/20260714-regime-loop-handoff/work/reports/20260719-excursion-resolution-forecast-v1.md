# 20260719-excursion-resolution-forecast-v1

## Exact scope

Structural resolution-family and arrival-time forecasting for confirmed, active, cluster-invariant excursions only. No economic outcome, payoff, execution, spread, broker, order, position, or strategy field was used.

## Frozen Part A identity

Part A decision: `emission_space_excursion_events_validated`. Part A binding: `5384e344d33315c128fd690cb7cebe45beed3e694f2547fb7e3af23d55ca33aa`. Event definition: `3d18758a31681b77780fcb2c33646e3956daafb8b5d6789b069d665f9e05c80f`.

## Forecast target and population

Each completed bar strictly after departure confirmation and strictly before resolution forecasts the first structural resolution family and bars to resolution. `UNRESOLVED_AT_HORIZON` is right-censored; `REMAIN_LOCAL` is excluded from this active-excursion population.

## Causal features

The frozen manifest contains emission geometry, posterior/age context, causal market context, prior resolved-event history, and clock variables. All preprocessing is fit inside each 2024 training fold.

## Baselines and candidates

Strongest development-only simple baseline: `B2`. Selected development-only structural candidate: `B7`.

## Multiclass proper scores

Validation event-level log loss: candidate 0.450054, baseline 0.417135. Brier: candidate 0.219095, baseline 0.197626.

## Class-specific metrics, calibration, and event level

Candidate top-one accuracy: 0.8833; top-two: 0.9624; ECE: 0.038231. Full class, calibration, confusion, and event-level tables are artifact-bound.

## Timing and lead time

Timing tables cover 3/6/12-bar cumulative incidence. Median correct first-forecast lead: 12.000 bars.

## Quarter, stock deletion, and sensitivity

Calendar-quarter paired losses and every leave-one-stock-out pooled recomputation are reported. Because Part A validated emission space only, no posterior/hybrid representation is promoted as a required sensitivity.

## Binary return/non-return sensitivity

The preregistered binary diagnostic remains separate. Its support status is `False`.

## Failure cases and missing evidence

Resolution classes are highly imbalanced and most Part A excursions were right-censored at the structural horizon. The specifically named Research Pipeline Correctness Audit V1 remained unlocated; underlying source, causality, gap, rerun, and independent-audit evidence was verified.

## Part B scientific decision

`excursion_resolution_timing_validated`

## Economic research status

Later economic testing justified by this experiment: `False`. This report makes no profitability or trading claim.

## Exact next step

Run a separately preregistered structural replication; do not infer economic value.
