# Causal payoff-model paths V1

Date: 2026-07-14

Decision: **`all_tested_model_paths_rejected_or_unknown`**

Scientific status: causal retrospective cross-surface mechanism development on already-opened 2024 data. This is not validation.

Safety:

- `research_only: true`
- `live_ordering_enabled: false`
- `order_placement: disabled`
- no trading app, broker, paper/demo account, deployment, position, or order path was changed or used
- provider volume remains labelled `historical_volume_activity_proxy`; it was not used

## Direct answer

None of the three tested causal paths can yet distinguish profitable from losing occurrences of the same predicted loop orientation with usable uncertainty:

1. The compact admission payoff model abstained on every scored opportunity.
2. The five-way route forecast did not beat its causal candidate prior consistently or securely.
3. Adding completed-bar route plus MFE/MAE/retracement state produced no uncertainty-qualified hold or exit decisions.

The result does not challenge structural parent-loop identity conditional on completion. It shows again that loop identity, opportunity completion, and cost-adjusted payoff are separate targets. Even a perfectly known future parent label would not identify which occurrence pays.

The 2024 cross-surface baseline itself split sharply: `cycle_04|state4` averaged **-15.17 bps** after 5 bps per side, while `cycle_07|state5` averaged **+102.82 bps**. This is a descriptive opened-data observation on a different OCO-breakout surface, not validation or a rule. It reinforces that payoff depends on the admission surface and time, not merely the loop name.

## Why 2024 was used

No genuinely unseen sealed sessions remain. The expected-leg-then-diversion hypothesis therefore could not be tested honestly and remains deferred.

A distinct 2024 causal surface was available for mechanism-transfer development:

- source: `causal_setup_conditions_v1` OCO anchor breakouts;
- setup/family: `oco_anchor_breakout_all` / `oco_baseline`;
- filled anchor+24 observations only;
- the original 20-stock universe, excluding 2024-only `AXTI` and `OKLO`;
- 128 base-surface sessions from 2024-07-01 through 2024-12-31;
- the first 60 completed sessions used only as warm-up;
- 68 base score sessions, with candidate opportunities on 67 of them;
- primary cost: 5 bps per side.

This is not the earlier `breakout_loop_scores_range_p75` signal surface. It tests whether the proposed mechanisms transfer; it does not validate the 2023/2025 candidates.

| Candidate | Full rows | Scored rows | Scored candidate sessions | Mean net at frozen close |
|---|---:|---:|---:|---:|
| `cycle_04|state4` | 259 | 112 | 46 | -15.17 bps |
| `cycle_07|state5` | 432 | 289 | 64 | +102.82 bps |
| Pooled | 691 | 401 | 67 | +69.87 bps |

The contract, runner, frozen auditor, tests, inputs, state runs, reports, and all 20 provider files were SHA-256 frozen before scoring.

## Path 1: direct causal payoff state

Two rolling Bayesian-ridge models were fit from prior completed sessions only:

- compact admission information only;
- the same information plus five prequential route-class probabilities.

The target was fixed-close net payoff after 10 bps round-trip cost. A prediction could be positive only if the lower endpoint of its frozen 90% posterior-predictive interval exceeded zero, negative only if its upper endpoint was below zero, and otherwise had to abstain.

Both variants labelled **100% of 401 scored opportunities `unknown_abstain`**. Positive coverage was 0%, so no uncertainty-aware selector existed. The minimum 5% coverage gate failed before economic promotion could be considered.

Forecast quality was also weak:

| Model | Group | RMSE | Expanding baseline RMSE | Spearman |
|---|---|---:|---:|---:|
| Admission only | Pooled | 342.99 | 341.06 | +0.164 |
| Admission only | `cycle_04|state4` | 200.88 | 197.07 | -0.013 |
| Admission only | `cycle_07|state5` | 384.18 | 382.56 | +0.124 |
| Admission + route probabilities | Pooled | 342.70 | 341.06 | +0.173 |
| Admission + route probabilities | `cycle_04|state4` | 202.12 | 197.07 | -0.051 |
| Admission + route probabilities | `cycle_07|state5` | 383.57 | 382.56 | +0.133 |

The admission features therefore did not beat the expanding payoff baseline and did not rank both candidates in the correct direction.

Point means were inspected only because the contract explicitly labelled them nonqualifying diagnostics. They selected many rows and improved per-opportunity payoff by +5.62 bps pooled for admission-only and +7.86 bps pooled with route probabilities. Those apparent gains cannot qualify because they ignore predictive uncertainty, were not the multiplicity-controlled primary policy, and still selected negative mean payoff for `cycle_04` (-5.20 and -8.40 bps respectively). They are not rules to carry forward.

Decision: **`direct_payoff_state: rejected_or_unknown`**.

## Path 2: causal route-branch forecast

The frozen route target had five exhaustive outcome labels:

- no transition;
- expected leg only;
- exact parent completion;
- incompatible first transition;
- expected leg followed by diversion.

Realised topology was never used as an admission feature. A rolling L2 multinomial model used prior sessions only, with a 2% uniform probability mixture to prevent zero-probability classes. Its comparator was a candidate-specific Dirichlet-smoothed class prior.

| Group | Rows | Log-loss improvement vs prior | 95% session-block interval | Model / prior Brier |
|---|---:|---:|---:|---:|
| Pooled | 401 | -0.0146 | -0.0813 to +0.0462 | 0.650 / 0.661 |
| `cycle_04|state4` | 112 | -0.0544 | -0.1171 to +0.0128 | 0.594 / 0.575 |
| `cycle_07|state5` | 289 | +0.0009 | -0.0829 to +0.0829 | 0.672 / 0.694 |

No endpoint passed Holm control. Log-loss improvement was positive in only 1/4 scored months for `cycle_04` and 2/4 for `cycle_07`. Leave-one-stock-out improvement was negative for 19/20 pooled deletions and all 20 `cycle_04` deletions.

The route probabilities also failed as auxiliary payoff inputs. They worsened admission RMSE for `cycle_04`, improved it only slightly for `cycle_07`, and did not improve the nonqualifying point selector in both candidates.

Decisions:

- **`route_branch_forecast: rejected_or_unknown`**
- **`predicted_route_increment: rejected_or_unknown`**

This rejects this compact route model on this opened surface. It does not establish that future route branches are inherently unforecastable.

## Path 3: route joined to causal price path

Three checkpoints were fixed before scoring at 25%, 50%, and 75% of each post-entry holding window. Every checkpoint used only fully completed bars; the entry bar was excluded from excursion calculations. A hypothetical state change could act only at the following provider open.

The price-path model added causal running return, favorable and adverse excursion, time to each excursion, prior-ATR-normalised movement, running peak/trough, retracement, post-entry volatility, and range support to the route state. It was compared with the same sequential model using route state only.

There were 1,200 scored checkpoint rows across 400 opportunities. Route plus price marginally reduced RMSE relative to route-only:

| Group | Route + price RMSE | Route-only RMSE | Route + price Spearman |
|---|---:|---:|---:|
| Pooled | 217.18 | 217.82 | +0.055 |
| `cycle_04|state4` | 141.93 | 142.06 | -0.037 |
| `cycle_07|state5` | 240.16 | 240.93 | +0.074 |

That small error reduction did not become a usable decision. Every checkpoint prediction remained inside the frozen uncertainty interval, so **100% were `unknown_abstain`** and the uncertainty-aware policy made zero early-exit actions.

The nonqualifying point-mean diagnostic confirms that forcing action is unsafe:

| Model | Group | Action coverage | Paired policy minus frozen close |
|---|---|---:|---:|
| Route + price | Pooled | 56.6% | -15.15 bps |
| Route + price | `cycle_04|state4` | 67.0% | +0.14 bps |
| Route + price | `cycle_07|state5` | 52.6% | -21.07 bps |
| Route only | Pooled | 45.6% | -8.21 bps |
| Route only | `cycle_04|state4` | 57.1% | -1.51 bps |
| Route only | `cycle_07|state5` | 41.2% | -10.80 bps |

Decision: **`sequential_route_plus_price: rejected_or_unknown`**.

This does not prove that MFE, MAE, or retracement can never help. It shows that the frozen compact features, quartile checkpoints, support, and linear partially pooled proxy did not identify a sufficiently certain remaining-payoff state.

## Path 4: expected leg then diversion

Decision: **`diversion_specific_payoff: deferred_no_sealed_data`**.

No diversion-specific payoff or exit table was produced in this experiment. The negative 2023/2025 pattern was found after inspection, and opening another already-seen period would not validate it. It remains a named secondary hypothesis for observations logged after a future immutable activation timestamp.

## What is now retired

Do not keep searching the opened 2023, 2024, or 2025 data for a better variation of:

- the frozen compact admission-only Bayesian-ridge model;
- the admission model plus the frozen five route probabilities;
- the frozen five-way multinomial route forecast;
- quartile-checkpoint route-only or route-plus-price Bayesian-ridge state;
- uncertainty-ignoring point-mean selectors;
- any diversion, MFE/MAE, retracement, ATR, bar-count, child, or morph threshold.

Changing the regression family, interactions, thresholds, or checkpoints now would be post-score tuning on the newly opened 2024 outcomes.

## Most useful path forward

The next experiment should not be another retrospective dictionary search. It should begin only after a prospective research ledger is activated under a separate authorization:

1. Freeze an activation timestamp, model source, causal feature clock, costs, support floor, uncertainty method, and evaluation code before observing any new outcomes.
2. Log the two candidates and matched orientations without execution, including admission probabilities, regime state/age/hazard, completed-bar route state, MFE/MAE/retracement, ATR support, and next-open counterfactuals.
3. Keep direct payoff, route occurrence, and route-conditioned remaining payoff as separate targets.
4. Require nonzero conservative coverage before comparing economic performance; a model that abstains everywhere remains unknown, not successful.
5. Test expected-leg diversion only as a secondary frozen hypothesis after the prospective seal.

`work/contracts/20260714-loop-payoff-prospective-log-v3.json` records this specification. It is not activated or implemented in the trading application.

## Integrity and reproducibility

- Pre-score manifest SHA-256: `0b717f57be949f4764bc0fcbaa545a5067e2dcf79eb745ee32a64171be85ed1a`.
- Frozen contract SHA-256: `d0179b25d07d33841e57a3030acb25e73aaefaa8f00c2ff5047281b55ede4f6c`.
- Frozen runner SHA-256: `0d6b175616071714ec5f341d50be10fa3d7c1887e69624bd5e5dbd33f8cd7d75`.
- Frozen auditor SHA-256: `468c680c69c0c07d2cf673776cad2cc7ce44a772612c49ea3d98a4001bd22275`.
- Outcome-blind population-audit addendum SHA-256: `ea76313628e001e97eef6447db76a08a0480be41fcf80eea2aed203be43a6b6a`.
- Focused tests SHA-256: `21a769fde412776eead8d4542c6c1c968434a3c68248acdb144ea997a5be2a2c`.
- Artifact-manifest SHA-256: `06407441f2a0db1aed7e7e18aef7baa0f4b2dd56db8747e8d26a0ba583c492b6`.
- Focused tests: 9/9 passed; lint passed.
- Frozen independent audit: 15 unaffected checks passed; one population-calendar check was rejected because it incorrectly demanded the base surface's 128 sessions from the candidate-only surface.
- The frozen auditor was preserved. An outcome-blind addendum replayed the correct base calendar, candidate counts, warm-up boundary, calendar indices, and score flags: 8/8 passed.
- Maximum fixed-payoff and snapshot-payoff replay errors: 0.
- Maximum aggregate replay error: `1.42e-14`; bootstrap replay error: `7.11e-15`; Holm replay error: 0.
- Exact rerun: all 30 files were byte-identical.
- No diversion-specific payoff artifact exists.
- The pre-existing dirty `StockerLocal` worktree was not modified by this experiment.

Primary artifacts:

`work/artifacts/20260714-causal-payoff-model-paths-v1/primary`

Exact rerun:

`work/artifacts/20260714-causal-payoff-model-paths-v1/exact_rerun`
