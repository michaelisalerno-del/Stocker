# Dynamic loop edge-state V2

Date: 2026-07-14

Decision: **`temporary_payoff_state_hypothesis_rejected`**

Scientific status: causal retrospective development on already-opened 2023 and 2025 surfaces; not prospective validation and not strategy approval.

Safety: `research_only: true`; `live_ordering_enabled: false`; `order_placement: disabled`. No broker, paper/demo, deployment, position-management, or frozen-exit code was changed.

## 1. Hypothesis

A loop orientation may enter and leave a temporary latent net-payoff state. Structural loop occurrence remains a separate prediction target from economic payoff and admission.

## 2. Existing V1 baseline

V1 is preserved byte-for-byte. It uses 60 completed sessions, raw filled-trade support of 20, a fixed 50-trade pseudocount, and activation when the shrunk net mean is above zero. Its exact summary was reproduced before V2 scoring. At 24 bars V1 averaged -0.01 bps/trade in 2025 and +1.25 bps/trade in backward-2023 and rejected the overall hypothesis.

## 3. Data and field definitions

The source is the frozen `breakout_loop_scores_range_p75` accepted-signal ledger. `loop_id` is the top causal parent cycle; `orientation` is the current causal state within that rotation-invariant parent. Stock is `symbol_norm`; signal bar start is `start_timestamp`; decision is that five-minute bar's close; entry is the triggering-bar start proxy; exit/settlement is the anchor+24 bar close. Gross payoff is the frozen directional simple return. Net payoff subtracts the frozen 5 bps entry and 5 bps exit assumptions.

The original V1 anchor panels and 2023 provider files were ephemeral and expired after the exact V1 rerun. V2 therefore uses a registered, hash-verified recovery adapter: the surviving accepted-trade ledger, the prior 250-session causal loop-scoring artifact, and V1's sealed score/state ledgers. Before any V2 score, the adapter verifies hashes and proves exact equality of top loop, top probability, state, and history token for every one of V1's scored rows. This preserves historical predictions; it does not regenerate them from revised data. Raw volume was not retained, so liquidity stresses use the documented anchor-price × causal-volume-ratio activity proxy and must not be read as true dollar-volume tests.

No bid/ask, spread, slippage, commission, financing, borrow, market-impact, or FX component observations exist in this source. Those component columns are retained as unavailable zero fields rather than fabricated estimates. The provider metadata labels these US symbols' currency `GBP`; V1 computes dimensionless returns, but this inconsistency remains a data-quality warning. Sector metadata is unavailable, so the sector slice is explicitly `unavailable`.

## 4. Registered horizon

The only confirmatory horizon is **24 five-minute bars** (about 120 minutes), selected from the prior V1 follow-up rather than re-optimised here. No horizon search was run.

## 5. Decision-time and settlement-time conventions

Each loop/orientation forecast is frozen at regular-session open, before any current-session anchor or payoff. Only complete session observations whose maximum settlement timestamp is strictly earlier than that open can update the state. Current-session outcomes, unresolved outcomes, and later feature rows cannot train the current gate. Entry trigger time is known only to its five-minute bar, so `entry_timestamp` is the triggering bar start proxy; payoff availability waits until the fixed exit bar closes.

## 6. Session-level aggregation

The statistical unit is session × loop × orientation × 24 bars. Multiple fills first collapse to one capped contribution per stock. The primary observation is an equal-stock mean after 10% winsorisation per tail and a ±500 bps stock cap. Raw fill count, independent-stock count, and Kish equal-weight ESS remain separate. A no-opportunity session is absent, never zero. Median aggregation is a predeclared sensitivity.

## 7. Model implementation

The four registered selectors were compared: V1, a 10-observation-half-life EWMA with support and uncertainty, payoff-only Student-t BOCPD, and the full hierarchical Student-t BOCPD. A fifth diagnostic, hierarchical payoff-history BOCPD without leading features, isolates feature value from pooling value. The primary BOCPD hazard is 0.05 per observed session (broad geometric mean 20 sessions), with 1/30 and 1/14 sensitivities. Run-length branches are bounded at 120 sessions. A Normal-Inverse-Gamma update supplies Student-t predictives, and one observation is clipped at four branch-predictive scales for robust sufficient-statistic updates.

`p_edge_positive` concerns the latent state mean. Calibration and Brier scoring instead use `p_next_payoff_positive`, the Student-t probability that the next settled session observation is positive, with observation predictive uncertainty. `p_on_next` is conditional on currently being inactive; `p_off_next` is conditional on currently being active; `p_survive_horizon` compounds the conditional off probability over 24/78 of a session. These outputs drive `unknown`, `active`, `decaying`, and `retired` states. Only `active` admits a new entry; the existing frozen exit is always retained.

## 8. Hierarchical pooling approximation

The shared environment is an online winsorised mean across eligible loop/orientation session cells. Each cell retains its own BOCPD. The empirical-Bayes cell evidence is `effective sessions × sqrt(capped independent-stock factor × capped ESS-per-session factor)`, compared with a frozen 12-session pooling strength. Shared and cell uncertainty are combined, with extra sparse-cell variance. This is a practical approximation: it does not learn dynamic loop loadings or a joint covariance matrix, and population contamination remains possible. Every leave-one-stock-out stress fully rebuilds breadth, market context, session payoffs, shared state, cell posteriors, features, and forecasts before rescoring.

## 9. Leakage controls

The processing order is explicit: settle complete prior sessions; update shared state; update cells; transform current lagged features against past-only moments; forecast; freeze; then join to current opportunities. Hindsight episode labels are generated only after forecasts and are never model inputs. Focused tests cover settlement, same-session exclusion, appended-future invariance, rolling-scaler isolation, session boundaries, shared-state timing, costs, correlated fills, metadata, and unchanged exits.

## 10. Test results

The focused V2 suite and full repository suite were executed before the final historical run; exact counts and commands are recorded in the run note and handoff response. The exact V1 summary SHA-256 matched its archived exact rerun. Regression coverage includes settlement timing, same-session exclusion, immutable frozen history, next-observation calibration semantics, hierarchy evidence weighting, conditional transition probabilities, decay-boundary diagnostics, and leave-one-stock-out context rebuilding.

## 11. Model comparison

| model_name | predictive_log_loss | brier_score | expected_calibration_error | detection_lag_ratio | accepted_trade_count | net_pnl_bps | net_return_per_accepted_trade_bps | coverage | descriptive_sharpe_zero_rate | maximum_drawdown |
|---|---|---|---|---|---|---|---|---|---|---|
| v1_60_session_selector | 0.7104 | 0.2579 | 0.0633 | 0.1458 | 4230 | 2903.6962 | 0.6865 | 0.4252 | 0.0522 | -0.1707 |
| ewma_short_memory | 0.7134 | 0.2585 | 0.0741 | 0.5897 | 165 | -3360.8252 | -20.3686 | 0.0162 | -0.8496 | -0.0220 |
| payoff_only_change_point | 0.8524 | 0.2961 | 0.1485 | 0.4983 | 324 | 8380.7581 | 25.8665 | 0.0327 | 0.6947 | -0.0426 |
| hierarchical_payoff_history_change_point | 0.7580 | 0.2757 | 0.1123 | 0.4284 | 375 | -4466.4249 | -11.9105 | 0.0375 | -0.3554 | -0.0497 |
| hierarchical_change_point | 0.9523 | 0.3250 | 0.2385 | 0.4932 | 275 | -9416.3830 | -34.2414 | 0.0275 | -1.0045 | -0.0516 |
| no_payoff_state_filter | NA | NA | NA | NA | 9926 | -12317.0834 | -1.2409 | 1.0000 | -0.2379 | -0.3041 |

## 12. Probability calibration

Positive target: the next settled robust session net payoff is strictly above zero after the 10 bps round trip. Calibration uses the frozen next-observation Student-t probability and fixed decile bins; rows were scored prequentially. The machine-readable calibration table has 47 pooled bin rows. An abstaining model is `unknown`, not a correct positive prediction.

## 13. Activation and termination delays

| model_name | hindsight_positive_episodes | episodes_detected | fraction_hindsight_positive_episodes_detected | mean_activation_delay_sessions | median_activation_delay_sessions | mean_termination_delay_sessions | detected_change_points | false_change_points | operational_state_transitions | mean_fraction_episode_captured_after_activation | net_payoff_captured_after_activation_bps | net_payoff_missed_before_activation_bps | loss_incurred_during_decaying_bps | detection_lag_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1_60_session_selector | 215 | 127 | 0.5907 | 2.3543 | 0.0000 | 13.5337 | 301 | 198 | 301 | 0.7929 | 48349.2688 | 47658.9868 | 0.0000 | 0.1458 |
| ewma_short_memory | 215 | 15 | 0.0698 | 24.3333 | 11.0000 | 10.4920 | 0 | 0 | 318 | 0.2794 | 982.8261 | 95025.4295 | -32795.2857 | 0.5897 |
| payoff_only_change_point | 215 | 24 | 0.1116 | 8.2500 | 5.5000 | 16.7836 | 537 | 275 | 509 | 0.3259 | 3114.6993 | 92893.5563 | -4520.7083 | 0.4983 |
| hierarchical_payoff_history_change_point | 215 | 54 | 0.2512 | 9.2407 | 4.0000 | 9.9415 | 568 | 471 | 1025 | 0.2374 | 580.4425 | 95427.8131 | -16455.1872 | 0.4284 |
| hierarchical_change_point | 215 | 47 | 0.2186 | 10.4468 | 4.0000 | 10.6108 | 568 | 471 | 956 | 0.2060 | -357.0623 | 96365.3179 | -11430.7172 | 0.4932 |

## 14. Detection-lag ratio

`detection_lag_ratio = causal activation delay / hindsight positive episode length`. Missing ratios mean the model never activated during a qualifying hindsight episode; they are not treated as zero delay.

## 15. Trading results after costs

The table in section 11 reports frozen-exit results after 5 bps per side. Accepted/rejected counts include all frozen signal opportunities; trade counts include only hypothetical fills. The payoff-state model is only an admission overlay and does not refill later overlapping opportunities.

## 16. Twice-cost stress

| model_name | accepted_trade_count | net_pnl_bps | net_return_per_accepted_trade_bps | cumulative_return | maximum_drawdown |
|---|---|---|---|---|---|
| v1_60_session_selector | 4230 | -39396.3038 | -9.3135 | -0.1925 | -0.3034 |
| ewma_short_memory | 165 | -5010.8252 | -30.3686 | -0.0251 | -0.0290 |
| payoff_only_change_point | 324 | 5140.7581 | 15.8665 | 0.0247 | -0.0502 |
| hierarchical_payoff_history_change_point | 375 | -8216.4249 | -21.9105 | -0.0424 | -0.0630 |
| hierarchical_change_point | 275 | -12166.3830 | -44.2414 | -0.0601 | -0.0623 |
| no_payoff_state_filter | 9926 | -111577.0834 | -11.2409 | -0.4541 | -0.5184 |

## 17. Delayed-entry stress

| model_name | accepted_trade_count | net_pnl_bps | net_return_per_accepted_trade_bps | cumulative_return | maximum_drawdown |
|---|---|---|---|---|---|
| v1_60_session_selector | 4200 | 1407.8903 | 0.3352 | -0.0084 | -0.1659 |
| ewma_short_memory | 141 | -2891.9719 | -20.5104 | -0.0145 | -0.0163 |
| payoff_only_change_point | 318 | 3373.2753 | 10.6078 | 0.0158 | -0.0531 |
| hierarchical_payoff_history_change_point | 345 | 15748.7019 | 45.6484 | 0.0797 | -0.0357 |
| hierarchical_change_point | 259 | 9014.5353 | 34.8052 | 0.0446 | -0.0397 |
| no_payoff_state_filter | 9837 | -10396.0490 | -1.0568 | -0.0933 | -0.2875 |

## 18. Leave-one-stock-out

The full model produced 20 fully retrained leave-one-stock-out rows. Each deletion rebuilt population features, payoff panels, shared/cell states, forecasts, and admissions. Net P&L range: -10516.26 to 7306.34 bps; positive deletions: 9/20.

## 19. Episode analysis

The hindsight diagnostic found 215 positive/decaying episodes. Median finite causal activation delay was 4.00 sessions. These are evaluation labels, not predictions. A loop is called predicted only when its frozen session-open forecast activated before the associated payoff was observed.

## 20. Failure cases

Most common full-model rejection combinations:

- `insufficient_sessions`: 2888
- `insufficient_sessions|insufficient_effective_sample_size`: 2273
- `retired_state|edge_probability_too_low|lower_bound_not_positive`: 1748
- `edge_probability_too_low|lower_bound_not_positive`: 1523
- `decaying_state|lower_bound_not_positive`: 400
- `lower_bound_not_positive`: 376
- `posterior_uncertainty_too_high`: 214
- `edge_probability_too_low|lower_bound_not_positive|survival_probability_too_low|termination_probability_too_high`: 115

## 21. Concentration analysis

Best-stock diagnostic: {'model_name': 'hierarchical_change_point', 'concentration_type': 'best_stocks', 'top_item': 'IREN', 'top_item_net_pnl_bps': 1793.2454840748492, 'top_five_share_of_positive_contribution': 0.883620676397056}. Machine-readable stock, loop, orientation, month, episode, and other slices are exported. Sector concentration cannot be assessed on this frozen source.

## 22. Did breadth/coherence lead payoff changes?

Breadth increased before 37.2% of hindsight episodes and top-versus-second coherence increased before 40.5%. Dispersion and structural surprise are measured at each episode's hindsight-estimated decay onset, not its payoff onset: dispersion increased before decay in 35.3%; structural surprise increased in 43.3%. These are hindsight descriptive diagnostics. The isolated causal feature test compares the full model with `hierarchical_payoff_history_change_point`, which has identical pooling but no leading features; the unpooled payoff-only comparison separately tests pooling.

## 23. Hypothesis assessment

**rejected.**

- breadth/coherence did not improve both Brier and log loss over the same hierarchy without leading features.
- hierarchical pooling improved Brier over unpooled payoff-only BOCPD.
- V2 did not reduce detection-lag ratio versus V1.
- V2 net payoff after costs was non-positive.
- V2 failed twice-cost stress.
- positive contribution was concentrated or absent.

Higher filtered P&L alone is not treated as success. Calibration, delay, costs, delayed entry, abstention, and concentration are part of this decision.

## 24. Exact next recommendation

Freeze this V2 implementation and log it prospectively on genuinely new sessions without execution. Do not retune thresholds on 2023/2025. The single highest-value next experiment is a sealed prospective comparison of the full model against the identical hierarchical model without leading features, with immutable forecasts and enough independent session/stock support to estimate activation and termination calibration.

## Reproducibility

- Run ID: `20260714-dynamic-loop-edge-state-v2-8c68a7c1-66be016d`
- Git SHA: `ca3537a0f337097a9a75abf87ae4bf419fae6a5d`
- Branch: `agent/slrno-research-handoff`
- Configuration SHA-256: `8c68a7c10a23ba39f6d067147601d574f12753ea15ae1a91920b29bf604c5373`
- Data snapshot SHA-256: `66be016db2be4b55e0309dfe7a1ec4ee6b99a0b490af8d9abfa11b5832dc1a6a`
- Command: `/Users/michaelsalerno/Documents/Codex/2026-07-14-you-are-working-inside-my-stocker/.venv/bin/python research/slrno-v2/20260714-regime-loop-handoff/work/run_dynamic_loop_edge_state_v2.py`
