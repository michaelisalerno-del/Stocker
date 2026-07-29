# Directed Economic Loop–Regime Rotation V1

## 1–3. Hypothesis, prior boundary, and scientific status

The frozen question is whether the causal decay or retirement of one structural family improves prediction of a **different** family's new positive economic episode within three trading sessions. It is not same-pair persistence, structural loop recurrence, a rolling winner table, or the within-opportunity competitor veto.

The prior rolling selector estimated each pair's own recent payoff; V2 estimated each pair's own lifecycle; lead-lag tested same-pair structural features; recurrence predicted structural paths; and Sequential Competitor Veto eliminated simultaneously compatible loops inside one opportunity. None previously estimated `P(destination activates soon | different source decays/retires)`, separated no/one/multiple family activations, or used a past-only directed economic graph. These opened 2023/2025 surfaces are attribution data, not validation or trading approval.

## 4–8. Frozen taxonomy, state construction, target, and clock

The outcome-free taxonomy maps all 24 frozen two-transition return-cycle pairs to eight orientation families: two_transition_return_cycle__state_0, two_transition_return_cycle__state_1, two_transition_return_cycle__state_2, two_transition_return_cycle__state_3, two_transition_return_cycle__state_4, two_transition_return_cycle__state_5, two_transition_return_cycle__state_6, two_transition_return_cycle__state_7. No family was merged after payoff inspection. The source economic state is the frozen V2 `hierarchical_payoff_history_change_point`, with structural leading features disabled. Pair probabilities are support-weighted into family states; active takes precedence, then decaying, while mixed retired/unknown remains unknown.

The primary label is a new family-positive episode onset in the next **three explicit trading sessions**; one and five sessions are sensitivities. Current active episodes are excluded, multiple activations remain multi-label, and missing future payoff support is unavailable—not zero. Positive labels mature only after the full unioned family episode end. Forecast freeze is the V2 regular-session-open decision timestamp.

## 9–12. Walk-forward graph, source events, and destination base rates

At each session the runner first matures labels with availability strictly before the freeze, then updates beta-smoothed destination rates and cross-family graph counts, constructs M1/M2/M3 features, and freezes forecasts. Same-family edges are excluded. The graph uses alpha=beta=1 edge smoothing, pooling strength 20, and a minimum eight source-event sessions.

Source lifecycle census:

- two_transition_return_cycle__state_0: active onsets 5, decays 4, retirements 2
- two_transition_return_cycle__state_1: active onsets 8, decays 6, retirements 0
- two_transition_return_cycle__state_2: active onsets 1, decays 1, retirements 0
- two_transition_return_cycle__state_3: active onsets 7, decays 4, retirements 0
- two_transition_return_cycle__state_4: active onsets 35, decays 33, retirements 0
- two_transition_return_cycle__state_5: active onsets 36, decays 29, retirements 3
- two_transition_return_cycle__state_6: active onsets 11, decays 10, retirements 0
- two_transition_return_cycle__state_7: active onsets 11, decays 5, retirements 3

Observed three-session destination activation rates (evaluation labels, not live priors):

- state 0: 11.8% (39/330)
- state 1: 6.4% (9/140)
- state 2: 9.3% (8/86)
- state 3: 0.0% (0/116)
- state 4: 13.6% (40/295)
- state 5: 19.3% (61/316)
- state 6: 19.4% (70/360)
- state 7: 17.7% (62/351)

Supported end-of-period family edges (descriptive graph table; not individually promoted):

- 2025: two_transition_return_cycle__state_4 `active` → two_transition_return_cycle__state_7: shrunk lift 0.566, support 46, activations 2
- 2025: two_transition_return_cycle__state_4 `active` → two_transition_return_cycle__state_6: shrunk lift 0.916, support 40, activations 6
- 2025: two_transition_return_cycle__state_4 `active` → two_transition_return_cycle__state_0: shrunk lift 1.223, support 38, activations 6
- 2025: two_transition_return_cycle__state_5 `active` → two_transition_return_cycle__state_7: shrunk lift 0.569, support 33, activations 1
- 2025: two_transition_return_cycle__state_5 `active` → two_transition_return_cycle__state_6: shrunk lift 1.327, support 31, activations 9
- 2025: two_transition_return_cycle__state_5 `active` → two_transition_return_cycle__state_0: shrunk lift 0.964, support 31, activations 3
- 2023: two_transition_return_cycle__state_4 `active` → two_transition_return_cycle__state_6: shrunk lift 0.946, support 29, activations 4
- 2025: two_transition_return_cycle__state_4 `active` → two_transition_return_cycle__state_5: shrunk lift 1.199, support 27, activations 3
- 2023: two_transition_return_cycle__state_4 `active` → two_transition_return_cycle__state_5: shrunk lift 1.487, support 26, activations 14
- 2023: two_transition_return_cycle__state_4 `active` → two_transition_return_cycle__state_0: shrunk lift 1.642, support 25, activations 6
- 2023: two_transition_return_cycle__state_4 `active` → two_transition_return_cycle__state_7: shrunk lift 0.904, support 25, activations 3
- 2023: two_transition_return_cycle__state_5 `active` → two_transition_return_cycle__state_0: shrunk lift 0.972, support 18, activations 1

## 13–18. M0–M3 and primary paired tests

| Model | Brier | Log loss | ECE | Coverage |
|---|---:|---:|---:|---:|
| M0 base rate | 0.120707 | 0.396032 | 0.018072 | 0.0% |
| M1 destination history | 0.120359 | 0.394737 | 0.027522 | 3.0% |
| M2 undirected system | 0.120483 | 0.395103 | 0.030126 | 3.6% |
| M3 directed family | 0.120486 | 0.395117 | 0.030652 | 3.6% |

Primary M3-versus-M1 paired Brier improvement: **-0.000126** (session-block 95% interval -0.000342 to 0.000093); paired log-loss improvement: **-0.000380**. Directional M3-versus-M2 Brier improvement: **-0.000003**; log-loss improvement: **-0.000015**. Period and 1/5-session shapes are in `paired_model_metrics.csv`; no window was selected after scoring.

## 19–23. Pair refinement, system outcomes, timing, and economic translation

M4 is secondary: pair activation rates are shown only at frozen support (20 rows and four activations) and shrunk toward M3 family forecasts. Unsupported pair edges remain unknown. No-activation and multiple-activation probabilities are exported separately; multi-label scoring is not mixed with first-activation ranking.

Across 330 observable primary system windows, 149 had no activation and 58 had multiple activations. M3 no-activation Brier was 0.246585; multiple-activation Brier was 0.143628. M1 was slightly better on both (0.245905 and 0.143190).

Activation timing: median lead 2.0 sessions; mean episode payoff available 1032.69 bps. Because all primary onsets follow the forecast, the mapped episode payoff is entirely post-forecast at the family-label level; this does not imply that a qualifying trade exists.

Opportunity translation uses only later frozen no-filter V2 opportunities with the exact predicted destination family inside the three-session window. Missing families are not replaced and overlap/capacity is not refilled. M3 result: -23526.80 bps over 887 eligible opportunities; twice-cost -32396.80 bps; one-bar-delay -9047.44 bps over 653 observable delayed rows.

## 24–29. Cost, nulls, leave-one-stock-out, and concentration

Registered nulls:

- wrong_lag_10: Brier improvement -0.000172, log-loss improvement -0.000640
- source_permutation: Brier improvement -0.000135, log-loss improvement -0.000405
- destination_label_permutation: Brier improvement -0.000137, log-loss improvement -0.000457

Fully rebuilt leave-one-stock-out: not completed; blocker: missing hash-pinned V1 recovery inputs: accepted_signal_ledger=/private/tmp/stocker_frozen_regime_loop_pnl_sanity_v1_20260712/accepted_signal_ledger.parquet, loop_scoring_2023=/private/tmp/stocker_hierarchical_loop_quality_algorithm_v1_20260711/scoring_predictions_2023.parquet, loop_scoring_2025=/private/tmp/stocker_hierarchical_loop_quality_algorithm_v1_20260711/scoring_predictions_2025.parquet, v1_scored_signal_ledger=/private/tmp/stocker_dynamic_loop_context_edge_v1_20260713_exact_rerun/scored_signal_ledger.parquet, v1_primary_cell_states=/private/tmp/stocker_dynamic_loop_context_edge_v1_20260713_exact_rerun/primary_cell_states.parquet, v1_summary=/private/tmp/stocker_dynamic_loop_context_edge_v1_20260713_exact_rerun/summary.json, v1_source_hashes=/private/tmp/stocker_dynamic_loop_context_edge_v1_20260713_exact_rerun/source_hashes.json. Neither the median aggregation nor the leave-one-stock-out states could be rebuilt; no aggregate approximation or imputed result was substituted. The minimum-two-bar dwell and alternate taxonomy sensitivities are explicitly not applicable because neither was registered for this session-level source; primary states were not silently changed.

Concentration diagnostics:

- destination_family `two_transition_return_cycle__state_6`: 95.0% of absolute contribution
- pair `cycle_06`: 88.7% of absolute contribution
- source_state_vector `{}`: 55.5% of absolute contribution
- period `2023`: 53.4% of absolute contribution
- period `2025`: 46.6% of absolute contribution
- month `2025-09`: 36.5% of absolute contribution

## 30–34. Failure cases, decision, and recommendation

Failure modes include sparse newly-decaying/retired edges, long positive family unions that suppress new-onset labels, family aggregation where one active pair masks another pair's retirement, calibrated episode prediction without a later eligible opportunity, and contribution concentration. A structured graph or one high-lift edge is not evidence of tradeability.

Scientific decision: **`destination_own_history_sufficient`**.

The directed graph is descriptive rather than incrementally predictive on these opened periods. Destination own history is sufficient under the registered comparison; M3 neither forecasts the next profitable family better nor translates into positive subsequent opportunity payoff.

This decision distinguishes historical description from prediction: every scored forecast was frozen before its target, but the periods were already opened. The independent audit passed 18/18 checks. The exact rerun matched all 35 comparable machine-readable files byte-for-byte.

Exact next recommendation: freeze this contract and prospectively log M1/M2/M3 family forecasts on a genuinely unopened data snapshot until enough new decay/retirement events mature; do not refine pairs or thresholds during collection.

## Reproducibility and assumptions

- Run ID: `directed-rotation-02cbec96e5441a5217116e74`
- Git SHA: `1b89071b41139b87640de28c8f9abba17cd750a0`
- Contract SHA-256: `f982891bfb8e3b539ed5f6816b71b94a92979b57ae8a29dd121cfa888805a813`
- Data snapshot: `728b145c73c2f82fe2ba707a077a9ff52f597e065876ded40790f0818c5de684`
- Fixed horizon: 24 bars; frozen cost translation: 5 bps per side.
- A family positive episode is the calendar union of overlapping or adjacent frozen pair-positive episodes. This union is an evaluation label, never a feature.
- Family taxonomy uses current frozen regime orientation because the repository had no prior outcome-free named branch taxonomy spanning all eligible pairs.
- The V2 decision timestamp is the regular-session-open family scoring freeze; within-session opportunities are later economic translations, not state-feature inputs.
- Existing entries, exits, overlap, positions, broker paths, and runtime behaviour remain unchanged.
