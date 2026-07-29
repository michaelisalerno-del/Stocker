# Loop Event Semantics and Causal Infrastructure V2

`research_only=true` · `execution_enabled=false` · `order_placement=disabled` · `broker_connected=false` · `strategy_promotion=false`

## 1. Exact scope

This migration reconstructs structural state and loop events only. It does not read payoff, MFE, MAE, P&L, order, position, broker, or execution data; it trains no predictor and makes no economic-edge claim.

## 2. Active versus frozen implementation map

The pre-rewrite census contains 2640 implementation entries. Historical files at baseline `6d43807dbe3287afb1587f7ab4f5dafdeedad426` remain byte unchanged. V2 behavior is isolated in typed package modules, a new runner, contract, tests, and independent auditor.

## 3. Confirmed defects

- The historical run builder combines the first row's start timestamp with last-row B0/context fields.
- Legacy targets are overlapping compatible rotated-cycle labels, not a mutually exclusive first next event.
- Primitive loops, repeated traversals, and composites shared a flat cycle namespace.
- Legacy IDs depend on discovery rank; discovery/load length contracts differed (2–5 versus 2–4).
- Duration 24 was an `>=24` bucket, exact 24 was omitted from convolution, age 24 was forced to exit, and terminal runs were excluded rather than censored.
- The causal filter discarded all but MAP state/age, and forecast anchors were run entries only.
- A historical limited-path baseline repeated anchor context through hypothetical transitions.
- The shuffled-order null destroyed transition, duration, occupancy-order, and phase structure; raw support dominated dictionary selection.

## 4. Suspected defects not confirmed

Actual 2024 B0 value leakage was not observed: both B0 fields were constant within every audited hard run. This disproves an empirical 2024 B0-value change while leaving the source-position implementation defect confirmed. No protected prospective dataset was opened.

## 5. B0 timing result

- `b0_high_stress`: 0 of 110,949 runs differ.
- `b0_state_numeric`: 0 of 110,949 runs differ.
- `clock_cos`: 70,488 of 110,949 runs differ.
- `clock_sin`: 70,028 of 110,949 runs differ.

Provider timestamps are bar starts. Completed-bar availability and decisions are five minutes later. B0 uses an explicit prior-session source/availability timestamp; missing warm-up values remain missing.

## 6. Historical experiments affected by B0 timing

No audited 2024 B0 anchor changed. Raw-run clock consumers are affected because clock fields changed in more than 70,000 runs. Context-model results remain provenance-limited even where session-level B0 happened to be invariant.

## 7. Legacy loop-target semantics

Legacy labels independently ask whether each compatible rotated whole cycle occurs. They can have several positives and have no mutually exclusive no-loop outcome. Historical top-three recall is therefore overlapping-label recall.

## 8. V2 first-event semantics

V2 resolves the earliest completion after each completed bar across prefixes already open at the decision and loops initiated later. It separately labels registered primitive/repeat/composite completions, unregistered loops, no registered completion, session end, ties, and unavailable sources.

## 9. Quantified legacy versus V2 target difference

Across 398,304 source-eligible decisions, 369,745 (92.83%) differ under the full semantic contract; 175,054 differ when comparing only registered event sets. Legacy had multiple simultaneous positives at 14,331 decisions, and 317,181 decisions had active prefixes. Source-unavailable decisions are reported separately and never counted as semantic differences.

## 10. Primitive/repeat/composite decomposition

The selected dictionary contains 11 primitives, 8 repeated traversals, and 1 composite motif. The legacy migration retains explicit component mappings rather than deleting its composite motif; 0 migrations are ambiguous.

## 11. Semantic ID design

IDs derive from canonical paths: `loop_p_<path>`, `loop_rN_<primitive-path>`, and `loop_c_<hash>`. Rotation preserves identity; orientation is separate metadata; reverse direction is not silently merged; repeat depth changes identity.

## 12. Prefix automaton design

An Aho–Corasick-style automaton advances only on causal state-change events and retains every suffix matching a proper oriented-loop prefix. Per-bar decisions map to the most recent causal state event without replacing their own completed-bar timestamp.

## 13. Tie and nested-event handling

Same-event structural ties remain `TIED_REGISTERED_COMPLETION` with ordered IDs in a secondary field. Primitive completions precede later repeats/composites; nested completions remain secondary events.

Observed rates are: tie 2.59%, no registered loop 2.58%, unregistered loop 63.95%, session end 7.14%, and unavailable 6.19%.

## 14. Session-boundary handling

Prefixes reset at every regular-session boundary. In-session gaps fail closed as `UNAVAILABLE`; no prefix crosses overnight.

## 15. Duration-24 correction

Durations 1–78 are exact. Exact duration 24 contributes at horizon 24; duration 25 is distinct; no final hazard is forced to one.

## 16. Long-duration and censoring design

The duration model is discrete survival with hierarchical smoothing. It contains 5,128 right-censored terminal runs, 390 exact-24 runs, and 491 runs longer than 24. Remaining session time truncates completion while preserving terminal/no-completion mass.

## 17. Posterior-state export

All 424,583 bars export eight state probabilities, entropy, top two states and margin, persistence/transition probability, next-state probabilities, expected age, and the complete 8×78 V2 state-age posterior. Ages 1–23 preserve the frozen hazards; the former forced exit at age 24 is replaced by an explicit geometric tail through the regular-session support. Frozen hard-MAP labels remain a separate compatibility surface.

## 18. Hard versus hysteretic versus soft representations

`LEGACY_HARD_MAP` exactly reproduces frozen causal labels. `CAUSAL_HYSTERETIC_STATE` uses only the current posterior and prior causal state and resets by session. `SOFT_POSTERIOR` exports probability mass without asserting hard completion; soft prefix propagation was bounded to 20,592 preregistered sample rows.

## 19. Per-bar versus run-entry populations

There are 424,583 completed-bar decisions and 110,949 run-entry baseline rows. Run entries are a strict subset.

## 20. Static future-context audit

Direct V2 event models may use current known context once. Unknown future B0, price, activity, volatility, and market context are never fabricated. Historical history-only models are distinguishable from the rejected static-context limited-path baseline in the lineage table.

## 21. Old null limitations

Within-session state permutation destroys transition probabilities, dwell structure, persistence, higher-order order, and phase. It is retained only as historical evidence.

## 22. V2 semi-Markov null

Primary results use 2,000 fitted semi-Markov draws on a balanced 264-session sample with original lengths. Selected-loop rate ratios span 1.30–10.32; selected q-values are at most 0.0991. Clock-conditioned, first-order analytical, and whole-session circular controls are exported separately. These are structural results, not economic evidence.

## 23. Dictionary-selection redesign

The 20-entry dictionary separates eligible anchors, observed and null-expected completions, excess, rate ratio, conditional information, current-state and second-order increments, breadth, period consistency, and complexity. Primitive dependencies are inserted before repeats/composites. Raw frequency alone does not determine rank.

## 24. Allowed-length consistency

Discovery, decomposition, storage, loading, scoring, tests, and audit share primitive lengths 2–5, composite lengths 4–8, and maximum event length 8. Unsupported legacy entries fail closed.

## 25. Historical lineage impact table

| experiment_name | severity | historical_result_interpretability |
| --- | --- | --- |
| audit_causal_loop_state_path_v1 | semantic limitation | structurally interpretable only |
| audit_causal_state_pattern_discovery_v1 | none | fully interpretable |
| audit_directed_economic_loop_regime_rotation_v1 | semantic limitation | structurally interpretable only |
| audit_dynamic_loop_context_edge_v1 | semantic limitation | structurally interpretable only |
| audit_dynamic_loop_edge_state_lead_lag_v1 | none | fully interpretable |
| audit_dynamic_loop_edge_state_v2 | none | fully interpretable |
| audit_factor_conditioned_loop_occurrence_v1 | semantic limitation | structurally interpretable only |
| audit_frozen_loop_price_consequence_test | semantic limitation | structurally interpretable only |
| audit_frozen_named_loop_t0_execution_v1 | semantic limitation | structurally interpretable only |
| audit_frozen_regime_loop_pnl_sanity_v1 | none | fully interpretable |
| audit_hierarchical_loop_quality_algorithm_v1 | semantic limitation | structurally interpretable only |
| audit_joint_history_semimarkov_loop_forecast | semantic limitation | structurally interpretable only |
| audit_loop_burst_mechanism_v1 | semantic limitation | structurally interpretable only |
| audit_loop_payoff_phase_path_v1 | semantic limitation | structurally interpretable only |
| audit_loop_payoff_phase_path_v2 | semantic limitation | structurally interpretable only |
| audit_loop_quality_feature_ablation_v2 | semantic limitation | structurally interpretable only |
| audit_loop_quality_feature_ablation_v3 | semantic limitation | structurally interpretable only |
| audit_per_loop_movement_quality | semantic limitation | structurally interpretable only |
| audit_profitable_loop_episode_anatomy_v1 | semantic limitation | structurally interpretable only |
| audit_regime_loop_individual_expert_selection_v1 | semantic limitation | structurally interpretable only |
| audit_regime_loop_linkage_ideas_v3 | semantic limitation | structurally interpretable only |
| audit_regime_loop_orientation_calibration_algorithms_v1 | semantic limitation | structurally interpretable only |
| audit_sequential_loop_competitor_veto_v1 | semantic limitation | structurally interpretable only |
| factor_conditioned_loop_occurrence_core | semantic limitation | structurally interpretable only |
| factor_conditioned_loop_occurrence_eval | semantic limitation | structurally interpretable only |
| frozen_loop_movement_shadow_core | semantic limitation | structurally interpretable only |
| frozen_loop_movement_shadow_core | semantic limitation | structurally interpretable only |
| historical | semantic limitation | structurally interpretable only |
| log_frozen_named_loop_t0_opportunities_v1 | none | fully interpretable |
| per_loop_quality_shadow_core | semantic limitation | structurally interpretable only |
| per_loop_quality_shadow_core | semantic limitation | structurally interpretable only |
| run_causal_loop_prefix_path_forecast | semantic limitation | structurally interpretable only |
| run_causal_loop_state_path_v1 | none | fully interpretable |
| run_causal_semimarkov_regime_loops | provenance limitation | structurally interpretable only |
| run_causal_state_pattern_discovery_v1 | semantic limitation | structurally interpretable only |
| run_directed_economic_loop_regime_rotation_v1 | semantic limitation | structurally interpretable only |
| run_dynamic_loop_context_edge_v1 | semantic limitation | structurally interpretable only |
| run_dynamic_loop_edge_state_lead_lag_v1 | none | fully interpretable |
| run_dynamic_loop_edge_state_v2 | semantic limitation | structurally interpretable only |
| run_factor_conditioned_loop_occurrence_v1 | semantic limitation | structurally interpretable only |
| run_frozen_loop_movement_shadow | semantic limitation | structurally interpretable only |
| run_frozen_loop_movement_shadow | semantic limitation | structurally interpretable only |
| run_frozen_loop_price_consequence_test | semantic limitation | structurally interpretable only |
| run_frozen_named_loop_t0_execution_reference_v1 | semantic limitation | structurally interpretable only |
| run_frozen_regime_loop_pnl_sanity_v1 | none | fully interpretable |
| run_hierarchical_loop_quality_algorithm_v1 | semantic limitation | structurally interpretable only |
| run_joint_history_semimarkov_loop_forecast | semantic limitation | structurally interpretable only |
| run_loop_burst_mechanism_v1 | semantic limitation | structurally interpretable only |
| run_loop_payoff_phase_path_v1 | semantic limitation | structurally interpretable only |
| run_loop_payoff_phase_path_v2 | semantic limitation | structurally interpretable only |
| run_loop_quality_failure_diagnostics | semantic limitation | structurally interpretable only |
| run_loop_quality_feature_ablation_v2 | semantic limitation | structurally interpretable only |
| run_loop_quality_feature_ablation_v3 | semantic limitation | structurally interpretable only |
| run_per_loop_movement_quality | semantic limitation | structurally interpretable only |
| run_per_loop_quality_shadow | semantic limitation | structurally interpretable only |
| run_per_loop_quality_shadow | semantic limitation | structurally interpretable only |
| run_profitable_loop_episode_anatomy_v1 | semantic limitation | structurally interpretable only |
| run_regime_loop_individual_expert_selection_v1 | semantic limitation | structurally interpretable only |
| run_regime_loop_linkage_ideas_v1 | semantic limitation | structurally interpretable only |
| run_regime_loop_linkage_ideas_v2 | semantic limitation | structurally interpretable only |
| run_regime_loop_linkage_ideas_v3 | semantic limitation | structurally interpretable only |
| run_regime_loop_orientation_calibration_algorithms_v1 | semantic limitation | structurally interpretable only |
| run_sequential_loop_competitor_veto_v1 | semantic limitation | structurally interpretable only |
| settle_frozen_named_loop_t0_outcomes_v1 | none | fully interpretable |
| state_lifecycle_context_lab_v0 | semantic limitation | structurally interpretable only |
| test_causal_loop_state_path_v1 | none | fully interpretable |
| test_causal_state_pattern_discovery_v1 | semantic limitation | structurally interpretable only |
| test_causal_state_pattern_discovery_v1_audit | none | fully interpretable |
| test_factor_conditioned_loop_occurrence_core | semantic limitation | structurally interpretable only |
| test_factor_conditioned_loop_occurrence_eval | semantic limitation | structurally interpretable only |
| test_factor_conditioned_loop_occurrence_v1 | semantic limitation | structurally interpretable only |
| test_factor_conditioned_loop_occurrence_v1_audit | semantic limitation | structurally interpretable only |
| test_frozen_loop_movement_shadow | none | fully interpretable |
| test_frozen_loop_movement_shadow | none | fully interpretable |
| test_frozen_regime_loop_pnl_sanity_v1 | none | fully interpretable |
| test_hierarchical_loop_quality_algorithm | semantic limitation | structurally interpretable only |
| test_hierarchical_loop_quality_algorithm_v1_audit | semantic limitation | structurally interpretable only |
| test_joint_history_semimarkov_loop_forecast | none | fully interpretable |
| test_loop_burst_mechanism_v1 | none | fully interpretable |
| test_loop_burst_mechanism_v1_audit | semantic limitation | structurally interpretable only |
| test_loop_payoff_phase_path_v1 | none | fully interpretable |
| test_loop_payoff_phase_path_v2 | semantic limitation | structurally interpretable only |
| test_loop_quality_failure_diagnostics | semantic limitation | structurally interpretable only |
| test_loop_quality_feature_ablation_v2_audit | semantic limitation | structurally interpretable only |
| test_loop_quality_feature_ablation_v2_stop | none | fully interpretable |
| test_loop_quality_feature_ablation_v3 | semantic limitation | structurally interpretable only |
| test_loop_quality_feature_ablation_v3_audit | semantic limitation | structurally interpretable only |
| test_per_loop_movement_quality | semantic limitation | structurally interpretable only |
| test_per_loop_quality_shadow | semantic limitation | structurally interpretable only |
| test_per_loop_quality_shadow | semantic limitation | structurally interpretable only |
| test_regime_loop_individual_expert_selection_v1 | semantic limitation | structurally interpretable only |
| test_regime_loop_individual_expert_selection_v1_audit | semantic limitation | structurally interpretable only |
| test_regime_loop_linkage_ideas_v1 | none | fully interpretable |
| test_regime_loop_linkage_ideas_v2 | none | fully interpretable |
| test_regime_loop_linkage_ideas_v3 | semantic limitation | structurally interpretable only |
| test_regime_loop_linkage_ideas_v3_audit | none | fully interpretable |
| test_regime_loop_orientation_calibration_algorithms_v1 | none | fully interpretable |
| test_regime_loop_orientation_calibration_algorithms_v1_audit | none | fully interpretable |

## 26. Tests

Focused tests cover causal provenance, semantic identity, prefix matching, duration/censoring, posterior export, per-bar ledgers, nulls, dictionary closure, migration, frozen hashes, and safety.

- V2 package tests: 75 passed.
- V2 research and artifact tests: 21 passed.
- Existing structural loop suite: 109 passed and 9 failed because hash-pinned historical files under `/private/tmp` were absent.
- Existing state suite: 60 passed and 1 failed because its hash-pinned historical anchor under `/private/tmp` was absent.
- Full repository suite: 832 passed, 13 failed, and 19 errored. All listed failures/errors were older integration surfaces missing frozen artifact inputs, principally `20260714-dynamic-loop-edge-state-v2/primary/trade_decisions.parquet`; no V2-focused test failed.
- Scoped Ruff format, scoped Ruff lint, strict mypy for all seven reusable V2 modules, and `git diff --check`: passed.

## 27. Independent audit

The independent auditor passed 16 of 16 checks. It imported no production V2 module and reconstructed from detailed decisions, state runs, posterior arrays, source files, dictionaries, and raw null draws.

## 28. Exact rerun

Exact rerun identity: `byte_identical`. Artifact-level mismatch details, if any, are in `independent_audit.json`.

## 29. Remaining blockers

| item | affected_rows | status |
| --- | --- | --- |
| initial causal B0 warm-up | 49590 | known_missingness_preserved |
| incomplete or ambiguous in-session source sequence | 26279 | posterior_reset_at_gap; excluded_from_duration_dictionary_and_null; failed_closed_as_UNAVAILABLE |
| state-age posterior support | 424583 | V2_posterior_uses_78_age_support_with_explicit_geometric_tail; legacy_hard_map_remains_separate |
| soft prefix computation | 403991 | bounded_deterministic_sample_by_contract |

These are known limitations, not permission to widen scope. No event source blocker remains for eligible rows.

## 30. Scientific decision

`loop_event_v2_ready_with_known_limitations`

No edge, profitability, strategy, paper-trading, or live-readiness claim is made.

## 31. Exact next experiment

A separately preregistered structural forecast comparing simple baselines and competing-event models for first next-loop identity and arrival time, with no payoff or economic target.
