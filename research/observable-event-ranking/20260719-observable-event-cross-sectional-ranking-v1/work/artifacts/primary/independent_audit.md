# Independent audit — Observable Event Ranking V1

- Audit passed: `true`
- Decision audited: `blocked_missing_point_in_time_sector_membership`
- Main runner imported: `false`
- Candidate event, metric, gate, and prediction helpers imported: `false`

## Checks

- `artifact_account_identifiers_absent`: `true`
- `candidate_event_functions_not_imported`: `true`
- `candidate_gate_functions_not_imported`: `true`
- `candidate_metric_functions_not_imported`: `true`
- `candidate_prediction_helpers_not_imported`: `true`
- `contract_hash_binding`: `true`
- `contract_safety_flags`: `true`
- `credentials_and_account_identifiers_absent`: `true`
- `event_ledger_outcome_free`: `true`
- `event_population_empty_before_target_gate`: `true`
- `exact_artifact_manifest_entries_match_files`: `true`
- `exact_rerun_identity`: `true`
- `forbidden_ibkr_calls_absent`: `true`
- `implementation_manifest_hash`: `true`
- `implementation_source_files_match_manifest`: `true`
- `independent_event_and_peer_formula_fixtures`: `true`
- `independent_timing_rank_metric_fixtures`: `true`
- `independent_weight_bootstrap_gate_fixtures`: `true`
- `main_runner_not_imported`: `true`
- `model_not_permitted`: `true`
- `order_models_not_imported`: `true`
- `outcomes_not_read`: `true`
- `primary_artifact_manifest_entries_match_files`: `true`
- `processed_data_unopened`: `true`
- `protected_data_unopened`: `true`
- `sector_effective_dates_fail_closed`: `true`
- `source_inventory_hash`: `true`
- `static_primary_provenance_audit`: `true`
- `targets_not_permitted`: `true`
- `threshold_not_fitted_after_data_blocker`: `true`
- `universe_effective_dates_fail_closed`: `true`

## Not applicable after the pre-target blocker

- `artifact_event_calculation_sample`
- `artifact_leave_one_out_market_sector_medians`
- `artifact_trailing_only_robust_scaling`
- `artifact_threshold_application`
- `artifact_first_event_deduplication`
- `artifact_grid_assignment`
- `artifact_entry_t_plus_2_and_target_60m_timing`
- `artifact_within_slate_ranks_and_equal_weights`
- `artifact_serialized_model_and_baseline_predictions`
- `artifact_spearman_top_two_and_bootstrap_grouping`
- `artifact_concentration_and_gate_logic`

The audit validates a fail-closed pre-target blocker only; no structural, directional, economic, or executable result exists.
