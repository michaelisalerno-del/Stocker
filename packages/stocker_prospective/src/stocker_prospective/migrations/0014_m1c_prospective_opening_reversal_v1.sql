PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS opening_reversal_activation_v1 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    experiment_id TEXT NOT NULL,
    experiment_version TEXT NOT NULL,
    activation_timestamp_utc TEXT NOT NULL,
    new_york_trading_date TEXT NOT NULL,
    configuration_hash TEXT NOT NULL,
    frozen_rule_hash TEXT NOT NULL,
    configured_reserved_line_count INTEGER NOT NULL
        CHECK (configured_reserved_line_count = 12),
    order_routing_disabled INTEGER NOT NULL
        CHECK (order_routing_disabled = 1),
    activation_receipt_hash TEXT NOT NULL UNIQUE,
    receipt_json TEXT NOT NULL,
    UNIQUE(run_id, experiment_id, experiment_version)
);

CREATE TABLE IF NOT EXISTS opening_reversal_prediction_v1 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    experiment_id TEXT NOT NULL,
    experiment_version TEXT NOT NULL,
    session_date TEXT NOT NULL,
    stock TEXT NOT NULL,
    checkpoint INTEGER NOT NULL CHECK (checkpoint = 6),
    signal_timestamp_utc TEXT NOT NULL,
    entry_timestamp_utc TEXT NOT NULL,
    receipt_created_at_utc TEXT NOT NULL,
    m1c_probability REAL,
    m1c_threshold REAL NOT NULL,
    high_tail_membership INTEGER NOT NULL CHECK (high_tail_membership IN (0, 1)),
    fresh_episode_id TEXT,
    tail_phase_v1 TEXT NOT NULL,
    market_opening_return_v1 REAL,
    market_opening_range_v1 REAL,
    opening_market_transition_state_v1 TEXT NOT NULL,
    opening_transition_sign_v1 INTEGER
        CHECK (opening_transition_sign_v1 IS NULL OR opening_transition_sign_v1 IN (-1, 1)),
    opening_transition_event_id_v1 TEXT,
    data_source TEXT NOT NULL,
    transfer_status TEXT NOT NULL,
    cohort_phase TEXT NOT NULL
        CHECK (
            cohort_phase IN (
                'engineering_transfer',
                'prospective_development',
                'untouched_confirmation'
            )
        ),
    prediction_v1 TEXT NOT NULL CHECK (prediction_v1 IN ('CALL', 'PUT', 'ABSTAIN')),
    prediction_sign_v1 INTEGER NOT NULL CHECK (prediction_sign_v1 IN (-1, 0, 1)),
    eligibility_v1 INTEGER NOT NULL CHECK (eligibility_v1 IN (0, 1)),
    ineligibility_reasons_v1_json TEXT NOT NULL,
    completeness_status_v1 TEXT NOT NULL CHECK (completeness_status_v1 IN ('complete', 'incomplete')),
    scientific_outcome_eligible_v1 INTEGER NOT NULL
        CHECK (scientific_outcome_eligible_v1 IN (0, 1)),
    scientific_exclusion_reason_v1 TEXT,
    capacity_snapshot_id TEXT,
    previous_close_atm_iv_scale_15m REAL,
    frozen_comparisons_json TEXT NOT NULL,
    rule_hash_v1 TEXT NOT NULL,
    receipt_hash_v1 TEXT NOT NULL UNIQUE,
    receipt_json TEXT NOT NULL,
    UNIQUE(run_id, experiment_id, experiment_version, session_date, stock, checkpoint)
);

CREATE TABLE IF NOT EXISTS opening_reversal_prediction_correction_v1 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    original_receipt_hash_v1 TEXT NOT NULL
        REFERENCES opening_reversal_prediction_v1(receipt_hash_v1),
    correction_version INTEGER NOT NULL CHECK (correction_version > 0),
    correction_reason TEXT NOT NULL,
    correction_payload_json TEXT NOT NULL,
    correction_hash_v1 TEXT NOT NULL UNIQUE,
    UNIQUE(run_id, original_receipt_hash_v1, correction_version)
);

CREATE TABLE IF NOT EXISTS opening_reversal_promotion_v1 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    session_date TEXT NOT NULL,
    opening_transition_event_id_v1 TEXT NOT NULL,
    promoted_receipt_hash_v1 TEXT
        REFERENCES opening_reversal_prediction_v1(receipt_hash_v1),
    promoted_stock TEXT,
    eligible_count INTEGER NOT NULL CHECK (eligible_count >= 0),
    maximum_promoted_count INTEGER NOT NULL CHECK (maximum_promoted_count = 1),
    selection_rule TEXT NOT NULL,
    non_promoted_json TEXT NOT NULL,
    promotion_hash_v1 TEXT NOT NULL UNIQUE,
    UNIQUE(run_id, opening_transition_event_id_v1)
);

CREATE TABLE IF NOT EXISTS opening_reversal_capacity_snapshot_v1 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    timestamp_utc TEXT NOT NULL,
    configured_budget INTEGER NOT NULL,
    reserved_lines INTEGER NOT NULL CHECK (reserved_lines >= 12),
    mandatory_lines INTEGER NOT NULL,
    optional_lines INTEGER NOT NULL,
    pending_lines INTEGER NOT NULL,
    cancelled_lines INTEGER NOT NULL,
    lines_awaiting_acknowledgement_or_cleanup INTEGER NOT NULL,
    estimated_free_lines INTEGER NOT NULL,
    current_promoted_episode_id TEXT,
    snapshot_hash_v1 TEXT NOT NULL UNIQUE,
    snapshot_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS opening_reversal_degradation_event_v1 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    timestamp_utc TEXT NOT NULL,
    episode_id TEXT NOT NULL,
    feed TEXT,
    subscription_ids_json TEXT NOT NULL,
    reason TEXT NOT NULL,
    raw_capacity_reason TEXT,
    capacity_snapshot_hash_v1 TEXT,
    primary_direction_evidence_remains_complete INTEGER NOT NULL
        CHECK (primary_direction_evidence_remains_complete = 1),
    primary_option_evidence_remains_complete INTEGER NOT NULL
        CHECK (primary_option_evidence_remains_complete IN (0, 1)),
    event_hash_v1 TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS opening_reversal_contract_discovery_v1 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    episode_id TEXT NOT NULL,
    discovery_timestamp_utc TEXT NOT NULL,
    contract_source TEXT NOT NULL,
    cache_hit INTEGER NOT NULL CHECK (cache_hit IN (0, 1)),
    candidates_inspected INTEGER NOT NULL,
    call_con_id INTEGER,
    put_con_id INTEGER,
    expiry TEXT,
    strike REAL,
    tie_break_rule TEXT NOT NULL,
    live_market_data_lines_consumed INTEGER NOT NULL,
    planned_live_market_data_lines INTEGER NOT NULL,
    metadata_request_ended INTEGER NOT NULL CHECK (metadata_request_ended = 1),
    full_chain_live_subscription_created INTEGER NOT NULL
        CHECK (full_chain_live_subscription_created = 0),
    status TEXT NOT NULL,
    missing_reason TEXT,
    audit_hash_v1 TEXT NOT NULL UNIQUE,
    audit_json TEXT NOT NULL,
    UNIQUE(run_id, episode_id)
);

CREATE TABLE IF NOT EXISTS opening_reversal_transfer_session_v1 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    session_date TEXT NOT NULL,
    valid INTEGER NOT NULL CHECK (valid IN (0, 1)),
    valid_session_ordinal INTEGER,
    decision TEXT NOT NULL,
    ibkr_opening_return REAL,
    eodhd_opening_return REAL,
    ibkr_opening_range REAL,
    eodhd_opening_range REAL,
    severe_state_agreement INTEGER CHECK (severe_state_agreement IN (0, 1)),
    sign_agreement INTEGER CHECK (sign_agreement IN (0, 1)),
    timestamp_alignment INTEGER CHECK (timestamp_alignment IN (0, 1)),
    checkpoint_6_episode_identity_agreement INTEGER
        CHECK (checkpoint_6_episode_identity_agreement IN (0, 1)),
    operational_checks_pass INTEGER NOT NULL
        CHECK (operational_checks_pass IN (0, 1)),
    operational_evidence_json TEXT NOT NULL,
    outcome_fields_accessed INTEGER NOT NULL CHECK (outcome_fields_accessed = 0),
    report_json TEXT NOT NULL,
    report_hash_v1 TEXT NOT NULL UNIQUE,
    UNIQUE(run_id, session_date)
);

CREATE TABLE IF NOT EXISTS opening_reversal_underlying_outcome_v1 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    prediction_receipt_hash_v1 TEXT NOT NULL UNIQUE
        REFERENCES opening_reversal_prediction_v1(receipt_hash_v1),
    opening_transition_event_id_v1 TEXT NOT NULL,
    session_date TEXT NOT NULL,
    stock TEXT NOT NULL,
    prediction_v1 TEXT NOT NULL CHECK (prediction_v1 IN ('CALL', 'PUT')),
    r_15m REAL,
    absolute_return_15m REAL,
    threshold_15m REAL NOT NULL,
    outcome_state_v1 TEXT
        CHECK (outcome_state_v1 IN ('MATERIAL_UP', 'MATERIAL_DOWN', 'NO_MATERIAL_MOVE')),
    opening_reversal_aligned_return_v1 REAL,
    correct_predicted_material_direction_v1 INTEGER
        CHECK (
            correct_predicted_material_direction_v1 IS NULL
            OR correct_predicted_material_direction_v1 IN (0, 1)
        ),
    accuracy_counting_no_move_as_failure_v1 INTEGER
        CHECK (accuracy_counting_no_move_as_failure_v1 IN (0, 1)),
    maximum_favourable_excursion_v1 REAL,
    maximum_adverse_excursion_v1 REAL,
    canonical_post_entry_local_range_share_v1 REAL,
    iv_residual_v1 REAL,
    exceed_iv_v1 INTEGER CHECK (exceed_iv_v1 IN (0, 1)),
    outcome_completeness_v1 TEXT NOT NULL
        CHECK (outcome_completeness_v1 IN ('complete', 'incomplete')),
    missing_reason_v1 TEXT,
    outcome_created_at_utc TEXT NOT NULL,
    outcome_receipt_hash_v1 TEXT NOT NULL UNIQUE,
    outcome_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS opening_reversal_primary_option_outcome_v1 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    prediction_receipt_hash_v1 TEXT NOT NULL
        REFERENCES opening_reversal_prediction_v1(receipt_hash_v1),
    con_id INTEGER NOT NULL,
    right TEXT NOT NULL CHECK (right IN ('C', 'P')),
    role TEXT NOT NULL CHECK (role IN ('predicted_leg', 'opposite_leg')),
    expiry TEXT NOT NULL,
    strike REAL NOT NULL,
    entry_bid REAL,
    entry_ask REAL,
    entry_midpoint_diagnostic REAL,
    entry_quote_timestamp_utc TEXT NOT NULL,
    exit_bid REAL,
    exit_ask REAL,
    exit_midpoint_diagnostic REAL,
    exit_quote_timestamp_utc TEXT NOT NULL,
    entry_spread REAL,
    exit_spread REAL,
    entry_quote_age_seconds REAL,
    exit_quote_age_seconds REAL,
    entry_locked_or_crossed INTEGER NOT NULL CHECK (entry_locked_or_crossed IN (0, 1)),
    exit_locked_or_crossed INTEGER NOT NULL CHECK (exit_locked_or_crossed IN (0, 1)),
    entry_stale INTEGER NOT NULL CHECK (entry_stale IN (0, 1)),
    exit_stale INTEGER NOT NULL CHECK (exit_stale IN (0, 1)),
    subscription_start_utc TEXT NOT NULL,
    subscription_end_utc TEXT NOT NULL,
    capacity_line_owner TEXT NOT NULL,
    conservative_return_v1 REAL,
    complete INTEGER NOT NULL CHECK (complete IN (0, 1)),
    missing_reason TEXT,
    outcome_hash_v1 TEXT NOT NULL UNIQUE,
    outcome_json TEXT NOT NULL,
    UNIQUE(run_id, prediction_receipt_hash_v1, role)
);

CREATE TABLE IF NOT EXISTS opening_reversal_decision_receipt_v1 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    receipt_kind TEXT NOT NULL
        CHECK (
            receipt_kind IN (
                'transfer',
                'development',
                'confirmation_start',
                'confirmation',
                'option_economics'
            )
        ),
    boundary_timestamp_utc TEXT NOT NULL,
    decision TEXT NOT NULL,
    cohort_first_session TEXT,
    cohort_last_session TEXT,
    receipt_hash_v1 TEXT NOT NULL UNIQUE,
    receipt_json TEXT NOT NULL,
    UNIQUE(run_id, receipt_kind)
);

CREATE INDEX IF NOT EXISTS idx_opening_reversal_prediction_event_v1
    ON opening_reversal_prediction_v1(
        run_id,
        opening_transition_event_id_v1,
        scientific_outcome_eligible_v1
    );
CREATE INDEX IF NOT EXISTS idx_opening_reversal_prediction_phase_v1
    ON opening_reversal_prediction_v1(run_id, cohort_phase, session_date);
CREATE INDEX IF NOT EXISTS idx_opening_reversal_transfer_valid_v1
    ON opening_reversal_transfer_session_v1(run_id, valid, session_date);
