PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS quiet_state_checkpoint_v0 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    checkpoint_id INTEGER NOT NULL UNIQUE REFERENCES m1c_checkpoint_v0(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    symbol TEXT NOT NULL,
    session_date TEXT NOT NULL,
    checkpoint INTEGER NOT NULL,
    m1c_probability REAL NOT NULL,
    previous_m1c_probability REAL,
    bottom_5 INTEGER NOT NULL CHECK (bottom_5 IN (0, 1)),
    bottom_10 INTEGER NOT NULL CHECK (bottom_10 IN (0, 1)),
    bottom_20 INTEGER NOT NULL CHECK (bottom_20 IN (0, 1)),
    high_tail INTEGER NOT NULL CHECK (high_tail IN (0, 1)),
    distance_from_bottom_10 REAL NOT NULL,
    model_hash TEXT NOT NULL,
    feature_hash TEXT NOT NULL,
    eligible INTEGER NOT NULL CHECK (eligible IN (0, 1)),
    data_quality_status TEXT NOT NULL,
    data_quality_flags_json TEXT NOT NULL,
    claims_json TEXT NOT NULL,
    UNIQUE(run_id, symbol, session_date, checkpoint)
);

CREATE TABLE IF NOT EXISTS quiet_state_observation_v0 (
    observation_id TEXT PRIMARY KEY,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    quiet_checkpoint_id INTEGER NOT NULL REFERENCES quiet_state_checkpoint_v0(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    observation_kind TEXT NOT NULL CHECK (
        observation_kind IN (
            'quiet_bottom_10',
            'neutral_control',
            'high_tail_control'
        )
    ),
    symbol TEXT NOT NULL,
    session_date TEXT NOT NULL,
    trigger_checkpoint INTEGER NOT NULL,
    trigger_timestamp_utc TEXT NOT NULL,
    prospective_entry_timestamp_utc TEXT NOT NULL,
    m1c_probability REAL NOT NULL,
    previous_m1c_probability REAL,
    bottom_5 INTEGER NOT NULL CHECK (bottom_5 IN (0, 1)),
    bottom_10 INTEGER NOT NULL CHECK (bottom_10 IN (0, 1)),
    bottom_20 INTEGER NOT NULL CHECK (bottom_20 IN (0, 1)),
    high_tail INTEGER NOT NULL CHECK (high_tail IN (0, 1)),
    episode_number INTEGER,
    minutes_since_previous_quiet_episode REAL,
    previous_high_tail_within_60_minutes INTEGER NOT NULL
        CHECK (previous_high_tail_within_60_minutes IN (0, 1)),
    following_high_tail_within_60_minutes INTEGER NOT NULL
        CHECK (following_high_tail_within_60_minutes IN (0, 1)),
    neutral_hash_hex TEXT,
    neutral_hash_fraction REAL,
    neutral_sampling_fraction REAL,
    neutral_salt_id TEXT,
    option_plan_recorded INTEGER NOT NULL DEFAULT 0
        CHECK (option_plan_recorded IN (0, 1)),
    option_plan_requested_contract_count INTEGER NOT NULL DEFAULT 0,
    option_plan_selected_contract_count INTEGER NOT NULL DEFAULT 0,
    option_plan_capacity_reduced INTEGER NOT NULL DEFAULT 0
        CHECK (option_plan_capacity_reduced IN (0, 1)),
    option_plan_missing_buckets_json TEXT NOT NULL DEFAULT '[]',
    option_context_valid INTEGER NOT NULL DEFAULT 0
        CHECK (option_context_valid IN (0, 1)),
    scientific_recording_valid INTEGER NOT NULL
        CHECK (scientific_recording_valid IN (0, 1)),
    data_quality_flags_json TEXT NOT NULL,
    phase TEXT NOT NULL,
    completion_status TEXT NOT NULL,
    completed_at_utc TEXT,
    claims_json TEXT NOT NULL,
    UNIQUE(run_id, symbol, session_date, trigger_checkpoint, observation_kind)
);

CREATE TABLE IF NOT EXISTS quiet_state_microstructure_v0 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    observation_id TEXT NOT NULL REFERENCES quiet_state_observation_v0(observation_id),
    window_name TEXT NOT NULL,
    window_start_utc TEXT NOT NULL,
    window_end_utc TEXT NOT NULL,
    summary_json TEXT NOT NULL,
    quality_flags_json TEXT NOT NULL,
    claims_json TEXT NOT NULL,
    UNIQUE(observation_id, window_name, window_end_utc)
);

CREATE TABLE IF NOT EXISTS quiet_state_underlying_path_v0 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    observation_id TEXT NOT NULL REFERENCES quiet_state_observation_v0(observation_id),
    horizon_label TEXT NOT NULL,
    target_timestamp_utc TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    quality_flags_json TEXT NOT NULL,
    claims_json TEXT NOT NULL,
    UNIQUE(observation_id, horizon_label)
);

CREATE TABLE IF NOT EXISTS quiet_state_option_contract_v0 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    observation_id TEXT NOT NULL REFERENCES quiet_state_observation_v0(observation_id),
    underlying_con_id INTEGER NOT NULL,
    con_id INTEGER,
    expiry TEXT NOT NULL,
    dte INTEGER NOT NULL,
    dte_bucket TEXT NOT NULL CHECK (
        dte_bucket IN ('0DTE', '1DTE', '3_TO_5_DTE')
    ),
    strike REAL NOT NULL,
    right TEXT NOT NULL CHECK (right IN ('C', 'P')),
    multiplier INTEGER NOT NULL,
    exchange TEXT NOT NULL,
    trading_class TEXT NOT NULL,
    selection_rank INTEGER NOT NULL,
    selection_roles_json TEXT NOT NULL,
    resolution_status TEXT NOT NULL,
    rejection_reason TEXT,
    recording_started_at_utc TEXT,
    recording_ends_at_utc TEXT,
    claims_json TEXT NOT NULL,
    UNIQUE(observation_id, expiry, strike, right)
);

CREATE TABLE IF NOT EXISTS quiet_state_option_quote_state_v0 (
    option_contract_id INTEGER PRIMARY KEY
        REFERENCES quiet_state_option_contract_v0(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    observation_id TEXT NOT NULL REFERENCES quiet_state_observation_v0(observation_id),
    provider_timestamp_utc TEXT,
    received_timestamp_utc TEXT NOT NULL,
    bid REAL,
    bid_size REAL,
    ask REAL,
    ask_size REAL,
    last REAL,
    last_size REAL,
    market_data_type TEXT NOT NULL,
    option_model_price REAL,
    implied_volatility REAL,
    delta REAL,
    gamma REAL,
    theta REAL,
    vega REAL,
    underlying_reference_price REAL,
    volume REAL,
    open_interest REAL,
    quote_attributes_json TEXT NOT NULL,
    recording_status TEXT NOT NULL,
    quote_quality_flags_json TEXT NOT NULL,
    claims_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS quiet_state_shadow_outcome_v0 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    observation_id TEXT NOT NULL REFERENCES quiet_state_observation_v0(observation_id),
    structure_type TEXT NOT NULL CHECK (
        structure_type IN (
            'LONG_CALL',
            'LONG_PUT',
            'ATM_STRADDLE',
            'ATM_IRON_BUTTERFLY',
            'DELTA_IRON_CONDOR',
            'CALL_CREDIT_SPREAD',
            'PUT_CREDIT_SPREAD'
        )
    ),
    dte_bucket TEXT NOT NULL CHECK (
        dte_bucket IN ('0DTE', '1DTE', '3_TO_5_DTE')
    ),
    horizon_label TEXT NOT NULL,
    horizon_minutes INTEGER,
    opening_credit_or_debit REAL,
    maximum_defined_risk REAL,
    conservative_pnl REAL,
    return_on_maximum_risk REAL,
    short_strike_touched INTEGER,
    protective_wing_touched INTEGER,
    attempted INTEGER NOT NULL CHECK (attempted IN (0, 1)),
    complete_quote_quality INTEGER NOT NULL CHECK (complete_quote_quality IN (0, 1)),
    strict_quote_quality INTEGER NOT NULL CHECK (strict_quote_quality IN (0, 1)),
    quality_status TEXT NOT NULL,
    quality_flags_json TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    live_decision_panel_visible INTEGER NOT NULL
        CHECK (live_decision_panel_visible = 0),
    claims_json TEXT NOT NULL,
    UNIQUE(observation_id, structure_type, dte_bucket, horizon_label)
);

CREATE INDEX IF NOT EXISTS idx_quiet_checkpoint_live
    ON quiet_state_checkpoint_v0(run_id, symbol, checkpoint);
CREATE INDEX IF NOT EXISTS idx_quiet_observation_live
    ON quiet_state_observation_v0(run_id, trigger_timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_quiet_shadow_live
    ON quiet_state_shadow_outcome_v0(observation_id, horizon_label);
