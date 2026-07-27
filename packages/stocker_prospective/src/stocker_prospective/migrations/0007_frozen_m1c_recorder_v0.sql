PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS m1c_checkpoint_v0 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    symbol TEXT NOT NULL,
    session_date TEXT NOT NULL,
    checkpoint INTEGER NOT NULL,
    bar_start_utc TEXT NOT NULL,
    bar_end_utc TEXT NOT NULL,
    feature_as_of_utc TEXT NOT NULL,
    model_id TEXT NOT NULL CHECK (model_id = 'M1C'),
    model_version TEXT NOT NULL,
    model_hash TEXT NOT NULL,
    feature_hash TEXT NOT NULL,
    session_context_hash TEXT NOT NULL,
    feature_values_json TEXT NOT NULL,
    probability REAL NOT NULL,
    threshold REAL NOT NULL,
    threshold_passed INTEGER NOT NULL CHECK (threshold_passed IN (0, 1)),
    eligible INTEGER NOT NULL CHECK (eligible IN (0, 1)),
    feature_freshness TEXT NOT NULL,
    missing_feature_count INTEGER NOT NULL,
    rejection_reasons_json TEXT NOT NULL,
    claims_json TEXT NOT NULL,
    UNIQUE(run_id, symbol, session_date, checkpoint)
);

CREATE TABLE IF NOT EXISTS m1c_episode_v0 (
    episode_id TEXT PRIMARY KEY,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    checkpoint_id INTEGER NOT NULL REFERENCES m1c_checkpoint_v0(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    symbol TEXT NOT NULL,
    session_date TEXT NOT NULL,
    trigger_checkpoint INTEGER NOT NULL,
    trigger_bar_end_utc TEXT NOT NULL,
    prospective_entry_timestamp_utc TEXT NOT NULL,
    m1c_probability REAL NOT NULL,
    previous_m1c_probability REAL,
    episode_number INTEGER NOT NULL,
    minutes_since_previous_episode REAL,
    scientific_recording_valid INTEGER NOT NULL
        CHECK (scientific_recording_valid IN (0, 1)),
    rejection_reasons_json TEXT NOT NULL,
    phase TEXT NOT NULL,
    completion_status TEXT NOT NULL,
    completed_at_utc TEXT,
    claims_json TEXT NOT NULL,
    UNIQUE(run_id, symbol, session_date, trigger_checkpoint, trigger_bar_end_utc)
);

CREATE TABLE IF NOT EXISTS direction_classification_v0 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    episode_id TEXT NOT NULL REFERENCES m1c_episode_v0(episode_id),
    archetype TEXT NOT NULL CHECK (archetype IN ('A1', 'C1', 'R1')),
    probability_up REAL NOT NULL,
    confidence REAL NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('CALL', 'PUT', 'ABSTAIN')),
    confidence_boundary REAL NOT NULL,
    classification_label TEXT NOT NULL,
    model_hash TEXT NOT NULL,
    preprocessing_hash TEXT NOT NULL,
    feature_hash TEXT NOT NULL,
    maximum_feature_timestamp_utc TEXT NOT NULL,
    trigger_bar_excluded INTEGER NOT NULL CHECK (trigger_bar_excluded = 1),
    valid INTEGER NOT NULL CHECK (valid IN (0, 1)),
    payload_json TEXT NOT NULL,
    claims_json TEXT NOT NULL,
    UNIQUE(episode_id, archetype)
);

CREATE TABLE IF NOT EXISTS group_o_session_context_v0 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    symbol TEXT NOT NULL,
    signal_session TEXT NOT NULL,
    required_option_observation_session TEXT NOT NULL,
    actual_option_observation_session TEXT,
    front_expiry TEXT,
    dte INTEGER,
    atm_strike REAL,
    features_json TEXT NOT NULL,
    missing_indicators_json TEXT NOT NULL,
    quality_status TEXT NOT NULL,
    source_receipt_hashes_json TEXT NOT NULL,
    context_hash TEXT NOT NULL,
    claims_json TEXT NOT NULL,
    UNIQUE(run_id, symbol, signal_session)
);

CREATE TABLE IF NOT EXISTS microstructure_summary_v0 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    episode_id TEXT REFERENCES m1c_episode_v0(episode_id),
    symbol TEXT NOT NULL,
    window_name TEXT NOT NULL,
    window_start_utc TEXT NOT NULL,
    window_end_utc TEXT NOT NULL,
    calculated_at_utc TEXT NOT NULL,
    level1_valid INTEGER NOT NULL CHECK (level1_valid IN (0, 1)),
    tick_valid INTEGER NOT NULL CHECK (tick_valid IN (0, 1)),
    depth_valid INTEGER NOT NULL CHECK (depth_valid IN (0, 1)),
    summary_json TEXT NOT NULL,
    component_json TEXT NOT NULL,
    archetype_relationship_json TEXT NOT NULL,
    quality_flags_json TEXT NOT NULL,
    claims_json TEXT NOT NULL,
    UNIQUE(run_id, symbol, window_name, window_end_utc)
);

CREATE TABLE IF NOT EXISTS raw_partition_manifest_v0 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    data_source TEXT NOT NULL,
    session_date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    event_type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    minimum_timestamp_utc TEXT NOT NULL,
    maximum_timestamp_utc TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    complete INTEGER NOT NULL CHECK (complete IN (0, 1)),
    gap_count INTEGER NOT NULL,
    recorder_version TEXT NOT NULL,
    contract_version TEXT NOT NULL,
    recorded_at_utc TEXT NOT NULL,
    claims_json TEXT NOT NULL,
    UNIQUE(run_id, content_hash)
);

CREATE TABLE IF NOT EXISTS subscription_lifecycle_v0 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    subscription_key TEXT NOT NULL,
    request_id INTEGER NOT NULL,
    subscription_kind TEXT NOT NULL,
    symbol TEXT NOT NULL,
    con_id INTEGER NOT NULL,
    priority INTEGER NOT NULL,
    owner_episode TEXT,
    started_at_utc TEXT NOT NULL,
    cancelled_at_utc TEXT,
    cancellation_reason TEXT,
    ibkr_error_codes_json TEXT NOT NULL,
    capacity_denied INTEGER NOT NULL CHECK (capacity_denied IN (0, 1)),
    claims_json TEXT NOT NULL,
    UNIQUE(run_id, subscription_key, started_at_utc)
);

CREATE TABLE IF NOT EXISTS shadow_quote_outcome_v0 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    episode_id TEXT NOT NULL REFERENCES m1c_episode_v0(episode_id),
    archetype TEXT NOT NULL,
    direction TEXT NOT NULL,
    dte_bucket TEXT NOT NULL,
    con_id INTEGER,
    contract_identity TEXT NOT NULL,
    horizon_minutes INTEGER NOT NULL,
    target_timestamp_utc TEXT NOT NULL,
    entry_ask REAL,
    entry_bid REAL,
    exit_bid REAL,
    exit_ask REAL,
    ask_to_bid_return REAL,
    dollar_pnl_per_contract REAL,
    payload_json TEXT NOT NULL,
    quality_flags_json TEXT NOT NULL,
    valid INTEGER NOT NULL CHECK (valid IN (0, 1)),
    claims_json TEXT NOT NULL,
    UNIQUE(episode_id, archetype, dte_bucket, contract_identity, horizon_minutes)
);

CREATE TABLE IF NOT EXISTS shadow_structure_outcome_v0 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    episode_id TEXT NOT NULL REFERENCES m1c_episode_v0(episode_id),
    structure_type TEXT NOT NULL
        CHECK (structure_type IN ('ATM_STRADDLE', 'RETROSPECTIVE_ORACLE')),
    dte_bucket TEXT NOT NULL,
    horizon_minutes INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    valid INTEGER NOT NULL CHECK (valid IN (0, 1)),
    live_decision_panel_visible INTEGER NOT NULL
        CHECK (live_decision_panel_visible = 0),
    claims_json TEXT NOT NULL,
    UNIQUE(episode_id, structure_type, dte_bucket, horizon_minutes)
);

CREATE TABLE IF NOT EXISTS episode_option_contract_v0 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    episode_id TEXT NOT NULL REFERENCES m1c_episode_v0(episode_id),
    underlying_con_id INTEGER NOT NULL,
    con_id INTEGER,
    expiry TEXT NOT NULL,
    dte INTEGER NOT NULL,
    dte_bucket TEXT NOT NULL,
    strike REAL NOT NULL,
    right TEXT NOT NULL CHECK (right IN ('C', 'P')),
    multiplier INTEGER NOT NULL,
    exchange TEXT NOT NULL,
    trading_class TEXT NOT NULL,
    selection_rank INTEGER NOT NULL,
    resolution_status TEXT NOT NULL,
    rejection_reason TEXT,
    recording_started_at_utc TEXT,
    recording_ends_at_utc TEXT,
    claims_json TEXT NOT NULL,
    UNIQUE(episode_id, expiry, strike, right)
);

CREATE TABLE IF NOT EXISTS option_quote_state_v0 (
    option_contract_id INTEGER PRIMARY KEY
        REFERENCES episode_option_contract_v0(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    episode_id TEXT NOT NULL REFERENCES m1c_episode_v0(episode_id),
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

CREATE TABLE IF NOT EXISTS underlying_live_state_v0 (
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    symbol TEXT NOT NULL,
    con_id INTEGER NOT NULL,
    request_id INTEGER NOT NULL,
    provider_timestamp_utc TEXT,
    received_timestamp_utc TEXT NOT NULL,
    received_monotonic_ns INTEGER NOT NULL,
    source_sequence INTEGER NOT NULL,
    bid REAL,
    bid_size REAL,
    ask REAL,
    ask_size REAL,
    last REAL,
    last_size REAL,
    midpoint REAL,
    spread REAL,
    quote_size_imbalance REAL,
    microprice_edge_bps REAL,
    market_data_type TEXT NOT NULL,
    quote_valid INTEGER NOT NULL CHECK (quote_valid IN (0, 1)),
    tick_by_tick_status TEXT NOT NULL,
    depth_status TEXT NOT NULL,
    claims_json TEXT NOT NULL,
    PRIMARY KEY(run_id, symbol)
);

CREATE TABLE IF NOT EXISTS completed_bar_state_v0 (
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    symbol TEXT NOT NULL,
    session_date TEXT NOT NULL,
    bar_start_utc TEXT NOT NULL,
    bar_end_utc TEXT NOT NULL,
    checkpoint INTEGER NOT NULL,
    source TEXT NOT NULL,
    source_completeness TEXT NOT NULL,
    received_timestamp_utc TEXT NOT NULL,
    claims_json TEXT NOT NULL,
    PRIMARY KEY(run_id, symbol)
);

CREATE TABLE IF NOT EXISTS recorder_session_report_v0 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    session_date TEXT NOT NULL,
    report_json TEXT NOT NULL,
    partition_hashes_json TEXT NOT NULL,
    complete INTEGER NOT NULL CHECK (complete IN (0, 1)),
    generated_at_utc TEXT NOT NULL,
    claims_json TEXT NOT NULL,
    UNIQUE(run_id, session_date)
);

CREATE INDEX IF NOT EXISTS idx_m1c_checkpoint_live
    ON m1c_checkpoint_v0(run_id, symbol, bar_end_utc);
CREATE INDEX IF NOT EXISTS idx_m1c_episode_live
    ON m1c_episode_v0(run_id, trigger_bar_end_utc);
CREATE INDEX IF NOT EXISTS idx_direction_episode_live
    ON direction_classification_v0(episode_id, archetype);
CREATE INDEX IF NOT EXISTS idx_microstructure_episode_live
    ON microstructure_summary_v0(episode_id, window_end_utc);
CREATE INDEX IF NOT EXISTS idx_shadow_episode_live
    ON shadow_quote_outcome_v0(episode_id, horizon_minutes);
