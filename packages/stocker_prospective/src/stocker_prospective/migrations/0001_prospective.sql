PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS prospective_run (
    run_id TEXT PRIMARY KEY,
    prospective_start_utc TEXT NOT NULL,
    app_version TEXT NOT NULL,
    git_commit TEXT NOT NULL,
    model_artifact_id TEXT NOT NULL,
    universe_id TEXT NOT NULL,
    cohort TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT 'record_only',
    status TEXT NOT NULL DEFAULT 'active',
    scientific_classification TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence_envelope (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    prospective_start_utc TEXT NOT NULL,
    app_version TEXT NOT NULL,
    git_commit TEXT NOT NULL,
    model_artifact_id TEXT NOT NULL,
    universe_id TEXT NOT NULL,
    cohort TEXT NOT NULL,
    source_timestamps_json TEXT NOT NULL,
    recorded_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS installed_bundle (
    bundle_id TEXT PRIMARY KEY,
    manifest_sha256 TEXT NOT NULL,
    installed_path TEXT NOT NULL UNIQUE,
    installed_at_utc TEXT NOT NULL,
    installed_by TEXT NOT NULL,
    verified INTEGER NOT NULL CHECK (verified IN (0, 1))
);

CREATE TABLE IF NOT EXISTS active_bundle (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    bundle_id TEXT NOT NULL REFERENCES installed_bundle(bundle_id),
    manifest_sha256 TEXT NOT NULL,
    activated_at_utc TEXT NOT NULL,
    activated_by TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS universe_membership (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    universe_id TEXT NOT NULL,
    cohort TEXT NOT NULL,
    symbol TEXT NOT NULL,
    operational_status TEXT NOT NULL,
    rejection_reason TEXT,
    recorded_at_utc TEXT NOT NULL,
    UNIQUE(run_id, cohort, symbol)
);

CREATE TABLE IF NOT EXISTS runtime_session (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    session_date TEXT NOT NULL,
    opened_at_utc TEXT NOT NULL,
    closed_at_utc TEXT,
    status TEXT NOT NULL,
    UNIQUE(run_id, session_date)
);

CREATE TABLE IF NOT EXISTS recorder_lease (
    lease_key TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    acquired_at_utc TEXT NOT NULL,
    heartbeat_at_utc TEXT NOT NULL,
    generation INTEGER NOT NULL,
    recovered_stale_owner INTEGER NOT NULL CHECK (recovered_stale_owner IN (0, 1))
);

CREATE TABLE IF NOT EXISTS underlying_contract (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    con_id INTEGER,
    exchange TEXT,
    currency TEXT,
    local_symbol TEXT,
    qualification_status TEXT NOT NULL,
    rejection_reason TEXT,
    UNIQUE(run_id, symbol, con_id)
);

CREATE TABLE IF NOT EXISTS underlying_bar (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    con_id INTEGER,
    bar_start_utc TEXT NOT NULL,
    bar_end_utc TEXT NOT NULL,
    session_date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    activity_value REAL,
    activity_semantic_label TEXT NOT NULL,
    bar_source TEXT NOT NULL,
    source_timestamp_utc TEXT,
    receive_timestamp_utc TEXT NOT NULL,
    completeness TEXT NOT NULL,
    feature_as_of_utc TEXT,
    m0_probability REAL,
    m1_probability REAL,
    frozen_threshold REAL,
    model_bundle_id TEXT,
    feature_schema_hash TEXT,
    eligibility INTEGER NOT NULL CHECK (eligibility IN (0, 1)),
    rejection_reason TEXT,
    UNIQUE(run_id, symbol, bar_end_utc)
);

CREATE TABLE IF NOT EXISTS underlying_quote (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL,
    signal_episode_id TEXT,
    target_timestamp_utc TEXT NOT NULL,
    actual_quote_timestamp_utc TEXT,
    capture_lag_seconds REAL,
    bid REAL,
    ask REAL,
    bid_size REAL,
    ask_size REAL,
    last REAL,
    last_size REAL,
    midpoint REAL,
    spread REAL,
    provider_timestamp_utc TEXT,
    receive_timestamp_utc TEXT,
    market_data_type TEXT,
    freshness TEXT,
    completeness TEXT NOT NULL,
    capture_status TEXT NOT NULL,
    missing_quote_reason TEXT,
    UNIQUE(run_id, signal_episode_id, target_timestamp_utc)
);

CREATE TABLE IF NOT EXISTS previous_session_options_context (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL,
    current_session_date TEXT NOT NULL,
    required_previous_session TEXT NOT NULL,
    observation_date TEXT,
    provider_identity TEXT NOT NULL,
    source_record_identity_json TEXT NOT NULL,
    context_hash TEXT NOT NULL,
    schema_hash TEXT NOT NULL,
    feature_hash TEXT NOT NULL,
    completeness TEXT NOT NULL,
    freshness TEXT NOT NULL,
    eligibility INTEGER NOT NULL CHECK (eligibility IN (0, 1)),
    rejection_reason TEXT,
    payload_json TEXT NOT NULL,
    UNIQUE(run_id, current_session_date, provider_identity, context_hash)
);

CREATE TABLE IF NOT EXISTS feature_snapshot (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    bar_end_utc TEXT NOT NULL,
    feature_as_of_utc TEXT NOT NULL,
    feature_schema_hash TEXT NOT NULL,
    feature_values_json TEXT NOT NULL,
    parity_status TEXT NOT NULL,
    eligibility INTEGER NOT NULL CHECK (eligibility IN (0, 1)),
    rejection_reason TEXT,
    UNIQUE(run_id, symbol, bar_end_utc, feature_schema_hash)
);

CREATE TABLE IF NOT EXISTS model_score (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL,
    cohort TEXT NOT NULL,
    symbol TEXT NOT NULL,
    bar_end_utc TEXT NOT NULL,
    session_date TEXT NOT NULL,
    feature_as_of_utc TEXT NOT NULL,
    m0_probability REAL,
    m1_probability REAL,
    frozen_threshold REAL NOT NULL,
    model_bundle_id TEXT NOT NULL,
    feature_schema_hash TEXT NOT NULL,
    eligibility INTEGER NOT NULL CHECK (eligibility IN (0, 1)),
    rejection_reason TEXT,
    score_label TEXT NOT NULL,
    UNIQUE(run_id, cohort, symbol, model_bundle_id, bar_end_utc)
);

CREATE TABLE IF NOT EXISTS signal_episode (
    id TEXT PRIMARY KEY,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL,
    cohort TEXT NOT NULL,
    symbol TEXT NOT NULL,
    model_bundle_id TEXT NOT NULL,
    crossing_timestamp_utc TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    startup_above_threshold INTEGER NOT NULL DEFAULT 0 CHECK (startup_above_threshold IN (0, 1)),
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS signal_checkpoint (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL,
    signal_episode_id TEXT NOT NULL REFERENCES signal_episode(id),
    model_score_id INTEGER NOT NULL REFERENCES model_score(id),
    checkpoint_timestamp_utc TEXT NOT NULL,
    m1_probability REAL NOT NULL,
    frozen_threshold REAL NOT NULL,
    UNIQUE(signal_episode_id, model_score_id)
);

CREATE TABLE IF NOT EXISTS option_contract (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL,
    underlying_con_id INTEGER,
    con_id INTEGER NOT NULL,
    local_symbol TEXT NOT NULL,
    expiry TEXT NOT NULL,
    strike REAL NOT NULL,
    right TEXT NOT NULL,
    multiplier TEXT,
    exchange TEXT,
    trading_class TEXT,
    dte_bucket TEXT NOT NULL,
    qualification_status TEXT NOT NULL,
    rejection_reason TEXT,
    UNIQUE(run_id, con_id)
);

CREATE TABLE IF NOT EXISTS option_surface_capture (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL,
    signal_episode_id TEXT NOT NULL REFERENCES signal_episode(id),
    dte_bucket TEXT NOT NULL,
    target_timestamp_utc TEXT NOT NULL,
    actual_quote_timestamp_utc TEXT,
    capture_lag_seconds REAL,
    market_data_type TEXT,
    quote_freshness TEXT,
    completeness TEXT NOT NULL,
    connection_status TEXT NOT NULL,
    budget_status TEXT NOT NULL,
    missing_contract_reason TEXT,
    missing_quote_reason TEXT,
    subscription_error TEXT,
    capture_status TEXT NOT NULL,
    UNIQUE(signal_episode_id, dte_bucket, target_timestamp_utc)
);

CREATE TABLE IF NOT EXISTS option_quote (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL,
    surface_capture_id INTEGER NOT NULL REFERENCES option_surface_capture(id),
    option_contract_id INTEGER NOT NULL REFERENCES option_contract(id),
    bid REAL,
    ask REAL,
    bid_size REAL,
    ask_size REAL,
    last REAL,
    last_size REAL,
    volume REAL,
    open_interest REAL,
    bid_implied_volatility REAL,
    ask_implied_volatility REAL,
    last_implied_volatility REAL,
    model_implied_volatility REAL,
    bid_delta REAL,
    ask_delta REAL,
    last_delta REAL,
    model_delta REAL,
    gamma REAL,
    theta REAL,
    vega REAL,
    underlying_reference_price REAL,
    computation_source TEXT,
    provider_timestamp_utc TEXT,
    receive_timestamp_utc TEXT,
    market_data_type TEXT,
    staleness_seconds REAL,
    completeness TEXT NOT NULL,
    permission_error TEXT,
    UNIQUE(surface_capture_id, option_contract_id)
);

CREATE TABLE IF NOT EXISTS shadow_structure (
    id TEXT PRIMARY KEY,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL,
    signal_episode_id TEXT NOT NULL REFERENCES signal_episode(id),
    cohort TEXT NOT NULL,
    symbol TEXT NOT NULL,
    dte_bucket TEXT NOT NULL,
    structure_type TEXT NOT NULL,
    entry_debit REAL,
    multiplier REAL,
    estimated_fees REAL NOT NULL,
    spread_quality TEXT,
    completeness TEXT NOT NULL,
    rejection_reason TEXT,
    quoted_research_ledger INTEGER NOT NULL DEFAULT 1 CHECK (quoted_research_ledger = 1),
    UNIQUE(signal_episode_id, dte_bucket, structure_type)
);

CREATE TABLE IF NOT EXISTS shadow_leg (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL,
    shadow_structure_id TEXT NOT NULL REFERENCES shadow_structure(id),
    option_contract_id INTEGER NOT NULL REFERENCES option_contract(id),
    leg_role TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    entry_side TEXT NOT NULL,
    entry_price REAL,
    quote_timestamp_utc TEXT,
    UNIQUE(shadow_structure_id, option_contract_id, leg_role)
);

CREATE TABLE IF NOT EXISTS shadow_horizon_valuation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL,
    shadow_structure_id TEXT NOT NULL REFERENCES shadow_structure(id),
    horizon_minutes INTEGER NOT NULL,
    target_timestamp_utc TEXT NOT NULL,
    actual_quote_timestamp_utc TEXT,
    capture_lag_seconds REAL,
    exit_credit REAL,
    gross_return_on_debit REAL,
    gross_pnl REAL,
    estimated_fees REAL NOT NULL,
    market_data_type TEXT,
    completeness TEXT NOT NULL,
    rejection_reason TEXT,
    UNIQUE(shadow_structure_id, horizon_minutes)
);

CREATE TABLE IF NOT EXISTS data_health_event (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL,
    severity TEXT NOT NULL,
    blocker_code TEXT,
    component TEXT NOT NULL,
    message TEXT NOT NULL,
    details_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ibkr_connection_event (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL,
    state TEXT NOT NULL,
    error_code INTEGER,
    message TEXT NOT NULL,
    data_maintained INTEGER,
    reconnect_attempt INTEGER,
    details_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_event (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    actor TEXT NOT NULL,
    message TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    UNIQUE(run_id, sequence)
);

CREATE INDEX IF NOT EXISTS idx_model_score_symbol_time
    ON model_score(run_id, cohort, symbol, bar_end_utc);
CREATE INDEX IF NOT EXISTS idx_signal_episode_time
    ON signal_episode(run_id, crossing_timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_audit_event_sequence
    ON audit_event(run_id, sequence);

