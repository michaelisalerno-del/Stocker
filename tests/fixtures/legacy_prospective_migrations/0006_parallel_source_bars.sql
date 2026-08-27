CREATE TABLE IF NOT EXISTS source_bar_observation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    provider TEXT NOT NULL,
    provider_record_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    session_date TEXT NOT NULL,
    bar_start_utc TEXT NOT NULL,
    bar_end_utc TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    activity_value REAL,
    activity_semantic_label TEXT NOT NULL,
    source_timestamp_utc TEXT NOT NULL,
    receive_timestamp_utc TEXT NOT NULL,
    completeness TEXT NOT NULL,
    eligibility INTEGER NOT NULL CHECK (eligibility = 0),
    rejection_reason TEXT NOT NULL CHECK (rejection_reason = 'parallel_validation_only'),
    record_hash TEXT NOT NULL,
    UNIQUE(run_id, provider, provider_record_id),
    UNIQUE(run_id, provider, symbol, bar_end_utc)
);

CREATE INDEX IF NOT EXISTS idx_source_bar_observation_session
    ON source_bar_observation(run_id, provider, session_date, symbol, bar_end_utc);

CREATE TABLE IF NOT EXISTS source_capture_completion (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    provider TEXT NOT NULL,
    session_date TEXT NOT NULL,
    status TEXT NOT NULL,
    requested_symbol_count INTEGER NOT NULL,
    captured_symbol_count INTEGER NOT NULL,
    bar_count INTEGER NOT NULL,
    missing_symbols_json TEXT NOT NULL,
    completed_at_utc TEXT NOT NULL,
    UNIQUE(run_id, provider, session_date)
);
