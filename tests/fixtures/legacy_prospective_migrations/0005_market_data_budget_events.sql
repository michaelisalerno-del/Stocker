CREATE TABLE IF NOT EXISTS market_data_budget_event (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL,
    line_limit INTEGER NOT NULL,
    reserved_headroom INTEGER NOT NULL,
    usable_lines INTEGER NOT NULL,
    active_lines INTEGER NOT NULL,
    pending_requests INTEGER NOT NULL,
    awaiting_cancellation INTEGER NOT NULL,
    current_request_rate INTEGER NOT NULL,
    waiting_signals INTEGER NOT NULL,
    rejected_signals INTEGER NOT NULL,
    recorded_at_utc TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_market_data_budget_event_run
    ON market_data_budget_event(run_id, id);
