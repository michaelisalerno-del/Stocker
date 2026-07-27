PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS promotion_decision_v0 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    promotion_time_utc TEXT NOT NULL,
    symbol TEXT NOT NULL,
    m1c_probability REAL NOT NULL,
    rank INTEGER NOT NULL,
    capacity_available INTEGER NOT NULL,
    subscription_type TEXT NOT NULL,
    reason TEXT NOT NULL,
    claims_json TEXT NOT NULL,
    UNIQUE(run_id, promotion_time_utc, symbol, subscription_type)
);
