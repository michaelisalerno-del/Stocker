CREATE TABLE IF NOT EXISTS signal_eventization (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL,
    model_score_id INTEGER NOT NULL REFERENCES model_score(id),
    eventization_status TEXT NOT NULL,
    signal_episode_id TEXT REFERENCES signal_episode(id),
    recorded_at_utc TEXT NOT NULL,
    UNIQUE(model_score_id)
);

CREATE INDEX IF NOT EXISTS idx_signal_eventization_status
    ON signal_eventization(run_id, eventization_status);

CREATE VIEW IF NOT EXISTS future_paper_execution_ledger AS
SELECT
    CAST(NULL AS TEXT) AS external_fill_id,
    CAST(NULL AS TEXT) AS signal_episode_id,
    CAST(NULL AS TEXT) AS recorded_at_utc,
    CAST(NULL AS TEXT) AS status
WHERE 0;
