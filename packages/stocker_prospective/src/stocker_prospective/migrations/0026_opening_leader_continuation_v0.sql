PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS opening_leader_evidence_v0 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    stable_id TEXT NOT NULL UNIQUE,
    recorder_version TEXT NOT NULL
        CHECK (recorder_version = 'opening-leader-continuation-recorder-v0'),
    deployment_receipt_id TEXT NOT NULL,
    session_date TEXT NOT NULL,
    checkpoint INTEGER NOT NULL CHECK (checkpoint IN (6, 12)),
    signal_timestamp_utc TEXT NOT NULL,
    selected_symbol TEXT,
    record_type TEXT NOT NULL,
    observation_name TEXT NOT NULL,
    observed_at_utc TEXT NOT NULL,
    original_stable_id TEXT REFERENCES opening_leader_evidence_v0(stable_id),
    cohort_hash TEXT NOT NULL,
    contract_hash TEXT NOT NULL,
    code_hash TEXT NOT NULL,
    data_quality_flags_json TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    UNIQUE(run_id, stable_id)
);

CREATE INDEX IF NOT EXISTS idx_opening_leader_evidence_session_v0
    ON opening_leader_evidence_v0(run_id, session_date, checkpoint, id);
CREATE INDEX IF NOT EXISTS idx_opening_leader_evidence_type_v0
    ON opening_leader_evidence_v0(run_id, record_type, observation_name, id);

CREATE TRIGGER IF NOT EXISTS opening_leader_evidence_no_update_v0
BEFORE UPDATE ON opening_leader_evidence_v0
BEGIN
    SELECT RAISE(ABORT, 'opening-leader evidence is append-only');
END;

CREATE TRIGGER IF NOT EXISTS opening_leader_evidence_no_delete_v0
BEFORE DELETE ON opening_leader_evidence_v0
BEGIN
    SELECT RAISE(ABORT, 'opening-leader evidence is append-only');
END;
