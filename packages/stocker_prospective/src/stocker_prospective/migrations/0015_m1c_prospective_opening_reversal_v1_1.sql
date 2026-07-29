PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS opening_reversal_causal_barrier_audit_v1_1 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    experiment_id TEXT NOT NULL,
    experiment_version TEXT NOT NULL CHECK (experiment_version = '1.1'),
    activation_receipt_hash_v1_1 TEXT NOT NULL,
    session_date TEXT NOT NULL,
    nominal_entry_timestamp_utc TEXT NOT NULL,
    prediction_receipt_count INTEGER NOT NULL
        CHECK (prediction_receipt_count BETWEEN 0 AND 20),
    prediction_receipt_hashes_json TEXT NOT NULL,
    deferred_event_count INTEGER NOT NULL CHECK (deferred_event_count >= 0),
    first_deferred_event_received_at_utc TEXT,
    entry_or_post_entry_data_admitted_before_receipts INTEGER NOT NULL
        CHECK (entry_or_post_entry_data_admitted_before_receipts IN (0, 1)),
    raw_event_archive_write_allowed INTEGER NOT NULL
        CHECK (raw_event_archive_write_allowed = 1),
    core_recorder_continued INTEGER NOT NULL
        CHECK (core_recorder_continued = 1),
    barrier_status TEXT NOT NULL
        CHECK (barrier_status IN ('passed', 'failed_closed')),
    failure_reason TEXT,
    release_authorized_at_utc TEXT NOT NULL,
    audit_hash_v1_1 TEXT NOT NULL UNIQUE,
    audit_json TEXT NOT NULL,
    UNIQUE(run_id, session_date)
);

CREATE INDEX IF NOT EXISTS idx_opening_reversal_barrier_status_v1_1
    ON opening_reversal_causal_barrier_audit_v1_1(
        run_id,
        barrier_status,
        session_date
    );
