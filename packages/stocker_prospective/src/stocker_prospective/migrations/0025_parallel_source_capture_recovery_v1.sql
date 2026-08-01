PRAGMA foreign_keys = ON;

-- Preserve the original once-only source_capture_completion row and append
-- bounded same-run recovery attempts when a transient vendor delay produced a
-- partial completion. No recovery row may cross a run or replace the original.
CREATE TABLE IF NOT EXISTS source_capture_recovery_v1 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    provider TEXT NOT NULL,
    session_date TEXT NOT NULL,
    supersedes_source_capture_completion_id INTEGER NOT NULL
        REFERENCES source_capture_completion(id),
    attempt_number INTEGER NOT NULL CHECK (attempt_number >= 1),
    status TEXT NOT NULL CHECK (status IN ('complete', 'partial')),
    requested_symbol_count INTEGER NOT NULL CHECK (requested_symbol_count >= 0),
    captured_symbol_count INTEGER NOT NULL CHECK (captured_symbol_count >= 0),
    bar_count INTEGER NOT NULL CHECK (bar_count >= 0),
    missing_symbols_json TEXT NOT NULL,
    completed_at_utc TEXT NOT NULL,
    recovery_reason TEXT NOT NULL
        CHECK (recovery_reason = 'retry_after_partial_capture'),
    recovery_provenance_json TEXT NOT NULL,
    UNIQUE(run_id, provider, session_date, attempt_number)
);

CREATE INDEX IF NOT EXISTS idx_source_capture_recovery_v1_latest
ON source_capture_recovery_v1(
    run_id,
    provider,
    session_date,
    attempt_number DESC
);

CREATE VIEW IF NOT EXISTS source_capture_effective_v1 AS
SELECT
    recovery.id AS id,
    recovery.envelope_id AS envelope_id,
    recovery.run_id AS run_id,
    recovery.provider AS provider,
    recovery.session_date AS session_date,
    recovery.status AS status,
    recovery.requested_symbol_count AS requested_symbol_count,
    recovery.captured_symbol_count AS captured_symbol_count,
    recovery.bar_count AS bar_count,
    recovery.missing_symbols_json AS missing_symbols_json,
    recovery.completed_at_utc AS completed_at_utc,
    'same_run_recovery' AS capture_record_type,
    recovery.attempt_number AS recovery_attempt_number,
    recovery.supersedes_source_capture_completion_id
        AS supersedes_source_capture_completion_id,
    recovery.recovery_reason AS recovery_reason,
    recovery.recovery_provenance_json AS recovery_provenance_json
FROM source_capture_recovery_v1 AS recovery
WHERE recovery.attempt_number = (
    SELECT MAX(candidate.attempt_number)
    FROM source_capture_recovery_v1 AS candidate
    WHERE candidate.run_id = recovery.run_id
      AND candidate.provider = recovery.provider
      AND candidate.session_date = recovery.session_date
)
UNION ALL
SELECT
    base.id,
    base.envelope_id,
    base.run_id,
    base.provider,
    base.session_date,
    base.status,
    base.requested_symbol_count,
    base.captured_symbol_count,
    base.bar_count,
    base.missing_symbols_json,
    base.completed_at_utc,
    'initial_capture' AS capture_record_type,
    0 AS recovery_attempt_number,
    NULL AS supersedes_source_capture_completion_id,
    NULL AS recovery_reason,
    NULL AS recovery_provenance_json
FROM source_capture_completion AS base
WHERE NOT EXISTS (
    SELECT 1
    FROM source_capture_recovery_v1 AS recovery
    WHERE recovery.run_id = base.run_id
      AND recovery.provider = base.provider
      AND recovery.session_date = base.session_date
);

-- Resolution messages are component-specific. An arbitrary informational
-- event must never clear an active blocker.
DROP TRIGGER IF EXISTS web_active_runtime_blocker_resolve_v0;

CREATE TRIGGER IF NOT EXISTS web_active_runtime_blocker_resolve_v1
AFTER INSERT ON data_health_event
WHEN NEW.blocker_code IS NULL
 AND (
    (
        NEW.component = 'previous_session_options_context'
        AND NEW.message = 'previous_session_options_context_ready'
    )
    OR (
        NEW.component = 'parallel_feature_validation'
        AND NEW.message = 'parallel_source_capture_ready'
    )
 )
BEGIN
    DELETE FROM web_active_runtime_blocker_v0
    WHERE run_id = NEW.run_id
      AND component = NEW.component
      AND event_id < NEW.id;
END;
