PRAGMA foreign_keys = ON;

ALTER TABLE quiet_state_checkpoint_v0
ADD COLUMN selected_underlying_quote_event_id TEXT;

ALTER TABLE quiet_state_checkpoint_v0
ADD COLUMN selected_underlying_quote_timestamp_utc TEXT;

ALTER TABLE quiet_state_checkpoint_v0
ADD COLUMN selected_underlying_quote_age_seconds REAL
    CHECK (
        selected_underlying_quote_age_seconds IS NULL
        OR selected_underlying_quote_age_seconds >= 0.0
    );

ALTER TABLE quiet_state_checkpoint_v0
ADD COLUMN underlying_quote_selection_policy TEXT;

-- Keep the original prospective rows immutable.  This separate, explicitly
-- non-prospective dataset records which pre-fix classifications were exposed
-- to the mutable-latest-quote instrumentation defect.  It cannot create or
-- authorize observations, episodes, or scientific claims.
CREATE TABLE quiet_quote_instrumentation_defect_v0 (
    defect_id TEXT PRIMARY KEY,
    recorded_at_utc TEXT NOT NULL,
    description TEXT NOT NULL,
    affected_checkpoint_count INTEGER NOT NULL
        CHECK (affected_checkpoint_count >= 0),
    dataset_scope TEXT NOT NULL
        CHECK (dataset_scope = 'non_prospective_derived_instrumentation_audit'),
    original_evidence_modified INTEGER NOT NULL
        CHECK (original_evidence_modified = 0),
    recomputation_authorized INTEGER NOT NULL
        CHECK (recomputation_authorized = 0),
    may_create_quiet_observation INTEGER NOT NULL
        CHECK (may_create_quiet_observation = 0)
);

CREATE TABLE quiet_quote_instrumentation_defect_checkpoint_v0 (
    defect_id TEXT NOT NULL,
    quiet_checkpoint_id INTEGER NOT NULL UNIQUE,
    run_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    session_date TEXT NOT NULL,
    checkpoint INTEGER NOT NULL,
    original_eligible INTEGER NOT NULL,
    original_data_quality_flags_json TEXT NOT NULL,
    PRIMARY KEY (defect_id, quiet_checkpoint_id),
    FOREIGN KEY (defect_id)
        REFERENCES quiet_quote_instrumentation_defect_v0(defect_id),
    FOREIGN KEY (quiet_checkpoint_id)
        REFERENCES quiet_state_checkpoint_v0(id)
);

INSERT INTO quiet_quote_instrumentation_defect_v0(
    defect_id,
    recorded_at_utc,
    description,
    affected_checkpoint_count,
    dataset_scope,
    original_evidence_modified,
    recomputation_authorized,
    may_create_quiet_observation
)
SELECT
    'quiet-boundary-quote-latest-overwrite-v0',
    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
    'Pre-fix checkpoint used mutable latest quote instead of deterministic boundary as-of quote',
    COUNT(*),
    'non_prospective_derived_instrumentation_audit',
    0,
    0,
    0
FROM quiet_state_checkpoint_v0
WHERE underlying_quote_selection_policy IS NULL
  AND data_quality_flags_json LIKE '%"underlying_quote_stale"%';

INSERT INTO quiet_quote_instrumentation_defect_checkpoint_v0(
    defect_id,
    quiet_checkpoint_id,
    run_id,
    symbol,
    session_date,
    checkpoint,
    original_eligible,
    original_data_quality_flags_json
)
SELECT
    'quiet-boundary-quote-latest-overwrite-v0',
    id,
    run_id,
    symbol,
    session_date,
    checkpoint,
    eligible,
    data_quality_flags_json
FROM quiet_state_checkpoint_v0
WHERE underlying_quote_selection_policy IS NULL
  AND data_quality_flags_json LIKE '%"underlying_quote_stale"%';
