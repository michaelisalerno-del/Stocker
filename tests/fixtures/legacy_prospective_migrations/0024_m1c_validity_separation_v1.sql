-- Numbered after the highest deployed migration observed during the repair audit.
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS m1c_checkpoint_validity_v1 (
    checkpoint_id INTEGER PRIMARY KEY REFERENCES m1c_checkpoint_v0(id),
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    symbol TEXT NOT NULL,
    session_date TEXT NOT NULL,
    checkpoint INTEGER NOT NULL,
    m1c_computation_valid INTEGER NOT NULL
        CHECK (m1c_computation_valid IN (0, 1)),
    m1c_computation_reasons_json TEXT NOT NULL,
    source_transfer_valid INTEGER NOT NULL
        CHECK (source_transfer_valid IN (0, 1)),
    source_transfer_reasons_json TEXT NOT NULL,
    opening_reversal_prediction_eligible INTEGER NOT NULL
        CHECK (opening_reversal_prediction_eligible IN (0, 1)),
    opening_reversal_prediction_reasons_json TEXT NOT NULL,
    promotion_eligible INTEGER NOT NULL
        CHECK (promotion_eligible IN (0, 1)),
    promotion_reasons_json TEXT NOT NULL,
    option_recording_ready INTEGER NOT NULL
        CHECK (option_recording_ready IN (0, 1)),
    option_recording_reasons_json TEXT NOT NULL,
    recording_mode TEXT NOT NULL
        CHECK (
            recording_mode IN (
                'live_checkpoint',
                'same_run_durable_reconstruction'
            )
        ),
    recovery_provenance_json TEXT NOT NULL,
    recorded_at_utc TEXT NOT NULL,
    claims_json TEXT NOT NULL,
    UNIQUE(run_id, symbol, session_date, checkpoint)
);

CREATE INDEX IF NOT EXISTS idx_m1c_checkpoint_validity_transfer_v1
ON m1c_checkpoint_validity_v1(
    run_id,
    session_date,
    source_transfer_valid,
    symbol,
    checkpoint
);

CREATE TABLE IF NOT EXISTS provider_m1c_transfer_validity_v1 (
    provider_observation_id INTEGER PRIMARY KEY
        REFERENCES provider_m1c_observation_v0(id),
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    provider TEXT NOT NULL CHECK (provider IN ('ibkr', 'eodhd')),
    symbol TEXT NOT NULL,
    session_date TEXT NOT NULL,
    checkpoint INTEGER NOT NULL,
    bar_complete INTEGER NOT NULL CHECK (bar_complete IN (0, 1)),
    source_transfer_valid INTEGER NOT NULL
        CHECK (source_transfer_valid IN (0, 1)),
    source_transfer_reasons_json TEXT NOT NULL,
    quiet_episode INTEGER NOT NULL CHECK (quiet_episode IN (0, 1)),
    high_tail_episode INTEGER NOT NULL CHECK (high_tail_episode IN (0, 1)),
    adjudication_mode TEXT NOT NULL CHECK (
        adjudication_mode IN (
            'live_provider_projection',
            'same_run_durable_reconstruction'
        )
    ),
    provenance_json TEXT NOT NULL,
    recorded_at_utc TEXT NOT NULL,
    claims_json TEXT NOT NULL,
    UNIQUE(run_id, provider, symbol, session_date, checkpoint)
);

CREATE INDEX IF NOT EXISTS idx_provider_m1c_transfer_validity_v1
ON provider_m1c_transfer_validity_v1(
    run_id,
    session_date,
    provider,
    source_transfer_valid,
    symbol,
    checkpoint
);

CREATE TABLE IF NOT EXISTS opening_reversal_operational_state_v1 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    session_date TEXT NOT NULL,
    episode_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    prediction_receipt_hash_v1 TEXT NOT NULL
        REFERENCES opening_reversal_prediction_v1(receipt_hash_v1),
    state TEXT NOT NULL CHECK (
        state IN (
            'promotion_candidate',
            'promoted',
            'level1_subscription_started',
            'level1_ready',
            'level1_unavailable',
            'option_pair_selected',
            'option_recording_ready',
            'option_recording_unavailable'
        )
    ),
    reason TEXT,
    occurred_at_utc TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    event_hash_v1 TEXT NOT NULL UNIQUE,
    claims_json TEXT NOT NULL,
    UNIQUE(run_id, episode_id, state)
);

CREATE INDEX IF NOT EXISTS idx_opening_reversal_operational_state_v1
ON opening_reversal_operational_state_v1(
    run_id,
    session_date,
    state,
    symbol
);

CREATE TABLE IF NOT EXISTS subscription_capacity_denial_v1 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    occurred_at_utc TEXT NOT NULL,
    subscription_key TEXT NOT NULL,
    subscription_kind TEXT NOT NULL,
    subscription_class INTEGER NOT NULL CHECK (subscription_class BETWEEN 0 AND 5),
    symbol TEXT NOT NULL,
    reason TEXT NOT NULL,
    budget_state TEXT NOT NULL,
    mandatory_feed INTEGER NOT NULL CHECK (mandatory_feed IN (0, 1)),
    claims_json TEXT NOT NULL,
    UNIQUE(run_id, occurred_at_utc, subscription_key, reason)
);

CREATE INDEX IF NOT EXISTS idx_subscription_capacity_denial_v1
ON subscription_capacity_denial_v1(
    run_id,
    occurred_at_utc,
    mandatory_feed
);

CREATE TABLE IF NOT EXISTS source_transfer_session_revision_v1 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    session_date TEXT NOT NULL,
    supersedes_source_transfer_session_id INTEGER NOT NULL
        REFERENCES source_transfer_session_v0(id),
    revision_number INTEGER NOT NULL CHECK (revision_number >= 1),
    valid INTEGER NOT NULL CHECK (valid IN (0, 1)),
    decision TEXT NOT NULL,
    report_json TEXT NOT NULL,
    generated_at_utc TEXT NOT NULL,
    revision_reason TEXT NOT NULL,
    revision_provenance_json TEXT NOT NULL,
    claims_json TEXT NOT NULL,
    UNIQUE(run_id, session_date, revision_number)
);

CREATE VIEW IF NOT EXISTS source_transfer_session_effective_v1 AS
SELECT
    revision.id AS id,
    revision.envelope_id AS envelope_id,
    revision.run_id AS run_id,
    revision.session_date AS session_date,
    revision.valid AS valid,
    revision.decision AS decision,
    revision.report_json AS report_json,
    revision.generated_at_utc AS generated_at_utc,
    revision.claims_json AS claims_json,
    revision.revision_number AS revision_number
FROM source_transfer_session_revision_v1 AS revision
WHERE revision.revision_number = (
    SELECT MAX(candidate.revision_number)
    FROM source_transfer_session_revision_v1 AS candidate
    WHERE candidate.run_id = revision.run_id
      AND candidate.session_date = revision.session_date
)
UNION ALL
SELECT
    base.id,
    base.envelope_id,
    base.run_id,
    base.session_date,
    base.valid,
    base.decision,
    base.report_json,
    base.generated_at_utc,
    base.claims_json,
    0 AS revision_number
FROM source_transfer_session_v0 AS base
WHERE NOT EXISTS (
    SELECT 1
    FROM source_transfer_session_revision_v1 AS revision
    WHERE revision.run_id = base.run_id
      AND revision.session_date = base.session_date
);

CREATE TABLE IF NOT EXISTS prospective_session_phase_revision_v1 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    session_date TEXT NOT NULL,
    supersedes_prospective_session_phase_id INTEGER NOT NULL
        REFERENCES prospective_session_phase_v0(id),
    revision_number INTEGER NOT NULL CHECK (revision_number >= 1),
    valid_session_ordinal INTEGER,
    phase TEXT NOT NULL CHECK (
        phase IN ('engineering_transfer', 'option_development', 'untouched_confirmation')
    ),
    valid INTEGER NOT NULL CHECK (valid IN (0, 1)),
    source_transfer_decision TEXT,
    strategy_rule_changes_allowed INTEGER NOT NULL
        CHECK (strategy_rule_changes_allowed = 0),
    recorded_at_utc TEXT NOT NULL,
    revision_reason TEXT NOT NULL,
    revision_provenance_json TEXT NOT NULL,
    claims_json TEXT NOT NULL,
    UNIQUE(run_id, session_date, revision_number)
);

CREATE VIEW IF NOT EXISTS prospective_session_phase_effective_v1 AS
SELECT
    revision.id AS id,
    revision.envelope_id AS envelope_id,
    revision.run_id AS run_id,
    revision.session_date AS session_date,
    revision.valid_session_ordinal AS valid_session_ordinal,
    revision.phase AS phase,
    revision.valid AS valid,
    revision.source_transfer_decision AS source_transfer_decision,
    revision.strategy_rule_changes_allowed AS strategy_rule_changes_allowed,
    revision.recorded_at_utc AS recorded_at_utc,
    revision.claims_json AS claims_json,
    revision.revision_number AS revision_number
FROM prospective_session_phase_revision_v1 AS revision
WHERE revision.revision_number = (
    SELECT MAX(candidate.revision_number)
    FROM prospective_session_phase_revision_v1 AS candidate
    WHERE candidate.run_id = revision.run_id
      AND candidate.session_date = revision.session_date
)
UNION ALL
SELECT
    base.id,
    base.envelope_id,
    base.run_id,
    base.session_date,
    base.valid_session_ordinal,
    base.phase,
    base.valid,
    base.source_transfer_decision,
    base.strategy_rule_changes_allowed,
    base.recorded_at_utc,
    base.claims_json,
    0 AS revision_number
FROM prospective_session_phase_v0 AS base
WHERE NOT EXISTS (
    SELECT 1
    FROM prospective_session_phase_revision_v1 AS revision
    WHERE revision.run_id = base.run_id
      AND revision.session_date = base.session_date
);

CREATE TABLE IF NOT EXISTS opening_reversal_transfer_session_revision_v1 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    session_date TEXT NOT NULL,
    supersedes_opening_transfer_session_id INTEGER NOT NULL
        REFERENCES opening_reversal_transfer_session_v1(id),
    revision_number INTEGER NOT NULL CHECK (revision_number >= 1),
    valid INTEGER NOT NULL CHECK (valid IN (0, 1)),
    valid_session_ordinal INTEGER,
    decision TEXT NOT NULL,
    ibkr_opening_return REAL,
    eodhd_opening_return REAL,
    ibkr_opening_range REAL,
    eodhd_opening_range REAL,
    severe_state_agreement INTEGER CHECK (severe_state_agreement IN (0, 1)),
    sign_agreement INTEGER CHECK (sign_agreement IN (0, 1)),
    timestamp_alignment INTEGER CHECK (timestamp_alignment IN (0, 1)),
    checkpoint_6_episode_identity_agreement INTEGER
        CHECK (checkpoint_6_episode_identity_agreement IN (0, 1)),
    operational_checks_pass INTEGER NOT NULL
        CHECK (operational_checks_pass IN (0, 1)),
    operational_evidence_json TEXT NOT NULL,
    outcome_fields_accessed INTEGER NOT NULL CHECK (outcome_fields_accessed = 0),
    report_json TEXT NOT NULL,
    report_hash_v1 TEXT NOT NULL UNIQUE,
    revision_reason TEXT NOT NULL,
    revision_provenance_json TEXT NOT NULL,
    UNIQUE(run_id, session_date, revision_number)
);

CREATE VIEW IF NOT EXISTS opening_reversal_transfer_session_effective_v1 AS
SELECT
    revision.id AS id,
    revision.envelope_id AS envelope_id,
    revision.run_id AS run_id,
    revision.session_date AS session_date,
    revision.valid AS valid,
    revision.valid_session_ordinal AS valid_session_ordinal,
    revision.decision AS decision,
    revision.ibkr_opening_return AS ibkr_opening_return,
    revision.eodhd_opening_return AS eodhd_opening_return,
    revision.ibkr_opening_range AS ibkr_opening_range,
    revision.eodhd_opening_range AS eodhd_opening_range,
    revision.severe_state_agreement AS severe_state_agreement,
    revision.sign_agreement AS sign_agreement,
    revision.timestamp_alignment AS timestamp_alignment,
    revision.checkpoint_6_episode_identity_agreement
        AS checkpoint_6_episode_identity_agreement,
    revision.operational_checks_pass AS operational_checks_pass,
    revision.operational_evidence_json AS operational_evidence_json,
    revision.outcome_fields_accessed AS outcome_fields_accessed,
    revision.report_json AS report_json,
    revision.report_hash_v1 AS report_hash_v1,
    revision.revision_number AS revision_number
FROM opening_reversal_transfer_session_revision_v1 AS revision
WHERE revision.revision_number = (
    SELECT MAX(candidate.revision_number)
    FROM opening_reversal_transfer_session_revision_v1 AS candidate
    WHERE candidate.run_id = revision.run_id
      AND candidate.session_date = revision.session_date
)
UNION ALL
SELECT
    base.id,
    base.envelope_id,
    base.run_id,
    base.session_date,
    base.valid,
    base.valid_session_ordinal,
    base.decision,
    base.ibkr_opening_return,
    base.eodhd_opening_return,
    base.ibkr_opening_range,
    base.eodhd_opening_range,
    base.severe_state_agreement,
    base.sign_agreement,
    base.timestamp_alignment,
    base.checkpoint_6_episode_identity_agreement,
    base.operational_checks_pass,
    base.operational_evidence_json,
    base.outcome_fields_accessed,
    base.report_json,
    base.report_hash_v1,
    0 AS revision_number
FROM opening_reversal_transfer_session_v1 AS base
WHERE NOT EXISTS (
    SELECT 1
    FROM opening_reversal_transfer_session_revision_v1 AS revision
    WHERE revision.run_id = base.run_id
      AND revision.session_date = base.session_date
);
