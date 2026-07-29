PRAGMA foreign_keys = ON;

ALTER TABLE m1c_checkpoint_v0
    ADD COLUMN bar_identity TEXT NOT NULL DEFAULT '';
ALTER TABLE m1c_checkpoint_v0
    ADD COLUMN configuration_hash TEXT NOT NULL DEFAULT '';

ALTER TABLE quiet_state_shadow_outcome_v0
    ADD COLUMN cohort_phase TEXT NOT NULL DEFAULT 'engineering_transfer';
ALTER TABLE quiet_state_shadow_outcome_v0
    ADD COLUMN scientific_option_evidence INTEGER NOT NULL DEFAULT 0
        CHECK (scientific_option_evidence IN (0, 1));
ALTER TABLE shadow_quote_outcome_v0
    ADD COLUMN cohort_phase TEXT NOT NULL DEFAULT 'engineering_transfer';
ALTER TABLE shadow_quote_outcome_v0
    ADD COLUMN scientific_option_evidence INTEGER NOT NULL DEFAULT 0
        CHECK (scientific_option_evidence IN (0, 1));
ALTER TABLE shadow_structure_outcome_v0
    ADD COLUMN cohort_phase TEXT NOT NULL DEFAULT 'engineering_transfer';
ALTER TABLE shadow_structure_outcome_v0
    ADD COLUMN scientific_option_evidence INTEGER NOT NULL DEFAULT 0
        CHECK (scientific_option_evidence IN (0, 1));

CREATE TABLE IF NOT EXISTS ibkr_runtime_capacity_v0 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    observed_at_utc TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    claims_json TEXT NOT NULL,
    UNIQUE(run_id, observed_at_utc)
);

CREATE TABLE IF NOT EXISTS subscription_lifecycle_event_v0 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    occurred_at_utc TEXT NOT NULL,
    subscription_key TEXT NOT NULL,
    request_id INTEGER NOT NULL,
    subscription_kind TEXT NOT NULL,
    subscription_class INTEGER NOT NULL CHECK (subscription_class BETWEEN 0 AND 5),
    symbol TEXT NOT NULL,
    con_id INTEGER NOT NULL,
    status TEXT NOT NULL,
    owner_ids_json TEXT NOT NULL,
    owner_count INTEGER NOT NULL,
    generation INTEGER NOT NULL,
    reason TEXT,
    payload_json TEXT NOT NULL,
    claims_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS option_episode_allocation_v0 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    episode_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    episode_kind TEXT NOT NULL CHECK (
        episode_kind IN ('quiet', 'high_tail', 'neutral_control')
    ),
    state TEXT NOT NULL CHECK (
        state IN (
            'IDLE',
            'UNIVERSE_MONITORING',
            'EPISODE_QUEUED',
            'CONTRACT_DISCOVERY',
            'PRIMARY_LEGS_STREAMING',
            'COMPARISON_LEGS_STREAMING',
            'HORIZON_FINALISING',
            'CANCELLING_SUBSCRIPTIONS',
            'COMPLETE',
            'DEGRADED',
            'FAILED'
        )
    ),
    requested_subscriptions_json TEXT NOT NULL,
    approved_subscriptions_json TEXT NOT NULL,
    queued_subscriptions_json TEXT NOT NULL,
    denied_subscriptions_json TEXT NOT NULL,
    degradation_reason TEXT,
    capacity_before_json TEXT NOT NULL,
    capacity_after_json TEXT NOT NULL,
    cohort_phase TEXT NOT NULL CHECK (
        cohort_phase IN (
            'engineering_transfer',
            'option_development',
            'untouched_confirmation'
        )
    ),
    scientific_option_evidence INTEGER NOT NULL
        CHECK (
            scientific_option_evidence IN (0, 1)
            AND NOT (
                cohort_phase = 'engineering_transfer'
                AND scientific_option_evidence = 1
            )
        ),
    updated_at_utc TEXT NOT NULL,
    claims_json TEXT NOT NULL,
    UNIQUE(run_id, episode_id, state, updated_at_utc)
);

CREATE TABLE IF NOT EXISTS skipped_recording_v0 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    session_date TEXT NOT NULL,
    episode_id TEXT,
    symbol TEXT,
    recording_kind TEXT NOT NULL,
    reason TEXT NOT NULL,
    requested_payload_json TEXT NOT NULL,
    occurred_at_utc TEXT NOT NULL,
    cohort_phase TEXT NOT NULL,
    scientific_option_evidence INTEGER NOT NULL
        CHECK (scientific_option_evidence IN (0, 1)),
    claims_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS provider_m1c_observation_v0 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    provider TEXT NOT NULL CHECK (provider IN ('ibkr', 'eodhd')),
    symbol TEXT NOT NULL,
    session_date TEXT NOT NULL,
    checkpoint INTEGER NOT NULL,
    bar_identity TEXT NOT NULL,
    bar_start_utc TEXT NOT NULL,
    bar_end_utc TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    feature_values_json TEXT NOT NULL,
    probability REAL NOT NULL,
    quiet_episode INTEGER NOT NULL CHECK (quiet_episode IN (0, 1)),
    high_tail_episode INTEGER NOT NULL CHECK (high_tail_episode IN (0, 1)),
    data_quality_status TEXT NOT NULL,
    model_hash TEXT NOT NULL,
    configuration_hash TEXT NOT NULL,
    claims_json TEXT NOT NULL,
    UNIQUE(run_id, provider, symbol, session_date, checkpoint)
);

CREATE TABLE IF NOT EXISTS source_transfer_session_v0 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    session_date TEXT NOT NULL,
    valid INTEGER NOT NULL CHECK (valid IN (0, 1)),
    decision TEXT NOT NULL,
    report_json TEXT NOT NULL,
    generated_at_utc TEXT NOT NULL,
    claims_json TEXT NOT NULL,
    UNIQUE(run_id, session_date)
);

CREATE TABLE IF NOT EXISTS prospective_session_phase_v0 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    session_date TEXT NOT NULL,
    valid_session_ordinal INTEGER,
    phase TEXT NOT NULL CHECK (
        phase IN ('engineering_transfer', 'option_development', 'untouched_confirmation')
    ),
    valid INTEGER NOT NULL CHECK (valid IN (0, 1)),
    source_transfer_decision TEXT,
    strategy_rule_changes_allowed INTEGER NOT NULL
        CHECK (strategy_rule_changes_allowed = 0),
    recorded_at_utc TEXT NOT NULL,
    claims_json TEXT NOT NULL,
    UNIQUE(run_id, session_date)
);

CREATE INDEX IF NOT EXISTS idx_subscription_lifecycle_event_v0
    ON subscription_lifecycle_event_v0(run_id, occurred_at_utc, subscription_class);
CREATE INDEX IF NOT EXISTS idx_option_episode_allocation_v0
    ON option_episode_allocation_v0(run_id, state, updated_at_utc);
CREATE INDEX IF NOT EXISTS idx_provider_m1c_observation_v0
    ON provider_m1c_observation_v0(run_id, session_date, provider, symbol, checkpoint);
CREATE INDEX IF NOT EXISTS idx_skipped_recording_v0
    ON skipped_recording_v0(run_id, session_date, recording_kind);
