PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS callback_inbox_v1 (
    source_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    inbox_event_id TEXT NOT NULL UNIQUE,
    callback_kind TEXT NOT NULL,
    request_id INTEGER NOT NULL,
    received_utc TEXT NOT NULL,
    received_monotonic_ns INTEGER,
    provider_timestamp_utc TEXT,
    original_payload_json TEXT NOT NULL,
    admission_run_id TEXT,
    admission_recorder_generation INTEGER
        CHECK (
            admission_recorder_generation IS NULL
            OR admission_recorder_generation > 0
        ),
    connection_generation INTEGER NOT NULL CHECK (connection_generation >= 0),
    subscription_owner TEXT,
    symbol TEXT,
    callback_classification TEXT NOT NULL
        CHECK (
            callback_classification IN (
                'accepted_active_callback',
                'expected_late_callback_after_cancellation',
                'callback_from_previous_connection_generation',
                'duplicate_callback',
                'unknown_callback',
                'callback_after_data_loss_latch',
                'control_callback'
            )
        ),
    provider_envelope_event_id TEXT
        REFERENCES callback_inbox_v1(inbox_event_id),
    lease_owner TEXT,
    lease_generation INTEGER NOT NULL DEFAULT 0 CHECK (lease_generation >= 0),
    lease_batch_id TEXT,
    lease_timestamp_utc TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    status TEXT NOT NULL
        CHECK (
            status IN (
                'provider_pending',
                'pending',
                'leased',
                'acknowledged',
                'quarantined',
                'diagnostic'
            )
        ),
    acknowledgement_timestamp_utc TEXT,
    failure_classification TEXT,
    associated_raw_partition_hashes_json TEXT NOT NULL DEFAULT '[]',
    admitted_at_utc TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_callback_inbox_v1_available
    ON callback_inbox_v1(status, source_sequence);
CREATE INDEX IF NOT EXISTS idx_callback_inbox_v1_lease
    ON callback_inbox_v1(status, lease_timestamp_utc, source_sequence);
CREATE INDEX IF NOT EXISTS idx_callback_inbox_v1_request
    ON callback_inbox_v1(request_id, connection_generation, source_sequence);
CREATE INDEX IF NOT EXISTS idx_callback_inbox_v1_run_status
    ON callback_inbox_v1(admission_run_id, status, source_sequence);
CREATE INDEX IF NOT EXISTS idx_callback_inbox_v1_batch
    ON callback_inbox_v1(admission_run_id, lease_batch_id, source_sequence);
CREATE INDEX IF NOT EXISTS idx_callback_inbox_v1_provider_envelope
    ON callback_inbox_v1(provider_envelope_event_id);

CREATE TABLE IF NOT EXISTS callback_raw_materialization_v1 (
    inbox_event_id TEXT PRIMARY KEY
        REFERENCES callback_inbox_v1(inbox_event_id) ON DELETE CASCADE,
    source_sequence INTEGER NOT NULL UNIQUE,
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    recorder_generation INTEGER NOT NULL CHECK (recorder_generation > 0),
    lease_batch_id TEXT NOT NULL,
    raw_partition_hashes_json TEXT NOT NULL,
    raw_event_ids_json TEXT NOT NULL,
    materialized_at_utc TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_callback_raw_materialization_v1_batch
    ON callback_raw_materialization_v1(run_id, lease_batch_id, source_sequence);

CREATE TABLE IF NOT EXISTS callback_processing_commit_v1 (
    inbox_event_id TEXT PRIMARY KEY
        REFERENCES callback_inbox_v1(inbox_event_id) ON DELETE CASCADE,
    source_sequence INTEGER NOT NULL UNIQUE,
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    recorder_generation INTEGER NOT NULL CHECK (recorder_generation > 0),
    raw_partition_hashes_json TEXT NOT NULL,
    committed_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS callback_request_tombstone_v1 (
    request_id INTEGER NOT NULL,
    connection_generation INTEGER NOT NULL CHECK (connection_generation >= 0),
    subscription_owner TEXT,
    symbol TEXT,
    cancellation_reason TEXT NOT NULL,
    cancelled_at_utc TEXT NOT NULL,
    expires_at_utc TEXT NOT NULL,
    PRIMARY KEY(request_id, connection_generation)
);

CREATE INDEX IF NOT EXISTS idx_callback_request_tombstone_v1_expiry
    ON callback_request_tombstone_v1(expires_at_utc);

CREATE TABLE IF NOT EXISTS recorder_generation_v1 (
    -- Startup evidence must be writable even when artifact verification
    -- fails before the scientific run identity can be created.
    run_id TEXT NOT NULL,
    recorder_generation INTEGER NOT NULL CHECK (recorder_generation > 0),
    owner_id TEXT NOT NULL,
    started_at_utc TEXT NOT NULL,
    stopping_at_utc TEXT,
    stopped_at_utc TEXT,
    termination_reason TEXT,
    stopped_cleanly INTEGER CHECK (stopped_cleanly IN (0, 1)),
    PRIMARY KEY(run_id, recorder_generation),
    UNIQUE(run_id, owner_id, recorder_generation)
);

CREATE TABLE IF NOT EXISTS recorder_operational_state_v1 (
    run_id TEXT PRIMARY KEY,
    recorder_generation INTEGER NOT NULL CHECK (recorder_generation > 0),
    owner_id TEXT NOT NULL,
    state TEXT NOT NULL
        CHECK (
            state IN (
                'INACTIVE',
                'STARTING',
                'WAITING_FOR_PROSPECTIVE_START',
                'MARKET_CLOSED',
                'RECORDING_HEALTHY',
                'RECORDING_DEGRADED',
                'RECONNECTING',
                'STALE_HEARTBEAT',
                'INGESTION_FATAL',
                'STORAGE_FATAL',
                'SCIENTIFICALLY_BLOCKED',
                'STOPPING',
                'STOPPED_CLEANLY'
            )
        ),
    state_reason_code TEXT,
    process_heartbeat_at_utc TEXT,
    latest_callback_received_at_utc TEXT,
    latest_callback_durably_admitted_at_utc TEXT,
    latest_raw_partition_committed_at_utc TEXT,
    latest_inbox_acknowledgement_at_utc TEXT,
    latest_completed_five_minute_bar_at_utc TEXT,
    latest_successful_checkpoint_at_utc TEXT,
    inbox_backlog INTEGER NOT NULL DEFAULT 0 CHECK (inbox_backlog >= 0),
    oldest_unacknowledged_at_utc TEXT,
    market_session_open INTEGER NOT NULL DEFAULT 0 CHECK (market_session_open IN (0, 1)),
    callbacks_expected INTEGER NOT NULL DEFAULT 0 CHECK (callbacks_expected IN (0, 1)),
    ibkr_connection_state TEXT,
    required_market_data_mode TEXT,
    observed_market_data_mode TEXT,
    scientific_prerequisites_valid INTEGER NOT NULL DEFAULT 0
        CHECK (scientific_prerequisites_valid IN (0, 1)),
    expected_artifact_count INTEGER NOT NULL DEFAULT 0
        CHECK (expected_artifact_count >= 0),
    frozen_artifacts_verified INTEGER NOT NULL DEFAULT 0
        CHECK (frozen_artifacts_verified IN (0, 1)),
    unresolved_required_gap_count INTEGER NOT NULL DEFAULT 0
        CHECK (unresolved_required_gap_count >= 0),
    fatal_ingestion_code TEXT,
    fatal_storage_code TEXT,
    scientific_recording_valid INTEGER NOT NULL DEFAULT 0
        CHECK (scientific_recording_valid IN (0, 1)),
    broker_state_mutation_count INTEGER NOT NULL DEFAULT 0
        CHECK (broker_state_mutation_count >= 0),
    updated_at_utc TEXT NOT NULL,
    FOREIGN KEY(run_id, recorder_generation)
        REFERENCES recorder_generation_v1(run_id, recorder_generation)
);

CREATE TABLE IF NOT EXISTS recorder_fatal_latch_v1 (
    latch_id TEXT PRIMARY KEY,
    run_id TEXT,
    recorder_generation INTEGER,
    connection_generation INTEGER,
    latch_kind TEXT NOT NULL CHECK (latch_kind IN ('ingestion', 'storage')),
    stable_error_code TEXT NOT NULL,
    first_possibly_lost_source_sequence INTEGER,
    callback_kind TEXT,
    request_id INTEGER,
    error_class TEXT NOT NULL,
    evidence_loss_possible INTEGER NOT NULL CHECK (evidence_loss_possible IN (0, 1)),
    latched_at_utc TEXT NOT NULL,
    resolved_at_utc TEXT,
    resolution_evidence TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_recorder_fatal_latch_v1_active
    ON recorder_fatal_latch_v1(run_id, latch_kind)
    WHERE resolved_at_utc IS NULL;

CREATE TABLE IF NOT EXISTS operational_incident_v1 (
    incident_id TEXT PRIMARY KEY,
    run_id TEXT,
    recorder_generation INTEGER,
    connection_generation INTEGER,
    occurred_at_utc TEXT NOT NULL,
    component TEXT NOT NULL,
    severity TEXT NOT NULL,
    stable_error_code TEXT NOT NULL,
    callback_kind TEXT,
    request_id INTEGER,
    source_sequence INTEGER,
    subscription_owner TEXT,
    symbol TEXT,
    error_class TEXT NOT NULL,
    evidence_loss_possible INTEGER NOT NULL CHECK (evidence_loss_possible IN (0, 1)),
    details_json TEXT NOT NULL,
    resolved_at_utc TEXT,
    resolution_evidence TEXT
);

CREATE INDEX IF NOT EXISTS idx_operational_incident_v1_run
    ON operational_incident_v1(run_id, occurred_at_utc, incident_id);
CREATE INDEX IF NOT EXISTS idx_operational_incident_v1_active
    ON operational_incident_v1(run_id, resolved_at_utc, severity);

CREATE TABLE IF NOT EXISTS gap_incident_v1 (
    gap_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    recorder_generation INTEGER NOT NULL CHECK (recorder_generation > 0),
    symbol TEXT NOT NULL,
    stream_kind TEXT NOT NULL,
    request_id INTEGER,
    connection_generation INTEGER NOT NULL CHECK (connection_generation >= 0),
    start_timestamp_utc TEXT NOT NULL,
    end_timestamp_utc TEXT,
    detection_timestamp_utc TEXT NOT NULL,
    cause_code TEXT NOT NULL,
    severity TEXT NOT NULL,
    recoverability TEXT NOT NULL,
    backfill_attempted INTEGER NOT NULL DEFAULT 0 CHECK (backfill_attempted IN (0, 1)),
    backfill_result TEXT,
    affected_first_source_sequence INTEGER,
    affected_last_source_sequence INTEGER,
    affected_episode_ids_json TEXT NOT NULL DEFAULT '[]',
    resolution_timestamp_utc TEXT,
    resolution_evidence TEXT
);

CREATE INDEX IF NOT EXISTS idx_gap_incident_v1_active
    ON gap_incident_v1(run_id, resolution_timestamp_utc, severity);
CREATE INDEX IF NOT EXISTS idx_gap_incident_v1_session
    ON gap_incident_v1(run_id, start_timestamp_utc, end_timestamp_utc);

CREATE TABLE IF NOT EXISTS runtime_artifact_verification_v1 (
    verification_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    recorder_generation INTEGER NOT NULL CHECK (recorder_generation > 0),
    artifact_bundle_id TEXT NOT NULL,
    artifact_name TEXT NOT NULL,
    expected_hash TEXT NOT NULL,
    observed_hash TEXT,
    feature_contract_version TEXT NOT NULL,
    activation_receipt_identity TEXT NOT NULL,
    found INTEGER NOT NULL CHECK (found IN (0, 1)),
    loaded INTEGER NOT NULL CHECK (loaded IN (0, 1)),
    schema_validated INTEGER NOT NULL CHECK (schema_validated IN (0, 1)),
    hash_verified INTEGER NOT NULL CHECK (hash_verified IN (0, 1)),
    contract_compatible INTEGER NOT NULL CHECK (contract_compatible IN (0, 1)),
    used_by_active_generation INTEGER NOT NULL CHECK (used_by_active_generation IN (0, 1)),
    load_timestamp_utc TEXT NOT NULL,
    verification_result TEXT NOT NULL CHECK (verification_result IN ('verified', 'blocked')),
    blocker TEXT,
    details_json TEXT NOT NULL,
    UNIQUE(run_id, recorder_generation, artifact_name)
);

CREATE INDEX IF NOT EXISTS idx_runtime_artifact_verification_v1_active
    ON runtime_artifact_verification_v1(
        run_id,
        recorder_generation,
        verification_result
    );

CREATE VIEW IF NOT EXISTS opening_reversal_v1_1_eligible_episode AS
SELECT
    prediction.run_id,
    prediction.fresh_episode_id AS episode_id,
    prediction.receipt_hash_v1 AS prediction_receipt_hash_v1,
    prediction.session_date,
    prediction.stock,
    prediction.entry_timestamp_utc,
    activation.activation_receipt_hash AS activation_receipt_identity,
    barrier.audit_hash_v1_1 AS causal_barrier_audit_identity,
    promotion.promotion_hash_v1 AS promotion_identity
FROM opening_reversal_prediction_v1 AS prediction
JOIN opening_reversal_activation_v1 AS activation
  ON activation.run_id = prediction.run_id
 AND activation.experiment_id = prediction.experiment_id
 AND activation.experiment_version = prediction.experiment_version
JOIN opening_reversal_causal_barrier_audit_v1_1 AS barrier
  ON barrier.run_id = prediction.run_id
 AND barrier.session_date = prediction.session_date
 AND barrier.experiment_id = prediction.experiment_id
 AND barrier.experiment_version = prediction.experiment_version
 AND barrier.activation_receipt_hash_v1_1 =
     activation.activation_receipt_hash
 AND barrier.barrier_status = 'passed'
JOIN opening_reversal_promotion_v1 AS promotion
  ON promotion.run_id = prediction.run_id
 AND promotion.promoted_receipt_hash_v1 = prediction.receipt_hash_v1
JOIN m1c_episode_v0 AS episode
  ON episode.run_id = prediction.run_id
 AND episode.episode_id = prediction.fresh_episode_id
 AND episode.symbol = prediction.stock
 AND episode.session_date = prediction.session_date
 AND episode.prospective_entry_timestamp_utc =
     prediction.entry_timestamp_utc
WHERE prediction.experiment_id = 'm1c-prospective-opening-reversal-v1'
  AND prediction.experiment_version = '1.1'
  AND prediction.eligibility_v1 = 1
  AND prediction.scientific_outcome_eligible_v1 = 1
  AND prediction.fresh_episode_id IS NOT NULL
  AND activation.activation_timestamp_utc < prediction.entry_timestamp_utc
  AND activation.new_york_trading_date <= prediction.session_date
  AND barrier.nominal_entry_timestamp_utc =
      prediction.entry_timestamp_utc
  AND prediction.receipt_created_at_utc <=
      barrier.release_authorized_at_utc
  AND EXISTS (
      SELECT 1
      FROM json_each(barrier.prediction_receipt_hashes_json) AS receipt
      WHERE receipt.value = prediction.receipt_hash_v1
  )
  AND NOT EXISTS (
      SELECT 1
      FROM quiet_state_observation_v0 AS quiet
      WHERE quiet.run_id = prediction.run_id
        AND quiet.observation_id = prediction.fresh_episode_id
  );

CREATE TRIGGER IF NOT EXISTS trg_opening_reversal_v1_1_discovery_guard
BEFORE INSERT ON opening_reversal_contract_discovery_v1
WHEN EXISTS (
    SELECT 1
    FROM opening_reversal_prediction_v1 AS prediction
    WHERE prediction.run_id = NEW.run_id
      AND prediction.fresh_episode_id = NEW.episode_id
      AND prediction.experiment_version = '1.1'
)
BEGIN
    SELECT CASE
        WHEN NEW.status = 'selected'
          AND (
              NEW.call_con_id IS NULL
              OR NEW.put_con_id IS NULL
              OR NEW.call_con_id = NEW.put_con_id
              OR NEW.expiry IS NULL
              OR NEW.strike IS NULL
              OR NEW.planned_live_market_data_lines <> 2
              OR NEW.live_market_data_lines_consumed <> 0
              OR NEW.full_chain_live_subscription_created <> 0
              OR NEW.metadata_request_ended <> 1
              OR NEW.missing_reason IS NOT NULL
          )
        THEN RAISE(ABORT, 'blocked_v1_1_option_protocol_not_exactly_two_lines')
    END;
    SELECT CASE
        WHEN NEW.status = 'failed'
          AND (
              NEW.call_con_id IS NOT NULL
              OR NEW.put_con_id IS NOT NULL
              OR NEW.expiry IS NOT NULL
              OR NEW.strike IS NOT NULL
              OR NEW.planned_live_market_data_lines <> 0
              OR NEW.live_market_data_lines_consumed <> 0
              OR NEW.full_chain_live_subscription_created <> 0
              OR NEW.metadata_request_ended <> 1
              OR NEW.missing_reason IS NULL
          )
        THEN RAISE(ABORT, 'blocked_v1_1_failed_discovery_contains_contract_allocation')
    END;
    SELECT CASE
        WHEN NEW.status NOT IN ('selected', 'failed')
        THEN RAISE(ABORT, 'blocked_v1_1_discovery_status_invalid')
    END;
    SELECT CASE
        WHEN NOT EXISTS (
            SELECT 1
            FROM opening_reversal_v1_1_eligible_episode AS eligible
            WHERE eligible.run_id = NEW.run_id
              AND eligible.episode_id = NEW.episode_id
        )
        THEN RAISE(ABORT, 'blocked_v1_1_episode_not_eligible')
    END;
    SELECT CASE
        WHEN NEW.status = 'selected'
          AND NEW.expiry <> (
            SELECT date(eligible.session_date, '+1 day')
            FROM opening_reversal_v1_1_eligible_episode AS eligible
            WHERE eligible.run_id = NEW.run_id
              AND eligible.episode_id = NEW.episode_id
        )
        THEN RAISE(ABORT, 'blocked_v1_1_primary_expiry_not_1dte')
    END;
END;

CREATE TRIGGER IF NOT EXISTS trg_opening_reversal_discovery_identity_guard
BEFORE INSERT ON opening_reversal_contract_discovery_v1
WHEN NOT EXISTS (
    SELECT 1
    FROM opening_reversal_prediction_v1 AS prediction
    WHERE prediction.run_id = NEW.run_id
      AND prediction.fresh_episode_id = NEW.episode_id
      AND prediction.experiment_id = 'm1c-prospective-opening-reversal-v1'
)
BEGIN
    SELECT RAISE(ABORT, 'blocked_opening_reversal_episode_identity_missing');
END;

CREATE TRIGGER IF NOT EXISTS trg_opening_reversal_v1_1_outcome_guard
BEFORE INSERT ON opening_reversal_primary_option_outcome_v1
WHEN EXISTS (
    SELECT 1
    FROM opening_reversal_prediction_v1 AS prediction
    WHERE prediction.run_id = NEW.run_id
      AND prediction.receipt_hash_v1 = NEW.prediction_receipt_hash_v1
      AND prediction.experiment_version = '1.1'
)
BEGIN
    SELECT CASE
        WHEN NOT EXISTS (
            SELECT 1
            FROM opening_reversal_v1_1_eligible_episode AS eligible
            WHERE eligible.run_id = NEW.run_id
              AND eligible.prediction_receipt_hash_v1 = NEW.prediction_receipt_hash_v1
        )
        THEN RAISE(ABORT, 'blocked_v1_1_episode_not_eligible')
    END;
    SELECT CASE
        WHEN NOT EXISTS (
            SELECT 1
            FROM opening_reversal_contract_discovery_v1 AS discovery
            JOIN opening_reversal_v1_1_eligible_episode AS eligible
              ON eligible.run_id = discovery.run_id
             AND eligible.episode_id = discovery.episode_id
            WHERE eligible.prediction_receipt_hash_v1 =
                  NEW.prediction_receipt_hash_v1
              AND discovery.status = 'selected'
              AND discovery.planned_live_market_data_lines = 2
              AND discovery.expiry = NEW.expiry
              AND discovery.strike = NEW.strike
              AND (
                  (NEW.right = 'C' AND discovery.call_con_id = NEW.con_id)
                  OR
                  (NEW.right = 'P' AND discovery.put_con_id = NEW.con_id)
              )
        )
        THEN RAISE(ABORT, 'blocked_v1_1_outcome_contract_not_selected_pair')
    END;
    SELECT CASE
        WHEN (
            SELECT COUNT(*)
            FROM opening_reversal_primary_option_outcome_v1 AS existing
            WHERE existing.run_id = NEW.run_id
              AND existing.prediction_receipt_hash_v1 =
                  NEW.prediction_receipt_hash_v1
        ) >= 2
        THEN RAISE(ABORT, 'blocked_v1_1_outcome_more_than_two_legs')
    END;
    SELECT CASE
        WHEN EXISTS (
            SELECT 1
            FROM opening_reversal_primary_option_outcome_v1 AS existing
            WHERE existing.run_id = NEW.run_id
              AND existing.prediction_receipt_hash_v1 =
                  NEW.prediction_receipt_hash_v1
              AND existing.right = NEW.right
        )
        THEN RAISE(ABORT, 'blocked_v1_1_outcome_duplicate_option_right')
    END;
    SELECT CASE
        WHEN (
            NEW.role = 'predicted_leg'
            AND NEW.right <> (
                SELECT CASE prediction.prediction_v1
                    WHEN 'CALL' THEN 'C'
                    WHEN 'PUT' THEN 'P'
                END
                FROM opening_reversal_prediction_v1 AS prediction
                WHERE prediction.run_id = NEW.run_id
                  AND prediction.receipt_hash_v1 =
                      NEW.prediction_receipt_hash_v1
            )
        )
        OR (
            NEW.role = 'opposite_leg'
            AND NEW.right = (
                SELECT CASE prediction.prediction_v1
                    WHEN 'CALL' THEN 'C'
                    WHEN 'PUT' THEN 'P'
                END
                FROM opening_reversal_prediction_v1 AS prediction
                WHERE prediction.run_id = NEW.run_id
                  AND prediction.receipt_hash_v1 =
                      NEW.prediction_receipt_hash_v1
            )
        )
        THEN RAISE(ABORT, 'blocked_v1_1_outcome_role_right_mismatch')
    END;
END;

CREATE TRIGGER IF NOT EXISTS trg_opening_reversal_v1_1_discovery_immutable
BEFORE UPDATE ON opening_reversal_contract_discovery_v1
WHEN EXISTS (
    SELECT 1
    FROM opening_reversal_prediction_v1 AS prediction
    WHERE prediction.run_id = OLD.run_id
      AND prediction.fresh_episode_id = OLD.episode_id
      AND prediction.experiment_id = 'm1c-prospective-opening-reversal-v1'
      AND prediction.experiment_version = '1.1'
)
BEGIN
    SELECT RAISE(ABORT, 'blocked_v1_1_contract_discovery_is_immutable');
END;

CREATE TRIGGER IF NOT EXISTS trg_opening_reversal_v1_1_outcome_immutable
BEFORE UPDATE ON opening_reversal_primary_option_outcome_v1
WHEN EXISTS (
    SELECT 1
    FROM opening_reversal_prediction_v1 AS prediction
    WHERE prediction.run_id = OLD.run_id
      AND prediction.receipt_hash_v1 = OLD.prediction_receipt_hash_v1
      AND prediction.experiment_id = 'm1c-prospective-opening-reversal-v1'
      AND prediction.experiment_version = '1.1'
)
BEGIN
    SELECT RAISE(ABORT, 'blocked_v1_1_primary_option_outcome_is_immutable');
END;
