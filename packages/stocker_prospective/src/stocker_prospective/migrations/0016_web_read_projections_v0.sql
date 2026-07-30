CREATE INDEX IF NOT EXISTS idx_raw_partition_manifest_web_window
ON raw_partition_manifest_v0(
    run_id,
    symbol,
    event_type,
    maximum_timestamp_utc,
    minimum_timestamp_utc
);

CREATE INDEX IF NOT EXISTS idx_m1c_episode_web_latest
ON m1c_episode_v0(run_id, symbol, trigger_bar_end_utc DESC);

CREATE TABLE IF NOT EXISTS web_audit_projection_v0 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    audit_type TEXT NOT NULL,
    identity TEXT NOT NULL,
    recorded_at_utc TEXT NOT NULL,
    details TEXT,
    UNIQUE(run_id, audit_type, identity, recorded_at_utc)
);

CREATE INDEX IF NOT EXISTS idx_web_audit_projection_page
ON web_audit_projection_v0(run_id, recorded_at_utc DESC, id DESC);

INSERT OR IGNORE INTO web_audit_projection_v0(
    run_id,
    audit_type,
    identity,
    recorded_at_utc,
    details
)
SELECT run_id, audit_type, identity, recorded_at_utc, details
FROM (
    SELECT run_id,
           'raw_partition' AS audit_type,
           content_hash AS identity,
           maximum_timestamp_utc AS recorded_at_utc,
           json_object(
               'event_type', event_type,
               'symbol', symbol,
               'row_count', row_count,
               'complete', complete,
               'gap_count', gap_count
           ) AS details
    FROM raw_partition_manifest_v0
    UNION ALL
    SELECT run_id,
           'subscription' AS audit_type,
           subscription_key AS identity,
           started_at_utc AS recorded_at_utc,
           json_object(
               'subscription_kind', subscription_kind,
               'symbol', symbol,
               'cancelled_at_utc', cancelled_at_utc,
               'cancellation_reason', cancellation_reason,
               'capacity_denied', capacity_denied
           ) AS details
    FROM subscription_lifecycle_v0
    UNION ALL
    SELECT run_id,
           'm1c_prediction' AS audit_type,
           feature_hash AS identity,
           bar_end_utc AS recorded_at_utc,
           json_object(
               'symbol', symbol,
               'checkpoint', checkpoint,
               'model_hash', model_hash,
               'probability', probability,
               'threshold', threshold,
               'threshold_passed', threshold_passed,
               'eligible', eligible
           ) AS details
    FROM m1c_checkpoint_v0
    UNION ALL
    SELECT run_id,
           'episode_decision' AS audit_type,
           episode_id AS identity,
           trigger_bar_end_utc AS recorded_at_utc,
           json_object(
               'symbol', symbol,
               'checkpoint', trigger_checkpoint,
               'entry', prospective_entry_timestamp_utc,
               'probability', m1c_probability,
               'scientific_recording_valid', scientific_recording_valid,
               'completion_status', completion_status
           ) AS details
    FROM m1c_episode_v0
    UNION ALL
    SELECT run_id,
           'shadow_quote_selection' AS audit_type,
           episode_id || ':' || archetype || ':' ||
               contract_identity || ':' || horizon_minutes AS identity,
           target_timestamp_utc AS recorded_at_utc,
           json_object(
               'episode_id', episode_id,
               'archetype', archetype,
               'direction', direction,
               'horizon_minutes', horizon_minutes,
               'valid', valid
           ) AS details
    FROM shadow_quote_outcome_v0
)
ORDER BY recorded_at_utc, audit_type, identity;

CREATE TRIGGER IF NOT EXISTS web_audit_raw_partition_insert_v0
AFTER INSERT ON raw_partition_manifest_v0
BEGIN
    INSERT OR IGNORE INTO web_audit_projection_v0(
        run_id, audit_type, identity, recorded_at_utc, details
    ) VALUES (
        NEW.run_id,
        'raw_partition',
        NEW.content_hash,
        NEW.maximum_timestamp_utc,
        json_object(
            'event_type', NEW.event_type,
            'symbol', NEW.symbol,
            'row_count', NEW.row_count,
            'complete', NEW.complete,
            'gap_count', NEW.gap_count
        )
    );
END;

CREATE TRIGGER IF NOT EXISTS web_audit_subscription_insert_v0
AFTER INSERT ON subscription_lifecycle_v0
BEGIN
    INSERT OR IGNORE INTO web_audit_projection_v0(
        run_id, audit_type, identity, recorded_at_utc, details
    ) VALUES (
        NEW.run_id,
        'subscription',
        NEW.subscription_key,
        NEW.started_at_utc,
        json_object(
            'subscription_kind', NEW.subscription_kind,
            'symbol', NEW.symbol,
            'cancelled_at_utc', NEW.cancelled_at_utc,
            'cancellation_reason', NEW.cancellation_reason,
            'capacity_denied', NEW.capacity_denied
        )
    );
END;

CREATE TRIGGER IF NOT EXISTS web_audit_m1c_checkpoint_insert_v0
AFTER INSERT ON m1c_checkpoint_v0
BEGIN
    INSERT OR IGNORE INTO web_audit_projection_v0(
        run_id, audit_type, identity, recorded_at_utc, details
    ) VALUES (
        NEW.run_id,
        'm1c_prediction',
        NEW.feature_hash,
        NEW.bar_end_utc,
        json_object(
            'symbol', NEW.symbol,
            'checkpoint', NEW.checkpoint,
            'model_hash', NEW.model_hash,
            'probability', NEW.probability,
            'threshold', NEW.threshold,
            'threshold_passed', NEW.threshold_passed,
            'eligible', NEW.eligible
        )
    );
END;

CREATE TRIGGER IF NOT EXISTS web_audit_m1c_episode_insert_v0
AFTER INSERT ON m1c_episode_v0
BEGIN
    INSERT OR IGNORE INTO web_audit_projection_v0(
        run_id, audit_type, identity, recorded_at_utc, details
    ) VALUES (
        NEW.run_id,
        'episode_decision',
        NEW.episode_id,
        NEW.trigger_bar_end_utc,
        json_object(
            'symbol', NEW.symbol,
            'checkpoint', NEW.trigger_checkpoint,
            'entry', NEW.prospective_entry_timestamp_utc,
            'probability', NEW.m1c_probability,
            'scientific_recording_valid', NEW.scientific_recording_valid,
            'completion_status', NEW.completion_status
        )
    );
END;

CREATE TRIGGER IF NOT EXISTS web_audit_shadow_quote_insert_v0
AFTER INSERT ON shadow_quote_outcome_v0
BEGIN
    INSERT OR IGNORE INTO web_audit_projection_v0(
        run_id, audit_type, identity, recorded_at_utc, details
    ) VALUES (
        NEW.run_id,
        'shadow_quote_selection',
        NEW.episode_id || ':' || NEW.archetype || ':' ||
            NEW.contract_identity || ':' || NEW.horizon_minutes,
        NEW.target_timestamp_utc,
        json_object(
            'episode_id', NEW.episode_id,
            'archetype', NEW.archetype,
            'direction', NEW.direction,
            'horizon_minutes', NEW.horizon_minutes,
            'valid', NEW.valid
        )
    );
END;
