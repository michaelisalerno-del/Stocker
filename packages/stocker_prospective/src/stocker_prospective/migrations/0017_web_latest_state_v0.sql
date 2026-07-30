CREATE TABLE IF NOT EXISTS web_run_event_summary_v0 (
    run_id TEXT PRIMARY KEY REFERENCES prospective_run(run_id),
    last_event_timestamp TEXT,
    data_gaps INTEGER NOT NULL CHECK (data_gaps >= 0),
    manifest_count INTEGER NOT NULL CHECK (manifest_count >= 0)
);

INSERT OR IGNORE INTO web_run_event_summary_v0(
    run_id,
    last_event_timestamp,
    data_gaps,
    manifest_count
)
SELECT run_id,
       MAX(maximum_timestamp_utc),
       COALESCE(SUM(gap_count), 0),
       COUNT(*)
FROM raw_partition_manifest_v0
GROUP BY run_id;

CREATE TRIGGER IF NOT EXISTS web_run_event_summary_insert_v0
AFTER INSERT ON raw_partition_manifest_v0
BEGIN
    INSERT INTO web_run_event_summary_v0(
        run_id,
        last_event_timestamp,
        data_gaps,
        manifest_count
    ) VALUES (
        NEW.run_id,
        NEW.maximum_timestamp_utc,
        NEW.gap_count,
        1
    )
    ON CONFLICT(run_id) DO UPDATE SET
        last_event_timestamp = CASE
            WHEN web_run_event_summary_v0.last_event_timestamp IS NULL
              OR excluded.last_event_timestamp >
                 web_run_event_summary_v0.last_event_timestamp
            THEN excluded.last_event_timestamp
            ELSE web_run_event_summary_v0.last_event_timestamp
        END,
        data_gaps = web_run_event_summary_v0.data_gaps + excluded.data_gaps,
        manifest_count = web_run_event_summary_v0.manifest_count + 1;
END;

CREATE INDEX IF NOT EXISTS idx_data_health_event_web_active
ON data_health_event(run_id, component, id, blocker_code, message);

CREATE TABLE IF NOT EXISTS web_active_runtime_blocker_v0 (
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    component TEXT NOT NULL,
    event_id INTEGER NOT NULL REFERENCES data_health_event(id),
    blocker_code TEXT NOT NULL,
    message TEXT NOT NULL,
    severity TEXT NOT NULL,
    PRIMARY KEY(run_id, component)
);

INSERT OR REPLACE INTO web_active_runtime_blocker_v0(
    run_id,
    component,
    event_id,
    blocker_code,
    message,
    severity
)
SELECT blocked.run_id,
       blocked.component,
       blocked.id,
       blocked.blocker_code,
       blocked.message,
       blocked.severity
FROM data_health_event AS blocked
WHERE blocked.blocker_code IS NOT NULL
  AND NOT EXISTS (
    SELECT 1
    FROM data_health_event AS resolved
    WHERE resolved.run_id = blocked.run_id
      AND resolved.component = blocked.component
      AND resolved.id > blocked.id
      AND resolved.blocker_code IS NULL
      AND resolved.message = 'previous_session_options_context_ready'
  )
  AND NOT EXISTS (
    SELECT 1
    FROM data_health_event AS newer_blocker
    WHERE newer_blocker.run_id = blocked.run_id
      AND newer_blocker.component = blocked.component
      AND newer_blocker.id > blocked.id
      AND newer_blocker.blocker_code IS NOT NULL
  );

CREATE TRIGGER IF NOT EXISTS web_active_runtime_blocker_insert_v0
AFTER INSERT ON data_health_event
WHEN NEW.blocker_code IS NOT NULL
BEGIN
    INSERT INTO web_active_runtime_blocker_v0(
        run_id,
        component,
        event_id,
        blocker_code,
        message,
        severity
    ) VALUES (
        NEW.run_id,
        NEW.component,
        NEW.id,
        NEW.blocker_code,
        NEW.message,
        NEW.severity
    )
    ON CONFLICT(run_id, component) DO UPDATE SET
        event_id = excluded.event_id,
        blocker_code = excluded.blocker_code,
        message = excluded.message,
        severity = excluded.severity
    WHERE excluded.event_id > web_active_runtime_blocker_v0.event_id;
END;

CREATE TRIGGER IF NOT EXISTS web_active_runtime_blocker_resolve_v0
AFTER INSERT ON data_health_event
WHEN NEW.blocker_code IS NULL
 AND NEW.message = 'previous_session_options_context_ready'
BEGIN
    DELETE FROM web_active_runtime_blocker_v0
    WHERE run_id = NEW.run_id
      AND component = NEW.component
      AND event_id < NEW.id;
END;

CREATE INDEX IF NOT EXISTS idx_web_active_runtime_blocker
ON web_active_runtime_blocker_v0(run_id, event_id);

CREATE INDEX IF NOT EXISTS idx_runtime_session_web_latest
ON runtime_session(run_id, opened_at_utc DESC, id DESC);

CREATE INDEX IF NOT EXISTS idx_m1c_checkpoint_web_latest
ON m1c_checkpoint_v0(
    run_id,
    session_date,
    symbol,
    checkpoint DESC,
    bar_end_utc DESC,
    id DESC
);

CREATE INDEX IF NOT EXISTS idx_m1c_episode_web_latest_exact
ON m1c_episode_v0(
    run_id,
    symbol,
    trigger_bar_end_utc DESC,
    episode_id DESC
);

CREATE INDEX IF NOT EXISTS idx_underlying_quote_web_latest
ON underlying_quote(
    run_id,
    symbol,
    target_timestamp_utc DESC,
    id DESC
);

CREATE INDEX IF NOT EXISTS idx_underlying_bar_web_latest
ON underlying_bar(run_id, bar_end_utc DESC, id DESC);

CREATE INDEX IF NOT EXISTS idx_model_score_web_latest
ON model_score(run_id, bar_end_utc DESC, id DESC);

CREATE INDEX IF NOT EXISTS idx_previous_session_context_web_latest
ON previous_session_options_context(
    run_id,
    current_session_date DESC,
    id DESC
);

CREATE INDEX IF NOT EXISTS idx_ibkr_connection_event_web_latest
ON ibkr_connection_event(run_id, id DESC)
WHERE COALESCE(
    json_extract(details_json, '$.event_kind'),
    'state_transition'
) = 'state_transition';

CREATE INDEX IF NOT EXISTS idx_option_surface_capture_web_latest
ON option_surface_capture(run_id, target_timestamp_utc DESC, id DESC);

CREATE INDEX IF NOT EXISTS idx_source_capture_completion_web_latest
ON source_capture_completion(run_id, session_date DESC, id DESC);

CREATE INDEX IF NOT EXISTS idx_completed_bar_state_web_latest
ON completed_bar_state_v0(run_id, bar_end_utc DESC, symbol);

CREATE INDEX IF NOT EXISTS idx_subscription_lifecycle_web_active
ON subscription_lifecycle_v0(run_id, subscription_kind)
WHERE cancelled_at_utc IS NULL;
