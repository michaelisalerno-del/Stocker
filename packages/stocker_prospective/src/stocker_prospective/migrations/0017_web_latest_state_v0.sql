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
