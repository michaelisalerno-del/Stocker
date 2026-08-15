PRAGMA foreign_keys = ON;

ALTER TABLE raw_partition_manifest_v0
ADD COLUMN retention_state TEXT NOT NULL DEFAULT 'ACTIVE'
CHECK (retention_state IN ('ACTIVE', 'RETIREMENT_PREPARED', 'RETIRED', 'RETAINED'));

CREATE TABLE IF NOT EXISTS m1c_event_window_retention_session_v0 (
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    session_date TEXT NOT NULL,
    dataset_version TEXT NOT NULL CHECK (dataset_version = 'm1c_event_window_retention_v0'),
    status TEXT NOT NULL CHECK (status IN ('PREPARED', 'COMPLETE')),
    plan_json TEXT NOT NULL,
    episode_count INTEGER NOT NULL CHECK (episode_count >= 0),
    source_partition_count INTEGER NOT NULL CHECK (source_partition_count >= 0),
    source_row_count INTEGER NOT NULL CHECK (source_row_count >= 0),
    retained_row_count INTEGER NOT NULL CHECK (retained_row_count >= 0),
    summarised_row_count INTEGER NOT NULL CHECK (summarised_row_count >= 0),
    summary_file_path TEXT NOT NULL,
    summary_sha256 TEXT NOT NULL CHECK (length(summary_sha256) = 64),
    prepared_at_utc TEXT NOT NULL,
    completed_at_utc TEXT,
    PRIMARY KEY (run_id, session_date, dataset_version)
);

CREATE TABLE IF NOT EXISTS m1c_event_window_retention_partition_v0 (
    run_id TEXT NOT NULL,
    session_date TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    source_manifest_id INTEGER NOT NULL REFERENCES raw_partition_manifest_v0(id),
    source_content_hash TEXT NOT NULL CHECK (length(source_content_hash) = 64),
    source_file_path TEXT NOT NULL,
    source_row_count INTEGER NOT NULL CHECK (source_row_count >= 0),
    retained_manifest_id INTEGER REFERENCES raw_partition_manifest_v0(id),
    retained_content_hash TEXT,
    retained_file_path TEXT,
    retained_row_count INTEGER NOT NULL CHECK (retained_row_count >= 0),
    summarised_row_count INTEGER NOT NULL CHECK (summarised_row_count >= 0),
    source_verified INTEGER NOT NULL CHECK (source_verified = 1),
    retained_verified INTEGER NOT NULL CHECK (retained_verified = 1),
    source_deleted INTEGER NOT NULL DEFAULT 0 CHECK (source_deleted IN (0, 1)),
    deleted_at_utc TEXT,
    PRIMARY KEY (run_id, source_content_hash),
    FOREIGN KEY (run_id, session_date, dataset_version)
        REFERENCES m1c_event_window_retention_session_v0(
            run_id, session_date, dataset_version
        ),
    CHECK (
        (retained_row_count = 0 AND retained_content_hash IS NULL
            AND retained_file_path IS NULL AND retained_manifest_id IS NULL)
        OR
        (retained_row_count > 0 AND length(retained_content_hash) = 64
            AND retained_file_path IS NOT NULL AND retained_manifest_id IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_raw_partition_retention_active_v0
ON raw_partition_manifest_v0(run_id, session_date, retention_state, event_type);

CREATE INDEX IF NOT EXISTS idx_event_window_retention_session_v0
ON m1c_event_window_retention_session_v0(run_id, status, session_date);

CREATE TABLE IF NOT EXISTS m1c_event_window_callback_purge_v0 (
    run_id TEXT NOT NULL,
    session_date TEXT NOT NULL,
    dataset_version TEXT NOT NULL CHECK (dataset_version = 'm1c_event_window_retention_v0'),
    purged_callback_count INTEGER NOT NULL CHECK (purged_callback_count >= 0),
    callback_identity_set_sha256 TEXT NOT NULL
        CHECK (length(callback_identity_set_sha256) = 64),
    purged_at_utc TEXT NOT NULL,
    PRIMARY KEY (run_id, session_date, dataset_version),
    FOREIGN KEY (run_id, session_date, dataset_version)
        REFERENCES m1c_event_window_retention_session_v0(
            run_id, session_date, dataset_version
        )
);
