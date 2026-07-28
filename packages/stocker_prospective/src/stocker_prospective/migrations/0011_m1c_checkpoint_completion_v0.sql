CREATE TABLE IF NOT EXISTS m1c_checkpoint_completion_v0 (
    checkpoint_id INTEGER PRIMARY KEY REFERENCES m1c_checkpoint_v0(id),
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    symbol TEXT NOT NULL,
    session_date TEXT NOT NULL,
    checkpoint INTEGER NOT NULL,
    completed_at_utc TEXT NOT NULL,
    claims_json TEXT NOT NULL,
    UNIQUE(run_id, symbol, session_date, checkpoint)
);

CREATE INDEX IF NOT EXISTS idx_m1c_checkpoint_completion_run_session
ON m1c_checkpoint_completion_v0(run_id, session_date, symbol, checkpoint);

CREATE TABLE IF NOT EXISTS option_episode_schedule_v0 (
    episode_id TEXT PRIMARY KEY,
    checkpoint_id INTEGER NOT NULL REFERENCES m1c_checkpoint_v0(id),
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    symbol TEXT NOT NULL,
    session_date TEXT NOT NULL,
    entry_timestamp_utc TEXT NOT NULL,
    episode_kind TEXT NOT NULL,
    probability REAL NOT NULL,
    quiet_state INTEGER NOT NULL CHECK (quiet_state IN (0, 1)),
    directional_actions_json TEXT NOT NULL,
    recording_duration_seconds INTEGER NOT NULL,
    strike_steps INTEGER NOT NULL,
    maximum_contracts INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('scheduled', 'streaming', 'complete', 'rejected', 'expired')
    ),
    updated_at_utc TEXT NOT NULL,
    claims_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_option_episode_schedule_restore
ON option_episode_schedule_v0(run_id, status, entry_timestamp_utc);
