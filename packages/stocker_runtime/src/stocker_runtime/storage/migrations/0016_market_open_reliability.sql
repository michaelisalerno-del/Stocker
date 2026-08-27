ALTER TABLE recorder_generations ADD COLUMN ownership_protocol TEXT
    CHECK(ownership_protocol IS NULL OR ownership_protocol = 'local_flock_v1');
ALTER TABLE recorder_generations ADD COLUMN git_commit TEXT;
ALTER TABLE recorder_generations ADD COLUMN input_hash TEXT
    CHECK(input_hash IS NULL OR length(input_hash) = 64);
ALTER TABLE recorder_generations ADD COLUMN fatal_recovery_authorized_at_us INTEGER
    CHECK(fatal_recovery_authorized_at_us IS NULL OR fatal_recovery_authorized_at_us >= 0);
ALTER TABLE recorder_generations ADD COLUMN fatal_recovery_operator TEXT;
ALTER TABLE recorder_generations ADD COLUMN fatal_recovery_reason TEXT;
ALTER TABLE recorder_generations ADD COLUMN recovered_fatal_code TEXT;

ALTER TABLE subscriptions ADD COLUMN stale_after_us INTEGER
    CHECK(stale_after_us IS NULL OR stale_after_us > 0);
ALTER TABLE subscriptions ADD COLUMN retry_count INTEGER NOT NULL DEFAULT 0
    CHECK(retry_count >= 0);
ALTER TABLE subscriptions ADD COLUMN next_retry_at_us INTEGER
    CHECK(next_retry_at_us IS NULL OR next_retry_at_us >= 0);
ALTER TABLE subscriptions ADD COLUMN last_attempt_at_us INTEGER
    CHECK(last_attempt_at_us IS NULL OR last_attempt_at_us >= 0);
ALTER TABLE subscriptions ADD COLUMN last_error_code TEXT;
ALTER TABLE subscriptions ADD COLUMN permanent_failure INTEGER NOT NULL DEFAULT 0
    CHECK(permanent_failure IN (0, 1));

CREATE INDEX subscriptions_retry_due_idx
    ON subscriptions(run_id, recorder_generation, connection_generation, next_retry_at_us,
                     subscription_id)
    WHERE lifecycle IN ('disconnected', 'degraded', 'paused') AND permanent_failure = 0;
