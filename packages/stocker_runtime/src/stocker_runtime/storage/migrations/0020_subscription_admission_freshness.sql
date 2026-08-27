ALTER TABLE subscriptions ADD COLUMN last_admitted_callback_at_us INTEGER
    CHECK(
        last_admitted_callback_at_us IS NULL
        OR last_admitted_callback_at_us >= opened_at_us
    );

DROP INDEX callback_inbox_readiness_latest_idx;
