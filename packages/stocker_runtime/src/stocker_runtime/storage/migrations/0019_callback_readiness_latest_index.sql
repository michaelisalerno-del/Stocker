CREATE INDEX callback_inbox_readiness_latest_idx
    ON callback_inbox(
        run_id,
        recorder_generation,
        connection_generation,
        request_id,
        received_at_us DESC
    )
    WHERE lifecycle = 'acknowledged';
