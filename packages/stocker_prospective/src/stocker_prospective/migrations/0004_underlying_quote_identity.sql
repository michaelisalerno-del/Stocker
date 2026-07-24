ALTER TABLE underlying_quote ADD COLUMN symbol TEXT;
ALTER TABLE underlying_quote ADD COLUMN con_id INTEGER;

CREATE UNIQUE INDEX IF NOT EXISTS idx_underlying_quote_capture_identity
    ON underlying_quote(
        run_id,
        COALESCE(signal_episode_id, ''),
        COALESCE(symbol, ''),
        target_timestamp_utc
    );
