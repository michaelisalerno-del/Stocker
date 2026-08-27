ALTER TABLE shadow_legs ADD COLUMN entry_bid_market_event_id TEXT;
ALTER TABLE shadow_legs ADD COLUMN entry_ask_market_event_id TEXT;
ALTER TABLE shadow_legs ADD COLUMN exit_bid_market_event_id TEXT;
ALTER TABLE shadow_legs ADD COLUMN exit_ask_market_event_id TEXT;
ALTER TABLE shadow_progress ADD COLUMN schedule_count INTEGER NOT NULL DEFAULT 0
    CHECK(schedule_count >= 0);

CREATE INDEX shadow_progress_schedule_idx
    ON shadow_progress(schedule_count, position_id);

CREATE INDEX market_events_shadow_scan_idx
    ON market_events(run_id, instrument_id, source_sequence, event_id);

CREATE TABLE shadow_quote_state (
    position_id TEXT NOT NULL REFERENCES shadow_positions(position_id) ON DELETE CASCADE,
    leg_number INTEGER NOT NULL CHECK(leg_number BETWEEN 0 AND 7),
    instrument_id TEXT NOT NULL REFERENCES instruments(instrument_id),
    bid_event_id TEXT,
    bid_source_sequence INTEGER,
    bid_at_us INTEGER,
    bid_value REAL,
    ask_event_id TEXT,
    ask_source_sequence INTEGER,
    ask_at_us INTEGER,
    ask_value REAL,
    PRIMARY KEY(position_id, leg_number),
    CHECK(
        (bid_event_id IS NULL AND bid_source_sequence IS NULL
            AND bid_at_us IS NULL AND bid_value IS NULL)
        OR
        (bid_event_id IS NOT NULL AND bid_source_sequence >= 0
            AND bid_at_us >= 0 AND bid_value IS NOT NULL)
    ),
    CHECK(
        (ask_event_id IS NULL AND ask_source_sequence IS NULL
            AND ask_at_us IS NULL AND ask_value IS NULL)
        OR
        (ask_event_id IS NOT NULL AND ask_source_sequence >= 0
            AND ask_at_us >= 0 AND ask_value IS NOT NULL)
    )
) STRICT;

CREATE INDEX shadow_quote_state_instrument_idx
    ON shadow_quote_state(position_id, instrument_id, leg_number);

INSERT INTO shadow_quote_state(
    position_id, leg_number, instrument_id,
    bid_event_id, bid_source_sequence, bid_at_us, bid_value,
    ask_event_id, ask_source_sequence, ask_at_us, ask_value
)
SELECT position.position_id, proposal.leg_number, proposal.instrument_id,
    CASE WHEN event.bid_value IS NOT NULL THEN event.event_id END,
    CASE WHEN event.bid_value IS NOT NULL THEN event.source_sequence END,
    CASE WHEN event.bid_value IS NOT NULL THEN event.event_at_us END,
    event.bid_value,
    CASE WHEN event.ask_value IS NOT NULL THEN event.event_id END,
    CASE WHEN event.ask_value IS NOT NULL THEN event.source_sequence END,
    CASE WHEN event.ask_value IS NOT NULL THEN event.event_at_us END,
    event.ask_value
FROM shadow_positions position
JOIN idea_output_legs proposal
  ON proposal.output_id = position.proposed_trade_output_id
LEFT JOIN shadow_legs leg
  ON leg.position_id = position.position_id AND leg.leg_number = proposal.leg_number
LEFT JOIN market_events event ON event.event_id = leg.entry_market_event_id
WHERE position.lifecycle IN ('pending', 'open');

DROP TRIGGER shadow_legs_exit_event_provenance_update;
CREATE TRIGGER shadow_legs_exit_event_provenance_update
BEFORE UPDATE OF position_id, instrument_id, exit_market_event_id ON shadow_legs
WHEN NEW.exit_market_event_id IS NOT NULL AND (
    NEW.exit_market_event_id IS NOT OLD.exit_market_event_id
    OR NEW.position_id IS NOT OLD.position_id
    OR NEW.instrument_id IS NOT OLD.instrument_id
)
BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM shadow_positions position
        JOIN market_events event ON event.event_id = NEW.exit_market_event_id
        WHERE position.position_id = NEW.position_id
          AND event.run_id = position.run_id
          AND event.instrument_id = NEW.instrument_id
    ) AND NOT EXISTS (
        SELECT 1 FROM shadow_quote_state state
        WHERE state.position_id = NEW.position_id
          AND state.instrument_id = NEW.instrument_id
          AND NEW.exit_market_event_id IN (state.bid_event_id, state.ask_event_id)
    ) THEN RAISE(ABORT, 'shadow_leg_exit_event_provenance_mismatch') END;
END;

CREATE TRIGGER shadow_legs_side_event_provenance_insert
BEFORE INSERT ON shadow_legs
BEGIN
    SELECT CASE WHEN NEW.entry_bid_market_event_id IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM shadow_positions position
        JOIN market_events event ON event.event_id = NEW.entry_bid_market_event_id
        WHERE position.position_id = NEW.position_id
          AND event.run_id = position.run_id
          AND event.instrument_id = NEW.instrument_id
          AND event.bid_value IS NOT NULL
    ) THEN RAISE(ABORT, 'shadow_leg_entry_bid_event_provenance_mismatch') END;
    SELECT CASE WHEN NEW.entry_ask_market_event_id IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM shadow_positions position
        JOIN market_events event ON event.event_id = NEW.entry_ask_market_event_id
        WHERE position.position_id = NEW.position_id
          AND event.run_id = position.run_id
          AND event.instrument_id = NEW.instrument_id
          AND event.ask_value IS NOT NULL
    ) THEN RAISE(ABORT, 'shadow_leg_entry_ask_event_provenance_mismatch') END;
    SELECT CASE WHEN NEW.exit_bid_market_event_id IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM shadow_positions position
        JOIN market_events event ON event.event_id = NEW.exit_bid_market_event_id
        WHERE position.position_id = NEW.position_id
          AND event.run_id = position.run_id
          AND event.instrument_id = NEW.instrument_id
          AND event.bid_value IS NOT NULL
    ) THEN RAISE(ABORT, 'shadow_leg_exit_bid_event_provenance_mismatch') END;
    SELECT CASE WHEN NEW.exit_ask_market_event_id IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM shadow_positions position
        JOIN market_events event ON event.event_id = NEW.exit_ask_market_event_id
        WHERE position.position_id = NEW.position_id
          AND event.run_id = position.run_id
          AND event.instrument_id = NEW.instrument_id
          AND event.ask_value IS NOT NULL
    ) THEN RAISE(ABORT, 'shadow_leg_exit_ask_event_provenance_mismatch') END;
END;

CREATE TRIGGER shadow_legs_side_event_provenance_update
BEFORE UPDATE OF position_id, instrument_id,
    entry_bid_market_event_id, entry_ask_market_event_id,
    exit_bid_market_event_id, exit_ask_market_event_id
ON shadow_legs
BEGIN
    SELECT CASE WHEN NEW.entry_bid_market_event_id IS NOT NULL
      AND (NEW.entry_bid_market_event_id IS NOT OLD.entry_bid_market_event_id
        OR NEW.position_id IS NOT OLD.position_id
        OR NEW.instrument_id IS NOT OLD.instrument_id) AND NOT EXISTS (
        SELECT 1 FROM shadow_positions position
        JOIN market_events event ON event.event_id = NEW.entry_bid_market_event_id
        WHERE position.position_id = NEW.position_id
          AND event.run_id = position.run_id
          AND event.instrument_id = NEW.instrument_id
          AND event.bid_value IS NOT NULL
    ) AND NOT EXISTS (
        SELECT 1 FROM shadow_quote_state state
        WHERE state.position_id = NEW.position_id
          AND state.instrument_id = NEW.instrument_id
          AND state.bid_event_id = NEW.entry_bid_market_event_id
    ) THEN RAISE(ABORT, 'shadow_leg_entry_bid_event_provenance_mismatch') END;
    SELECT CASE WHEN NEW.entry_ask_market_event_id IS NOT NULL
      AND (NEW.entry_ask_market_event_id IS NOT OLD.entry_ask_market_event_id
        OR NEW.position_id IS NOT OLD.position_id
        OR NEW.instrument_id IS NOT OLD.instrument_id) AND NOT EXISTS (
        SELECT 1 FROM shadow_positions position
        JOIN market_events event ON event.event_id = NEW.entry_ask_market_event_id
        WHERE position.position_id = NEW.position_id
          AND event.run_id = position.run_id
          AND event.instrument_id = NEW.instrument_id
          AND event.ask_value IS NOT NULL
    ) AND NOT EXISTS (
        SELECT 1 FROM shadow_quote_state state
        WHERE state.position_id = NEW.position_id
          AND state.instrument_id = NEW.instrument_id
          AND state.ask_event_id = NEW.entry_ask_market_event_id
    ) THEN RAISE(ABORT, 'shadow_leg_entry_ask_event_provenance_mismatch') END;
    SELECT CASE WHEN NEW.exit_bid_market_event_id IS NOT NULL
      AND (NEW.exit_bid_market_event_id IS NOT OLD.exit_bid_market_event_id
        OR NEW.position_id IS NOT OLD.position_id
        OR NEW.instrument_id IS NOT OLD.instrument_id) AND NOT EXISTS (
        SELECT 1 FROM shadow_positions position
        JOIN market_events event ON event.event_id = NEW.exit_bid_market_event_id
        WHERE position.position_id = NEW.position_id
          AND event.run_id = position.run_id
          AND event.instrument_id = NEW.instrument_id
          AND event.bid_value IS NOT NULL
    ) AND NOT EXISTS (
        SELECT 1 FROM shadow_quote_state state
        WHERE state.position_id = NEW.position_id
          AND state.instrument_id = NEW.instrument_id
          AND state.bid_event_id = NEW.exit_bid_market_event_id
    ) THEN RAISE(ABORT, 'shadow_leg_exit_bid_event_provenance_mismatch') END;
    SELECT CASE WHEN NEW.exit_ask_market_event_id IS NOT NULL
      AND (NEW.exit_ask_market_event_id IS NOT OLD.exit_ask_market_event_id
        OR NEW.position_id IS NOT OLD.position_id
        OR NEW.instrument_id IS NOT OLD.instrument_id) AND NOT EXISTS (
        SELECT 1 FROM shadow_positions position
        JOIN market_events event ON event.event_id = NEW.exit_ask_market_event_id
        WHERE position.position_id = NEW.position_id
          AND event.run_id = position.run_id
          AND event.instrument_id = NEW.instrument_id
          AND event.ask_value IS NOT NULL
    ) AND NOT EXISTS (
        SELECT 1 FROM shadow_quote_state state
        WHERE state.position_id = NEW.position_id
          AND state.instrument_id = NEW.instrument_id
          AND state.ask_event_id = NEW.exit_ask_market_event_id
    ) THEN RAISE(ABORT, 'shadow_leg_exit_ask_event_provenance_mismatch') END;
END;
