ALTER TABLE shadow_progress ADD COLUMN pending_retention_deadline_us INTEGER NOT NULL DEFAULT 0
    CHECK(pending_retention_deadline_us >= 0);

UPDATE shadow_progress AS progress
SET pending_retention_deadline_us = (
    SELECT output.emitted_at_us + 2592000000000
    FROM shadow_positions position
    JOIN idea_outputs output ON output.output_id = position.proposed_trade_output_id
    WHERE position.position_id = progress.position_id
);

CREATE TRIGGER shadow_progress_pending_retention_deadline_immutable
BEFORE UPDATE OF pending_retention_deadline_us ON shadow_progress
WHEN NEW.pending_retention_deadline_us IS NOT OLD.pending_retention_deadline_us
BEGIN
    SELECT RAISE(ABORT, 'shadow_pending_retention_deadline_immutable');
END;

CREATE TABLE shadow_schedule (
    output_id TEXT PRIMARY KEY REFERENCES idea_outputs(output_id) ON DELETE CASCADE,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    schedule_count INTEGER NOT NULL DEFAULT 0 CHECK(schedule_count >= 0)
) STRICT;

CREATE INDEX shadow_schedule_run_count_idx
    ON shadow_schedule(run_id, schedule_count, output_id);

CREATE TRIGGER shadow_schedule_provenance_insert
BEFORE INSERT ON shadow_schedule
BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM idea_outputs output
        WHERE output.output_id = NEW.output_id
          AND output.run_id = NEW.run_id
          AND output.output_kind = 'proposed_trade'
          AND output.authority_status = 'unapproved'
          AND output.data_class = 'shadow_protected'
    ) THEN RAISE(ABORT, 'shadow_schedule_proposal_provenance_mismatch') END;
END;

CREATE TRIGGER shadow_schedule_identity_immutable
BEFORE UPDATE OF output_id, run_id ON shadow_schedule
BEGIN
    SELECT RAISE(ABORT, 'shadow_schedule_identity_immutable');
END;

INSERT INTO shadow_schedule(output_id, run_id, schedule_count)
SELECT output.output_id, output.run_id, coalesce(progress.schedule_count, 0)
FROM idea_outputs output
LEFT JOIN shadow_positions position
  ON position.proposed_trade_output_id = output.output_id
LEFT JOIN shadow_progress progress ON progress.position_id = position.position_id
WHERE output.output_kind = 'proposed_trade'
  AND output.authority_status = 'unapproved'
  AND output.data_class = 'shadow_protected'
  AND (position.position_id IS NULL OR position.lifecycle IN ('pending', 'open'));

CREATE TRIGGER idea_outputs_shadow_schedule_insert
AFTER INSERT ON idea_outputs
WHEN NEW.output_kind = 'proposed_trade'
  AND NEW.authority_status = 'unapproved'
  AND NEW.data_class = 'shadow_protected'
BEGIN
    INSERT INTO shadow_schedule(output_id, run_id) VALUES (NEW.output_id, NEW.run_id);
END;

CREATE TRIGGER shadow_positions_schedule_terminal_insert
AFTER INSERT ON shadow_positions
WHEN NEW.lifecycle IN ('closed', 'invalid')
BEGIN
    DELETE FROM shadow_schedule WHERE output_id = NEW.proposed_trade_output_id;
END;

CREATE TRIGGER shadow_positions_schedule_terminal_update
AFTER UPDATE OF lifecycle ON shadow_positions
WHEN NEW.lifecycle IN ('closed', 'invalid')
BEGIN
    DELETE FROM shadow_schedule WHERE output_id = NEW.proposed_trade_output_id;
END;

CREATE INDEX market_events_shadow_raw_idx
    ON market_events(run_id, instrument_id, event_kind, source_sequence, event_id);
