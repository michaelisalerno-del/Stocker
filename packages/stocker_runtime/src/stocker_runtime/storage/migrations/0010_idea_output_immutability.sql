CREATE TABLE idea_output_seals (
    output_id TEXT PRIMARY KEY REFERENCES idea_outputs(output_id) ON DELETE CASCADE
) STRICT;

CREATE TRIGGER idea_outputs_immutable
BEFORE UPDATE ON idea_outputs
BEGIN
    SELECT RAISE(ABORT, 'idea_output_immutable');
END;

CREATE TRIGGER idea_output_inputs_commit_boundary_insert
BEFORE INSERT ON idea_output_inputs
WHEN NOT EXISTS (
    SELECT 1 FROM idea_output_inputs input
    WHERE input.output_id = NEW.output_id
      AND input.event_id = NEW.event_id
      AND input.input_ordinal = NEW.input_ordinal
)
AND (
    NEW.input_ordinal >= 256 OR NOT EXISTS (
        SELECT 1
        FROM idea_outputs output
        JOIN idea_output_commit_boundaries boundary ON boundary.output_id = output.output_id
        JOIN market_events event ON event.event_id = NEW.event_id
        WHERE output.output_id = NEW.output_id
          AND event.run_id = output.run_id
          AND coalesce(event.source_sequence, event.derived_after_source_sequence)
              <= boundary.committed_after_source_sequence
    )
)
BEGIN
    SELECT RAISE(ABORT, 'idea_output_input_after_commit_boundary');
END;

CREATE TRIGGER idea_output_inputs_sealed_insert
BEFORE INSERT ON idea_output_inputs
WHEN EXISTS (
    SELECT 1 FROM idea_output_seals seal WHERE seal.output_id = NEW.output_id
)
BEGIN
    SELECT RAISE(ABORT, 'idea_output_input_immutable');
END;

CREATE TRIGGER idea_output_inputs_immutable
BEFORE UPDATE ON idea_output_inputs
BEGIN
    SELECT RAISE(ABORT, 'idea_output_input_immutable');
END;

CREATE TRIGGER idea_output_inputs_delete_guard
BEFORE DELETE ON idea_output_inputs
WHEN EXISTS (
    SELECT 1 FROM idea_outputs output WHERE output.output_id = OLD.output_id
)
BEGIN
    SELECT RAISE(ABORT, 'idea_output_input_immutable');
END;

CREATE TRIGGER idea_output_legs_bounded_insert
BEFORE INSERT ON idea_output_legs
-- Preserve bounded malformed evidence; ShadowEngine's semantic maximum remains eight.
WHEN NEW.leg_number >= 16
BEGIN
    SELECT RAISE(ABORT, 'idea_output_leg_out_of_bounds');
END;

CREATE TRIGGER idea_output_legs_sealed_insert
BEFORE INSERT ON idea_output_legs
WHEN EXISTS (
    SELECT 1 FROM idea_output_seals seal WHERE seal.output_id = NEW.output_id
)
BEGIN
    SELECT RAISE(ABORT, 'idea_output_leg_immutable');
END;

CREATE TRIGGER idea_output_legs_immutable
BEFORE UPDATE ON idea_output_legs
BEGIN
    SELECT RAISE(ABORT, 'idea_output_leg_immutable');
END;

CREATE TRIGGER idea_output_legs_delete_guard
BEFORE DELETE ON idea_output_legs
WHEN EXISTS (
    SELECT 1 FROM idea_outputs output WHERE output.output_id = OLD.output_id
)
BEGIN
    SELECT RAISE(ABORT, 'idea_output_leg_immutable');
END;

CREATE TRIGGER idea_output_seals_complete_insert
BEFORE INSERT ON idea_output_seals
WHEN NOT EXISTS (
    SELECT 1
    FROM idea_outputs output
    JOIN idea_output_commit_boundaries boundary ON boundary.output_id = output.output_id
    WHERE output.output_id = NEW.output_id
      AND output.input_watermark = output.last_input_event_id
      AND (SELECT count(*) FROM idea_output_inputs input
           WHERE input.output_id = output.output_id) BETWEEN 1 AND 256
      AND (SELECT min(input.input_ordinal) FROM idea_output_inputs input
           WHERE input.output_id = output.output_id) = 0
      AND (SELECT max(input.input_ordinal) FROM idea_output_inputs input
           WHERE input.output_id = output.output_id)
          = (SELECT count(*) - 1 FROM idea_output_inputs input
             WHERE input.output_id = output.output_id)
      AND EXISTS (
          SELECT 1 FROM idea_output_inputs input
          WHERE input.output_id = output.output_id
            AND input.input_ordinal = 0
            AND input.event_id = output.first_input_event_id
      )
      AND EXISTS (
          SELECT 1 FROM idea_output_inputs input
          WHERE input.output_id = output.output_id
            AND input.input_ordinal = (
                SELECT max(last_input.input_ordinal)
                FROM idea_output_inputs last_input
                WHERE last_input.output_id = output.output_id
            )
            AND input.event_id = output.last_input_event_id
      )
      AND NOT EXISTS (
          SELECT 1
          FROM idea_output_inputs input
          JOIN market_events event ON event.event_id = input.event_id
          WHERE input.output_id = output.output_id
            AND (
                event.run_id != output.run_id
                OR coalesce(event.source_sequence, event.derived_after_source_sequence)
                    > boundary.committed_after_source_sequence
            )
      )
      AND (
          (
              output.output_kind = 'proposed_trade'
              AND (SELECT count(*) FROM idea_output_legs leg
                   WHERE leg.output_id = output.output_id) BETWEEN 1 AND 16
              AND (SELECT min(leg.leg_number) FROM idea_output_legs leg
                   WHERE leg.output_id = output.output_id) = 0
              AND (SELECT max(leg.leg_number) FROM idea_output_legs leg
                   WHERE leg.output_id = output.output_id)
                  = (SELECT count(*) - 1 FROM idea_output_legs leg
                     WHERE leg.output_id = output.output_id)
          )
          OR (
              output.output_kind != 'proposed_trade'
              AND NOT EXISTS (
                  SELECT 1 FROM idea_output_legs leg WHERE leg.output_id = output.output_id
              )
          )
      )
)
BEGIN
    SELECT RAISE(ABORT, 'idea_output_seal_incomplete');
END;

CREATE TRIGGER idea_output_seals_immutable
BEFORE UPDATE ON idea_output_seals
BEGIN
    SELECT RAISE(ABORT, 'idea_output_seal_immutable');
END;

CREATE TRIGGER idea_output_seals_delete_guard
BEFORE DELETE ON idea_output_seals
WHEN EXISTS (
    SELECT 1 FROM idea_outputs output WHERE output.output_id = OLD.output_id
)
BEGIN
    SELECT RAISE(ABORT, 'idea_output_seal_immutable');
END;

INSERT INTO idea_output_seals(output_id)
SELECT output_id FROM idea_outputs;

DROP TRIGGER idea_outputs_shadow_schedule_insert;
DROP TRIGGER shadow_schedule_provenance_insert;

CREATE TRIGGER shadow_schedule_provenance_insert
BEFORE INSERT ON shadow_schedule
BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1
        FROM idea_outputs output
        JOIN idea_output_seals seal ON seal.output_id = output.output_id
        WHERE output.output_id = NEW.output_id
          AND output.run_id = NEW.run_id
          AND output.output_kind = 'proposed_trade'
          AND output.authority_status = 'unapproved'
          AND output.data_class = 'shadow_protected'
    ) THEN RAISE(ABORT, 'shadow_schedule_proposal_provenance_mismatch') END;
END;

CREATE TRIGGER idea_output_seals_shadow_schedule_insert
AFTER INSERT ON idea_output_seals
WHEN EXISTS (
    SELECT 1 FROM idea_outputs output
    WHERE output.output_id = NEW.output_id
      AND output.output_kind = 'proposed_trade'
      AND output.authority_status = 'unapproved'
      AND output.data_class = 'shadow_protected'
)
BEGIN
    INSERT INTO shadow_schedule(output_id, run_id)
    SELECT output.output_id, output.run_id
    FROM idea_outputs output
    WHERE output.output_id = NEW.output_id;
END;
