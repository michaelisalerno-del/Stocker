CREATE TABLE idea_output_commit_boundaries (
    output_id TEXT PRIMARY KEY REFERENCES idea_outputs(output_id) ON DELETE CASCADE,
    committed_after_source_sequence INTEGER NOT NULL
        CHECK(committed_after_source_sequence >= 0)
) STRICT;

CREATE TRIGGER idea_output_commit_boundaries_provenance_insert
BEFORE INSERT ON idea_output_commit_boundaries
WHEN NOT EXISTS (
    SELECT 1 FROM idea_outputs output WHERE output.output_id = NEW.output_id
)
BEGIN
    SELECT RAISE(ABORT, 'idea_output_commit_boundary_provenance_mismatch');
END;

CREATE TRIGGER idea_output_commit_boundaries_immutable
BEFORE UPDATE ON idea_output_commit_boundaries
BEGIN
    SELECT RAISE(ABORT, 'idea_output_commit_boundary_immutable');
END;

CREATE TRIGGER idea_output_commit_boundaries_lineage_floor_insert
BEFORE INSERT ON idea_output_commit_boundaries
WHEN EXISTS (
    SELECT 1 FROM idea_outputs output
    JOIN market_events event ON event.event_id = output.last_input_event_id
    WHERE output.output_id = NEW.output_id
      AND NEW.committed_after_source_sequence
          < coalesce(event.source_sequence, event.derived_after_source_sequence)
)
BEGIN
    SELECT RAISE(ABORT, 'idea_output_commit_boundary_precedes_lineage');
END;

CREATE TRIGGER idea_output_commit_boundaries_delete_guard
BEFORE DELETE ON idea_output_commit_boundaries
WHEN EXISTS (
    SELECT 1 FROM idea_outputs output WHERE output.output_id = OLD.output_id
)
BEGIN
    SELECT RAISE(ABORT, 'idea_output_commit_boundary_immutable');
END;

CREATE TRIGGER idea_output_commit_boundaries_shadow_history_guard
BEFORE INSERT ON idea_output_commit_boundaries
WHEN EXISTS (
    SELECT 1 FROM shadow_positions position
    WHERE position.proposed_trade_output_id = NEW.output_id
      AND position.lifecycle IN ('open', 'closed')
)
BEGIN
    SELECT RAISE(ABORT, 'idea_output_commit_boundary_requires_shadow_recreation');
END;

WITH run_watermarks AS (
    SELECT run.run_id, coalesce((
        SELECT max(sequence) FROM (
            SELECT max(callback.source_sequence) AS sequence
            FROM callback_inbox callback
            INDEXED BY callback_inbox_run_sequence_idx
            WHERE callback.run_id = run.run_id
            UNION ALL
            SELECT max(coalesce(event.source_sequence, event.derived_after_source_sequence))
            FROM market_events event
            INDEXED BY market_events_causal_sequence_idx
            WHERE event.run_id = run.run_id
            UNION ALL
            SELECT max(receipt.last_source_sequence)
            FROM callback_receipts receipt
            INDEXED BY callback_receipts_run_sequence_idx
            WHERE receipt.run_id = run.run_id
            UNION ALL
            SELECT max(watermark.compacted_through_sequence)
            FROM callback_compaction_watermarks watermark
            WHERE watermark.run_id = run.run_id
        )
    ), 0) AS source_sequence
    FROM runs run
)
INSERT INTO idea_output_commit_boundaries(output_id, committed_after_source_sequence)
SELECT output.output_id, watermark.source_sequence
FROM idea_outputs output
JOIN run_watermarks watermark ON watermark.run_id = output.run_id;

UPDATE shadow_progress AS progress
SET entry_after_source_sequence = max(
        progress.entry_after_source_sequence,
        (
            SELECT boundary.committed_after_source_sequence
            FROM shadow_positions position
            JOIN idea_output_commit_boundaries boundary
              ON boundary.output_id = position.proposed_trade_output_id
            WHERE position.position_id = progress.position_id
        )
    ),
    next_source_sequence = max(
        progress.next_source_sequence,
        (
            SELECT boundary.committed_after_source_sequence + 1
            FROM shadow_positions position
            JOIN idea_output_commit_boundaries boundary
              ON boundary.output_id = position.proposed_trade_output_id
            WHERE position.position_id = progress.position_id
        )
    )
WHERE EXISTS (
    SELECT 1 FROM shadow_positions position
    WHERE position.position_id = progress.position_id
      AND position.lifecycle = 'pending'
);

UPDATE shadow_quote_state
SET bid_event_id = NULL,
    bid_source_sequence = NULL,
    bid_at_us = NULL,
    bid_value = NULL,
    ask_event_id = NULL,
    ask_source_sequence = NULL,
    ask_at_us = NULL,
    ask_value = NULL
WHERE EXISTS (
    SELECT 1 FROM shadow_positions position
    WHERE position.position_id = shadow_quote_state.position_id
      AND position.lifecycle = 'pending'
);

CREATE TRIGGER idea_outputs_commit_boundary_insert
AFTER INSERT ON idea_outputs
BEGIN
    INSERT INTO idea_output_commit_boundaries(output_id, committed_after_source_sequence)
    SELECT NEW.output_id, coalesce(max(sequence), 0)
    FROM (
        SELECT max(callback.source_sequence) AS sequence
        FROM callback_inbox callback
        INDEXED BY callback_inbox_run_sequence_idx
        WHERE callback.run_id = NEW.run_id
        UNION ALL
        SELECT max(coalesce(event.source_sequence, event.derived_after_source_sequence))
        FROM market_events event
        INDEXED BY market_events_causal_sequence_idx
        WHERE event.run_id = NEW.run_id
        UNION ALL
        SELECT max(receipt.last_source_sequence)
        FROM callback_receipts receipt
        INDEXED BY callback_receipts_run_sequence_idx
        WHERE receipt.run_id = NEW.run_id
        UNION ALL
        SELECT max(watermark.compacted_through_sequence)
        FROM callback_compaction_watermarks watermark
        WHERE watermark.run_id = NEW.run_id
    );
END;
