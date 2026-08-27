CREATE INDEX callback_inbox_payload_run_sequence_idx
    ON callback_inbox(run_id, lifecycle, source_sequence)
    WHERE payload_json IS NOT NULL;
