CREATE INDEX callback_inbox_unreceipted_terminal_idx
    ON callback_inbox(run_id, source_sequence)
    WHERE receipt_batch_id IS NULL
        AND lifecycle IN ('acknowledged', 'failed');
