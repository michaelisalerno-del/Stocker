CREATE TABLE shadow_progress (
    position_id TEXT PRIMARY KEY REFERENCES shadow_positions(position_id) ON DELETE CASCADE,
    entry_after_source_sequence INTEGER NOT NULL CHECK(entry_after_source_sequence >= 0),
    next_source_sequence INTEGER NOT NULL CHECK(next_source_sequence > entry_after_source_sequence),
    entry_source_sequence INTEGER CHECK(entry_source_sequence > entry_after_source_sequence),
    final_target_at_us INTEGER CHECK(final_target_at_us IS NULL OR final_target_at_us >= 0),
    next_horizon_index INTEGER NOT NULL CHECK(next_horizon_index BETWEEN 0 AND 8),
    mfe REAL,
    mae REAL,
    mfe_event_ids_json TEXT NOT NULL DEFAULT '[]' CHECK(
        stocker_canonical_json(mfe_event_ids_json) = 1
        AND length(CAST(mfe_event_ids_json AS BLOB)) <= 4096
    ),
    mae_event_ids_json TEXT NOT NULL DEFAULT '[]' CHECK(
        stocker_canonical_json(mae_event_ids_json) = 1
        AND length(CAST(mae_event_ids_json AS BLOB)) <= 4096
    ),
    updated_at_us INTEGER NOT NULL CHECK(updated_at_us >= 0)
) STRICT;

CREATE INDEX shadow_progress_cursor_idx
    ON shadow_progress(next_source_sequence, position_id);
