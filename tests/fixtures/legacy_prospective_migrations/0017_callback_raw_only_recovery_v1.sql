PRAGMA foreign_keys = ON;

-- 0016 may already be present on a recorder database. Keep its filename and
-- schema immutable; this forward migration adds the ownership receipt and an
-- explicit distinction between normal projection and fatal-latched raw-only
-- recovery without deleting or rewriting prior evidence.
ALTER TABLE callback_inbox_v1
    ADD COLUMN stream_owner_json TEXT;

ALTER TABLE callback_processing_commit_v1
    ADD COLUMN processing_disposition TEXT NOT NULL
        DEFAULT 'normal_scientific_projection'
        CHECK (
            processing_disposition IN (
                'normal_scientific_projection',
                'scientifically_blocked_raw_only'
            )
        );

ALTER TABLE callback_processing_commit_v1
    ADD COLUMN scientific_projection_complete INTEGER NOT NULL
        DEFAULT 1
        CHECK (scientific_projection_complete IN (0, 1));

-- The first hardening migration protected only V1.1 rows from UPDATE. All
-- opening-reversal discovery/outcome rows are append-only evidence, including
-- legacy and generic identities, so replace the tracked triggers forward.
DROP TRIGGER IF EXISTS trg_opening_reversal_v1_1_discovery_immutable;
DROP TRIGGER IF EXISTS trg_opening_reversal_v1_1_outcome_immutable;

CREATE TRIGGER IF NOT EXISTS trg_opening_reversal_discovery_immutable
BEFORE UPDATE ON opening_reversal_contract_discovery_v1
BEGIN
    SELECT RAISE(ABORT, 'blocked_opening_reversal_contract_discovery_is_immutable');
END;

CREATE TRIGGER IF NOT EXISTS trg_opening_reversal_option_outcome_immutable
BEFORE UPDATE ON opening_reversal_primary_option_outcome_v1
BEGIN
    SELECT RAISE(ABORT, 'blocked_opening_reversal_option_outcome_is_immutable');
END;
