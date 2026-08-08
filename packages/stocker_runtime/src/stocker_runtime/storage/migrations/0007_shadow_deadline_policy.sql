ALTER TABLE shadow_progress ADD COLUMN pending_evidence_drained INTEGER NOT NULL DEFAULT 0
    CHECK(pending_evidence_drained IN (0, 1));

CREATE TRIGGER shadow_progress_pending_evidence_drained_monotonic
BEFORE UPDATE OF pending_evidence_drained ON shadow_progress
WHEN OLD.pending_evidence_drained = 1 AND NEW.pending_evidence_drained != 1
BEGIN
    SELECT RAISE(ABORT, 'shadow_pending_evidence_drained_monotonic');
END;

CREATE INDEX shadow_progress_pending_expiry_idx
    ON shadow_progress(pending_retention_deadline_us, position_id)
    WHERE pending_evidence_drained = 1;

CREATE TABLE shadow_run_policies (
    run_id TEXT PRIMARY KEY REFERENCES runs(run_id),
    policy_hash TEXT NOT NULL CHECK(length(policy_hash) = 64),
    policy_json TEXT NOT NULL CHECK(
        stocker_canonical_json(policy_json) = 1
        AND length(CAST(policy_json AS BLOB)) <= 16384
    )
) STRICT;

CREATE TRIGGER shadow_run_policies_provenance_insert
BEFORE INSERT ON shadow_run_policies
BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM runs
        WHERE runs.run_id = NEW.run_id
          AND runs.mode = 'shadow'
          AND runs.data_class = 'shadow_protected'
    ) THEN RAISE(ABORT, 'shadow_run_policy_run_mismatch') END;
END;

CREATE TRIGGER shadow_run_policies_immutable
BEFORE UPDATE ON shadow_run_policies
BEGIN
    SELECT RAISE(ABORT, 'shadow_run_policy_immutable');
END;

CREATE TRIGGER shadow_run_policies_delete_forbidden
BEFORE DELETE ON shadow_run_policies
BEGIN
    SELECT RAISE(ABORT, 'shadow_run_policy_immutable');
END;

INSERT INTO shadow_run_policies(run_id, policy_hash, policy_json)
SELECT run_id, policy_hash, policy_json
FROM shadow_positions
GROUP BY run_id, policy_hash, policy_json;
