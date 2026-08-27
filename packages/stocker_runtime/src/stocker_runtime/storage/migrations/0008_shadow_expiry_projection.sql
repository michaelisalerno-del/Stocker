ALTER TABLE shadow_progress ADD COLUMN pending_expiry_active INTEGER NOT NULL DEFAULT 1
    CHECK(pending_expiry_active IN (0, 1));

UPDATE shadow_progress AS progress
SET pending_expiry_active = 0
WHERE EXISTS (
    SELECT 1 FROM shadow_positions position
    WHERE position.position_id = progress.position_id
      AND position.lifecycle != 'pending'
);

DROP INDEX shadow_progress_pending_expiry_idx;

CREATE INDEX shadow_progress_pending_expiry_idx
    ON shadow_progress(pending_retention_deadline_us, position_id)
    WHERE pending_evidence_drained = 1 AND pending_expiry_active = 1;

CREATE TRIGGER shadow_progress_pending_expiry_active_monotonic
BEFORE UPDATE OF pending_expiry_active ON shadow_progress
WHEN OLD.pending_expiry_active = 0 AND NEW.pending_expiry_active != 0
BEGIN
    SELECT RAISE(ABORT, 'shadow_pending_expiry_active_monotonic');
END;

CREATE TRIGGER shadow_progress_pending_expiry_deactivate_terminal_only
BEFORE UPDATE OF pending_expiry_active ON shadow_progress
WHEN OLD.pending_expiry_active = 1
  AND NEW.pending_expiry_active = 0
  AND EXISTS (
      SELECT 1 FROM shadow_positions position
      WHERE position.position_id = NEW.position_id
        AND position.lifecycle = 'pending'
  )
BEGIN
    SELECT RAISE(ABORT, 'shadow_pending_expiry_deactivate_requires_terminal');
END;

CREATE TRIGGER shadow_progress_terminal_expiry_insert
AFTER INSERT ON shadow_progress
WHEN EXISTS (
    SELECT 1 FROM shadow_positions position
    WHERE position.position_id = NEW.position_id
      AND position.lifecycle != 'pending'
)
BEGIN
    UPDATE shadow_progress SET pending_expiry_active = 0
    WHERE position_id = NEW.position_id;
END;

CREATE TRIGGER shadow_positions_pending_expiry_deactivate
AFTER UPDATE OF lifecycle ON shadow_positions
WHEN OLD.lifecycle = 'pending' AND NEW.lifecycle != 'pending'
BEGIN
    UPDATE shadow_progress SET pending_expiry_active = 0
    WHERE position_id = NEW.position_id;
END;
