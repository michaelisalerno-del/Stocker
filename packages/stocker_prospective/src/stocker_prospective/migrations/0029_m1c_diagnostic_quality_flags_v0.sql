PRAGMA foreign_keys = ON;

ALTER TABLE m1c_checkpoint_v0
ADD COLUMN diagnostic_quality_flags_json TEXT NOT NULL DEFAULT '[]';
