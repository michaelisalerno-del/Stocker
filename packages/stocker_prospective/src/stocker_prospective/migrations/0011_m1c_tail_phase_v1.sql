PRAGMA foreign_keys = ON;

ALTER TABLE group_o_session_context_v0
    ADD COLUMN previous_close_implied_movement_15m REAL;

ALTER TABLE m1c_checkpoint_v0
    ADD COLUMN m1c_high_tail_v1 INTEGER
        CHECK (m1c_high_tail_v1 IS NULL OR m1c_high_tail_v1 IN (0, 1));
ALTER TABLE m1c_checkpoint_v0
    ADD COLUMN m1c_tail_phase_v1 TEXT
        CHECK (
            m1c_tail_phase_v1 IS NULL
            OR m1c_tail_phase_v1 IN (
                'FIRST_ENTRY',
                'PERSISTENT',
                'RE_ENTRY',
                'OUTSIDE_TAIL',
                'UNKNOWN_INCOMPLETE'
            )
        );
ALTER TABLE m1c_checkpoint_v0 ADD COLUMN tail_entry_number_v1 INTEGER;
ALTER TABLE m1c_checkpoint_v0 ADD COLUMN tail_run_length_checkpoints_v1 INTEGER;
ALTER TABLE m1c_checkpoint_v0 ADD COLUMN tail_run_age_minutes_v1 REAL;
ALTER TABLE m1c_checkpoint_v0 ADD COLUMN prior_tail_entries_v1 INTEGER;
ALTER TABLE m1c_checkpoint_v0
    ADD COLUMN previous_checkpoint_above_tail_v1 INTEGER
        CHECK (
            previous_checkpoint_above_tail_v1 IS NULL
            OR previous_checkpoint_above_tail_v1 IN (0, 1)
        );
ALTER TABLE m1c_checkpoint_v0 ADD COLUMN minutes_since_previous_tail_exit_v1 REAL;
ALTER TABLE m1c_checkpoint_v0
    ADD COLUMN phase_history_complete_v1 INTEGER
        CHECK (
            phase_history_complete_v1 IS NULL
            OR phase_history_complete_v1 IN (0, 1)
        );
ALTER TABLE m1c_checkpoint_v0 ADD COLUMN phase_missing_reason_v1 TEXT;
ALTER TABLE m1c_checkpoint_v0 ADD COLUMN movement_consumed_v1 REAL;
ALTER TABLE m1c_checkpoint_v0 ADD COLUMN movement_consumed_numerator_v1 REAL;
ALTER TABLE m1c_checkpoint_v0 ADD COLUMN movement_consumed_denominator_v1 REAL;
ALTER TABLE m1c_checkpoint_v0
    ADD COLUMN movement_consumed_complete_v1 INTEGER
        CHECK (
            movement_consumed_complete_v1 IS NULL
            OR movement_consumed_complete_v1 IN (0, 1)
        );
ALTER TABLE m1c_checkpoint_v0 ADD COLUMN movement_consumed_missing_reason_v1 TEXT;
ALTER TABLE m1c_checkpoint_v0
    ADD COLUMN movement_consumed_bucket_v1 TEXT
        CHECK (
            movement_consumed_bucket_v1 IS NULL
            OR movement_consumed_bucket_v1 IN (
                'LOW_OR_EQUAL',
                'HIGH',
                'UNKNOWN_INCOMPLETE'
            )
        );
ALTER TABLE m1c_checkpoint_v0 ADD COLUMN tail_phase_source_v1_json TEXT;

ALTER TABLE m1c_episode_v0 ADD COLUMN m1c_model_version_v1 TEXT;
ALTER TABLE m1c_episode_v0 ADD COLUMN m1c_high_tail_threshold_v1 REAL;
ALTER TABLE m1c_episode_v0
    ADD COLUMN phase_at_trigger_v1 TEXT
        CHECK (
            phase_at_trigger_v1 IS NULL
            OR phase_at_trigger_v1 IN (
                'FIRST_ENTRY',
                'PERSISTENT',
                'RE_ENTRY',
                'OUTSIDE_TAIL',
                'UNKNOWN_INCOMPLETE'
            )
        );
ALTER TABLE m1c_episode_v0 ADD COLUMN tail_entry_number_v1 INTEGER;
ALTER TABLE m1c_episode_v0 ADD COLUMN tail_run_length_checkpoints_v1 INTEGER;
ALTER TABLE m1c_episode_v0 ADD COLUMN tail_run_age_at_trigger_v1 REAL;
ALTER TABLE m1c_episode_v0 ADD COLUMN prior_tail_entries_v1 INTEGER;
ALTER TABLE m1c_episode_v0
    ADD COLUMN previous_checkpoint_above_tail_v1 INTEGER
        CHECK (
            previous_checkpoint_above_tail_v1 IS NULL
            OR previous_checkpoint_above_tail_v1 IN (0, 1)
        );
ALTER TABLE m1c_episode_v0 ADD COLUMN minutes_since_previous_tail_exit_v1 REAL;
ALTER TABLE m1c_episode_v0
    ADD COLUMN phase_history_complete_v1 INTEGER
        CHECK (
            phase_history_complete_v1 IS NULL
            OR phase_history_complete_v1 IN (0, 1)
        );
ALTER TABLE m1c_episode_v0 ADD COLUMN phase_missing_reason_v1 TEXT;
ALTER TABLE m1c_episode_v0 ADD COLUMN movement_consumed_at_trigger_v1 REAL;
ALTER TABLE m1c_episode_v0 ADD COLUMN movement_consumed_numerator_v1 REAL;
ALTER TABLE m1c_episode_v0 ADD COLUMN movement_consumed_denominator_v1 REAL;
ALTER TABLE m1c_episode_v0
    ADD COLUMN movement_consumed_complete_v1 INTEGER
        CHECK (
            movement_consumed_complete_v1 IS NULL
            OR movement_consumed_complete_v1 IN (0, 1)
        );
ALTER TABLE m1c_episode_v0 ADD COLUMN movement_consumed_missing_reason_v1 TEXT;
ALTER TABLE m1c_episode_v0
    ADD COLUMN movement_consumed_bucket_v1 TEXT
        CHECK (
            movement_consumed_bucket_v1 IS NULL
            OR movement_consumed_bucket_v1 IN (
                'LOW_OR_EQUAL',
                'HIGH',
                'UNKNOWN_INCOMPLETE'
            )
        );
ALTER TABLE m1c_episode_v0 ADD COLUMN tail_phase_source_v1_json TEXT;
