PRAGMA foreign_keys = ON;

ALTER TABLE m1c_checkpoint_v0
    ADD COLUMN canonical_market_proxy_v1 TEXT
        CHECK (
            canonical_market_proxy_v1 IS NULL
            OR canonical_market_proxy_v1 = 'VTI'
        );
ALTER TABLE m1c_checkpoint_v0 ADD COLUMN market_return_w0_v1 REAL;
ALTER TABLE m1c_checkpoint_v0 ADD COLUMN market_range_w0_v1 REAL;
ALTER TABLE m1c_checkpoint_v0 ADD COLUMN market_return_w1_v1 REAL;
ALTER TABLE m1c_checkpoint_v0 ADD COLUMN market_range_w1_v1 REAL;
ALTER TABLE m1c_checkpoint_v0 ADD COLUMN market_shock_thresholds_v1_json TEXT;
ALTER TABLE m1c_checkpoint_v0
    ADD COLUMN market_shock_state_v1 TEXT
        CHECK (
            market_shock_state_v1 IS NULL
            OR market_shock_state_v1 IN (
                'NEGATIVE_SHOCK_ONSET',
                'POSITIVE_SHOCK_ONSET',
                'ONGOING_NEGATIVE_SHOCK',
                'ONGOING_POSITIVE_SHOCK',
                'ELEVATED_RANGE_NONDIRECTIONAL',
                'NORMAL_OTHER',
                'UNKNOWN_INCOMPLETE'
            )
        );
ALTER TABLE m1c_checkpoint_v0 ADD COLUMN market_shock_event_id_v1 TEXT;
ALTER TABLE m1c_checkpoint_v0
    ADD COLUMN shock_sign_v1 INTEGER
        CHECK (shock_sign_v1 IS NULL OR shock_sign_v1 IN (-1, 1));
ALTER TABLE m1c_checkpoint_v0 ADD COLUMN stock_return_w0_v1 REAL;
ALTER TABLE m1c_checkpoint_v0 ADD COLUMN stock_absolute_alignment_v1 REAL;
ALTER TABLE m1c_checkpoint_v0 ADD COLUMN shock_relative_response_v1 REAL;
ALTER TABLE m1c_checkpoint_v0
    ADD COLUMN shock_response_class_v1 TEXT
        CHECK (
            shock_response_class_v1 IS NULL
            OR shock_response_class_v1 IN (
                'AMPLIFYING',
                'RESISTING',
                'NEUTRAL_EXACT',
                'NOT_SHOCK_ONSET',
                'UNKNOWN_INCOMPLETE'
            )
        );
ALTER TABLE m1c_checkpoint_v0 ADD COLUMN shock_resisting_subtype_v1 TEXT;
ALTER TABLE m1c_checkpoint_v0
    ADD COLUMN market_shock_complete_v1 INTEGER
        CHECK (
            market_shock_complete_v1 IS NULL
            OR market_shock_complete_v1 IN (0, 1)
        );
ALTER TABLE m1c_checkpoint_v0
    ADD COLUMN shock_response_complete_v1 INTEGER
        CHECK (
            shock_response_complete_v1 IS NULL
            OR shock_response_complete_v1 IN (0, 1)
        );
ALTER TABLE m1c_checkpoint_v0
    ADD COLUMN market_shock_missing_reasons_v1_json TEXT;
ALTER TABLE m1c_checkpoint_v0
    ADD COLUMN shock_response_missing_reasons_v1_json TEXT;
ALTER TABLE m1c_checkpoint_v0 ADD COLUMN signed_market_shock_source_v1_json TEXT;
