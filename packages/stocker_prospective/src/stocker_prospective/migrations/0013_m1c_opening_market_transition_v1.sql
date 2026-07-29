PRAGMA foreign_keys = ON;

ALTER TABLE m1c_checkpoint_v0
    ADD COLUMN opening_market_proxy_v1 TEXT
        CHECK (
            opening_market_proxy_v1 IS NULL
            OR opening_market_proxy_v1 = 'VTI'
        );
ALTER TABLE m1c_checkpoint_v0 ADD COLUMN vti_session_open_v1 REAL;
ALTER TABLE m1c_checkpoint_v0
    ADD COLUMN vti_prior_regular_session_close_v1 REAL;
ALTER TABLE m1c_checkpoint_v0 ADD COLUMN opening_expected_bar_count_v1 INTEGER;
ALTER TABLE m1c_checkpoint_v0 ADD COLUMN opening_observed_bar_count_v1 INTEGER;
ALTER TABLE m1c_checkpoint_v0 ADD COLUMN market_opening_return_v1 REAL;
ALTER TABLE m1c_checkpoint_v0 ADD COLUMN market_opening_range_v1 REAL;
ALTER TABLE m1c_checkpoint_v0 ADD COLUMN market_overnight_gap_v1 REAL;
ALTER TABLE m1c_checkpoint_v0 ADD COLUMN market_total_transition_v1 REAL;
ALTER TABLE m1c_checkpoint_v0
    ADD COLUMN market_gap_open_alignment_v1 TEXT
        CHECK (
            market_gap_open_alignment_v1 IS NULL
            OR market_gap_open_alignment_v1 IN (
                'ALIGNED_POSITIVE',
                'ALIGNED_NEGATIVE',
                'GAP_UP_OPENING_DOWN',
                'GAP_DOWN_OPENING_UP',
                'ZERO_OR_NEUTRAL',
                'UNKNOWN_INCOMPLETE'
            )
        );
ALTER TABLE m1c_checkpoint_v0 ADD COLUMN opening_thresholds_v1_json TEXT;
ALTER TABLE m1c_checkpoint_v0
    ADD COLUMN opening_market_transition_state_v1 TEXT
        CHECK (
            opening_market_transition_state_v1 IS NULL
            OR opening_market_transition_state_v1 IN (
                'NEGATIVE_SEVERE_OPENING_TRANSITION',
                'POSITIVE_SEVERE_OPENING_TRANSITION',
                'ELEVATED_OPENING_RANGE_NONDIRECTIONAL',
                'NORMAL_OPENING',
                'UNKNOWN_INCOMPLETE'
            )
        );
ALTER TABLE m1c_checkpoint_v0
    ADD COLUMN opening_transition_sign_v1 INTEGER
        CHECK (
            opening_transition_sign_v1 IS NULL
            OR opening_transition_sign_v1 IN (-1, 1)
        );
ALTER TABLE m1c_checkpoint_v0 ADD COLUMN opening_transition_event_id_v1 TEXT;
ALTER TABLE m1c_checkpoint_v0 ADD COLUMN stock_opening_return_v1 REAL;
ALTER TABLE m1c_checkpoint_v0 ADD COLUMN stock_opening_range_v1 REAL;
ALTER TABLE m1c_checkpoint_v0 ADD COLUMN stock_opening_alignment_v1 REAL;
ALTER TABLE m1c_checkpoint_v0
    ADD COLUMN stock_relative_opening_response_v1 REAL;
ALTER TABLE m1c_checkpoint_v0
    ADD COLUMN stock_opening_response_class_v1 TEXT
        CHECK (
            stock_opening_response_class_v1 IS NULL
            OR stock_opening_response_class_v1 IN (
                'AMPLIFYING',
                'RESISTING',
                'NEUTRAL_EXACT',
                'NOT_SEVERE_OPENING_TRANSITION',
                'UNKNOWN_INCOMPLETE'
            )
        );
ALTER TABLE m1c_checkpoint_v0 ADD COLUMN stock_opening_resisting_subtype_v1 TEXT;
ALTER TABLE m1c_checkpoint_v0
    ADD COLUMN opening_market_complete_v1 INTEGER
        CHECK (
            opening_market_complete_v1 IS NULL
            OR opening_market_complete_v1 IN (0, 1)
        );
ALTER TABLE m1c_checkpoint_v0
    ADD COLUMN stock_opening_response_complete_v1 INTEGER
        CHECK (
            stock_opening_response_complete_v1 IS NULL
            OR stock_opening_response_complete_v1 IN (0, 1)
        );
ALTER TABLE m1c_checkpoint_v0
    ADD COLUMN opening_market_missing_reasons_v1_json TEXT;
ALTER TABLE m1c_checkpoint_v0
    ADD COLUMN stock_opening_response_missing_reasons_v1_json TEXT;
ALTER TABLE m1c_checkpoint_v0
    ADD COLUMN opening_market_transition_source_v1_json TEXT;
