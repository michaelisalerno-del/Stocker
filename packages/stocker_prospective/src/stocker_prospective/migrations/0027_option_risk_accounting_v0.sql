PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS quiet_option_risk_observation_v0 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    observation_id TEXT NOT NULL REFERENCES quiet_state_observation_v0(observation_id),
    candidate_id TEXT NOT NULL,
    strategy_type TEXT NOT NULL CHECK (
        strategy_type IN (
            'UNDERLYING_LONG',
            'LONG_OPTION',
            'SHORT_PUT',
            'BULL_PUT_SPREAD'
        )
    ),
    dte_bucket TEXT NOT NULL CHECK (
        dte_bucket IN ('0DTE', '1DTE', '3_TO_5_DTE')
    ),
    horizon_label TEXT NOT NULL,
    observed_at_utc TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    executable_pnl_primary INTEGER NOT NULL CHECK (executable_pnl_primary = 1),
    policy_gate INTEGER NOT NULL CHECK (policy_gate = 0),
    can_authorize_trade INTEGER NOT NULL CHECK (can_authorize_trade = 0),
    claims_json TEXT NOT NULL,
    UNIQUE(observation_id, candidate_id, dte_bucket, horizon_label, observed_at_utc)
);

CREATE TABLE IF NOT EXISTS quiet_option_strategy_comparison_v0 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    observation_id TEXT NOT NULL REFERENCES quiet_state_observation_v0(observation_id),
    dte_bucket TEXT NOT NULL CHECK (
        dte_bucket IN ('0DTE', '1DTE', '3_TO_5_DTE')
    ),
    horizon_label TEXT NOT NULL,
    horizon_minutes INTEGER,
    payload_json TEXT NOT NULL,
    executable_pnl_primary INTEGER NOT NULL CHECK (executable_pnl_primary = 1),
    greek_attribution_diagnostic_only INTEGER NOT NULL
        CHECK (greek_attribution_diagnostic_only = 1),
    policy_gate INTEGER NOT NULL CHECK (policy_gate = 0),
    can_authorize_trade INTEGER NOT NULL CHECK (can_authorize_trade = 0),
    claims_json TEXT NOT NULL,
    UNIQUE(observation_id, dte_bucket, horizon_label)
);

CREATE INDEX IF NOT EXISTS idx_quiet_option_risk_observation_parent
    ON quiet_option_risk_observation_v0(
        observation_id,
        dte_bucket,
        horizon_label,
        observed_at_utc
    );

CREATE INDEX IF NOT EXISTS idx_quiet_option_strategy_comparison_parent
    ON quiet_option_strategy_comparison_v0(observation_id, dte_bucket, horizon_label);
