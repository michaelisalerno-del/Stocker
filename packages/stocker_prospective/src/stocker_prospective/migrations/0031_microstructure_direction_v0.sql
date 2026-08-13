PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS microstructure_direction_v0 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    microstructure_summary_id INTEGER NOT NULL REFERENCES microstructure_summary_v0(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    episode_id TEXT NOT NULL REFERENCES m1c_episode_v0(episode_id),
    symbol TEXT NOT NULL,
    trigger_timestamp_utc TEXT NOT NULL,
    decision_timestamp_utc TEXT NOT NULL,
    information_cutoff_utc TEXT NOT NULL,
    confirmation_delay_seconds REAL NOT NULL CHECK (confirmation_delay_seconds >= 0.0),
    window_name TEXT NOT NULL,
    direction_method TEXT NOT NULL CHECK (
        direction_method IN (
            'M01_TRADE_IMBALANCE_SIGN',
            'M02_QUOTE_IMBALANCE_SIGN',
            'M03_MICROPRICE_EDGE_SIGN',
            'M04_MC_MD_DIFFERENCE',
            'M05_STRICT_CONSENSUS',
            'M06_MAJORITY_SIGN'
        )
    ),
    action TEXT NOT NULL CHECK (action IN ('UP', 'DOWN', 'ABSTAIN')),
    signed_score REAL,
    component_values_json TEXT NOT NULL,
    component_validity_json TEXT NOT NULL,
    quote_count INTEGER NOT NULL CHECK (quote_count >= 0),
    trade_count INTEGER NOT NULL CHECK (trade_count >= 0),
    classified_trade_count INTEGER NOT NULL CHECK (classified_trade_count >= 0),
    trade_classification_valid_fraction REAL NOT NULL
        CHECK (trade_classification_valid_fraction BETWEEN 0.0 AND 1.0),
    stale_quote_fraction REAL CHECK (stale_quote_fraction BETWEEN 0.0 AND 1.0),
    unclassified_trade_fraction REAL CHECK (unclassified_trade_fraction BETWEEN 0.0 AND 1.0),
    unknown_trade_volume_fraction REAL NOT NULL
        CHECK (unknown_trade_volume_fraction BETWEEN 0.0 AND 1.0),
    probable_buyer_initiated_volume REAL NOT NULL,
    probable_seller_initiated_volume REAL NOT NULL,
    tick_by_tick_status TEXT NOT NULL CHECK (
        tick_by_tick_status IN (
            'BIDASK_AND_LAST_PRESENT', 'BIDASK_ONLY', 'LAST_ONLY', 'ABSENT'
        )
    ),
    depth_status TEXT NOT NULL CHECK (
        depth_status IN ('PRESENT_VALID', 'PRESENT_INVALID', 'ABSENT')
    ),
    market_data_type TEXT NOT NULL,
    data_quality_flags_json TEXT NOT NULL,
    causal_valid INTEGER NOT NULL CHECK (causal_valid IN (0, 1)),
    formulas_version TEXT NOT NULL,
    research_label TEXT NOT NULL CHECK (
        research_label = 'RESEARCH ONLY — MICROSTRUCTURE DIRECTION V0 — NOT VALIDATED — NO RECOMMENDATION'
    ),
    payload_json TEXT NOT NULL,
    claims_json TEXT NOT NULL,
    CHECK (information_cutoff_utc = decision_timestamp_utc),
    UNIQUE(run_id, episode_id, window_name, direction_method)
);

CREATE INDEX IF NOT EXISTS idx_microstructure_direction_episode_v0
ON microstructure_direction_v0(episode_id, decision_timestamp_utc, direction_method);
