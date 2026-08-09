CREATE TABLE IF NOT EXISTS option_quote_computation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL,
    option_quote_id INTEGER NOT NULL REFERENCES option_quote(id),
    computation_source TEXT NOT NULL,
    implied_volatility REAL,
    delta REAL,
    gamma REAL,
    theta REAL,
    vega REAL,
    option_price REAL,
    present_value_dividend REAL,
    underlying_reference_price REAL,
    provider_timestamp_utc TEXT,
    receive_timestamp_utc TEXT NOT NULL,
    market_data_type TEXT,
    completeness TEXT NOT NULL,
    UNIQUE(option_quote_id, computation_source)
);

CREATE INDEX IF NOT EXISTS idx_option_quote_computation_quote
    ON option_quote_computation(option_quote_id, computation_source);
