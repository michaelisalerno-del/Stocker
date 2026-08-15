PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS m1c_directionless_shadow_run_v0 (
    run_id TEXT PRIMARY KEY,
    strategy_version TEXT NOT NULL CHECK (strategy_version = 'm1c_directionless_t20_d_v0'),
    activation_timestamp_utc TEXT NOT NULL,
    first_eligible_session TEXT NOT NULL,
    planned_session_count INTEGER NOT NULL CHECK (planned_session_count = 20),
    price_path TEXT NOT NULL CHECK (price_path = 'LEVEL1_BBO_EXECUTABLE_SIDE'),
    contract_sha256 TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    code_hash TEXT NOT NULL,
    m1c_artifact_hash TEXT NOT NULL,
    analysis_enabled INTEGER NOT NULL CHECK (analysis_enabled = 0),
    execution_enabled INTEGER NOT NULL CHECK (execution_enabled = 0),
    orders_enabled INTEGER NOT NULL CHECK (orders_enabled = 0)
);

CREATE TABLE IF NOT EXISTS m1c_directionless_shadow_v0 (
    shadow_episode_id TEXT PRIMARY KEY,
    strategy_version TEXT NOT NULL CHECK (strategy_version = 'm1c_directionless_t20_d_v0'),
    run_id TEXT NOT NULL,
    m1c_episode_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    con_id INTEGER NOT NULL,
    session_date TEXT NOT NULL,
    t0_utc TEXT NOT NULL,
    p0 REAL NOT NULL CHECK (p0 > 0.0),
    movement_price REAL NOT NULL CHECK (movement_price > 0.0),
    family TEXT NOT NULL CHECK (family = 'HARD_M1C'),
    probability REAL NOT NULL,
    consumed_ratio REAL,
    upper_trigger REAL NOT NULL,
    lower_trigger REAL NOT NULL,
    state TEXT NOT NULL CHECK (state IN (
        'CREATED', 'WAITING_FOR_ENTRY', 'LONG_ACTIVE', 'SHORT_ACTIVE',
        'TARGET_EXITED', 'STOP_EXITED', 'TIME_EXITED', 'NO_ENTRY_TIMEOUT',
        'ENTRY_UNRESOLVED', 'ACTIVE_PATH_UNRESOLVED'
    )),
    direction TEXT CHECK (direction IS NULL OR direction IN ('LONG', 'SHORT')),
    entry_timestamp_utc TEXT,
    entry_price REAL,
    entry_source_event_id TEXT,
    stop_price REAL,
    target_price REAL,
    exit_timestamp_utc TEXT,
    exit_price REAL,
    exit_reason TEXT,
    exit_source_event_id TEXT,
    payload_json TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL,
    code_hash TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    m1c_artifact_hash TEXT NOT NULL,
    UNIQUE (run_id, m1c_episode_id, strategy_version)
);

CREATE INDEX IF NOT EXISTS idx_directionless_shadow_session_symbol_v0
ON m1c_directionless_shadow_v0(run_id, session_date, symbol, t0_utc);

CREATE INDEX IF NOT EXISTS idx_directionless_shadow_state_v0
ON m1c_directionless_shadow_v0(run_id, state, updated_at_utc);

CREATE TABLE IF NOT EXISTS m1c_directionless_shadow_transition_v0 (
    transition_id TEXT PRIMARY KEY,
    shadow_episode_id TEXT NOT NULL
        REFERENCES m1c_directionless_shadow_v0(shadow_episode_id),
    run_id TEXT NOT NULL,
    state TEXT NOT NULL,
    transition_timestamp_utc TEXT NOT NULL,
    source_event_id TEXT,
    source_partition_reference TEXT,
    connection_generation INTEGER,
    payload_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_directionless_shadow_transition_episode_v0
ON m1c_directionless_shadow_transition_v0(shadow_episode_id, transition_timestamp_utc);
