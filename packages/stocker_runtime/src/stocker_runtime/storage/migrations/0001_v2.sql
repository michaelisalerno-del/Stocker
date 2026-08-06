CREATE TABLE schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    sha256 TEXT NOT NULL CHECK(length(sha256) = 64),
    applied_at_us INTEGER NOT NULL CHECK(applied_at_us >= 0)
) STRICT;

CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    mode TEXT NOT NULL CHECK(mode IN ('prospective_record', 'shadow')),
    source TEXT NOT NULL CHECK(source = 'ibkr'),
    started_at_us INTEGER NOT NULL CHECK(started_at_us >= 0),
    ended_at_us INTEGER CHECK(ended_at_us IS NULL OR ended_at_us >= started_at_us),
    config_hash TEXT NOT NULL CHECK(length(config_hash) = 64),
    git_commit TEXT NOT NULL,
    data_class TEXT NOT NULL CHECK(data_class IN ('prospective_protected', 'shadow_protected')),
    status TEXT NOT NULL CHECK(status IN ('created', 'running', 'stopped', 'fatal')),
    prior_run_id TEXT REFERENCES runs(run_id),
    CHECK(
        (mode = 'prospective_record' AND data_class = 'prospective_protected')
        OR (mode = 'shadow' AND data_class = 'shadow_protected')
    ),
    UNIQUE(run_id, mode, data_class)
) STRICT;

CREATE TABLE recorder_generations (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    generation INTEGER NOT NULL CHECK(generation >= 0),
    owner_id TEXT NOT NULL,
    started_at_us INTEGER NOT NULL CHECK(started_at_us >= 0),
    ended_at_us INTEGER,
    clean_stop INTEGER NOT NULL DEFAULT 0 CHECK(clean_stop IN (0, 1)),
    termination_code TEXT,
    PRIMARY KEY(run_id, generation)
) STRICT;

CREATE TABLE runtime_state (
    run_id TEXT PRIMARY KEY REFERENCES runs(run_id),
    recorder_generation INTEGER NOT NULL CHECK(recorder_generation >= 0),
    lifecycle TEXT NOT NULL,
    reason TEXT,
    process_heartbeat_at_us INTEGER,
    callback_heartbeat_at_us INTEGER,
    admission_heartbeat_at_us INTEGER,
    projection_heartbeat_at_us INTEGER,
    connection_generation INTEGER NOT NULL DEFAULT 0 CHECK(connection_generation >= 0),
    inbox_nonterminal_count INTEGER NOT NULL DEFAULT 0 CHECK(inbox_nonterminal_count >= 0),
    inbox_bytes INTEGER NOT NULL DEFAULT 0 CHECK(inbox_bytes >= 0),
    database_bytes INTEGER NOT NULL DEFAULT 0 CHECK(database_bytes >= 0),
    wal_bytes INTEGER NOT NULL DEFAULT 0 CHECK(wal_bytes >= 0),
    order_capability_observed INTEGER NOT NULL DEFAULT 0 CHECK(order_capability_observed = 0),
    FOREIGN KEY(run_id, recorder_generation)
        REFERENCES recorder_generations(run_id, generation)
) STRICT;

CREATE TABLE incidents (
    incident_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    scope TEXT NOT NULL,
    severity TEXT NOT NULL CHECK(severity IN ('info', 'degraded', 'fatal')),
    code TEXT NOT NULL,
    plugin_instance_id TEXT,
    subscription_id TEXT,
    opened_at_us INTEGER NOT NULL CHECK(opened_at_us >= 0),
    resolved_at_us INTEGER,
    details_json TEXT NOT NULL CHECK(
        stocker_canonical_json(details_json) = 1 AND length(CAST(details_json AS BLOB)) <= 16384
    )
) STRICT;
CREATE INDEX incidents_run_opened_idx ON incidents(run_id, opened_at_us DESC, incident_id);
CREATE INDEX incidents_unresolved_idx ON incidents(run_id, severity, opened_at_us)
    WHERE resolved_at_us IS NULL;
CREATE INDEX incidents_retention_idx
    ON incidents(resolved_at_us, incident_id) WHERE resolved_at_us IS NOT NULL;

CREATE TABLE gaps (
    gap_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    subscription_id TEXT,
    started_at_us INTEGER NOT NULL CHECK(started_at_us >= 0),
    ended_at_us INTEGER,
    reason TEXT NOT NULL,
    data_loss_possible INTEGER NOT NULL CHECK(data_loss_possible IN (0, 1)),
    continuity_required INTEGER NOT NULL CHECK(continuity_required IN (0, 1)),
    resolved_at_us INTEGER
) STRICT;
CREATE INDEX gaps_run_started_idx ON gaps(run_id, started_at_us DESC, gap_id);
CREATE INDEX gaps_unresolved_idx ON gaps(run_id, started_at_us) WHERE resolved_at_us IS NULL;
CREATE INDEX gaps_retention_idx
    ON gaps(resolved_at_us, gap_id) WHERE resolved_at_us IS NOT NULL;

CREATE TABLE instruments (
    instrument_id TEXT PRIMARY KEY,
    identity_hash TEXT NOT NULL UNIQUE CHECK(length(identity_hash) = 64),
    ibkr_con_id INTEGER,
    kind TEXT NOT NULL,
    symbol TEXT NOT NULL,
    exchange TEXT NOT NULL,
    currency TEXT NOT NULL,
    option_expiry TEXT,
    option_strike TEXT,
    option_right TEXT CHECK(option_right IS NULL OR option_right IN ('call', 'put')),
    option_multiplier TEXT
) STRICT;
CREATE UNIQUE INDEX instruments_ibkr_con_id_idx ON instruments(ibkr_con_id)
    WHERE ibkr_con_id IS NOT NULL;
CREATE INDEX instruments_symbol_idx ON instruments(symbol, kind, instrument_id);

CREATE TABLE subscriptions (
    subscription_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    recorder_generation INTEGER NOT NULL,
    connection_generation INTEGER NOT NULL CHECK(connection_generation >= 0),
    instrument_id TEXT NOT NULL REFERENCES instruments(instrument_id),
    feed_kind TEXT NOT NULL,
    request_id INTEGER NOT NULL,
    lifecycle TEXT NOT NULL,
    requirements_hash TEXT NOT NULL CHECK(length(requirements_hash) = 64),
    opened_at_us INTEGER NOT NULL,
    closed_at_us INTEGER,
    latest_event_id TEXT,
    FOREIGN KEY(run_id, recorder_generation)
        REFERENCES recorder_generations(run_id, generation),
    UNIQUE(run_id, connection_generation, request_id)
) STRICT;
CREATE INDEX subscriptions_run_lifecycle_idx
    ON subscriptions(run_id, lifecycle, subscription_id);
CREATE INDEX subscriptions_instrument_feed_idx
    ON subscriptions(instrument_id, feed_kind, opened_at_us DESC);
CREATE INDEX subscriptions_retention_idx
    ON subscriptions(closed_at_us, subscription_id) WHERE closed_at_us IS NOT NULL;

CREATE TABLE callback_inbox (
    source_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    event_uid TEXT NOT NULL UNIQUE,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    recorder_generation INTEGER NOT NULL,
    connection_generation INTEGER NOT NULL CHECK(connection_generation >= 0),
    request_id INTEGER,
    callback_kind TEXT NOT NULL,
    received_at_us INTEGER NOT NULL CHECK(received_at_us >= 0),
    provider_at_us INTEGER,
    payload_json TEXT CHECK(
        payload_json IS NULL OR (
            stocker_canonical_json(payload_json) = 1 AND length(CAST(payload_json AS BLOB)) <= 65536
        )
    ),
    payload_sha256 TEXT NOT NULL CHECK(length(payload_sha256) = 64),
    lifecycle TEXT NOT NULL CHECK(lifecycle IN ('pending', 'leased', 'acknowledged', 'failed')),
    lease_owner TEXT,
    lease_expires_at_us INTEGER,
    attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts >= 0),
    failure_code TEXT,
    normalized_event_id TEXT,
    acknowledged_at_us INTEGER,
    receipt_batch_id TEXT,
    FOREIGN KEY(run_id, recorder_generation)
        REFERENCES recorder_generations(run_id, generation)
) STRICT;
CREATE INDEX callback_inbox_lease_idx
    ON callback_inbox(lifecycle, lease_expires_at_us, source_sequence);
CREATE INDEX callback_inbox_run_sequence_idx ON callback_inbox(run_id, source_sequence);
CREATE INDEX callback_inbox_terminal_idx
    ON callback_inbox(acknowledged_at_us, source_sequence)
    WHERE lifecycle = 'acknowledged' AND payload_json IS NOT NULL;
CREATE INDEX callback_inbox_failed_payload_idx
    ON callback_inbox(received_at_us, source_sequence)
    WHERE lifecycle = 'failed' AND payload_json IS NOT NULL;
CREATE INDEX callback_inbox_ack_tombstone_idx
    ON callback_inbox(acknowledged_at_us, source_sequence)
    WHERE lifecycle = 'acknowledged' AND payload_json IS NULL;
CREATE INDEX callback_inbox_failed_tombstone_idx
    ON callback_inbox(received_at_us, source_sequence)
    WHERE lifecycle = 'failed' AND payload_json IS NULL;

CREATE TABLE callback_receipts (
    batch_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    first_source_sequence INTEGER NOT NULL,
    last_source_sequence INTEGER NOT NULL CHECK(last_source_sequence >= first_source_sequence),
    callback_count INTEGER NOT NULL CHECK(callback_count > 0),
    first_received_at_us INTEGER NOT NULL,
    last_received_at_us INTEGER NOT NULL,
    kind_counts_json TEXT NOT NULL CHECK(
        stocker_canonical_json(kind_counts_json) = 1 AND length(CAST(kind_counts_json AS BLOB)) <= 16384
    ),
    status_counts_json TEXT NOT NULL CHECK(
        stocker_canonical_json(status_counts_json) = 1 AND length(CAST(status_counts_json AS BLOB)) <= 16384
    ),
    callback_rows_hash TEXT NOT NULL CHECK(length(callback_rows_hash) = 64),
    prior_chain_hash TEXT NOT NULL CHECK(length(prior_chain_hash) = 64),
    chained_payload_hash TEXT NOT NULL CHECK(length(chained_payload_hash) = 64),
    first_normalized_event_id TEXT,
    last_normalized_event_id TEXT,
    created_at_us INTEGER NOT NULL,
    UNIQUE(run_id, first_source_sequence, last_source_sequence)
) STRICT;
CREATE INDEX callback_receipts_run_sequence_idx
    ON callback_receipts(run_id, last_source_sequence, batch_id);
CREATE INDEX callback_receipts_run_created_idx
    ON callback_receipts(run_id, created_at_us, batch_id);

CREATE TABLE callback_compaction_watermarks (
    run_id TEXT PRIMARY KEY REFERENCES runs(run_id),
    compacted_through_sequence INTEGER NOT NULL CHECK(compacted_through_sequence >= 0),
    cumulative_callback_count INTEGER NOT NULL CHECK(cumulative_callback_count >= 0),
    first_received_at_us INTEGER,
    last_received_at_us INTEGER,
    rolled_receipt_chain_hash TEXT NOT NULL CHECK(length(rolled_receipt_chain_hash) = 64),
    last_receipt_chain_hash TEXT NOT NULL CHECK(length(last_receipt_chain_hash) = 64),
    updated_at_us INTEGER NOT NULL
) STRICT;

CREATE TABLE market_events (
    event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    source_sequence INTEGER NOT NULL UNIQUE,
    instrument_id TEXT NOT NULL REFERENCES instruments(instrument_id),
    feed_kind TEXT NOT NULL,
    event_kind TEXT NOT NULL,
    event_at_us INTEGER NOT NULL CHECK(event_at_us >= 0),
    received_at_us INTEGER NOT NULL CHECK(received_at_us >= 0),
    connection_generation INTEGER NOT NULL CHECK(connection_generation >= 0),
    quality_bits INTEGER NOT NULL DEFAULT 0 CHECK(quality_bits >= 0),
    open_value REAL,
    high_value REAL,
    low_value REAL,
    close_value REAL,
    volume_value REAL,
    bid_value REAL,
    ask_value REAL,
    last_value REAL,
    size_value REAL,
    payload_json TEXT NOT NULL CHECK(
        stocker_canonical_json(payload_json) = 1 AND length(CAST(payload_json AS BLOB)) <= 65536
    ),
    payload_sha256 TEXT NOT NULL CHECK(length(payload_sha256) = 64),
    UNIQUE(event_id, run_id, instrument_id, feed_kind)
) STRICT;
CREATE INDEX market_events_run_time_idx ON market_events(run_id, event_at_us DESC, event_id);
CREATE INDEX market_events_instrument_kind_time_idx
    ON market_events(instrument_id, event_kind, event_at_us DESC, event_id);
CREATE INDEX market_events_run_kind_time_idx
    ON market_events(run_id, event_kind, event_at_us DESC, event_id);
CREATE INDEX market_events_retention_idx ON market_events(event_at_us, event_kind, event_id);

CREATE TRIGGER market_events_callback_provenance_insert
BEFORE INSERT ON market_events
BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM callback_inbox callback
        WHERE callback.source_sequence = NEW.source_sequence
          AND callback.run_id = NEW.run_id
    ) THEN RAISE(ABORT, 'market_event_callback_provenance_mismatch') END;
END;
CREATE TRIGGER market_events_callback_provenance_update
BEFORE UPDATE OF run_id, source_sequence ON market_events
BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM callback_inbox callback
        WHERE callback.source_sequence = NEW.source_sequence
          AND callback.run_id = NEW.run_id
    ) THEN RAISE(ABORT, 'market_event_callback_provenance_mismatch') END;
END;

CREATE TRIGGER callback_inbox_provenance_update
BEFORE UPDATE OF run_id ON callback_inbox
WHEN EXISTS (SELECT 1 FROM market_events event
    WHERE event.source_sequence = OLD.source_sequence)
BEGIN
    SELECT CASE WHEN EXISTS (
        SELECT 1 FROM market_events event
        WHERE event.source_sequence = OLD.source_sequence
          AND event.run_id != NEW.run_id
    ) THEN RAISE(ABORT, 'callback_market_event_provenance_mismatch') END;
END;

CREATE TRIGGER callback_inbox_acknowledged_event_insert
BEFORE INSERT ON callback_inbox
WHEN NEW.lifecycle = 'acknowledged'
BEGIN
    SELECT CASE WHEN NEW.normalized_event_id IS NULL OR NOT EXISTS (
        SELECT 1 FROM market_events event
        WHERE event.event_id = NEW.normalized_event_id
          AND event.run_id = NEW.run_id
          AND event.source_sequence = NEW.source_sequence
    ) THEN RAISE(ABORT, 'callback_acknowledgement_event_mismatch') END;
END;
CREATE TRIGGER callback_inbox_acknowledged_event_update
BEFORE UPDATE ON callback_inbox
WHEN NEW.lifecycle = 'acknowledged'
BEGIN
    SELECT CASE WHEN NEW.normalized_event_id IS NULL OR NOT EXISTS (
        SELECT 1 FROM market_events event
        WHERE event.event_id = NEW.normalized_event_id
          AND event.run_id = NEW.run_id
          AND event.source_sequence = NEW.source_sequence
    ) THEN RAISE(ABORT, 'callback_acknowledgement_event_mismatch') END;
END;

CREATE TRIGGER subscriptions_latest_event_provenance_insert
BEFORE INSERT ON subscriptions
WHEN NEW.latest_event_id IS NOT NULL
BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM market_events event
        WHERE event.event_id = NEW.latest_event_id
          AND event.run_id = NEW.run_id
          AND event.instrument_id = NEW.instrument_id
          AND event.feed_kind = NEW.feed_kind
    ) THEN RAISE(ABORT, 'subscription_latest_event_provenance_mismatch') END;
END;
CREATE TRIGGER subscriptions_latest_event_provenance_update
BEFORE UPDATE OF run_id, instrument_id, feed_kind, latest_event_id ON subscriptions
WHEN NEW.latest_event_id IS NOT NULL
BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM market_events event
        WHERE event.event_id = NEW.latest_event_id
          AND event.run_id = NEW.run_id
          AND event.instrument_id = NEW.instrument_id
          AND event.feed_kind = NEW.feed_kind
    ) THEN RAISE(ABORT, 'subscription_latest_event_provenance_mismatch') END;
END;

CREATE TABLE market_latest (
    run_id TEXT NOT NULL,
    instrument_id TEXT NOT NULL REFERENCES instruments(instrument_id),
    feed_kind TEXT NOT NULL,
    event_id TEXT NOT NULL UNIQUE REFERENCES market_events(event_id),
    event_at_us INTEGER NOT NULL,
    received_at_us INTEGER NOT NULL,
    event_kind TEXT NOT NULL,
    quality_bits INTEGER NOT NULL CHECK(quality_bits >= 0),
    bid_value REAL,
    ask_value REAL,
    last_value REAL,
    close_value REAL,
    PRIMARY KEY(instrument_id, feed_kind),
    FOREIGN KEY(event_id, run_id, instrument_id, feed_kind)
        REFERENCES market_events(event_id, run_id, instrument_id, feed_kind)
) STRICT;
CREATE INDEX market_latest_event_time_idx ON market_latest(event_at_us DESC, instrument_id);

CREATE TABLE idea_plugins (
    idea_id TEXT NOT NULL,
    idea_version TEXT NOT NULL,
    api_version INTEGER NOT NULL CHECK(api_version = 1),
    display_name TEXT NOT NULL,
    description TEXT NOT NULL,
    manifest_hash TEXT NOT NULL CHECK(length(manifest_hash) = 64),
    code_hash TEXT NOT NULL CHECK(length(code_hash) = 64),
    manifest_json TEXT NOT NULL CHECK(
        stocker_canonical_json(manifest_json) = 1 AND length(CAST(manifest_json AS BLOB)) <= 65536
    ),
    discovered_at_us INTEGER NOT NULL,
    PRIMARY KEY(idea_id, idea_version)
) STRICT;

CREATE TABLE idea_instances (
    instance_id TEXT PRIMARY KEY,
    idea_id TEXT NOT NULL,
    idea_version TEXT NOT NULL,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    mode TEXT NOT NULL CHECK(mode IN ('prospective_record', 'shadow')),
    parameters_json TEXT NOT NULL CHECK(
        stocker_canonical_json(parameters_json) = 1 AND length(CAST(parameters_json AS BLOB)) <= 65536
    ),
    parameters_hash TEXT NOT NULL CHECK(length(parameters_hash) = 64),
    activated_at_us INTEGER NOT NULL,
    deactivated_at_us INTEGER,
    health TEXT NOT NULL CHECK(health IN ('healthy', 'degraded', 'disabled')),
    error_code TEXT,
    data_class TEXT NOT NULL CHECK(data_class IN ('prospective_protected', 'shadow_protected')),
    FOREIGN KEY(idea_id, idea_version) REFERENCES idea_plugins(idea_id, idea_version),
    FOREIGN KEY(run_id, mode, data_class) REFERENCES runs(run_id, mode, data_class),
    CHECK(
        (mode = 'prospective_record' AND data_class = 'prospective_protected')
        OR (mode = 'shadow' AND data_class = 'shadow_protected')
    ),
    UNIQUE(run_id, idea_id, idea_version, parameters_hash, activated_at_us),
    UNIQUE(instance_id, run_id, data_class)
) STRICT;
CREATE INDEX idea_instances_run_health_idx ON idea_instances(run_id, health, instance_id);

CREATE TABLE idea_checkpoints (
    instance_id TEXT PRIMARY KEY REFERENCES idea_instances(instance_id),
    last_market_event_id TEXT,
    state_json TEXT NOT NULL CHECK(
        stocker_canonical_json(state_json) = 1 AND length(CAST(state_json AS BLOB)) <= 65536
    ),
    state_hash TEXT NOT NULL CHECK(length(state_hash) = 64),
    last_success_at_us INTEGER,
    updated_at_us INTEGER NOT NULL,
    consecutive_failures INTEGER NOT NULL DEFAULT 0 CHECK(consecutive_failures >= 0)
) STRICT;

CREATE TRIGGER idea_checkpoints_event_provenance_insert
BEFORE INSERT ON idea_checkpoints
WHEN NEW.last_market_event_id IS NOT NULL
BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM idea_instances instance
        JOIN market_events event ON event.event_id = NEW.last_market_event_id
        WHERE instance.instance_id = NEW.instance_id
          AND event.run_id = instance.run_id
    ) THEN RAISE(ABORT, 'idea_checkpoint_event_provenance_mismatch') END;
END;
CREATE TRIGGER idea_checkpoints_event_provenance_update
BEFORE UPDATE OF instance_id, last_market_event_id ON idea_checkpoints
WHEN NEW.last_market_event_id IS NOT NULL
BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM idea_instances instance
        JOIN market_events event ON event.event_id = NEW.last_market_event_id
        WHERE instance.instance_id = NEW.instance_id
          AND event.run_id = instance.run_id
    ) THEN RAISE(ABORT, 'idea_checkpoint_event_provenance_mismatch') END;
END;

CREATE TABLE idea_outputs (
    output_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    instance_id TEXT NOT NULL REFERENCES idea_instances(instance_id),
    output_kind TEXT NOT NULL CHECK(output_kind IN (
        'observation', 'signal', 'proposed_position', 'proposed_trade'
    )),
    subject_instrument_id TEXT NOT NULL REFERENCES instruments(instrument_id),
    emitted_at_us INTEGER NOT NULL,
    as_of_at_us INTEGER NOT NULL,
    valid_until_at_us INTEGER,
    direction TEXT,
    strength REAL,
    confidence REAL,
    horizon_us INTEGER,
    first_input_event_id TEXT NOT NULL,
    last_input_event_id TEXT NOT NULL,
    output_ordinal INTEGER NOT NULL CHECK(output_ordinal >= 0),
    payload_json TEXT NOT NULL CHECK(
        stocker_canonical_json(payload_json) = 1 AND length(CAST(payload_json AS BLOB)) <= 16384
    ),
    payload_hash TEXT NOT NULL CHECK(length(payload_hash) = 64),
    content_hash TEXT NOT NULL CHECK(length(content_hash) = 64),
    data_class TEXT NOT NULL CHECK(data_class IN ('prospective_protected', 'shadow_protected')),
    authority_status TEXT NOT NULL CHECK(
        (output_kind IN ('observation', 'signal') AND authority_status = 'recorded')
        OR
        (output_kind IN ('proposed_position', 'proposed_trade')
            AND authority_status = 'unapproved')
    ),
    FOREIGN KEY(instance_id, run_id, data_class)
        REFERENCES idea_instances(instance_id, run_id, data_class),
    UNIQUE(output_id, run_id, instance_id, data_class)
) STRICT;

CREATE INDEX idea_outputs_run_kind_time_idx
    ON idea_outputs(run_id, output_kind, as_of_at_us DESC, output_id);
CREATE INDEX idea_outputs_instance_kind_time_idx
    ON idea_outputs(instance_id, output_kind, as_of_at_us DESC, output_id);
CREATE INDEX idea_outputs_instrument_time_idx
    ON idea_outputs(subject_instrument_id, as_of_at_us DESC, output_id);
CREATE INDEX idea_outputs_retention_idx ON idea_outputs(emitted_at_us, output_id);

CREATE TRIGGER idea_outputs_event_provenance_insert
BEFORE INSERT ON idea_outputs
BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM market_events first_event
        JOIN market_events last_event ON last_event.event_id = NEW.last_input_event_id
        WHERE first_event.event_id = NEW.first_input_event_id
          AND first_event.run_id = NEW.run_id
          AND last_event.run_id = NEW.run_id
    ) THEN RAISE(ABORT, 'idea_output_event_provenance_mismatch') END;
END;
CREATE TRIGGER idea_outputs_event_provenance_update
BEFORE UPDATE OF run_id, first_input_event_id, last_input_event_id ON idea_outputs
BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM market_events first_event
        JOIN market_events last_event ON last_event.event_id = NEW.last_input_event_id
        WHERE first_event.event_id = NEW.first_input_event_id
          AND first_event.run_id = NEW.run_id
          AND last_event.run_id = NEW.run_id
    ) THEN RAISE(ABORT, 'idea_output_event_provenance_mismatch') END;
END;

CREATE TABLE idea_output_legs (
    output_id TEXT NOT NULL REFERENCES idea_outputs(output_id) ON DELETE CASCADE,
    leg_number INTEGER NOT NULL CHECK(leg_number >= 0),
    instrument_id TEXT NOT NULL REFERENCES instruments(instrument_id),
    action TEXT NOT NULL,
    target TEXT NOT NULL,
    quantity_value REAL,
    notional_value REAL,
    currency TEXT,
    price_hint REAL,
    PRIMARY KEY(output_id, leg_number)
) STRICT;

CREATE TABLE shadow_positions (
    position_id TEXT PRIMARY KEY,
    proposed_trade_output_id TEXT NOT NULL UNIQUE,
    run_id TEXT NOT NULL,
    instance_id TEXT NOT NULL,
    opened_at_us INTEGER,
    closed_at_us INTEGER,
    lifecycle TEXT NOT NULL CHECK(lifecycle IN ('pending', 'open', 'closed', 'invalid')),
    cost_model_id TEXT NOT NULL,
    fill_model_id TEXT NOT NULL,
    currency TEXT NOT NULL,
    invalid_reason TEXT,
    data_class TEXT NOT NULL CHECK(data_class = 'shadow_protected'),
    FOREIGN KEY(instance_id, run_id, data_class)
        REFERENCES idea_instances(instance_id, run_id, data_class),
    FOREIGN KEY(proposed_trade_output_id, run_id, instance_id, data_class)
        REFERENCES idea_outputs(output_id, run_id, instance_id, data_class)
) STRICT;
CREATE INDEX shadow_positions_run_lifecycle_idx
    ON shadow_positions(run_id, lifecycle, opened_at_us DESC, position_id);
CREATE INDEX shadow_positions_instance_time_idx
    ON shadow_positions(instance_id, opened_at_us DESC, position_id);
CREATE INDEX shadow_positions_retention_idx
    ON shadow_positions(closed_at_us, position_id) WHERE closed_at_us IS NOT NULL;

CREATE TRIGGER shadow_positions_proposed_trade_insert
BEFORE INSERT ON shadow_positions
BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM idea_outputs output
        WHERE output.output_id = NEW.proposed_trade_output_id
          AND output.run_id = NEW.run_id
          AND output.instance_id = NEW.instance_id
          AND output.data_class = NEW.data_class
          AND output.output_kind = 'proposed_trade'
          AND output.authority_status = 'unapproved'
    ) THEN RAISE(ABORT, 'shadow_position_proposal_provenance_mismatch') END;
END;
CREATE TRIGGER shadow_positions_proposed_trade_update
BEFORE UPDATE OF proposed_trade_output_id, run_id, instance_id, data_class ON shadow_positions
BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM idea_outputs output
        WHERE output.output_id = NEW.proposed_trade_output_id
          AND output.run_id = NEW.run_id
          AND output.instance_id = NEW.instance_id
          AND output.data_class = NEW.data_class
          AND output.output_kind = 'proposed_trade'
          AND output.authority_status = 'unapproved'
    ) THEN RAISE(ABORT, 'shadow_position_proposal_provenance_mismatch') END;
END;

CREATE TABLE shadow_legs (
    position_id TEXT NOT NULL REFERENCES shadow_positions(position_id) ON DELETE CASCADE,
    leg_number INTEGER NOT NULL CHECK(leg_number >= 0),
    instrument_id TEXT NOT NULL REFERENCES instruments(instrument_id),
    side TEXT NOT NULL CHECK(side IN ('buy', 'sell')),
    quantity REAL NOT NULL CHECK(quantity > 0),
    entry_market_event_id TEXT,
    entry_price REAL,
    exit_market_event_id TEXT,
    exit_price REAL,
    PRIMARY KEY(position_id, leg_number)
) STRICT;

CREATE TRIGGER shadow_legs_event_provenance_insert
BEFORE INSERT ON shadow_legs
BEGIN
    SELECT CASE WHEN NEW.entry_market_event_id IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM shadow_positions position
        JOIN market_events event ON event.event_id = NEW.entry_market_event_id
        WHERE position.position_id = NEW.position_id
          AND event.run_id = position.run_id
          AND event.instrument_id = NEW.instrument_id
    ) THEN RAISE(ABORT, 'shadow_leg_entry_event_provenance_mismatch') END;
    SELECT CASE WHEN NEW.exit_market_event_id IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM shadow_positions position
        JOIN market_events event ON event.event_id = NEW.exit_market_event_id
        WHERE position.position_id = NEW.position_id
          AND event.run_id = position.run_id
          AND event.instrument_id = NEW.instrument_id
    ) THEN RAISE(ABORT, 'shadow_leg_exit_event_provenance_mismatch') END;
END;
CREATE TRIGGER shadow_legs_entry_event_provenance_update
BEFORE UPDATE OF position_id, instrument_id, entry_market_event_id ON shadow_legs
WHEN NEW.entry_market_event_id IS NOT NULL AND (
    NEW.entry_market_event_id IS NOT OLD.entry_market_event_id
    OR NEW.position_id IS NOT OLD.position_id
    OR NEW.instrument_id IS NOT OLD.instrument_id
)
BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM shadow_positions position
        JOIN market_events event ON event.event_id = NEW.entry_market_event_id
        WHERE position.position_id = NEW.position_id
          AND event.run_id = position.run_id
          AND event.instrument_id = NEW.instrument_id
    ) THEN RAISE(ABORT, 'shadow_leg_entry_event_provenance_mismatch') END;
END;
CREATE TRIGGER shadow_legs_exit_event_provenance_update
BEFORE UPDATE OF position_id, instrument_id, exit_market_event_id ON shadow_legs
WHEN NEW.exit_market_event_id IS NOT NULL AND (
    NEW.exit_market_event_id IS NOT OLD.exit_market_event_id
    OR NEW.position_id IS NOT OLD.position_id
    OR NEW.instrument_id IS NOT OLD.instrument_id
)
BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM shadow_positions position
        JOIN market_events event ON event.event_id = NEW.exit_market_event_id
        WHERE position.position_id = NEW.position_id
          AND event.run_id = position.run_id
          AND event.instrument_id = NEW.instrument_id
    ) THEN RAISE(ABORT, 'shadow_leg_exit_event_provenance_mismatch') END;
END;

CREATE TABLE shadow_marks (
    position_id TEXT NOT NULL REFERENCES shadow_positions(position_id) ON DELETE CASCADE,
    marked_at_us INTEGER NOT NULL,
    gross_value REAL,
    gross_pnl REAL,
    net_pnl REAL,
    return_value REAL,
    quality_bits INTEGER NOT NULL DEFAULT 0 CHECK(quality_bits >= 0),
    payload_json TEXT NOT NULL CHECK(
        stocker_canonical_json(payload_json) = 1 AND length(CAST(payload_json AS BLOB)) <= 16384
    ),
    PRIMARY KEY(position_id, marked_at_us)
) STRICT;
CREATE INDEX shadow_marks_time_idx ON shadow_marks(marked_at_us DESC, position_id);

CREATE TABLE shadow_outcomes (
    position_id TEXT PRIMARY KEY REFERENCES shadow_positions(position_id) ON DELETE CASCADE,
    outcome_at_us INTEGER NOT NULL,
    reason TEXT NOT NULL,
    gross_pnl REAL,
    net_pnl REAL,
    return_value REAL,
    mfe REAL,
    mae REAL,
    completeness TEXT NOT NULL CHECK(completeness IN ('complete', 'incomplete', 'invalid')),
    payload_json TEXT NOT NULL CHECK(
        stocker_canonical_json(payload_json) = 1 AND length(CAST(payload_json AS BLOB)) <= 16384
    )
) STRICT;
CREATE INDEX shadow_outcomes_time_idx ON shadow_outcomes(outcome_at_us DESC, position_id);

CREATE TABLE migration_manifests (
    migration_id TEXT PRIMARY KEY,
    source_database_hash TEXT NOT NULL CHECK(length(source_database_hash) = 64),
    source_schema_digest TEXT NOT NULL CHECK(length(source_schema_digest) = 64),
    importer_version TEXT NOT NULL,
    started_at_us INTEGER NOT NULL,
    completed_at_us INTEGER,
    source_row_count INTEGER NOT NULL CHECK(source_row_count >= 0),
    imported_row_count INTEGER NOT NULL CHECK(imported_row_count >= 0),
    omitted_row_count INTEGER NOT NULL CHECK(omitted_row_count >= 0),
    target_digest TEXT CHECK(target_digest IS NULL OR length(target_digest) = 64),
    verification_status TEXT NOT NULL CHECK(verification_status IN ('pending', 'verified', 'failed'))
) STRICT;
