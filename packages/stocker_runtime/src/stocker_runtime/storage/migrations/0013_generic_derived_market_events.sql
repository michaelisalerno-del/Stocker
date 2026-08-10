CREATE TABLE market_events_v2 (
    event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    source_sequence INTEGER UNIQUE,
    derived_after_source_sequence INTEGER,
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
    bid_size_value REAL,
    ask_size_value REAL,
    last_value REAL,
    size_value REAL,
    payload_json TEXT NOT NULL CHECK(
        stocker_canonical_json(payload_json) = 1
        AND length(CAST(payload_json AS BLOB)) <= 65536
    ),
    payload_sha256 TEXT NOT NULL CHECK(length(payload_sha256) = 64),
    CHECK(
        (source_sequence IS NOT NULL AND derived_after_source_sequence IS NULL)
        OR
        (source_sequence IS NULL AND derived_after_source_sequence IS NOT NULL
            AND event_kind IN (
                'bar_5m',
                'bar_5m_session_prefix',
                'session_volume_baseline',
                'option_snapshot_capture'
            ))
    ),
    UNIQUE(event_id, run_id, instrument_id, feed_kind)
) STRICT;

CREATE TABLE market_event_derivations_v2 (
    derived_event_id TEXT NOT NULL
        REFERENCES market_events_v2(event_id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED,
    input_event_id TEXT NOT NULL REFERENCES market_events_v2(event_id),
    input_ordinal INTEGER NOT NULL CHECK(input_ordinal >= 0),
    input_role TEXT NOT NULL CHECK(
        input_role IN ('constituent', 'progress', 'prior_receipt', 'context', 'completion')
    ),
    created_at_us INTEGER NOT NULL CHECK(created_at_us >= 0),
    PRIMARY KEY(derived_event_id, input_ordinal),
    UNIQUE(derived_event_id, input_event_id)
) STRICT;

CREATE TABLE market_latest_v2 (
    run_id TEXT NOT NULL,
    instrument_id TEXT NOT NULL REFERENCES instruments(instrument_id),
    feed_kind TEXT NOT NULL,
    event_id TEXT NOT NULL UNIQUE REFERENCES market_events_v2(event_id),
    event_at_us INTEGER NOT NULL,
    received_at_us INTEGER NOT NULL,
    event_kind TEXT NOT NULL,
    quality_bits INTEGER NOT NULL CHECK(quality_bits >= 0),
    bid_value REAL,
    bid_source_event_id TEXT REFERENCES market_events_v2(event_id),
    ask_value REAL,
    ask_source_event_id TEXT REFERENCES market_events_v2(event_id),
    bid_size_value REAL,
    bid_size_source_event_id TEXT REFERENCES market_events_v2(event_id),
    ask_size_value REAL,
    ask_size_source_event_id TEXT REFERENCES market_events_v2(event_id),
    last_value REAL,
    last_source_event_id TEXT REFERENCES market_events_v2(event_id),
    size_value REAL,
    size_source_event_id TEXT REFERENCES market_events_v2(event_id),
    close_value REAL,
    close_source_event_id TEXT REFERENCES market_events_v2(event_id),
    PRIMARY KEY(instrument_id, feed_kind),
    FOREIGN KEY(event_id, run_id, instrument_id, feed_kind)
        REFERENCES market_events_v2(event_id, run_id, instrument_id, feed_kind)
) STRICT;

CREATE TABLE market_data_interests_v2 (
    interest_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    instance_id TEXT NOT NULL REFERENCES idea_instances(instance_id),
    interest_key TEXT NOT NULL CHECK(length(interest_key) BETWEEN 1 AND 128),
    underlying_instrument_id TEXT NOT NULL REFERENCES instruments(instrument_id),
    asset_kind TEXT NOT NULL CHECK(asset_kind = 'option'),
    minimum_days_to_expiry INTEGER NOT NULL CHECK(minimum_days_to_expiry BETWEEN 0 AND 365),
    maximum_days_to_expiry INTEGER NOT NULL CHECK(
        maximum_days_to_expiry BETWEEN minimum_days_to_expiry AND 365
    ),
    option_right TEXT NOT NULL CHECK(option_right IN ('call', 'put')),
    strike_offset INTEGER NOT NULL CHECK(strike_offset BETWEEN -32 AND 32),
    reference_price REAL NOT NULL CHECK(reference_price > 0),
    feed_kind TEXT NOT NULL CHECK(feed_kind = 'quotes'),
    cadence TEXT NOT NULL CHECK(cadence IN ('snapshot', 'stream')),
    as_of_at_us INTEGER NOT NULL CHECK(as_of_at_us >= 0),
    expires_at_us INTEGER NOT NULL CHECK(
        expires_at_us > as_of_at_us AND expires_at_us - as_of_at_us <= 604800000000
    ),
    required INTEGER NOT NULL CHECK(required IN (0, 1)),
    priority INTEGER NOT NULL CHECK(priority BETWEEN 0 AND 1000),
    maximum_contracts INTEGER NOT NULL CHECK(maximum_contracts = 1),
    input_event_id TEXT NOT NULL REFERENCES market_events_v2(event_id),
    content_hash TEXT NOT NULL CHECK(length(content_hash) = 64),
    lifecycle TEXT NOT NULL CHECK(
        lifecycle IN (
            'pending', 'resolved', 'active', 'fulfilled', 'denied', 'expired', 'cancelled'
        )
    ),
    reason_code TEXT CHECK(reason_code IS NULL OR length(reason_code) BETWEEN 1 AND 128),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts BETWEEN 0 AND 5),
    next_attempt_at_us INTEGER NOT NULL CHECK(next_attempt_at_us >= 0),
    created_at_us INTEGER NOT NULL CHECK(created_at_us >= 0),
    updated_at_us INTEGER NOT NULL CHECK(updated_at_us >= created_at_us),
    bound_subscription_id TEXT REFERENCES subscriptions(subscription_id) ON DELETE SET NULL,
    UNIQUE(instance_id, interest_key, as_of_at_us)
) STRICT;

CREATE TABLE instrument_discovery_receipts_v2 (
    receipt_id TEXT PRIMARY KEY,
    interest_id TEXT NOT NULL UNIQUE
        REFERENCES market_data_interests_v2(interest_id) ON DELETE CASCADE,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    instance_id TEXT NOT NULL REFERENCES idea_instances(instance_id),
    status TEXT NOT NULL CHECK(status IN ('resolved', 'denied')),
    reason_code TEXT CHECK(reason_code IS NULL OR length(reason_code) BETWEEN 1 AND 128),
    instrument_id TEXT REFERENCES instruments(instrument_id),
    expiry TEXT CHECK(
        expiry IS NULL OR (length(expiry) = 8 AND expiry NOT GLOB '*[^0-9]*')
    ),
    strike REAL CHECK(strike IS NULL OR strike > 0),
    option_right TEXT CHECK(option_right IS NULL OR option_right IN ('call', 'put')),
    multiplier TEXT CHECK(multiplier IS NULL OR length(multiplier) BETWEEN 1 AND 16),
    candidates_inspected INTEGER NOT NULL CHECK(candidates_inspected BETWEEN 0 AND 4096),
    completed_at_us INTEGER NOT NULL CHECK(completed_at_us >= 0),
    CHECK(
        (status = 'resolved' AND reason_code IS NULL AND instrument_id IS NOT NULL
            AND expiry IS NOT NULL AND strike IS NOT NULL AND option_right IS NOT NULL
            AND multiplier IS NOT NULL)
        OR
        (status = 'denied' AND reason_code IS NOT NULL AND instrument_id IS NULL
            AND expiry IS NULL AND strike IS NULL AND option_right IS NULL
            AND multiplier IS NULL)
    )
) STRICT;

INSERT INTO market_events_v2 SELECT * FROM market_events;
INSERT INTO market_event_derivations_v2 SELECT * FROM market_event_derivations;
INSERT INTO market_latest_v2 SELECT * FROM market_latest;
INSERT INTO market_data_interests_v2 SELECT * FROM market_data_interests;
INSERT INTO instrument_discovery_receipts_v2 SELECT * FROM instrument_discovery_receipts;

PRAGMA legacy_alter_table = ON;

DROP TABLE instrument_discovery_receipts;
DROP TABLE market_data_interests;
DROP TABLE market_event_derivations;
DROP TABLE market_latest;

ALTER TABLE market_events RENAME TO market_events_v1;
ALTER TABLE market_events_v2 RENAME TO market_events;
ALTER TABLE market_event_derivations_v2 RENAME TO market_event_derivations;
ALTER TABLE market_latest_v2 RENAME TO market_latest;
ALTER TABLE market_data_interests_v2 RENAME TO market_data_interests;
ALTER TABLE instrument_discovery_receipts_v2 RENAME TO instrument_discovery_receipts;
DROP TABLE market_events_v1;

PRAGMA legacy_alter_table = OFF;

CREATE INDEX market_events_run_time_idx ON market_events(run_id, event_at_us DESC, event_id);
CREATE INDEX market_events_instrument_kind_time_idx
    ON market_events(instrument_id, event_kind, event_at_us DESC, event_id);
CREATE INDEX market_events_run_kind_time_idx
    ON market_events(run_id, event_kind, event_at_us DESC, event_id);
CREATE INDEX market_events_retention_idx ON market_events(event_at_us, event_kind, event_id);
CREATE INDEX market_events_causal_sequence_idx ON market_events(
    run_id, coalesce(source_sequence, derived_after_source_sequence), event_id
);
CREATE INDEX market_events_shadow_raw_idx
    ON market_events(run_id, instrument_id, event_kind, source_sequence, event_id);
CREATE INDEX market_event_derivations_input_idx
    ON market_event_derivations(input_event_id, derived_event_id);
CREATE INDEX market_event_derivations_retention_idx
    ON market_event_derivations(created_at_us, derived_event_id, input_ordinal);
CREATE INDEX market_latest_event_time_idx ON market_latest(event_at_us DESC, instrument_id);
CREATE INDEX market_latest_run_event_time_idx
    ON market_latest(run_id, event_at_us DESC, event_id);
CREATE INDEX market_data_interests_run_lifecycle_idx
    ON market_data_interests(run_id, lifecycle, priority DESC, interest_id);
CREATE INDEX market_data_interests_instance_expiry_idx
    ON market_data_interests(instance_id, expires_at_us, interest_id);
CREATE INDEX market_data_interests_instance_updated_idx
    ON market_data_interests(instance_id, updated_at_us DESC, interest_id);
CREATE INDEX market_data_interests_input_event_idx
    ON market_data_interests(input_event_id, interest_id);
CREATE INDEX market_data_interests_bound_subscription_idx
    ON market_data_interests(bound_subscription_id, lifecycle, interest_id)
    WHERE bound_subscription_id IS NOT NULL;
CREATE INDEX market_data_interests_retention_idx
    ON market_data_interests(updated_at_us, interest_id)
    WHERE lifecycle IN ('fulfilled', 'denied', 'expired', 'cancelled');
CREATE INDEX instrument_discovery_receipts_instance_time_idx
    ON instrument_discovery_receipts(instance_id, completed_at_us DESC, receipt_id);

CREATE TRIGGER market_events_callback_provenance_insert
BEFORE INSERT ON market_events
WHEN NEW.source_sequence IS NOT NULL
BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM callback_inbox callback
        WHERE callback.source_sequence = NEW.source_sequence
          AND callback.run_id = NEW.run_id
    ) THEN RAISE(ABORT, 'market_event_callback_provenance_mismatch') END;
END;
CREATE TRIGGER market_events_callback_provenance_update
BEFORE UPDATE OF run_id, source_sequence ON market_events
WHEN NEW.source_sequence IS NOT NULL
BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM callback_inbox callback
        WHERE callback.source_sequence = NEW.source_sequence
          AND callback.run_id = NEW.run_id
    ) THEN RAISE(ABORT, 'market_event_callback_provenance_mismatch') END;
END;
CREATE TRIGGER market_events_immutable_update
BEFORE UPDATE ON market_events
BEGIN
    SELECT RAISE(ABORT, 'market_event_immutable');
END;

CREATE TRIGGER market_event_derivations_provenance_insert
BEFORE INSERT ON market_event_derivations
BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM market_events derived
        JOIN market_events input ON input.event_id=NEW.input_event_id
        WHERE derived.event_id=NEW.derived_event_id
          AND derived.source_sequence IS NULL
          AND derived.derived_after_source_sequence IS NOT NULL
          AND input.event_id!=derived.event_id
          AND input.run_id=derived.run_id
          AND input.instrument_id=derived.instrument_id
          AND input.feed_kind=derived.feed_kind
          AND coalesce(input.source_sequence, input.derived_after_source_sequence)
              <=derived.derived_after_source_sequence
          AND input.received_at_us<=derived.received_at_us
          AND (
              (derived.event_kind='bar_5m'
                  AND input.event_kind='bar'
                  AND NEW.input_role IN ('constituent', 'progress')
                  AND (
                      (NEW.input_role='constituent'
                          AND input.event_at_us>=derived.event_at_us-300000000
                          AND input.event_at_us<derived.event_at_us)
                      OR
                      (NEW.input_role='progress' AND input.event_at_us>=derived.event_at_us)
                  ))
              OR
              (derived.event_kind='bar_5m_session_prefix'
                  AND input.event_at_us<=derived.event_at_us
                  AND (
                      (NEW.input_role='constituent' AND input.event_kind='bar_5m')
                      OR
                      (NEW.input_role='prior_receipt'
                          AND input.event_kind='bar_5m_session_prefix')
                      OR
                      (NEW.input_role='context'
                          AND input.event_kind='session_volume_baseline')
                  ))
              OR
              (derived.event_kind='session_volume_baseline'
                  AND input.event_at_us<=derived.event_at_us
                  AND (
                      (NEW.input_role='constituent'
                          AND input.event_kind='bar_5m_session_prefix')
                      OR
                      (NEW.input_role='prior_receipt'
                          AND input.event_kind='session_volume_baseline')
                  ))
              OR
              (derived.event_kind='option_snapshot_capture'
                  AND input.event_at_us<=derived.event_at_us
                  AND (
                      (NEW.input_role='constituent'
                          AND input.event_kind IN ('quote', 'option_computation'))
                      OR
                      (NEW.input_role='completion'
                          AND input.event_kind='option_snapshot_end')
                  ))
          )
    ) THEN RAISE(ABORT, 'market_event_derivation_provenance_mismatch') END;
END;
CREATE TRIGGER market_event_derivations_immutable_update
BEFORE UPDATE ON market_event_derivations
BEGIN
    SELECT RAISE(ABORT, 'market_event_derivation_immutable');
END;

CREATE TRIGGER market_latest_field_provenance_insert
BEFORE INSERT ON market_latest
BEGIN
    SELECT CASE WHEN
        (NEW.bid_value IS NULL) != (NEW.bid_source_event_id IS NULL)
        OR (NEW.ask_value IS NULL) != (NEW.ask_source_event_id IS NULL)
        OR (NEW.bid_size_value IS NULL) != (NEW.bid_size_source_event_id IS NULL)
        OR (NEW.ask_size_value IS NULL) != (NEW.ask_size_source_event_id IS NULL)
        OR (NEW.last_value IS NULL) != (NEW.last_source_event_id IS NULL)
        OR (NEW.size_value IS NULL) != (NEW.size_source_event_id IS NULL)
        OR (NEW.close_value IS NULL) != (NEW.close_source_event_id IS NULL)
    THEN RAISE(ABORT, 'market_latest_field_provenance_missing') END;
    SELECT CASE WHEN
        (NEW.bid_source_event_id IS NOT NULL AND NOT EXISTS (
            SELECT 1 FROM market_events event WHERE event.event_id=NEW.bid_source_event_id
              AND event.run_id=NEW.run_id AND event.instrument_id=NEW.instrument_id
              AND event.feed_kind=NEW.feed_kind AND event.bid_value IS NEW.bid_value))
        OR (NEW.ask_source_event_id IS NOT NULL AND NOT EXISTS (
            SELECT 1 FROM market_events event WHERE event.event_id=NEW.ask_source_event_id
              AND event.run_id=NEW.run_id AND event.instrument_id=NEW.instrument_id
              AND event.feed_kind=NEW.feed_kind AND event.ask_value IS NEW.ask_value))
        OR (NEW.bid_size_source_event_id IS NOT NULL AND NOT EXISTS (
            SELECT 1 FROM market_events event WHERE event.event_id=NEW.bid_size_source_event_id
              AND event.run_id=NEW.run_id AND event.instrument_id=NEW.instrument_id
              AND event.feed_kind=NEW.feed_kind
              AND event.bid_size_value IS NEW.bid_size_value))
        OR (NEW.ask_size_source_event_id IS NOT NULL AND NOT EXISTS (
            SELECT 1 FROM market_events event WHERE event.event_id=NEW.ask_size_source_event_id
              AND event.run_id=NEW.run_id AND event.instrument_id=NEW.instrument_id
              AND event.feed_kind=NEW.feed_kind
              AND event.ask_size_value IS NEW.ask_size_value))
        OR (NEW.last_source_event_id IS NOT NULL AND NOT EXISTS (
            SELECT 1 FROM market_events event WHERE event.event_id=NEW.last_source_event_id
              AND event.run_id=NEW.run_id AND event.instrument_id=NEW.instrument_id
              AND event.feed_kind=NEW.feed_kind AND event.last_value IS NEW.last_value))
        OR (NEW.size_source_event_id IS NOT NULL AND NOT EXISTS (
            SELECT 1 FROM market_events event WHERE event.event_id=NEW.size_source_event_id
              AND event.run_id=NEW.run_id AND event.instrument_id=NEW.instrument_id
              AND event.feed_kind=NEW.feed_kind AND event.size_value IS NEW.size_value))
        OR (NEW.close_source_event_id IS NOT NULL AND NOT EXISTS (
            SELECT 1 FROM market_events event WHERE event.event_id=NEW.close_source_event_id
              AND event.run_id=NEW.run_id AND event.instrument_id=NEW.instrument_id
              AND event.feed_kind=NEW.feed_kind AND event.close_value IS NEW.close_value))
    THEN RAISE(ABORT, 'market_latest_field_provenance_mismatch') END;
END;
CREATE TRIGGER market_latest_field_provenance_update
BEFORE UPDATE ON market_latest
BEGIN
    SELECT CASE WHEN
        (NEW.bid_value IS NULL) != (NEW.bid_source_event_id IS NULL)
        OR (NEW.ask_value IS NULL) != (NEW.ask_source_event_id IS NULL)
        OR (NEW.bid_size_value IS NULL) != (NEW.bid_size_source_event_id IS NULL)
        OR (NEW.ask_size_value IS NULL) != (NEW.ask_size_source_event_id IS NULL)
        OR (NEW.last_value IS NULL) != (NEW.last_source_event_id IS NULL)
        OR (NEW.size_value IS NULL) != (NEW.size_source_event_id IS NULL)
        OR (NEW.close_value IS NULL) != (NEW.close_source_event_id IS NULL)
    THEN RAISE(ABORT, 'market_latest_field_provenance_missing') END;
    SELECT CASE WHEN
        (NEW.bid_source_event_id IS NOT NULL AND NOT EXISTS (
            SELECT 1 FROM market_events event WHERE event.event_id=NEW.bid_source_event_id
              AND event.run_id=NEW.run_id AND event.instrument_id=NEW.instrument_id
              AND event.feed_kind=NEW.feed_kind AND event.bid_value IS NEW.bid_value))
        OR (NEW.ask_source_event_id IS NOT NULL AND NOT EXISTS (
            SELECT 1 FROM market_events event WHERE event.event_id=NEW.ask_source_event_id
              AND event.run_id=NEW.run_id AND event.instrument_id=NEW.instrument_id
              AND event.feed_kind=NEW.feed_kind AND event.ask_value IS NEW.ask_value))
        OR (NEW.bid_size_source_event_id IS NOT NULL AND NOT EXISTS (
            SELECT 1 FROM market_events event WHERE event.event_id=NEW.bid_size_source_event_id
              AND event.run_id=NEW.run_id AND event.instrument_id=NEW.instrument_id
              AND event.feed_kind=NEW.feed_kind
              AND event.bid_size_value IS NEW.bid_size_value))
        OR (NEW.ask_size_source_event_id IS NOT NULL AND NOT EXISTS (
            SELECT 1 FROM market_events event WHERE event.event_id=NEW.ask_size_source_event_id
              AND event.run_id=NEW.run_id AND event.instrument_id=NEW.instrument_id
              AND event.feed_kind=NEW.feed_kind
              AND event.ask_size_value IS NEW.ask_size_value))
        OR (NEW.last_source_event_id IS NOT NULL AND NOT EXISTS (
            SELECT 1 FROM market_events event WHERE event.event_id=NEW.last_source_event_id
              AND event.run_id=NEW.run_id AND event.instrument_id=NEW.instrument_id
              AND event.feed_kind=NEW.feed_kind AND event.last_value IS NEW.last_value))
        OR (NEW.size_source_event_id IS NOT NULL AND NOT EXISTS (
            SELECT 1 FROM market_events event WHERE event.event_id=NEW.size_source_event_id
              AND event.run_id=NEW.run_id AND event.instrument_id=NEW.instrument_id
              AND event.feed_kind=NEW.feed_kind AND event.size_value IS NEW.size_value))
        OR (NEW.close_source_event_id IS NOT NULL AND NOT EXISTS (
            SELECT 1 FROM market_events event WHERE event.event_id=NEW.close_source_event_id
              AND event.run_id=NEW.run_id AND event.instrument_id=NEW.instrument_id
              AND event.feed_kind=NEW.feed_kind AND event.close_value IS NEW.close_value))
    THEN RAISE(ABORT, 'market_latest_field_provenance_mismatch') END;
END;

CREATE TRIGGER market_data_interests_input_provenance_insert
BEFORE INSERT ON market_data_interests
BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM market_events event
        WHERE event.event_id=NEW.input_event_id AND event.run_id=NEW.run_id
          AND event.event_at_us<=NEW.as_of_at_us
    ) OR NOT EXISTS (
        SELECT 1 FROM idea_instances instance
        WHERE instance.instance_id=NEW.instance_id AND instance.run_id=NEW.run_id
    ) OR NOT EXISTS (
        SELECT 1 FROM idea_instances instance, json_each(instance.universe_json) member
        WHERE instance.instance_id=NEW.instance_id
          AND member.value=NEW.underlying_instrument_id
    ) OR NOT EXISTS (
        SELECT 1 FROM instruments instrument
        WHERE instrument.instrument_id=NEW.underlying_instrument_id
          AND instrument.kind='stock'
    ) THEN RAISE(ABORT, 'market_data_interest_provenance_mismatch') END;
END;
CREATE TRIGGER market_data_interests_identity_immutable
BEFORE UPDATE OF
    interest_id, run_id, instance_id, interest_key, underlying_instrument_id,
    asset_kind, minimum_days_to_expiry, maximum_days_to_expiry, option_right,
    strike_offset, reference_price, feed_kind, cadence, as_of_at_us, expires_at_us,
    required, priority, maximum_contracts, input_event_id, content_hash, created_at_us
ON market_data_interests
BEGIN
    SELECT RAISE(ABORT, 'market_data_interest_identity_immutable');
END;
CREATE TRIGGER market_data_interests_lifecycle_update
BEFORE UPDATE OF lifecycle, attempts, next_attempt_at_us ON market_data_interests
BEGIN
    SELECT CASE WHEN
        (OLD.lifecycle='pending' AND NEW.lifecycle NOT IN (
            'pending', 'resolved', 'denied', 'expired', 'cancelled'
        )) OR
        (OLD.lifecycle='resolved' AND NEW.lifecycle NOT IN (
            'resolved', 'active', 'fulfilled', 'expired', 'cancelled'
        )) OR
        (OLD.lifecycle='active' AND NEW.lifecycle NOT IN (
            'active', 'resolved', 'fulfilled', 'expired', 'cancelled'
        )) OR
        (OLD.lifecycle IN ('fulfilled', 'denied', 'expired', 'cancelled')
            AND NEW.lifecycle != OLD.lifecycle) OR
        NEW.next_attempt_at_us > NEW.expires_at_us
    THEN RAISE(ABORT, 'market_data_interest_lifecycle_invalid') END;
END;
CREATE TRIGGER market_data_interests_subscription_binding_update
BEFORE UPDATE OF bound_subscription_id ON market_data_interests
WHEN NEW.bound_subscription_id IS NOT NULL
BEGIN
    SELECT CASE WHEN NEW.lifecycle NOT IN ('resolved', 'active') OR NOT EXISTS (
        SELECT 1 FROM subscriptions subscription
        JOIN runtime_state state
          ON state.run_id=subscription.run_id
         AND state.recorder_generation=subscription.recorder_generation
         AND state.connection_generation=subscription.connection_generation
        JOIN instrument_discovery_receipts receipt
          ON receipt.interest_id=NEW.interest_id
         AND receipt.status='resolved'
         AND receipt.instrument_id=subscription.instrument_id
        WHERE subscription.subscription_id=NEW.bound_subscription_id
          AND subscription.run_id=NEW.run_id
          AND subscription.feed_kind=NEW.feed_kind
          AND subscription.lifecycle IN ('connecting', 'active', 'degraded', 'cancelling')
          AND (NEW.cadence='snapshot' OR subscription.snapshot=0)
    ) THEN RAISE(ABORT, 'market_data_interest_subscription_binding_invalid') END;
END;

CREATE TRIGGER instrument_discovery_receipts_scope_insert
BEFORE INSERT ON instrument_discovery_receipts
BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM market_data_interests interest
        WHERE interest.interest_id=NEW.interest_id
          AND interest.run_id=NEW.run_id
          AND interest.instance_id=NEW.instance_id
          AND NEW.completed_at_us BETWEEN interest.as_of_at_us AND interest.expires_at_us
    ) OR (
        NEW.status='resolved' AND NOT EXISTS (
            SELECT 1 FROM instruments instrument
            WHERE instrument.instrument_id=NEW.instrument_id
              AND instrument.kind='option'
              AND instrument.option_expiry=NEW.expiry
              AND CAST(instrument.option_strike AS REAL)=NEW.strike
              AND instrument.option_right=NEW.option_right
              AND instrument.option_multiplier=NEW.multiplier
        )
    ) THEN RAISE(ABORT, 'instrument_discovery_receipt_scope_mismatch') END;
END;
CREATE TRIGGER instrument_discovery_receipts_immutable_update
BEFORE UPDATE ON instrument_discovery_receipts
BEGIN
    SELECT RAISE(ABORT, 'instrument_discovery_receipt_immutable');
END;
