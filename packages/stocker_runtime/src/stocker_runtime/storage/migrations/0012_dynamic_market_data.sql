ALTER TABLE subscriptions ADD COLUMN snapshot INTEGER NOT NULL DEFAULT 0
    CHECK(snapshot IN (0, 1));

CREATE TABLE market_data_interests (
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
    input_event_ids_json TEXT NOT NULL CHECK(
        stocker_canonical_json(input_event_ids_json) = 1
        AND json_type(input_event_ids_json) = 'array'
        AND json_array_length(input_event_ids_json) BETWEEN 1 AND 256
        AND length(CAST(input_event_ids_json AS BLOB)) <= 32768
    ),
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
    UNIQUE(instance_id, interest_key, as_of_at_us)
) STRICT;
CREATE INDEX market_data_interests_run_lifecycle_idx
    ON market_data_interests(run_id, lifecycle, priority DESC, interest_id);
CREATE INDEX market_data_interests_instance_expiry_idx
    ON market_data_interests(instance_id, expires_at_us, interest_id);
CREATE INDEX market_data_interests_instance_updated_idx
    ON market_data_interests(instance_id, updated_at_us DESC, interest_id);
CREATE INDEX market_data_interests_retention_idx
    ON market_data_interests(updated_at_us, interest_id)
    WHERE lifecycle IN ('fulfilled', 'denied', 'expired', 'cancelled');

CREATE TRIGGER market_data_interests_input_provenance_insert
BEFORE INSERT ON market_data_interests
BEGIN
    SELECT CASE WHEN EXISTS (
        SELECT value FROM json_each(NEW.input_event_ids_json)
        GROUP BY value HAVING count(*) > 1
    ) OR EXISTS (
        SELECT 1 FROM json_each(NEW.input_event_ids_json) input
        LEFT JOIN market_events event
          ON event.event_id=input.value AND event.run_id=NEW.run_id
        WHERE event.event_id IS NULL OR event.event_at_us>NEW.as_of_at_us
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
    required, priority, maximum_contracts, input_event_ids_json, content_hash, created_at_us
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

CREATE TABLE instrument_discovery_receipts (
    receipt_id TEXT PRIMARY KEY,
    interest_id TEXT NOT NULL UNIQUE
        REFERENCES market_data_interests(interest_id) ON DELETE CASCADE,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    instance_id TEXT NOT NULL REFERENCES idea_instances(instance_id),
    status TEXT NOT NULL CHECK(status IN ('resolved', 'denied')),
    reason_code TEXT CHECK(reason_code IS NULL OR length(reason_code) BETWEEN 1 AND 128),
    instrument_id TEXT REFERENCES instruments(instrument_id),
    expiry TEXT CHECK(
        expiry IS NULL OR (
            length(expiry) = 8 AND expiry NOT GLOB '*[^0-9]*'
        )
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
CREATE INDEX instrument_discovery_receipts_instance_time_idx
    ON instrument_discovery_receipts(instance_id, completed_at_us DESC, receipt_id);

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
