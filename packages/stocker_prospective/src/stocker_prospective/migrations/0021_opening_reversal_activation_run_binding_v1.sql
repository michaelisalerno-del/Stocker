PRAGMA foreign_keys = ON;

-- Activation receipts are immutable experiment identities, while a recorder
-- run is an operational lineage boundary. Rebuild the association table so an
-- audited replacement run can bind to the exact existing receipt bytes and
-- hash without manufacturing a new activation.
PRAGMA legacy_alter_table = ON;

ALTER TABLE opening_reversal_activation_v1
    RENAME TO opening_reversal_activation_v1_before_run_binding;

CREATE TABLE opening_reversal_activation_v1 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id INTEGER NOT NULL REFERENCES evidence_envelope(id),
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    experiment_id TEXT NOT NULL,
    experiment_version TEXT NOT NULL,
    activation_timestamp_utc TEXT NOT NULL,
    new_york_trading_date TEXT NOT NULL,
    configuration_hash TEXT NOT NULL,
    frozen_rule_hash TEXT NOT NULL,
    configured_reserved_line_count INTEGER NOT NULL
        CHECK (configured_reserved_line_count = 12),
    order_routing_disabled INTEGER NOT NULL
        CHECK (order_routing_disabled = 1),
    activation_receipt_hash TEXT NOT NULL,
    receipt_json TEXT NOT NULL,
    source_activation_id INTEGER REFERENCES opening_reversal_activation_v1(id),
    binding_kind TEXT NOT NULL
        CHECK (binding_kind IN ('original_activation', 'audited_run_rollover')),
    UNIQUE(run_id, experiment_id, experiment_version),
    UNIQUE(run_id, activation_receipt_hash),
    CHECK (
        (binding_kind = 'original_activation' AND source_activation_id IS NULL)
        OR
        (binding_kind = 'audited_run_rollover' AND source_activation_id IS NOT NULL)
    )
);

INSERT INTO opening_reversal_activation_v1(
    id,
    envelope_id,
    run_id,
    experiment_id,
    experiment_version,
    activation_timestamp_utc,
    new_york_trading_date,
    configuration_hash,
    frozen_rule_hash,
    configured_reserved_line_count,
    order_routing_disabled,
    activation_receipt_hash,
    receipt_json,
    source_activation_id,
    binding_kind
)
SELECT
    id,
    envelope_id,
    run_id,
    experiment_id,
    experiment_version,
    activation_timestamp_utc,
    new_york_trading_date,
    configuration_hash,
    frozen_rule_hash,
    configured_reserved_line_count,
    order_routing_disabled,
    activation_receipt_hash,
    receipt_json,
    NULL,
    'original_activation'
FROM opening_reversal_activation_v1_before_run_binding
ORDER BY id;

DROP TABLE opening_reversal_activation_v1_before_run_binding;

PRAGMA legacy_alter_table = OFF;

CREATE UNIQUE INDEX idx_opening_reversal_activation_v1_original_receipt
    ON opening_reversal_activation_v1(activation_receipt_hash)
    WHERE binding_kind = 'original_activation';

CREATE INDEX idx_opening_reversal_activation_v1_source
    ON opening_reversal_activation_v1(source_activation_id);

CREATE TRIGGER trg_opening_reversal_activation_v1_binding_guard
BEFORE INSERT ON opening_reversal_activation_v1
WHEN NEW.binding_kind = 'audited_run_rollover'
BEGIN
    SELECT CASE
        WHEN NOT EXISTS (
            SELECT 1
            FROM opening_reversal_activation_v1 AS source
            WHERE source.id = NEW.source_activation_id
              AND source.binding_kind = 'original_activation'
              AND source.run_id <> NEW.run_id
              AND source.experiment_id = NEW.experiment_id
              AND source.experiment_version = NEW.experiment_version
              AND source.activation_timestamp_utc =
                  NEW.activation_timestamp_utc
              AND source.new_york_trading_date =
                  NEW.new_york_trading_date
              AND source.configuration_hash = NEW.configuration_hash
              AND source.frozen_rule_hash = NEW.frozen_rule_hash
              AND source.configured_reserved_line_count =
                  NEW.configured_reserved_line_count
              AND source.order_routing_disabled =
                  NEW.order_routing_disabled
              AND source.activation_receipt_hash =
                  NEW.activation_receipt_hash
              AND source.receipt_json = NEW.receipt_json
        )
        THEN RAISE(
            ABORT,
            'blocked_opening_reversal_activation_binding_not_byte_identical'
        )
    END;
END;

CREATE TRIGGER trg_opening_reversal_activation_v1_no_update
BEFORE UPDATE ON opening_reversal_activation_v1
BEGIN
    SELECT RAISE(ABORT, 'opening reversal activation binding is immutable');
END;

CREATE TRIGGER trg_opening_reversal_activation_v1_no_delete
BEFORE DELETE ON opening_reversal_activation_v1
BEGIN
    SELECT RAISE(ABORT, 'opening reversal activation binding is append-only');
END;
