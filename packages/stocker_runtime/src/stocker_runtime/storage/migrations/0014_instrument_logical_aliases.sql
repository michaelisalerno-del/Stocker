DROP INDEX instruments_ibkr_con_id_idx;

CREATE INDEX instruments_ibkr_con_id_idx ON instruments(ibkr_con_id)
    WHERE ibkr_con_id IS NOT NULL;

CREATE TRIGGER instruments_ibkr_physical_identity_insert
BEFORE INSERT ON instruments
WHEN NEW.ibkr_con_id IS NOT NULL
BEGIN
    SELECT CASE WHEN NEW.option_strike IS NOT NULL
        AND stocker_same_positive_decimal(NEW.option_strike, NEW.option_strike) <> 1
        THEN RAISE(ABORT, 'instrument_ibkr_physical_identity_mismatch') END;
    SELECT CASE WHEN EXISTS (
        SELECT 1 FROM instruments existing
        WHERE existing.ibkr_con_id = NEW.ibkr_con_id
          AND existing.instrument_id <> NEW.instrument_id
          AND (
              existing.kind IS NOT NEW.kind
              OR existing.symbol IS NOT NEW.symbol
              OR existing.exchange IS NOT NEW.exchange
              OR existing.currency IS NOT NEW.currency
              OR existing.option_expiry IS NOT NEW.option_expiry
              OR (existing.option_strike IS NULL) IS NOT (NEW.option_strike IS NULL)
              OR (
                  existing.option_strike IS NOT NULL
                  AND NEW.option_strike IS NOT NULL
                  AND stocker_same_positive_decimal(
                      existing.option_strike,
                      NEW.option_strike
                  ) <> 1
              )
              OR existing.option_right IS NOT NEW.option_right
              OR existing.option_multiplier IS NOT NEW.option_multiplier
          )
    ) THEN RAISE(ABORT, 'instrument_ibkr_physical_identity_mismatch') END;
END;

CREATE TRIGGER instruments_ibkr_physical_identity_update
BEFORE UPDATE OF
    ibkr_con_id,
    kind,
    symbol,
    exchange,
    currency,
    option_expiry,
    option_strike,
    option_right,
    option_multiplier
ON instruments
WHEN NEW.ibkr_con_id IS NOT NULL
BEGIN
    SELECT CASE WHEN NEW.option_strike IS NOT NULL
        AND stocker_same_positive_decimal(NEW.option_strike, NEW.option_strike) <> 1
        THEN RAISE(ABORT, 'instrument_ibkr_physical_identity_mismatch') END;
    SELECT CASE WHEN EXISTS (
        SELECT 1 FROM instruments existing
        WHERE existing.ibkr_con_id = NEW.ibkr_con_id
          AND existing.instrument_id <> NEW.instrument_id
          AND (
              existing.kind IS NOT NEW.kind
              OR existing.symbol IS NOT NEW.symbol
              OR existing.exchange IS NOT NEW.exchange
              OR existing.currency IS NOT NEW.currency
              OR existing.option_expiry IS NOT NEW.option_expiry
              OR (existing.option_strike IS NULL) IS NOT (NEW.option_strike IS NULL)
              OR (
                  existing.option_strike IS NOT NULL
                  AND NEW.option_strike IS NOT NULL
                  AND stocker_same_positive_decimal(
                      existing.option_strike,
                      NEW.option_strike
                  ) <> 1
              )
              OR existing.option_right IS NOT NEW.option_right
              OR existing.option_multiplier IS NOT NEW.option_multiplier
          )
    ) THEN RAISE(ABORT, 'instrument_ibkr_physical_identity_mismatch') END;
END;
