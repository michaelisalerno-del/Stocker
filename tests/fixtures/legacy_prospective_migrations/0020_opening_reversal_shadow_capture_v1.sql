PRAGMA foreign_keys = ON;

-- V1.1 capture eligibility and scientific eligibility are deliberately
-- separate. Engineering-shadow rows may reach the exact frozen two-line
-- recorder, but only rows whose parent episode is scientifically valid enter
-- the scientific view.
DROP VIEW IF EXISTS opening_reversal_virtual_position_v1;
DROP TRIGGER IF EXISTS trg_opening_reversal_v1_1_discovery_guard;
DROP TRIGGER IF EXISTS trg_opening_reversal_v1_1_outcome_guard;
DROP VIEW IF EXISTS opening_reversal_v1_1_eligible_episode;

CREATE VIEW opening_reversal_v1_1_capture_eligible_episode AS
SELECT
    prediction.run_id,
    prediction.fresh_episode_id AS episode_id,
    prediction.receipt_hash_v1 AS prediction_receipt_hash_v1,
    prediction.session_date,
    prediction.stock,
    prediction.entry_timestamp_utc,
    activation.activation_receipt_hash AS activation_receipt_identity,
    barrier.audit_hash_v1_1 AS causal_barrier_audit_identity,
    promotion.promotion_hash_v1 AS promotion_identity,
    CASE
        WHEN prediction.scientific_outcome_eligible_v1 = 1
         AND episode.scientific_recording_valid = 1
            THEN 1
        ELSE 0
    END AS scientific_option_evidence
FROM opening_reversal_prediction_v1 AS prediction
JOIN opening_reversal_activation_v1 AS activation
  ON activation.run_id = prediction.run_id
 AND activation.experiment_id = prediction.experiment_id
 AND activation.experiment_version = prediction.experiment_version
JOIN opening_reversal_causal_barrier_audit_v1_1 AS barrier
  ON barrier.run_id = prediction.run_id
 AND barrier.session_date = prediction.session_date
 AND barrier.experiment_id = prediction.experiment_id
 AND barrier.experiment_version = prediction.experiment_version
 AND barrier.activation_receipt_hash_v1_1 =
     activation.activation_receipt_hash
 AND barrier.barrier_status = 'passed'
JOIN opening_reversal_promotion_v1 AS promotion
  ON promotion.run_id = prediction.run_id
 AND promotion.promoted_receipt_hash_v1 = prediction.receipt_hash_v1
JOIN m1c_episode_v0 AS episode
  ON episode.run_id = prediction.run_id
 AND episode.episode_id = prediction.fresh_episode_id
 AND episode.symbol = prediction.stock
 AND episode.session_date = prediction.session_date
 AND episode.prospective_entry_timestamp_utc =
     prediction.entry_timestamp_utc
WHERE prediction.experiment_id = 'm1c-prospective-opening-reversal-v1'
  AND prediction.experiment_version = '1.1'
  AND prediction.eligibility_v1 = 1
  AND prediction.fresh_episode_id IS NOT NULL
  AND activation.activation_timestamp_utc < prediction.entry_timestamp_utc
  AND activation.new_york_trading_date <= prediction.session_date
  AND barrier.nominal_entry_timestamp_utc =
      prediction.entry_timestamp_utc
  AND prediction.receipt_created_at_utc <=
      barrier.release_authorized_at_utc
  AND EXISTS (
      SELECT 1
      FROM json_each(barrier.prediction_receipt_hashes_json) AS receipt
      WHERE receipt.value = prediction.receipt_hash_v1
  )
  AND NOT EXISTS (
      SELECT 1
      FROM quiet_state_observation_v0 AS quiet
      WHERE quiet.run_id = prediction.run_id
        AND quiet.observation_id = prediction.fresh_episode_id
  );

CREATE VIEW opening_reversal_v1_1_eligible_episode AS
SELECT
    run_id,
    episode_id,
    prediction_receipt_hash_v1,
    session_date,
    stock,
    entry_timestamp_utc,
    activation_receipt_identity,
    causal_barrier_audit_identity,
    promotion_identity
FROM opening_reversal_v1_1_capture_eligible_episode
WHERE scientific_option_evidence = 1;

CREATE TRIGGER trg_opening_reversal_v1_1_discovery_guard
BEFORE INSERT ON opening_reversal_contract_discovery_v1
WHEN EXISTS (
    SELECT 1
    FROM opening_reversal_prediction_v1 AS prediction
    WHERE prediction.run_id = NEW.run_id
      AND prediction.fresh_episode_id = NEW.episode_id
      AND prediction.experiment_version = '1.1'
)
BEGIN
    SELECT CASE
        WHEN NEW.status = 'selected'
          AND (
              NEW.call_con_id IS NULL
              OR NEW.put_con_id IS NULL
              OR NEW.call_con_id = NEW.put_con_id
              OR NEW.expiry IS NULL
              OR NEW.strike IS NULL
              OR NEW.planned_live_market_data_lines <> 2
              OR NEW.live_market_data_lines_consumed <> 0
              OR NEW.full_chain_live_subscription_created <> 0
              OR NEW.metadata_request_ended <> 1
              OR NEW.missing_reason IS NOT NULL
          )
        THEN RAISE(ABORT, 'blocked_v1_1_option_protocol_not_exactly_two_lines')
    END;
    SELECT CASE
        WHEN NEW.status = 'failed'
          AND (
              NEW.call_con_id IS NOT NULL
              OR NEW.put_con_id IS NOT NULL
              OR NEW.expiry IS NOT NULL
              OR NEW.strike IS NOT NULL
              OR NEW.planned_live_market_data_lines <> 0
              OR NEW.live_market_data_lines_consumed <> 0
              OR NEW.full_chain_live_subscription_created <> 0
              OR NEW.metadata_request_ended <> 1
              OR NEW.missing_reason IS NULL
          )
        THEN RAISE(ABORT, 'blocked_v1_1_failed_discovery_contains_contract_allocation')
    END;
    SELECT CASE
        WHEN NEW.status NOT IN ('selected', 'failed')
        THEN RAISE(ABORT, 'blocked_v1_1_discovery_status_invalid')
    END;
    SELECT CASE
        WHEN NOT EXISTS (
            SELECT 1
            FROM opening_reversal_v1_1_capture_eligible_episode AS eligible
            WHERE eligible.run_id = NEW.run_id
              AND eligible.episode_id = NEW.episode_id
        )
        THEN RAISE(ABORT, 'blocked_v1_1_episode_not_eligible')
    END;
    SELECT CASE
        WHEN NEW.status = 'selected'
          AND NEW.expiry <> (
            SELECT date(eligible.session_date, '+1 day')
            FROM opening_reversal_v1_1_capture_eligible_episode AS eligible
            WHERE eligible.run_id = NEW.run_id
              AND eligible.episode_id = NEW.episode_id
        )
        THEN RAISE(ABORT, 'blocked_v1_1_primary_expiry_not_1dte')
    END;
END;

CREATE TRIGGER trg_opening_reversal_v1_1_outcome_guard
BEFORE INSERT ON opening_reversal_primary_option_outcome_v1
WHEN EXISTS (
    SELECT 1
    FROM opening_reversal_prediction_v1 AS prediction
    WHERE prediction.run_id = NEW.run_id
      AND prediction.receipt_hash_v1 = NEW.prediction_receipt_hash_v1
      AND prediction.experiment_version = '1.1'
)
BEGIN
    SELECT CASE
        WHEN NOT EXISTS (
            SELECT 1
            FROM opening_reversal_v1_1_capture_eligible_episode AS eligible
            WHERE eligible.run_id = NEW.run_id
              AND eligible.prediction_receipt_hash_v1 =
                  NEW.prediction_receipt_hash_v1
        )
        THEN RAISE(ABORT, 'blocked_v1_1_episode_not_eligible')
    END;
    SELECT CASE
        WHEN NOT EXISTS (
            SELECT 1
            FROM opening_reversal_contract_discovery_v1 AS discovery
            JOIN opening_reversal_v1_1_capture_eligible_episode AS eligible
              ON eligible.run_id = discovery.run_id
             AND eligible.episode_id = discovery.episode_id
            WHERE eligible.prediction_receipt_hash_v1 =
                  NEW.prediction_receipt_hash_v1
              AND discovery.status = 'selected'
              AND discovery.planned_live_market_data_lines = 2
              AND discovery.expiry = NEW.expiry
              AND discovery.strike = NEW.strike
              AND (
                  (NEW.right = 'C' AND discovery.call_con_id = NEW.con_id)
                  OR
                  (NEW.right = 'P' AND discovery.put_con_id = NEW.con_id)
              )
        )
        THEN RAISE(ABORT, 'blocked_v1_1_outcome_contract_not_selected_pair')
    END;
    SELECT CASE
        WHEN (
            SELECT COUNT(*)
            FROM opening_reversal_primary_option_outcome_v1 AS existing
            WHERE existing.run_id = NEW.run_id
              AND existing.prediction_receipt_hash_v1 =
                  NEW.prediction_receipt_hash_v1
        ) >= 2
        THEN RAISE(ABORT, 'blocked_v1_1_outcome_more_than_two_legs')
    END;
    SELECT CASE
        WHEN EXISTS (
            SELECT 1
            FROM opening_reversal_primary_option_outcome_v1 AS existing
            WHERE existing.run_id = NEW.run_id
              AND existing.prediction_receipt_hash_v1 =
                  NEW.prediction_receipt_hash_v1
              AND existing.right = NEW.right
        )
        THEN RAISE(ABORT, 'blocked_v1_1_outcome_duplicate_option_right')
    END;
    SELECT CASE
        WHEN (
            NEW.role = 'predicted_leg'
            AND NEW.right <> (
                SELECT CASE prediction.prediction_v1
                    WHEN 'CALL' THEN 'C'
                    WHEN 'PUT' THEN 'P'
                END
                FROM opening_reversal_prediction_v1 AS prediction
                WHERE prediction.run_id = NEW.run_id
                  AND prediction.receipt_hash_v1 =
                      NEW.prediction_receipt_hash_v1
            )
        )
        OR (
            NEW.role = 'opposite_leg'
            AND NEW.right = (
                SELECT CASE prediction.prediction_v1
                    WHEN 'CALL' THEN 'C'
                    WHEN 'PUT' THEN 'P'
                END
                FROM opening_reversal_prediction_v1 AS prediction
                WHERE prediction.run_id = NEW.run_id
                  AND prediction.receipt_hash_v1 =
                      NEW.prediction_receipt_hash_v1
            )
        )
        THEN RAISE(ABORT, 'blocked_v1_1_outcome_role_right_mismatch')
    END;
END;

CREATE VIEW opening_reversal_virtual_position_v1 AS
WITH primary_pair AS (
    SELECT
        run_id,
        prediction_receipt_hash_v1,
        COUNT(*) AS pair_outcome_count,
        SUM(complete) AS pair_complete_count,
        MAX(CASE WHEN role = 'predicted_leg' THEN 1 ELSE 0 END)
            AS predicted_outcome_present,
        MAX(CASE WHEN role = 'opposite_leg' THEN 1 ELSE 0 END)
            AS opposite_outcome_present,
        MAX(
            CASE
                WHEN complete = 0 THEN missing_reason
            END
        ) AS pair_missing_reason
    FROM opening_reversal_primary_option_outcome_v1
    GROUP BY run_id, prediction_receipt_hash_v1
)
SELECT
    'opening-reversal-v1.1:' || prediction.receipt_hash_v1
        AS virtual_position_id,
    'opening_reversal_v1_1' AS ledger_scope,
    eligible.run_id,
    prediction.experiment_id,
    prediction.experiment_version,
    eligible.activation_receipt_identity,
    eligible.causal_barrier_audit_identity,
    eligible.promotion_identity,
    prediction.receipt_hash_v1 AS prediction_receipt_hash_v1,
    eligible.episode_id,
    eligible.session_date,
    eligible.stock AS symbol,
    prediction.entry_timestamp_utc,
    prediction.prediction_v1 AS predicted_direction,
    CASE prediction.prediction_v1
        WHEN 'CALL' THEN 'C'
        WHEN 'PUT' THEN 'P'
    END AS right,
    'predicted_leg' AS role,
    CASE
        WHEN discovery.status = 'failed' THEN 'INVALID'
        WHEN COALESCE(pair.pair_outcome_count, 0) = 2
         AND COALESCE(pair.pair_complete_count, 0) = 2
         AND predicted.complete = 1
         AND predicted.entry_ask > 0
         AND predicted.exit_bid IS NOT NULL
            THEN 'CLOSED'
        WHEN COALESCE(pair.pair_outcome_count, 0) = 2 THEN 'INVALID'
        WHEN schedule.status IN ('rejected', 'expired', 'complete')
            THEN 'INVALID'
        WHEN discovery.status = 'selected' THEN 'CAPTURING'
        WHEN schedule.status = 'streaming' THEN 'CAPTURING'
        ELSE 'SCHEDULED'
    END AS lifecycle_state,
    CASE
        WHEN discovery.status = 'failed' THEN discovery.missing_reason
        WHEN COALESCE(pair.pair_outcome_count, 0) = 2
         AND COALESCE(pair.pair_complete_count, 0) = 2
         AND predicted.complete = 1
         AND predicted.entry_ask > 0
         AND predicted.exit_bid IS NOT NULL
            THEN NULL
        WHEN COALESCE(pair.pair_outcome_count, 0) = 2
         AND COALESCE(pair.pair_complete_count, 0) < 2
            THEN COALESCE(
                pair.pair_missing_reason,
                'primary_pair_outcome_incomplete'
            )
        WHEN COALESCE(pair.pair_outcome_count, 0) = 2
            THEN 'predicted_leg_quote_evidence_incomplete'
        WHEN schedule.status IN ('rejected', 'expired')
            THEN COALESCE(
                schedule.degradation_reason,
                'option_episode_' || schedule.status
            )
        WHEN schedule.status = 'complete'
            THEN 'primary_pair_outcome_missing_after_schedule_completion'
        WHEN discovery.status = 'selected'
         AND COALESCE(pair.pair_outcome_count, 0) = 1
         AND COALESCE(pair.predicted_outcome_present, 0) = 1
            THEN 'awaiting_opposite_control_outcome'
        WHEN discovery.status = 'selected'
         AND COALESCE(pair.pair_outcome_count, 0) = 1
         AND COALESCE(pair.opposite_outcome_present, 0) = 1
            THEN 'awaiting_predicted_leg_outcome'
        WHEN discovery.status = 'selected'
            THEN 'awaiting_frozen_primary_pair_bid_ask_outcomes'
        WHEN schedule.status = 'streaming'
            THEN 'awaiting_primary_pair_contract_discovery'
        ELSE 'awaiting_primary_1dte_common_strike'
    END AS status_reason,
    schedule.status AS option_schedule_status,
    CASE prediction.prediction_v1
        WHEN 'CALL' THEN discovery.call_con_id
        WHEN 'PUT' THEN discovery.put_con_id
    END AS con_id,
    discovery.expiry,
    CASE
        WHEN discovery.expiry IS NULL THEN NULL
        ELSE CAST(
            julianday(discovery.expiry) - julianday(eligible.session_date)
            AS INTEGER
        )
    END AS dte,
    discovery.strike,
    1 AS quantity,
    COALESCE(
        contract.multiplier,
        CAST(json_extract(predicted.outcome_json, '$.contract.multiplier') AS INTEGER)
    ) AS multiplier,
    discovery.planned_live_market_data_lines,
    COALESCE(pair.pair_outcome_count, 0) AS pair_outcome_count,
    COALESCE(pair.pair_complete_count, 0) AS pair_complete_count,
    COALESCE(pair.predicted_outcome_present, 0) AS predicted_outcome_present,
    COALESCE(pair.opposite_outcome_present, 0) AS opposite_outcome_present,
    predicted.entry_bid,
    predicted.entry_ask,
    predicted.entry_quote_timestamp_utc,
    predicted.exit_bid,
    predicted.exit_ask,
    predicted.exit_quote_timestamp_utc,
    predicted.conservative_return_v1,
    CASE
        WHEN predicted.entry_ask IS NULL
          OR predicted.exit_bid IS NULL
          OR COALESCE(
                contract.multiplier,
                CAST(
                    json_extract(
                        predicted.outcome_json,
                        '$.contract.multiplier'
                    ) AS INTEGER
                )
             ) IS NULL
            THEN NULL
        ELSE (
            predicted.exit_bid - predicted.entry_ask
        ) * COALESCE(
            contract.multiplier,
            CAST(json_extract(predicted.outcome_json, '$.contract.multiplier') AS INTEGER)
        )
    END AS gross_quote_pnl,
    quote.bid AS latest_observed_bid,
    quote.ask AS latest_observed_ask,
    quote.received_timestamp_utc AS latest_quote_received_at_utc,
    quote.recording_status AS latest_quote_recording_status,
    quote.quote_quality_flags_json AS latest_quote_quality_flags_json,
    'first_valid_live_ask_at_or_after_entry' AS entry_convention,
    'first_valid_live_bid_at_or_after_frozen_15m_horizon' AS exit_convention,
    eligible.scientific_option_evidence AS scientific_eligible,
    0 AS execution_claimed,
    0 AS paper_fill_claimed
FROM opening_reversal_v1_1_capture_eligible_episode AS eligible
JOIN opening_reversal_prediction_v1 AS prediction
  ON prediction.run_id = eligible.run_id
 AND prediction.receipt_hash_v1 = eligible.prediction_receipt_hash_v1
LEFT JOIN opening_reversal_contract_discovery_v1 AS discovery
  ON discovery.run_id = eligible.run_id
 AND discovery.episode_id = eligible.episode_id
LEFT JOIN option_episode_schedule_v0 AS schedule
  ON schedule.run_id = eligible.run_id
 AND schedule.episode_id = eligible.episode_id
LEFT JOIN primary_pair AS pair
  ON pair.run_id = eligible.run_id
 AND pair.prediction_receipt_hash_v1 = eligible.prediction_receipt_hash_v1
LEFT JOIN opening_reversal_primary_option_outcome_v1 AS predicted
  ON predicted.run_id = eligible.run_id
 AND predicted.prediction_receipt_hash_v1 =
     eligible.prediction_receipt_hash_v1
 AND predicted.role = 'predicted_leg'
LEFT JOIN episode_option_contract_v0 AS contract
  ON contract.run_id = eligible.run_id
 AND contract.episode_id = eligible.episode_id
 AND contract.con_id = CASE prediction.prediction_v1
        WHEN 'CALL' THEN discovery.call_con_id
        WHEN 'PUT' THEN discovery.put_con_id
     END
LEFT JOIN option_quote_state_v0 AS quote
  ON quote.option_contract_id = contract.id;
