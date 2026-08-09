PRAGMA foreign_keys = ON;

-- These are read projections over immutable experiment evidence.  They do not
-- create fills, orders, account state, or a second mutable source of truth.
CREATE VIEW IF NOT EXISTS opening_reversal_virtual_position_v1 AS
WITH primary_pair AS (
    SELECT
        run_id,
        prediction_receipt_hash_v1,
        COUNT(*) AS pair_outcome_count,
        SUM(complete) AS pair_complete_count,
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
            THEN 'CLOSED'
        WHEN COALESCE(pair.pair_outcome_count, 0) = 2 THEN 'INVALID'
        WHEN discovery.status = 'selected' THEN 'CAPTURING'
        ELSE 'SCHEDULED'
    END AS lifecycle_state,
    CASE
        WHEN discovery.status = 'failed' THEN discovery.missing_reason
        WHEN COALESCE(pair.pair_outcome_count, 0) = 2
         AND COALESCE(pair.pair_complete_count, 0) = 2
         AND predicted.complete = 1
            THEN NULL
        WHEN COALESCE(pair.pair_outcome_count, 0) = 2
         AND COALESCE(pair.pair_complete_count, 0) < 2
            THEN COALESCE(
                pair.pair_missing_reason,
                'primary_pair_outcome_incomplete'
            )
        WHEN discovery.status = 'selected'
         AND COALESCE(pair.pair_outcome_count, 0) = 1
            THEN 'awaiting_opposite_control_outcome'
        WHEN discovery.status = 'selected'
            THEN 'awaiting_frozen_primary_pair_bid_ask_outcomes'
        ELSE 'awaiting_primary_1dte_common_strike'
    END AS status_reason,
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
    'last_valid_live_bid_at_or_before_frozen_15m_horizon' AS exit_convention,
    1 AS scientific_eligible,
    0 AS execution_claimed,
    0 AS paper_fill_claimed
FROM opening_reversal_v1_1_eligible_episode AS eligible
JOIN opening_reversal_prediction_v1 AS prediction
  ON prediction.run_id = eligible.run_id
 AND prediction.receipt_hash_v1 = eligible.prediction_receipt_hash_v1
LEFT JOIN opening_reversal_contract_discovery_v1 AS discovery
  ON discovery.run_id = eligible.run_id
 AND discovery.episode_id = eligible.episode_id
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

CREATE VIEW IF NOT EXISTS quiet_state_virtual_position_v1 AS
SELECT
    'quiet-short-premium:'
        || outcome.observation_id || ':'
        || outcome.structure_type || ':'
        || outcome.dte_bucket || ':'
        || outcome.horizon_label AS virtual_position_id,
    'quiet_state_short_premium' AS ledger_scope,
    outcome.run_id,
    outcome.observation_id,
    observation.observation_kind,
    observation.session_date,
    observation.symbol,
    observation.trigger_timestamp_utc,
    observation.prospective_entry_timestamp_utc AS entry_timestamp_utc,
    outcome.structure_type,
    outcome.dte_bucket,
    outcome.horizon_label,
    outcome.horizon_minutes,
    CASE
        WHEN outcome.attempted = 1
         AND outcome.complete_quote_quality = 1
            THEN 'CLOSED'
        ELSE 'INVALID'
    END AS lifecycle_state,
    CASE
        WHEN outcome.attempted = 0 THEN 'structure_not_attempted'
        WHEN outcome.complete_quote_quality = 0
            THEN 'incomplete_quote_quality'
        ELSE NULL
    END AS status_reason,
    outcome.opening_credit_or_debit AS opening_net_credit,
    CAST(json_extract(outcome.payload_json, '$.closing_debit') AS REAL)
        AS closing_net_debit,
    outcome.conservative_pnl,
    CAST(
        json_extract(outcome.payload_json, '$.configured_commission_pnl')
        AS REAL
    ) AS configured_commission_pnl,
    outcome.maximum_defined_risk,
    outcome.return_on_maximum_risk,
    outcome.short_strike_touched,
    outcome.protective_wing_touched,
    outcome.attempted,
    outcome.complete_quote_quality,
    outcome.strict_quote_quality,
    outcome.quality_status,
    outcome.quality_flags_json,
    json_extract(outcome.payload_json, '$.legs') AS legs_json,
    COALESCE(json_array_length(outcome.payload_json, '$.legs'), 0)
        AS leg_count,
    observation.scientific_recording_valid,
    outcome.scientific_option_evidence,
    outcome.cohort_phase,
    'open_short_bid_long_ask_close_short_ask_long_bid'
        AS conservative_fill_convention,
    0 AS execution_claimed,
    0 AS paper_fill_claimed
FROM quiet_state_shadow_outcome_v0 AS outcome
JOIN quiet_state_observation_v0 AS observation
  ON observation.run_id = outcome.run_id
 AND observation.observation_id = outcome.observation_id
WHERE observation.observation_kind = 'quiet_bottom_10'
  AND outcome.structure_type IN (
      'ATM_IRON_BUTTERFLY',
      'DELTA_IRON_CONDOR',
      'CALL_CREDIT_SPREAD',
      'PUT_CREDIT_SPREAD'
  );
