from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

import stocker_ideas.plugins.m1c_quiet_state_options_v0 as quiet_plugin
from stocker_ideas.plugins.frozen_m1c_v0 import CAUSAL_GROUP_I_FEATURES, COHORT, score_m1c
from stocker_ideas.plugins.m1c_quiet_state_options_v0 import (
    PARAMETERS,
    M1CQuietStateOptionsV0,
)
from stocker_ideas.plugins.m1c_quiet_state_v0 import (
    BOTTOM_10_THRESHOLD,
    OptionPanelContract,
    classify_quiet_state,
    select_defined_risk_structures,
)
from stocker_research.legacy_prospective.frozen_m1c import FrozenM1CRuntime
from stocker_runtime.domain import (
    IdeaOutput,
    JsonValue,
    MarketEvent,
    ProtectedDataClass,
    RuntimeMode,
    canonical_json_bytes,
)
from stocker_runtime.ideas.contract import (
    DiscoveryReceipt,
    IdeaActivation,
    IdeaBatch,
    IdeaEvaluation,
    MarketDataInterest,
)
from stocker_runtime.ideas.discovery import IdeaConfig, discover_plugins, reviewed_code_hash
from stocker_runtime.ingestion.dynamic_market_data import (
    ContractCandidate,
    InstrumentResolver,
    InterestResolutionRequest,
    MarketDataCapacity,
    MarketDataDemand,
    OptionParameterSet,
    plan_market_data,
)
from stocker_runtime.ingestion.recorder import InstrumentSpec


def _hash(value: Mapping[str, JsonValue]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _activation() -> IdeaActivation:
    return IdeaActivation(
        instance_id="quiet-1",
        parameters=PARAMETERS,
        parameters_hash=_hash(PARAMETERS),
        plugin_code_hash="1" * 64,
        activated_at_us=1,
        run_id="run-1",
        protected_data_class=ProtectedDataClass.PROSPECTIVE,
        universe=COHORT,
    )


def _event(
    event_id: str,
    instrument_id: str,
    event_kind: str,
    at_us: int,
    payload: Mapping[str, JsonValue],
) -> MarketEvent:
    return MarketEvent(
        event_id=event_id,
        instrument_id=instrument_id,
        feed_kind="quotes" if event_kind in {"quote", "option_snapshot_capture"} else "bars",
        event_kind=event_kind,
        event_at_us=at_us,
        received_at_us=at_us,
        payload=payload,
    )


def _replace_event(event: MarketEvent, **updates: JsonValue) -> MarketEvent:
    values = event.model_dump(mode="python")
    values.update(updates)
    return MarketEvent.model_validate(values)


def _prefix_payload(*, symbol: str, session: str, checkpoint: int = 6) -> dict[str, JsonValue]:
    base = 40.0
    trailing: list[dict[str, JsonValue]] = []
    for number in range(1, checkpoint + 1):
        opening = base + number / 10.0
        trailing.append(
            {
                "bar_number": number,
                "event_at_us": 1_000_000 * number,
                "open": opening,
                "high": opening + 1.0,
                "low": opening - 1.0,
                "close": opening + 0.2,
                "volume": 1_000.0 + number,
                "historical_relative_activity": 1.2,
                "return_bps": 10.0,
                "true_range_bps": 200.0,
                "upper_wick_fraction": 0.4,
                "lower_wick_fraction": 0.4,
                "session_vwap": base + number / 20.0,
            }
        )
    return {
        "schema_version": 1,
        "session": session,
        "bar_number": checkpoint,
        "source_completeness": "complete",
        "historical_session_count": 21,
        "historical_relative_activity": 1.2,
        "activity_ready": True,
        "trailing_bars": trailing,
        "session_volumes": [1_000.0 + number for number in range(1, checkpoint + 1)],
        "accumulator": {
            "bar_count": checkpoint,
            "session_open": base,
            "session_high": base + checkpoint / 10.0 + 1.0,
            "session_low": base - 0.9,
            "last_close": base + checkpoint / 10.0 + 0.2,
            "activity_sum": checkpoint * 1.2,
            "range_sum": checkpoint * 200.0,
            "travel_sum": checkpoint * 10.0,
            "return_sum": checkpoint * 10.0,
            "width_sum": checkpoint * 2.0,
            "positive_return_count": checkpoint,
            "negative_return_count": 0,
            "zero_return_count": 0,
        },
        "fixture_symbol": symbol,
    }


def _ordinary_batch(
    events: tuple[MarketEvent, ...],
    *,
    retained: tuple[str, ...] = (),
    receipts: tuple[DiscoveryReceipt, ...] = (),
) -> IdeaBatch:
    return IdeaBatch(
        mode=RuntimeMode.SHADOW,
        events=events,
        input_watermark=events[-1].event_id,
        causal_from_at_us=min(event.event_at_us for event in events),
        causal_through_at_us=max(event.event_at_us for event in events),
        prior_state_input_event_ids=retained,
        discovery_receipts=receipts,
    )


def _prime_low_tail_context(
    plugin: M1CQuietStateOptionsV0,
    *,
    symbol: str = "RGTI",
    state: JsonValue | None = None,
    retained: tuple[str, ...] = (),
    all_cohort_low: bool = False,
) -> tuple[JsonValue, tuple[str, ...]]:
    prior_close = 22.0 if all_cohort_low else 40.0
    realised_volatility = 1.2 if all_cohort_low else 0.58
    baseline_event = _event(
        f"baseline-{symbol}",
        symbol,
        "session_volume_baseline",
        1_000_000,
        {
            "session": "2026-08-07",
            "complete_session_count": 21,
            "session_closes": [prior_close - 0.5, prior_close],
            "realised_volatility_20d": realised_volatility,
        },
    )
    baseline = plugin.evaluate(
        _ordinary_batch((baseline_event,), retained=retained),
        {} if state is None else state,
    )
    assert len(baseline.interests) == 2
    receipts = tuple(
        DiscoveryReceipt(
            receipt_id=f"receipt-{symbol}-{right}",
            interest_id=f"interest-{symbol}-{right}",
            interest_key=f"quiet:m1c:d1:2026-08-07:{symbol}:{right}",
            instance_id="quiet-1",
            status="resolved",
            instrument_id=f"{symbol}-d1-{right}",
            expiry="20260918",
            strike=prior_close,
            option_right=right,
            multiplier="100",
            candidates_inspected=20,
            completed_at_us=1_100_000,
        )
        for right in ("call", "put")
    )
    captures = tuple(
        _event(
            f"capture-{symbol}-{right}",
            f"{symbol}-d1-{right}",
            "option_snapshot_capture",
            1_200_000 + index,
            {
                "source_completeness": "complete",
                "bid": (
                    (3.2 if right == "call" else 2.1)
                    if all_cohort_low
                    else (4.0 if right == "call" else 3.2)
                ),
                "ask": (
                    (3.4 if right == "call" else 5.3)
                    if all_cohort_low
                    else (11.5 if right == "call" else 9.1)
                ),
                "model_implied_volatility": (
                    (4.9 if right == "call" else 5.0)
                    if all_cohort_low
                    else (3.4 if right == "call" else 1.1)
                ),
            },
        )
        for index, right in enumerate(("call", "put"))
    )
    captured = plugin.evaluate(
        _ordinary_batch(
            captures,
            retained=baseline.retained_input_event_ids,
            receipts=receipts,
        ),
        baseline.state,
    )
    return captured.state, captured.retained_input_event_ids


def _planned_entry_panel() -> tuple[
    M1CQuietStateOptionsV0,
    IdeaEvaluation,
    MarketEvent,
    MarketEvent,
]:
    plugin = M1CQuietStateOptionsV0()
    state, retained = _prime_low_tail_context(plugin)
    prefix = _event(
        "prefix-rgti-6",
        "RGTI",
        "bar_5m_session_prefix",
        2_000_000_000,
        _prefix_payload(symbol="RGTI", session="2026-08-10"),
    )
    triggered = plugin.evaluate(_ordinary_batch((prefix,), retained=retained), state)
    quote = _event(
        "quote-rgti-entry",
        "RGTI",
        "quote",
        prefix.event_at_us + 1,
        {"bid": 40.0, "ask": 40.2},
    )
    planned = plugin.evaluate(
        _ordinary_batch((quote,), retained=triggered.retained_input_event_ids),
        triggered.state,
    )
    return plugin, planned, prefix, quote


def _entry_panel_evidence(
    planned: IdeaEvaluation,
    *,
    incomplete_key: str | None = None,
    zero_bid_key: str | None = None,
) -> tuple[tuple[DiscoveryReceipt, ...], tuple[MarketEvent, ...]]:
    expiries = {(0, 0): "20260810", (1, 1): "20260811", (3, 5): "20260813"}
    call_deltas = (0.95, 0.85, 0.75, 0.55, 0.50, 0.40, 0.25, 0.10, 0.05)
    put_deltas = (-0.05, -0.10, -0.25, -0.40, -0.50, -0.55, -0.75, -0.85, -0.95)
    receipts: list[DiscoveryReceipt] = []
    captures: list[MarketEvent] = []
    for index, interest in enumerate(planned.interests):
        bucket = (interest.minimum_days_to_expiry, interest.maximum_days_to_expiry)
        offset = interest.strike_offset
        right = interest.option_right
        instrument = f"option-{bucket[0]}-{bucket[1]}-{offset}-{right}"
        receipts.append(
            DiscoveryReceipt(
                receipt_id=f"receipt-{index}",
                interest_id=f"interest-{index}",
                interest_key=interest.interest_key,
                instance_id="quiet-1",
                status="resolved",
                instrument_id=instrument,
                expiry=expiries[bucket],
                strike=40.0 + offset,
                option_right=right,
                multiplier="100",
                candidates_inspected=40,
                completed_at_us=2_000_000_100 + index,
            )
        )
        bid, ask = (
            (1.50, 1.60) if offset == 0 else (0.85, 0.95) if abs(offset) == 2 else (0.25, 0.30)
        )
        if interest.interest_key == zero_bid_key:
            bid, ask = 0.0, 0.10
        delta = call_deltas[offset + 4] if right == "call" else put_deltas[offset + 4]
        captures.append(
            _event(
                f"panel-capture-{index}",
                instrument,
                "option_snapshot_capture",
                2_000_000_200 + index,
                {
                    "source_completeness": (
                        "incomplete" if interest.interest_key == incomplete_key else "complete"
                    ),
                    "bid": bid,
                    "ask": ask,
                    "model_implied_volatility": 0.50,
                    "model_delta": delta,
                },
            )
        )
    return tuple(receipts), tuple(captures)


def _stream_evidence(
    completed: IdeaEvaluation,
) -> tuple[tuple[DiscoveryReceipt, ...], tuple[MarketEvent, ...], int]:
    state = cast(Mapping[str, object], completed.state)
    active = cast(Mapping[str, object], state["active"])
    instruments = cast(Mapping[str, str], active["m"])
    deadline = cast(int, active["x"])
    receipts: list[DiscoveryReceipt] = []
    quotes: list[MarketEvent] = []
    for index, interest in enumerate(completed.interests):
        instrument_id = instruments[interest.interest_key]
        completed_at_us = 2_000_001_000 + index * 2
        receipts.append(
            DiscoveryReceipt(
                receipt_id=f"stream-receipt-{index}",
                interest_id=f"stream-interest-{index}",
                interest_key=interest.interest_key,
                instance_id="quiet-1",
                status="resolved",
                instrument_id=instrument_id,
                expiry={0: "20260810", 1: "20260811", 3: "20260813"}[
                    interest.minimum_days_to_expiry
                ],
                strike=40.0 + interest.strike_offset,
                option_right=interest.option_right,
                multiplier="100",
                candidates_inspected=1,
                completed_at_us=completed_at_us,
            )
        )
        quotes.append(
            _event(
                f"stream-quote-{index}",
                instrument_id,
                "quote",
                completed_at_us + 1,
                {"bid": 0.50, "ask": 0.60},
            )
        )
    return tuple(receipts), tuple(quotes), deadline


class _PanelDiscoveryBackend:
    def __init__(self) -> None:
        self.parameter_calls = 0
        self.contract_calls: list[tuple[str, float, str]] = []

    def option_parameters(
        self,
        *,
        underlying_con_id: int,
        symbol: str,
    ) -> tuple[OptionParameterSet, ...]:
        assert underlying_con_id == 1234
        assert symbol == "RGTI"
        self.parameter_calls += 1
        return (
            OptionParameterSet(
                exchange="SMART",
                trading_class="RGTI",
                multiplier="100",
                expirations=("20260810", "20260811", "20260813"),
                strikes=tuple(float(value) for value in range(36, 45)),
            ),
        )

    def option_contracts(
        self,
        *,
        symbol: str,
        expiry: str,
        strike: float,
        right: str,
        multiplier: str,
        trading_class: str,
    ) -> tuple[ContractCandidate, ...]:
        self.contract_calls.append((expiry, strike, right))
        return (
            ContractCandidate(
                con_id=int(expiry[-2:]) * 10_000 + int(strike * 10) * 10 + int(right == "P"),
                symbol=symbol,
                expiry=expiry,
                strike=strike,
                right=right,
                multiplier=multiplier,
                exchange="SMART",
                currency="USD",
                trading_class=trading_class,
            ),
        )


def _bucket_contracts() -> tuple[OptionPanelContract, ...]:
    strikes = tuple(float(value) for value in range(96, 105))
    call_deltas = (0.95, 0.85, 0.75, 0.55, 0.50, 0.40, 0.25, 0.10, 0.05)
    put_deltas = (-0.05, -0.10, -0.25, -0.40, -0.50, -0.55, -0.75, -0.85, -0.95)
    contracts: list[OptionPanelContract] = []
    for offset, strike in zip(range(-4, 5), strikes, strict=True):
        bid, ask = (
            (1.50, 1.60) if offset == 0 else (0.85, 0.95) if abs(offset) == 2 else (0.25, 0.30)
        )
        for right, delta in (
            ("call", call_deltas[offset + 4]),
            ("put", put_deltas[offset + 4]),
        ):
            contracts.append(
                OptionPanelContract(
                    event_id=f"capture-{offset}-{right}",
                    interest_key=f"quiet:episode:entry:0DTE:{offset}:{right}",
                    instrument_id=f"option-{offset}-{right}",
                    bucket="0DTE",
                    offset=offset,
                    right=right,
                    expiry="20260810",
                    strike=strike,
                    multiplier="100",
                    bid=bid,
                    ask=ask,
                    delta=delta,
                )
            )
    return tuple(contracts)


def test_quiet_plugin_declares_only_the_existing_generic_market_data_seams() -> None:
    plugin = M1CQuietStateOptionsV0()
    requirements = plugin.requirements(_activation())

    assert plugin.manifest.idea_id == "m1c_quiet_state_options"
    assert plugin.manifest.modes == (RuntimeMode.PROSPECTIVE_RECORD, RuntimeMode.SHADOW)
    assert plugin.manifest.maximum_outputs_per_batch == 64
    assert plugin.manifest.maximum_interests_per_batch == 64
    assert len(requirements) == 60
    assert {item.instrument_id for item in requirements} == set(COHORT)
    assert {(item.feed_kind, item.event_kind, item.cadence) for item in requirements} == {
        ("bars", "bar_5m_session_prefix", "5s"),
        ("bars", "session_volume_baseline", "5s"),
        ("quotes", "quote", "stream"),
    }


def test_quiet_constants_remain_bound_to_the_frozen_research_contracts() -> None:
    root = Path("research/prospective/frozen-m1c-microstructure-recorder-v0")
    thresholds = json.loads((root / "quiet_state_threshold_manifest.json").read_text())
    option_selection = json.loads((root / "quiet_state_option_selection_contract.json").read_text())
    structures = json.loads((root / "defined_risk_structure_contract.json").read_text())

    assert thresholds["m1c_bottom_10_threshold"] == BOTTOM_10_THRESHOLD
    assert thresholds["primary_quiet_state"] == "bottom_10_percent"
    assert option_selection["maximum_contracts_per_quiet_observation"] == 54
    assert option_selection["maximum_neighbouring_strike_steps"] == 4
    assert option_selection["complete_chain_stream_allowed"] is False
    assert option_selection["capacity_reduced_plan_complete"] is False
    assert structures["naked_short_options_allowed"] is False
    assert structures["long_premium_structures"] == ["ATM_CALL", "ATM_PUT", "ATM_STRADDLE"]

    replay_path = root / "quiet_state_replay_fixture.json"
    replay = json.loads(replay_path.read_text())
    determinism = json.loads((root / "quiet_state_determinism_check.json").read_text())
    assert hashlib.sha256(replay_path.read_bytes()).hexdigest() == determinism["fixture_sha256"]
    assert determinism["fixture_sha256"] == (
        "974477f1f6d1a9c74d2f0949bd0421934dbe267fc98f3efbd112e1cbb63a82a5"
    )
    assert determinism["canonical_replay_hash_a"] == determinism["canonical_replay_hash_b"]
    assert determinism["canonical_replay_hash_a"] == (
        "b911ad3b942ba5b1d7870fe51003d29c00131000626e4687484b7723b951ecbf"
    )
    assert replay["prediction_rows"] == 100
    assert replay["defined_risk_cases"] == 50


def test_frozen_replay_exercises_the_new_quiet_classifier_and_structure_selector() -> None:
    root = Path("research/prospective/frozen-m1c-microstructure-recorder-v0")
    replay = json.loads((root / "quiet_state_replay_fixture.json").read_text())
    artifact_root = Path(
        "research/directional-readiness/"
        "20260726-stock-local-directional-archetypes-v0/artifacts/primary"
    )
    runtime = FrozenM1CRuntime.from_artifacts(
        feature_manifest_path=artifact_root / "causal_movement_feature_manifest.json",
        threshold_path=artifact_root / "causal_movement_threshold.json",
    )

    def replay_once() -> dict[str, JsonValue]:
        predictions: list[dict[str, JsonValue]] = []
        prior_probability: dict[tuple[str, str], float] = {}
        prior_episode_at: dict[tuple[str, str], int] = {}
        for fixture_row in replay["prediction_inputs"]:
            group_o: dict[str, object] = {}
            group_i: dict[str, object] = {}
            seed = int(fixture_row["feature_offset_seed"])
            missing = fixture_row["missing_feature_index"]
            for index, name in enumerate(runtime.numeric_features):
                if name.startswith("checkpoint_"):
                    continue
                value: object = (
                    float(runtime.numeric_medians[index])
                    + (((seed + index) % 9) - 4) * float(runtime.numeric_scales[index]) * 0.025
                )
                if missing == index:
                    value = None
                target = group_i if name in CAUSAL_GROUP_I_FEATURES else group_o
                target[name] = value
            symbol = str(fixture_row["symbol"])
            session = str(fixture_row["session"])
            checkpoint = int(fixture_row["checkpoint"])
            legacy = runtime.score(
                symbol=symbol,
                checkpoint=checkpoint,
                group_o_context=group_o,
                causal_group_i=group_i,
            )
            current = score_m1c(
                symbol=symbol,
                checkpoint=checkpoint,
                group_o=group_o,
                group_i=group_i,
            )
            probability = cast(float, current["probability"])
            assert probability == pytest.approx(legacy.probability, abs=1e-15)
            key = (session, symbol)
            at_us = int(
                datetime.fromisoformat(str(fixture_row["timestamp_utc"])).timestamp() * 1_000_000
            )
            previous_at = prior_episode_at.get(key)
            quiet = classify_quiet_state(
                probability=probability,
                previous_probability=prior_probability.get(key),
                minutes_since_previous_episode=(
                    None if previous_at is None else (at_us - previous_at) / 60_000_000.0
                ),
            )
            prior_probability[key] = probability
            if quiet.fresh_episode:
                prior_episode_at[key] = at_us
            predictions.append(
                {
                    "row": int(fixture_row["row"]),
                    "probability": probability,
                    "bottom_5": quiet.bottom_5,
                    "bottom_10": quiet.bottom_10,
                    "bottom_20": quiet.bottom_20,
                    "fresh_episode": quiet.fresh_episode,
                }
            )

        option_cases: list[dict[str, JsonValue]] = []
        for fixture_case in replay["defined_risk_inputs"]:
            contracts = tuple(
                OptionPanelContract(
                    event_id=f"fixture-{fixture_case['case']}-{item['con_id']}",
                    interest_key=f"fixture:{fixture_case['case']}:{item['con_id']}",
                    instrument_id=str(item["con_id"]),
                    bucket="3_5_dte",
                    offset=int(float(item["strike"]) - 100.0),
                    right="call" if item["right"] == "C" else "put",
                    expiry="20260731",
                    strike=float(item["strike"]),
                    multiplier="100",
                    bid=float(item["entry_bid"]),
                    ask=float(item["entry_ask"]),
                    delta=float(item["delta"]),
                )
                for item in fixture_case["contracts"]
            )
            attempts = select_defined_risk_structures(
                contracts=contracts,
                underlying_reference_price=float(fixture_case["underlying_entry"]),
            )
            expected = (
                (
                    ("short", 100.0, "call"),
                    ("short", 100.0, "put"),
                    ("long", 102.0, "call"),
                    ("long", 98.0, "put"),
                ),
                (
                    ("short", 102.0, "call"),
                    ("short", 98.0, "put"),
                    ("long", 105.0, "call"),
                    ("long", 95.0, "put"),
                ),
            )
            assert (
                tuple(
                    tuple(
                        (leg.side, leg.contract.strike, leg.contract.right) for leg in attempt.legs
                    )
                    for attempt in attempts[:2]
                )
                == expected
            )
            option_cases.append(
                {
                    "case": int(fixture_case["case"]),
                    "attempts": [
                        {
                            "structure": attempt.structure_type,
                            "available": attempt.available,
                            "reason": attempt.reason,
                            "opening_credit": attempt.opening_credit,
                            "legs": [
                                {
                                    "side": leg.side,
                                    "strike": leg.contract.strike,
                                    "right": leg.contract.right,
                                }
                                for leg in attempt.legs
                            ],
                        }
                        for attempt in attempts
                    ],
                }
            )
        return {"predictions": predictions, "option_cases": option_cases}

    first = replay_once()
    second = replay_once()
    assert first == second
    assert (
        hashlib.sha256(
            json.dumps(first, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        == "5359dacc1bf8e793ee2111fc3595a0f0163155e4546405d7bd9fc9a7caf58d53"
    )


def test_quiet_plugin_starts_through_isolated_reviewed_discovery() -> None:
    module = "stocker_ideas.plugins.m1c_quiet_state_options_v0"
    discovered = discover_plugins(
        (
            IdeaConfig(
                module=module,
                expected_code_hash=reviewed_code_hash(module),
                expected_manifest_hash=hashlib.sha256(
                    quiet_plugin.MANIFEST.to_canonical_json()
                ).hexdigest(),
                parameters=quiet_plugin.PARAMETERS,
                universe=COHORT,
                enabled=True,
            ),
        )
    )

    assert len(discovered) == 1
    assert discovered[0].manifest.idea_id == "m1c_quiet_state_options"
    assert len(discovered[0].requirements) == 60


def test_d1_snapshot_planning_uses_only_the_latest_open_window_and_never_backfills() -> None:
    def baseline(event_id: str, session: str, at_us: int) -> MarketEvent:
        return _event(
            event_id,
            "RGTI",
            "session_volume_baseline",
            at_us,
            {
                "session": session,
                "complete_session_count": 21,
                "session_closes": [39.5, 40.0],
                "realised_volatility_20d": 0.58,
            },
        )

    latest_open = M1CQuietStateOptionsV0().evaluate(
        _ordinary_batch(
            (
                baseline("baseline-1", "2026-08-05", 1_000_000),
                baseline("baseline-2", "2026-08-06", 2_000_000),
                baseline("baseline-3", "2026-08-07", 3_000_000),
                _event(
                    "heartbeat-open",
                    "RGTI",
                    "quote",
                    3_000_001,
                    {"bid": 39.9, "ask": 40.1},
                ),
            )
        ),
        {},
    )

    assert len(latest_open.interests) == 2
    assert {interest.interest_key for interest in latest_open.interests} == {
        "quiet:m1c:d1:2026-08-07:RGTI:call",
        "quiet:m1c:d1:2026-08-07:RGTI:put",
    }

    expired = M1CQuietStateOptionsV0().evaluate(
        _ordinary_batch(
            (
                baseline("baseline-expired", "2026-08-07", 1_000_000),
                _event(
                    "heartbeat-after-expiry",
                    "RGTI",
                    "quote",
                    1_801_000_001,
                    {"bid": 39.9, "ask": 40.1},
                ),
            )
        ),
        {},
    )

    assert expired.interests == ()


def test_next_session_prefix_must_not_precede_the_immutable_d1_cutoff() -> None:
    plugin = M1CQuietStateOptionsV0()
    state, retained = _prime_low_tail_context(plugin)
    before_cutoff = _event(
        "prefix-before-d1-cutoff",
        "RGTI",
        "bar_5m_session_prefix",
        1_800_999_999,
        _prefix_payload(symbol="RGTI", session="2026-08-10"),
    )

    with pytest.raises(ValueError, match="precedes the D-1 cutoff"):
        plugin.evaluate(_ordinary_batch((before_cutoff,), retained=retained), state)

    equality_plugin = M1CQuietStateOptionsV0()
    equality_state, equality_retained = _prime_low_tail_context(equality_plugin)
    at_cutoff = _replace_event(
        before_cutoff,
        event_id="prefix-at-d1-cutoff",
        event_at_us=1_801_000_000,
        received_at_us=1_801_000_000,
    )
    evaluated = equality_plugin.evaluate(
        _ordinary_batch((at_cutoff,), retained=equality_retained),
        equality_state,
    )
    assert evaluated.outputs[0].payload["status"] == "complete"


def test_d1_denial_and_missing_right_keep_exact_terminal_attribution_and_lineage() -> None:
    plugin = M1CQuietStateOptionsV0()
    baseline = _event(
        "baseline-rgti-denial",
        "RGTI",
        "session_volume_baseline",
        1_000_000,
        {
            "session": "2026-08-07",
            "complete_session_count": 21,
            "session_closes": [39.5, 40.0],
            "realised_volatility_20d": 0.58,
        },
    )
    planned = plugin.evaluate(_ordinary_batch((baseline,)), {})
    denied = DiscoveryReceipt(
        receipt_id="d1-call-denied",
        interest_id="d1-call-interest",
        interest_key="quiet:m1c:d1:2026-08-07:RGTI:call",
        instance_id="quiet-1",
        status="denied",
        reason_code="OPTION_PERMISSION_DENIED",
        candidates_inspected=0,
        completed_at_us=1_100_000,
    )
    prefix = _event(
        "prefix-rgti-denied-d1",
        "RGTI",
        "bar_5m_session_prefix",
        1_801_000_000,
        _prefix_payload(symbol="RGTI", session="2026-08-10"),
    )

    evaluated = plugin.evaluate(
        _ordinary_batch(
            (prefix,),
            retained=planned.retained_input_event_ids,
            receipts=(denied,),
        ),
        planned.state,
    )

    assert len(evaluated.outputs) == 1
    payload = evaluated.outputs[0].payload
    assert payload["status"] == "unavailable"
    assert payload["interest_keys"] == (
        "quiet:m1c:d1:2026-08-07:RGTI:call",
        "quiet:m1c:d1:2026-08-07:RGTI:put",
    )
    assert payload["terminal_statuses"] == {
        "call": "denied",
        "put": "window_elapsed",
    }
    assert payload["denial_reasons"] == {"call": "OPTION_PERMISSION_DENIED"}
    assert payload["cutoff_at_us"] == 1_801_000_000
    assert payload["terminal_basis"] == "mixed_terminal_option_evidence"
    assert evaluated.output_input_event_ids == (("baseline-rgti-denial", "prefix-rgti-denied-d1"),)


def test_d1_capture_after_cutoff_remains_exactly_attributed_as_late() -> None:
    plugin = M1CQuietStateOptionsV0()
    baseline = _event(
        "baseline-rgti-late",
        "RGTI",
        "session_volume_baseline",
        1_000_000,
        {
            "session": "2026-08-07",
            "complete_session_count": 21,
            "session_closes": [39.5, 40.0],
            "realised_volatility_20d": 0.58,
        },
    )
    planned = plugin.evaluate(_ordinary_batch((baseline,)), {})
    cutoff = 1_801_000_000
    late_receipt = DiscoveryReceipt(
        receipt_id="d1-call-late-receipt",
        interest_id="d1-call-late-interest",
        interest_key="quiet:m1c:d1:2026-08-07:RGTI:call",
        instance_id="quiet-1",
        status="resolved",
        instrument_id="RGTI-d1-call-late",
        expiry="20260821",
        strike=40.0,
        option_right="call",
        multiplier="100",
        candidates_inspected=20,
        completed_at_us=cutoff + 1,
    )
    late_capture = _event(
        "capture-rgti-call-late",
        "RGTI-d1-call-late",
        "option_snapshot_capture",
        cutoff + 1,
        {
            "source_completeness": "complete",
            "bid": 1.0,
            "ask": 1.1,
            "model_implied_volatility": 0.5,
        },
    )
    prefix = _event(
        "prefix-rgti-after-late-d1",
        "RGTI",
        "bar_5m_session_prefix",
        cutoff + 2,
        _prefix_payload(symbol="RGTI", session="2026-08-10"),
    )

    captured = plugin.evaluate(
        _ordinary_batch(
            (late_capture,),
            retained=planned.retained_input_event_ids,
            receipts=(late_receipt,),
        ),
        planned.state,
    )
    assert late_capture.event_id in captured.retained_input_event_ids
    evaluated = M1CQuietStateOptionsV0().evaluate(
        _ordinary_batch((prefix,), retained=captured.retained_input_event_ids),
        captured.state,
    )

    payload = evaluated.outputs[0].payload
    assert payload["terminal_statuses"] == {"call": "late", "put": "window_elapsed"}
    assert payload["terminal_basis"] == "mixed_terminal_option_evidence"
    assert payload["evidence_available_at_us"] == {"call": cutoff + 1}
    assert late_capture.event_id in evaluated.output_input_event_ids[0]


def test_quiet_crossing_waits_for_a_quote_then_requests_the_exact_54_contract_panel() -> None:
    plugin = M1CQuietStateOptionsV0()
    state, retained = _prime_low_tail_context(plugin)
    prefix = _event(
        "prefix-rgti-6",
        "RGTI",
        "bar_5m_session_prefix",
        2_000_000_000,
        _prefix_payload(symbol="RGTI", session="2026-08-10"),
    )

    triggered = plugin.evaluate(
        _ordinary_batch((prefix,), retained=retained),
        state,
    )

    assert triggered.interests == ()
    assert len(triggered.outputs) == 1
    assert triggered.outputs[0].kind == "observation"
    assert triggered.outputs[0].payload["status"] == "complete"
    assert triggered.outputs[0].payload["fresh_episode"] is True
    assert triggered.outputs[0].payload["bottom_10"] is True
    assert triggered.outputs[0].payload["probability"] == pytest.approx(0.08276432334461586)

    quote = _event(
        "quote-rgti-entry",
        "RGTI",
        "quote",
        prefix.event_at_us + 1,
        {"bid": 40.0, "ask": 40.2},
    )
    planned = plugin.evaluate(
        _ordinary_batch((quote,), retained=triggered.retained_input_event_ids),
        triggered.state,
    )

    assert len(planned.interests) == 54
    assert len({interest.interest_key for interest in planned.interests}) == 54
    assert {
        (
            interest.minimum_days_to_expiry,
            interest.maximum_days_to_expiry,
            interest.strike_offset,
            interest.option_right,
        )
        for interest in planned.interests
    } == {
        (minimum, maximum, offset, right)
        for minimum, maximum in ((0, 0), (1, 1), (3, 5))
        for offset in range(-4, 5)
        for right in ("call", "put")
    }
    assert all(interest.required for interest in planned.interests)
    assert all(interest.cadence == "snapshot" for interest in planned.interests)
    assert all(interest.reference_price == pytest.approx(40.1) for interest in planned.interests)
    assert all(interest.input_event_id == quote.event_id for interest in planned.interests)
    assert len(planned.state_json()) < 65_536


def test_exact_54_panel_resolves_and_plans_within_the_existing_market_data_bound() -> None:
    _plugin, planned, _prefix, _quote = _planned_entry_panel()
    as_of = int(datetime(2026, 8, 10, 14, tzinfo=UTC).timestamp() * 1_000_000)
    backend = _PanelDiscoveryBackend()
    resolver = InstrumentResolver(
        backend,
        underlyings={"RGTI": InstrumentSpec("RGTI", 1234, "stock", "RGTI", "SMART", "USD")},
        completed_at_us=lambda: as_of + 1,
    )
    resolved = tuple(
        resolver.resolve(
            InterestResolutionRequest(
                interest_id=f"panel-{index}",
                instance_id="quiet-1",
                interest=type(interest).model_validate(
                    {
                        **interest.model_dump(),
                        "as_of_at_us": as_of,
                        "expires_at_us": as_of + 3_600_000_000,
                    }
                ),
            )
        )
        for index, interest in enumerate(planned.interests)
    )

    assert all(receipt.status == "resolved" for receipt in resolved)
    assert backend.parameter_calls == 1
    assert len(backend.contract_calls) == 54
    assert {(receipt.expiry, receipt.strike, receipt.option_right) for receipt in resolved} == {
        (expiry, 40.0 + offset, right)
        for expiry in ("20260810", "20260811", "20260813")
        for offset in range(-4, 5)
        for right in ("call", "put")
    }
    demands = tuple(
        MarketDataDemand(
            source_id=receipt.interest_id,
            instrument_id=cast(str, receipt.instrument_id),
            feed_kind="quotes",
            required=True,
            priority=200,
            stale_after_us=60_000_000,
            snapshot=True,
        )
        for receipt in resolved
    )
    plan = plan_market_data((), demands, MarketDataCapacity(line_limit=54))
    assert plan.required_complete is True
    assert plan.line_count == 54
    assert plan.deferred_source_ids == ()


def test_complete_panel_emits_nine_long_observations_and_twelve_protected_attempts() -> None:
    plugin, planned, prefix, quote = _planned_entry_panel()
    receipts, captures = _entry_panel_evidence(planned)

    completed = plugin.evaluate(
        _ordinary_batch(
            captures,
            retained=planned.retained_input_event_ids,
            receipts=receipts,
        ),
        planned.state,
    )

    observations = tuple(output for output in completed.outputs if output.kind == "observation")
    proposals = tuple(output for output in completed.outputs if output.kind == "proposed_trade")
    assert len(completed.outputs) == 21
    assert len(observations) == 9
    assert len(proposals) == 12
    assert {output.payload["structure_type"] for output in observations} == {
        "ATM_CALL",
        "ATM_PUT",
        "ATM_STRADDLE",
    }
    assert {output.payload["structure_type"] for output in proposals} == {
        "ATM_IRON_BUTTERFLY",
        "DELTA_IRON_CONDOR",
        "CALL_CREDIT_SPREAD",
        "PUT_CREDIT_SPREAD",
    }
    for proposal in proposals:
        assert proposal.status == "unapproved"
        assert proposal.payload["defined_risk"] is True
        assert proposal.payload["research_only"] is True
        assert proposal.payload["execution_enabled"] is False
        assert any(leg.action == "sell" and leg.target == "short" for leg in proposal.legs)
        assert any(leg.action == "buy" and leg.target == "long" for leg in proposal.legs)
        assert {leg.currency for leg in proposal.legs} == {"USD"}
    assert len(completed.interests) == 24
    assert all(interest.cadence == "stream" for interest in completed.interests)
    assert all(interest.required for interest in completed.interests)
    assert len({interest.interest_key for interest in completed.interests}) == 24
    assert len(completed.output_input_event_ids) == 21
    assert all(prefix.event_id in lineage for lineage in completed.output_input_event_ids)
    assert all(quote.event_id in lineage for lineage in completed.output_input_event_ids)
    assert all(len(lineage) == 59 for lineage in completed.output_input_event_ids)
    assert len(completed.state_json()) < 65_536
    assert len(completed.retained_input_event_ids) <= 256


def test_zero_bid_short_leg_is_unavailable_instead_of_crashing_the_plugin() -> None:
    plugin, planned, _prefix, _quote = _planned_entry_panel()
    zero_bid_key = next(
        interest.interest_key
        for interest in planned.interests
        if interest.minimum_days_to_expiry == 0
        and interest.strike_offset == 0
        and interest.option_right == "call"
    )
    receipts, captures = _entry_panel_evidence(planned, zero_bid_key=zero_bid_key)

    completed = plugin.evaluate(
        _ordinary_batch(
            captures,
            retained=planned.retained_input_event_ids,
            receipts=receipts,
        ),
        planned.state,
    )

    unavailable = tuple(
        output
        for output in completed.outputs
        if output.payload.get("reason") == "short_leg_bid_not_positive"
    )
    assert {output.payload["structure_type"] for output in unavailable} == {
        "ATM_IRON_BUTTERFLY",
        "CALL_CREDIT_SPREAD",
    }
    assert all(output.kind == "observation" for output in unavailable)
    assert all(
        not (
            output.kind == "proposed_trade"
            and output.payload.get("dte_bucket") == "0DTE"
            and output.payload.get("structure_type") in {"ATM_IRON_BUTTERFLY", "CALL_CREDIT_SPREAD"}
        )
        for output in completed.outputs
    )


def test_one_incomplete_capture_fails_the_whole_panel_closed() -> None:
    plugin, planned, _prefix, _quote = _planned_entry_panel()
    incomplete_key = planned.interests[17].interest_key
    receipts, captures = _entry_panel_evidence(planned, incomplete_key=incomplete_key)

    completed = plugin.evaluate(
        _ordinary_batch(
            captures,
            retained=planned.retained_input_event_ids,
            receipts=receipts,
        ),
        planned.state,
    )

    assert len(completed.outputs) == 1
    assert completed.outputs[0].kind == "observation"
    assert completed.outputs[0].payload["status"] == "incomplete"
    assert completed.outputs[0].payload["reason"] == "option_panel_capture_incomplete"
    assert completed.interests == ()
    assert all(output.kind != "proposed_trade" for output in completed.outputs)


def test_one_denied_required_entry_contract_fails_the_whole_panel_closed() -> None:
    plugin, planned, _prefix, _quote = _planned_entry_panel()
    denied_interest = planned.interests[0]
    denied = DiscoveryReceipt(
        receipt_id="entry-denied",
        interest_id="entry-interest-denied",
        interest_key=denied_interest.interest_key,
        instance_id="quiet-1",
        status="denied",
        reason_code="no_matching_expiry",
        candidates_inspected=20,
        completed_at_us=2_000_000_100,
    )
    heartbeat = _event(
        "quote-rgti-entry-denial",
        "RGTI",
        "quote",
        2_000_000_101,
        {"bid": 40.0, "ask": 40.2},
    )

    evaluated = plugin.evaluate(
        _ordinary_batch(
            (heartbeat,),
            retained=planned.retained_input_event_ids,
            receipts=(denied,),
        ),
        planned.state,
    )

    assert len(evaluated.outputs) == 1
    assert evaluated.outputs[0].payload["reason"] == "option_panel_discovery_denied"
    assert evaluated.outputs[0].payload["interest_statuses"][denied_interest.interest_key] == {
        "status": "denied",
        "reason": "no_matching_expiry",
        "completed_at_us": 2_000_000_100,
    }
    assert heartbeat.event_id in evaluated.output_input_event_ids[0]
    assert all(output.kind != "proposed_trade" for output in evaluated.outputs)
    assert evaluated.interests == ()


def test_entry_panel_timeout_uses_exact_cutoff_event_and_stored_score_lineage() -> None:
    _plugin, planned, prefix, quote = _planned_entry_panel()
    planned_state = cast(Mapping[str, object], planned.state)
    active = cast(Mapping[str, object], planned_state["active"])
    deadline = cast(int, active["x"])
    cutoff = _event(
        "entry-panel-cutoff",
        "RGTI",
        "quote",
        deadline,
        {"bid": 40.0, "ask": 40.2},
    )

    timed_out = M1CQuietStateOptionsV0().evaluate(
        _ordinary_batch((cutoff,), retained=planned.retained_input_event_ids),
        planned.state,
    )

    assert len(timed_out.outputs) == 1
    payload = timed_out.outputs[0].payload
    assert payload["reason"] == "option_panel_capture_missing"
    assert payload["cutoff_at_us"] == deadline
    assert payload["cutoff_crossing_event_id"] == cutoff.event_id
    assert len(payload["interest_statuses"]) == 54
    assert {prefix.event_id, quote.event_id, cutoff.event_id}.issubset(
        timed_out.output_input_event_ids[0]
    )
    assert timed_out.state["active"] == {}


def test_entry_receipt_after_cutoff_cannot_override_the_terminal_evidence() -> None:
    _plugin, planned, _prefix, _quote = _planned_entry_panel()
    planned_state = cast(Mapping[str, object], planned.state)
    active = cast(Mapping[str, object], planned_state["active"])
    deadline = cast(int, active["x"])
    key = planned.interests[0].interest_key
    late_denial = DiscoveryReceipt(
        receipt_id="entry-denial-after-cutoff",
        interest_id="entry-interest-after-cutoff",
        interest_key=key,
        instance_id="quiet-1",
        status="denied",
        reason_code="line_capacity",
        candidates_inspected=0,
        completed_at_us=deadline + 1,
    )
    cutoff = _event(
        "entry-receipt-cutoff",
        "RGTI",
        "quote",
        deadline,
        {"bid": 40.0, "ask": 40.2},
    )
    after = _event(
        "entry-receipt-after-cutoff",
        "RGTI",
        "quote",
        deadline + 1,
        {"bid": 40.0, "ask": 40.2},
    )

    combined = M1CQuietStateOptionsV0().evaluate(
        _ordinary_batch(
            (cutoff, after),
            retained=planned.retained_input_event_ids,
            receipts=(late_denial,),
        ),
        planned.state,
    )
    split = M1CQuietStateOptionsV0().evaluate(
        _ordinary_batch((cutoff,), retained=planned.retained_input_event_ids),
        planned.state,
    )

    assert combined.outputs[0].payload == split.outputs[0].payload
    assert combined.outputs[0].payload["reason"] == "option_panel_capture_missing"
    assert combined.outputs[0].payload["interest_statuses"][key] == {
        "status": "missing",
        "reason": None,
        "completed_at_us": None,
    }
    assert after.event_id not in combined.output_input_event_ids[0]


def test_entry_captures_after_the_cutoff_event_cannot_be_admitted_by_backdating() -> None:
    _plugin, planned, _prefix, _quote = _planned_entry_panel()
    receipts, captures = _entry_panel_evidence(planned)
    planned_state = cast(Mapping[str, object], planned.state)
    active = cast(Mapping[str, object], planned_state["active"])
    deadline = cast(int, active["x"])
    cutoff = _event(
        "entry-cutoff-before-backdated-captures",
        "RGTI",
        "quote",
        deadline,
        {"bid": 40.0, "ask": 40.2},
    )

    rejected = M1CQuietStateOptionsV0().evaluate(
        _ordinary_batch(
            (cutoff, *captures),
            retained=planned.retained_input_event_ids,
            receipts=receipts,
        ),
        planned.state,
    )

    assert len(rejected.outputs) == 1
    assert rejected.outputs[0].payload["reason"] == "option_panel_capture_missing"
    assert rejected.outputs[0].payload["capture_count"] == 0
    assert rejected.outputs[0].payload["cutoff_crossing_event_id"] == cutoff.event_id
    assert all(output.kind != "proposed_trade" for output in rejected.outputs)
    assert rejected.state["active"] == {}


def test_partial_panel_resumes_on_a_fresh_plugin_instance_without_duplicate_interests() -> None:
    plugin, planned, _prefix, _quote = _planned_entry_panel()
    receipts, captures = _entry_panel_evidence(planned)
    partial = plugin.evaluate(
        _ordinary_batch(
            captures[:27],
            retained=planned.retained_input_event_ids,
            receipts=receipts,
        ),
        planned.state,
    )

    assert partial.outputs == ()
    assert partial.interests == ()
    assert len(partial.retained_input_event_ids) == 32
    assert len(partial.state_json()) < 65_536

    completed = M1CQuietStateOptionsV0().evaluate(
        _ordinary_batch(
            captures[27:],
            retained=partial.retained_input_event_ids,
            receipts=receipts,
        ),
        partial.state,
    )

    assert len(completed.outputs) == 21
    assert len(completed.interests) == 24
    assert all(len(lineage) == 59 for lineage in completed.output_input_event_ids)


def test_entry_completion_is_committed_before_a_later_stream_cutoff() -> None:
    plugin, planned, _prefix, _quote = _planned_entry_panel()
    receipts, captures = _entry_panel_evidence(planned)
    planned_active = cast(Mapping[str, object], cast(Mapping[str, object], planned.state)["active"])
    planned_deadline = cast(int, planned_active["x"])
    planned_cutoff = _event(
        "entry-normal-delivery-cutoff",
        "RGTI",
        "quote",
        planned_deadline,
        {"bid": 40.0, "ask": 40.2},
    )
    assert (
        plugin.select_input_prefix(
            _ordinary_batch(
                (*captures, planned_cutoff),
                retained=planned.retained_input_event_ids,
                receipts=receipts,
            ),
            planned.state,
        )
        == 1
    )
    partial = plugin.evaluate(
        _ordinary_batch(
            captures[:53],
            retained=planned.retained_input_event_ids,
            receipts=receipts,
        ),
        planned.state,
    )
    partial_active = cast(Mapping[str, object], cast(Mapping[str, object], partial.state)["active"])
    deadline = cast(int, partial_active["x"])
    cutoff = _event(
        "stream-cutoff-after-final-entry-capture",
        "RGTI",
        "quote",
        deadline,
        {"bid": 40.0, "ask": 40.2},
    )
    fetched = _ordinary_batch(
        (captures[53], cutoff),
        retained=partial.retained_input_event_ids,
        receipts=receipts,
    )

    assert plugin.select_input_prefix(fetched, partial.state) == 1
    completed = plugin.evaluate(
        _ordinary_batch(
            (captures[53],),
            retained=partial.retained_input_event_ids,
            receipts=receipts,
        ),
        partial.state,
    )
    assert len(completed.outputs) == 21
    assert len(completed.interests) == 24
    assert completed.state["active"]["g"] == "streams"

    terminal = M1CQuietStateOptionsV0().evaluate(
        _ordinary_batch((cutoff,), retained=completed.retained_input_event_ids),
        completed.state,
    )
    assert len(terminal.outputs) == 1
    assert terminal.outputs[0].payload["reason"] == "selected_leg_stream_window_incomplete"
    assert terminal.interests == ()
    assert terminal.state["active"] == {}


def test_all_twenty_simultaneous_crossings_service_one_and_record_the_other_nineteen() -> None:
    plugin = M1CQuietStateOptionsV0()
    state: JsonValue | None = None
    retained: tuple[str, ...] = ()
    for symbol in COHORT:
        state, retained = _prime_low_tail_context(
            plugin,
            symbol=symbol,
            state=state,
            retained=retained,
            all_cohort_low=True,
        )
    assert state is not None
    at_us = 2_000_000_000
    events = tuple(
        event
        for index, symbol in enumerate(COHORT)
        for event in (
            _event(
                f"prefix-{symbol.lower()}-6",
                symbol,
                "bar_5m_session_prefix",
                at_us + index * 2,
                _prefix_payload(symbol=symbol, session="2026-08-10"),
            ),
            _event(
                f"quote-{symbol.lower()}",
                symbol,
                "quote",
                at_us + index * 2 + 1,
                {"bid": 40.0, "ask": 40.2},
            ),
        )
    )

    def drain(
        chunk_size: int,
    ) -> tuple[
        JsonValue,
        tuple[str, ...],
        tuple[IdeaOutput, ...],
        tuple[MarketDataInterest, ...],
    ]:
        current_state = state
        current_retained = retained
        cursor = 0
        outputs: list[IdeaOutput] = []
        interests: list[MarketDataInterest] = []
        runner_plugin = M1CQuietStateOptionsV0()
        while cursor < len(events):
            candidate_events = events[cursor : cursor + chunk_size]
            candidate_batch = _ordinary_batch(candidate_events, retained=current_retained)
            selected_count = runner_plugin.select_input_prefix(candidate_batch, current_state)
            assert selected_count >= 1
            evaluated = runner_plugin.evaluate(
                _ordinary_batch(
                    candidate_events[:selected_count],
                    retained=current_retained,
                ),
                current_state,
            )
            outputs.extend(evaluated.outputs)
            interests.extend(evaluated.interests)
            current_state = evaluated.state
            current_retained = evaluated.retained_input_event_ids
            cursor += selected_count
        return current_state, current_retained, tuple(outputs), tuple(interests)

    combined_state, combined_retained, combined_outputs, combined_interests = drain(len(events))
    split_state, split_retained, split_outputs, split_interests = drain(1)

    assert combined_state == split_state
    assert combined_retained == split_retained
    assert tuple(output.model_dump(mode="json") for output in combined_outputs) == tuple(
        output.model_dump(mode="json") for output in split_outputs
    )
    assert tuple(interest.model_dump(mode="json") for interest in combined_interests) == tuple(
        interest.model_dump(mode="json") for interest in split_interests
    )
    assert len(combined_interests) == 54
    assert {interest.underlying_instrument_id for interest in combined_interests} == {COHORT[0]}
    incomplete = tuple(
        output
        for output in combined_outputs
        if output.payload.get("reason") == "complete_panel_capacity_unavailable"
    )
    assert len(incomplete) == 19
    assert {output.subject_instrument_id for output in incomplete} == set(COHORT[1:])
    assert len(combined_outputs) == 39

    planned = IdeaEvaluation(
        state=combined_state,
        outputs=(),
        retained_input_event_ids=combined_retained,
        output_input_event_ids=(),
        interests=combined_interests,
    )
    receipts, captures = _entry_panel_evidence(planned)
    worst_case = plugin.evaluate(
        _ordinary_batch(
            captures[:53],
            retained=combined_retained,
            receipts=receipts,
        ),
        combined_state,
    )
    assert worst_case.outputs == ()
    assert len(worst_case.state_json()) < 65_536
    assert len(worst_case.retained_input_event_ids) <= 256


def test_expired_episode_is_terminalized_before_a_later_candidate_independent_of_partition() -> (
    None
):
    plugin = M1CQuietStateOptionsV0()
    state, retained = _prime_low_tail_context(plugin, symbol="AAL", all_cohort_low=True)
    state, retained = _prime_low_tail_context(
        plugin,
        symbol="RGTI",
        state=state,
        retained=retained,
        all_cohort_low=True,
    )
    aal_prefix = _event(
        "prefix-aal-expiring",
        "AAL",
        "bar_5m_session_prefix",
        2_000_000_000,
        _prefix_payload(symbol="AAL", session="2026-08-10"),
    )
    triggered = plugin.evaluate(_ordinary_batch((aal_prefix,), retained=retained), state)
    cutoff = 5_600_000_000
    clock = _event(
        "clock-at-aal-cutoff",
        "AAL",
        "quote",
        cutoff,
        {"bid": 20.0, "ask": 20.2},
    )
    rgti_prefix = _event(
        "prefix-rgti-after-aal-cutoff",
        "RGTI",
        "bar_5m_session_prefix",
        cutoff + 1,
        _prefix_payload(symbol="RGTI", session="2026-08-10"),
    )

    combined_batch = _ordinary_batch(
        (clock, rgti_prefix),
        retained=triggered.retained_input_event_ids,
    )
    assert M1CQuietStateOptionsV0().select_input_prefix(combined_batch, triggered.state) == 1
    combined_clock = M1CQuietStateOptionsV0().evaluate(
        _ordinary_batch(
            (clock,),
            retained=triggered.retained_input_event_ids,
        ),
        triggered.state,
    )
    combined = M1CQuietStateOptionsV0().evaluate(
        _ordinary_batch(
            (rgti_prefix,),
            retained=combined_clock.retained_input_event_ids,
        ),
        combined_clock.state,
    )
    split_clock = M1CQuietStateOptionsV0().evaluate(
        _ordinary_batch((clock,), retained=triggered.retained_input_event_ids),
        triggered.state,
    )
    split_prefix = M1CQuietStateOptionsV0().evaluate(
        _ordinary_batch(
            (rgti_prefix,),
            retained=split_clock.retained_input_event_ids,
        ),
        split_clock.state,
    )

    assert combined.state == split_prefix.state
    assert combined.state["pending"]["y"] == "RGTI"
    combined_payloads = sorted(
        (output.subject_instrument_id, output.payload.get("reason"))
        for output in (*combined_clock.outputs, *combined.outputs)
    )
    split_payloads = sorted(
        (output.subject_instrument_id, output.payload.get("reason"))
        for output in (*split_clock.outputs, *split_prefix.outputs)
    )
    assert combined_payloads == split_payloads
    timeout_index = next(
        index
        for index, output in enumerate(combined_clock.outputs)
        if output.payload.get("reason") == "underlying_reference_quote_unavailable"
    )
    timeout = combined_clock.outputs[timeout_index]
    assert timeout.payload["cutoff_at_us"] == cutoff
    assert timeout.payload["cutoff_crossing_event_id"] == clock.event_id
    assert {
        "baseline-AAL",
        "capture-AAL-call",
        "capture-AAL-put",
        aal_prefix.event_id,
        clock.event_id,
    }.issubset(combined_clock.output_input_event_ids[timeout_index])

    cutoff_prefix = _replace_event(
        rgti_prefix,
        event_id="prefix-rgti-at-aal-cutoff",
        event_at_us=cutoff,
        received_at_us=cutoff,
    )
    same_event = M1CQuietStateOptionsV0().evaluate(
        _ordinary_batch((cutoff_prefix,), retained=triggered.retained_input_event_ids),
        triggered.state,
    )
    assert tuple(output.subject_instrument_id for output in same_event.outputs) == ("AAL", "RGTI")
    assert same_event.outputs[0].payload["reason"] == "underlying_reference_quote_unavailable"
    assert same_event.outputs[1].payload.get("reason") is None
    assert same_event.state["pending"]["y"] == "RGTI"


def test_pending_quote_stage_is_committed_before_a_later_cutoff_in_the_same_fetch() -> None:
    plugin = M1CQuietStateOptionsV0()
    state, retained = _prime_low_tail_context(plugin)
    prefix = _event(
        "prefix-rgti-pending-stage",
        "RGTI",
        "bar_5m_session_prefix",
        2_000_000_000,
        _prefix_payload(symbol="RGTI", session="2026-08-10"),
    )
    triggered = plugin.evaluate(_ordinary_batch((prefix,), retained=retained), state)
    quote = _event(
        "quote-before-panel-cutoff",
        "RGTI",
        "quote",
        2_000_000_001,
        {"bid": 40.0, "ask": 40.2},
    )
    cutoff_event = _event(
        "clock-after-entry-stage",
        "AAL",
        "quote",
        5_600_000_000,
        {"bid": 20.0, "ask": 20.2},
    )
    fetched = _ordinary_batch(
        (quote, cutoff_event),
        retained=triggered.retained_input_event_ids,
    )

    assert plugin.select_input_prefix(fetched, triggered.state) == 1
    entry = plugin.evaluate(
        _ordinary_batch((quote,), retained=triggered.retained_input_event_ids),
        triggered.state,
    )
    assert len(entry.interests) == 54
    assert entry.state["active"]["g"] == "entry"

    timed_out = M1CQuietStateOptionsV0().evaluate(
        _ordinary_batch((cutoff_event,), retained=entry.retained_input_event_ids),
        entry.state,
    )
    assert timed_out.interests == ()
    assert timed_out.outputs[0].payload["reason"] == "option_panel_capture_missing"
    assert timed_out.state["active"] == {}


def test_stream_window_completes_only_with_exact_in_window_quote_proof_after_restart() -> None:
    plugin, planned, _prefix, _quote = _planned_entry_panel()
    receipts, captures = _entry_panel_evidence(planned)
    completed = plugin.evaluate(
        _ordinary_batch(
            captures,
            retained=planned.retained_input_event_ids,
            receipts=receipts,
        ),
        planned.state,
    )
    stream_receipts, quotes, deadline = _stream_evidence(completed)
    clock = _event(
        "stream-window-cutoff",
        "RGTI",
        "quote",
        deadline,
        {"bid": 40.0, "ask": 40.2},
    )

    observed = M1CQuietStateOptionsV0().evaluate(
        _ordinary_batch(
            quotes,
            retained=completed.retained_input_event_ids,
            receipts=stream_receipts,
        ),
        completed.state,
    )
    assert observed.outputs == ()
    assert len(observed.state_json()) < 65_536
    assert len(observed.retained_input_event_ids) <= 256

    terminal = M1CQuietStateOptionsV0().evaluate(
        _ordinary_batch(
            (clock,),
            retained=observed.retained_input_event_ids,
            receipts=stream_receipts,
        ),
        observed.state,
    )

    assert len(terminal.outputs) == 1
    payload = terminal.outputs[0].payload
    assert payload["status"] == "complete"
    assert payload["reason"] is None
    assert payload["resolved_stream_count"] == len(completed.interests) == 24
    assert len(payload["quote_proof_event_ids"]) == 24
    assert payload["cutoff_crossing_event_id"] == clock.event_id
    assert {event.event_id for event in (*quotes, clock)}.issubset(
        terminal.output_input_event_ids[0]
    )
    assert terminal.state["active"] == {}


def test_first_stream_quote_proof_is_checkpointed_without_per_tick_boundaries() -> None:
    plugin, planned, _prefix, _quote = _planned_entry_panel()
    receipts, captures = _entry_panel_evidence(planned)
    completed = plugin.evaluate(
        _ordinary_batch(
            captures,
            retained=planned.retained_input_event_ids,
            receipts=receipts,
        ),
        planned.state,
    )
    stream_receipts, quotes, deadline = _stream_evidence(completed)
    receipt_clock = _event(
        "stream-receipt-clock",
        "RGTI",
        "quote",
        max(receipt.completed_at_us for receipt in stream_receipts) + 1,
        {"bid": 40.0, "ask": 40.2},
    )
    statuses = plugin.evaluate(
        _ordinary_batch(
            (receipt_clock,),
            retained=completed.retained_input_event_ids,
            receipts=stream_receipts,
        ),
        completed.state,
    )
    cutoff = _event(
        "stream-cutoff-after-first-proof",
        "RGTI",
        "quote",
        deadline,
        {"bid": 40.0, "ask": 40.2},
    )
    assert (
        plugin.select_input_prefix(
            _ordinary_batch(
                (quotes[0], cutoff),
                retained=completed.retained_input_event_ids,
                receipts=stream_receipts,
            ),
            completed.state,
        )
        == 1
    )
    fetched = _ordinary_batch(
        (quotes[0], cutoff),
        retained=statuses.retained_input_event_ids,
        receipts=stream_receipts,
    )

    assert plugin.select_input_prefix(fetched, statuses.state) == 1
    proved = plugin.evaluate(
        _ordinary_batch(
            (quotes[0],),
            retained=statuses.retained_input_event_ids,
            receipts=stream_receipts,
        ),
        statuses.state,
    )
    assert len(proved.state["active"]["z"]) == 1

    duplicates = tuple(
        _replace_event(
            quotes[0],
            event_id=f"duplicate-stream-quote-{index}",
            event_at_us=quotes[0].event_at_us + index + 1,
            received_at_us=quotes[0].received_at_us + index + 1,
        )
        for index in range(5)
    )
    duplicate_batch = _ordinary_batch(
        duplicates,
        retained=proved.retained_input_event_ids,
        receipts=stream_receipts,
    )
    assert plugin.select_input_prefix(duplicate_batch, proved.state) == len(duplicates)


def test_stream_receipt_after_cutoff_cannot_pollute_terminal_attribution() -> None:
    plugin, planned, _prefix, _quote = _planned_entry_panel()
    receipts, captures = _entry_panel_evidence(planned)
    completed = plugin.evaluate(
        _ordinary_batch(
            captures,
            retained=planned.retained_input_event_ids,
            receipts=receipts,
        ),
        planned.state,
    )
    completed_active = cast(
        Mapping[str, object], cast(Mapping[str, object], completed.state)["active"]
    )
    deadline = cast(int, completed_active["x"])
    stream_receipts, _quotes, _deadline = _stream_evidence(completed)
    key = stream_receipts[0].interest_key
    late_receipt = DiscoveryReceipt.model_validate(
        {
            **stream_receipts[0].model_dump(mode="python"),
            "receipt_id": "stream-resolved-after-cutoff",
            "interest_id": "stream-interest-after-cutoff",
            "completed_at_us": deadline + 1,
        }
    )
    cutoff = _event(
        "stream-receipt-cutoff",
        "RGTI",
        "quote",
        deadline,
        {"bid": 40.0, "ask": 40.2},
    )
    after = _event(
        "stream-receipt-after-cutoff-event",
        "RGTI",
        "quote",
        deadline + 1,
        {"bid": 40.0, "ask": 40.2},
    )

    combined = plugin.evaluate(
        _ordinary_batch(
            (cutoff, after),
            retained=completed.retained_input_event_ids,
            receipts=(late_receipt,),
        ),
        completed.state,
    )
    split = M1CQuietStateOptionsV0().evaluate(
        _ordinary_batch((cutoff,), retained=completed.retained_input_event_ids),
        completed.state,
    )

    assert combined.outputs[0].payload == split.outputs[0].payload
    assert combined.outputs[0].payload["stream_interest_statuses"][key] == {
        "status": "missing",
        "reason": None,
        "completed_at_us": None,
    }
    assert after.event_id not in combined.output_input_event_ids[0]


@pytest.mark.parametrize(
    "fault", ("missing", "pre_receipt", "wrong_instrument", "crossed", "at_cutoff")
)
def test_stream_window_rejects_receipt_only_or_invalid_quote_evidence(fault: str) -> None:
    plugin, planned, _prefix, _quote = _planned_entry_panel()
    receipts, captures = _entry_panel_evidence(planned)
    completed = plugin.evaluate(
        _ordinary_batch(
            captures,
            retained=planned.retained_input_event_ids,
            receipts=receipts,
        ),
        planned.state,
    )
    stream_receipts, valid_quotes, deadline = _stream_evidence(completed)
    selected_quotes = list(valid_quotes)
    first_receipt = stream_receipts[0]
    first_quote = selected_quotes[0]
    if fault == "missing":
        selected_quotes = []
    elif fault == "pre_receipt":
        selected_quotes[0] = _replace_event(
            first_quote,
            event_at_us=first_receipt.completed_at_us - 1,
            received_at_us=first_receipt.completed_at_us - 1,
        )
    elif fault == "wrong_instrument":
        selected_quotes[0] = _replace_event(first_quote, instrument_id="wrong-option")
    elif fault == "crossed":
        selected_quotes[0] = _replace_event(
            first_quote,
            payload={"bid": 0.70, "ask": 0.60},
        )
    else:
        selected_quotes[0] = _replace_event(
            first_quote,
            event_at_us=deadline,
            received_at_us=deadline,
        )
    clock = _event(
        f"stream-cutoff-{fault}",
        "RGTI",
        "quote",
        deadline,
        {"bid": 40.0, "ask": 40.2},
    )

    terminal = M1CQuietStateOptionsV0().evaluate(
        _ordinary_batch(
            (*selected_quotes, clock),
            retained=completed.retained_input_event_ids,
            receipts=stream_receipts,
        ),
        completed.state,
    )

    assert len(terminal.outputs) == 1
    assert terminal.outputs[0].payload["status"] == "incomplete"
    assert terminal.outputs[0].payload["reason"] == "selected_leg_stream_window_incomplete"
    expected = 0 if fault in {"missing", "at_cutoff"} else 23
    assert terminal.outputs[0].payload["resolved_stream_count"] == expected
    assert terminal.outputs[0].payload["cutoff_crossing_event_id"] in {
        clock.event_id,
        first_quote.event_id,
    }
    assert terminal.state["active"] == {}


def test_stream_quotes_after_the_cutoff_event_cannot_be_admitted_by_backdating() -> None:
    plugin, planned, _prefix, _quote = _planned_entry_panel()
    receipts, captures = _entry_panel_evidence(planned)
    completed = plugin.evaluate(
        _ordinary_batch(
            captures,
            retained=planned.retained_input_event_ids,
            receipts=receipts,
        ),
        planned.state,
    )
    stream_receipts, quotes, deadline = _stream_evidence(completed)
    cutoff = _event(
        "stream-cutoff-before-backdated-quotes",
        "RGTI",
        "quote",
        deadline,
        {"bid": 40.0, "ask": 40.2},
    )

    rejected = M1CQuietStateOptionsV0().evaluate(
        _ordinary_batch(
            (cutoff, *quotes),
            retained=completed.retained_input_event_ids,
            receipts=stream_receipts,
        ),
        completed.state,
    )

    assert len(rejected.outputs) == 1
    assert rejected.outputs[0].payload["status"] == "incomplete"
    assert rejected.outputs[0].payload["resolved_stream_count"] == 0
    assert rejected.outputs[0].payload["cutoff_crossing_event_id"] == cutoff.event_id
    assert rejected.state["active"] == {}


def test_denied_selected_leg_stream_is_recorded_once_as_incomplete() -> None:
    plugin, planned, _prefix, _quote = _planned_entry_panel()
    receipts, captures = _entry_panel_evidence(planned)
    completed = plugin.evaluate(
        _ordinary_batch(
            captures,
            retained=planned.retained_input_event_ids,
            receipts=receipts,
        ),
        planned.state,
    )
    denied_interest = completed.interests[0]
    denied = DiscoveryReceipt(
        receipt_id="stream-denied",
        interest_id="stream-interest-denied",
        interest_key=denied_interest.interest_key,
        instance_id="quiet-1",
        status="denied",
        reason_code="line_capacity",
        candidates_inspected=0,
        completed_at_us=2_000_001_000,
    )
    completed_state = cast(Mapping[str, object], completed.state)
    active_state = cast(Mapping[str, object], completed_state["active"])
    stream_deadline = cast(int, active_state["x"])
    heartbeat = _event(
        "quote-rgti-heartbeat",
        "RGTI",
        "quote",
        stream_deadline,
        {"bid": 40.0, "ask": 40.2},
    )

    reported = plugin.evaluate(
        _ordinary_batch(
            (heartbeat,),
            retained=completed.retained_input_event_ids,
            receipts=(denied,),
        ),
        completed.state,
    )

    assert len(reported.outputs) == 1
    assert reported.outputs[0].kind == "observation"
    assert reported.outputs[0].payload["status"] == "incomplete"
    assert reported.outputs[0].payload["reason"] == "selected_leg_stream_window_incomplete"
    assert reported.outputs[0].payload["stream_interest_statuses"][
        denied_interest.interest_key
    ] == {
        "status": "denied",
        "reason": "line_capacity",
        "completed_at_us": 2_000_001_000,
    }
    assert reported.outputs[0].payload["cutoff_crossing_event_id"] == heartbeat.event_id
    assert heartbeat.event_id in reported.output_input_event_ids[0]
    assert reported.interests == ()

    replayed = M1CQuietStateOptionsV0().evaluate(
        _ordinary_batch(
            (
                _event(
                    "quote-rgti-heartbeat-2",
                    "RGTI",
                    "quote",
                    stream_deadline + 1,
                    {"bid": 40.0, "ask": 40.2},
                ),
            ),
            retained=reported.retained_input_event_ids,
            receipts=(denied,),
        ),
        reported.state,
    )
    assert replayed.outputs == ()


def test_quiet_state_uses_the_frozen_inclusive_downward_crossing() -> None:
    exact = classify_quiet_state(
        probability=BOTTOM_10_THRESHOLD,
        previous_probability=BOTTOM_10_THRESHOLD + 0.01,
        minutes_since_previous_episode=None,
    )
    already_quiet = classify_quiet_state(
        probability=BOTTOM_10_THRESHOLD - 0.01,
        previous_probability=BOTTOM_10_THRESHOLD,
        minutes_since_previous_episode=45.0,
    )
    too_close = classify_quiet_state(
        probability=BOTTOM_10_THRESHOLD - 0.01,
        previous_probability=BOTTOM_10_THRESHOLD + 0.01,
        minutes_since_previous_episode=29.999,
    )

    assert exact.bottom_10 is True
    assert exact.fresh_episode is True
    assert already_quiet.fresh_episode is False
    assert too_close.fresh_episode is False


def test_quiet_episode_spacing_uses_market_bar_time_across_restart_and_receipt_skew() -> None:
    plugin = M1CQuietStateOptionsV0()
    state, retained = _prime_low_tail_context(plugin, all_cohort_low=True)
    seeded = dict(cast(Mapping[str, JsonValue], state))
    seeded["episodes"] = {
        "RGTI": {
            "s": "2026-08-10",
            "p": 0.20,
            "l": 800_000_000,
            "c": 1,
        }
    }
    delayed = _event(
        "prefix-rgti-delayed-before-spacing",
        "RGTI",
        "bar_5m_session_prefix",
        2_000_000_000,
        _prefix_payload(symbol="RGTI", session="2026-08-10"),
    )
    delayed = _replace_event(delayed, received_at_us=3_860_000_000)

    before_spacing = M1CQuietStateOptionsV0().evaluate(
        _ordinary_batch((delayed,), retained=retained),
        cast(JsonValue, seeded),
    )

    assert before_spacing.outputs[0].payload["fresh_episode"] is False
    assert before_spacing.state["episodes"]["RGTI"]["l"] == 800_000_000
    assert before_spacing.state["pending"] == {}

    at_spacing = _replace_event(
        delayed,
        event_id="prefix-rgti-delayed-at-spacing",
        event_at_us=2_600_000_000,
        received_at_us=4_000_000_000,
    )
    admitted = M1CQuietStateOptionsV0().evaluate(
        _ordinary_batch((at_spacing,), retained=retained),
        cast(JsonValue, seeded),
    )
    assert admitted.outputs[0].payload["fresh_episode"] is True
    assert admitted.state["episodes"]["RGTI"]["l"] == 2_600_000_000
    assert admitted.state["pending"]["y"] == "RGTI"


def test_defined_risk_selection_matches_the_frozen_protective_contracts() -> None:
    attempts = select_defined_risk_structures(
        contracts=_bucket_contracts(),
        underlying_reference_price=100.0,
    )

    assert tuple(attempt.structure_type for attempt in attempts) == (
        "ATM_IRON_BUTTERFLY",
        "DELTA_IRON_CONDOR",
        "CALL_CREDIT_SPREAD",
        "PUT_CREDIT_SPREAD",
    )
    assert all(attempt.available for attempt in attempts)
    assert tuple(
        (leg.side, leg.contract.offset, leg.contract.right) for leg in attempts[0].legs
    ) == (
        ("short", 0, "call"),
        ("short", 0, "put"),
        ("long", 1, "call"),
        ("long", -1, "put"),
    )
    assert tuple(
        (leg.side, leg.contract.offset, leg.contract.right) for leg in attempts[1].legs
    ) == (
        ("short", 2, "call"),
        ("short", -2, "put"),
        ("long", 3, "call"),
        ("long", -3, "put"),
    )
    assert all(
        any(leg.side == "short" for leg in attempt.legs)
        and any(leg.side == "long" for leg in attempt.legs)
        for attempt in attempts
    )


def test_short_premium_selection_never_emits_an_unprotected_attempt() -> None:
    contracts = tuple(contract for contract in _bucket_contracts() if contract.offset == 0)

    attempts = select_defined_risk_structures(
        contracts=contracts,
        underlying_reference_price=100.0,
    )

    assert all(not attempt.available for attempt in attempts)
    assert all(attempt.legs == () for attempt in attempts)
    assert {attempt.reason for attempt in attempts} == {
        "symmetric_wings_unavailable",
        "delta_tolerance_failed",
        "protective_wing_unavailable",
    }


def test_non_credit_entry_never_becomes_a_short_premium_proposal_attempt() -> None:
    contracts = tuple(replace(contract, bid=0.1, ask=1.0) for contract in _bucket_contracts())

    attempts = select_defined_risk_structures(
        contracts=contracts,
        underlying_reference_price=100.0,
    )

    assert all(not attempt.available for attempt in attempts)
    assert {attempt.reason for attempt in attempts} == {"non_positive_opening_credit"}
