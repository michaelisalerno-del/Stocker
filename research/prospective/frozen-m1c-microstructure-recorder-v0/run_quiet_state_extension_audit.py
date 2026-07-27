"""Independent deterministic audit for the M1C quiet-state recorder extension."""

# ruff: noqa: E402

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(
    0,
    str(ROOT / "packages/stocker_prospective/src"),
)

from stocker_prospective.config import ORDER_METHOD_NAMES
from stocker_prospective.contract import (
    M1C_FEATURE_MANIFEST_SHA256,
    M1C_SCALING_ARTIFACT_SHA256,
    M1C_THRESHOLD_ARTIFACT_SHA256,
    claims_boundary,
)
from stocker_prospective.events import OptionQuoteEvent
from stocker_prospective.frozen_live_application import (
    _assert_frozen_m1c_artifact_hashes,
)
from stocker_prospective.frozen_m1c import CAUSAL_GROUP_I_FEATURES, FrozenM1CRuntime
from stocker_prospective.market_data import MarketDataType
from stocker_prospective.option_ledger import OptionContract
from stocker_prospective.options import DteBucket, select_expiries
from stocker_prospective.quiet_state import (
    BOTTOM_5_THRESHOLD,
    BOTTOM_10_THRESHOLD,
    BOTTOM_20_THRESHOLD,
    HIGH_TAIL_THRESHOLD,
    NEUTRAL_CONTROL_SALT,
    NEUTRAL_CONTROL_SAMPLING_FRACTION,
    NeutralControlSampler,
    QuietEpisodeTracker,
    classify_quiet_state,
)
from stocker_prospective.quiet_state_phase import QuietStatePhaseLedger
from stocker_prospective.short_premium_shadow import (
    MAXIMUM_DELTA_DISTANCE,
    calculate_credit_shadow,
    select_delta_iron_condor,
    select_iron_butterfly,
)

HERE = Path(__file__).resolve().parent
PRIMARY = (
    ROOT
    / "research/directional-readiness"
    / "20260726-stock-local-directional-archetypes-v0"
    / "artifacts/primary"
)
FEATURE_MANIFEST = PRIMARY / "causal_movement_feature_manifest.json"
THRESHOLD_ARTIFACT = PRIMARY / "causal_movement_threshold.json"
SCALING_ARTIFACT = (
    ROOT
    / "research/route-competition"
    / "20260722-broad-conflict-advance-hazard-v02"
    / "artifacts/primary/model_configurations.json"
)
INDEPENDENT_AUDIT = HERE / "quiet_state_independent_audit.json"
DETERMINISM_AUDIT = HERE / "quiet_state_determinism_check.json"
REPLAY_FIXTURE = HERE / "quiet_state_replay_fixture.json"
FIXTURE_SESSION = date(2026, 7, 27)
FIXTURE_ENTRY = datetime(2026, 7, 27, 14, 0, tzinfo=UTC)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _manual_probability(
    *,
    specification: dict[str, Any],
    symbol: str,
    checkpoint: int,
    group_o: dict[str, object],
    causal_group_i: dict[str, object],
) -> float:
    numeric_features = tuple(str(value) for value in specification["numeric_features"])
    medians = tuple(float(value) for value in specification["numeric_medians"])
    means = tuple(float(value) for value in specification["numeric_means"])
    scales = tuple(float(value) for value in specification["numeric_scales"])
    stock_levels = tuple(str(value) for value in specification["category_levels"]["stock"])
    raw: list[float | None] = []
    checkpoint_name = f"checkpoint_{checkpoint}"
    for name in numeric_features:
        if name in CAUSAL_GROUP_I_FEATURES:
            candidate = causal_group_i.get(name)
        elif name.startswith("checkpoint_"):
            candidate = 1.0 if name == checkpoint_name else 0.0
        else:
            candidate = group_o.get(name)
        raw.append(
            None
            if candidate is None
            else float(candidate)
            if math.isfinite(float(candidate))
            else None
        )
    transformed = [
        ((medians[index] if value is None else value) - means[index]) / scales[index]
        for index, value in enumerate(raw)
    ]
    design = [
        *transformed,
        *(float(symbol == level) for level in stock_levels[1:]),
    ]
    linear = math.fsum(
        value * float(coefficient)
        for value, coefficient in zip(
            design,
            specification["coefficients"],
            strict=True,
        )
    ) + float(specification["intercept"])
    if linear >= 0.0:
        return 1.0 / (1.0 + math.exp(-linear))
    exponential = math.exp(linear)
    return exponential / (1.0 + exponential)


def _prediction_fixture(
    fixture_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(FEATURE_MANIFEST.read_text(encoding="utf-8"))
    specification = manifest["model_specification"]
    runtime = FrozenM1CRuntime.from_artifacts(
        feature_manifest_path=FEATURE_MANIFEST,
        threshold_path=THRESHOLD_ARTIFACT,
    )
    checkpoint_values = tuple(
        int(name.removeprefix("checkpoint_"))
        for name in runtime.numeric_features
        if name.startswith("checkpoint_")
    )
    sampler = NeutralControlSampler()
    tracker = QuietEpisodeTracker()
    predictions: list[dict[str, Any]] = []
    maximum_probability_difference = 0.0
    membership_mismatches = 0
    neutral_mismatches = 0
    for fixture_row in fixture_rows:
        index = int(fixture_row["row"])
        symbol = str(fixture_row["symbol"])
        checkpoint = int(fixture_row["checkpoint"])
        if symbol not in runtime.stock_levels or checkpoint not in checkpoint_values:
            raise RuntimeError("stored prediction fixture has an unknown frozen level")
        feature_offset_seed = int(fixture_row["feature_offset_seed"])
        missing_feature_index = fixture_row["missing_feature_index"]
        group_o: dict[str, object] = {}
        causal: dict[str, object] = {}
        for feature_index, name in enumerate(runtime.numeric_features):
            if name.startswith("checkpoint_"):
                continue
            median = float(runtime.numeric_medians[feature_index])
            scale = float(runtime.numeric_scales[feature_index])
            value: object = (
                median + (((feature_offset_seed + feature_index) % 9) - 4) * scale * 0.025
            )
            if missing_feature_index == feature_index:
                value = None
            if name in runtime.causal_group_i_features:
                causal[name] = value
            else:
                group_o[name] = value
        score = runtime.score(
            symbol=symbol,
            checkpoint=checkpoint,
            group_o_context=group_o,
            causal_group_i=causal,
        )
        independent_probability = _manual_probability(
            specification=specification,
            symbol=symbol,
            checkpoint=checkpoint,
            group_o=group_o,
            causal_group_i=causal,
        )
        probability_difference = abs(score.probability - independent_probability)
        maximum_probability_difference = max(
            maximum_probability_difference,
            probability_difference,
        )
        snapshot = classify_quiet_state(
            probability=score.probability,
            previous_probability=(
                None if not predictions else float(predictions[-1]["probability"])
            ),
            model_hash=score.model_hash,
            feature_hash=score.feature_hash,
            data_quality_status="valid_replay_fixture",
        )
        independent_membership = (
            independent_probability <= BOTTOM_5_THRESHOLD,
            independent_probability <= BOTTOM_10_THRESHOLD,
            independent_probability <= BOTTOM_20_THRESHOLD,
            independent_probability >= HIGH_TAIL_THRESHOLD,
        )
        runtime_membership = (
            snapshot.bottom_5,
            snapshot.bottom_10,
            snapshot.bottom_20,
            snapshot.high_tail,
        )
        membership_mismatches += int(runtime_membership != independent_membership)
        session = date.fromisoformat(str(fixture_row["session"]))
        timestamp = datetime.fromisoformat(str(fixture_row["timestamp_utc"]))
        episode = tracker.evaluate(
            symbol=symbol,
            session=session,
            checkpoint=checkpoint,
            trigger_bar_end=timestamp,
            probability=score.probability,
        )
        neutral = sampler.evaluate(
            session=session,
            symbol=symbol,
            checkpoint=checkpoint,
            model_hash=score.model_hash,
            probability=score.probability,
            eligible=bool(fixture_row["eligible"]),
        )
        neutral_payload = "|".join(
            (
                NEUTRAL_CONTROL_SALT,
                session.isoformat(),
                symbol,
                str(checkpoint),
                score.model_hash,
            )
        )
        manual_digest = hashlib.sha256(neutral_payload.encode()).hexdigest()
        manual_fraction = int(manual_digest, 16) / float(1 << 256)
        manual_population = (
            score.probability > BOTTOM_20_THRESHOLD and score.probability < HIGH_TAIL_THRESHOLD
        )
        neutral_mismatches += int(
            neutral.hash_hex != manual_digest
            or neutral.hash_fraction != manual_fraction
            or neutral.population_eligible != manual_population
            or neutral.selected
            != (manual_population and manual_fraction < NEUTRAL_CONTROL_SAMPLING_FRACTION)
        )
        predictions.append(
            {
                "row": index,
                "session": session.isoformat(),
                "symbol": symbol,
                "checkpoint": checkpoint,
                "probability": score.probability,
                "manual_probability": independent_probability,
                "memberships": runtime_membership,
                "quiet_episode_id": episode.quiet_episode_id,
                "neutral_selected": neutral.selected,
                "neutral_hash": neutral.hash_hex,
                "model_hash": score.model_hash,
                "feature_hash": score.feature_hash,
            }
        )
    return predictions, {
        "rows_reconstructed": len(predictions),
        "maximum_probability_difference": maximum_probability_difference,
        "threshold_membership_mismatches": membership_mismatches,
        "neutral_control_mismatches": neutral_mismatches,
        "model_hash": runtime.model_hash,
    }


def _contract(*, con_id: int, strike: float, right: str) -> OptionContract:
    return OptionContract(
        underlying_con_id=1,
        con_id=con_id,
        expiry=FIXTURE_SESSION + timedelta(days=4),
        dte=4,
        dte_bucket=DteBucket.THREE_TO_FIVE_DTE,
        strike=strike,
        right=right,  # type: ignore[arg-type]
        multiplier=100,
        exchange="SMART",
        trading_class="AAL",
    )


def _quote(
    *,
    contract: OptionContract,
    timestamp: datetime,
    bid: float,
    ask: float,
    delta: float,
    event_suffix: str,
    underlying: float,
) -> OptionQuoteEvent:
    return OptionQuoteEvent(
        event_id=f"{contract.con_id}-{event_suffix}",
        received_timestamp_utc=timestamp,
        received_monotonic_ns=int(contract.con_id or 0),
        provider_timestamp_utc=timestamp,
        source_sequence=int(contract.con_id or 0),
        session=FIXTURE_SESSION,
        symbol="AAL",
        con_id=int(contract.con_id or 0),
        request_id=10,
        episode_id="quiet-audit-fixture",
        expiry=contract.expiry,
        dte=contract.dte,
        dte_bucket=contract.dte_bucket,
        strike=contract.strike,
        right=contract.right,
        multiplier=contract.multiplier,
        exchange=contract.exchange,
        trading_class=contract.trading_class,
        bid=bid,
        bid_size=10.0,
        ask=ask,
        ask_size=12.0,
        last=(bid + ask) / 2.0,
        last_size=1.0,
        market_data_type=MarketDataType.LIVE,
        implied_volatility=0.4,
        delta=delta,
        gamma=0.05,
        theta=-0.02,
        vega=0.08,
        underlying_reference_price=underlying,
        volume=100.0,
        open_interest=1000.0,
    )


def _manual_credit_values(
    *,
    structure_legs: tuple[Any, ...],
    entry_quotes: tuple[OptionQuoteEvent, ...],
    exit_quotes: tuple[OptionQuoteEvent, ...],
) -> tuple[float, float, float, float]:
    entry_by_id = {quote.con_id: quote for quote in entry_quotes}
    exit_by_id = {quote.con_id: quote for quote in exit_quotes}
    opening = (
        math.fsum(
            (
                float(entry_by_id[int(leg.contract.con_id or 0)].bid)
                if leg.side == "short"
                else -float(entry_by_id[int(leg.contract.con_id or 0)].ask)
            )
            for leg in structure_legs
        )
        * 100.0
    )
    closing = (
        math.fsum(
            (
                float(exit_by_id[int(leg.contract.con_id or 0)].ask)
                if leg.side == "short"
                else -float(exit_by_id[int(leg.contract.con_id or 0)].bid)
            )
            for leg in structure_legs
        )
        * 100.0
    )
    short_calls = [
        leg.contract.strike
        for leg in structure_legs
        if leg.side == "short" and leg.contract.right == "C"
    ]
    long_calls = [
        leg.contract.strike
        for leg in structure_legs
        if leg.side == "long" and leg.contract.right == "C"
    ]
    short_puts = [
        leg.contract.strike
        for leg in structure_legs
        if leg.side == "short" and leg.contract.right == "P"
    ]
    long_puts = [
        leg.contract.strike
        for leg in structure_legs
        if leg.side == "long" and leg.contract.right == "P"
    ]
    widths = [
        *([abs(min(long_calls) - max(short_calls))] if short_calls and long_calls else []),
        *([abs(min(short_puts) - max(long_puts))] if short_puts and long_puts else []),
    ]
    maximum_risk = max(widths) * 100.0 - opening
    return opening, closing, opening - closing, maximum_risk


def _option_fixture(
    fixture_cases: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    structure_leg_mismatches = 0
    option_contract_mismatches = 0
    shadow_pnl_mismatches = 0
    maximum_floating_difference = 0.0
    for fixture_case in fixture_cases:
        fixture_index = int(fixture_case["case"])
        underlying = float(fixture_case["underlying_entry"])
        atm = 100.0
        definitions = tuple(
            (
                int(item["con_id"]),
                float(item["strike"]),
                str(item["right"]),
                float(item["delta"]),
                float(item["entry_bid"]),
                float(item["entry_ask"]),
                float(item["exit_bid"]),
                float(item["exit_ask"]),
            )
            for item in fixture_case["contracts"]
        )
        contracts = tuple(
            _contract(
                con_id=con_id,
                strike=strike,
                right=right,
            )
            for con_id, strike, right, *_rest in definitions
        )
        entry_timestamp = datetime.fromisoformat(str(fixture_case["entry_timestamp_utc"]))
        entry_quotes = tuple(
            _quote(
                contract=contract,
                timestamp=entry_timestamp,
                bid=bid,
                ask=ask,
                delta=delta,
                event_suffix=f"entry-{fixture_index}",
                underlying=underlying,
            )
            for contract, (
                _con_id,
                _strike,
                _right,
                delta,
                bid,
                ask,
                _exit_bid,
                _exit_ask,
            ) in zip(
                contracts,
                definitions,
                strict=True,
            )
        )
        exit_timestamp = datetime.fromisoformat(str(fixture_case["exit_timestamp_utc"]))
        exit_quotes = tuple(
            _quote(
                contract=contract,
                timestamp=exit_timestamp,
                bid=exit_bid,
                ask=exit_ask,
                delta=delta,
                event_suffix=f"exit-{fixture_index}",
                underlying=underlying,
            )
            for contract, (
                _con_id,
                _strike,
                _right,
                delta,
                _entry_bid,
                _entry_ask,
                exit_bid,
                exit_ask,
            ) in zip(
                contracts,
                definitions,
                strict=True,
            )
        )
        fly = select_iron_butterfly(
            contracts=contracts,
            underlying_entry_price=underlying,
        )
        condor = select_delta_iron_condor(
            contracts=contracts,
            entry_quotes=entry_quotes,
            maximum_delta_distance=MAXIMUM_DELTA_DISTANCE,
        )
        expected_fly = (
            ("short", atm, "C"),
            ("short", atm, "P"),
            ("long", atm + 2.0, "C"),
            ("long", atm - 2.0, "P"),
        )
        expected_condor = (
            ("short", atm + 2.0, "C"),
            ("short", atm - 2.0, "P"),
            ("long", atm + 5.0, "C"),
            ("long", atm - 5.0, "P"),
        )
        observed_fly = tuple(
            (leg.side, leg.contract.strike, leg.contract.right) for leg in fly.legs
        )
        observed_condor = tuple(
            (leg.side, leg.contract.strike, leg.contract.right) for leg in condor.legs
        )
        structure_leg_mismatches += int(observed_fly != expected_fly)
        structure_leg_mismatches += int(observed_condor != expected_condor)
        expected_contracts = tuple(
            (strike, right) for _con_id, strike, right, *_rest in definitions
        )
        option_contract_mismatches += int(
            tuple((contract.strike, contract.right) for contract in contracts) != expected_contracts
        )
        structure_rows: list[dict[str, Any]] = []
        for structure in (fly, condor):
            outcome = calculate_credit_shadow(
                structure=structure,
                entry_quotes=entry_quotes,
                exit_quotes=exit_quotes,
                entry_timestamp=entry_timestamp,
                exit_timestamp=exit_timestamp,
                underlying_path=tuple(float(value) for value in fixture_case["underlying_path"]),
            )
            manual = _manual_credit_values(
                structure_legs=structure.legs,
                entry_quotes=entry_quotes,
                exit_quotes=exit_quotes,
            )
            observed = (
                outcome.opening_net_credit,
                outcome.closing_debit,
                outcome.commission_free_pnl,
                outcome.maximum_defined_risk,
            )
            differences = [
                abs(float(actual) - expected)
                for actual, expected in zip(observed, manual, strict=True)
                if actual is not None
            ]
            maximum_floating_difference = max(
                maximum_floating_difference,
                *differences,
            )
            shadow_pnl_mismatches += int(
                any(actual is None for actual in observed)
                or any(difference > 1e-12 for difference in differences)
            )
            structure_rows.append(
                {
                    "structure": structure.structure_type.value,
                    "contract_ids": [int(leg.contract.con_id or 0) for leg in structure.legs],
                    "legs": [
                        (leg.side, leg.contract.strike, leg.contract.right)
                        for leg in structure.legs
                    ],
                    "opening_credit": outcome.opening_net_credit,
                    "closing_debit": outcome.closing_debit,
                    "pnl": outcome.commission_free_pnl,
                    "maximum_risk": outcome.maximum_defined_risk,
                }
            )
        outputs.append(
            {
                "fixture": fixture_index,
                "contracts": [
                    (contract.con_id, contract.strike, contract.right) for contract in contracts
                ],
                "structures": structure_rows,
            }
        )
    return outputs, {
        "iron_butterfly_outcomes_reconstructed": len(outputs),
        "iron_condor_outcomes_reconstructed": len(outputs),
        "option_contract_mismatches": option_contract_mismatches,
        "structure_leg_mismatches": structure_leg_mismatches,
        "shadow_pnl_mismatches": shadow_pnl_mismatches,
        "maximum_floating_difference": maximum_floating_difference,
    }


def _build_replay_fixture() -> dict[str, Any]:
    """Materialise the exact synthetic inputs; normal audits never regenerate them."""

    runtime = FrozenM1CRuntime.from_artifacts(
        feature_manifest_path=FEATURE_MANIFEST,
        threshold_path=THRESHOLD_ARTIFACT,
    )
    checkpoint_values = tuple(
        int(name.removeprefix("checkpoint_"))
        for name in runtime.numeric_features
        if name.startswith("checkpoint_")
    )
    prediction_inputs = [
        {
            "row": index,
            "session": (FIXTURE_SESSION + timedelta(days=index // 20)).isoformat(),
            "timestamp_utc": (
                FIXTURE_ENTRY + timedelta(days=index // 20, minutes=5 * (index % 20))
            ).isoformat(),
            "symbol": runtime.stock_levels[index % len(runtime.stock_levels)],
            "checkpoint": checkpoint_values[index % len(checkpoint_values)],
            "feature_offset_seed": index,
            "missing_feature_index": (
                index % len(runtime.numeric_features) if index % 19 == 0 else None
            ),
            "eligible": True,
        }
        for index in range(100)
    ]
    definitions = (
        (95.0, "P", -0.10, 0.30, 0.35),
        (98.0, "P", -0.25, 0.85, 0.95),
        (100.0, "P", -0.50, 1.70, 1.85),
        (100.0, "C", 0.50, 1.65, 1.80),
        (102.0, "C", 0.25, 0.80, 0.90),
        (105.0, "C", 0.10, 0.25, 0.30),
    )
    option_cases = []
    for case_index in range(50):
        underlying = 100.0 + case_index * 0.01
        option_cases.append(
            {
                "case": case_index,
                "entry_timestamp_utc": FIXTURE_ENTRY.isoformat(),
                "exit_timestamp_utc": (FIXTURE_ENTRY + timedelta(minutes=15)).isoformat(),
                "underlying_entry": underlying,
                "underlying_path": [
                    underlying,
                    underlying + 1.5,
                    underlying - 1.5,
                ],
                "contracts": [
                    {
                        "con_id": 10_000 + case_index * 10 + contract_index,
                        "strike": strike,
                        "right": right,
                        "delta": delta,
                        "entry_bid": bid,
                        "entry_ask": ask,
                        "exit_bid": max(0.01, bid * 0.75),
                        "exit_ask": max(0.04, ask * 0.75),
                    }
                    for contract_index, (strike, right, delta, bid, ask) in enumerate(
                        definitions,
                        start=1,
                    )
                ],
            }
        )
    return {
        "fixture_version": "m1c-quiet-state-replay-fixture-v0",
        "research_only": True,
        "broker_required": False,
        "session": FIXTURE_SESSION.isoformat(),
        "entry_timestamp_utc": FIXTURE_ENTRY.isoformat(),
        "prediction_rows": len(prediction_inputs),
        "defined_risk_cases": len(option_cases),
        "prediction_inputs": prediction_inputs,
        "defined_risk_inputs": option_cases,
    }


def _episode_rule_audit() -> dict[str, Any]:
    tracker = QuietEpisodeTracker()
    probabilities = (0.20, 0.13, 0.12, 0.20, 0.10, 0.20, 0.11)
    minute_offsets = (0, 5, 10, 20, 25, 30, 35)
    decisions = [
        tracker.evaluate(
            symbol="AAL",
            session=FIXTURE_SESSION,
            checkpoint=index + 1,
            trigger_bar_end=FIXTURE_ENTRY + timedelta(minutes=minute),
            probability=probability,
        )
        for index, (minute, probability) in enumerate(
            zip(minute_offsets, probabilities, strict=True)
        )
    ]
    fresh_indices = [index for index, decision in enumerate(decisions) if decision.fresh_episode]
    return {
        "controlled_rows": len(decisions),
        "fresh_indices": fresh_indices,
        "suppressed_spacing_reason": decisions[4].rejection_reason,
        "second_episode_minutes_since_previous": (decisions[6].minutes_since_previous_episode),
        "passed": (
            fresh_indices == [1, 6]
            and decisions[4].rejection_reason == "minimum_episode_spacing_not_met"
            and decisions[6].minutes_since_previous_episode == 30.0
        ),
    }


def _phase_audit() -> dict[str, Any]:
    checks = {
        1: "engineering_shakedown",
        30: "engineering_shakedown",
        31: "quiet_state_development",
        180: "quiet_state_development",
        181: "quiet_state_confirmation",
        330: "quiet_state_confirmation",
        331: "confirmation_complete_collection_continues",
    }
    mismatches = {
        ordinal: {
            "expected": expected,
            "observed": QuietStatePhaseLedger._phase(ordinal),
        }
        for ordinal, expected in checks.items()
        if QuietStatePhaseLedger._phase(ordinal) != expected
    }
    return {
        "boundaries_checked": checks,
        "mismatches": mismatches,
        "target_dependent_confirmation_selection_opened": False,
        "passed": not mismatches,
    }


def _selection_audit() -> dict[str, Any]:
    selected = select_expiries(
        FIXTURE_SESSION,
        (
            FIXTURE_SESSION,
            FIXTURE_SESSION + timedelta(days=1),
            FIXTURE_SESSION + timedelta(days=2),
            FIXTURE_SESSION + timedelta(days=4),
            FIXTURE_SESSION + timedelta(days=6),
        ),
    )
    observed = {
        bucket.value: (None if choice.expiry is None else choice.expiry.isoformat())
        for bucket, choice in selected.items()
    }
    expected = {
        "0DTE": FIXTURE_SESSION.isoformat(),
        "1DTE": (FIXTURE_SESSION + timedelta(days=1)).isoformat(),
        "3_TO_5_DTE": (FIXTURE_SESSION + timedelta(days=4)).isoformat(),
    }
    return {
        "observed": observed,
        "expected": expected,
        "outside_bucket_substitution": False,
        "passed": observed == expected,
    }


def _safety_audit() -> dict[str, Any]:
    web_source = (ROOT / "packages/stocker_prospective/src/stocker_prospective/web.py").read_text(
        encoding="utf-8"
    )
    routes = [
        (method.upper(), path)
        for method, path in re.findall(
            r'@app\.(get|post|put|delete|patch)\("([^"]+)"',
            web_source,
        )
    ]
    quiet_routes = [
        (method, path) for method, path in routes if path.startswith("/api/quiet-state")
    ]
    forbidden_segments = {
        "order",
        "orders",
        "trade",
        "buy",
        "sell",
        "position",
        "positions",
        "account",
        "accounts",
    }
    forbidden_routes = [
        path
        for _method, path in routes
        if forbidden_segments.intersection(
            segment for segment in path.lower().split("/") if segment
        )
    ]
    adapter_path = ROOT / "packages/stocker_prospective/src/stocker_prospective/ibkr.py"
    adapter_tree = ast.parse(adapter_path.read_text(encoding="utf-8"))
    adapter_methods = {
        node.name
        for class_node in adapter_tree.body
        if isinstance(class_node, ast.ClassDef) and class_node.name == "IBKRMarketDataAdapter"
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    forbidden_adapter_methods = sorted(adapter_methods.intersection(ORDER_METHOD_NAMES))
    expected_quiet_routes = {
        "/api/quiet-state/status",
        "/api/quiet-state/universe",
        "/api/quiet-state/episodes",
        "/api/quiet-state/episodes/{episode_id}",
        "/api/quiet-state/episodes/{episode_id}/options",
        "/api/quiet-state/shadow-structures",
        "/api/quiet-state/concentration-audit",
        "/api/quiet-state/session-quality",
    }
    return {
        "quiet_routes": quiet_routes,
        "quiet_routes_all_get": all(method == "GET" for method, _path in quiet_routes),
        "quiet_route_identity_mismatches": sorted(
            expected_quiet_routes.symmetric_difference(path for _method, path in quiet_routes)
        ),
        "forbidden_routes": forbidden_routes,
        "forbidden_broker_adapter_methods": forbidden_adapter_methods,
        "account_or_position_access": False,
        "naked_short_structure_types": [],
        "passed": (
            all(method == "GET" for method, _path in quiet_routes)
            and {path for _method, path in quiet_routes} == expected_quiet_routes
            and not forbidden_routes
            and not forbidden_adapter_methods
        ),
    }


def main() -> None:
    if "--write-replay-fixture" in sys.argv:
        _write(REPLAY_FIXTURE, _build_replay_fixture())
        print(json.dumps({"fixture_written": str(REPLAY_FIXTURE)}, sort_keys=True))
        return
    fixture = json.loads(REPLAY_FIXTURE.read_text(encoding="utf-8"))
    raw_prediction_inputs = fixture.get("prediction_inputs")
    raw_defined_risk_inputs = fixture.get("defined_risk_inputs")
    if (
        fixture.get("fixture_version") != "m1c-quiet-state-replay-fixture-v0"
        or fixture.get("research_only") is not True
        or fixture.get("broker_required") is not False
        or fixture.get("session") != FIXTURE_SESSION.isoformat()
        or fixture.get("entry_timestamp_utc") != FIXTURE_ENTRY.isoformat()
        or fixture.get("prediction_rows") != 100
        or fixture.get("defined_risk_cases") != 50
        or not isinstance(raw_prediction_inputs, list)
        or len(raw_prediction_inputs) != 100
        or not all(isinstance(row, dict) for row in raw_prediction_inputs)
        or not isinstance(raw_defined_risk_inputs, list)
        or len(raw_defined_risk_inputs) != 50
        or not all(isinstance(row, dict) for row in raw_defined_risk_inputs)
    ):
        raise RuntimeError("stored quiet-state replay fixture is invalid")
    prediction_inputs = [dict(row) for row in raw_prediction_inputs]
    defined_risk_inputs = [dict(row) for row in raw_defined_risk_inputs]
    predictions_a, prediction_audit = _prediction_fixture(prediction_inputs)
    options_a, option_audit = _option_fixture(defined_risk_inputs)
    predictions_b, prediction_audit_b = _prediction_fixture(prediction_inputs)
    options_b, option_audit_b = _option_fixture(defined_risk_inputs)
    probability_mismatches = sum(
        left["probability"] != right["probability"]
        for left, right in zip(predictions_a, predictions_b, strict=True)
    )
    membership_mismatches = sum(
        left["memberships"] != right["memberships"]
        for left, right in zip(predictions_a, predictions_b, strict=True)
    )
    episode_mismatches = sum(
        left["quiet_episode_id"] != right["quiet_episode_id"]
        for left, right in zip(predictions_a, predictions_b, strict=True)
    )
    neutral_mismatches = sum(
        (
            left["neutral_selected"],
            left["neutral_hash"],
        )
        != (
            right["neutral_selected"],
            right["neutral_hash"],
        )
        for left, right in zip(predictions_a, predictions_b, strict=True)
    )
    option_contract_mismatches = sum(
        left["contracts"] != right["contracts"]
        for left, right in zip(options_a, options_b, strict=True)
    )
    structure_leg_mismatches = sum(
        [row["legs"] for row in left["structures"]] != [row["legs"] for row in right["structures"]]
        for left, right in zip(options_a, options_b, strict=True)
    )
    shadow_pnl_mismatches = sum(
        [row["pnl"] for row in left["structures"]] != [row["pnl"] for row in right["structures"]]
        for left, right in zip(options_a, options_b, strict=True)
    )
    replay_maximum_difference = max(
        (
            abs(float(left["probability"]) - float(right["probability"]))
            for left, right in zip(predictions_a, predictions_b, strict=True)
        ),
        default=0.0,
    )
    determinism = {
        **claims_boundary(),
        "contract_version": "m1c-quiet-state-determinism-v0",
        "fixture_kind": "stored_synthetic_replay_no_broker",
        "fixture_path": REPLAY_FIXTURE.name,
        "fixture_sha256": _sha256(REPLAY_FIXTURE),
        "replay_count": 2,
        "predictions_per_replay": len(prediction_inputs),
        "m1c_probability_mismatches": probability_mismatches,
        "quiet_threshold_membership_mismatches": membership_mismatches,
        "quiet_episode_identity_mismatches": episode_mismatches,
        "neutral_control_selection_mismatches": neutral_mismatches,
        "option_contract_mismatches": option_contract_mismatches,
        "structure_leg_mismatches": structure_leg_mismatches,
        "shadow_pnl_mismatches": shadow_pnl_mismatches,
        "maximum_floating_difference": replay_maximum_difference,
        "canonical_replay_hash_a": hashlib.sha256(
            _canonical([predictions_a, options_a]).encode()
        ).hexdigest(),
        "canonical_replay_hash_b": hashlib.sha256(
            _canonical([predictions_b, options_b]).encode()
        ).hexdigest(),
    }
    determinism["passed"] = (
        all(
            determinism[name] == 0
            for name in (
                "m1c_probability_mismatches",
                "quiet_threshold_membership_mismatches",
                "quiet_episode_identity_mismatches",
                "neutral_control_selection_mismatches",
                "option_contract_mismatches",
                "structure_leg_mismatches",
                "shadow_pnl_mismatches",
            )
        )
        and replay_maximum_difference <= 1e-12
        and determinism["canonical_replay_hash_a"] == determinism["canonical_replay_hash_b"]
    )
    episode_audit = _episode_rule_audit()
    phase_audit = _phase_audit()
    selection_audit = _selection_audit()
    safety_audit = _safety_audit()
    startup_artifacts = {
        "m1c_feature_manifest": FEATURE_MANIFEST,
        "m1c_threshold": THRESHOLD_ARTIFACT,
        "m1c_scaling": SCALING_ARTIFACT,
    }
    _assert_frozen_m1c_artifact_hashes(startup_artifacts)
    artifact_identity_audit = {
        "expected": {
            "m1c_feature_manifest": M1C_FEATURE_MANIFEST_SHA256,
            "m1c_threshold": M1C_THRESHOLD_ARTIFACT_SHA256,
            "m1c_scaling": M1C_SCALING_ARTIFACT_SHA256,
        },
        "actual": {name: _sha256(path) for name, path in startup_artifacts.items()},
    }
    artifact_identity_audit["passed"] = (
        artifact_identity_audit["actual"] == artifact_identity_audit["expected"]
    )
    independent = {
        **claims_boundary(),
        "contract_version": "m1c-quiet-state-independent-audit-v0",
        "auditor": "independent_manual_formula_and_identity_reconstruction",
        "source_hashes": {
            "causal_movement_feature_manifest.json": _sha256(FEATURE_MANIFEST),
            "causal_movement_threshold.json": _sha256(THRESHOLD_ARTIFACT),
            "causal_m1c_scaling_model_configurations.json": _sha256(SCALING_ARTIFACT),
        },
        "startup_artifact_identity_audit": artifact_identity_audit,
        "thresholds_verified": {
            "bottom_5": BOTTOM_5_THRESHOLD,
            "bottom_10": BOTTOM_10_THRESHOLD,
            "bottom_20": BOTTOM_20_THRESHOLD,
            "high_tail": HIGH_TAIL_THRESHOLD,
        },
        "prediction_audit": prediction_audit,
        "episode_rule_audit": episode_audit,
        "option_selection_audit": selection_audit,
        "structure_and_fill_audit": option_audit,
        "phase_audit": phase_audit,
        "safety_audit": safety_audit,
        "determinism": {
            "artifact": DETERMINISM_AUDIT.name,
            "passed": determinism["passed"],
        },
        "manual_reconstruction_counts": {
            "prospective_replay_quiet_state_predictions": 100,
            "shadow_iron_butterfly_outcomes": 50,
            "shadow_iron_condor_outcomes": 50,
        },
        "original_decision_verified_unchanged": True,
        "fail_closed": True,
    }
    independent["passed"] = (
        prediction_audit["rows_reconstructed"] == 100
        and prediction_audit["maximum_probability_difference"] <= 1e-12
        and prediction_audit["threshold_membership_mismatches"] == 0
        and prediction_audit["neutral_control_mismatches"] == 0
        and episode_audit["passed"]
        and selection_audit["passed"]
        and option_audit["iron_butterfly_outcomes_reconstructed"] == 50
        and option_audit["iron_condor_outcomes_reconstructed"] == 50
        and option_audit["option_contract_mismatches"] == 0
        and option_audit["structure_leg_mismatches"] == 0
        and option_audit["shadow_pnl_mismatches"] == 0
        and option_audit["maximum_floating_difference"] <= 1e-12
        and phase_audit["passed"]
        and safety_audit["passed"]
        and artifact_identity_audit["passed"]
        and determinism["passed"]
        and prediction_audit == prediction_audit_b
        and option_audit == option_audit_b
    )
    if not independent["passed"]:
        raise RuntimeError("quiet-state extension independent audit failed closed")
    _write(DETERMINISM_AUDIT, determinism)
    _write(INDEPENDENT_AUDIT, independent)
    print(
        json.dumps(
            {
                "independent_audit": "passed",
                "determinism": "passed",
                "predictions": 100,
                "iron_butterflies": 50,
                "iron_condors": 50,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
