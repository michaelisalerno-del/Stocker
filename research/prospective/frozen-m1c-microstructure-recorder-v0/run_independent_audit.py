"""Independent deterministic audit for the frozen prospective recorder V0."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "packages/stocker_prospective/src"
for package_name in (
    "stocker_prospective",
    "stocker_research",
    "stocker_data",
    "stocker_core",
):
    sys.path.insert(0, str(ROOT / "packages" / package_name / "src"))

from stocker_prospective.contract import claims_boundary  # noqa: E402
from stocker_prospective.events import (  # noqa: E402
    OptionQuoteEvent,
    UnderlyingLevel1QuoteEvent,
    UnderlyingTickTradeEvent,
)
from stocker_prospective.market_data import MarketDataType  # noqa: E402
from stocker_prospective.microstructure import (  # noqa: E402
    summarise_microstructure_window,
)
from stocker_prospective.option_ledger import (  # noqa: E402
    OptionContract,
    build_contract_plan,
    build_shadow_outcomes,
)
from stocker_prospective.options import DteBucket  # noqa: E402
from stocker_prospective.replay_v0 import (  # noqa: E402
    ReplayMode,
    deterministic_replay,
)

PRIMARY = (
    ROOT
    / "research/directional-readiness"
    / "20260726-stock-local-directional-archetypes-v0"
    / "artifacts/primary"
)
OUTPUT = Path(__file__).with_name("independent_audit.json")
START = datetime(2026, 7, 24, 14, 0, tzinfo=UTC)
M1C_THRESHOLD = 0.488333710794033
ARCHETYPE_IDS = ("A1", "C1", "R1")


def _finite(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _logistic(linear: float) -> float:
    if linear >= 0.0:
        return 1.0 / (1.0 + math.exp(-linear))
    exponential = math.exp(linear)
    return exponential / (1.0 + exponential)


def level1(
    sequence: int,
    observed: datetime,
    *,
    bid: float,
    ask: float,
    bid_size: float,
    ask_size: float,
) -> UnderlyingLevel1QuoteEvent:
    return UnderlyingLevel1QuoteEvent(
        event_id=f"quote-{sequence}",
        received_timestamp_utc=observed,
        received_monotonic_ns=10_000 + sequence,
        provider_timestamp_utc=observed,
        source_sequence=10_000 + sequence,
        session=observed.date(),
        symbol="AAL",
        con_id=1,
        request_id=10,
        bid=bid,
        bid_size=bid_size,
        ask=ask,
        ask_size=ask_size,
        last=(bid + ask) / 2.0,
        last_size=1.0,
        market_data_type=MarketDataType.LIVE,
        source="independent_audit",
        quote_valid=True,
        staleness_ms=0.0,
        tick_type="state_change",
        exchange="SMART",
    )


def option_quote(
    sequence: int,
    contract: OptionContract,
    observed: datetime,
    *,
    bid: float,
    ask: float,
) -> OptionQuoteEvent:
    assert contract.con_id is not None
    return OptionQuoteEvent(
        event_id=f"option-{sequence}",
        received_timestamp_utc=observed,
        received_monotonic_ns=20_000 + sequence,
        provider_timestamp_utc=observed,
        source_sequence=20_000 + sequence,
        session=observed.date(),
        symbol="AAL",
        con_id=contract.con_id,
        request_id=20,
        episode_id=f"episode-{contract.con_id}",
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
        ask_size=10.0,
        last=(bid + ask) / 2.0,
        last_size=1.0,
        market_data_type=MarketDataType.LIVE,
        option_model_price=(bid + ask) / 2.0,
        implied_volatility=0.5,
        delta=0.5,
        gamma=0.05,
        theta=-0.02,
        vega=0.04,
        underlying_reference_price=100.0,
    )


def _research_runner() -> ModuleType:
    path = (
        ROOT
        / "research/directional-readiness"
        / "20260726-stock-local-directional-archetypes-v0"
        / "run_screen_v0.py"
    )
    specification = importlib.util.spec_from_file_location(
        "independent_frozen_archetype_runner",
        path,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("could not load frozen research runner")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def audit_m1c() -> dict[str, Any]:
    runner = _research_runner()
    historical, stress, _states, _manifest = runner.load_inputs()
    _m0, _m1c, _threshold, scored, _metrics, _audit = runner.phase_zero(
        historical,
        stress,
    )
    frame = scored.loc[scored["period"].astype(str).eq("assessment")].copy()
    manifest = json.loads(
        (PRIMARY / "causal_movement_feature_manifest.json").read_text(encoding="utf-8")
    )
    threshold_artifact = json.loads(
        (PRIMARY / "causal_movement_threshold.json").read_text(encoding="utf-8")
    )
    specification = manifest["model_specification"]
    numeric_features = tuple(specification["numeric_features"])
    medians = tuple(float(value) for value in specification["numeric_medians"])
    means = tuple(float(value) for value in specification["numeric_means"])
    scales = tuple(float(value) for value in specification["numeric_scales"])
    stock_levels = tuple(specification["category_levels"]["stock"])
    coefficients = tuple(float(value) for value in specification["coefficients"])
    intercept = float(specification["intercept"])
    selected = frame.sort_values(
        ["session", "stock", "checkpoint"],
        kind="mergesort",
    ).head(100)
    maximum = 0.0
    threshold_mismatches = 0
    records = cast(list[dict[str, Any]], selected.to_dict(orient="records"))
    for values in records:
        checkpoint_name = f"checkpoint_{int(values['checkpoint'])}"
        raw = [
            1.0
            if name == checkpoint_name
            else 0.0
            if name.startswith("checkpoint_")
            else _finite(values.get(name))
            for name in numeric_features
        ]
        transformed = [
            ((medians[index] if value is None else value) - means[index]) / scales[index]
            for index, value in enumerate(raw)
        ]
        design = [
            *transformed,
            *(float(str(values["stock"]) == stock) for stock in stock_levels[1:]),
        ]
        probability = _logistic(
            math.fsum(
                value * coefficient for value, coefficient in zip(design, coefficients, strict=True)
            )
            + intercept
        )
        reference = float(values["M1C_probability"])
        maximum = max(maximum, abs(probability - reference))
        threshold_mismatches += int((probability >= M1C_THRESHOLD) != (reference >= M1C_THRESHOLD))
    contaminated = {"signed_pressure", "tension"}.intersection(numeric_features)
    return {
        "rows_reconstructed": len(selected),
        "maximum_probability_difference": maximum,
        "threshold_membership_mismatches": threshold_mismatches,
        "contaminated_features_present": sorted(contaminated),
        "manual_formula": "artifact vector -> median -> scale -> stock one-hot -> logistic",
        "artifact_threshold": float(threshold_artifact["threshold"]),
        "passed": (
            len(selected) >= 100
            and maximum <= 1e-12
            and threshold_mismatches == 0
            and not contaminated
            and abs(float(threshold_artifact["threshold"]) - M1C_THRESHOLD) <= 1e-15
        ),
    }


def audit_direction(frame: pd.DataFrame) -> dict[str, Any]:
    configurations = json.loads((PRIMARY / "model_configurations.json").read_text(encoding="utf-8"))
    normalisation = json.loads(
        (PRIMARY / "stock_local_normalisation_parameters.json").read_text(encoding="utf-8")
    )
    thresholds = json.loads(
        (PRIMARY / "frozen_archetype_thresholds.json").read_text(encoding="utf-8")
    )
    exact: dict[tuple[str, str, int], dict[str, Any]] = {}
    pooled: dict[str, dict[str, Any]] = {}
    for parameter in normalisation["parameters"]:
        if parameter["stock"] == "__POOLED__":
            pooled[str(parameter["feature"])] = parameter
        else:
            exact[
                (
                    str(parameter["feature"]),
                    str(parameter["stock"]),
                    int(parameter["checkpoint"]),
                )
            ] = parameter
    selected = frame.sort_values(
        ["session", "stock", "checkpoint"],
        kind="mergesort",
    ).head(100)
    maximum = {model_id: 0.0 for model_id in ARCHETYPE_IDS}
    mismatches = {model_id: 0 for model_id in ARCHETYPE_IDS}
    records = cast(list[dict[str, Any]], selected.to_dict(orient="records"))
    for values in records:
        for model_id in ARCHETYPE_IDS:
            model = configurations["full_models"][model_id]
            numeric = tuple(model["numeric_features"])
            normalised: list[float] = []
            for name in numeric:
                parameter = exact.get(
                    (name, str(values["stock"]), int(values["checkpoint"])),
                    pooled[name],
                )
                raw = _finite(values.get(f"raw__{name}"))
                present = float(parameter["missing_value"]) if raw is None else raw
                clipped = min(
                    max(present, float(parameter["clip_lower"])),
                    float(parameter["clip_upper"]),
                )
                normalised.append((clipped - float(parameter["median"])) / float(parameter["iqr"]))
            design: list[float] = []
            for index, name in enumerate(numeric):
                value = normalised[index]
                missing = not math.isfinite(value)
                imputed = float(model["medians"][name]) if missing else value
                standardized = (imputed - float(model["robust_centers"][name])) / float(
                    model["robust_scales"][name]
                )
                design.extend((standardized, float(missing)))
            categories = {
                "stock": str(values["stock"]),
                "checkpoint_category": str(values["checkpoint_category"]),
                "day_of_week": str(values["day_of_week"]),
            }
            for category in model["categorical_features"]:
                levels = tuple(model["categorical_levels"][category])
                observed = categories[category]
                if observed not in levels:
                    observed = "__UNKNOWN__"
                design.extend(float(observed == level) for level in levels)
            probability = _logistic(
                math.fsum(
                    value * float(coefficient)
                    for value, coefficient in zip(
                        design,
                        model["coefficients"],
                        strict=True,
                    )
                )
                + float(model["intercept"])
            )
            boundary = float(thresholds[model_id]["boundary"])
            action = (
                "CALL"
                if probability >= 0.5 + boundary
                else "PUT"
                if probability <= 0.5 - boundary
                else "ABSTAIN"
            )
            maximum[model_id] = max(
                maximum[model_id],
                abs(probability - float(values[f"{model_id}_probability"])),
            )
            mismatches[model_id] += int(action != str(values[f"{model_id}_action"]))
    return {
        "rows_reconstructed_per_archetype": len(selected),
        "total_probabilities_reconstructed": len(selected) * len(ARCHETYPE_IDS),
        "maximum_probability_difference": maximum,
        "action_mismatches": mismatches,
        "manual_formula": (
            "stock/checkpoint transform -> frozen robust standardisation -> "
            "categorical one-hot -> logistic -> symmetric OOF boundary"
        ),
        "passed": (
            len(selected) >= 100
            and max(maximum.values()) <= 1e-12
            and sum(mismatches.values()) == 0
        ),
    }


def audit_microstructure() -> tuple[dict[str, Any], tuple[UnderlyingLevel1QuoteEvent, ...]]:
    maximum = 0.0
    all_quotes: list[UnderlyingLevel1QuoteEvent] = []
    for index in range(100):
        start = START + timedelta(minutes=index)
        quote_start = level1(
            index * 2,
            start,
            bid=100.0,
            ask=100.02,
            bid_size=100.0,
            ask_size=100.0,
        )
        quote_end = level1(
            index * 2 + 1,
            start + timedelta(seconds=5),
            bid=100.01,
            ask=100.03,
            bid_size=120.0,
            ask_size=80.0,
        )
        trade = UnderlyingTickTradeEvent(
            event_id=f"trade-{index}",
            received_timestamp_utc=start + timedelta(seconds=2),
            received_monotonic_ns=30_000 + index,
            provider_timestamp_utc=start + timedelta(seconds=2),
            source_sequence=30_000 + index,
            session=start.date(),
            symbol="AAL",
            con_id=1,
            request_id=11,
            price=100.02,
            size=10.0,
            exchange="NYSE",
            conditions=(),
            market_data_type=MarketDataType.LIVE,
        )
        summary = summarise_microstructure_window(
            symbol="AAL",
            window_start=start,
            window_end=start + timedelta(seconds=5),
            quotes=(quote_start, quote_end),
            trades=(trade,),
            maximum_quote_age=timedelta(seconds=3),
            minimum_classification_valid_fraction=0.5,
        )
        manually_reconstructed_midpoint_change = (quote_end.bid + quote_end.ask) / 2.0 - (
            quote_start.bid + quote_start.ask
        ) / 2.0
        start_midpoint = (quote_start.bid + quote_start.ask) / 2.0
        end_midpoint = (quote_end.bid + quote_end.ask) / 2.0
        end_size = quote_end.bid_size + quote_end.ask_size
        end_microprice = (
            quote_end.ask * quote_end.bid_size + quote_end.bid * quote_end.ask_size
        ) / end_size
        manual_values = {
            "midpoint_change": manually_reconstructed_midpoint_change,
            "best_bid_change": quote_end.bid - quote_start.bid,
            "best_ask_change": quote_end.ask - quote_start.ask,
            "probable_buy_volume": 10.0,
            "probable_sell_volume": 0.0,
            "trade_imbalance": 1.0,
            "classification_valid_fraction": 1.0,
            "buy_impact_bps": (manually_reconstructed_midpoint_change / start_midpoint * 10_000.0),
            "latest_microprice_edge": end_microprice - end_midpoint,
        }
        assert summary.quote_flow.midpoint_change is not None
        assert summary.buy_impact.price_impact_bps is not None
        actual_values = {
            "midpoint_change": summary.quote_flow.midpoint_change,
            "best_bid_change": summary.quote_flow.best_bid_change,
            "best_ask_change": summary.quote_flow.best_ask_change,
            "probable_buy_volume": summary.trade_flow.probable_buy_volume,
            "probable_sell_volume": summary.trade_flow.probable_sell_volume,
            "trade_imbalance": summary.trade_flow.trade_imbalance,
            "classification_valid_fraction": summary.trade_flow.classification_valid_fraction,
            "buy_impact_bps": summary.buy_impact.price_impact_bps,
            "latest_microprice_edge": summary.scores["MC"].components["microprice_edge"]
            * ((quote_end.ask - quote_end.bid) / 2.0),
        }
        maximum = max(
            maximum,
            *(
                abs(float(actual_values[name]) - expected)
                for name, expected in manual_values.items()
            ),
        )
        all_quotes.extend((quote_start, quote_end))
    return (
        {
            "windows_reconstructed": 100,
            "independently_reconstructed_fields_per_window": 9,
            "maximum_floating_difference": maximum,
            "passed": maximum <= 1e-12,
        },
        tuple(all_quotes),
    )


def audit_options() -> dict[str, Any]:
    maximum = 0.0
    for index in range(50):
        entry = START + timedelta(minutes=index)
        contract = OptionContract(
            underlying_con_id=1,
            con_id=1000 + index,
            expiry=date(2026, 7, 24),
            dte=0,
            dte_bucket=DteBucket.ZERO_DTE,
            strike=100.0,
            right="C" if index % 2 == 0 else "P",
            multiplier=100,
            exchange="SMART",
            trading_class="AAL",
        )
        entry_ask = 1.0 + index / 100.0
        exit_bid = entry_ask * (1.0 + (index % 5 - 2) / 100.0)
        quotes = (
            option_quote(
                index * 3,
                contract,
                entry,
                bid=entry_ask - 0.05,
                ask=entry_ask,
            ),
            option_quote(
                index * 3 + 1,
                contract,
                entry + timedelta(minutes=5) - timedelta(seconds=1),
                bid=exit_bid,
                ask=exit_bid + 0.05,
            ),
            option_quote(
                index * 3 + 2,
                contract,
                entry + timedelta(minutes=5) + timedelta(seconds=1),
                bid=exit_bid + 0.01,
                ask=exit_bid + 0.06,
            ),
        )
        outcome = build_shadow_outcomes(
            episode_id=f"episode-{contract.con_id}",
            symbol="AAL",
            entry_timestamp=entry,
            contracts=(contract,),
            quotes=quotes,
            horizons=(timedelta(minutes=5),),
            maximum_quote_age=timedelta(seconds=5),
        )[0]
        expected = exit_bid / entry_ask - 1.0
        assert outcome.ask_to_bid_return is not None
        expected_sensitivity = (exit_bid + 0.01) / entry_ask - 1.0
        expected_dollar = (exit_bid - entry_ask) * contract.multiplier
        assert outcome.first_after_horizon_sensitivity_return is not None
        assert outcome.dollar_pnl_per_contract is not None
        maximum = max(
            maximum,
            abs(outcome.entry_ask - entry_ask),  # type: ignore[operator]
            abs(outcome.exit_bid - exit_bid),  # type: ignore[operator]
            abs(outcome.ask_to_bid_return - expected),
            abs(outcome.first_after_horizon_sensitivity_return - expected_sensitivity),
            abs(outcome.dollar_pnl_per_contract - expected_dollar),
        )
    return {
        "outcomes_reconstructed": 50,
        "entry_rule": "first valid ask at or after prospective entry",
        "exit_rule": "last valid bid at or before frozen horizon",
        "independently_reconstructed_fields_per_outcome": 5,
        "maximum_floating_difference": maximum,
        "passed": maximum <= 1e-12,
    }


def audit_episode_timing() -> dict[str, Any]:
    from stocker_prospective.frozen_m1c import FreshEpisodeTracker

    tracker = FreshEpisodeTracker(threshold=M1C_THRESHOLD)
    probabilities = (0.60, 0.61, 0.20, 0.62, 0.20, 0.63)
    offsets = (0, 10, 20, 25, 30, 35)
    actual: list[int] = []
    for index, (probability, offset) in enumerate(
        zip(probabilities, offsets, strict=True),
        start=1,
    ):
        decision = tracker.evaluate(
            symbol="AAL",
            session=START.date(),
            checkpoint=index * 2 + 4,
            trigger_bar_end=START + timedelta(minutes=offset),
            probability=probability,
        )
        if decision.fresh_episode:
            actual.append(offset)
    direction_source = (PACKAGE / "stocker_prospective/direction_features.py").read_text(
        encoding="utf-8"
    )
    marker_excludes_trigger = all(
        fragment in direction_source
        for fragment in (
            "marker_ordinal = checkpoint - 2",
            "trigger_ordinal = checkpoint - 1",
            "prefix = completed_bars[: marker_ordinal + 1]",
            "trigger_bar_excluded=True",
        )
    )
    return {
        "manual_expected_episode_offsets_minutes": [0, 35],
        "runtime_episode_offsets_minutes": actual,
        "minimum_spacing_minutes": 30,
        "direction_features_end": "T-1",
        "trigger_bar_excluded": marker_excludes_trigger,
        "passed": actual == [0, 35] and marker_excludes_trigger,
    }


def audit_subscription_and_raw_contract(
    replay_events: tuple[UnderlyingLevel1QuoteEvent, ...],
) -> dict[str, Any]:
    from stocker_prospective.subscriptions import (
        SubscriptionBudgetManager,
        SubscriptionKind,
        SubscriptionPriority,
    )

    manager = SubscriptionBudgetManager(
        limits={
            SubscriptionKind.LEVEL1: 2,
            SubscriptionKind.TICK_BY_TICK: 1,
            SubscriptionKind.DEPTH: 0,
            SubscriptionKind.OPTION: 1,
            SubscriptionKind.BAR: 0,
            SubscriptionKind.MARKET_PROXY: 0,
        },
        request_rate_limit=100,
    )
    protected = manager.allocate(
        key="underlying:AAL:l1",
        kind=SubscriptionKind.LEVEL1,
        symbol="AAL",
        con_id=1,
        request_id=1,
        priority=SubscriptionPriority.UNIVERSE_LEVEL1,
        protected=True,
        now_monotonic=1.0,
    )
    armed = manager.allocate(
        key="underlying:AAL:tbt",
        kind=SubscriptionKind.TICK_BY_TICK,
        symbol="AAL",
        con_id=1,
        request_id=2,
        priority=SubscriptionPriority.ARMED_CANDIDATE,
        now_monotonic=2.0,
    )
    active = manager.allocate(
        key="underlying:AAOI:tbt",
        kind=SubscriptionKind.TICK_BY_TICK,
        symbol="AAOI",
        con_id=2,
        request_id=3,
        priority=SubscriptionPriority.ACTIVE_EPISODE,
        owner_episode="episode-audit",
        now_monotonic=3.0,
    )
    unique_raw_ids = len({event.event_id for event in replay_events})
    raw_payload_digest = hashlib.sha256(
        json.dumps(
            [event.model_dump(mode="json") for event in replay_events],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        "protected_level1_accepted": protected.accepted,
        "armed_candidate_accepted": armed.accepted,
        "active_episode_accepted": active.accepted,
        "active_episode_evicted": active.evicted_key,
        "protected_level1_retained": manager.get("underlying:AAL:l1") is not None,
        "raw_events_preserved": len(replay_events),
        "unique_raw_event_ids": unique_raw_ids,
        "raw_payload_digest": raw_payload_digest,
        "passed": (
            protected.accepted
            and armed.accepted
            and active.accepted
            and active.evicted_key == "underlying:AAL:tbt"
            and manager.get("underlying:AAL:l1") is not None
            and unique_raw_ids == len(replay_events)
        ),
    }


def audit_option_selection() -> dict[str, Any]:
    expiries = {
        DteBucket.ZERO_DTE: date(2026, 7, 24),
        DteBucket.ONE_DTE: date(2026, 7, 25),
        DteBucket.THREE_TO_FIVE_DTE: date(2026, 7, 28),
    }
    plan = build_contract_plan(
        underlying_con_id=1,
        session_date=date(2026, 7, 24),
        underlying_reference=102.0,
        expiries=expiries,
        strikes_by_expiry_right={
            (expiry, right): (95.0, 100.0, 105.0)
            for expiry in expiries.values()
            for right in ("C", "P")
        },
        strike_steps=1,
        maximum_contracts=18,
        exchange="SMART",
        trading_class="AAL",
    )
    expected_per_bucket = [
        (100.0, "C"),
        (100.0, "P"),
        (105.0, "C"),
        (105.0, "P"),
        (95.0, "C"),
        (95.0, "P"),
    ]
    actual_per_bucket = {
        bucket.value: [
            (contract.strike, contract.right)
            for contract in plan.contracts
            if contract.dte_bucket is bucket
        ]
        for bucket in DteBucket
    }
    return {
        "expected_per_bucket": expected_per_bucket,
        "actual_per_bucket": actual_per_bucket,
        "contract_count": len(plan.contracts),
        "deduplicated_contract_identities": len(
            {contract.con_id_key for contract in plan.contracts}
        ),
        "passed": (
            len(plan.contracts) == 18
            and all(actual_per_bucket[bucket.value] == expected_per_bucket for bucket in DteBucket)
            and len({contract.con_id_key for contract in plan.contracts}) == 18
        ),
    }


def audit_phase_and_web_boundary() -> dict[str, Any]:
    phase_contract = json.loads(
        Path(__file__).with_name("prospective_phase_contract.json").read_text(encoding="utf-8")
    )
    counts = [int(item["count"]) for item in phase_contract["phases"]]
    web_source = (PACKAGE / "stocker_prospective/web.py").read_text(encoding="utf-8")
    static_source = (PACKAGE / "stocker_prospective/web_static/index.html").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(web_source)
    routes = {
        argument.value
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for decorator in node.decorator_list
        if isinstance(decorator, ast.Call)
        for argument in decorator.args[:1]
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
    }
    forbidden_fragments = ("/orders", "/positions", "/accounts", "/trade", "/buy", "/sell")
    forbidden_routes = sorted(
        route
        for route in routes
        if any(fragment in route.lower() for fragment in forbidden_fragments)
    )
    banner_present = "RECORD ONLY — ORDER ROUTING DISABLED" in static_source
    return {
        "phase_counts": counts,
        "confirmation_target_dependent_selection_opened": phase_contract["phases"][2][
            "target_dependent_selection_opened"
        ],
        "forbidden_web_routes": forbidden_routes,
        "record_only_banner_present": banner_present,
        "passed": (
            counts == [30, 100, 100]
            and phase_contract["immutable_after_activation"] is True
            and phase_contract["phases"][2]["target_dependent_selection_opened"] is False
            and not forbidden_routes
            and banner_present
        ),
    }


def audit_artifact_and_safety_identity() -> dict[str, Any]:
    source_manifest = json.loads(
        Path(__file__).with_name("source_manifest.json").read_text(encoding="utf-8")
    )
    mismatches: list[str] = []
    for name, identity in source_manifest["frozen_sources"].items():
        path = ROOT / str(identity["path"])
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if actual != identity["sha256"]:
            mismatches.append(name)
    capability_source = (PACKAGE / "stocker_prospective/capability.py").read_text(encoding="utf-8")
    safety_source = (PACKAGE / "stocker_prospective/safety.py").read_text(encoding="utf-8")
    market_type_fail_closed = all(
        fragment in capability_source + safety_source
        for fragment in (
            "MarketDataType.LIVE",
            "market_data_not_live",
            "scientific_recording_valid",
        )
    )
    return {
        "artifact_hash_mismatches": mismatches,
        "market_data_type_validation_present": market_type_fail_closed,
        "passed": not mismatches and market_type_fail_closed,
    }


def audit_order_surface() -> dict[str, Any]:
    forbidden = {
        "placeOrder",
        "cancelOrder",
        "reqOpenOrders",
        "reqPositions",
        "place_order",
        "cancel_order",
    }
    found: list[str] = []
    source_root = PACKAGE / "stocker_prospective"
    for path in sorted(source_root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in forbidden:
                found.append(f"{path.name}:{node.name}")
    return {
        "forbidden_callable_definitions": found,
        "passed": not found,
    }


def main() -> None:
    frame = pd.read_parquet(PRIMARY / "assessment_predictions.parquet")
    m1c = audit_m1c()
    direction = audit_direction(frame)
    microstructure, replay_events = audit_microstructure()
    options = audit_options()
    episode_timing = audit_episode_timing()
    subscription_and_raw = audit_subscription_and_raw_contract(replay_events)
    option_selection = audit_option_selection()
    phase_and_web = audit_phase_and_web_boundary()
    artifact_and_safety = audit_artifact_and_safety_identity()
    order_surface = audit_order_surface()
    first = deterministic_replay(
        replay_events,
        mode=ReplayMode.ACCELERATED,
    )
    second = deterministic_replay(
        replay_events,
        mode=ReplayMode.ACCELERATED,
    )
    replay = {
        "events_replayed": first.event_count,
        "first_digest": first.digest,
        "second_digest": second.digest,
        "event_order_mismatches": int(first.event_ids != second.event_ids),
        "ibkr_connections_attempted": (
            first.ibkr_connections_attempted + second.ibkr_connections_attempted
        ),
        "maximum_floating_difference": max(
            first.maximum_floating_difference,
            second.maximum_floating_difference,
        ),
        "passed": (
            first.digest == second.digest
            and first.event_ids == second.event_ids
            and first.ibkr_connections_attempted == 0
            and second.ibkr_connections_attempted == 0
        ),
    }
    sections = {
        "m1c": m1c,
        "direction": direction,
        "microstructure": microstructure,
        "options": options,
        "episode_timing": episode_timing,
        "subscription_budget_and_raw_preservation": subscription_and_raw,
        "option_selection": option_selection,
        "prospective_phases_and_web": phase_and_web,
        "artifact_and_safety_identity": artifact_and_safety,
        "replay": replay,
        "order_surface": order_surface,
    }
    report = {
        "contract_version": "frozen-m1c-microstructure-recorder-v0",
        "claims_boundary": claims_boundary(),
        "audit_kind": "independent_manual_reconstruction",
        "tolerance": 1e-12,
        "sections": sections,
        "passed": all(bool(section["passed"]) for section in sections.values()),
    }
    canonical = json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    report["report_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
    OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not report["passed"]:
        raise SystemExit("independent audit failed")


if __name__ == "__main__":
    main()
