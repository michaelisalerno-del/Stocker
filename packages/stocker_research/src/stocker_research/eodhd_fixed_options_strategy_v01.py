"""Exact-date repair mechanics for the fixed EODHD options strategy screen V0.1."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import pandas as pd

PROTECTED_LAST_OBSERVATION_DATE = date(2025, 8, 22)

SAFETY_FLAGS_V01: dict[str, object] = {
    "research_only": True,
    "retrospective_options_strategy_screen": True,
    "repair_version": "v0.1",
    "exact_observation_date_filtering": True,
    "extra_provider_dates_discarded_before_materialisation": True,
    "protected_boundary_applied_to_observation_date": True,
    "contract_expiration_may_cross_protected_boundary": True,
    "strategies_evaluated_independently": True,
    "directional_strategy_requires_audited_mapping": True,
    "options_data_granularity": "end_of_day",
    "stock_signal_time": "15:30 America/New_York",
    "option_entry_time_proxy": "same_session_end_of_day_quote",
    "option_exit_time_proxy": "future_end_of_day_quote_or_expiry_intrinsic",
    "intraday_option_fill_simulated": False,
    "daily_option_high_low_used": False,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_integration_required": False,
    "strategy_promotion": False,
    "prospective_validation": False,
    "achieved_pnl_claimed": False,
}

StrategyStatus = Literal[
    "supported",
    "descriptive_only",
    "not_supported",
    "insufficient_support",
    "blocked_direction_mapping_unavailable",
    "blocked_contract_coverage",
    "blocked_quote_integrity",
    "blocked_chronology",
    "blocked_resource_limit",
]
STRATEGY_STATUSES_V01: tuple[StrategyStatus, ...] = (
    "supported",
    "descriptive_only",
    "not_supported",
    "insufficient_support",
    "blocked_direction_mapping_unavailable",
    "blocked_contract_coverage",
    "blocked_quote_integrity",
    "blocked_chronology",
    "blocked_resource_limit",
)

OVERALL_DECISIONS_V01 = (
    "multiple_eodhd_options_strategies_show_feasibility",
    "overnight_straddle_feasible_only",
    "directional_debit_spread_feasible_only",
    "dte1_straddle_feasible_only",
    "hidden_diversion_veto_improves_directional_spreads",
    "descriptive_options_strategy_results_only",
    "no_eodhd_options_strategy_feasibility",
    "all_supported_strategies_insufficient_support",
    "blocked_exact_date_filter_failure",
    "blocked_options_contract_reconstruction_failure",
    "blocked_options_quote_integrity_failure",
    "blocked_protected_boundary_failure",
    "blocked_quick_options_strategy_resource_limit",
    "blocked_reproducibility_or_audit_failure",
)

ExactDateStatus = Literal[
    "exact_date_complete",
    "exact_date_absent",
    "extra_dates_discarded",
    "ambiguous_date_mapping",
    "incomplete_pagination",
    "schema_failure",
]


@dataclass(frozen=True)
class ExactDateFilterResult:
    """Records and audit facts emitted before option canonicalisation."""

    status: ExactDateStatus
    retained_records: tuple[Mapping[str, Any], ...]
    returned_observation_dates: tuple[date, ...]
    requested_date_present: bool
    exact_date_record_count: int
    discarded_other_date_record_count: int
    discarded_post_boundary_record_count: int
    pagination_complete: bool
    response_hash: str
    exact_date_hash: str
    discarded_fragment_hash: str


class _ObservationSchemaFailure(ValueError):
    """Provider observation-date metadata cannot be parsed."""


class _ObservationMappingAmbiguity(ValueError):
    """Provider observation-date metadata disagree."""


def _timestamp_date(value: object) -> date:
    if not isinstance(value, str):
        raise _ObservationSchemaFailure("missing provider quote observation timestamp")
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise _ObservationSchemaFailure("invalid provider quote observation timestamp") from error
    if pd.isna(timestamp):
        raise _ObservationSchemaFailure("missing provider quote observation timestamp")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("America/New_York")
    else:
        timestamp = timestamp.tz_convert("America/New_York")
    return timestamp.date()


def _provider_identity_date(record: Mapping[str, Any]) -> date:
    resource_id = record.get("id")
    if not isinstance(resource_id, str):
        raise _ObservationSchemaFailure("provider observation-date metadata are missing")
    try:
        return date.fromisoformat(resource_id[-10:])
    except ValueError as error:
        raise _ObservationSchemaFailure(
            "provider resource identity lacks an observation date"
        ) from error


def _provider_observation_date(record: Mapping[str, Any]) -> date:
    observation_date = _provider_identity_date(record)
    attributes = record.get("attributes")
    if not isinstance(attributes, Mapping):
        raise _ObservationSchemaFailure("provider observation-date metadata are missing")
    if any(attributes.get(field) is None for field in ("bid", "ask", "bid_date", "ask_date")):
        return observation_date
    for side in ("bid", "ask"):
        quote_value = attributes.get(side)
        timestamp_value = attributes.get(f"{side}_date")
        if quote_value is None and timestamp_value is None:
            continue
        quote_date = _timestamp_date(timestamp_value)
        if observation_date != quote_date:
            raise _ObservationMappingAmbiguity("provider observation-date metadata disagree")
    return observation_date


def _records_hash(records: Sequence[Mapping[str, Any]]) -> str:
    serialised = sorted(
        json.dumps(
            record,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            default=str,
            separators=(",", ":"),
        )
        for record in records
    )
    return hashlib.sha256(json.dumps(serialised, separators=(",", ":")).encode("utf-8")).hexdigest()


def filter_exact_observation_date(
    records: Sequence[Mapping[str, Any]],
    *,
    requested_observation_date: date,
    response_hash: str,
    pagination_complete: bool,
) -> ExactDateFilterResult:
    """Retain only the requested provider observation date."""

    try:
        dated = [(_provider_identity_date(record), record) for record in records]
    except _ObservationSchemaFailure:
        return ExactDateFilterResult(
            status="schema_failure",
            retained_records=(),
            returned_observation_dates=(),
            requested_date_present=False,
            exact_date_record_count=0,
            discarded_other_date_record_count=0,
            discarded_post_boundary_record_count=0,
            pagination_complete=pagination_complete,
            response_hash=response_hash,
            exact_date_hash=_records_hash(()),
            discarded_fragment_hash=_records_hash(records),
        )
    returned_dates = tuple(sorted({observation_date for observation_date, _record in dated}))
    exact_records = tuple(
        record
        for observation_date, record in dated
        if observation_date == requested_observation_date
    )
    discarded = tuple(
        record
        for observation_date, record in dated
        if observation_date != requested_observation_date
    )
    discarded_post_boundary = sum(
        observation_date > PROTECTED_LAST_OBSERVATION_DATE
        for observation_date, _record in dated
        if observation_date != requested_observation_date
    )
    requested_date_present = bool(exact_records)
    try:
        for record in exact_records:
            _provider_observation_date(record)
    except _ObservationMappingAmbiguity:
        return ExactDateFilterResult(
            status="ambiguous_date_mapping",
            retained_records=(),
            returned_observation_dates=returned_dates,
            requested_date_present=requested_date_present,
            exact_date_record_count=len(exact_records),
            discarded_other_date_record_count=len(discarded),
            discarded_post_boundary_record_count=discarded_post_boundary,
            pagination_complete=pagination_complete,
            response_hash=response_hash,
            exact_date_hash=_records_hash(exact_records),
            discarded_fragment_hash=_records_hash(discarded),
        )
    except _ObservationSchemaFailure:
        return ExactDateFilterResult(
            status="schema_failure",
            retained_records=(),
            returned_observation_dates=returned_dates,
            requested_date_present=requested_date_present,
            exact_date_record_count=len(exact_records),
            discarded_other_date_record_count=len(discarded),
            discarded_post_boundary_record_count=discarded_post_boundary,
            pagination_complete=pagination_complete,
            response_hash=response_hash,
            exact_date_hash=_records_hash(exact_records),
            discarded_fragment_hash=_records_hash(discarded),
        )
    status: ExactDateStatus
    if not pagination_complete:
        status = "incomplete_pagination"
    elif not requested_date_present:
        status = "exact_date_absent"
    elif discarded:
        status = "extra_dates_discarded"
    else:
        status = "exact_date_complete"
    retained = exact_records if status in {"exact_date_complete", "extra_dates_discarded"} else ()
    return ExactDateFilterResult(
        status=status,
        retained_records=retained,
        returned_observation_dates=returned_dates,
        requested_date_present=requested_date_present,
        exact_date_record_count=len(exact_records),
        discarded_other_date_record_count=len(discarded),
        discarded_post_boundary_record_count=discarded_post_boundary,
        pagination_complete=pagination_complete,
        response_hash=response_hash,
        exact_date_hash=_records_hash(exact_records),
        discarded_fragment_hash=_records_hash(discarded),
    )


def assert_safety_flags_v01(values: Mapping[str, object]) -> None:
    """Require every frozen V0.1 research and execution-safety value."""

    for name, expected in SAFETY_FLAGS_V01.items():
        if (
            name not in values
            or values[name] != expected
            or type(values[name]) is not type(expected)
        ):
            raise ValueError(f"V0.1 safety flag differs: {name}")


def validate_observation_and_expiration_boundary(
    *, observation_date: date, expiration_date: date
) -> None:
    """Protect market observations while allowing causal expiration metadata."""

    if observation_date > PROTECTED_LAST_OBSERVATION_DATE:
        raise ValueError(
            f"quote crossed protected observation boundary: {observation_date.isoformat()}"
        )
    if expiration_date < observation_date:
        raise ValueError("contract expiration precedes its quote observation")


def validate_strategy_statuses(statuses: Mapping[str, str]) -> None:
    """Validate the four independently reported strategy status fields."""

    expected = {"S1", "S2_ALL", "S2_VETO", "S3"}
    if set(statuses) != expected:
        raise ValueError("strategy statuses must cover S1, S2_ALL, S2_VETO, and S3")
    invalid = {
        name: status for name, status in statuses.items() if status not in STRATEGY_STATUSES_V01
    }
    if invalid:
        raise ValueError(f"invalid strategy status values: {invalid}")


def choose_overall_decision_v01(
    *,
    statuses: Mapping[str, str],
    strategy_positive: Mapping[str, bool],
    hidden_veto_positive: bool,
    fatal_blocker: str | None = None,
) -> str:
    """Derive V0.1's result without propagating an isolated strategy blocker."""

    validate_strategy_statuses(statuses)
    if fatal_blocker is not None:
        if fatal_blocker not in OVERALL_DECISIONS_V01 or not fatal_blocker.startswith("blocked_"):
            raise ValueError(f"invalid global blocker: {fatal_blocker}")
        return fatal_blocker
    positive = {
        strategy for strategy in ("S1", "S2", "S3") if bool(strategy_positive.get(strategy, False))
    }
    if len(positive) >= 2:
        return "multiple_eodhd_options_strategies_show_feasibility"
    if positive == {"S1"}:
        return "overnight_straddle_feasible_only"
    if positive == {"S2"}:
        return "directional_debit_spread_feasible_only"
    if positive == {"S3"}:
        return "dte1_straddle_feasible_only"
    if hidden_veto_positive:
        return "hidden_diversion_veto_improves_directional_spreads"

    independently_evaluable = [
        statuses["S1"],
        statuses["S3"],
        *(
            [statuses["S2_ALL"]]
            if statuses["S2_ALL"] != "blocked_direction_mapping_unavailable"
            else []
        ),
    ]
    if independently_evaluable and all(
        status == "insufficient_support" for status in independently_evaluable
    ):
        return "all_supported_strategies_insufficient_support"
    if any(status in {"supported", "not_supported"} for status in independently_evaluable):
        return "no_eodhd_options_strategy_feasibility"
    return "descriptive_options_strategy_results_only"


__all__ = [
    "ExactDateFilterResult",
    "ExactDateStatus",
    "OVERALL_DECISIONS_V01",
    "PROTECTED_LAST_OBSERVATION_DATE",
    "SAFETY_FLAGS_V01",
    "STRATEGY_STATUSES_V01",
    "StrategyStatus",
    "assert_safety_flags_v01",
    "choose_overall_decision_v01",
    "filter_exact_observation_date",
    "validate_observation_and_expiration_boundary",
    "validate_strategy_statuses",
]
