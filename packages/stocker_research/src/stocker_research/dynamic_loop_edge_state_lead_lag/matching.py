"""Fail-closed reconstruction and matching for the V2 delay attribution.

The frozen V2 delay stress shifted a cell-level policy to later opportunities.
It did not delay an identified trade.  This module keeps that reconstruction
separate from the stricter same-setup counterfactual requested by the follow-up
contract.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

POLICY_CELL: Final[tuple[str, ...]] = ("period", "loop_id", "orientation", "horizon")
POLICY_KEY: Final[tuple[str, ...]] = (*POLICY_CELL, "score_session")
STRUCTURAL_LINEAGE_FIELDS: Final[tuple[str, ...]] = (
    "symbol_norm",
    "loop_id",
    "orientation",
    "history_token",
    "top_loop_cycle",
    "strategy",
    "family",
    "representation",
    "direction",
)
PERSISTENT_ID_FIELDS: Final[tuple[str, ...]] = (
    "event_lineage_id",
    "setup_id",
    "persistent_anchor_id",
    "parent_setup_id",
    "anchor_id",
)


def _require(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"missing {label} columns: {missing}")


def reconstruct_v2_shifted_policy(decisions: pd.DataFrame) -> pd.DataFrame:
    """Independently reproduce the frozen V2 one-opportunity-session shift.

    The returned payoff and timing columns remain those of each current row.
    ``policy_source_session`` makes explicit that the source can be more than
    one calendar step earlier when a cell has no intervening opportunity.
    """

    _require(
        decisions,
        {*POLICY_KEY, "accepted", "status", "opportunity_id"},
        "V2 decision",
    )
    frame = decisions.copy()
    policy_values = frame.groupby(list(POLICY_KEY), observed=True)["accepted"].nunique(dropna=False)
    if policy_values.gt(1).any():
        raise ValueError("V2 accepted policy is not unique within a cell/session")

    policy_columns = [*POLICY_KEY, "accepted"]
    if "session_index" in frame:
        policy_columns.append("session_index")
    policy = (
        frame.loc[:, policy_columns]
        .drop_duplicates(list(POLICY_KEY))
        .sort_values(list(POLICY_KEY), kind="stable")
    )
    if "session_index" not in policy:
        sessions = (
            frame.loc[:, ["period", "score_session"]]
            .drop_duplicates()
            .sort_values(["period", "score_session"], kind="stable")
        )
        sessions["session_index"] = sessions.groupby("period", observed=True).cumcount()
        policy = policy.merge(
            sessions,
            on=["period", "score_session"],
            how="left",
            validate="many_to_one",
        )

    grouped = policy.groupby(list(POLICY_CELL), observed=True, sort=False)
    policy["delayed_accepted"] = grouped["accepted"].shift(1, fill_value=False).astype(bool)
    policy["policy_source_session"] = grouped["score_session"].shift(1)
    policy["policy_source_session_index"] = grouped["session_index"].shift(1)
    policy["policy_gap_sessions"] = (
        pd.to_numeric(policy["session_index"], errors="coerce")
        - pd.to_numeric(policy["policy_source_session_index"], errors="coerce")
    ).astype("Int64")

    result = frame.merge(
        policy.loc[
            :,
            [
                *POLICY_KEY,
                "delayed_accepted",
                "policy_source_session",
                "policy_source_session_index",
                "policy_gap_sessions",
            ],
        ],
        on=list(POLICY_KEY),
        how="left",
        validate="many_to_one",
    )
    result["immediate_accepted"] = result["accepted"].eq(True)  # noqa: E712
    result["delayed_accepted"] = result["delayed_accepted"].eq(True)  # noqa: E712
    immediate = result["immediate_accepted"]
    delayed = result["delayed_accepted"]
    result["population_category"] = np.select(
        [immediate & delayed, immediate & ~delayed, ~immediate & delayed],
        ["retained", "dropped", "introduced"],
        default="rejected_both",
    )
    result["immediate_accepted_filled"] = immediate & result["status"].eq("filled")
    result["delayed_accepted_filled"] = delayed & result["status"].eq("filled")
    return result


def _next_session_map(calendar: pd.DataFrame) -> pd.DataFrame:
    _require(calendar, {"period", "score_session"}, "session calendar")
    sessions = (
        calendar.loc[:, ["period", "score_session"]]
        .drop_duplicates()
        .sort_values(["period", "score_session"], kind="stable")
    )
    sessions["target_session"] = sessions.groupby("period", observed=True)["score_session"].shift(
        -1
    )
    return sessions


def _equal_nonmissing(left: object, values: pd.Series) -> pd.Series:
    if left is None or left is pd.NA or bool(pd.isna(left)):  # type: ignore[call-overload]
        return pd.Series(False, index=values.index)
    return values.notna() & values.astype("string").eq(str(left))


def _identity_candidates(source: pd.Series, candidates: pd.DataFrame) -> pd.DataFrame:
    context_fields = [
        field
        for field in ("symbol_norm", "loop_id", "orientation", "horizon")
        if field in candidates and field in source.index
    ]
    contextual = candidates.copy()
    for field in context_fields:
        contextual = contextual.loc[_equal_nonmissing(source[field], contextual[field])]
    if contextual.empty:
        return contextual

    exact = contextual["opportunity_id"].astype(str).eq(str(source["opportunity_id"]))
    for field in PERSISTENT_ID_FIELDS:
        if field in contextual and field in source.index:
            exact |= _equal_nonmissing(source[field], contextual[field])
    return contextual.loc[exact]


def _structural_candidates(source: pd.Series, candidates: pd.DataFrame) -> pd.DataFrame:
    available = [
        field
        for field in STRUCTURAL_LINEAGE_FIELDS
        if field in source.index and field in candidates
    ]
    if set(STRUCTURAL_LINEAGE_FIELDS) - set(available):
        return candidates.iloc[0:0]
    structural = candidates.copy()
    for field in available:
        structural = structural.loc[_equal_nonmissing(source[field], structural[field])]
    return structural


def match_next_session_setups(
    source_opportunities: pd.DataFrame,
    all_opportunities: pd.DataFrame,
    session_calendar: pd.DataFrame,
) -> pd.DataFrame:
    """Classify next-session setup availability without outcome-based matching.

    Structural lineage is deliberately reported in a different column and is
    never promoted into the exact matched population.
    """

    required = {
        "period",
        "score_session",
        "opportunity_id",
        "symbol_norm",
        "loop_id",
        "orientation",
        "horizon",
    }
    _require(source_opportunities, required, "source opportunity")
    _require(all_opportunities, required, "opportunity universe")
    if source_opportunities["opportunity_id"].duplicated().any():
        raise ValueError("source opportunity IDs must be unique")

    mapped = source_opportunities.merge(
        _next_session_map(session_calendar),
        on=["period", "score_session"],
        how="left",
        validate="many_to_one",
    )
    records: list[dict[str, object]] = []
    for _, source in mapped.iterrows():
        base: dict[str, object] = {
            "source_opportunity_id": source["opportunity_id"],
            "source_period": source["period"],
            "source_session": source["score_session"],
            "target_session": source["target_session"],
            "exact_match": False,
            "matched_opportunity_id": pd.NA,
            "structural_lineage_opportunity_id": pd.NA,
            "exact_candidate_count": 0,
            "structural_lineage_candidate_count": 0,
        }
        if pd.isna(source["target_session"]):
            base["match_category"] = "delayed_entry_impossible_session_boundary"
            records.append(base)
            continue
        candidates = all_opportunities.loc[
            all_opportunities["period"].eq(source["period"])
            & all_opportunities["score_session"].eq(source["target_session"])
        ]
        exact = _identity_candidates(source, candidates)
        structural = _structural_candidates(source, candidates)
        base["exact_candidate_count"] = len(exact)
        base["structural_lineage_candidate_count"] = len(structural)
        if len(exact) == 1:
            base["match_category"] = "exact_same_setup_remained_available"
            base["exact_match"] = True
            base["matched_opportunity_id"] = exact.iloc[0]["opportunity_id"]
        elif len(exact) > 1:
            base["match_category"] = "ambiguous_identity"
        elif len(structural) == 1:
            base["match_category"] = "same_structural_lineage_not_exact_setup"
            base["structural_lineage_opportunity_id"] = structural.iloc[0]["opportunity_id"]
        elif len(structural) > 1:
            base["match_category"] = "ambiguous_identity"
        else:
            base["match_category"] = "setup_no_longer_available"
        records.append(base)
    return pd.DataFrame.from_records(records)


def _payoff_columns(row: pd.Series) -> tuple[float, float, float]:
    gross_field = "gross_payoff_bps" if "gross_payoff_bps" in row.index else "gross_return_bps"
    gross = float(row[gross_field])
    if "primary_total_cost_bps" in row.index and pd.notna(row["primary_total_cost_bps"]):
        cost = float(row["primary_total_cost_bps"])
    else:
        cost = sum(
            float(row[field])
            for field in (
                "entry_cost_bps",
                "exit_cost_bps",
                "spread_cost_bps",
                "slippage_cost_bps",
                "commission_cost_bps",
                "financing_cost_bps",
                "fx_cost_bps",
                "other_cost_bps",
            )
            if field in row.index and pd.notna(row[field])
        )
    return gross, cost, gross - cost


@dataclass(frozen=True)
class TradeDelayTables:
    exact_matches: pd.DataFrame
    restarted_horizon: pd.DataFrame
    constant_terminal: pd.DataFrame


def build_trade_delay_tables(
    matches: pd.DataFrame,
    opportunities: pd.DataFrame,
) -> TradeDelayTables:
    """Build separated exact/lineage and terminal-clock diagnostics."""

    _require(
        matches,
        {
            "source_opportunity_id",
            "exact_match",
            "matched_opportunity_id",
            "structural_lineage_opportunity_id",
            "match_category",
        },
        "match",
    )
    _require(
        opportunities,
        {
            "opportunity_id",
            "entry_timestamp",
            "exit_timestamp",
            "entry_price",
            "exit_price",
            "direction",
            "horizon",
        },
        "delay payoff",
    )
    if opportunities["opportunity_id"].duplicated().any():
        raise ValueError("opportunity IDs must be unique for delay payoff reconstruction")
    indexed = opportunities.set_index("opportunity_id", drop=False)
    restarted_records: list[dict[str, object]] = []
    constant_records: list[dict[str, object]] = []
    exact_records: list[dict[str, object]] = []

    for _, match in matches.iterrows():
        exact = bool(match["exact_match"])
        delayed_id = (
            match["matched_opportunity_id"] if exact else match["structural_lineage_opportunity_id"]
        )
        if pd.isna(delayed_id):
            continue
        source_id = match["source_opportunity_id"]
        if source_id not in indexed.index or delayed_id not in indexed.index:
            raise ValueError("matched opportunity is absent from payoff universe")
        source = indexed.loc[source_id]
        delayed = indexed.loc[delayed_id]
        if isinstance(source, pd.DataFrame) or isinstance(delayed, pd.DataFrame):
            raise ValueError("ambiguous opportunity lookup")
        immediate_gross, immediate_cost, immediate_net = _payoff_columns(source)
        delayed_gross, delayed_cost, delayed_net = _payoff_columns(delayed)
        basis = "exact_same_setup" if exact else "structural_lineage_diagnostic"
        restarted: dict[str, object] = {
            "source_opportunity_id": source_id,
            "delayed_opportunity_id": delayed_id,
            "match_basis": basis,
            "immediate_gross_payoff_bps": immediate_gross,
            "immediate_total_cost_bps": immediate_cost,
            "immediate_net_payoff_bps": immediate_net,
            "delayed_gross_payoff_bps": delayed_gross,
            "delayed_total_cost_bps": delayed_cost,
            "delayed_net_payoff_bps": delayed_net,
            "delayed_horizon_bars": int(delayed["horizon"]),
            "existing_position_action": delayed.get(
                "existing_position_action", "unchanged_existing_exit_rule"
            ),
        }
        restarted_records.append(restarted)
        if exact:
            exact_records.append(restarted.copy())

        original_terminal = pd.Timestamp(source["exit_timestamp"])
        delayed_entry = pd.Timestamp(delayed["entry_timestamp"])
        constant: dict[str, object] = {
            "source_opportunity_id": source_id,
            "delayed_opportunity_id": delayed_id,
            "match_basis": basis,
            "constant_terminal_available": False,
            "unavailable_reason": pd.NA,
            "delayed_constant_terminal_gross_bps": np.nan,
            "delayed_constant_terminal_cost_bps": delayed_cost,
            "delayed_constant_terminal_net_bps": np.nan,
        }
        if original_terminal <= delayed_entry:
            constant["unavailable_reason"] = "original_terminal_precedes_delayed_entry"
        elif pd.isna(source["exit_price"]) or pd.isna(delayed["entry_price"]):
            constant["unavailable_reason"] = "required_price_missing"
        else:
            direction = float(delayed["direction"])
            gross = (
                direction
                * (float(source["exit_price"]) / float(delayed["entry_price"]) - 1.0)
                * 10_000.0
            )
            constant["constant_terminal_available"] = True
            constant["unavailable_reason"] = pd.NA
            constant["delayed_constant_terminal_gross_bps"] = gross
            constant["delayed_constant_terminal_net_bps"] = gross - delayed_cost
        constant_records.append(constant)

    return TradeDelayTables(
        exact_matches=pd.DataFrame.from_records(exact_records),
        restarted_horizon=pd.DataFrame.from_records(restarted_records),
        constant_terminal=pd.DataFrame.from_records(constant_records),
    )


__all__ = [
    "STRUCTURAL_LINEAGE_FIELDS",
    "TradeDelayTables",
    "build_trade_delay_tables",
    "match_next_session_setups",
    "reconstruct_v2_shifted_policy",
]
