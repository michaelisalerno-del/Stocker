"""Immutable predicted-family opportunity attribution and cost stresses."""

from __future__ import annotations

import hashlib
import json

import pandas as pd


def _require(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"missing {label} columns: {missing}")


def _stable_id(values: tuple[object, ...]) -> str:
    payload = json.dumps([str(value) for value in values], separators=(",", ":"))
    return f"translation-{hashlib.sha256(payload.encode()).hexdigest()[:24]}"


def translate_predictions_to_opportunities(
    predictions: pd.DataFrame,
    opportunities: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    """Assign each eligible trade to the most recent same-family nomination only."""

    _require(
        predictions,
        {
            "forecast_id",
            "period",
            "forecast_session",
            "destination_family",
            "target_window_sessions",
            "model_name",
            "prediction_state",
            "predicted_activation_probability",
        },
        "prediction",
    )
    _require(
        opportunities,
        {
            "opportunity_id",
            "period",
            "score_session",
            "destination_family",
            "status",
            "stock_id",
            "entry_timestamp",
            "exit_timestamp",
            "gross_payoff_bps",
            "primary_total_cost_bps",
            "primary_net_payoff_bps",
        },
        "opportunity",
    )
    _require(calendar, {"period", "score_session"}, "calendar")
    session_index = {
        (int(str(period)), str(session)): index
        for period, group in calendar.groupby("period", sort=True, observed=True)
        for index, session in enumerate(
            group["score_session"].astype(str).drop_duplicates().sort_values()
        )
    }
    nominated = predictions.loc[predictions["prediction_state"].eq("nominated")].copy()
    nominated["_forecast_index"] = [
        session_index.get((int(str(period)), str(session)), -1)
        for period, session in nominated[["period", "forecast_session"]].itertuples(
            index=False, name=None
        )
    ]
    fills = opportunities.loc[
        opportunities["status"].eq("filled") & opportunities["primary_net_payoff_bps"].notna()
    ].copy()
    fills["_opportunity_index"] = [
        session_index.get((int(str(period)), str(session)), -1)
        for period, session in fills[["period", "score_session"]].itertuples(index=False, name=None)
    ]
    matched_records: list[dict[str, object]] = []
    used_forecasts: set[str] = set()
    for opportunity in fills.to_dict(orient="records"):
        candidates = nominated.loc[
            nominated["period"].eq(opportunity["period"])
            & nominated["destination_family"].eq(opportunity["destination_family"])
            & nominated["_forecast_index"].lt(int(opportunity["_opportunity_index"]))
            & (nominated["_forecast_index"] + nominated["target_window_sessions"].astype(int)).ge(
                int(opportunity["_opportunity_index"])
            )
        ].copy()
        for model_name, model_candidates in candidates.groupby(
            "model_name", sort=True, observed=True
        ):
            chosen = model_candidates.sort_values(
                ["_forecast_index", "forecast_id"], kind="stable"
            ).iloc[-1]
            forecast_id = str(chosen["forecast_id"])
            used_forecasts.add(forecast_id)
            record = {
                str(key): value
                for key, value in opportunity.items()
                if not str(key).startswith("_")
            }
            record.update(
                {
                    "forecast_id": forecast_id,
                    "forecast_session": str(chosen["forecast_session"]),
                    "model_name": str(model_name),
                    "predicted_activation_probability": float(
                        chosen["predicted_activation_probability"]
                    ),
                    "economic_translation_status": "eligible_opportunity",
                    "replacement_opportunity_used": False,
                    "capacity_refill_used": False,
                    "translation_id": _stable_id(
                        (forecast_id, opportunity["opportunity_id"], model_name)
                    ),
                }
            )
            matched_records.append(record)
    all_columns = [column for column in opportunities.columns]
    for forecast in nominated.to_dict(orient="records"):
        forecast_id = str(forecast["forecast_id"])
        if forecast_id in used_forecasts:
            continue
        possible = fills.loc[
            fills["period"].eq(forecast["period"])
            & fills["destination_family"].eq(forecast["destination_family"])
            & fills["_opportunity_index"].gt(int(forecast["_forecast_index"]))
            & fills["_opportunity_index"].le(
                int(forecast["_forecast_index"]) + int(forecast["target_window_sessions"])
            )
        ]
        status = (
            "superseded_by_later_forecast"
            if not possible.empty
            else "no_tradeable_destination_opportunity"
        )
        record = {column: pd.NA for column in all_columns}
        record.update(
            {
                "period": int(str(forecast["period"])),
                "destination_family": str(forecast["destination_family"]),
                "forecast_id": forecast_id,
                "forecast_session": str(forecast["forecast_session"]),
                "model_name": str(forecast["model_name"]),
                "predicted_activation_probability": float(
                    forecast["predicted_activation_probability"]
                ),
                "economic_translation_status": status,
                "replacement_opportunity_used": False,
                "capacity_refill_used": False,
                "translation_id": _stable_id((forecast_id, "missing", status)),
            }
        )
        matched_records.append(record)
    result = pd.DataFrame.from_records(matched_records)
    if result.empty:
        return result
    if (
        result.loc[result["economic_translation_status"].eq("eligible_opportunity")]
        .duplicated(["model_name", "opportunity_id"])
        .any()
    ):
        raise AssertionError("an opportunity was assigned more than once per model")
    return result.sort_values(
        ["period", "model_name", "forecast_session", "translation_id"], kind="stable"
    ).reset_index(drop=True)


def apply_cost_stress(frame: pd.DataFrame, *, multiplier: float) -> pd.DataFrame:
    """Reprice the existing immutable trade without changing entry or exit clocks."""

    if multiplier <= 0.0:
        raise ValueError("cost multiplier must be positive")
    _require(
        frame,
        {"gross_payoff_bps", "primary_total_cost_bps", "primary_net_payoff_bps"},
        "cost-stress",
    )
    result = frame.copy()
    gross = pd.to_numeric(result["gross_payoff_bps"], errors="coerce")
    cost = pd.to_numeric(result["primary_total_cost_bps"], errors="coerce")
    result["cost_multiplier"] = float(multiplier)
    result["stressed_total_cost_bps"] = cost * multiplier
    result["stressed_net_payoff_bps"] = gross - result["stressed_total_cost_bps"]
    result["entry_exit_rules_changed"] = False
    return result


__all__ = ["apply_cost_stress", "translate_predictions_to_opportunities"]
