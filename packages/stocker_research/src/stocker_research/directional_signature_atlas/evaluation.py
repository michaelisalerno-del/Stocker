"""Economic evaluation and unchanged-population baselines."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from stocker_research.directional_signature_atlas.signatures import Signature, apply_signature


def evaluate_signature(frame: pd.DataFrame, signature: Signature) -> dict[str, Any]:
    mask = apply_signature(frame, signature)
    selected = frame.loc[mask]
    direction = signature.direction
    payoff_column = "long_net_bps" if direction == "LONG" else "short_net_bps"
    if payoff_column not in selected:
        alternate = "net_long_return_bps" if direction == "LONG" else "net_short_return_bps"
        payoff_column = alternate
    base_rate = float(frame["target"].eq(direction).mean()) if len(frame) else float("nan")
    direction_rate = (
        float(selected["target"].eq(direction).mean()) if len(selected) else float("nan")
    )
    payoff = pd.Series(
        pd.to_numeric(selected.get(payoff_column, pd.Series(dtype=float)), errors="coerce"),
        dtype=float,
    )
    if "round_trip_cost_bps" not in selected:
        raise ValueError("exact round_trip_cost_bps is required for economic evaluation")
    cost = pd.Series(
        pd.to_numeric(selected["round_trip_cost_bps"], errors="coerce"),
        dtype=float,
    )
    return {
        "signature_id": signature.signature_id,
        "direction": direction,
        "rows": int(len(selected)),
        "sessions": int(selected.get("session", pd.Series(dtype=object)).nunique()),
        "stocks": int(selected.get("symbol", pd.Series(dtype=object)).nunique()),
        "direction_rate": direction_rate,
        "base_direction_rate": base_rate,
        "directional_lift": direction_rate - base_rate if len(selected) else float("nan"),
        "mean_directional_net_bps": float(payoff.mean()) if len(payoff) else float("nan"),
        "median_directional_net_bps": float(payoff.median()) if len(payoff) else float("nan"),
        "positive_payoff_rate": float(payoff.gt(0).mean()) if len(payoff) else float("nan"),
        "double_cost_mean_net_bps": float((payoff.to_numpy(float) - cost.to_numpy(float)).mean())
        if len(payoff)
        else float("nan"),
    }


def survives_validation(
    discovery: dict[str, Any],
    validation: dict[str, Any],
    require_double_cost: bool,
) -> bool:
    checks = [
        float(discovery["mean_directional_net_bps"]) > 0.0,
        float(validation["mean_directional_net_bps"]) > 0.0,
        float(discovery["directional_lift"]) > 0.0,
        float(validation["directional_lift"]) > 0.0,
    ]
    if require_double_cost:
        checks.append(float(validation["double_cost_mean_net_bps"]) > 0.0)
    return all(checks)


def null_permute_outcomes_within_period(frame: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """Permute the complete outcome tuple within each chronological period."""

    output = frame.copy()
    outcome_columns = [
        column
        for column in (
            "target",
            "long_net_bps",
            "short_net_bps",
            "net_long_return_bps",
            "net_short_return_bps",
        )
        if column in output
    ]
    rng = np.random.default_rng(seed)
    for _, positions in output.groupby("period", sort=False).groups.items():
        location = np.asarray(list(positions))
        permutation = rng.permutation(len(location))
        output.loc[location, outcome_columns] = output.loc[location, outcome_columns].to_numpy()[
            permutation
        ]
    return output


def baseline_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Return mandatory baselines on the exact input opportunity population."""

    output = frame[["opportunity_id", "decision_clock"]].copy()
    returns = pd.to_numeric(frame["return_1"], errors="coerce")
    output["always_neutral"] = "NEUTRAL"
    output["one_bar_momentum"] = np.where(
        returns.gt(0), "LONG", np.where(returns.lt(0), "SHORT", "NEUTRAL")
    )
    output["one_bar_reversal"] = np.where(
        returns.gt(0), "SHORT", np.where(returns.lt(0), "LONG", "NEUTRAL")
    )
    opening = pd.to_numeric(
        frame.get("opening_range_position", pd.Series(np.nan, index=frame.index)), errors="coerce"
    )
    output["opening_range_sign"] = np.where(
        opening.gt(0.5), "LONG", np.where(opening.lt(0.5), "SHORT", "NEUTRAL")
    )
    return output
