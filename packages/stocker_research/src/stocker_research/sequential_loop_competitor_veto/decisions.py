"""Posterior summaries, operational decisions, and fixed-population veto accounting."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import PosteriorSnapshot


@dataclass(frozen=True)
class DecisionConfig:
    bad_mass_veto_threshold: float = 0.5
    minimum_good_to_bad_odds: float = 1.0
    maximum_unknown_mass: float = 0.5
    maximum_normalised_entropy: float = 0.65
    minimum_bars_remaining: int = 6
    lower_bound_z: float = 1.6448536269514722


@dataclass(frozen=True)
class PosteriorSummary:
    good_mass: float
    bad_mass: float
    unknown_mass: float
    good_to_bad_odds: float
    good_to_non_good_odds: float
    compatible_good_count: int
    compatible_bad_count: int
    compatible_unknown_count: int
    posterior_entropy: float
    normalised_entropy: float
    expected_remaining_net_bps: float
    expected_remaining_std_bps: float
    conservative_remaining_net_bps: float
    p_positive_remaining: float
    bars_remaining: int
    target_compatible: bool = True


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def summarise_posterior(
    snapshot: PosteriorSnapshot,
    payoff_classes: Mapping[str, str],
    posterior_means: Mapping[str, float],
    posterior_stds: Mapping[str, float],
    *,
    bars_remaining: int,
    horizon_bars: int = 24,
    round_trip_cost_bps: float = 10.0,
    lower_bound_z: float = 1.6448536269514722,
    target_loop: str | None = None,
) -> PosteriorSummary:
    """Collapse loop probabilities without treating unknown as a known-good loop."""

    if horizon_bars <= 0:
        raise ValueError("horizon must be positive")
    fraction = min(1.0, max(0.0, float(bars_remaining) / float(horizon_bars)))
    component_rows: list[tuple[float, float, float]] = []
    good_mass = 0.0
    bad_mass = 0.0
    unclassified_known = 0.0
    good_count = 0
    bad_count = 0
    unknown_count = 0
    for loop, mass in snapshot.known.items():
        if mass <= 0.0:
            continue
        payoff_class = str(payoff_classes.get(loop, "unknown"))
        if payoff_class == "good":
            good_mass += mass
            good_count += 1
        elif payoff_class == "bad":
            bad_mass += mass
            bad_count += 1
        else:
            unclassified_known += mass
            unknown_count += 1
        mean = float(posterior_means.get(loop, 0.0))
        std = max(1e-9, float(posterior_stds.get(loop, 120.0)))
        remaining_mean = (mean + round_trip_cost_bps) * fraction - round_trip_cost_bps
        remaining_std = std * math.sqrt(max(fraction, 1e-12))
        component_rows.append((mass, remaining_mean, remaining_std))

    total_unknown = float(snapshot.unknown) + unclassified_known
    if total_unknown > 0.0:
        component_rows.append(
            (
                total_unknown,
                -round_trip_cost_bps,
                120.0 * math.sqrt(max(fraction, 1e-12)),
            )
        )
    expected = sum(mass * mean for mass, mean, _ in component_rows)
    variance = sum(
        mass * (std * std + (mean - expected) ** 2) for mass, mean, std in component_rows
    )
    std = math.sqrt(max(variance, 0.0))
    positive = sum(
        mass * _normal_cdf(mean / std_component) for mass, mean, std_component in component_rows
    )
    probabilities = [mass for mass, _, _ in component_rows if mass > 0.0]
    entropy = -sum(mass * math.log(mass) for mass in probabilities)
    normaliser = math.log(len(probabilities)) if len(probabilities) > 1 else 1.0
    normalised_entropy = entropy / normaliser if len(probabilities) > 1 else 0.0
    non_good = bad_mass + total_unknown
    return PosteriorSummary(
        good_mass=good_mass,
        bad_mass=bad_mass,
        unknown_mass=total_unknown,
        good_to_bad_odds=(good_mass / bad_mass if bad_mass > 0.0 else math.inf),
        good_to_non_good_odds=(good_mass / non_good if non_good > 0.0 else math.inf),
        compatible_good_count=good_count,
        compatible_bad_count=bad_count,
        compatible_unknown_count=unknown_count,
        posterior_entropy=entropy,
        normalised_entropy=normalised_entropy,
        expected_remaining_net_bps=expected,
        expected_remaining_std_bps=std,
        conservative_remaining_net_bps=expected - lower_bound_z * std,
        p_positive_remaining=min(1.0, max(0.0, positive)),
        bars_remaining=int(bars_remaining),
        target_compatible=(
            target_loop is None
            or (
                snapshot.statuses.get(target_loop) != "impossible"
                and snapshot.known.get(target_loop, 0.0) > 0.0
            )
        ),
    )


def classify_decision(summary: PosteriorSummary, config: DecisionConfig) -> str:
    """Apply the frozen rejection-first operational state rule."""

    if not summary.target_compatible:
        return "reject"
    if summary.bars_remaining < config.minimum_bars_remaining:
        return "reject"
    if (
        summary.bad_mass >= config.bad_mass_veto_threshold
        and summary.conservative_remaining_net_bps <= 0.0
    ):
        return "reject"
    admit = (
        summary.compatible_good_count > 0
        and summary.good_to_bad_odds > config.minimum_good_to_bad_odds
        and summary.unknown_mass <= config.maximum_unknown_mass
        and summary.normalised_entropy <= config.maximum_normalised_entropy
        and summary.conservative_remaining_net_bps > 0.0
    )
    return "admit" if admit else "unresolved"


def apply_irreversible_decisions(
    timeline: pd.DataFrame,
    *,
    identity_columns: tuple[str, ...] = ("opportunity_id",),
) -> pd.DataFrame:
    """Make rejection absorbing without changing the opportunity population."""

    required = {*identity_columns, "checkpoint_timestamp", "proposed_decision"}
    missing = required - set(timeline.columns)
    if missing:
        raise ValueError(f"missing decision columns: {sorted(missing)}")
    frame = timeline.copy()
    frame["checkpoint_timestamp"] = pd.to_datetime(
        frame["checkpoint_timestamp"], utc=True, errors="raise"
    )
    frame["_position"] = np.arange(len(frame))
    frame = frame.sort_values(
        [*identity_columns, "checkpoint_timestamp", "_position"], kind="stable"
    )
    decisions: list[str] = []
    reasons: list[str] = []
    rejected: set[tuple[str, ...]] = set()
    for row in frame.itertuples(index=False):
        identity = tuple(str(getattr(row, column)) for column in identity_columns)
        proposed = str(row.proposed_decision)
        if identity in rejected:
            decisions.append("reject")
            reasons.append("prior_rejection_irreversible")
            continue
        if proposed == "reject":
            rejected.add(identity)
        decisions.append(proposed)
        reasons.append(f"proposed_{proposed}")
    frame["decision_state"] = decisions
    frame["reason_codes"] = reasons
    return (
        frame.sort_values("_position", kind="stable")
        .drop(columns="_position")
        .reset_index(drop=True)
    )


def apply_veto_accounting(
    base_opportunities: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    payoff_column: str = "net_payoff_bps",
) -> pd.DataFrame:
    """Account for a veto without replacement, overlap refill, or exit changes."""

    if base_opportunities["opportunity_id"].duplicated().any():
        raise ValueError("base opportunity IDs must be unique")
    if decisions["opportunity_id"].duplicated().any():
        raise ValueError("policy decisions must contain one row per opportunity")
    base_ids = set(base_opportunities["opportunity_id"].astype(str))
    decision_ids = set(decisions["opportunity_id"].astype(str))
    if base_ids != decision_ids:
        raise ValueError("decision population differs from immutable base opportunities")
    frame = base_opportunities.merge(
        decisions[["opportunity_id", "decision_state"]],
        on="opportunity_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    payoff = pd.to_numeric(frame[payoff_column], errors="coerce")
    retained = frame["decision_state"].eq("admit")
    frame["policy_net_payoff_bps"] = np.where(retained, payoff, 0.0)
    frame["veto_value_bps"] = frame["policy_net_payoff_bps"] - payoff
    frame["existing_position_action"] = "unchanged"
    frame["replacement_opportunity_id"] = pd.NA
    return frame
