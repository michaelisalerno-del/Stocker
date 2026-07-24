"""Machine-readable parity gate for the frozen M1 feature contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

FROZEN_M1_FEATURES = (
    "front_options_implied_tension",
    "front_options_premium_richness",
    "front_options_downside_asymmetry",
    "front_options_liquidity_stress",
    "front_options_positioning_concentration",
    "front_options_directional_positioning",
    "front_options_surface_disagreement",
    "front_options_regime_p_0",
    "front_options_regime_p_1",
    "front_options_regime_p_2",
    "front_options_regime_p_3",
    "front_options_regime_entropy",
    "front_options_regime_margin",
    "skew_25d_missing",
    "near_spot_oi_concentration_missing",
    "call_put_oi_imbalance_missing",
    "checkpoint_6",
    "checkpoint_8",
    "checkpoint_10",
    "checkpoint_12",
    "checkpoint_14",
    "checkpoint_16",
    "checkpoint_18",
    "checkpoint_20",
    "checkpoint_22",
    "checkpoint_24",
    "checkpoint_26",
    "checkpoint_28",
    "checkpoint_30",
    "checkpoint_32",
    "checkpoint_34",
    "arousal",
    "conviction",
    "tension",
    "signed_pressure",
    "posterior_entropy",
    "transition_probability",
    "persistence_probability",
    "expected_state_age",
    "top_state_probability",
    "top_second_margin",
    "prior_6_mean_range",
    "prior_6_price_travel",
    "prior_6_absolute_net_movement",
    "prior_6_activity_proxy",
    "recent_vs_earlier_range_ratio",
    "recent_vs_earlier_activity_ratio",
    "current_bar_range_vs_prior_6",
    "current_bar_activity_vs_prior_6",
    "current_bar_body_fraction",
    "current_bar_extreme_wick_fraction",
    "any_registered_completion_prior_6",
    "any_registered_completion_prior_12",
    "same_identity_active_prefix_with_prior_completion",
    "any_hidden_event_prior_6",
    "hidden_2_3_2_prior_6",
    "bars_since_latest_registered_completion",
)


class FeatureParityError(RuntimeError):
    """A parity report cannot authorize frozen-model scoring."""


class FeatureParityItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    feature_name: str
    historical_source: str
    runtime_source: str
    historical_definition: str
    runtime_definition: str
    parity_status: Literal[
        "exact",
        "verified_equivalent",
        "requires_parallel_validation",
        "incompatible",
        "missing",
    ]
    explanation: str
    scoring_allowed: bool


class FeatureParityReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"]
    generated_at_utc: str
    frozen_model_source_artifact: str
    frozen_model_source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    frozen_feature_list_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    historical_bar_source: str
    planned_runtime_bar_source: str
    overall_scoring_allowed: bool
    overall_blocker: str
    features: tuple[FeatureParityItem, ...]

    def require_scoring_allowed(self) -> None:
        if not self.overall_scoring_allowed:
            raise FeatureParityError(self.overall_blocker)
        blocked = [item.feature_name for item in self.features if not item.scoring_allowed]
        if blocked:
            raise FeatureParityError(
                f"blocked_feature_source_semantics_mismatch: {','.join(blocked)}"
            )


def load_feature_parity_report(path: str | Path) -> FeatureParityReport:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        report = FeatureParityReport.model_validate(payload)
    except Exception as exc:
        raise FeatureParityError("blocked_feature_schema_mismatch: invalid parity report") from exc
    actual = tuple(item.feature_name for item in report.features)
    if actual != FROZEN_M1_FEATURES or len(set(actual)) != len(actual):
        raise FeatureParityError(
            "blocked_feature_schema_mismatch: parity report does not match frozen M1 order"
        )
    if report.overall_scoring_allowed != all(item.scoring_allowed for item in report.features):
        raise FeatureParityError(
            "blocked_feature_schema_mismatch: overall parity gate is inconsistent"
        )
    if any(
        item.scoring_allowed and item.parity_status not in {"exact", "verified_equivalent"}
        for item in report.features
    ):
        raise FeatureParityError(
            "blocked_feature_schema_mismatch: unsafe feature parity authorization"
        )
    return report
