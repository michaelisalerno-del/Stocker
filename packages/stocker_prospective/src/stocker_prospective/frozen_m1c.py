"""Exact no-fit runtime for the frozen causal M1C movement model."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
from pydantic import BaseModel, ConfigDict

M1C_THRESHOLD = 0.488333710794033
MINIMUM_EPISODE_SPACING_MINUTES = 30

CAUSAL_GROUP_I_FEATURES = (
    "arousal",
    "conviction",
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
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"frozen artifact must be a mapping: {path}")
    return payload


def _finite_or_none(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class FrozenM1CScore(BaseModel):
    """One immutable score and its exact preprocessing identity."""

    model_config = ConfigDict(frozen=True)

    model_id: str = "M1C"
    model_hash: str
    probability: float
    threshold: float
    threshold_passed: bool
    feature_order: tuple[str, ...]
    feature_values: tuple[float | None, ...]
    transformed_values: tuple[float, ...]
    feature_hash: str
    missing_feature_count: int


class FrozenM1CRuntime:
    """Load and score the committed M1C specification without fitting."""

    def __init__(
        self,
        *,
        numeric_features: tuple[str, ...],
        numeric_medians: np.ndarray[Any, np.dtype[np.float64]],
        numeric_means: np.ndarray[Any, np.dtype[np.float64]],
        numeric_scales: np.ndarray[Any, np.dtype[np.float64]],
        stock_levels: tuple[str, ...],
        design_columns: tuple[str, ...],
        coefficients: np.ndarray[Any, np.dtype[np.float64]],
        intercept: float,
        model_hash: str,
    ) -> None:
        self.numeric_features = numeric_features
        self.numeric_medians = numeric_medians
        self.numeric_means = numeric_means
        self.numeric_scales = numeric_scales
        self.stock_levels = stock_levels
        self.design_columns = design_columns
        self.coefficients = coefficients
        self.intercept = intercept
        self.model_hash = model_hash
        self.threshold = M1C_THRESHOLD
        self.causal_group_i_features = CAUSAL_GROUP_I_FEATURES
        self.required_group_o_features = tuple(
            name
            for name in self.numeric_features
            if name not in self.causal_group_i_features and not name.startswith("checkpoint_")
        )

    def missing_group_o_features(
        self,
        group_o_context: Mapping[str, object],
    ) -> tuple[str, ...]:
        """Distinguish an explicit missing value from an absent frozen feature key."""

        return tuple(name for name in self.required_group_o_features if name not in group_o_context)

    @classmethod
    def from_artifacts(
        cls,
        *,
        feature_manifest_path: str | Path,
        threshold_path: str | Path,
    ) -> FrozenM1CRuntime:
        """Verify the frozen artifact surface and construct a no-fit scorer."""

        manifest_path = Path(feature_manifest_path)
        gate_path = Path(threshold_path)
        manifest = _mapping(manifest_path)
        threshold = _mapping(gate_path)
        if manifest.get("model") != "M1C" or threshold.get("model") != "M1C":
            raise ValueError("frozen M1C artifact identity differs")
        artifact_threshold = float(threshold["threshold"])
        if abs(artifact_threshold - M1C_THRESHOLD) > 1e-15:
            raise ValueError("frozen M1C threshold differs from the binding threshold")
        causal = tuple(str(value) for value in manifest.get("causally_valid_group_i", ()))
        if causal != CAUSAL_GROUP_I_FEATURES:
            raise ValueError("frozen causal Group I order differs")
        removed = {
            str(value)
            for key in (
                "removed_future_contaminated_group_i",
                "removed_other_peer_normalised_group_i",
            )
            for value in manifest.get(key, ())
        }
        if not {"signed_pressure", "tension"}.issubset(removed):
            raise ValueError("contaminated peer-slate lineage was not fully removed")
        if manifest.get("replacement_features_added") != []:
            raise ValueError("M1C must not contain replacement features")
        specification = manifest.get("model_specification")
        if not isinstance(specification, dict):
            raise ValueError("M1C model specification is absent")
        if (
            specification.get("model_id") != "M1C"
            or specification.get("kind") != "logistic"
            or tuple(specification.get("category_controls", ())) != ("stock",)
        ):
            raise ValueError("M1C model contract differs")
        numeric_features = tuple(str(value) for value in specification["numeric_features"])
        if numeric_features[-len(CAUSAL_GROUP_I_FEATURES) :] != CAUSAL_GROUP_I_FEATURES:
            raise ValueError("M1C numerical feature order differs")
        if {"signed_pressure", "tension"}.intersection(numeric_features):
            raise ValueError("M1C contains a contaminated feature")
        stock_levels = tuple(str(value) for value in specification["category_levels"]["stock"])
        design_columns = tuple(str(value) for value in specification["design_columns"])
        expected_design = (
            *numeric_features,
            *(f"control_stock__{stock}" for stock in stock_levels[1:]),
        )
        if design_columns != expected_design:
            raise ValueError("M1C design feature order differs")
        arrays = {
            "numeric_medians": np.asarray(specification["numeric_medians"], dtype=np.float64),
            "numeric_means": np.asarray(specification["numeric_means"], dtype=np.float64),
            "numeric_scales": np.asarray(specification["numeric_scales"], dtype=np.float64),
            "coefficients": np.asarray(specification["coefficients"], dtype=np.float64),
        }
        if any(not np.isfinite(values).all() for values in arrays.values()):
            raise ValueError("M1C artifact contains non-finite parameters")
        if any(
            len(arrays[name]) != len(numeric_features)
            for name in ("numeric_medians", "numeric_means", "numeric_scales")
        ):
            raise ValueError("M1C preprocessing width differs")
        if len(arrays["coefficients"]) != len(design_columns):
            raise ValueError("M1C coefficient width differs")
        if bool((arrays["numeric_scales"] <= 0.0).any()):
            raise ValueError("M1C numerical scales must be positive")
        intercept = float(specification["intercept"])
        if not math.isfinite(intercept):
            raise ValueError("M1C intercept must be finite")
        return cls(
            numeric_features=numeric_features,
            numeric_medians=arrays["numeric_medians"],
            numeric_means=arrays["numeric_means"],
            numeric_scales=arrays["numeric_scales"],
            stock_levels=stock_levels,
            design_columns=design_columns,
            coefficients=arrays["coefficients"],
            intercept=intercept,
            model_hash=_sha256(manifest_path),
        )

    def score(
        self,
        *,
        symbol: str,
        checkpoint: int,
        group_o_context: Mapping[str, object],
        causal_group_i: Mapping[str, object],
    ) -> FrozenM1CScore:
        """Apply exact imputation, scaling, category encoding, and logistic score."""

        if symbol not in self.stock_levels:
            raise ValueError(f"unknown frozen M1C stock level: {symbol}")
        checkpoint_name = f"checkpoint_{int(checkpoint)}"
        checkpoint_features = {
            name for name in self.numeric_features if name.startswith("checkpoint_")
        }
        if checkpoint_name not in checkpoint_features:
            raise ValueError(f"checkpoint outside frozen M1C grid: {checkpoint}")
        raw_values: list[float | None] = []
        for name in self.numeric_features:
            if name in self.causal_group_i_features:
                raw = causal_group_i.get(name)
            elif name.startswith("checkpoint_"):
                raw = 1.0 if name == checkpoint_name else 0.0
            else:
                raw = group_o_context.get(name)
            raw_values.append(_finite_or_none(raw))
        missing = np.asarray([value is None for value in raw_values], dtype=bool)
        numeric = np.asarray(
            [
                self.numeric_medians[index] if value is None else value
                for index, value in enumerate(raw_values)
            ],
            dtype=np.float64,
        )
        transformed = (numeric - self.numeric_means) / self.numeric_scales
        stock_controls = np.asarray(
            [float(symbol == stock) for stock in self.stock_levels[1:]],
            dtype=np.float64,
        )
        design = np.concatenate([transformed, stock_controls])
        if len(design) != len(self.design_columns) or not np.isfinite(design).all():
            raise ValueError("M1C design construction failed")
        linear = float(design @ self.coefficients + self.intercept)
        if linear >= 0.0:
            probability = 1.0 / (1.0 + math.exp(-linear))
        else:
            exponential = math.exp(linear)
            probability = exponential / (1.0 + exponential)
        serialized = json.dumps(
            {
                "symbol": symbol,
                "checkpoint": checkpoint,
                "feature_order": self.numeric_features,
                "feature_values": raw_values,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return FrozenM1CScore(
            model_hash=self.model_hash,
            probability=probability,
            threshold=self.threshold,
            threshold_passed=probability >= self.threshold,
            feature_order=self.numeric_features,
            feature_values=tuple(raw_values),
            transformed_values=tuple(float(value) for value in transformed),
            feature_hash=hashlib.sha256(serialized).hexdigest(),
            missing_feature_count=int(missing.sum()),
        )


class EpisodeDecision(BaseModel):
    """One raw checkpoint decision and optional deterministic fresh episode."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    session: date
    checkpoint: int
    probability: float
    threshold: float
    raw_above_threshold: bool
    previous_probability: float | None
    fresh_episode: bool
    episode_id: str | None
    episode_number: int | None
    minutes_since_previous_episode: float | None
    trigger_bar_end: datetime
    prospective_entry_timestamp: datetime
    rejection_reason: str | None


class FreshEpisodeTracker:
    """Reproduce the exact frozen stock-session crossing and spacing rule."""

    def __init__(
        self,
        *,
        threshold: float = M1C_THRESHOLD,
        minimum_spacing_minutes: int = MINIMUM_EPISODE_SPACING_MINUTES,
    ) -> None:
        if threshold != M1C_THRESHOLD:
            raise ValueError("M1C threshold is frozen")
        if minimum_spacing_minutes != MINIMUM_EPISODE_SPACING_MINUTES:
            raise ValueError("M1C episode spacing is frozen at thirty minutes")
        self.threshold = threshold
        self.minimum_spacing_minutes = minimum_spacing_minutes
        self._previous_eligible: dict[tuple[str, date], float] = {}
        self._previous_episode: dict[tuple[str, date], datetime] = {}
        self._episode_count: dict[tuple[str, date], int] = {}

    def restore_session(
        self,
        *,
        symbol: str,
        session: date,
        previous_eligible_probability: float | None,
        previous_episode_timestamp: datetime | None,
        episode_count: int,
    ) -> None:
        """Restore persisted state once after a recorder restart."""

        key = (symbol, session)
        if (
            key in self._previous_eligible
            or key in self._previous_episode
            or key in self._episode_count
        ):
            raise ValueError("fresh-episode session state is already initialized")
        if previous_eligible_probability is not None and (
            not math.isfinite(previous_eligible_probability)
            or not 0.0 <= previous_eligible_probability <= 1.0
        ):
            raise ValueError("persisted M1C probability is invalid")
        if episode_count < 0:
            raise ValueError("persisted episode count is invalid")
        if previous_episode_timestamp is not None:
            if (
                previous_episode_timestamp.tzinfo is None
                or previous_episode_timestamp.utcoffset() is None
            ):
                raise ValueError("persisted episode timestamp must be timezone-aware")
            self._previous_episode[key] = previous_episode_timestamp.astimezone(UTC)
        if previous_eligible_probability is not None:
            self._previous_eligible[key] = previous_eligible_probability
        if episode_count:
            self._episode_count[key] = episode_count

    def evaluate(
        self,
        *,
        symbol: str,
        session: date,
        checkpoint: int,
        trigger_bar_end: datetime,
        probability: float,
        eligible: bool = True,
        rejection_reason: str | None = None,
    ) -> EpisodeDecision:
        if trigger_bar_end.tzinfo is None or trigger_bar_end.utcoffset() is None:
            raise ValueError("trigger timestamp must be timezone-aware")
        timestamp = trigger_bar_end.astimezone(UTC)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("M1C probability must lie in [0, 1]")
        key = (symbol, session)
        previous = self._previous_eligible.get(key)
        above = probability >= self.threshold
        crossing = eligible and above and (previous is None or previous < self.threshold)
        previous_episode = self._previous_episode.get(key)
        elapsed = (
            None
            if previous_episode is None
            else (timestamp - previous_episode).total_seconds() / 60.0
        )
        spacing_passed = elapsed is None or elapsed >= self.minimum_spacing_minutes
        fresh = crossing and spacing_passed
        episode_id: str | None = None
        episode_number: int | None = None
        reason = rejection_reason
        if crossing and not spacing_passed:
            reason = "minimum_episode_spacing_not_met"
        if fresh:
            episode_number = self._episode_count.get(key, 0) + 1
            raw_identity = "|".join(
                (
                    "M1C",
                    symbol,
                    session.isoformat(),
                    str(int(checkpoint)),
                    timestamp.isoformat(),
                )
            )
            episode_id = f"m1c-{hashlib.sha256(raw_identity.encode()).hexdigest()[:24]}"
            self._episode_count[key] = episode_number
            self._previous_episode[key] = timestamp
        if eligible:
            self._previous_eligible[key] = probability
        return EpisodeDecision(
            symbol=symbol,
            session=session,
            checkpoint=int(checkpoint),
            probability=probability,
            threshold=self.threshold,
            raw_above_threshold=eligible and above,
            previous_probability=previous,
            fresh_episode=fresh,
            episode_id=episode_id,
            episode_number=episode_number,
            minutes_since_previous_episode=elapsed,
            trigger_bar_end=timestamp,
            prospective_entry_timestamp=timestamp,
            rejection_reason=reason,
        )


__all__ = [
    "CAUSAL_GROUP_I_FEATURES",
    "M1C_THRESHOLD",
    "EpisodeDecision",
    "FreshEpisodeTracker",
    "FrozenM1CRuntime",
    "FrozenM1CScore",
]
