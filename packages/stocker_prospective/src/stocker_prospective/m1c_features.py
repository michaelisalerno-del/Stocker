"""Exact live construction of the frozen causal M1C Group-I features."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final, cast

from pydantic import BaseModel, ConfigDict, field_validator

EPSILON: Final[float] = 1e-12
AROUSAL_COMPONENTS: Final[tuple[str, ...]] = (
    "activity_effort",
    "range_effort",
    "travel_effort",
)
CONVICTION_COMPONENTS: Final[tuple[str, ...]] = (
    "absolute_efficiency",
    "close_retention",
    "directional_persistence",
)
CAUSAL_COMPONENTS: Final[tuple[str, ...]] = (
    *AROUSAL_COMPONENTS,
    *CONVICTION_COMPONENTS,
)
LOCAL_FEATURES: Final[tuple[str, ...]] = (
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
CAUSAL_M1C_FEATURES: Final[tuple[str, ...]] = (
    "arousal",
    "conviction",
    *LOCAL_FEATURES,
)
FROZEN_CHECKPOINTS: Final[tuple[int, ...]] = tuple(range(6, 35, 2))


class LiveFeatureBar(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str
    session: date
    bar_ordinal: int
    bar_start_timestamp: datetime
    bar_complete_timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    historical_relative_activity: float | None
    finalised: bool
    source: str

    @field_validator("bar_start_timestamp", "bar_complete_timestamp")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("bar timestamps must be timezone-aware")
        return value.astimezone(UTC)


class M1CCausalFeatureResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str
    session: date
    checkpoint: int
    trigger_bar_ordinal: int
    feature_available_timestamp_utc: datetime
    raw_components: dict[str, float]
    raw_local_features: dict[str, float]
    scaled_features: dict[str, float]
    feature_hash: str
    scaling_artifact_hash: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class HistoricalActivityBaseline:
    """Prior-session-only mean volume by stock and bar ordinal."""

    def __init__(self, *, minimum_sessions: int = 10) -> None:
        if minimum_sessions <= 0:
            raise ValueError("minimum_sessions must be positive")
        self.minimum_sessions = minimum_sessions
        self._values: dict[tuple[str, int], list[tuple[date, float]]] = {}
        self._latest_session: dict[str, date] = {}

    def commit_session(
        self,
        *,
        symbol: str,
        session: date,
        volume_by_ordinal: Mapping[int, float],
    ) -> None:
        latest = self._latest_session.get(symbol)
        if latest is not None and session <= latest:
            raise ValueError("activity baseline sessions must be appended chronologically")
        if any(
            ordinal < 0 or not math.isfinite(volume) or volume < 0.0
            for ordinal, volume in volume_by_ordinal.items()
        ):
            raise ValueError("activity baseline values are invalid")
        for ordinal, volume in sorted(volume_by_ordinal.items()):
            self._values.setdefault((symbol, ordinal), []).append((session, volume))
        self._latest_session[symbol] = session

    def relative_activity(
        self,
        *,
        symbol: str,
        session: date,
        bar_ordinal: int,
        volume: float,
    ) -> float | None:
        latest = self._latest_session.get(symbol)
        if latest is not None and session <= latest:
            raise ValueError("activity baseline chronology would include same/future session")
        history = [
            value
            for observed_session, value in self._values.get((symbol, bar_ordinal), ())
            if observed_session < session
        ]
        if len(history) < self.minimum_sessions:
            return None
        mean = math.fsum(history) / len(history)
        if mean <= EPSILON or not math.isfinite(volume) or volume < 0.0:
            return None
        return float(volume / mean)

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        *,
        latest_authorised_session: date,
        minimum_sessions: int = 10,
    ) -> HistoricalActivityBaseline:
        import pandas as pd

        frame = pd.read_parquet(
            path,
            columns=["symbol", "session", "bar_ordinal", "volume"],
            filters=[("session", "<=", latest_authorised_session.isoformat())],
        )
        baseline = cls(minimum_sessions=minimum_sessions)
        frame["session"] = pd.to_datetime(frame["session"], errors="raise").dt.date
        for (symbol, session), rows in frame.sort_values(
            ["symbol", "session", "bar_ordinal"],
            kind="mergesort",
        ).groupby(["symbol", "session"], sort=False):
            baseline.commit_session(
                symbol=str(symbol),
                session=cast(date, session),
                volume_by_ordinal={
                    int(cast(Any, row.bar_ordinal)): float(cast(Any, row.volume))
                    for row in rows.itertuples(index=False)
                },
            )
        return baseline


def _bar_components(bars: tuple[LiveFeatureBar, ...]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    previous_close = bars[0].open
    for bar in bars:
        width = bar.high - bar.low
        true_range = (
            10_000.0
            * max(width, abs(bar.high - previous_close), abs(bar.low - previous_close))
            / previous_close
        )
        return_bps = 10_000.0 * (bar.close / previous_close - 1.0)
        if width > EPSILON:
            close_location = (bar.close - bar.low) / width
            upper_wick = (bar.high - max(bar.open, bar.close)) / width
            lower_wick = (min(bar.open, bar.close) - bar.low) / width
        else:
            close_location = 0.5
            upper_wick = 0.0
            lower_wick = 0.0
        assert bar.historical_relative_activity is not None
        rows.append(
            {
                "return_bps": return_bps,
                "true_range_bps": true_range,
                "close_location": min(max(close_location, 0.0), 1.0),
                "upper_wick_fraction": min(max(upper_wick, 0.0), 1.0),
                "lower_wick_fraction": min(max(lower_wick, 0.0), 1.0),
                "historical_relative_activity": bar.historical_relative_activity,
            }
        )
        previous_close = bar.close
    return rows


def _raw_components(
    bars: tuple[LiveFeatureBar, ...],
    components: list[dict[str, float]],
) -> dict[str, float]:
    returns = [row["return_bps"] for row in components]
    ranges = [row["true_range_bps"] for row in components]
    activity = [row["historical_relative_activity"] for row in components]
    total_travel = math.fsum(abs(value) for value in returns)
    net = math.fsum(returns)
    cumulative = 10_000.0 * (bars[-1].close / bars[0].open - 1.0)
    if abs(cumulative) <= EPSILON:
        persistence = 0.5
    else:
        direction = 1.0 if cumulative > 0.0 else -1.0
        persistence = sum(
            (1.0 if value > 0.0 else -1.0 if value < 0.0 else 0.0) == direction for value in returns
        ) / len(returns)
    signed_efficiency = net / max(total_travel, EPSILON)
    return {
        "activity_effort": math.log1p(math.fsum(activity) / len(activity)),
        "range_effort": math.log1p(math.fsum(ranges)),
        "travel_effort": math.log1p(total_travel),
        "absolute_efficiency": abs(signed_efficiency),
        "close_retention": abs(bars[-1].close - bars[0].open)
        / max(math.fsum(bar.high - bar.low for bar in bars), EPSILON),
        "directional_persistence": persistence,
    }


def _raw_local(
    bars: tuple[LiveFeatureBar, ...],
    components: list[dict[str, float]],
) -> dict[str, float]:
    trailing_components = components[-6:]
    trailing_bars = bars[-6:]
    ranges = [row["true_range_bps"] for row in trailing_components]
    returns = [row["return_bps"] for row in trailing_components]
    activity = [row["historical_relative_activity"] for row in trailing_components]
    mean_range = math.fsum(ranges) / 6.0
    mean_activity = math.fsum(activity) / 6.0
    current = trailing_components[-1]
    current_bar = trailing_bars[-1]
    width = current_bar.high - current_bar.low
    body = abs(current_bar.close - current_bar.open) / max(width, EPSILON)
    return {
        "prior_6_mean_range": mean_range,
        "prior_6_price_travel": math.fsum(abs(value) for value in returns),
        "prior_6_absolute_net_movement": abs(math.fsum(returns)),
        "prior_6_activity_proxy": mean_activity,
        "recent_vs_earlier_range_ratio": (math.fsum(ranges[3:]) / 3.0)
        / max(math.fsum(ranges[:3]) / 3.0, EPSILON),
        "recent_vs_earlier_activity_ratio": (math.fsum(activity[3:]) / 3.0)
        / max(math.fsum(activity[:3]) / 3.0, EPSILON),
        "current_bar_range_vs_prior_6": current["true_range_bps"] / max(mean_range, EPSILON),
        "current_bar_activity_vs_prior_6": current["historical_relative_activity"]
        / max(mean_activity, EPSILON),
        "current_bar_body_fraction": min(max(body, 0.0), 1.0),
        "current_bar_extreme_wick_fraction": max(
            current["upper_wick_fraction"],
            current["lower_wick_fraction"],
        ),
    }


class M1CCausalFeatureBuilder:
    """Apply predecessor-frozen 2024 component and stock/checkpoint scaling."""

    def __init__(
        self,
        *,
        component_scaling: Mapping[str, Any],
        local_scaling: Mapping[str, Any],
        scaling_artifact_hash: str,
    ) -> None:
        self.component_scaling = component_scaling
        self.local_scaling = local_scaling
        self.scaling_artifact_hash = scaling_artifact_hash

    @classmethod
    def from_scaling_artifact(
        cls,
        path: str | Path,
    ) -> M1CCausalFeatureBuilder:
        artifact = Path(path)
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        return cls(
            component_scaling=cast(Mapping[str, Any], payload["component_development_scaling"]),
            local_scaling=cast(Mapping[str, Any], payload["local_development_scaling"]),
            scaling_artifact_hash=_sha256(artifact),
        )

    def build(
        self,
        *,
        symbol: str,
        checkpoint: int,
        completed_bars: tuple[LiveFeatureBar, ...],
    ) -> M1CCausalFeatureResult:
        if checkpoint not in FROZEN_CHECKPOINTS:
            raise ValueError("checkpoint is outside the frozen M1C grid")
        if len(completed_bars) != checkpoint:
            raise ValueError("completed bar count must equal checkpoint")
        if any(not bar.finalised for bar in completed_bars):
            raise ValueError("all feature bars must be finalised")
        if any(bar.symbol != symbol for bar in completed_bars):
            raise ValueError("bar symbol identity mismatch")
        sessions = {bar.session for bar in completed_bars}
        if len(sessions) != 1:
            raise ValueError("M1C bars must belong to one session")
        if [bar.bar_ordinal for bar in completed_bars] != list(range(checkpoint)):
            raise ValueError("M1C bar ordinals must be contiguous")
        if any(
            bar.historical_relative_activity is None
            or not math.isfinite(bar.historical_relative_activity)
            or bar.historical_relative_activity < 0.0
            for bar in completed_bars
        ):
            raise ValueError("historical relative activity is unavailable or invalid")
        if any(
            not all(
                math.isfinite(value) and value > 0.0
                for value in (bar.open, bar.high, bar.low, bar.close)
            )
            or bar.high < max(bar.open, bar.close, bar.low)
            or bar.low > min(bar.open, bar.close, bar.high)
            for bar in completed_bars
        ):
            raise ValueError("bar OHLC is invalid")
        components = _bar_components(completed_bars)
        raw_components = _raw_components(completed_bars, components)
        raw_local = _raw_local(completed_bars, components)
        checkpoint_scaling = cast(
            Mapping[str, Mapping[str, float]],
            self.component_scaling[str(checkpoint)],
        )
        scaled_components = {
            name: min(
                max(
                    (raw_components[name] - float(checkpoint_scaling[name]["center"]))
                    / float(checkpoint_scaling[name]["scale"]),
                    float(checkpoint_scaling[name].get("clip_lower", -5.0)),
                ),
                float(checkpoint_scaling[name].get("clip_upper", 5.0)),
            )
            for name in CAUSAL_COMPONENTS
        }
        local_key = f"{symbol}|{checkpoint}"
        if local_key not in self.local_scaling:
            raise ValueError(f"frozen local scaling unavailable for {local_key}")
        frozen_local = cast(
            Mapping[str, Mapping[str, float]],
            self.local_scaling[local_key],
        )
        scaled: dict[str, float] = {
            "arousal": math.fsum(scaled_components[name] for name in AROUSAL_COMPONENTS)
            / len(AROUSAL_COMPONENTS),
            "conviction": math.fsum(scaled_components[name] for name in CONVICTION_COMPONENTS)
            / len(CONVICTION_COMPONENTS),
        }
        for name in LOCAL_FEATURES:
            frozen = frozen_local[name]
            scaled[name] = min(
                max(
                    (raw_local[name] - float(frozen["center"])) / float(frozen["scale"]),
                    -5.0,
                ),
                5.0,
            )
        if tuple(scaled) != CAUSAL_M1C_FEATURES or not all(
            math.isfinite(value) for value in scaled.values()
        ):
            raise RuntimeError("causal M1C feature construction drifted")
        feature_payload = json.dumps(
            scaled,
            sort_keys=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        return M1CCausalFeatureResult(
            symbol=symbol,
            session=next(iter(sessions)),
            checkpoint=checkpoint,
            trigger_bar_ordinal=checkpoint - 1,
            feature_available_timestamp_utc=completed_bars[-1].bar_complete_timestamp,
            raw_components=raw_components,
            raw_local_features=raw_local,
            scaled_features=scaled,
            feature_hash=hashlib.sha256(feature_payload.encode("utf-8")).hexdigest(),
            scaling_artifact_hash=self.scaling_artifact_hash,
        )
