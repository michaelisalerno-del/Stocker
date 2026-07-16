"""Past-only directed family activation census with empirical-Bayes shrinkage."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_distribution

SOURCE_EVENT_TYPES = ("active", "newly_decaying", "newly_retired")


@dataclass(frozen=True)
class GraphSettings:
    base_alpha: float = 1.0
    base_beta: float = 9.0
    edge_alpha: float = 1.0
    edge_beta: float = 1.0
    pooling_strength: float = 20.0
    minimum_source_event_sessions: int = 8
    log_lift_clip: tuple[float, float] = (-math.log(4.0), math.log(4.0))

    def __post_init__(self) -> None:
        if min(self.base_alpha, self.base_beta, self.edge_alpha, self.edge_beta) <= 0.0:
            raise ValueError("beta prior values must be positive")
        if self.pooling_strength <= 0.0:
            raise ValueError("pooling strength must be positive")
        if self.minimum_source_event_sessions < 1:
            raise ValueError("minimum source-event support must be positive")
        if self.log_lift_clip[0] >= self.log_lift_clip[1]:
            raise ValueError("invalid lift clip")


@dataclass(frozen=True)
class MaturedRotationExample:
    example_id: str
    period: int
    forecast_session: str
    destination_family: str
    activation_target: bool
    label_availability_timestamp: pd.Timestamp
    source_events: Mapping[str, frozenset[str]]

    def __post_init__(self) -> None:
        timestamp = pd.Timestamp(self.label_availability_timestamp)
        if timestamp.tzinfo is None:
            raise ValueError("label availability must be timezone-aware")
        invalid = sorted(
            {
                event
                for events in self.source_events.values()
                for event in events
                if event not in SOURCE_EVENT_TYPES
            }
        )
        if invalid:
            raise ValueError(f"unregistered source events: {invalid}")


@dataclass(frozen=True)
class EdgeSummary:
    source_family: str
    source_event: str
    destination_family: str
    support: int
    activations: int
    raw_transition_probability: float
    destination_base_rate: float
    directed_lift: float
    shrunk_lift: float
    interval_lower: float
    interval_upper: float
    support_status: str


class PastOnlyRotationGraph:
    """Update a directed graph only after each target label becomes available."""

    def __init__(self, settings: GraphSettings) -> None:
        self.settings = settings
        self._base: dict[str, list[int]] = {}
        self._edges: dict[tuple[str, str, str], list[int]] = {}
        self._seen: set[str] = set()
        self.latest_label_availability: pd.Timestamp | None = None

    def base_counts(self, destination_family: str) -> tuple[int, int]:
        counts = self._base.get(str(destination_family), [0, 0])
        return int(counts[0]), int(counts[1])

    def base_rate(self, destination_family: str) -> float:
        support, activations = self.base_counts(destination_family)
        return float(
            (activations + self.settings.base_alpha)
            / (support + self.settings.base_alpha + self.settings.base_beta)
        )

    def base_interval(self, destination_family: str) -> tuple[float, float]:
        support, activations = self.base_counts(destination_family)
        alpha = activations + self.settings.base_alpha
        beta = support - activations + self.settings.base_beta
        return (
            float(beta_distribution.ppf(0.025, alpha, beta)),
            float(beta_distribution.ppf(0.975, alpha, beta)),
        )

    def update_matured(
        self,
        examples: Sequence[MaturedRotationExample],
        *,
        as_of: pd.Timestamp,
    ) -> int:
        cutoff = pd.Timestamp(as_of)
        if cutoff.tzinfo is None:
            raise ValueError("graph cutoff must be timezone-aware")
        eligible = sorted(
            (
                example
                for example in examples
                if example.example_id not in self._seen
                and pd.Timestamp(example.label_availability_timestamp) < cutoff
            ),
            key=lambda value: (
                pd.Timestamp(value.label_availability_timestamp),
                value.example_id,
            ),
        )
        for example in eligible:
            base = self._base.setdefault(example.destination_family, [0, 0])
            base[0] += 1
            base[1] += int(example.activation_target)
            for source_family, events in sorted(example.source_events.items()):
                if source_family == example.destination_family:
                    continue
                for event in sorted(events):
                    edge = self._edges.setdefault(
                        (source_family, event, example.destination_family), [0, 0]
                    )
                    edge[0] += 1
                    edge[1] += int(example.activation_target)
            self._seen.add(example.example_id)
            availability = pd.Timestamp(example.label_availability_timestamp)
            self.latest_label_availability = (
                availability
                if self.latest_label_availability is None
                else max(self.latest_label_availability, availability)
            )
        return len(eligible)

    def edge_summary(
        self,
        source_family: str,
        source_event: str,
        destination_family: str,
    ) -> EdgeSummary:
        if source_event not in SOURCE_EVENT_TYPES:
            raise ValueError(f"unregistered source event: {source_event}")
        same = source_family == destination_family
        support, activations = self._edges.get(
            (source_family, source_event, destination_family), [0, 0]
        )
        if same:
            support, activations = 0, 0
        alpha = activations + self.settings.edge_alpha
        beta = support - activations + self.settings.edge_beta
        edge_rate = float(alpha / (alpha + beta))
        base_rate = self.base_rate(destination_family)
        lift = edge_rate / max(base_rate, 1e-12)
        weight = support / (support + self.settings.pooling_strength)
        clipped_log_lift = float(np.clip(math.log(max(lift, 1e-12)), *self.settings.log_lift_clip))
        shrunk_lift = math.exp(weight * clipped_log_lift)
        status = (
            "same_family_excluded"
            if same
            else "supported"
            if support >= self.settings.minimum_source_event_sessions
            else "unknown"
        )
        return EdgeSummary(
            source_family=source_family,
            source_event=source_event,
            destination_family=destination_family,
            support=int(support),
            activations=int(activations),
            raw_transition_probability=edge_rate,
            destination_base_rate=base_rate,
            directed_lift=float(lift),
            shrunk_lift=float(shrunk_lift),
            interval_lower=float(beta_distribution.ppf(0.025, alpha, beta)),
            interval_upper=float(beta_distribution.ppf(0.975, alpha, beta)),
            support_status=status,
        )

    def directed_features(
        self,
        destination_family: str,
        source_events: Mapping[str, frozenset[str]],
    ) -> dict[str, float]:
        scores = {event: 0.0 for event in SOURCE_EVENT_TYPES}
        supported = 0
        possible = 0
        positive: list[float] = []
        for source_family, events in sorted(source_events.items()):
            if source_family == destination_family:
                continue
            for event in sorted(events):
                possible += 1
                edge = self.edge_summary(source_family, event, destination_family)
                if edge.support_status != "supported":
                    continue
                supported += 1
                log_lift = math.log(max(edge.shrunk_lift, 1e-12))
                scores[event] += log_lift
                positive.append(max(log_lift, 0.0))
        return {
            "directed_active_log_lift_score": scores["active"],
            "directed_newly_decaying_log_lift_score": scores["newly_decaying"],
            "directed_newly_retired_log_lift_score": scores["newly_retired"],
            "maximum_supported_positive_directed_log_lift": max(positive, default=0.0),
            "supported_source_edge_fraction": supported / possible if possible else 0.0,
        }

    def rows(self, families: Sequence[str]) -> pd.DataFrame:
        records = [
            self.edge_summary(source, event, destination).__dict__
            for source in sorted(families)
            for event in SOURCE_EVENT_TYPES
            for destination in sorted(families)
            if source != destination
        ]
        return pd.DataFrame.from_records(records)


__all__ = [
    "EdgeSummary",
    "GraphSettings",
    "MaturedRotationExample",
    "PastOnlyRotationGraph",
    "SOURCE_EVENT_TYPES",
]
