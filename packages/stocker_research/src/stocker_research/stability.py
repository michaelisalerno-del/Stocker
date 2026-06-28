"""Parameter stability analysis."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any


@dataclass(frozen=True)
class StabilityReport:
    """Compact stability summary for a parameter grid."""

    best_parameter_set_id: str
    best_test_return: float
    median_test_return: float
    profitable_neighbour_pct: float
    train_to_test_degradation: float
    stability_score: float
    isolated_warning: bool

    def to_dict(self) -> dict[str, float | bool | str]:
        """Return JSON-serializable report data."""

        return {
            "best_parameter_set_id": self.best_parameter_set_id,
            "best_test_return": self.best_test_return,
            "median_test_return": self.median_test_return,
            "profitable_neighbour_pct": self.profitable_neighbour_pct,
            "train_to_test_degradation": self.train_to_test_degradation,
            "stability_score": self.stability_score,
            "isolated_warning": self.isolated_warning,
        }


def analyze_stability(
    results: list[dict[str, Any]],
    *,
    best_parameter_set_id: str | None = None,
) -> StabilityReport:
    """Score whether performance survives neighbouring parameter sets."""

    if not results:
        raise ValueError("results must not be empty")
    best = max(results, key=lambda row: float(row["test_net_return"]))
    if best_parameter_set_id is not None:
        best = next(
            (row for row in results if row["parameter_set_id"] == best_parameter_set_id),
            best,
        )
    best_id = str(best["parameter_set_id"])
    best_test = float(best["test_net_return"])
    best_train = float(best["train_net_return"])
    neighbours = [row for row in results if row["parameter_set_id"] != best_id]
    if neighbours:
        profitable_pct = sum(float(row["test_net_return"]) > 0 for row in neighbours) / len(
            neighbours
        )
    else:
        profitable_pct = 0.0
    med = median(float(row["test_net_return"]) for row in results)
    degradation = best_train - best_test
    degradation_score = max(0.0, min(1.0, 1.0 - max(0.0, degradation)))
    stability_score = max(0.0, min(1.0, 0.6 * profitable_pct + 0.4 * degradation_score))
    return StabilityReport(
        best_parameter_set_id=best_id,
        best_test_return=best_test,
        median_test_return=float(med),
        profitable_neighbour_pct=float(profitable_pct),
        train_to_test_degradation=float(degradation),
        stability_score=float(stability_score),
        isolated_warning=profitable_pct < 0.5,
    )
