"""Supported pair-level probability refinement with family shrinkage."""

from __future__ import annotations


def shrink_pair_probability(
    *,
    pair_activations: int,
    pair_support: int,
    family_probability: float,
    pooling_strength: float,
) -> float:
    """Shrink a secondary pair rate toward its frozen family forecast."""

    if pair_support < 0 or pair_activations < 0 or pair_activations > pair_support:
        raise ValueError("invalid pair activation counts")
    if pooling_strength <= 0.0:
        raise ValueError("pooling strength must be positive")
    if not 0.0 <= family_probability <= 1.0:
        raise ValueError("family probability must be in [0, 1]")
    if pair_support == 0:
        return float(family_probability)
    pair_rate = pair_activations / pair_support
    weight = pair_support / (pair_support + pooling_strength)
    return float(weight * pair_rate + (1.0 - weight) * family_probability)


__all__ = ["shrink_pair_probability"]
