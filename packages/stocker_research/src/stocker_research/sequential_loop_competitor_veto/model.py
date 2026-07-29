"""Causal compatible-loop sets and transparent sequential posterior updates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PosteriorSnapshot:
    """Normalised known-loop probabilities plus an explicit residual unknown mass."""

    known: dict[str, float]
    unknown: float
    statuses: dict[str, str]
    eliminated: tuple[str, ...]
    completed: tuple[str, ...]


def parse_cycle(cycle: str) -> tuple[int, ...]:
    """Parse and validate a closed state cycle."""

    states = tuple(int(value) for value in cycle.split("->"))
    if len(states) < 3 or states[0] != states[-1]:
        raise ValueError(f"cycle must be closed and contain a transition: {cycle}")
    return states


def oriented_paths(cycle: str, anchor_state: int) -> tuple[tuple[int, ...], ...]:
    """Return every distinct rotation of ``cycle`` beginning at ``anchor_state``."""

    states = parse_cycle(cycle)
    open_path = states[:-1]
    rotations: list[tuple[int, ...]] = []
    for index, state in enumerate(open_path):
        if state != anchor_state:
            continue
        rotated_open = open_path[index:] + open_path[:index]
        rotated = rotated_open + (rotated_open[0],)
        if rotated not in rotations:
            rotations.append(rotated)
    return tuple(rotations)


def compatibility_status(
    cycle: str,
    anchor_state: int,
    observed_transitions: Sequence[int],
) -> str:
    """Classify a loop using only transitions observable at the checkpoint.

    A completed parent remains compatible with the observed occurrence.  A loop is
    impossible only when every rotation beginning at the anchor disagrees before
    completion.
    """

    paths = oriented_paths(cycle, anchor_state)
    if not paths:
        return "impossible"
    observed = tuple(int(value) for value in observed_transitions)
    compatible = False
    for path in paths:
        expected = path[1:]
        comparable = observed[: len(expected)]
        if comparable != expected[: len(comparable)]:
            continue
        if len(observed) >= len(expected):
            return "completed"
        compatible = True
    return "compatible" if compatible else "impossible"


def initial_posterior(loop_masses: Mapping[str, float]) -> PosteriorSnapshot:
    """Build the anchor posterior without forcing residual mass into known loops."""

    known = {str(loop): float(mass) for loop, mass in loop_masses.items()}
    if any(mass < 0.0 for mass in known.values()):
        raise ValueError("loop mass cannot be negative")
    total = sum(known.values())
    if total > 1.0 + 1e-12:
        raise ValueError("known loop masses exceed one; residual unknown is undefined")
    unknown = max(0.0, 1.0 - total)
    return PosteriorSnapshot(
        known=known,
        unknown=unknown,
        statuses={loop: "compatible" for loop in known},
        eliminated=(),
        completed=(),
    )


def update_posterior(
    prior: PosteriorSnapshot,
    cycles: Mapping[str, str],
    anchor_state: int,
    observed_transitions: Sequence[int],
    *,
    evidence_likelihoods: Mapping[str, float] | None = None,
    minimum_possible_likelihood: float = 0.05,
    unknown_likelihood: float = 1.0,
) -> PosteriorSnapshot:
    """Apply observable structural exclusions and smoothed causal likelihoods."""

    if minimum_possible_likelihood <= 0.0:
        raise ValueError("minimum possible likelihood must be positive")
    if unknown_likelihood < 0.0:
        raise ValueError("unknown likelihood cannot be negative")
    likelihoods = evidence_likelihoods or {}
    weights: dict[str, float] = {}
    statuses: dict[str, str] = {}
    eliminated = set(prior.eliminated)
    completed = set(prior.completed)
    for loop, prior_mass in prior.known.items():
        if loop not in cycles:
            raise KeyError(f"missing cycle definition for {loop}")
        status = compatibility_status(cycles[loop], anchor_state, observed_transitions)
        statuses[loop] = status
        if status == "impossible":
            weights[loop] = 0.0
            eliminated.add(loop)
            continue
        if status == "completed":
            completed.add(loop)
        likelihood = max(
            minimum_possible_likelihood,
            float(likelihoods.get(loop, 1.0)),
        )
        weights[loop] = float(prior_mass) * likelihood

    unknown_weight = float(prior.unknown) * unknown_likelihood
    denominator = sum(weights.values()) + unknown_weight
    if denominator <= 0.0:
        normalised = {loop: 0.0 for loop in weights}
        unknown = 1.0
    else:
        normalised = {loop: weight / denominator for loop, weight in weights.items()}
        unknown = unknown_weight / denominator
    return PosteriorSnapshot(
        known=normalised,
        unknown=unknown,
        statuses=statuses,
        eliminated=tuple(sorted(eliminated)),
        completed=tuple(sorted(completed)),
    )
