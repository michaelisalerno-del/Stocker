"""Interpretable rule definitions, bounded generation, support, and voting."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import asdict, dataclass
from itertools import combinations, product
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from stocker_research.directional_signature_atlas.features import (
    assert_outcome_free_feature_names,
)

Direction = Literal["LONG", "SHORT"]


@dataclass(frozen=True)
class Condition:
    feature: str
    operator: str
    value: Any
    family: str


@dataclass(frozen=True)
class Signature:
    signature_id: str
    direction: Direction
    conditions: tuple[Condition, ...]
    source: str = "bounded_census"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchCaps:
    univariate_and_pairwise: int
    triples: int
    tree: int
    retained: int


@dataclass(frozen=True)
class SupportRules:
    minimum_rows: int
    minimum_sessions: int
    minimum_stocks: int
    maximum_stock_fraction: float
    minimum_months: int
    minimum_directional_outcomes: int


@dataclass(frozen=True)
class ControllerDecision:
    state: str
    reason: str
    long_votes: int
    short_votes: int


def _slug(value: Any) -> str:
    text = str(value).replace(" ", "_").replace(">", "gt").replace("<", "lt")
    return "".join(character for character in text if character.isalnum() or character in "_-.")[
        :48
    ]


def validate_signature(signature: Signature) -> None:
    if signature.direction not in {"LONG", "SHORT"}:
        raise ValueError("signature direction must be LONG or SHORT")
    if not 1 <= len(signature.conditions) <= 3:
        raise ValueError("signature must contain one to three conditions")
    features = [condition.feature for condition in signature.conditions]
    try:
        assert_outcome_free_feature_names(features)
    except ValueError as exc:
        if any(
            name.lower() in {"symbol", "symbol_norm", "stock_identity", "month"}
            for name in features
        ):
            raise ValueError("stock or month identity is forbidden") from exc
        if any("episode" in name.lower() for name in features):
            raise ValueError("outcome-derived episode is forbidden") from exc
        raise
    if any(name.startswith("loop_score_") for name in features):
        raise ValueError("full raw loop-score vector fields are forbidden")
    if len({condition.feature for condition in signature.conditions}) != len(signature.conditions):
        raise ValueError("a signature cannot repeat a feature")


def _condition_mask(frame: pd.DataFrame, condition: Condition) -> pd.Series:
    values = frame[condition.feature]
    if condition.operator == "==":
        return values.eq(condition.value)
    if condition.operator == "!=":
        return values.ne(condition.value) & values.notna()
    numeric = pd.to_numeric(values, errors="coerce")
    threshold = float(condition.value)
    if condition.operator == ">":
        return numeric.gt(threshold)
    if condition.operator == ">=":
        return numeric.ge(threshold)
    if condition.operator == "<":
        return numeric.lt(threshold)
    if condition.operator == "<=":
        return numeric.le(threshold)
    raise ValueError(f"unsupported signature operator {condition.operator}")


def apply_signature(frame: pd.DataFrame, signature: Signature) -> pd.Series:
    validate_signature(signature)
    mask = pd.Series(True, index=frame.index)
    for condition in signature.conditions:
        if condition.feature not in frame:
            return pd.Series(False, index=frame.index)
        mask &= _condition_mask(frame, condition).fillna(False)
    return mask


def _levels(frame: pd.DataFrame, feature: str, *, minimum_count: int = 1) -> list[Any]:
    counts = frame[feature].dropna().value_counts()
    values = counts.loc[counts.ge(minimum_count)].index.tolist()
    return sorted(values, key=lambda value: str(value))


def _make_signature(
    direction: Direction, conditions: tuple[Condition, ...], source: str
) -> Signature:
    body = "__".join(
        f"{condition.feature}_{_slug(condition.operator)}_{_slug(condition.value)}"
        for condition in conditions
    )
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()[:10]
    return Signature(f"{direction.lower()}__{body[:120]}__{digest}", direction, conditions, source)


def candidate_search_space_counts(
    discovery: pd.DataFrame,
    feature_families: dict[str, str],
) -> dict[str, int]:
    """Count the outcome-free space before the frozen balanced search cap."""

    levels = {
        feature: len(_levels(discovery, feature, minimum_count=1))
        for feature in sorted(feature_families)
    }
    univariate_rules = sum(levels.values())
    pairwise_rules = sum(
        levels[left] * levels[right]
        for left, right in combinations(sorted(levels), 2)
    )
    return {
        "features": len(levels),
        "observed_univariate_condition_rules": univariate_rules,
        "observed_pairwise_condition_rules": pairwise_rules,
        "observed_univariate_directional_candidates": 2 * univariate_rules,
        "observed_pairwise_directional_candidates": 2 * pairwise_rules,
    }


def generate_bounded_candidates(
    discovery: pd.DataFrame,
    feature_families: dict[str, str],
    caps: SearchCaps,
    *,
    minimum_parent_support: int = 1,
) -> tuple[list[Signature], list[dict[str, Any]]]:
    """Generate deterministic one-to-three-condition equality rules.

    Inputs must already be coarse-binned.  Candidate creation sees discovery
    features and support only; validation/final outcomes never enter.
    """

    assert_outcome_free_feature_names(feature_families)
    features = sorted(feature_families)
    conditions_by_feature = {
        feature: [
            Condition(feature, "==", value, feature_families[feature])
            for value in _levels(
                discovery,
                feature,
                minimum_count=1,
            )
        ]
        for feature in features
    }
    candidates: list[Signature] = []
    registry: list[dict[str, Any]] = []

    def rule_key(conditions: tuple[Condition, ...]) -> str:
        payload = json.dumps(
            [asdict(condition) for condition in conditions],
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def add(conditions: tuple[Condition, ...], stage: str) -> None:
        mask = pd.Series(True, index=discovery.index)
        for condition in conditions:
            mask &= _condition_mask(discovery, condition).fillna(False)
        support = int(mask.sum())
        reasons = [] if support else ["zero_discovery_support"]
        directions: tuple[Direction, ...] = ("LONG", "SHORT")
        for direction in directions:
            signature = _make_signature(direction, conditions, stage)
            registry.append(
                {
                    "signature_id": signature.signature_id,
                    "direction": direction,
                    "stage": stage,
                    "condition_count": len(conditions),
                    "conditions": [asdict(condition) for condition in conditions],
                    "feature_support_rows": support,
                    "rejection_reasons": reasons.copy(),
                }
            )
            candidates.append(signature)

    broad_cap = caps.univariate_and_pairwise
    broad_rule_cap = broad_cap // 2
    univariate_rule_quota = min(
        sum(len(values) for values in conditions_by_feature.values()),
        broad_rule_cap // 2,
    )
    pairwise_rule_quota = broad_rule_cap - univariate_rule_quota
    broad_rows = 0
    selected_pairs: list[tuple[Condition, ...]] = []
    univariate_queues = {
        feature: deque(conditions_by_feature[feature]) for feature in features
    }
    selected_univariate_rules = 0
    while selected_univariate_rules < univariate_rule_quota:
        made_progress = False
        for feature in features:
            queue = univariate_queues[feature]
            if not queue or selected_univariate_rules >= univariate_rule_quota:
                continue
            add((queue.popleft(),), "univariate")
            selected_univariate_rules += 1
            broad_rows += 2
            made_progress = True
        if not made_progress:
            break

    pair_iterators: deque[Any] = deque()
    feature_pairs = [
        (left, right)
        for left, right in combinations(features, 2)
        if conditions_by_feature[left] and conditions_by_feature[right]
    ]
    feature_pairs.sort(
        key=lambda pair: hashlib.sha256(f"{pair[0]}|{pair[1]}".encode()).hexdigest()
    )
    for left, right in feature_pairs:
        pair_iterators.append(
            iter(product(conditions_by_feature[left], conditions_by_feature[right]))
        )
    selected_pair_rules = 0
    while pair_iterators and selected_pair_rules < pairwise_rule_quota:
        iterator = pair_iterators.popleft()
        try:
            selected_pair = tuple(next(iterator))
        except StopIteration:
            continue
        add(selected_pair, "pairwise")
        selected_pairs.append(selected_pair)
        selected_pair_rules += 1
        broad_rows += 2
        pair_iterators.append(iterator)

    triple_pool: dict[str, tuple[Condition, ...]] = {}
    for selected_pair in selected_pairs:
        parent_mask = _condition_mask(discovery, selected_pair[0]).fillna(False)
        parent_mask &= _condition_mask(discovery, selected_pair[1]).fillna(False)
        if int(parent_mask.sum()) < minimum_parent_support:
            continue
        used_features = {condition.feature for condition in selected_pair}
        used_families = {condition.family for condition in selected_pair}
        for third_feature in features:
            if third_feature in used_features or feature_families[third_feature] in used_families:
                continue
            for third in conditions_by_feature[third_feature]:
                triple = tuple(
                    sorted((*selected_pair, third), key=lambda condition: condition.feature)
                )
                triple_pool[rule_key(triple)] = triple
    triple_rows = 0
    for key in sorted(triple_pool):
        if triple_rows + 2 > caps.triples:
            break
        add(triple_pool[key], "three_condition")
        triple_rows += 2
    return candidates, registry


def retain_ranked_candidates(
    candidates: list[Signature],
    scores: dict[str, float],
    cap: int,
) -> list[Signature]:
    """Enforce a frozen post-scoring retention cap with deterministic ties."""

    return sorted(
        candidates,
        key=lambda signature: (
            -float(scores.get(signature.signature_id, -np.inf)),
            len(signature.conditions),
            signature.signature_id,
        ),
    )[:cap]


def extract_shallow_tree_candidates(
    discovery: pd.DataFrame,
    feature_families: dict[str, str],
    *,
    maximum_depth: int,
    minimum_leaf_rows: int,
    cap: int,
    seed: int,
    long_payoff_column: str = "long_net_bps",
    short_payoff_column: str = "short_net_bps",
) -> tuple[list[Signature], list[dict[str, Any]]]:
    """Use depth-three discovery trees only to propose interpretable rules."""

    if maximum_depth > 3:
        raise ValueError("tree candidate generator depth cannot exceed three")
    encoded_columns: list[tuple[str, Any]] = []
    encoded_parts: list[np.ndarray] = []
    for feature in sorted(feature_families):
        for value in _levels(discovery, feature, minimum_count=minimum_leaf_rows):
            encoded_columns.append((feature, value))
            encoded_parts.append(discovery[feature].eq(value).to_numpy(dtype=float))
    if not encoded_parts or cap <= 0:
        return [], []
    matrix = np.column_stack(encoded_parts)
    candidates: list[Signature] = []
    registry: list[dict[str, Any]] = []
    seen: set[str] = set()
    direction_columns: tuple[tuple[Direction, str], ...] = (
        ("LONG", long_payoff_column),
        ("SHORT", short_payoff_column),
    )
    for direction, payoff_column in direction_columns:
        target = pd.to_numeric(discovery[payoff_column], errors="coerce").fillna(0.0)
        tree = DecisionTreeRegressor(
            max_depth=maximum_depth,
            min_samples_leaf=minimum_leaf_rows,
            random_state=seed,
        )
        tree.fit(matrix, target.to_numpy(float))

        def walk(
            node: int,
            path: tuple[Condition, ...],
            *,
            fitted_tree: DecisionTreeRegressor = tree,
            fitted_direction: Direction = direction,
        ) -> None:
            if len(registry) >= cap:
                return
            feature_index = int(fitted_tree.tree_.feature[node])
            if feature_index < 0:
                support = int(fitted_tree.tree_.n_node_samples[node])
                value = float(fitted_tree.tree_.value[node][0][0])
                if support < minimum_leaf_rows or value <= 0.0:
                    return
                if len({condition.feature for condition in path}) != len(path):
                    return
                signature = _make_signature(fitted_direction, path, "shallow_tree")
                try:
                    validate_signature(signature)
                except ValueError:
                    return
                if signature.signature_id in seen:
                    return
                seen.add(signature.signature_id)
                candidates.append(signature)
                registry.append(
                    {
                        "signature_id": signature.signature_id,
                        "direction": fitted_direction,
                        "stage": "shallow_tree",
                        "condition_count": len(path),
                        "conditions": [asdict(condition) for condition in path],
                        "feature_support_rows": support,
                        "tree_leaf_mean_directional_net_bps": value,
                        "rejection_reasons": [],
                    }
                )
                return
            feature, category = encoded_columns[feature_index]
            family = feature_families[feature]
            left = Condition(feature, "!=", category, family)
            right = Condition(feature, "==", category, family)
            walk(
                int(fitted_tree.tree_.children_left[node]),
                (*path, left),
                fitted_tree=fitted_tree,
                fitted_direction=fitted_direction,
            )
            walk(
                int(fitted_tree.tree_.children_right[node]),
                (*path, right),
                fitted_tree=fitted_tree,
                fitted_direction=fitted_direction,
            )

        walk(0, ())
    return candidates[:cap], registry[:cap]


def passes_support(
    rows: pd.DataFrame,
    direction: Direction,
    rules: SupportRules,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if len(rows) < rules.minimum_rows:
        reasons.append("insufficient_rows")
    if rows.get("session", pd.Series(dtype=object)).nunique() < rules.minimum_sessions:
        reasons.append("insufficient_sessions")
    stock_count = rows.get("symbol", pd.Series(dtype=object)).nunique()
    if stock_count < rules.minimum_stocks:
        reasons.append("insufficient_stocks")
    if len(rows):
        concentration = float(rows["symbol"].value_counts(normalize=True).max())
        if concentration > rules.maximum_stock_fraction:
            reasons.append("stock_concentration")
    months = rows.get("session", pd.Series(dtype=object)).astype(str).str[:7].nunique()
    if months < rules.minimum_months:
        reasons.append("insufficient_months")
    if (
        rows.get("target", pd.Series(dtype=object)).eq(direction).sum()
        < rules.minimum_directional_outcomes
    ):
        reasons.append("insufficient_directional_outcomes")
    return not reasons, reasons


def complexity_penalty(condition_count: int, per_extra_condition: float) -> float:
    if condition_count < 1:
        raise ValueError("condition count must be positive")
    return per_extra_condition * condition_count


def apply_multiple_testing(p_values: list[float], *, method: str) -> list[float]:
    if not p_values:
        return []
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    count = len(values)
    if method == "fdr_bh":
        adjusted_ranked = ranked * count / np.arange(1, count + 1)
        adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    elif method == "holm":
        adjusted_ranked = ranked * (count - np.arange(count))
        adjusted_ranked = np.maximum.accumulate(adjusted_ranked)
    else:
        raise ValueError(f"unknown multiplicity method {method}")
    adjusted = np.empty(count, dtype=float)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted.tolist()


def split_libraries(signatures: list[Signature]) -> tuple[list[Signature], list[Signature]]:
    return (
        [signature for signature in signatures if signature.direction == "LONG"],
        [signature for signature in signatures if signature.direction == "SHORT"],
    )


def controller_decision(
    long_votes: int,
    short_votes: int,
    movement_permitted: bool,
    aggregate_value_positive: bool,
    *,
    final_holdout_weights: dict[str, float] | None = None,
) -> ControllerDecision:
    """Apply one-rule/one-vote logic; holdout weights are deliberately ignored."""

    del final_holdout_weights
    if not movement_permitted:
        return ControllerDecision("NEUTRAL", "movement_permission_failed", long_votes, short_votes)
    if long_votes and short_votes:
        return ControllerDecision("NEUTRAL", "conflicting_votes", long_votes, short_votes)
    if not long_votes and not short_votes:
        return ControllerDecision("NEUTRAL", "no_directional_vote", long_votes, short_votes)
    if not aggregate_value_positive:
        return ControllerDecision(
            "NEUTRAL", "non_positive_conservative_value", long_votes, short_votes
        )
    state = "LONG" if long_votes else "SHORT"
    return ControllerDecision(state, "supported_directional_vote", long_votes, short_votes)


def library_sha256(signatures: list[Signature]) -> str:
    payload = [signature.to_dict() for signature in signatures]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
