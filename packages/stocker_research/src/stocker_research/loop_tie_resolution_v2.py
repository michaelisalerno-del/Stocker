"""Resolve frozen V2 registered ties into primitive-first semantic outcomes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, cast

import pandas as pd

from stocker_research.semantic_loop_dictionary_v2 import decompose_semantic_path, safety_flags


class TieClass(StrEnum):
    NESTED_SAME_PRIMITIVE_TIE = "NESTED_SAME_PRIMITIVE_TIE"
    DISTINCT_PRIMITIVE_TIE = "DISTINCT_PRIMITIVE_TIE"
    PRIMITIVE_COMPOSITE_TIE = "PRIMITIVE_COMPOSITE_TIE"
    DISTINCT_COMPOSITE_TIE = "DISTINCT_COMPOSITE_TIE"
    MIGRATION_OR_IDENTITY_TIE = "MIGRATION_OR_IDENTITY_TIE"
    UNKNOWN_TIE = "UNKNOWN_TIE"


@dataclass(frozen=True, slots=True)
class TieResolutionBundle:
    classification: pd.DataFrame
    nested_mapping: pd.DataFrame
    primary_rewrite: pd.DataFrame
    summary: pd.DataFrame


def _root_for_row(
    row: Any,
    *,
    composite_components: Mapping[str, Sequence[str]],
) -> str | None:
    motif = str(row.motif_type) if pd.notna(row.motif_type) else ""
    if motif in {"primitive", "repeat"}:
        return str(row.primitive_loop_id) if pd.notna(row.primitive_loop_id) else None
    if motif == "composite":
        full_path = getattr(row, "full_path", None)
        path_values = cast(Sequence[Any], full_path) if pd.api.types.is_list_like(full_path) else ()
        if len(path_values) > 0:
            stack: list[int] = []
            final_root: str | None = None
            for raw_state in path_values:
                state = int(raw_state)
                if state not in stack:
                    stack.append(state)
                    continue
                start = stack.index(state)
                closed_component = tuple(stack[start:] + [state])
                final_root = decompose_semantic_path(closed_component).primitive_loop_id
                stack = stack[:start] + [state]
            return final_root
        components = tuple(composite_components.get(str(row.semantic_loop_id), ()))
        return str(components[-1]) if components else None
    return None


def resolve_registered_ties(
    outcomes: pd.DataFrame,
    completions: pd.DataFrame,
    *,
    composite_components: Mapping[str, Sequence[str]] | None = None,
    legacy_aliases: Mapping[str, str] | None = None,
) -> TieResolutionBundle:
    """Classify every old tied primary completion without lexical tie-breaking."""

    required_outcomes = {"decision_id", "primary_label"}
    required_completions = {
        "decision_id",
        "semantic_loop_id",
        "primitive_loop_id",
        "motif_type",
        "repeat_depth",
        "is_primary_completion",
    }
    if missing := sorted(required_outcomes.difference(outcomes.columns)):
        raise ValueError(f"outcomes lack tie fields: {missing}")
    if missing := sorted(required_completions.difference(completions.columns)):
        raise ValueError(f"completions lack tie fields: {missing}")
    if outcomes["decision_id"].duplicated().any():
        raise ValueError("outcome decision IDs are not unique")

    component_lookup = composite_components or {}
    aliases = legacy_aliases or {}
    tied = outcomes.loc[outcomes["primary_label"].astype(str).eq("TIED_REGISTERED_COMPLETION")]
    primary = completions.loc[completions["is_primary_completion"].astype(bool)]
    grouped = {str(key): group for key, group in primary.groupby("decision_id", sort=False)}
    classification_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []

    for outcome in tied.itertuples(index=False):
        decision_id = str(outcome.decision_id)
        group = grouped.get(decision_id, pd.DataFrame(columns=primary.columns))
        semantic_ids = sorted(group["semantic_loop_id"].dropna().astype(str).unique().tolist())
        mapped_ids = sorted({aliases.get(identifier, identifier) for identifier in semantic_ids})
        alias_only = len(semantic_ids) > 1 and len(mapped_ids) == 1
        roots: list[str] = []
        explicit_roots: list[str] = []
        composite_ids: list[str] = []
        unknown = group.empty
        maximum_repeat_depth = 1
        for event in group.itertuples(index=False):
            motif = str(event.motif_type) if pd.notna(event.motif_type) else ""
            root = _root_for_row(event, composite_components=component_lookup)
            if root is None:
                unknown = True
            else:
                roots.append(root)
                if motif in {"primitive", "repeat"}:
                    explicit_roots.append(root)
            if motif == "composite":
                composite_ids.append(str(event.semantic_loop_id))
            if pd.notna(event.repeat_depth):
                maximum_repeat_depth = max(maximum_repeat_depth, int(cast(Any, event.repeat_depth)))
            mapping_rows.append(
                {
                    "decision_id": decision_id,
                    "semantic_loop_id": (
                        str(event.semantic_loop_id) if pd.notna(event.semantic_loop_id) else None
                    ),
                    "motif_type": motif or None,
                    "primitive_root": root,
                    "nested_under_primary": root is not None,
                    **safety_flags(),
                }
            )

        unique_roots = sorted(set(roots))
        unique_explicit = sorted(set(explicit_roots))
        if unknown:
            tie_class = TieClass.UNKNOWN_TIE
            rewritten = "UNAVAILABLE_STRUCTURAL_GAP"
        elif alias_only:
            tie_class = TieClass.MIGRATION_OR_IDENTITY_TIE
            rewritten = unique_roots[0] if len(unique_roots) == 1 else "UNAVAILABLE_STRUCTURAL_GAP"
        elif len(unique_roots) == 1:
            tie_class = TieClass.NESTED_SAME_PRIMITIVE_TIE
            rewritten = unique_roots[0]
        elif len(unique_explicit) >= 2 and not composite_ids:
            tie_class = TieClass.DISTINCT_PRIMITIVE_TIE
            rewritten = "DISTINCT_PRIMITIVE_TIE"
        elif unique_explicit and composite_ids:
            tie_class = TieClass.PRIMITIVE_COMPOSITE_TIE
            rewritten = "DISTINCT_PRIMITIVE_TIE" if len(unique_roots) >= 2 else unique_roots[0]
        elif len(composite_ids) >= 2:
            tie_class = TieClass.DISTINCT_COMPOSITE_TIE
            rewritten = "DISTINCT_COMPOSITE_TIE"
        else:
            tie_class = TieClass.UNKNOWN_TIE
            rewritten = "UNAVAILABLE_STRUCTURAL_GAP"

        classification_rows.append(
            {
                "decision_id": decision_id,
                "tie_class": tie_class,
                "tied_semantic_ids": semantic_ids,
                "tied_primitive_ids": unique_roots,
                "composite_ids": sorted(set(composite_ids)),
                "maximum_repeat_depth": maximum_repeat_depth,
                "rewritten_primary_label": rewritten,
                "resolved_to_one_primitive": rewritten.startswith("loop_p_"),
                **safety_flags(),
            }
        )

    classification = pd.DataFrame(classification_rows)
    if classification.empty:
        classification = pd.DataFrame(
            columns=[
                "decision_id",
                "tie_class",
                "tied_semantic_ids",
                "tied_primitive_ids",
                "composite_ids",
                "maximum_repeat_depth",
                "rewritten_primary_label",
                "resolved_to_one_primitive",
                *safety_flags(),
            ]
        )
    rewrite = classification[
        ["decision_id", "rewritten_primary_label", "tie_class", "resolved_to_one_primitive"]
    ].copy()
    summary_rows = []
    total = len(classification)
    for tie_class in TieClass:
        count = int(classification["tie_class"].eq(tie_class).sum())
        summary_rows.append(
            {
                "tie_class": tie_class,
                "count": count,
                "share_of_old_ties": count / total if total else 0.0,
                **safety_flags(),
            }
        )
    resolved = int(classification["resolved_to_one_primitive"].sum()) if total else 0
    summary_rows.append(
        {
            "tie_class": "TOTAL_AND_RESOLUTION",
            "count": total,
            "share_of_old_ties": resolved / total if total else 0.0,
            **safety_flags(),
        }
    )
    return TieResolutionBundle(
        classification=classification,
        nested_mapping=pd.DataFrame(mapping_rows),
        primary_rewrite=rewrite,
        summary=pd.DataFrame(summary_rows),
    )


__all__ = ["TieClass", "TieResolutionBundle", "resolve_registered_ties"]
