"""Frozen named and control families for T0 execution-realism research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal


@dataclass(frozen=True)
class FamilySpec:
    """One immutable named or control orientation."""

    family: str
    classification: Literal["named", "control"]
    loop_id: str
    cycle: str
    orientation: str
    current_state: int
    alternate_state: int
    role: str


FROZEN_FAMILIES: Final[dict[str, FamilySpec]] = {
    "cycle_04|state_4": FamilySpec(
        family="cycle_04|state_4",
        classification="named",
        loop_id="cycle_04",
        cycle="2->4->2",
        orientation="state_4",
        current_state=4,
        alternate_state=2,
        role="named_candidate",
    ),
    "cycle_04|state_2": FamilySpec(
        family="cycle_04|state_2",
        classification="control",
        loop_id="cycle_04",
        cycle="2->4->2",
        orientation="state_2",
        current_state=2,
        alternate_state=4,
        role="neutral_control",
    ),
    "cycle_07|state_5": FamilySpec(
        family="cycle_07|state_5",
        classification="named",
        loop_id="cycle_07",
        cycle="5->6->5",
        orientation="state_5",
        current_state=5,
        alternate_state=6,
        role="named_candidate",
    ),
    "cycle_07|state_6": FamilySpec(
        family="cycle_07|state_6",
        classification="control",
        loop_id="cycle_07",
        cycle="5->6->5",
        orientation="state_6",
        current_state=6,
        alternate_state=5,
        role="negative_control",
    ),
}

NAMED_FAMILIES: Final[tuple[str, ...]] = ("cycle_04|state_4", "cycle_07|state_5")
CONTROL_FAMILIES: Final[tuple[str, ...]] = ("cycle_04|state_2", "cycle_07|state_6")


def family_spec(family: str) -> FamilySpec:
    """Return a frozen family or fail closed; replacement is never allowed."""

    try:
        return FROZEN_FAMILIES[family]
    except KeyError as error:
        raise ValueError(f"family is not frozen: {family}") from error
