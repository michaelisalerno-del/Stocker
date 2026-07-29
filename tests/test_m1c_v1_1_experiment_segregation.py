from __future__ import annotations

from datetime import date, timedelta

import pytest

from stocker_prospective.m1c_prospective_opening_reversal_v1 import (
    OptionContractCandidateV1,
    validate_primary_option_protocol_v1_1,
)

SESSION = date(2026, 7, 30)


def contract(
    *,
    con_id: int,
    right: str,
    strike: float = 100.0,
    expiry_offset: int = 1,
) -> OptionContractCandidateV1:
    return OptionContractCandidateV1(
        con_id=con_id,
        underlying="AAL",
        expiry=SESSION + timedelta(days=expiry_offset),
        strike=strike,
        right=right,
        multiplier=100,
        exchange="SMART",
        trading_class="AAL",
    )


def test_v1_1_accepts_only_primary_one_dte_common_strike_call_put_pair() -> None:
    call, put = validate_primary_option_protocol_v1_1(
        session=SESSION,
        contracts=(
            contract(con_id=101, right="C"),
            contract(con_id=102, right="P"),
        ),
    )

    assert call.right == "C"
    assert put.right == "P"
    assert call.expiry == SESSION + timedelta(days=1)
    assert call.strike == put.strike


@pytest.mark.parametrize(
    ("contracts", "error"),
    [
        (
            (contract(con_id=101, right="C"),),
            "requires_exactly_two_legs",
        ),
        (
            (
                contract(con_id=101, right="C", strike=99.0),
                contract(con_id=102, right="P", strike=99.0),
                contract(con_id=103, right="C", strike=101.0),
                contract(con_id=104, right="P", strike=101.0),
            ),
            "requires_exactly_two_legs",
        ),
        (
            (
                contract(con_id=101, right="C"),
                contract(con_id=102, right="C"),
            ),
            "rejects_condor_or_duplicate_roles",
        ),
        (
            (
                contract(con_id=101, right="C", strike=99.0),
                contract(con_id=102, right="P", strike=101.0),
            ),
            "requires_common_strike_pair",
        ),
        (
            (
                contract(con_id=101, right="C", expiry_offset=2),
                contract(con_id=102, right="P", expiry_offset=2),
            ),
            "primary_expiry_must_be_1dte",
        ),
    ],
)
def test_v1_1_rejects_secondary_dte_wrong_leg_count_condor_and_strike_mismatch(
    contracts: tuple[OptionContractCandidateV1, ...],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        validate_primary_option_protocol_v1_1(
            session=SESSION,
            contracts=contracts,
        )
