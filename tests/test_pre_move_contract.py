"""Shared PRE normalization boundary checks."""

from stocker_execution.stage5 import calculate_pre_move

STAGE6_PRE_MOVE_M_THRESHOLD = 0.475764059845861

def test_stage6_strategy_threshold_equality_does_not_qualify() -> None:
    result = calculate_pre_move(
        expected_absolute_return_15m=1.0,
        p0=1.0,
        raw_open_t0=1.0,
        raw_open_t0_minus_3m=1.0 - STAGE6_PRE_MOVE_M_THRESHOLD,
    )

    assert result.pre_move_m == STAGE6_PRE_MOVE_M_THRESHOLD
    assert not (result.pre_move_m > STAGE6_PRE_MOVE_M_THRESHOLD)


def test_generic_stage5_result_does_not_apply_strategy_qualification() -> None:
    result = calculate_pre_move(
        expected_absolute_return_15m=0.01,
        p0=100.0,
        raw_open_t0=100.0,
        raw_open_t0_minus_3m=99.0,
    )

    assert not hasattr(result, "passes_pre_move_threshold")
