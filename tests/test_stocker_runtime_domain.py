import pytest
from pydantic import ValidationError

from stocker_runtime import Observation, ProposedPosition, ProposedTrade, Signal


def test_observation_serializes_to_canonical_json() -> None:
    observation = Observation(
        subject_instrument_id="AAPL@SMART:USD",
        as_of_at_us=1_786_032_000_000_000,
        payload={"z_score": 1.25, "label": "leader"},
    )

    encoded = observation.to_canonical_json()

    assert encoded == (
        b'{"as_of_at_us":1786032000000000,"kind":"observation",'
        b'"payload":{"label":"leader","z_score":1.25},'
        b'"subject_instrument_id":"AAPL@SMART:USD"}'
    )
    assert Observation.model_validate_json(encoded) == observation


@pytest.mark.parametrize(
    "output_type,kind",
    [
        (Observation, "observation"),
        (Signal, "signal"),
        (ProposedPosition, "proposed_position"),
        (ProposedTrade, "proposed_trade"),
    ],
)
def test_all_output_types_are_unapproved_evidence_with_strict_fields(
    output_type: type[Observation | Signal | ProposedPosition | ProposedTrade],
    kind: str,
) -> None:
    output = output_type(
        subject_instrument_id="AAPL@SMART:USD",
        as_of_at_us=1_786_032_000_000_000,
        payload={"reason": "causal evidence"},
    )

    assert output.kind == kind
    if isinstance(output, (ProposedPosition, ProposedTrade)):
        assert output.status == "unapproved"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        output_type.model_validate(
            {
                **output.model_dump(),
                "account_id": "forbidden",
            }
        )


def test_output_payload_is_bounded_by_canonical_utf8_size() -> None:
    accepted = Observation(
        subject_instrument_id="AAPL@SMART:USD",
        as_of_at_us=1,
        payload={"value": "x" * 16_372},
    )

    assert len(accepted.payload_json()) == 16 * 1024

    with pytest.raises(ValidationError, match="payload exceeds 16384 bytes"):
        Observation(
            subject_instrument_id="AAPL@SMART:USD",
            as_of_at_us=1,
            payload={"value": "x" * 16_373},
        )
