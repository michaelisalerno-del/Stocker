import hashlib

import pytest
from pydantic import ValidationError

from stocker_runtime import (
    IdeaActivation,
    IdeaBatch,
    IdeaEvaluation,
    IdeaManifest,
    IdeaPlugin,
    MarketDataRequirement,
    MarketEvent,
    Observation,
    OutputKind,
    ProtectedDataClass,
    RuntimeMode,
    canonical_json_bytes,
)


def _manifest_data() -> dict[str, object]:
    return {
        "api_version": 1,
        "idea_id": "opening_leader_continuation",
        "idea_version": "v0",
        "display_name": "Opening Leader Continuation",
        "description": "First-party prospective evidence plugin.",
        "modes": ["prospective_record", "shadow"],
        "output_kinds": ["observation", "signal", "proposed_trade"],
        "parameter_schema_version": "v1",
        "parameter_schema": {"type": "object", "additionalProperties": False},
        "maximum_state_bytes": 65_536,
        "maximum_outputs_per_batch": 256,
    }


def test_manifest_accepts_only_immediate_runtime_modes() -> None:
    manifest = IdeaManifest.model_validate(_manifest_data())

    assert manifest.modes == (RuntimeMode.PROSPECTIVE_RECORD, RuntimeMode.SHADOW)
    assert manifest.output_kinds == (
        OutputKind.OBSERVATION,
        OutputKind.SIGNAL,
        OutputKind.PROPOSED_TRADE,
    )

    for unsupported_mode in ("paper", "live", "research", "unknown"):
        with pytest.raises(ValidationError):
            IdeaManifest.model_validate({**_manifest_data(), "modes": [unsupported_mode]})


def test_first_party_plugin_protocol_uses_bounded_authority_free_dtos() -> None:
    manifest = IdeaManifest.model_validate(_manifest_data())
    parameters = {"minimum_rank": 3}
    activation = IdeaActivation(
        instance_id="instance-001",
        parameters=parameters,
        parameters_hash=hashlib.sha256(canonical_json_bytes(parameters)).hexdigest(),
        plugin_code_hash="b" * 64,
        activated_at_us=1_786_032_000_000_000,
        run_id="run-001",
        protected_data_class=ProtectedDataClass.PROSPECTIVE,
        universe=("AAPL@SMART:USD",),
    )
    requirement = MarketDataRequirement(
        feed_kind="trades",
        instrument_id="AAPL@SMART:USD",
        cadence="tick",
        gaps_block=True,
        staleness_block=True,
    )
    batch = IdeaBatch(
        mode=RuntimeMode.PROSPECTIVE_RECORD,
        events=(
            MarketEvent(
                event_id="event-001",
                instrument_id="AAPL@SMART:USD",
                feed_kind="trades",
                event_kind="trade",
                event_at_us=1_786_032_000_000_000,
                received_at_us=1_786_032_000_001_000,
                payload={"price": 225.5, "size": 100},
            ),
        ),
        input_watermark="event-001",
        causal_from_at_us=1_786_032_000_000_000,
        causal_through_at_us=1_786_032_000_000_000,
    )
    evaluation = IdeaEvaluation(
        state={"last_event_id": "event-001"},
        outputs=(
            Observation(
                subject_instrument_id="AAPL@SMART:USD",
                as_of_at_us=1_786_032_000_000_000,
                payload={"rank": 1},
            ),
        ),
    )

    class ExamplePlugin:
        @property
        def manifest(self) -> IdeaManifest:
            return manifest

        def requirements(
            self, requested_activation: IdeaActivation
        ) -> tuple[MarketDataRequirement, ...]:
            assert requested_activation == activation
            return (requirement,)

        def evaluate(self, requested_batch: IdeaBatch, state: object) -> IdeaEvaluation:
            assert requested_batch == batch
            assert state == {}
            return evaluation

    plugin = ExamplePlugin()

    assert isinstance(plugin, IdeaPlugin)
    assert plugin.requirements(activation) == (requirement,)
    assert plugin.evaluate(batch, {}) == evaluation


def test_evaluation_enforces_global_state_and_output_bounds() -> None:
    accepted = IdeaEvaluation(state={"value": "x" * 65_524}, outputs=())

    assert len(accepted.state_json()) == 64 * 1024

    with pytest.raises(ValidationError, match="state exceeds 65536 bytes"):
        IdeaEvaluation(state={"value": "x" * 65_525}, outputs=())

    output = Observation(subject_instrument_id="AAPL", as_of_at_us=1, payload={})
    assert len(IdeaEvaluation(state=None, outputs=(output,) * 256).outputs) == 256

    with pytest.raises(ValidationError):
        IdeaEvaluation(state=None, outputs=(output,) * 257)


@pytest.mark.parametrize("unsupported_mode", ["paper", "live", "research", "unknown"])
def test_batch_rejects_unsupported_modes(unsupported_mode: str) -> None:
    with pytest.raises(ValidationError):
        IdeaBatch(
            mode=unsupported_mode,
            events=(
                MarketEvent(
                    event_id="event-001",
                    instrument_id="AAPL",
                    feed_kind="trades",
                    event_kind="trade",
                    event_at_us=1,
                    received_at_us=2,
                    payload={},
                ),
            ),
            input_watermark="event-001",
            causal_from_at_us=1,
            causal_through_at_us=1,
        )


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "approval_id",
        "account_id",
        "broker_order_id",
        "transmit",
        "mode",
        "risk_decision",
        "execution_id",
    ],
)
def test_plugin_json_cannot_carry_authority_fields(forbidden_field: str) -> None:
    with pytest.raises(ValidationError, match="forbidden authority field"):
        IdeaEvaluation(
            state={"nested": {forbidden_field: "forbidden"}},
            outputs=(),
        )

    with pytest.raises(ValidationError, match="forbidden authority field"):
        Observation(
            subject_instrument_id="AAPL",
            as_of_at_us=1,
            payload={"nested": {forbidden_field: "forbidden"}},
        )


def test_contract_json_rejects_non_finite_numbers_during_validation() -> None:
    with pytest.raises(ValidationError, match="finite number"):
        IdeaActivation(
            instance_id="instance-001",
            parameters={"threshold": float("nan")},
            parameters_hash="a" * 64,
            plugin_code_hash="b" * 64,
            activated_at_us=1,
            run_id="run-001",
            protected_data_class=ProtectedDataClass.PROSPECTIVE,
            universe=("AAPL",),
        )


def test_activation_rejects_a_parameter_hash_that_does_not_match_canonical_json() -> None:
    with pytest.raises(ValidationError, match="parameters_hash does not match"):
        IdeaActivation(
            instance_id="instance-001",
            parameters={"threshold": 1.25},
            parameters_hash="a" * 64,
            plugin_code_hash="b" * 64,
            activated_at_us=1,
            run_id="run-001",
            protected_data_class=ProtectedDataClass.PROSPECTIVE,
            universe=("AAPL",),
        )
