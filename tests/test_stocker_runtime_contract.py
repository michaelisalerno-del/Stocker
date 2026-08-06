import copy
import hashlib
import pickle
from collections.abc import Mapping
from typing import get_origin

import pytest
from pydantic import ValidationError

from stocker_runtime import (
    MAX_EVENTS_PER_BATCH,
    MAX_MARKET_EVENT_PAYLOAD_BYTES,
    IdeaActivation,
    IdeaBatch,
    IdeaEvaluation,
    IdeaManifest,
    IdeaPlugin,
    MarketDataRequirement,
    MarketEvent,
    Observation,
    OutputKind,
    ProposedPosition,
    ProposedTrade,
    ProtectedDataClass,
    RuntimeMode,
    Signal,
    canonical_json_bytes,
)


def _manifest_data() -> dict[str, object]:
    return {
        "api_version": 1,
        "idea_id": "opening_leader_continuation",
        "idea_version": "v0",
        "display_name": "Opening Leader Continuation",
        "description": "First-party prospective evidence plugin.",
        "modes": (RuntimeMode.PROSPECTIVE_RECORD, RuntimeMode.SHADOW),
        "output_kinds": (
            OutputKind.OBSERVATION,
            OutputKind.SIGNAL,
            OutputKind.PROPOSED_TRADE,
        ),
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
            IdeaManifest.model_validate({**_manifest_data(), "modes": (unsupported_mode,)})


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
        "approval-id",
        "ApprovalId",
        "isApproved",
        "IsApproved",
        "is-approved",
        "account_id",
        "accountId",
        "AccountId",
        "Account ID",
        "account.id",
        "broker_order_id",
        "broker-order-id",
        "brokerOrderId",
        "BrokerOrderId",
        "broker/order/id",
        "should_transmit",
        "should-transmit",
        "shouldTransmit",
        "ShouldTransmit",
        "runtime_mode",
        "runtime-mode",
        "runtimeMode",
        "RuntimeMode",
    ],
)
def test_proposals_cannot_carry_normalized_authority_fields(forbidden_field: str) -> None:
    for proposal_type in (ProposedPosition, ProposedTrade):
        with pytest.raises(ValidationError, match="forbidden authority field"):
            proposal_type(
                subject_instrument_id="AAPL",
                as_of_at_us=1,
                payload={"nested": {forbidden_field: "forbidden"}},
            )


def test_non_proposal_json_allows_legitimate_evidence_and_configuration_vocabulary() -> None:
    evidence = {
        "order_book_observation": {"imbalance": 0.25},
        "risk_score": 0.1,
        "strategy_mode": "momentum",
    }
    parameters_hash = hashlib.sha256(canonical_json_bytes(evidence)).hexdigest()
    manifest_data = _manifest_data()
    manifest_data["parameter_schema"] = {
        "type": "object",
        "properties": {
            "order_book_observation": {"type": "object"},
            "risk_score": {"type": "number"},
            "strategy_mode": {"type": "string"},
        },
    }

    models = (
        Observation(subject_instrument_id="AAPL", as_of_at_us=1, payload=evidence),
        Signal(subject_instrument_id="AAPL", as_of_at_us=1, payload=evidence),
        MarketEvent(
            event_id="event-001",
            instrument_id="AAPL",
            feed_kind="depth",
            event_kind="order_book",
            event_at_us=1,
            received_at_us=2,
            payload=evidence,
        ),
        IdeaManifest.model_validate(manifest_data),
        IdeaActivation(
            instance_id="instance-001",
            parameters=evidence,
            parameters_hash=parameters_hash,
            plugin_code_hash="b" * 64,
            activated_at_us=1,
            run_id="run-001",
            protected_data_class=ProtectedDataClass.PROSPECTIVE,
            universe=("AAPL",),
        ),
        IdeaEvaluation(state=evidence, outputs=()),
    )

    assert all(model.to_canonical_json() for model in models)


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


def test_public_dto_json_is_deeply_immutable_and_round_trips() -> None:
    parameters = {"rules": [{"threshold": 1.25}]}
    parameters_hash = hashlib.sha256(canonical_json_bytes(parameters)).hexdigest()
    activation = IdeaActivation(
        instance_id="instance-001",
        parameters=parameters,
        parameters_hash=parameters_hash,
        plugin_code_hash="b" * 64,
        activated_at_us=1,
        run_id="run-001",
        protected_data_class=ProtectedDataClass.PROSPECTIVE,
        universe=("AAPL",),
    )
    manifest_data = _manifest_data()
    manifest_data["parameter_schema"] = {
        "type": "object",
        "properties": {"threshold": {"type": "number", "examples": [1.25]}},
    }
    models_and_json_fields = (
        (
            Observation(
                subject_instrument_id="AAPL",
                as_of_at_us=1,
                payload={"nested": {"value": 1}, "items": ["first"]},
            ),
            "payload",
            ("nested",),
            "value",
            ("items",),
        ),
        (
            ProposedTrade(
                subject_instrument_id="AAPL",
                as_of_at_us=1,
                payload={"nested": {"value": 1}, "items": ["first"]},
            ),
            "payload",
            ("nested",),
            "value",
            ("items",),
        ),
        (
            MarketEvent(
                event_id="event-001",
                instrument_id="AAPL",
                feed_kind="trades",
                event_kind="trade",
                event_at_us=1,
                received_at_us=2,
                payload={"nested": {"value": 1}, "items": ["first"]},
            ),
            "payload",
            ("nested",),
            "value",
            ("items",),
        ),
        (activation, "parameters", ("rules", 0), "threshold", ("rules",)),
        (
            IdeaManifest.model_validate(manifest_data),
            "parameter_schema",
            ("properties", "threshold"),
            "type",
            ("properties", "threshold", "examples"),
        ),
        (
            IdeaEvaluation(
                state={"nested": {"value": 1}, "items": ["first"]},
                outputs=(
                    ProposedTrade(
                        subject_instrument_id="AAPL",
                        as_of_at_us=1,
                        payload={"reason": "causal evidence"},
                    ),
                ),
            ),
            "state",
            ("nested",),
            "value",
            ("items",),
        ),
    )

    activation_before_input_mutation = activation.to_canonical_json()
    parameters["rules"][0]["threshold"] = 9.0
    assert activation.to_canonical_json() == activation_before_input_mutation
    assert activation.parameters_hash == parameters_hash

    for model, field_name, object_path, object_key, list_path in models_and_json_fields:
        canonical_before = model.to_canonical_json()
        json_value = getattr(model, field_name)
        nested_object = json_value
        for path_part in object_path:
            nested_object = nested_object[path_part]
        nested_list = json_value
        for path_part in list_path:
            nested_list = nested_list[path_part]

        with pytest.raises(TypeError):
            nested_object[object_key] = "changed"
        with pytest.raises(TypeError):
            nested_list[0] = "changed"
        with pytest.raises(TypeError):
            dict.__setitem__(json_value, "broker_order_id", "forbidden")

        assert model.to_canonical_json() == canonical_before
        assert type(model).model_validate_json(canonical_before) == model
        assert type(model).model_validate(model.model_dump(mode="python")) == model

    assert activation.parameters_hash == parameters_hash


def test_public_json_contract_is_read_only_and_survives_deep_copy() -> None:
    observation = Observation(
        subject_instrument_id="AAPL",
        as_of_at_us=1,
        payload={"nested": {"values": [1, 2]}},
    )

    for model_type, field_name in (
        (Observation, "payload"),
        (MarketEvent, "payload"),
        (IdeaManifest, "parameter_schema"),
        (IdeaActivation, "parameters"),
    ):
        assert get_origin(model_type.model_fields[field_name].annotation) is Mapping
    assert isinstance(observation.payload, Mapping)
    assert not isinstance(observation.payload, dict)
    assert isinstance(observation.payload["nested"], Mapping)
    assert observation.payload["nested"]["values"] == (1, 2)

    for copied in (
        copy.deepcopy(observation),
        observation.model_copy(deep=True),
        pickle.loads(pickle.dumps(observation)),
    ):
        assert copied == observation
        with pytest.raises(TypeError):
            copied.payload["nested"]["values"][0] = 9
        with pytest.raises(TypeError):
            copied.payload["nested"]["extra"] = True

    assert Observation.model_validate_json(observation.to_canonical_json()) == observation


def test_immutable_dto_copy_rejects_every_non_empty_update() -> None:
    event = MarketEvent(
        event_id="event-001",
        instrument_id="AAPL",
        feed_kind="trades",
        event_kind="trade",
        event_at_us=1,
        received_at_us=2,
        payload={},
    )
    batch = IdeaBatch(
        mode=RuntimeMode.PROSPECTIVE_RECORD,
        events=(event,),
        input_watermark="event-001",
        causal_from_at_us=1,
        causal_through_at_us=1,
    )
    proposal = ProposedTrade(
        subject_instrument_id="AAPL",
        as_of_at_us=1,
        payload={"reason": "causal evidence"},
    )

    adversarial_updates = (
        (batch, {"mode": "live"}),
        (proposal, {"status": "approved"}),
        (proposal, {"payload": {"brokerOrderId": "forbidden"}}),
        (proposal, {"payload": {"value": "x" * (16 * 1024)}}),
        (proposal, {"payload": {"nested": []}}),
    )

    for model, update in adversarial_updates:
        with pytest.raises(TypeError, match="immutable DTO copies cannot be updated"):
            model.model_copy(update=update)

    with pytest.raises(TypeError, match="immutable DTO copies cannot be updated"):
        proposal.copy(update={"status": "approved"})

    assert proposal.model_copy(update={}) == proposal
    assert proposal.model_copy(deep=True) == proposal


def test_immutable_json_object_serialization_does_constant_work_per_lookup() -> None:
    from stocker_runtime.domain import FrozenJsonObject

    class CountingKey(str):
        comparisons = 0

        def __eq__(self, other: object) -> bool:
            type(self).comparisons += 1
            return super().__eq__(other)

        __hash__ = str.__hash__

    value = FrozenJsonObject({CountingKey(f"k{index:03}"): index for index in range(128)})
    CountingKey.comparisons = 0

    encoded = canonical_json_bytes(value)

    assert encoded.startswith(b'{"k000":0')
    assert CountingKey.comparisons <= len(value)


def test_market_event_serializes_a_valid_five_thousand_key_payload() -> None:
    event = MarketEvent(
        event_id="event-001",
        instrument_id="AAPL",
        feed_kind="trades",
        event_kind="trade",
        event_at_us=1,
        received_at_us=2,
        payload={f"{index:04x}": None for index in range(5_000)},
    )

    encoded = event.payload_json()

    assert len(encoded) == 60_001
    assert encoded.startswith(b'{"0000":null')
    assert encoded.endswith(b'"1387":null}')
    assert MarketEvent.model_validate_json(event.to_canonical_json()) == event


@pytest.mark.parametrize("invalid_integer", [True, "1", 1.0])
@pytest.mark.parametrize(
    "model_type,valid_data,field_name",
    [
        (
            MarketEvent,
            {
                "event_id": "event-001",
                "instrument_id": "AAPL",
                "feed_kind": "trades",
                "event_kind": "trade",
                "event_at_us": 1,
                "received_at_us": 2,
                "payload": {},
            },
            "event_at_us",
        ),
        (
            Observation,
            {"subject_instrument_id": "AAPL", "as_of_at_us": 1, "payload": {}},
            "as_of_at_us",
        ),
        (
            IdeaActivation,
            {
                "instance_id": "instance-001",
                "parameters": {},
                "parameters_hash": hashlib.sha256(b"{}").hexdigest(),
                "plugin_code_hash": "b" * 64,
                "activated_at_us": 1,
                "run_id": "run-001",
                "protected_data_class": ProtectedDataClass.PROSPECTIVE,
                "universe": ("AAPL",),
            },
            "activated_at_us",
        ),
        (IdeaManifest, _manifest_data(), "maximum_outputs_per_batch"),
    ],
)
def test_integer_timestamps_and_limits_reject_coercion(
    model_type: type[MarketEvent | Observation | IdeaActivation | IdeaManifest],
    valid_data: dict[str, object],
    field_name: str,
    invalid_integer: object,
) -> None:
    with pytest.raises(ValidationError):
        model_type.model_validate({**valid_data, field_name: invalid_integer})


def test_market_event_payload_and_idea_batch_have_explicit_input_bounds() -> None:
    assert MAX_MARKET_EVENT_PAYLOAD_BYTES == 64 * 1024
    assert MAX_EVENTS_PER_BATCH == 256

    event = MarketEvent(
        event_id="event-001",
        instrument_id="AAPL",
        feed_kind="trades",
        event_kind="trade",
        event_at_us=1,
        received_at_us=2,
        payload={"value": "x" * 65_524},
    )
    batch = IdeaBatch(
        mode=RuntimeMode.PROSPECTIVE_RECORD,
        events=(event,) * MAX_EVENTS_PER_BATCH,
        input_watermark="event-001",
        causal_from_at_us=1,
        causal_through_at_us=1,
    )

    assert len(event.payload_json()) == MAX_MARKET_EVENT_PAYLOAD_BYTES
    assert len(batch.events) == MAX_EVENTS_PER_BATCH

    with pytest.raises(ValidationError, match="market event payload exceeds 65536 bytes"):
        MarketEvent(
            event_id="event-002",
            instrument_id="AAPL",
            feed_kind="trades",
            event_kind="trade",
            event_at_us=1,
            received_at_us=2,
            payload={"value": "x" * 65_525},
        )

    with pytest.raises(ValidationError):
        IdeaBatch(
            mode=RuntimeMode.PROSPECTIVE_RECORD,
            events=(event,) * (MAX_EVENTS_PER_BATCH + 1),
            input_watermark="event-001",
            causal_from_at_us=1,
            causal_through_at_us=1,
        )
