from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pytest

from captain_harness.ports.capability import (
    CapabilityManifest,
    CapabilityResult,
    CapabilitySpec,
    FailureCategory,
    SideEffect,
    UnknownCapabilityError,
)
from captain_harness.ports.environment import (
    EnvironmentObservation,
    EnvironmentPort,
    HostCompletion,
    ImageObservation,
)
from captain_harness.ports.model import ModelBackend, ModelResponse
from captain_harness.ports.policy import PolicyExecutionResult, PolicyExecutor


def _echo_spec(name: str = "echo_runtime_value") -> CapabilitySpec:
    return CapabilitySpec(
        name=name,
        description="Return a value supplied by generated policy code.",
        arguments_schema={
            "type": "object",
            "properties": {"value": {"type": "number"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        result_schema={"type": "number"},
        units={"value": "dimensionless", "result": "dimensionless"},
        coordinate_frames={},
        timeout_s=0.5,
        side_effect=SideEffect.READ_ONLY,
    )


def test_manifest_is_versioned_schema_validated_and_runtime_defined() -> None:
    spec = _echo_spec()
    manifest = CapabilityManifest(
        schema_version="captain-capabilities/v1",
        provider="fake-environment/v3",
        capabilities=(spec,),
    )
    assert manifest.names == ("echo_runtime_value",)
    assert manifest.require("echo_runtime_value") is spec
    with pytest.raises(UnknownCapabilityError, match="not declared"):
        manifest.require("move_to_pose")
    with pytest.raises(ValueError, match="duplicate"):
        CapabilityManifest(
            schema_version="captain-capabilities/v1",
            provider="fake",
            capabilities=(spec, spec),
        )
    with pytest.raises(ValueError, match="arguments_schema"):
        CapabilitySpec(
            name="broken",
            description="invalid schema",
            arguments_schema={"type": "definitely-not-json-schema"},
            result_schema={},
            units={},
            coordinate_frames={},
            timeout_s=1.0,
            side_effect=SideEffect.READ_ONLY,
        )


@pytest.mark.parametrize(
    ("success", "executed", "failure_category", "message"),
    [
        (True, False, FailureCategory.NONE, "executed"),
        (True, True, FailureCategory.TIMEOUT, "failure_category"),
        (False, True, FailureCategory.NOT_EXECUTED, "not_executed"),
    ],
)
def test_capability_result_rejects_contradictory_states(
    success: bool,
    executed: bool,
    failure_category: FailureCategory,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        CapabilityResult(success, executed, failure_category, "")


@dataclass
class FakeEnvironment:
    manifest: CapabilityManifest

    def reset(self, seed: int) -> EnvironmentObservation:
        return self.observe()

    def observe(self) -> EnvironmentObservation:
        return EnvironmentObservation(
            step=0,
            images=(ImageObservation("head", "/tmp/head.png"),),
            public_state={"ready": True},
        )

    def capabilities(self) -> CapabilityManifest:
        return self.manifest

    def execute(self, capability: str, arguments: Mapping[str, object]) -> CapabilityResult:
        return CapabilityResult.ok(arguments["value"])

    def host_completion(self) -> HostCompletion:
        return HostCompletion(False, None, None)

    def finalize(self) -> HostCompletion:
        return HostCompletion(False, False, None)


class FakeModel:
    def respond(self, request, *, previous_response_id=None) -> ModelResponse:
        return ModelResponse("response-1", "FINISH", {})

    def compact(self, request) -> ModelResponse:
        return ModelResponse("response-2", "compacted", {})


class FakePolicy:
    def execute(self, policy: str, capabilities, broker) -> PolicyExecutionResult:
        return PolicyExecutionResult(True, "none", "", "", (), ())


def test_ports_are_runtime_checkable_without_robot_dependencies() -> None:
    environment = FakeEnvironment(
        CapabilityManifest("captain-capabilities/v1", "fake", (_echo_spec(),))
    )
    assert isinstance(environment, EnvironmentPort)
    assert isinstance(FakeModel(), ModelBackend)
    assert isinstance(FakePolicy(), PolicyExecutor)
