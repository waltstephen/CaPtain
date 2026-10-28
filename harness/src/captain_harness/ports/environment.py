from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, runtime_checkable

from .capability import CapabilityManifest, CapabilityResult


@dataclass(frozen=True, slots=True)
class ImageObservation:
    camera: str
    rgb_path: str
    depth_path: str | None = None
    role: str = "head"
    measurement_metadata: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class EnvironmentObservation:
    step: int
    images: tuple[ImageObservation, ...]
    public_state: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class HostCompletion:
    success_ever: bool | None
    final_success: bool | None
    first_success_step: int | None


@runtime_checkable
class EnvironmentPort(Protocol):
    def reset(self, seed: int) -> EnvironmentObservation: ...

    def observe(self) -> EnvironmentObservation: ...

    def capabilities(self) -> CapabilityManifest: ...

    def execute(
        self, capability: str, arguments: Mapping[str, object]
    ) -> CapabilityResult: ...

    def finalize(self) -> HostCompletion: ...


@runtime_checkable
class PublicEvidenceHistoryPort(Protocol):
    """Optional host interface; opaque sources resolve only stored public images."""
    def public_evidence_observations(self, source_ids: tuple[str, ...]) -> Mapping[str, EnvironmentObservation]: ...


@runtime_checkable
class CompletionGatePort(Protocol):
    """Optional host interface: check a completion declaration against the agent's own recorded public
    measurements. Returns None to pass, or a rejection mapping (failure, message) to hand back to the agent.
    Never exposes hidden state."""
    def completion_gate(self) -> Mapping[str, object] | None: ...


@runtime_checkable
class PublicExecutionRecoveryPort(Protocol):
    """Optional, read-only comparison of a prior request against a later public receipt.

    Only local execution recovery may be established; no task predicate, new
    observation, or hidden state is available through this interface.
    """
    def execution_recovery_evidence(
        self, *, failed_capability: str, failed_arguments: Mapping[str, object],
        capability: str, arguments: Mapping[str, object], result: CapabilityResult,
    ) -> Mapping[str, object] | None: ...
