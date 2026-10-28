from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, runtime_checkable

from .capability import CapabilityManifest, CapabilityResult


@runtime_checkable
class CapabilityBroker(Protocol):
    def invoke(self, name: str, arguments: Mapping[str, object]) -> CapabilityResult: ...


@dataclass(frozen=True, slots=True)
class PolicyExecutionResult:
    success: bool
    failure: str
    message: str
    stdout: str
    stderr: str
    capability_results: tuple[CapabilityResult, ...]
    session_usable: bool = True


@dataclass(frozen=True, slots=True)
class PolicyWorkspaceSymbol:
    name: str
    kind: str
    signature: str | None = None
    type_name: str | None = None
    preview: str | None = None
    lifetime: str | None = None
    source_frame_ids: tuple[str, ...] = ()
    target_ref: str | None = None
    stale: bool = False
    stale_reason: str | None = None


@runtime_checkable
class PolicyExecutor(Protocol):
    def execute(
        self,
        policy: str,
        capabilities: CapabilityManifest,
        broker: CapabilityBroker,
    ) -> PolicyExecutionResult: ...


@runtime_checkable
class PolicySession(Protocol):
    def execute(self, policy: str) -> PolicyExecutionResult: ...

    def workspace_symbols(self) -> tuple[PolicyWorkspaceSymbol, ...]: ...

    def prune_workspace(self) -> tuple[PolicyWorkspaceSymbol, ...]: ...

    def advance_lifecycle(self, event: str) -> tuple[PolicyWorkspaceSymbol, ...]: ...

    def close(self) -> None: ...


@runtime_checkable
class SessionPolicyExecutor(Protocol):
    def open_episode(
        self,
        capabilities: CapabilityManifest,
        broker: CapabilityBroker,
    ) -> PolicySession: ...
