from __future__ import annotations

import dataclasses
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from jsonschema import Draft202012Validator, ValidationError

from captain_harness.ports.capability import (
    CapabilityManifest,
    CapabilityResult,
    FailureCategory,
    UnknownCapabilityError,
)
from captain_harness.ports.environment import (
    EnvironmentObservation,
    HostCompletion,
    ImageObservation,
)

from .backend import LiberoActionBackend, reset_to_seeded_init_state
from .calibration import LiberoCalibrationProfile, load_calibration_profile
from .manifest import build_libero_manifest
from .scoring import HostOnlyFinalScorer


def _json_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"public capability value {type(value).__name__} is not JSON serializable")


class LiberoEnvironmentAdapter:
    """Expose one reset LIBERO environment through the generic environment port."""

    def __init__(
        self,
        env: Any,
        *,
        init_states: Sequence[Any],
        fresh_init_states: Any = None,
        observation_dir: Path,
        horizon: int,
        final_predicate: Callable[[], bool],
        backend_options: Mapping[str, object] | None = None,
        calibration_profile: LiberoCalibrationProfile | Path | str | None = None,
    ) -> None:
        self.__env = env
        self.__final_predicate = final_predicate
        self._init_states = init_states
        self._fresh_init_states = fresh_init_states
        self._observation_dir = observation_dir
        self._horizon = horizon
        self._backend_options = dict(backend_options or {})
        self._calibration_profile = (
            calibration_profile
            if isinstance(calibration_profile, LiberoCalibrationProfile)
            else load_calibration_profile(calibration_profile)
        )
        self._backend: LiberoActionBackend | None = None
        self._manifest: CapabilityManifest | None = None
        self._scorer: HostOnlyFinalScorer | None = None
        self._seed: int | None = None
        self._init_state_index: int | None = None

    def reset(self, seed: int) -> EnvironmentObservation:
        reset = reset_to_seeded_init_state(
            self.__env, self._init_states, seed=seed, fresh_init_states=self._fresh_init_states)
        joint_count = len(reset.observation["robot0_joint_pos"])
        self._backend = LiberoActionBackend(
            self.__env,
            observation_dir=self._observation_dir,
            initial_observation=reset.observation,
            horizon=self._horizon,
            calibration_profile=self._calibration_profile,
            **self._backend_options,
        )
        self._manifest = build_libero_manifest(joint_count=joint_count)
        self._scorer = HostOnlyFinalScorer(self.__final_predicate)
        self._seed = seed
        self._init_state_index = reset.init_state_index
        return self.observe()

    def _require_backend(self) -> LiberoActionBackend:
        if self._backend is None:
            raise RuntimeError("LIBERO adapter must be reset before use")
        return self._backend

    def capabilities(self) -> CapabilityManifest:
        if self._manifest is None:
            raise RuntimeError("LIBERO adapter must be reset before capabilities are available")
        return self._manifest

    def observe(self) -> EnvironmentObservation:
        backend = self._require_backend()
        result = backend.get_observation_data()
        if not result.success or result.value is None:
            raise RuntimeError(f"LIBERO observation failed: {result.message}")
        images = tuple(
            ImageObservation(
                camera.camera_name,
                camera.rgb_path,
                camera.depth_path,
                "head" if index == 0 else "wrist",
            )
            for index, camera in enumerate(result.value.cameras)
        )
        return EnvironmentObservation(
            step=backend.simulation_steps,
            images=images,
            public_state={
                "seed": self._seed,
                "init_state_index": self._init_state_index,
                "robot_state": _json_value(result.value.robot_state),
            },
        )

    def execute(
        self, capability: str, arguments: Mapping[str, object]
    ) -> CapabilityResult:
        backend = self._require_backend()
        manifest = self.capabilities()
        try:
            spec = manifest.require(capability)
        except UnknownCapabilityError as exc:
            return CapabilityResult.failed(FailureCategory.NOT_EXECUTED, str(exc))
        try:
            Draft202012Validator(spec.arguments_schema).validate(dict(arguments))
        except ValidationError as exc:
            return CapabilityResult.failed(
                FailureCategory.INVALID_ARGUMENT,
                f"arguments schema validation failed: {exc.message}",
            )
        try:
            values = [arguments[name] for name in spec.argument_order]
            result = getattr(backend, capability)(*values)
        except Exception as exc:
            return CapabilityResult.failed(
                FailureCategory.BACKEND_ERROR,
                f"{type(exc).__name__}: {exc}",
                executed=True,
            )
        if not isinstance(result, CapabilityResult):
            return CapabilityResult.failed(
                FailureCategory.BACKEND_ERROR,
                "LIBERO backend did not return CapabilityResult",
                executed=True,
            )
        if not result.success:
            return result
        try:
            public_value = _json_value(result.value)
            Draft202012Validator(spec.result_schema).validate(public_value)
        except (TypeError, ValidationError) as exc:
            message = exc.message if isinstance(exc, ValidationError) else str(exc)
            from captain_harness.ports.result_validation import result_validation_failure
            return result_validation_failure(result, message)
        return dataclasses.replace(result, value=public_value)

    def audit_hidden_success(self) -> bool:
        if self._scorer is None:
            raise RuntimeError(
                "LIBERO adapter must be reset before hidden audit sampling"
            )
        return self._scorer.score_after_action(
            environment_step=self._require_backend().simulation_steps
        )

    def finalize(self) -> HostCompletion:
        if self._scorer is None:
            raise RuntimeError("LIBERO adapter must be reset before final scoring")
        final_success = self._scorer.score_after_loop(loop_ended=True)
        return HostCompletion(
            self._scorer.success_ever,
            final_success,
            self._scorer.first_success_step,
        )
