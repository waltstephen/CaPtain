from __future__ import annotations
import json

import dataclasses
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from captain_harness.adapters.libero.backend import (
    LiberoActionBackend,
    reset_to_seeded_init_state,
)
from captain_harness.adapters.libero.calibration import (
    LiberoCalibrationProfile,
    load_calibration_profile,
)
from captain_harness.adapters.libero.scoring import HostOnlyFinalScorer
from captain_harness.adapters.libero.recorded_backend import ClosingMonitoredPossessionActionBackend
from captain_harness.ports.broker import ValidatedCapabilityBroker
from captain_harness.core.experience import ExperienceSnapshot
from captain_harness.ports.capability import CapabilityManifest, CapabilityResult, SideEffect, ControlSemantic
from captain_harness.ports.environment import (
    EnvironmentObservation,
    HostCompletion,
    ImageObservation,
)

from .artifacts import EpisodeArtifactRegistry
from .backend import ExternalToolServices, ToolCapabilityBackend
from .manifest import ToolApiLevel, build_capx_tool_manifest
from .runtime_manifest import runtime_manifest
from .protocol import (
    PublicCompletionProtocol,
    load_public_completion_protocol,
)


class CapXLiberoEnvironmentAdapter:
    """Expose LIBERO through sensor-only S2 or handle-based S3 CaP-X tools."""

    def __init__(
        self,
        env: Any,
        *,
        init_states: Sequence[Any],
        fresh_init_states: Any = None,
        observation_dir: Path,
        horizon: int,
        final_predicate: Callable[[], bool],
        services: ExternalToolServices,
        api_level: ToolApiLevel,
        backend_options: Mapping[str, object] | None = None,
        tool_backend_options: Mapping[str, object] | None = None,
        artifact_ttl_s: float | None = None,
        public_safe_release: bool = False,
        calibration_profile: LiberoCalibrationProfile | Path | str | None = None,
        service_aligned: bool = False,
        dehardcoded: bool = False,
        v15_protocol: bool = False,
        completion_protocol: (
            PublicCompletionProtocol | Path | str | None
        ) = None,
        experience_snapshot: ExperienceSnapshot | None = None,
        motion_tools: bool = False,
        coding_control: bool = False,
        completion_goals: Sequence[object] | None = None,
        completion_gate: bool = False,
    ) -> None:
        from .completion_gate import goals_from_bddl
        self._completion_goals = goals_from_bddl(completion_goals) if completion_goals else []
        self._completion_gate_enabled = bool(completion_gate)
        self._reset_gate_state()
        self.__env = env
        self.__final_predicate = final_predicate
        self._init_states = init_states
        self._fresh_init_states = fresh_init_states
        self._observation_dir = observation_dir
        self._horizon = horizon
        self._services = services
        self._api_level = ToolApiLevel(api_level)
        self._backend_options = dict(backend_options or {})
        self._tool_backend_options = dict(tool_backend_options or {})
        # Slow model calls must not erase identity anchors within a live episode.
        # reset/finalize revoke every handle; pose/scene freshness stays separate.
        self._artifact_ttl_s = artifact_ttl_s
        self._public_safe_release = bool(public_safe_release)
        self._service_aligned = bool(service_aligned)
        self._dehardcoded = bool(dehardcoded)
        self._v15_protocol = bool(v15_protocol)
        self._coding_control = bool(coding_control)
        self._motion_tools = bool(motion_tools or coding_control)
        if self._motion_tools and not (service_aligned and dehardcoded and v15_protocol):
            raise ValueError('motion tools require service_aligned, dehardcoded and v15_protocol')
        self._calibration_profile = (
            calibration_profile
            if isinstance(calibration_profile, LiberoCalibrationProfile)
            else load_calibration_profile(calibration_profile)
        )
        self._completion_protocol = (
            completion_protocol
            if isinstance(
                completion_protocol,
                PublicCompletionProtocol,
            )
            else load_public_completion_protocol(completion_protocol)
        )
        self._experience_snapshot = experience_snapshot
        self._robot: LiberoActionBackend | None = None
        self._tools: ToolCapabilityBackend | None = None
        self._registry: EpisodeArtifactRegistry | None = None
        self._manifest: CapabilityManifest | None = None
        self._scorer: HostOnlyFinalScorer | None = None
        self._seed: int | None = None
        self._init_state_index: int | None = None

    def reset(self, seed: int) -> EnvironmentObservation:
        if self._registry is not None:
            self._registry.close()
        from .attempt_evidence import AttemptEvidenceBinding
        self._attempt_evidence_binding = AttemptEvidenceBinding()
        reset = reset_to_seeded_init_state(
            self.__env, self._init_states, seed=seed, fresh_init_states=self._fresh_init_states)
        robot_backend = ClosingMonitoredPossessionActionBackend if self._motion_tools else LiberoActionBackend
        backend_options = dict(self._backend_options)
        contact_detector = str(backend_options.pop('contact_detector', 'legacy'))
        contact_calibration = bool(backend_options.pop('contact_calibration', False))
        self._robot = robot_backend(
            self.__env,
            observation_dir=self._observation_dir,
            initial_observation=reset.observation,
            horizon=self._horizon,
            calibration_profile=self._calibration_profile,
            **{'guarded_cartesian_motion': self._motion_tools, **backend_options},
        )
        self._robot._contact_detector_mode = contact_detector
        self._contact_calibration_receipt = None
        self._reset_gate_state()
        if contact_detector == 'residual' and contact_calibration:
            # v5: per-episode free-space force calibration before the first public observation.
            calibration, receipt = self._robot.calibrate_contact_residual()
            self._contact_calibration_receipt = {'calibration': calibration.to_json(), **receipt}
            Path(self._observation_dir).parent.mkdir(parents=True, exist_ok=True)
            (Path(self._observation_dir).parent / 'force_calibration.json').write_text(
                json.dumps(self._contact_calibration_receipt, indent=2))
        episode_id = f"libero-{self._api_level.value.lower()}-{seed}-{uuid.uuid4().hex}"
        self._registry = EpisodeArtifactRegistry(
            episode_id, ttl_s=self._artifact_ttl_s,
            storage_root=self._observation_dir.parent / "public_sensor_artifacts" / episode_id
        )
        self._tools = ToolCapabilityBackend(
            robot=self._robot,
            services=self._services,
            registry=self._registry,
            tcp_offset_local_m=self._calibration_profile.tcp_offset_local_m,
            gripper_profile=self._calibration_profile.gripper,
            default_approach_frame=(
                "world" if self._service_aligned else "tool"
            ),
            completion_protocol=self._completion_protocol,
            dehardcoded=self._dehardcoded,
            v15_protocol=self._v15_protocol,
            motion_tools=self._motion_tools,
            experience_snapshot=self._experience_snapshot,
            graspnet_to_eef_transform=(
                self._calibration_profile.graspnet_to_eef_transform
            ),
            **self._tool_backend_options,
        )
        joint_count = len(reset.observation["robot0_joint_pos"])
        self._manifest = build_capx_tool_manifest(
            self._api_level,
            joint_count=joint_count,
            public_safe_release=self._public_safe_release,
            service_aligned=self._service_aligned,
            dehardcoded=self._dehardcoded,
            completion_evidence=(
                self._completion_protocol.evidence_only
            ),
            completion_confidence=(
                self._completion_protocol.confidence_repair
            ),
            v15_protocol=self._v15_protocol,
        )
        self._scorer = HostOnlyFinalScorer(self.__final_predicate)
        self._manifest = runtime_manifest(self._manifest, controller=self._robot.controller,
                                          motion_tools=self._motion_tools)
        if self._coding_control:
            from .direct_programming import DirectProgramming, direct_manifest
            self._manifest = direct_manifest(self._manifest)
            self._direct_programming = DirectProgramming(self._tools)
            from captain_harness.ports.public_records import PublicRecordStore
            self._direct_programming._measurement_receipts = PublicRecordStore(
                Path(self._observation_dir).parent / "public_measurements")

        self._seed = seed
        self._init_state_index = reset.init_state_index
        return self.observe()

    def _require_robot(self) -> LiberoActionBackend:
        if self._robot is None:
            raise RuntimeError("CaP-X LIBERO adapter must be reset before use")
        return self._robot

    def capabilities(self) -> CapabilityManifest:
        if self._manifest is None:
            raise RuntimeError("CaP-X LIBERO adapter must be reset before use")
        return self._manifest

    @property
    def calibration_metadata(self) -> Mapping[str, object]:
        return self._calibration_profile.public_metadata()

    @property
    def completion_protocol_metadata(self) -> Mapping[str, object]:
        return self._completion_protocol.public_metadata()

    @property
    def experience_metadata(self) -> Mapping[str, object] | None:
        return (
            self._experience_snapshot.public_metadata()
            if self._experience_snapshot is not None
            else None
        )

    def observe(self) -> EnvironmentObservation:
        robot = self._require_robot()
        result = robot.get_observation_data()
        if not result.success or result.value is None:
            raise RuntimeError(f"LIBERO observation failed: {result.message}")
        images = tuple(
            ImageObservation(
                camera.camera_name,
                camera.rgb_path,
                camera.depth_path,
                "head" if index == 0 else "wrist",
                {"intrinsics_3x3":list(camera.intrinsics_3x3),
                 "camera_to_robot_base_4x4":list(camera.extrinsics_4x4),
                 "depth_units":"meter", "depth_convention":"camera optical z",
                 "pixel_origin":"top_left", "sample_time_s":result.value.time_s,
                 "source":"declared_rgbd_sensor_and_camera_calibration"},
            )
            for index, camera in enumerate(result.value.cameras)
        )
        state = result.value.robot_state
        return EnvironmentObservation(
            step=robot.simulation_steps,
            images=images,
            public_state={
                "seed": self._seed,
                "init_state_index": self._init_state_index,
                "robot_state": _json_value(state),
                **({"end_effector_geometry": self._direct_programming.geometry()}
                   if self._coding_control else {}),
            },
        )

    def public_measurement_evidence(self, measurement_ids=None):
        if not self._coding_control:
            return ()
        records = self._direct_programming._measurement_receipts
        return (records[key] for key in (measurement_ids if measurement_ids is not None else records)
                if key in records)

    def execution_recovery_evidence(self, *, failed_capability, failed_arguments,
                                   capability, arguments, result):
        if not self._coding_control or failed_capability != "move_to" or capability != "move_to":
            return None
        from captain_harness.adapters.robotics.execution_goal_evidence import cartesian_recovery_evidence
        return cartesian_recovery_evidence(failed_arguments, result,
            argument_schema=self.capabilities().require("move_to").arguments_schema)

    def execute(
        self, capability: str, arguments: Mapping[str, object]
    ) -> CapabilityResult:
        if self._tools is None:
            raise RuntimeError("CaP-X LIBERO adapter must be reset before use")
        invoke = self._tools.invoke
        if self._coding_control:
            invoke = self._direct_programming.invoke
        elif self._motion_tools:
            from .possession_tools import invoke_with_possession_evidence
            from .grasp_preparation import add_grasp_preparation
            def invoke(name, args):
                from .attempt_identity import resolve_attempt_identity
                spec = self.capabilities().require(name)
                identity = resolve_attempt_identity(self._tools, spec, args)
                result = invoke_with_possession_evidence(self._tools, spec, args)
                result = add_grasp_preparation(self._tools, name, result)
                from .attempt_evidence import AttemptEvidenceBinding
                if not hasattr(self, "_attempt_evidence_binding"):
                    self._attempt_evidence_binding = AttemptEvidenceBinding()
                return self._attempt_evidence_binding.annotate(self._tools, spec, args, result, identity)
        steps_before = getattr(self._require_robot(), 'simulation_steps', None)
        result = ValidatedCapabilityBroker(self.capabilities(), invoke).invoke(capability, arguments)
        steps_after = getattr(self._require_robot(), 'simulation_steps', None)
        configuration_advanced = (steps_after > steps_before
            if isinstance(steps_before, int) and isinstance(steps_after, int) else result.executed)
        if configuration_advanced:
            spec = self.capabilities().require(capability)
            if (spec.side_effect in {SideEffect.ROBOT_MOTION, SideEffect.ACTUATOR}
                    and spec.control_semantic is not ControlSemantic.TIME_ADVANCE):
                self._tools._last_configuration_motion_time_s = self._require_robot().time_s
            if capability in ('move_to', 'hold', 'release', 'recover'):
                self._last_physical_step = self._require_robot().simulation_steps
                if capability == 'move_to' and arguments.get('gripper') == 'close' and self._first_grasp_step is None:
                    self._first_grasp_step = self._last_physical_step
        self._mechanism_log(capability, arguments, result)
        return result

    # ------------------------------------------------------------------ completion gate (host-side, public records only)
    def _reset_gate_state(self) -> None:
        self._last_physical_step = None
        self._first_grasp_step = None
        self._geometry_records = []
        self._comparison_records = []
        self._gate_rejections = 0

    def completion_gate(self):
        """CompletionGatePort: None passes; a mapping rejects with the agent's own numbers."""
        if not self._completion_gate_enabled or not self._completion_goals:
            return None
        from .completion_gate import evaluate, MAX_REJECTIONS
        verdict = evaluate(self._completion_goals, self._geometry_records, self._comparison_records,
                           last_physical_step=self._last_physical_step, first_grasp_step=self._first_grasp_step,
                           rejections=self._gate_rejections)
        if verdict.get('verdict') == 'reject':
            self._gate_rejections += 1
        try:
            root = Path(self._observation_dir).parent
            with (root / 'completion_gate.jsonl').open('a') as out:
                out.write(json.dumps({'goals': self._completion_goals, 'last_physical_step': self._last_physical_step,
                                      'first_grasp_step': self._first_grasp_step, 'geometry_records': len(self._geometry_records),
                                      'comparison_records': len(self._comparison_records), 'rejections': self._gate_rejections,
                                      **verdict}, default=str) + '\n')
        except Exception:
            pass
        if verdict.get('verdict') != 'reject':
            return None
        return {'failure': verdict['failure'], 'message': verdict['message'], 'rejections': self._gate_rejections,
                'max_rejections': __import__('captain_harness.adapters.capx.completion_gate', fromlist=['MAX_REJECTIONS']).MAX_REJECTIONS}

    def _mechanism_log(self, capability, arguments, result):
        """Host-side per-call mechanism logs for attribution (never model input)."""
        try:
            root = Path(self._observation_dir).parent
            value = result.value if isinstance(result.value, Mapping) else {}
            if capability in ('move_to', 'hold', 'release', 'recover'):
                detector = value.get('contact_detector') or {}
                row = {'capability': capability, 'gripper': arguments.get('gripper'),
                       'contact_mode': arguments.get('contact_mode', 'guarded'), 'load_budget_n': arguments.get('load_budget_n'),
                       'success': result.success, 'failure_category': result.failure_category.value if result.failure_category else None,
                       'stopped_reason': value.get('stopped_reason'), 'detector_mode': detector.get('mode'),
                       'trigger': (detector.get('trigger') or {}).get('reason'),
                       'peak_contact_n': detector.get('peak_contact_n'), 'peak_residual_n': detector.get('peak_residual_n'),
                       'threshold_n': detector.get('threshold_n'), 'max_observed_force_n': value.get('max_observed_force_n'),
                       'executed_distance_m': value.get('executed_distance_m'), 'step': self._require_robot().simulation_steps}
                with (root / 'contact_events.jsonl').open('a') as out:
                    out.write(json.dumps(row) + '\n')
            elif capability == 'compare_object_observations' and result.success:
                from .completion_gate import comparison_record
                self._comparison_records.append(comparison_record(value, self._require_robot().simulation_steps))
            elif capability == 'measure_object_geometry':
                if result.success:
                    from .completion_gate import geometry_record
                    self._geometry_records.append(geometry_record(value, dict(arguments), self._require_robot().simulation_steps))
                obj = value.get('object') or {}
                row = {'success': result.success, 'quality': obj.get('quality'), 'model': (obj.get('footprint') or {}).get('model'),
                       'views_used': obj.get('views_used'), 'center_m': obj.get('center_m'),
                       'xy_distance_m': value.get('xy_distance_m'), 'has_destination': 'destination' in value,
                       'error': None if result.success else result.message, 'step': self._require_robot().simulation_steps}
                with (root / 'grounding_calls.jsonl').open('a') as out:
                    out.write(json.dumps(row) + '\n')
        except Exception:
            pass

    def public_evidence_observations(self, source_ids: Sequence[str]) -> Mapping[str, EnvironmentObservation]:
        """Host-only access to already captured public frames, never hidden state."""
        if self._tools is None:
            return {}
        return {source:self._tools._public_observation_history[source] for source in source_ids
                if source in self._tools._public_observation_history}

    def audit_hidden_success(self) -> bool:
        if self._scorer is None:
            raise RuntimeError(
                "CaP-X LIBERO adapter must be reset before hidden audit sampling"
            )
        return self._scorer.score_after_action(
            environment_step=self._require_robot().simulation_steps
        )

    def finalize(self) -> HostCompletion:
        if self._scorer is None:
            raise RuntimeError("CaP-X LIBERO adapter must be reset before scoring")
        try:
            final_success = self._scorer.score_after_loop(loop_ended=True)
            return HostCompletion(
                self._scorer.success_ever,
                final_success,
                self._scorer.first_success_step,
            )
        finally:
            if self._registry is not None:
                self._registry.close()


def _json_value(value: object) -> object:
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
    return value
