"""CaP-X tool adapter over the XR1 facade: same manifest, tools and receipts as LIBERO."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping

from captain_harness.adapters.capx.artifacts import EpisodeArtifactRegistry
from captain_harness.adapters.capx.backend import ToolCapabilityBackend
from captain_harness.adapters.capx.environment import CapXLiberoEnvironmentAdapter
from captain_harness.adapters.capx.manifest import build_capx_tool_manifest
from captain_harness.adapters.capx.runtime_manifest import runtime_manifest
from captain_harness.adapters.libero.scoring import HostOnlyFinalScorer

from .backend import XR1ActionBackend


class CapXXR1EnvironmentAdapter(CapXLiberoEnvironmentAdapter):
    """Identical public surface; the robot backend is XR1ActionBackend and reset never replays LIBERO states."""

    def __init__(self, env: Any, *, final_predicate: Callable[[], bool], camera_link_to_optical=None,
                 task_markdown_overrides: Mapping[str, object] | None = None, **kwargs) -> None:
        super().__init__(env, init_states=(), final_predicate=final_predicate, **kwargs)
        self._xr1_env = env
        self._xr1_final_predicate = final_predicate
        self._camera_link_to_optical = camera_link_to_optical
        self.task_markdown_overrides = dict(task_markdown_overrides or {})

    _event_hook = None  # optional mid-task event hook: .after(capability, advanced, robot) after every call

    def execute(self, capability, arguments):
        robot = self._require_robot()
        before = getattr(robot, "simulation_steps", None)
        result = super().execute(capability, arguments)
        if self._event_hook is not None:
            after = getattr(robot, "simulation_steps", None)
            advanced = (after > before) if isinstance(before, int) and isinstance(after, int) else bool(getattr(result, "executed", False))
            self._event_hook.after(capability, advanced, robot)
        return result

    def reset(self, seed: int):
        if self._registry is not None:
            self._registry.close()
        from captain_harness.adapters.capx.attempt_evidence import AttemptEvidenceBinding
        self._attempt_evidence_binding = AttemptEvidenceBinding()
        backend_options = dict(self._backend_options)
        contact_detector = str(backend_options.pop("contact_detector", "legacy"))
        contact_calibration = bool(backend_options.pop("contact_calibration", False))
        backend_options.pop("depth_converter", None)
        observation = self._xr1_env.current_observation()
        self._robot = XR1ActionBackend(
            self._xr1_env,
            observation_dir=self._observation_dir,
            initial_observation=observation,
            horizon=self._horizon,
            calibration_profile=self._calibration_profile,
            camera_link_to_optical=self._camera_link_to_optical,
            guarded_cartesian_motion=self._motion_tools,
            **backend_options,
        )
        self._robot._contact_detector_mode = contact_detector
        self._contact_calibration_receipt = None
        self._reset_gate_state()
        if contact_detector == "residual" and contact_calibration:
            calibration, receipt = self._robot.calibrate_contact_residual()
            self._contact_calibration_receipt = {"calibration": calibration.to_json(), **receipt}
            Path(self._observation_dir).parent.mkdir(parents=True, exist_ok=True)
            (Path(self._observation_dir).parent / "force_calibration.json").write_text(
                json.dumps(self._contact_calibration_receipt, indent=2))
        episode_id = f"xr1-{self._api_level.value.lower()}-{seed}-{uuid.uuid4().hex}"
        self._registry = EpisodeArtifactRegistry(
            episode_id, ttl_s=self._artifact_ttl_s,
            storage_root=self._observation_dir.parent / "public_sensor_artifacts" / episode_id)
        self._tools = ToolCapabilityBackend(
            robot=self._robot, services=self._services, registry=self._registry,
            tcp_offset_local_m=self._calibration_profile.tcp_offset_local_m,
            gripper_profile=self._calibration_profile.gripper,
            default_approach_frame="world" if self._service_aligned else "tool",
            completion_protocol=self._completion_protocol, dehardcoded=self._dehardcoded,
            v15_protocol=self._v15_protocol, motion_tools=self._motion_tools,
            experience_snapshot=self._experience_snapshot,
            graspnet_to_eef_transform=self._calibration_profile.graspnet_to_eef_transform,
            **self._tool_backend_options)
        self._manifest = build_capx_tool_manifest(
            self._api_level, joint_count=len(observation["robot0_joint_pos"]),
            public_safe_release=self._public_safe_release, service_aligned=self._service_aligned,
            dehardcoded=self._dehardcoded, completion_evidence=self._completion_protocol.evidence_only,
            completion_confidence=self._completion_protocol.confidence_repair, v15_protocol=self._v15_protocol)
        self._scorer = HostOnlyFinalScorer(self._xr1_final_predicate)
        self._manifest = runtime_manifest(self._manifest, controller=self._robot.controller,
                                          motion_tools=self._motion_tools)
        if self._coding_control:
            from captain_harness.adapters.capx.direct_programming import DirectProgramming, direct_manifest
            from captain_harness.ports.public_records import PublicRecordStore
            self._manifest = direct_manifest(self._manifest)
            self._direct_programming = DirectProgramming(self._tools)
            self._direct_programming._measurement_receipts = PublicRecordStore(
                Path(self._observation_dir).parent / "public_measurements")
        self._seed = seed
        self._init_state_index = None
        return self.observe()
