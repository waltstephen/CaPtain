from __future__ import annotations

from dataclasses import replace

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image

from captain_harness.ports.capability import CapabilityResult as ActionResult
from captain_harness.ports.capability import FailureCategory
from .calibration import (
    LiberoCalibrationProfile,
    load_calibration_profile,
)
from .geometry import DepthUnavailableError, pixel_to_world as project_pixel_to_world
from .types import CameraData, ObservationData, RobotState
from .contact_motion import ContactMotionMixin
from .fresh_init import FreshInitStates, apply_visual_variation, is_fresh_seed, select_fresh_state

IKSolver = Callable[[np.ndarray, np.ndarray, np.ndarray], Sequence[float] | None]
DepthConverter = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class ResetResult:
    seed: int
    init_state_index: int
    observation: Mapping[str, Any]
    init_state_source: str = "official"


def reset_to_seeded_init_state(
    env: Any,
    init_states: Sequence[Any],
    *,
    seed: int,
    settling_steps: int = 5,
    fresh_init_states: FreshInitStates | None = None,
) -> ResetResult:
    """Reset and select a deterministic LIBERO initial state.

    seed < FRESH_SEED_BASE (1000): official state ``init_states[seed % len(init_states)]`` (unchanged mapping).
    seed = FRESH_SEED_BASE + k: fresh state k of ``fresh_init_states`` (reset with its recorded generation seed first,
    so fixture poses match); there is no fallback to an official state.
    """
    if len(init_states) == 0:
        raise ValueError("LIBERO init_states must not be empty")
    if settling_steps < 0:
        raise ValueError("settling_steps must not be negative")
    if is_fresh_seed(seed):
        init_state_index, reset_seed, state = select_fresh_state(fresh_init_states, seed)
        source = "fresh"
    else:
        init_state_index, reset_seed = seed % len(init_states), seed
        state = init_states[init_state_index]
        source = "official"
    env.seed(reset_seed)
    env.reset()
    if source == "fresh" and fresh_init_states.visual is not None:
        apply_visual_variation(env, fresh_init_states.visual[init_state_index])
    observation = env.set_init_state(state)
    gripper_qpos = np.asarray(
        observation["robot0_gripper_qpos"],
        dtype=np.float64,
    ).reshape(-1)
    gripper_fraction = float(
        np.clip(gripper_qpos[0] / 0.04, 0.0, 1.0)
    )
    robot = env.robots[0]
    action_dim = int(
        getattr(
            robot,
            "action_dim",
            len(np.asarray(observation["robot0_joint_pos"]).reshape(-1)) + 1,
        )
    )
    dummy_action = np.zeros(action_dim, dtype=np.float64)
    dummy_action[-1] = 1.0 - 2.0 * gripper_fraction
    for _ in range(settling_steps):
        observation, _, _, _ = env.step(dummy_action)
    return ResetResult(seed, init_state_index, observation, source)


def _matrix_to_quaternion_wxyz(matrix: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        values = (0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale,
                  (matrix[0, 2] - matrix[2, 0]) / scale,
                  (matrix[1, 0] - matrix[0, 1]) / scale)
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            values = ((matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale,
                      (matrix[0, 1] + matrix[1, 0]) / scale,
                      (matrix[0, 2] + matrix[2, 0]) / scale)
        elif axis == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            values = ((matrix[0, 2] - matrix[2, 0]) / scale,
                      (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                      (matrix[1, 2] + matrix[2, 1]) / scale)
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            values = ((matrix[1, 0] - matrix[0, 1]) / scale,
                      (matrix[0, 2] + matrix[2, 0]) / scale,
                      (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale)
    norm = math.sqrt(sum(value * value for value in values))
    return tuple(float(value / norm) for value in values)  # type: ignore[return-value]


def _quaternion_wxyz_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    values = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(values))
    if not np.isfinite(values).all() or norm < 1e-12:
        raise ValueError("quaternion must be finite and nonzero")
    w, x, y, z = values / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _quaternion_rotation_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Return the shortest target * inverse(current) rotation vector."""
    tw, tx, ty, tz = target
    cw, cx, cy, cz = current
    error = np.array([
        tw * cw + tx * cx + ty * cy + tz * cz,
        -tw * cx + tx * cw - ty * cz + tz * cy,
        -tw * cy + tx * cz + ty * cw - tz * cx,
        -tw * cz - tx * cy + ty * cx + tz * cw,
    ])
    if error[0] < 0.0:
        error = -error
    vector_norm = float(np.linalg.norm(error[1:]))
    if vector_norm < 1e-12:
        return 2.0 * error[1:]
    angle = 2.0 * math.atan2(vector_norm, float(np.clip(error[0], -1.0, 1.0)))
    return error[1:] * (angle / vector_norm)


def _rotation_matrix_to_axis_angle(matrix: np.ndarray) -> np.ndarray:
    rotation = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1e-9:
        return np.zeros(3, dtype=np.float64)
    if math.pi - angle < 1e-5:
        diagonal = np.maximum((np.diag(rotation) + 1.0) * 0.5, 0.0)
        axis = np.sqrt(diagonal)
        axis[0] = math.copysign(
            axis[0],
            rotation[2, 1] - rotation[1, 2],
        )
        axis[1] = math.copysign(
            axis[1],
            rotation[0, 2] - rotation[2, 0],
        )
        axis[2] = math.copysign(
            axis[2],
            rotation[1, 0] - rotation[0, 1],
        )
        norm = float(np.linalg.norm(axis))
        if norm < 1e-9:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            axis /= norm
        return axis * angle
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    axis /= 2.0 * math.sin(angle)
    return axis * angle


# In-loop targets are strict; accept thresholds are what a stalled phase may
# end at and still count as converged. The V16 mainline showed a bimodal
# translate-failure distribution (near-misses at 0.02-0.055 m against a
# genuinely-unreachable tail from 0.068 m) and a rotate near-miss cluster at
# 0.13-0.18 rad; the hover waypoint needs no precision because settle and
# the 0.015 m descend gate enforce the final accuracy, and the guarded
# contact descent tolerates ~10 degrees of approach tilt.
_ALIGN_TRANSLATE_TOLERANCE_M = 0.030
_ALIGN_TRANSLATE_ACCEPT_TOLERANCE_M = 0.060
_ALIGN_ROTATE_ORIENTATION_TOLERANCE_RAD = 0.087
_ALIGN_ROTATE_ACCEPT_TOLERANCE_RAD = 0.18
_ALIGN_ROTATE_POSITION_SLACK_M = 0.050
_ALIGN_SETTLE_TOLERANCE_M = 0.020
_ALIGN_SETTLE_ACCEPT_TOLERANCE_M = 0.060
_ALIGN_DESCEND_TOLERANCE_M = 0.015


class LiberoActionBackend(ContactMotionMixin):
    """Strict adapter from the frozen CaP API to a LIBERO controller."""

    CAMERA_NAMES = ("agentview", "robot0_eye_in_hand")
    # v5 contact residual state (set by the adapter after construction / calibration)
    _force_calibration = None
    _contact_detector_mode = "legacy"
    _contact_direction_world = None

    def __init__(
        self,
        env: Any,
        *,
        observation_dir: Path,
        initial_observation: Mapping[str, Any],
        horizon: int,
        control_frequency: int = 20,
        motion_tolerance_rad: float = 0.01,
        motion_max_steps: int = 120,
        home_max_steps: int | None = None,
        gripper_steps: int = 40,
        gripper_settle_min_steps: int | None = None,
        image_size: tuple[int, int] = (512, 512),
        camera_names: Sequence[str] | None = None,
        inverse_kinematics: IKSolver | None = None,
        depth_converter: DepthConverter | None = None,
        cartesian_position_tolerance_m: float = 0.005,
        cartesian_orientation_tolerance_rad: float = 0.03,
        cartesian_damping: float = 0.05,
        cartesian_joint_step_rad: float = 0.05,
        cartesian_max_steps: int = 160,
        cartesian_stall_steps: int = 8,
        controller: str = "JOINT_POSITION",
        calibration_profile: LiberoCalibrationProfile | None = None,
        osc_position_tolerance_m: float = 0.004,
        osc_orientation_tolerance_rad: float = 0.05,
        osc_settle_velocity_m_s: float = 0.01,
        guarded_cartesian_motion: bool = False,
    ) -> None:
        if horizon <= 0 or control_frequency <= 0:
            raise ValueError("horizon and control_frequency must be positive")
        if motion_max_steps <= 0 or gripper_steps <= 0:
            raise ValueError("motion and gripper step limits must be positive")
        if min(cartesian_position_tolerance_m, cartesian_orientation_tolerance_rad,
               cartesian_damping, cartesian_joint_step_rad) <= 0:
            raise ValueError("Cartesian tolerances, damping, and joint step must be positive")
        if cartesian_max_steps <= 0 or cartesian_stall_steps <= 0:
            raise ValueError("Cartesian step limits must be positive")
        if min(
            osc_position_tolerance_m,
            osc_orientation_tolerance_rad,
            osc_settle_velocity_m_s,
        ) <= 0:
            raise ValueError("OSC tolerances and settle velocity must be positive")
        if controller not in {"JOINT_POSITION", "OSC_POSE"}:
            raise ValueError(f"unsupported LIBERO controller: {controller!r}")
        self.__env = env
        self._observation_dir = observation_dir
        self._observation_dir.mkdir(parents=True, exist_ok=True)
        self._observation = initial_observation
        self._horizon = horizon
        self._control_frequency = control_frequency
        self._guarded_cartesian_motion = bool(guarded_cartesian_motion)
        self._motion_tolerance_rad = motion_tolerance_rad
        self._motion_max_steps = motion_max_steps
        self._home_max_steps = home_max_steps or motion_max_steps * 2
        self._gripper_steps = gripper_steps
        # opt-in (RoboDojo X5): end an open/close once the measured aperture stopped changing, after at least N steps
        self._gripper_settle_min_steps = gripper_settle_min_steps
        self._image_size = image_size
        self._camera_names = tuple(camera_names) if camera_names is not None else tuple(
            name for name in self.CAMERA_NAMES if f'{name}_image' in initial_observation)
        if not self._camera_names or any(f'{name}_image' not in initial_observation for name in self._camera_names):
            raise ValueError('camera_names must identify available RGB observations')
        self._inverse_kinematics = inverse_kinematics
        self._depth_converter = depth_converter or (lambda depth: depth)
        self._cartesian_position_tolerance_m = cartesian_position_tolerance_m
        self._cartesian_orientation_tolerance_rad = cartesian_orientation_tolerance_rad
        self._cartesian_damping = cartesian_damping
        self._cartesian_joint_step_rad = cartesian_joint_step_rad
        self._cartesian_max_steps = cartesian_max_steps
        self._cartesian_stall_steps = cartesian_stall_steps
        self._controller = controller
        self._osc_position_tolerance_m = osc_position_tolerance_m
        self._osc_orientation_tolerance_rad = (
            osc_orientation_tolerance_rad
        )
        self._osc_settle_velocity_m_s = osc_settle_velocity_m_s
        self._calibration = calibration_profile or load_calibration_profile()
        self._pyroki_hand_to_libero_eef = (
            self._calibration.hand_to_eef_transform
        )
        self._sim_steps = 0
        self._snapshot_index = 0
        self._gripper_open_fraction = self._read_gripper_fraction()
        self._gripper_command_fraction = self._gripper_open_fraction
        self._home_joints = self._read_joints()
        self._base_body_id = env.sim.model.body_name2id("robot0_base")
        self._eef_body_id = env.sim.model.body_name2id("gripper0_eef")
        robot = env.robots[0]
        actual_controller = getattr(
            getattr(robot, "controller", None),
            "name",
            None,
        )
        if (
            isinstance(actual_controller, str)
            and actual_controller != controller
        ):
            raise ValueError(
                "configured controller does not match LIBERO runtime: "
                f"{controller!r} != {actual_controller!r}"
            )
        self._controller_action_dim = int(
            getattr(
                getattr(robot, "controller", None),
                "control_dim",
                7,
            )
        )
        self._action_dim = int(
            getattr(
                robot,
                "action_dim",
                self._controller_action_dim + 1,
            )
        )
        controller_output = np.asarray(
            getattr(
                getattr(robot, "controller", None),
                "output_max",
                (0.05, 0.05, 0.05, 0.5, 0.5, 0.5),
            ),
            dtype=np.float64,
        ).reshape(-1)
        self._osc_output_max = (
            controller_output[:6]
            if controller_output.size >= 6
            else np.asarray(
                (0.05, 0.05, 0.05, 0.5, 0.5, 0.5),
                dtype=np.float64,
            )
        )
        self._control_site_id: int | None = None
        self._eef_body_to_control_site = np.eye(4, dtype=np.float64)
        if controller == "OSC_POSE":
            gripper = getattr(robot, "gripper", None)
            important_sites = getattr(gripper, "important_sites", {})
            site_name = (
                important_sites.get("grip_site")
                if isinstance(important_sites, Mapping)
                else None
            )
            if not isinstance(site_name, str):
                raise ValueError(
                    "OSC_POSE requires the robot gripper grip_site"
                )
            self._control_site_id = env.sim.model.site_name2id(site_name)
            self._eef_body_to_control_site = (
                np.linalg.inv(self._eef_body_pose_world())
                @ self._control_site_pose_world()
            )
        self._joint_velocity_indexes = np.asarray(
            robot._ref_joint_vel_indexes, dtype=np.int64
        ).reshape(7)
        limits = []
        for name in robot.robot_model.joints:
            joint_id = env.sim.model.joint_name2id(name)
            if env.sim.model.jnt_limited[joint_id]:
                limits.append(np.asarray(env.sim.model.jnt_range[joint_id], dtype=np.float64))
            else:
                limits.append(np.array([-np.inf, np.inf]))
        self._joint_limits = np.asarray(limits, dtype=np.float64).reshape(7, 2)
        home_state = self._robot_state()
        self._home_eef_position = tuple(home_state.end_effector_position_m)
        self._home_eef_quaternion = tuple(
            home_state.end_effector_quaternion_wxyz
        )
        # Opt-in host-side speedups (fast_loop.py; default off = the original code path, values bit-identical when on).
        from .fast_loop import fast_loop_enabled, lazy_render_enabled
        self._fast_force = None
        self._lazy_cameras = None
        if fast_loop_enabled():
            from .fast_loop import ForceSensorCache
            try:
                self._fast_force = ForceSensorCache(env)
            except Exception:
                self._fast_force = None
        if lazy_render_enabled():
            from .fast_loop import LazyCameras
            self._lazy_cameras = LazyCameras(env)

    def _current_observation(self):
        """self._observation with any deferred camera sample rendered (CAPTAIN_LAZY_RENDER)."""
        lazy = getattr(self, "_lazy_cameras", None)
        if lazy is not None:
            lazy.materialize(self._observation)
        return self._observation

    @property
    def simulation_steps(self) -> int:
        return self._sim_steps

    @property
    def controller(self) -> str:
        return self._controller

    @property
    def time_s(self) -> float:
        return self._sim_steps / self._control_frequency

    def planning_model(self):
        """Build from robot XML only, never the assembled environment model."""
        from captain_harness.adapters.robotics.model import RobotCollisionModel
        if getattr(self, '_planning_model', None) is None:
            robot = self.__env.robots[0]
            self._planning_model = RobotCollisionModel(robot.robot_model.get_xml(),
                joints=list(robot.robot_model.joints), gripper_joints=list(robot.gripper.joints),
                base='robot0_base', end_effector='gripper0_eef', gripper_prefix='gripper0_',
                angle_unit='radian')
            if not np.allclose(self._planning_model.limits, self._joint_limits, rtol=0, atol=1e-9):
                self._planning_model = None
                raise ValueError('detached robot limits disagree with the hardware controller model')
        return self._planning_model

    def planned_path_tracking_policy(self):
        if self._controller != 'OSC_POSE':
            return None
        return {'position_tolerance_m': .003, 'orientation_tolerance_rad': .03,
            'scope': 'rigid_gripper_only',
            'basis': 'existing internal Cartesian waypoint acceptance tolerances',
            'limitation': 'Requested planning allowance, not a certified bound on servo error; arm posture remains independently monitored.'}

    def execute_joint_path(self, waypoints, collision_check, *, endpoint_target=None, endpoint_precision=None):
        """Follow planned configurations without switching the active controller.

        OSC execution uses FK waypoints and its nullspace posture target. Every
        measured joint configuration is checked; this is not a claim of exact
        joint-trajectory tracking by a Cartesian controller.
        """
        from scipy.spatial.transform import Rotation
        from captain_harness.adapters.robotics.motion_precision import motion_precision
        from captain_harness.adapters.libero.contact_motion import _pose
        path = np.asarray(waypoints, dtype=float)
        if path.ndim != 2 or path.shape[1] != 7 or len(path) < 2 or not np.isfinite(path).all():
            return ActionResult.invalid_argument('path must contain at least two finite 7-joint states')
        if np.max(np.abs(path[0] - self._read_joints())) > 1e-6:
            return ActionResult.failed(FailureCategory.NOT_EXECUTED, 'planned start joints changed')
        model = self.planning_model()
        trace, stop = [], {}
        start_steps = self._sim_steps
        old_guard = getattr(self, '_motion_guard', None)
        controller = self.__env.robots[0].controller
        start_aperture = self._read_gripper_fraction()
        start_pose = _pose(self.get_robot_state().value)
        endpoint = np.asarray(model.eef_pose(path[-1], start_aperture)
                              if endpoint_target is None else endpoint_target, dtype=float)
        if endpoint.shape != (4, 4) or not np.isfinite(endpoint).all():
            return ActionResult.invalid_argument('planned endpoint must be a finite 4x4 pose')
        precision = motion_precision(start_pose, endpoint, .003, .03)
        if endpoint_precision is not None:
            for key in ['position_tolerance_m', 'orientation_tolerance_rad']:
                value = float(endpoint_precision[key])
                if not np.isfinite(value) or value <= 0:
                    return ActionResult.invalid_argument('planned endpoint accuracy must be positive and finite')
                precision[key] = min(precision[key], value)
        tracking_policy = self.planned_path_tracking_policy()
        last_result = None
        def monitor():
            q = self._read_joints().copy()
            check = collision_check(q)
            # Euclidean distance to each segment (not just sparse endpoints).
            edges = path[1:] - path[:-1]
            fractions = np.clip(np.sum((q - path[:-1]) * edges, axis=1) /
                                np.maximum(np.sum(edges * edges, axis=1), 1e-12), 0., 1.)
            deviation = float(np.min(np.linalg.norm(q - (path[:-1] + fractions[:, None] * edges), axis=1)))
            trace.append({'step': self._sim_steps, 'joint_positions_rad': q.tolist(),
                          'path_deviation_rad': deviation, 'collision': check,
                          'force_n': self._eef_force_norm()})
            if check['collision']:
                stop.update(reason='observed_collision_during_execution', collision=check)
            elif deviation > .15:
                stop.update(reason='controller_left_planned_joint_corridor', path_deviation_rad=deviation)
            if old_guard is not None:
                try:
                    if not old_guard():
                        stop.setdefault('reason', 'motion_guard_rejected')
                        stop['outer_guard_rejected'] = True
                except Exception as exc:
                    stop.setdefault('reason', 'outer_motion_guard_failed')
                    stop['outer_guard_error'] = f'{type(exc).__name__}: {exc}'
            return not stop
        self._motion_guard = monitor
        completed = 0
        try:
            for index, target in enumerate(path[1:], 1):
                if not monitor():
                    break
                if self._controller == 'JOINT_POSITION':
                    result = self._move_to_joints(target, max_steps=self._motion_max_steps)
                else:
                    final_waypoint = index == len(path) - 1
                    pose = endpoint if final_waypoint else model.eef_pose(target, start_aperture)
                    quat = np.roll(Rotation.from_matrix(pose[:3, :3]).as_quat(), 1)
                    # Supported OSC nullspace reference, not a simulator state write.
                    controller.initial_joint = target.copy()
                    result = self._move_to_pose_osc(pose[:3, 3], quat, position_only=False,
                        max_steps=min(40, self._cartesian_max_steps),
                        position_tolerance_m=precision['position_tolerance_m'] if final_waypoint else tracking_policy['position_tolerance_m'],
                        orientation_tolerance_rad=precision['orientation_tolerance_rad'] if final_waypoint else tracking_policy['orientation_tolerance_rad'],
                        precision_reference='planned_path_endpoint' if final_waypoint else 'planned_path_waypoint')
                last_result = result
                if not result.success:
                    stop.setdefault('reason', result.message or 'waypoint_not_reached')
                    if isinstance(result.value, Mapping):
                        stop['controller_execution'] = dict(result.value)
                    break
                completed += 1
            monitor()
        except Exception as exc:
            stop.update(reason=f'{type(exc).__name__}: {exc}')
        finally:
            self._motion_guard = old_guard
            if self._controller == 'OSC_POSE':
                controller.initial_joint = self._read_joints().copy()
        final_pose = _pose(self.get_robot_state().value)
        position_error = float(np.linalg.norm(final_pose[:3, 3] - endpoint[:3, 3]))
        orientation_error = float(Rotation.from_matrix(endpoint[:3, :3] @ final_pose[:3, :3].T).magnitude())
        endpoint_reached = (position_error <= precision['position_tolerance_m']
                            and orientation_error <= precision['orientation_tolerance_rad'])
        if not stop and completed == len(path) - 1 and not endpoint_reached:
            stop['reason'] = 'planned_endpoint_not_reached'
        final_value = last_result.value if last_result is not None and isinstance(last_result.value, Mapping) else {}
        value = {'reason': stop.get('reason', 'path_executed'), 'trajectory': trace,
                 'endpoint_precision': precision, 'precision_reference': 'whole_planned_motion',
                 'position_residual_m': position_error, 'orientation_residual_rad': orientation_error,
                 'position_tolerance_m': precision['position_tolerance_m'],
                 'orientation_tolerance_rad': precision['orientation_tolerance_rad'],
                 'endpoint_settled': final_value.get('endpoint_settled'),
                 'converged': endpoint_reached, 'reached': endpoint_reached and not stop,
                 'control_steps': self._sim_steps - start_steps, 'completed_segments': completed,
                 'requested_segments': len(path) - 1, 'execution_controller': self._controller,
                 'joint_corridor_limit_rad': .15, 'stop_evidence': stop,
                 'execution_mode': 'joint_positions' if self._controller == 'JOINT_POSITION' else 'cartesian_waypoints_with_joint_monitor'}
        if stop or completed < len(path) - 1:
            if stop.get('reason') == 'outer_motion_guard_failed':
                failed = (last_result if last_result is not None else
                          ActionResult.failed(FailureCategory.BACKEND_ERROR, value['reason']))
                return replace(failed, success=False, failure_category=FailureCategory.BACKEND_ERROR,
                    message=value['reason'], executed=self._sim_steps > start_steps, value=value)
            if last_result is not None and not last_result.success:
                # Preserve the actual stop category and deadline diagnostics across
                # this composite action, while reporting all preceding motion.
                return replace(last_result, message=value['reason'],
                    executed=self._sim_steps > start_steps, value=value)
            return ActionResult.failed(FailureCategory.CONTROLLER_REJECTED, value['reason'],
                executed=self._sim_steps > start_steps, value=value)
        return ActionResult.ok(value)

    @property
    def calibration_profile(self) -> LiberoCalibrationProfile:
        return self._calibration

    def _read_joints(self) -> np.ndarray:
        return np.asarray(self._observation["robot0_joint_pos"], dtype=np.float64).reshape(7)

    def _read_gripper_fraction(self) -> float:
        qpos = np.asarray(self._observation["robot0_gripper_qpos"], dtype=np.float64)
        return float(np.clip(qpos[0] / 0.04, 0.0, 1.0))

    def _base_transform(self) -> tuple[np.ndarray, np.ndarray]:
        data = self.__env.sim.data
        rotation = np.asarray(data.xmat[self._base_body_id], dtype=np.float64).reshape(3, 3)
        position = np.asarray(data.xpos[self._base_body_id], dtype=np.float64)
        return rotation, position

    def _eef_body_pose_world(self) -> np.ndarray:
        data = self.__env.sim.data
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.asarray(
            data.xmat[self._eef_body_id],
            dtype=np.float64,
        ).reshape(3, 3)
        transform[:3, 3] = np.asarray(
            data.xpos[self._eef_body_id],
            dtype=np.float64,
        )
        return transform

    def _control_site_pose_world(self) -> np.ndarray:
        if self._control_site_id is None:
            return self._eef_body_pose_world()
        data = self.__env.sim.data
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.asarray(
            data.site_xmat[self._control_site_id],
            dtype=np.float64,
        ).reshape(3, 3)
        transform[:3, 3] = np.asarray(
            data.site_xpos[self._control_site_id],
            dtype=np.float64,
        )
        return transform

    def _robot_state(self) -> RobotState:
        data = self.__env.sim.data
        base_rotation, base_position = self._base_transform()
        eef_rotation_world = np.asarray(data.xmat[self._eef_body_id], dtype=np.float64).reshape(3, 3)
        eef_position_world = np.asarray(data.xpos[self._eef_body_id], dtype=np.float64)
        eef_position_base = base_rotation.T @ (eef_position_world - base_position)
        eef_rotation_base = base_rotation.T @ eef_rotation_world
        joint_velocity = np.asarray(
            self._observation.get("robot0_joint_vel", np.zeros(7)),
            dtype=np.float64,
        ).reshape(-1)
        return RobotState(
            joint_positions_rad=tuple(float(value) for value in self._read_joints()),
            end_effector_position_m=tuple(float(value) for value in eef_position_base),
            end_effector_quaternion_wxyz=_matrix_to_quaternion_wxyz(eef_rotation_base),
            gripper_open_fraction=self._read_gripper_fraction(),
            joint_velocity_norm_rad_s=float(np.linalg.norm(joint_velocity)),
        )

    def get_robot_state(self) -> ActionResult[RobotState]:
        try:
            return ActionResult.ok(self._robot_state())
        except Exception as exc:
            return ActionResult.failed(FailureCategory.BACKEND_ERROR, f"{type(exc).__name__}: {exc}")

    def external_ik_target_from_eef_pose(
        self,
        position: Sequence[float],
        quaternion: Sequence[float],
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        desired_eef = np.eye(4, dtype=np.float64)
        desired_eef[:3, :3] = _quaternion_wxyz_to_matrix(quaternion)
        desired_eef[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
        desired_hand = desired_eef @ np.linalg.inv(
            self._pyroki_hand_to_libero_eef
        )
        return (
            tuple(float(value) for value in desired_hand[:3, 3]),
            _matrix_to_quaternion_wxyz(desired_hand[:3, :3]),
        )

    def eef_pose_from_external_ik_target(
        self,
        position: Sequence[float],
        quaternion: Sequence[float],
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        desired_hand = np.eye(4, dtype=np.float64)
        desired_hand[:3, :3] = _quaternion_wxyz_to_matrix(quaternion)
        desired_hand[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
        desired_eef = desired_hand @ self._pyroki_hand_to_libero_eef
        return (
            tuple(float(value) for value in desired_eef[:3, 3]),
            _matrix_to_quaternion_wxyz(desired_eef[:3, :3]),
        )

    def gripper_contact_envelope(self) -> dict[str, object] | None:
        """Robot-only pad geometry, relative to EEF center in public hand axes.

        Read the robot's own two pad boxes and kinematics only. This exposes
        hardware dimensions, never environment bodies, contacts or object poses.
        The jaw-axis sweep is supplied by the public gripper-width profile.
        """
        model, data = self.__env.sim.model, self.__env.sim.data
        try:
            pad_ids = [model.geom_name2id(f"gripper0_finger{i}_pad_collision") for i in (1, 2)]
        except (ValueError, KeyError):
            return None
        if any(pad_id < 0 for pad_id in pad_ids):
            return None
        eef_rotation = np.asarray(data.xmat[self._eef_body_id]).reshape(3, 3)
        hand_rotation = eef_rotation @ self._pyroki_hand_to_libero_eef[:3, :3].T
        eef_position = np.asarray(data.xpos[self._eef_body_id])
        lower, upper = [], []
        for pad_id in pad_ids:
            center = hand_rotation.T @ (np.asarray(data.geom_xpos[pad_id]) - eef_position)
            orientation = hand_rotation.T @ np.asarray(data.geom_xmat[pad_id]).reshape(3, 3)
            half_extent = np.abs(orientation) @ np.asarray(model.geom_size[pad_id])
            lower.append(center - half_extent)
            upper.append(center + half_extent)
        axis_world = np.asarray(data.geom_xpos[pad_ids[1]]) - np.asarray(data.geom_xpos[pad_ids[0]])
        axis_local = hand_rotation.T @ axis_world
        axis_length = float(np.linalg.norm(axis_local))
        return {"pad_min_local_m": np.min(lower, axis=0).tolist(),
                "pad_max_local_m": np.max(upper, axis=0).tolist(),
                "origin_frame": "end_effector", "axes_frame": "hand", "units": "meter",
                "rotation_end_effector_from_axes": self._pyroki_hand_to_libero_eef[:3, :3].T.tolist(),
                "closing_axis_local": (axis_local / axis_length).tolist() if axis_length > 1e-9 else None,
                "closing_axis_direction": "finger1 pad center toward finger2 pad center; sign is an axis convention",
                "basis": "robot finger pad box dimensions and robot kinematics only"}

    def _camera_calibration(self, name: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
        model = self.__env.sim.model
        data = self.__env.sim.data
        width, height = self._image_size
        camera_id = model.camera_name2id(name)
        focal = 0.5 * height / math.tan(float(model.cam_fovy[camera_id]) * math.pi / 360.0)
        intrinsics = np.array(
            [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]
        )
        base_rotation, base_position = self._base_transform()
        camera_rotation_world = np.asarray(data.get_camera_xmat(name), dtype=np.float64).reshape(3, 3)
        camera_position_world = np.asarray(data.get_camera_xpos(name), dtype=np.float64)
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = base_rotation.T @ camera_rotation_world @ np.diag([1.0, -1.0, -1.0])
        extrinsics[:3, 3] = base_rotation.T @ (camera_position_world - base_position)
        return tuple(intrinsics.ravel()), tuple(extrinsics.ravel())

    def get_observation_data(self) -> ActionResult[ObservationData]:
        try:
            cameras = []
            width, height = self._image_size
            self._current_observation()
            for name in self._camera_names:
                rgb = np.asarray(self._observation[f"{name}_image"])[::-1]
                if rgb.shape != (height, width, 3):
                    raise ValueError(f"{name} RGB shape {rgb.shape} != {(height, width, 3)}")
                stem = f"observation_{self._snapshot_index:06d}_{name}"
                rgb_path = self._observation_dir / f"{stem}.png"
                Image.fromarray(rgb.astype(np.uint8, copy=False)).save(rgb_path)
                depth_path = None
                depth_key = f"{name}_depth"
                if depth_key in self._observation:
                    depth = np.asarray(self._observation[depth_key])[::-1]
                    depth = self._depth_converter(depth)
                    depth_path = self._observation_dir / f"{stem}_depth.npy"
                    np.save(depth_path, depth)
                intrinsics, extrinsics = self._camera_calibration(name)
                cameras.append(
                    CameraData(name, str(rgb_path), str(depth_path) if depth_path else None,
                               intrinsics, extrinsics)
                )
            self._snapshot_index += 1
            return ActionResult.ok(ObservationData(tuple(cameras), self._robot_state(), self._sim_steps / self._control_frequency))
        except Exception as exc:
            return ActionResult.failed(FailureCategory.BACKEND_ERROR, f"{type(exc).__name__}: {exc}")

    def pixel_to_world(self, camera: str, x_norm: float, y_norm: float) -> ActionResult[tuple[float, float, float]]:
        if camera not in self._camera_names:
            return ActionResult.failed(
                FailureCategory.INVALID_ARGUMENT, f"unknown camera: {camera!r}"
            )
        depth_key = f"{camera}_depth"
        self._current_observation()
        if depth_key not in self._observation:
            return ActionResult.failed(
                FailureCategory.NOT_EXECUTED, f"camera {camera!r} has no current depth"
            )
        try:
            depth = self._depth_converter(np.asarray(self._observation[depth_key])[::-1])
            intrinsics, extrinsics = self._camera_calibration(camera)
            point = project_pixel_to_world(
                depth,
                np.asarray(intrinsics).reshape(3, 3),
                np.asarray(extrinsics).reshape(4, 4),
                x_norm=float(x_norm),
                y_norm=float(y_norm),
            )
            return ActionResult.ok(point)
        except DepthUnavailableError as exc:
            return ActionResult.failed(FailureCategory.NOT_EXECUTED, str(exc))
        except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
            return ActionResult.failed(
                FailureCategory.INVALID_ARGUMENT, f"pixel projection failed: {exc}"
            )
        except Exception as exc:
            return ActionResult.failed(
                FailureCategory.BACKEND_ERROR, f"{type(exc).__name__}: {exc}"
            )

    def _deadline_rejection_result(self, executed, value=None):
        from captain_harness.ports.deadline import deadline_failure
        return deadline_failure(self._last_deadline, executed=executed, value=value)

    def _replace_gripper_target_with_measured_position(self, *, max_open_fraction_correction=None):
        from .gripper_target import replace_gripper_target
        return replace_gripper_target(self.__env, self._observation["robot0_gripper_qpos"],
            max_open_fraction_correction=max_open_fraction_correction)

    def cancel_direct_targets(self, *, cancel_gripper=False, max_open_fraction_correction=None):
        """Cancel accumulated control goals without advancing physics."""
        receipt = {'simulation_steps':self.simulation_steps}
        try:
            controller = self.__env.robots[0].controller
            controller.update(force=True)
            controller.reset_goal()
            receipt['arm_target'] = 'reset_to_measured_pose'
            if cancel_gripper:
                replacement = (self._replace_gripper_target_with_measured_position()
                    if max_open_fraction_correction is None else
                    self._replace_gripper_target_with_measured_position(
                        max_open_fraction_correction=max_open_fraction_correction))
                if replacement.get('status') != 'replaced':
                    raise RuntimeError('gripper target replacement unconfirmed')
                self._gripper_command_fraction = .5
                receipt['gripper_target'] = replacement
            else:
                receipt['gripper_target'] = 'preserved'
            receipt['status'] = 'cancelled'
        except Exception as exc:
            receipt.update(status='unconfirmed', error=type(exc).__name__)
            self._physical_execution_unknown = True
        return receipt

    def _step_action(self, action: Sequence[float]) -> bool:
        from captain_harness.ports.deadline import current_deadline
        self._last_step_rejection = None
        self._last_deadline = None
        deadline = current_deadline()
        if deadline is not None and deadline.remaining() <= 0:
            self._last_step_rejection = 'deadline_expired'
            self._last_deadline = deadline
            return False
        if self._sim_steps >= self._horizon:
            self._last_step_rejection = 'horizon_exhausted'
            return False
        guard = getattr(self, '_motion_guard', None)
        if guard is not None and not guard():
            self._last_step_rejection = 'motion_guard_rejected'
            return False
        values = np.asarray(action, dtype=np.float64).reshape(self._action_dim)
        if getattr(self, '_gripper_target_cancel_error', None) and values[-1] >= 0.:
            # An unverified stop must not resume the outstanding close target.
            # Explicit opening remains a normal native control command.
            self._last_step_rejection = 'motion_guard_rejected'
            return False
        if getattr(self, '_physical_execution_unknown', False):
            self._last_step_rejection = 'physical_execution_state_unknown'
            return False
        try:
            monitor=getattr(self,"_contact_substep_monitor",None)
            if monitor is None:
                observation, _, _, _ = self.__env.step(values)
            else:
                from .substep_force_guard import monitored_step
                needs_sample=(getattr(monitor,"detector",None) is not None or getattr(monitor,"collect_samples",False))
                observation, _, _, _ = monitored_step(self.__env,values,monitor,
                    read_force=self._eef_force_norm,
                    cancel=lambda:self.cancel_direct_targets(cancel_gripper=False),
                    read_sample=self._contact_sample if needs_sample else None)
        except Exception:
            # Simulator may have advanced before throwing; a counter is not proof
            # that nothing happened. No subsequent physical operation is permitted.
            self._physical_execution_unknown = True
            raise
        if values[-1] < 0.:
            self._gripper_target_cancel_error = None
        self._observation = observation
        self._sim_steps += 1
        return True

    def _step(self, joint_delta: np.ndarray, gripper_fraction: float) -> bool:
        if self._controller != "JOINT_POSITION":
            raise RuntimeError(
                "joint-delta stepping is unavailable under OSC_POSE"
            )
        action = np.concatenate(
            [
                joint_delta * self._control_frequency,
                [1.0 - 2.0 * gripper_fraction],
            ]
        )
        return self._step_action(action)

    def move_to_joints(self, joints: Sequence[float]) -> ActionResult[None]:
        if self._controller != "JOINT_POSITION":
            return ActionResult.failed(
                FailureCategory.CONTROLLER_REJECTED,
                "move_to_joints is unavailable while LIBERO is configured "
                "with OSC_POSE; use goto_pose, servo_to, or home",
            )
        return self._move_to_joints(joints, max_steps=self._motion_max_steps)

    def _move_to_joints(
        self, joints: Sequence[float], *, max_steps: int
    ) -> ActionResult[None]:
        target = np.asarray(joints, dtype=np.float64).reshape(7)
        executed = False
        try:
            for _ in range(max_steps):
                current = self._read_joints()
                if executed and np.linalg.norm(current - target) < self._motion_tolerance_rad:
                    return ActionResult.ok(None)
                if not self._step(target - current, self._gripper_command_fraction):
                    if self._last_step_rejection == 'deadline_expired':
                        return self._deadline_rejection_result(executed)
                    if self._last_step_rejection == 'motion_guard_rejected':
                        return ActionResult.failed(FailureCategory.CONTROLLER_REJECTED,
                            'motion_guard_rejected', executed=executed)
                    category = FailureCategory.TIMEOUT if executed else FailureCategory.NOT_EXECUTED
                    return ActionResult.failed(category, "LIBERO horizon exhausted", executed=executed)
                executed = True
        except Exception as exc:
            return ActionResult.failed(
                FailureCategory.BACKEND_ERROR, f"{type(exc).__name__}: {exc}", executed=executed
            )
        residual = target - self._read_joints()
        residual_norm = float(np.linalg.norm(residual))
        if executed and residual_norm < self._motion_tolerance_rad:
            return ActionResult.ok(None)
        max_residual_joint = int(np.argmax(np.abs(residual))) + 1
        return ActionResult.failed(
            FailureCategory.TIMEOUT,
            f"joint target did not converge in {max_steps} steps "
            f"(joint {max_residual_joint} max residual "
            f"{float(np.max(np.abs(residual))):.6f} rad, "
            f"L2 residual {residual_norm:.6f} rad)",
            executed=executed,
        )

    def move_to_pose(
        self,
        position_xyz,
        quaternion_wxyz,
        *,
        position_only: bool = False,
    ) -> ActionResult[None]:
        if self._controller == "OSC_POSE":
            return self._move_to_pose_osc(
                position_xyz,
                quaternion_wxyz,
                position_only=position_only,
            )
        if self._inverse_kinematics is not None:
            try:
                joints = self._inverse_kinematics(
                    np.asarray(position_xyz, dtype=np.float64),
                    np.asarray(quaternion_wxyz, dtype=np.float64),
                    self._read_joints().copy(),
                )
            except Exception as exc:
                return ActionResult.failed(
                    FailureCategory.CONTROLLER_REJECTED, f"IK failed: {type(exc).__name__}: {exc}"
                )
            if joints is None:
                return ActionResult.failed(FailureCategory.CONTROLLER_REJECTED, "IK found no solution")
            return self.move_to_joints(joints)
        return self._move_to_pose_dls(
            position_xyz,
            quaternion_wxyz,
            position_only=position_only,
        )

    @property
    def cartesian_control_step_limit(self):
        """Published bound for compound sampled-force Cartesian operations."""
        if self.controller != "OSC_POSE" or not self._guarded_cartesian_motion:
            return None
        return self._cartesian_max_steps

    def move_to_pose_bounded(self, position_xyz, quaternion_wxyz, *, max_steps,
                            orientation_reference_quaternion_wxyz):
        """Adapter-owned phase, including a guarded final one-step budget."""
        limit = self.cartesian_control_step_limit
        if limit is None:
            return ActionResult.failed(FailureCategory.NOT_EXECUTED,
                "bounded_cartesian_motion_unavailable")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 0 < max_steps <= limit:
            return ActionResult.failed(FailureCategory.INVALID_ARGUMENT,
                "max_steps must be a positive integer within the configured Cartesian limit")
        # Call the sweep explicitly: _move_to_pose_osc(max_steps=1) is its
        # internal microstep, not a guarded whole-target operation.
        return self._move_to_pose_guarded(position_xyz, quaternion_wxyz,
            max_steps=max_steps,
            orientation_reference_quaternion_wxyz=orientation_reference_quaternion_wxyz)

    def move_to_pose_with_orientation_reference(self, position_xyz, quaternion_wxyz, *,
            orientation_reference_quaternion_wxyz):
        """Internal compound phase: keep the original requested rotation precision."""
        return self._move_to_pose_osc(position_xyz, quaternion_wxyz, position_only=False,
            orientation_reference_quaternion_wxyz=orientation_reference_quaternion_wxyz)

    def _move_to_pose_osc(
        self,
        position_xyz: Sequence[float],
        quaternion_wxyz: Sequence[float],
        *,
        position_only: bool,
        max_steps: int | None = None,
        position_tolerance_m: float | None = None,
        orientation_tolerance_rad: float | None = None,
        precision_reference: str | None = None,
        orientation_reference_quaternion_wxyz=None,
    ) -> ActionResult[None]:
        if getattr(self, '_guarded_cartesian_motion', False) and max_steps != 1:
            return self._move_to_pose_guarded(position_xyz, quaternion_wxyz,
                position_only=position_only, max_steps=max_steps,
                position_tolerance_m=position_tolerance_m,
                orientation_tolerance_rad=orientation_tolerance_rad,
                precision_reference=precision_reference,
                orientation_reference_quaternion_wxyz=orientation_reference_quaternion_wxyz)
        executed = False
        try:
            target_position_base = np.asarray(
                position_xyz,
                dtype=np.float64,
            ).reshape(3)
            target_quaternion_base = np.asarray(
                quaternion_wxyz,
                dtype=np.float64,
            ).reshape(4)
            norm = float(np.linalg.norm(target_quaternion_base))
            if (
                not np.isfinite(target_position_base).all()
                or not np.isfinite(target_quaternion_base).all()
                or norm < 1e-12
            ):
                raise ValueError("OSC target pose must be finite and nonzero")
            target_quaternion_base /= norm
            base_rotation, base_position = self._base_transform()
            target_body_world = np.eye(4, dtype=np.float64)
            target_body_world[:3, :3] = (
                base_rotation
                @ _quaternion_wxyz_to_matrix(target_quaternion_base)
            )
            target_body_world[:3, 3] = (
                base_rotation @ target_position_base + base_position
            )
            target_site_world = (
                target_body_world @ self._eef_body_to_control_site
            )
            step_limit = max_steps or self._cartesian_max_steps
            position_tolerance = (
                position_tolerance_m
                if position_tolerance_m is not None
                else self._osc_position_tolerance_m
            )
            orientation_tolerance = (
                orientation_tolerance_rad
                if orientation_tolerance_rad is not None
                else self._osc_orientation_tolerance_rad
            )
            best_scaled_error = math.inf
            stalled_steps = 0
            for _ in range(step_limit):
                current_site = self._control_site_pose_world()
                position_error = (
                    target_site_world[:3, 3] - current_site[:3, 3]
                )
                orientation_vector = (
                    np.zeros(3, dtype=np.float64)
                    if position_only
                    else _rotation_matrix_to_axis_angle(
                        target_site_world[:3, :3]
                        @ current_site[:3, :3].T
                    )
                )
                position_residual = float(np.linalg.norm(position_error))
                orientation_residual = float(
                    np.linalg.norm(orientation_vector)
                )
                if (
                    position_residual <= position_tolerance
                    and (
                        position_only
                        or orientation_residual <= orientation_tolerance
                    )
                ):
                    return ActionResult.ok(
                        None,
                        "OSC_POSE reached the requested pose" if executed else
                        "OSC_POSE target already within tolerance; no simulation step required",
                    )
                scaled_error = (
                    position_residual / position_tolerance
                    + (
                        0.0
                        if position_only
                        else orientation_residual
                        / self._osc_orientation_tolerance_rad
                    )
                )
                if scaled_error < best_scaled_error - 1e-3:
                    best_scaled_error = scaled_error
                    stalled_steps = 0
                else:
                    stalled_steps += 1
                action = np.zeros(self._action_dim, dtype=np.float64)
                action[:3] = np.clip(
                    position_error / self._osc_output_max[:3],
                    -1.0,
                    1.0,
                )
                if self._controller_action_dim >= 6 and not position_only:
                    action[3:6] = np.clip(
                        orientation_vector / self._osc_output_max[3:6],
                        -1.0,
                        1.0,
                    )
                action[-1] = 1.0 - 2.0 * self._gripper_command_fraction
                if not self._step_action(action):
                    if self._last_step_rejection == 'deadline_expired':
                        return self._deadline_rejection_result(executed)
                    if self._last_step_rejection == 'motion_guard_rejected':
                        return ActionResult.failed(FailureCategory.CONTROLLER_REJECTED,
                            'motion_guard_rejected', executed=executed)
                    category = (
                        FailureCategory.TIMEOUT
                        if executed
                        else FailureCategory.NOT_EXECUTED
                    )
                    return ActionResult.failed(
                        category,
                        "LIBERO horizon exhausted",
                        executed=executed,
                    )
                executed = True
                if stalled_steps >= self._cartesian_stall_steps:
                    velocity = self._eef_linear_velocity_norm()
                    if velocity <= self._osc_settle_velocity_m_s:
                        return ActionResult.failed(
                            FailureCategory.TIMEOUT,
                            "OSC_POSE stalled before the requested pose "
                            f"(position residual {position_residual:.6f} m, "
                            "orientation residual "
                            f"{orientation_residual:.6f} rad)",
                            executed=executed,
                        )
            return ActionResult.failed(
                FailureCategory.TIMEOUT,
                f"OSC_POSE command exceeded {step_limit} control steps",
                executed=executed,
            )
        except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
            return ActionResult.failed(
                FailureCategory.CONTROLLER_REJECTED,
                f"OSC_POSE rejected target: {type(exc).__name__}: {exc}",
                executed=executed,
            )
        except Exception as exc:
            return ActionResult.failed(
                FailureCategory.BACKEND_ERROR,
                f"{type(exc).__name__}: {exc}",
                executed=executed,
            )

    def _eef_linear_velocity_norm(self) -> float:
        robot = self.__env.robots[0]
        recent = getattr(robot, "recent_ee_vel", None)
        current = getattr(recent, "current", None)
        if current is not None:
            values = np.asarray(current, dtype=np.float64).reshape(-1)
            if values.size >= 3 and np.isfinite(values[:3]).all():
                return float(np.linalg.norm(values[:3]))
        velocities = np.asarray(
            self._observation.get("robot0_joint_vel", np.zeros(7)),
            dtype=np.float64,
        ).reshape(-1)
        return float(np.linalg.norm(velocities))

    def _eef_force_world(self):
        """Wrist force sensor vector rotated into the world frame (public sensor only)."""
        robot = self.__env.robots[0]
        fast = getattr(self, "_fast_force", None)
        force = fast.ee_force() if fast is not None else getattr(robot, "ee_force", None)
        if force is None:
            return None
        values = np.asarray(force, dtype=np.float64).reshape(-1)
        if values.size < 3 or not np.isfinite(values[:3]).all():
            return None
        if fast is not None:
            try:
                return np.asarray(fast.site_xmat(), dtype=np.float64).reshape(3, 3) @ values[:3]
            except Exception:
                return values[:3]
        try:
            name = robot.gripper.important_sensors["force_ee"]
            sim = self.__env.sim
            model = getattr(sim.model, "_model", sim.model)
            data = getattr(sim.data, "_data", sim.data)
            sensor_id = sim.model.sensor_name2id(name)
            site = int(np.asarray(model.sensor_objid)[sensor_id])
            rotation = np.asarray(data.site_xmat[site], dtype=np.float64).reshape(3, 3)
            return rotation @ values[:3]
        except Exception:
            return values[:3]

    def _controller_gains(self):
        controller = getattr(self.__env.robots[0], "controller", None)
        try:
            return {"kp": np.asarray(controller.kp, dtype=np.float64).reshape(-1)[:3].tolist(),
                    "kd": np.asarray(controller.kd, dtype=np.float64).reshape(-1)[:3].tolist()}
        except Exception:
            return {"kp": None, "kd": None}

    def _osc_expected_force_world(self):
        """kp*(goal - ee) - kd*ee_vel: the operational-space controller's own commanded force (world frame)."""
        controller = getattr(self.__env.robots[0], "controller", None)
        try:
            kp = np.asarray(controller.kp, dtype=np.float64).reshape(-1)[:3]
            kd = np.asarray(controller.kd, dtype=np.float64).reshape(-1)[:3]
            return kp * (np.asarray(controller.goal_pos, dtype=np.float64) - np.asarray(controller.ee_pos, dtype=np.float64)) \
                - kd * np.asarray(controller.ee_pos_vel, dtype=np.float64)
        except Exception:
            return np.zeros(3)

    def _contact_sample(self):
        force = self._eef_force_world()
        if force is None:
            return None
        sample = {"f_world": force, "f_osc": self._osc_expected_force_world(),
                  "u_world": self._contact_direction_world}
        if getattr(self, "_velocity_samples", False):
            # r6 (opt-in): grip-site displacement over the last native tick (m/tick), for the velocity force model
            try:
                p = np.asarray(self._control_site_pose_world(), float)[:3, 3]
                s = int(self.simulation_steps)
                prev = getattr(self, "_vel_prev", None)
                if prev is None or s < prev[0]:
                    self._vel_prev, self._vel_last = (s, p), np.zeros(3)
                elif s > prev[0]:
                    self._vel_last = (p - prev[1]) / float(s - prev[0])
                    self._vel_prev = (s, p)
                sample["v_tick"] = getattr(self, "_vel_last", np.zeros(3))
            except Exception:  # noqa: BLE001
                pass
        return sample

    def make_contact_detector(self, contact_mode="guarded", load_budget_n=None):
        """Residual detector for one physical request, or None when the legacy fixed guard applies."""
        if self._contact_detector_mode != "residual" or self._force_calibration is None:
            return None
        from .contact_residual import ResidualContactDetector
        return ResidualContactDetector(self._force_calibration, mode=contact_mode, load_budget_n=load_budget_n,
                                       dt_s=1.0 / (self._control_frequency * getattr(self, "_force_samples_per_tick", 10.0)))

    def calibrate_contact_residual(self):
        from .contact_residual import calibrate_free_space
        return calibrate_free_space(self)

    def _eef_force_norm(self) -> float | None:
        fast = getattr(self, "_fast_force", None)
        force = fast.ee_force() if fast is not None else getattr(self.__env.robots[0], "ee_force", None)
        if force is None:
            return None
        values = np.asarray(force, dtype=np.float64).reshape(-1)
        if values.size < 3 or not np.isfinite(values[:3]).all():
            return None
        return float(np.linalg.norm(values[:3]))

    def gripper_state(self) -> ActionResult[dict[str, object]]:
        try:
            force = self._eef_force_norm()
            open_fraction = self._read_gripper_fraction()
            object_detected = bool(
                self._gripper_command_fraction <= 0.02
                and open_fraction >= 0.02
            )
            return ActionResult.ok(
                {
                    "open_fraction": open_fraction,
                    "object_detected": object_detected,
                    "force_n": force,
                    "force_available": force is not None,
                }
            )
        except Exception as exc:
            return ActionResult.failed(
                FailureCategory.BACKEND_ERROR,
                f"{type(exc).__name__}: {exc}",
            )

    def approach_and_close(
        self,
        position: Sequence[float],
        quaternion: Sequence[float],
        approach_direction_world: Sequence[float],
        *,
        max_contact_force: float | None = None,
        approach_distance_m: float = 0.06,
    ) -> ActionResult[dict[str, object]]:
        aligned = self.align_pregrasp(
            position,
            quaternion,
            approach_direction_world,
            approach_distance_m=approach_distance_m,
        )
        if not aligned.success:
            return aligned
        closed = self.descend_and_close(
            position,
            quaternion,
            max_contact_force=max_contact_force,
        )
        if not closed.success or not isinstance(closed.value, Mapping):
            return closed
        return ActionResult.ok(
            {
                **dict(closed.value),
                "pregrasp_alignment": dict(aligned.value or {}),
            },
            closed.message,
        )

    def align_pregrasp(
        self,
        position: Sequence[float],
        quaternion: Sequence[float],
        approach_direction_world: Sequence[float],
        *,
        approach_distance_m: float = 0.06,
        clearance_m: float = 0.09,
    ) -> ActionResult[dict[str, object]]:
        if self._controller != "OSC_POSE":
            return ActionResult.failed(
                FailureCategory.CONTROLLER_REJECTED,
                "align_pregrasp requires OSC_POSE",
            )
        executed = False
        try:
            target = np.asarray(position, dtype=np.float64).reshape(3)
            quat = np.asarray(quaternion, dtype=np.float64).reshape(4)
            approach = np.asarray(
                approach_direction_world,
                dtype=np.float64,
            ).reshape(3)
            approach_norm = float(np.linalg.norm(approach))
            if (
                not np.isfinite(target).all()
                or not np.isfinite(quat).all()
                or not np.isfinite(approach).all()
                or approach_norm < 1e-9
            ):
                raise ValueError("approach candidate must be finite")
            if (
                not math.isfinite(float(approach_distance_m))
                or not 0.005 <= float(approach_distance_m) <= 0.20
            ):
                raise ValueError(
                    "approach_distance_m must be in [0.005, 0.20]"
                )
            if (
                not math.isfinite(float(clearance_m))
                or not 0.02 <= float(clearance_m) <= 0.20
            ):
                raise ValueError("clearance_m must be in [0.02, 0.20]")
            quat_norm = float(np.linalg.norm(quat))
            if quat_norm < 1e-12:
                raise ValueError("approach quaternion must be nonzero")
            quat /= quat_norm
            approach /= approach_norm
            pregrasp = target - approach * float(approach_distance_m)
            current_state = self._robot_state()
            current_position = np.asarray(
                current_state.end_effector_position_m,
                dtype=np.float64,
            )
            current_quaternion = np.asarray(
                current_state.end_effector_quaternion_wxyz,
                dtype=np.float64,
            )
            hover = pregrasp.copy()
            hover[2] = max(
                float(pregrasp[2]) + float(clearance_m),
                float(current_position[2]),
            )
            opened = self._set_gripper(1.0)
            executed = opened.executed
            if not opened.success:
                return ActionResult.failed(
                    opened.failure_category,
                    opened.message,
                    executed=executed,
                )
            clearance = current_position.copy()
            clearance[2] = hover[2]
            # Leave low/contact poses vertically before any lateral movement.
            # This uses the existing hover height and phase tolerance; it is not
            # a collision-free path planner or a new object-specific waypoint.
            clearance_phases = ((
                "clearance", clearance, current_quaternion, True,
                _ALIGN_TRANSLATE_TOLERANCE_M, None,
                _ALIGN_TRANSLATE_ACCEPT_TOLERANCE_M,
            ),) if clearance[2] - current_position[2] > _ALIGN_TRANSLATE_TOLERANCE_M else ()
            phases = clearance_phases + (
                (
                    "translate",
                    hover,
                    current_quaternion,
                    True,
                    _ALIGN_TRANSLATE_TOLERANCE_M,
                    None,
                    _ALIGN_TRANSLATE_ACCEPT_TOLERANCE_M,
                ),
                (
                    "rotate",
                    hover,
                    quat,
                    False,
                    _ALIGN_ROTATE_POSITION_SLACK_M,
                    _ALIGN_ROTATE_ORIENTATION_TOLERANCE_RAD,
                    _ALIGN_ROTATE_POSITION_SLACK_M,
                ),
                (
                    "settle",
                    hover,
                    quat,
                    True,
                    _ALIGN_SETTLE_TOLERANCE_M,
                    None,
                    _ALIGN_SETTLE_ACCEPT_TOLERANCE_M,
                ),
                (
                    "descend_to_pregrasp",
                    pregrasp,
                    quat,
                    True,
                    _ALIGN_DESCEND_TOLERANCE_M,
                    None,
                    _ALIGN_DESCEND_TOLERANCE_M,
                ),
            )
            completed: list[str] = []
            phase_executions = []
            for (
                phase,
                phase_position,
                phase_quaternion,
                position_only,
                phase_position_tolerance,
                phase_orientation_tolerance,
                phase_accept_tolerance,
            ) in phases:
                moved = self._move_to_pose_osc(
                    phase_position,
                    phase_quaternion,
                    position_only=position_only,
                    position_tolerance_m=phase_position_tolerance,
                    orientation_tolerance_rad=phase_orientation_tolerance,
                    precision_reference='pregrasp_phase',
                )
                phase_executions.append({'phase': phase, 'success': moved.success,
                    'executed': moved.executed, 'message': moved.message, 'value': moved.value})
                executed = executed or moved.executed
                position_residual, orientation_residual = (
                    self._pose_residuals(
                        phase_position,
                        phase_quaternion,
                    )
                )
                if not moved.success:
                    # A stall whose end state already satisfies the phase
                    # accept threshold is convergence: waypoint phases carry
                    # looser accept limits than their in-loop targets because
                    # the descend gate enforces the final precision.
                    if phase == "rotate":
                        within_phase_tolerance = (
                            orientation_residual
                            <= _ALIGN_ROTATE_ACCEPT_TOLERANCE_RAD
                        )
                    else:
                        within_phase_tolerance = (
                            position_residual <= phase_accept_tolerance
                        )
                    ordinary_timeout = (moved.failure_category is FailureCategory.TIMEOUT
                        and (not isinstance(moved.value, Mapping)
                             or moved.value.get('stopped_reason') in {None, 'motion_stalled', 'duration_exhausted'}))
                    if within_phase_tolerance and ordinary_timeout:
                        completed.append(phase)
                        continue
                    payload = {
                        'phase_executions': phase_executions,
                        'controller_execution': moved.value,
                        "converged": False,
                        "phase_reached": completed[-1] if completed else "none",
                        "failed_phase": phase,
                        "position_residual_m": position_residual,
                        "orientation_residual_rad": orientation_residual,
                        "convergence_reason": moved.message,
                        "pregrasp_position_m": [
                            float(value) for value in pregrasp
                        ],
                        "hover_position_m": [
                            float(value) for value in hover
                        ],
                    }
                    return ActionResult.failed(
                        moved.failure_category,
                        f"align_pregrasp {phase} failed "
                        f"(position residual {position_residual:.6f} m, "
                        "orientation residual "
                        f"{orientation_residual:.6f} rad): {moved.message}",
                        executed=executed,
                        value=payload,
                    )
                completed.append(phase)
            position_residual, orientation_residual = self._pose_residuals(
                pregrasp,
                quat,
            )
            return ActionResult.ok(
                {
                    "converged": True,
                    "phase_reached": "pregrasp",
                    'phase_executions': phase_executions,
                    "failed_phase": None,
                    "position_residual_m": position_residual,
                    "orientation_residual_rad": orientation_residual,
                    "convergence_reason": "staged_pregrasp_reached",
                    "pregrasp_position_m": [
                        float(value) for value in pregrasp
                    ],
                    "hover_position_m": [
                        float(value) for value in hover
                    ],
                },
                "staged pregrasp alignment converged",
            )
        except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
            return ActionResult.failed(
                FailureCategory.INVALID_ARGUMENT,
                f"{type(exc).__name__}: {exc}",
                executed=executed,
            )
        except Exception as exc:
            return ActionResult.failed(
                FailureCategory.BACKEND_ERROR,
                f"{type(exc).__name__}: {exc}",
                executed=executed,
            )

    def descend_and_close(
        self,
        position: Sequence[float],
        quaternion: Sequence[float],
        *,
        max_contact_force: float | None = None,
        position_tolerance_m: float | None = None,
    ) -> ActionResult[dict[str, object]]:
        if self._controller != "OSC_POSE":
            return ActionResult.failed(
                FailureCategory.CONTROLLER_REJECTED,
                "descend_and_close requires OSC_POSE",
            )
        executed = False
        try:
            from captain_harness.adapters.robotics.contact_precision import contact_position_tolerance
            tolerance = contact_position_tolerance(position_tolerance_m)
            target = np.asarray(position, dtype=np.float64).reshape(3)
            quat = np.asarray(quaternion, dtype=np.float64).reshape(4)
            if not np.isfinite(target).all() or not np.isfinite(quat).all():
                raise ValueError("descent target must be finite")
            quat_norm = float(np.linalg.norm(quat))
            if quat_norm < 1e-12:
                raise ValueError("descent quaternion must be nonzero")
            quat /= quat_norm
            force_limit = (
                float(max_contact_force)
                if max_contact_force is not None
                else 18.0
            )
            if not math.isfinite(force_limit) or force_limit <= 0:
                raise ValueError("max_contact_force must be finite and positive")
            baseline_force = self._eef_force_norm()
            start_position = np.asarray(
                self._robot_state().end_effector_position_m,
                dtype=np.float64,
            )
            previous_position = start_position.copy()
            stalled = 0
            contacted = False
            stopped_reason = "target_reached"
            for _ in range(self._cartesian_max_steps):
                state = self._robot_state()
                current = np.asarray(
                    state.end_effector_position_m,
                    dtype=np.float64,
                )
                remaining = target - current
                if float(np.linalg.norm(remaining)) <= tolerance:
                    stopped_reason = "target_reached"
                    break
                body_target = current + remaining * min(
                    1.0,
                    0.004 / max(float(np.linalg.norm(remaining)), 1e-9),
                )
                step_result = self._move_to_pose_osc(
                    body_target,
                    quat,
                    position_only=False,
                    max_steps=1,
                    position_tolerance_m=tolerance,
                )
                executed = executed or step_result.executed
                if (not step_result.success
                        and step_result.failure_category is FailureCategory.TIMEOUT
                        and (step_result.diagnostics or {}).get("deadline")):
                    # A one-cycle nonconvergence timeout is expected; a shared
                    # deadline is terminal and must not become a position stall.
                    observed = self._robot_state()
                    final_position = np.asarray(observed.end_effector_position_m)
                    residual = target - final_position
                    return replace(step_result, executed=executed, value={
                        "failed_phase": "descent", "stopped_reason": "deadline_expired",
                        "closure_executed": False, "grasp_outcome": "not_attempted",
                        "observed_eef_position_m": final_position.tolist(),
                        "requested_eef_position_m": target.tolist(),
                        "position_residual_m": float(np.linalg.norm(residual)),
                        "remaining_vector_m": residual.tolist(),
                        "contact_travel_m": float(np.linalg.norm(final_position-start_position)),
                        "gripper_open_fraction": float(observed.gripper_open_fraction),
                        "observed_force_n": self._eef_force_norm(),
                        "force_limit_n": force_limit, "baseline_force_n": baseline_force,
                        "contact_identity": "unconfirmed",
                    })
                current_force = self._eef_force_norm()
                if (
                    current_force is not None and (
                        current_force >= force_limit
                        or (baseline_force is not None and current_force - baseline_force >= 5.0))
                ):
                    contacted = True
                    stopped_reason = "contact_force"
                    break
                updated = np.asarray(
                    self._robot_state().end_effector_position_m,
                    dtype=np.float64,
                )
                # Small impedance-control steps can still converge on a fine
                # endpoint. Stop on small motion without meaningful progress,
                # not on displacement alone; force/deadline gates stay above.
                progress = float(np.linalg.norm(remaining) - np.linalg.norm(target - updated))
                converging = progress > max(1e-7, .001 * tolerance)
                if float(np.linalg.norm(updated - previous_position)) < 0.0002 and not converging:
                    stalled += 1
                else:
                    stalled = 0
                previous_position = updated
                if stalled >= 3:
                    stopped_reason = "position_stall"
                    break
                if (
                    not step_result.success
                    and step_result.failure_category
                    not in {FailureCategory.TIMEOUT}
                ):
                    return ActionResult.failed(
                        step_result.failure_category,
                        step_result.message,
                        executed=executed,
                    )
            else:
                return ActionResult.failed(
                    FailureCategory.TIMEOUT,
                    "descend_and_close exceeded the OSC descent budget",
                    executed=executed,
                )
            descent_state = self._robot_state()
            descent_position = np.asarray(descent_state.end_effector_position_m)
            residual = target - descent_position
            descent_evidence = {
                "requested_position_tolerance_m": tolerance,
                "contacted": contacted,
                "stopped_reason": stopped_reason,
                "contact_travel_m": float(np.linalg.norm(descent_position - start_position)),
                "position_residual_m": float(np.linalg.norm(residual)),
                "remaining_vector_m": residual.tolist(),
                "observed_eef_position_m": descent_position.tolist(),
                "requested_eef_position_m": target.tolist(),
                "force_limit_n": force_limit,
                "force_increase_limit_n": 5.0,
                "baseline_force_n": baseline_force,
                "observed_force_n": self._eef_force_norm(),
                "contact_identity": "unconfirmed" if contacted else "not_measured",
            }
            if stopped_reason != "target_reached" and np.linalg.norm(residual) > tolerance:
                return ActionResult.failed(
                    FailureCategory.CONTROLLER_REJECTED,
                    f"descent_obstructed_before_target: stopped_reason={stopped_reason}; closure not executed; "
                    "inspect measured progress, the remaining vector and pre/post wrist images "
                    "before selecting a corrective action. A stop does not identify contacting geometry.",
                    executed=executed,
                    value={**descent_evidence, "reason": "descent_obstructed_before_target",
                           "closure_executed": False, "grasp_outcome": "not_attempted",
                           "gripper_open_fraction": float(descent_state.gripper_open_fraction)},
                )
            closed = self._set_gripper(0.0)
            executed = executed or closed.executed
            if not closed.success:
                # Closure can disturb the arm. Keep descent provenance while
                # reporting the measured pose after the interrupted closure.
                observed = self._robot_state()
                final_position = np.asarray(observed.end_effector_position_m)
                residual = target - final_position
                return replace(closed, executed=executed, value={
                    **descent_evidence, "failed_phase": "closure",
                    "descent_stopped_reason": descent_evidence["stopped_reason"],
                    "stopped_reason": ("deadline_expired"
                        if (closed.diagnostics or {}).get("deadline") else "closure_failed"),
                    "closure_executed": closed.executed,
                    "grasp_outcome": "closure_interrupted",
                    "observed_eef_position_m": final_position.tolist(),
                    "position_residual_m": float(np.linalg.norm(residual)),
                    "remaining_vector_m": residual.tolist(),
                    "gripper_open_fraction": float(observed.gripper_open_fraction),
                    "observed_force_n": self._eef_force_norm(),
                    "measurement_phase": "after_interrupted_closure",
                })
            gripper = self.gripper_state()
            final_state = self._robot_state()
            final_position = np.asarray(final_state.end_effector_position_m, dtype=np.float64)
            final_evidence = {
                **descent_evidence,
                "descent_execution": dict(descent_evidence),
                "contact_precision": {
                    "requested_position_tolerance_m": tolerance,
                    "preclosure_endpoint_within_tolerance": bool(descent_evidence["position_residual_m"] <= tolerance),
                    "postclosure_endpoint_within_tolerance": bool(np.linalg.norm(target-final_position) <= tolerance),
                    "scope": "Requested measured pre-closure EEF endpoint tolerance. Post-closure drift is reported separately; neither value verifies object contact or retention."},
                "closure_executed": closed.executed,
                "observed_eef_position_m": final_position.tolist(),
                "position_residual_m": float(np.linalg.norm(target - final_position)),
                "remaining_vector_m": (target - final_position).tolist(),
                "contact_travel_m": float(np.linalg.norm(final_position - start_position)),
                "gripper_open_fraction": float(final_state.gripper_open_fraction),
                "observed_force_n": self._eef_force_norm(),
                "measurement_phase": "after_closure",
            }
            if not gripper.success or not isinstance(gripper.value, Mapping):
                # A sensor failure after physical closure must keep its receipt
                # and the measured final arm pose, not erase the completed prefix.
                failed = gripper if not gripper.success else replace(gripper,
                    success=False, failure_category=FailureCategory.BACKEND_ERROR,
                    message="post-closure gripper measurement is not a mapping")
                return replace(failed, executed=executed, value={
                    **final_evidence, "failed_phase": "post_closure_measurement",
                    "stopped_reason": "gripper_measurement_failed",
                    "grasp_outcome": "unverified",
                    "closure_execution": {
                        "success": closed.success, "executed": closed.executed,
                        "failure_category": closed.failure_category.value,
                        "message": closed.message, "value": closed.value,
                        "diagnostics": closed.diagnostics,
                    },
                })
            target_quaternion = np.asarray(
                quaternion,
                dtype=np.float64,
            )
            target_quaternion /= np.linalg.norm(target_quaternion)
            final_quaternion = np.asarray(
                final_state.end_effector_quaternion_wxyz,
                dtype=np.float64,
            )
            return ActionResult.ok(
                {
                    **final_evidence,
                    "grasp_outcome": "object_detected" if gripper.value["object_detected"] else "empty_close",
                    "contacted": contacted,
                    "contact_travel_m": float(
                        np.linalg.norm(final_position - start_position)
                    ),
                    "gripper_open_fraction": float(
                        gripper.value["open_fraction"]
                    ),
                    "object_detected": bool(
                        gripper.value["object_detected"]
                    ),
                    "stopped_reason": stopped_reason,
                    "position_residual_m": float(
                        np.linalg.norm(final_position - target)
                    ),
                    "orientation_residual_rad": float(
                        np.linalg.norm(
                            _quaternion_rotation_error(
                                target_quaternion,
                                final_quaternion,
                            )
                        )
                    ),
                }
            )
        except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
            return ActionResult.failed(
                FailureCategory.INVALID_ARGUMENT,
                f"{type(exc).__name__}: {exc}",
                executed=executed,
            )
        except Exception as exc:
            return ActionResult.failed(
                FailureCategory.BACKEND_ERROR,
                f"{type(exc).__name__}: {exc}",
                executed=executed,
            )

    def _pose_residuals(
        self,
        position: Sequence[float],
        quaternion: Sequence[float],
    ) -> tuple[float, float]:
        state = self._robot_state()
        target_position = np.asarray(position, dtype=np.float64).reshape(3)
        target_quaternion = np.asarray(
            quaternion,
            dtype=np.float64,
        ).reshape(4)
        target_quaternion /= np.linalg.norm(target_quaternion)
        current_position = np.asarray(
            state.end_effector_position_m,
            dtype=np.float64,
        )
        current_quaternion = np.asarray(
            state.end_effector_quaternion_wxyz,
            dtype=np.float64,
        )
        return (
            float(np.linalg.norm(current_position - target_position)),
            float(
                np.linalg.norm(
                    _quaternion_rotation_error(
                        target_quaternion,
                        current_quaternion,
                    )
                )
            ),
        )

    def servo_to(
        self,
        position: Sequence[float],
        quaternion: Sequence[float],
        *,
        max_duration_s: float,
        tolerance_m: float,
    ) -> ActionResult[dict[str, object]]:
        if self._controller != "OSC_POSE":
            return ActionResult.failed(
                FailureCategory.CONTROLLER_REJECTED,
                "servo_to requires OSC_POSE",
            )
        if (
            not math.isfinite(float(max_duration_s))
            or not 0 < float(max_duration_s) <= 30.0
            or not math.isfinite(float(tolerance_m))
            or not 0 < float(tolerance_m) <= 0.10
        ):
            return ActionResult.invalid_argument(
                "max_duration_s must be in (0, 30] and tolerance_m in (0, 0.10]"
            )
        max_steps = max(
            1,
            int(math.ceil(float(max_duration_s) * self._control_frequency)),
        )
        start_steps = self._sim_steps
        result = self._move_to_pose_osc(
            position,
            quaternion,
            position_only=False,
            max_steps=max_steps,
            position_tolerance_m=float(tolerance_m),
        )
        state = self.get_robot_state()
        final_position = (
            np.asarray(state.value.end_effector_position_m, dtype=np.float64)
            if state.success and state.value is not None
            else np.zeros(3, dtype=np.float64)
        )
        target = np.asarray(position, dtype=np.float64).reshape(3)
        payload = {
            "reached": bool(
                result.success
                and float(np.linalg.norm(final_position - target))
                <= float(tolerance_m)
            ),
            "final_error_m": float(np.linalg.norm(final_position - target)),
            "iterations": min(
                max_steps,
                max(0, self._sim_steps - start_steps),
            ),
            "aborted_reason": (
                None if result.success else result.failure_category.value
            ),
        }
        if isinstance(result.value, Mapping):
            payload['controller_execution'] = dict(result.value)
        if result.success:
            return ActionResult.ok(payload, result.message)
        return ActionResult.failed(
            result.failure_category,
            result.message,
            executed=result.executed,
            value=payload,
        )

    def _move_to_pose_dls(
        self,
        position_xyz,
        quaternion_wxyz,
        *,
        position_only: bool = False,
    ) -> ActionResult[None]:
        executed = False
        try:
            target_position = np.asarray(position_xyz, dtype=np.float64).reshape(3)
            target_quaternion = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
            quaternion_norm = float(np.linalg.norm(target_quaternion))
            if not np.isfinite(target_position).all() or not math.isfinite(quaternion_norm):
                raise ValueError("Cartesian target must be finite")
            if quaternion_norm < 1e-12:
                raise ValueError("target quaternion has zero norm")
            target_quaternion /= quaternion_norm
            best_scaled_error = math.inf
            stalled_steps = 0
            for _ in range(self._cartesian_max_steps):
                state = self._robot_state()
                position_error = target_position - np.asarray(state.end_effector_position_m)
                orientation_error = _quaternion_rotation_error(
                    target_quaternion, np.asarray(state.end_effector_quaternion_wxyz)
                )
                position_norm = float(np.linalg.norm(position_error))
                orientation_norm = float(np.linalg.norm(orientation_error))
                reached = (
                    position_norm <= self._cartesian_position_tolerance_m
                    if position_only
                    else (
                        position_norm <= self._cartesian_position_tolerance_m
                        and orientation_norm
                        <= self._cartesian_orientation_tolerance_rad
                    )
                )
                if reached:
                    if executed:
                        return ActionResult.ok(None)
                    if not self._step(np.zeros(7), self._gripper_command_fraction):
                        if self._last_step_rejection == "deadline_expired":
                            return self._deadline_rejection_result(False)
                        if self._last_step_rejection == "motion_guard_rejected":
                            return ActionResult.failed(FailureCategory.CONTROLLER_REJECTED,
                                "motion_guard_rejected", executed=False)
                        return ActionResult.failed(
                            FailureCategory.NOT_EXECUTED, "LIBERO horizon exhausted"
                        )
                    return ActionResult.ok(None)

                scaled_error = (
                    position_norm / self._cartesian_position_tolerance_m
                    if position_only
                    else (
                        position_norm / self._cartesian_position_tolerance_m
                        + orientation_norm
                        / self._cartesian_orientation_tolerance_rad
                    )
                )
                if scaled_error < best_scaled_error - 1e-3:
                    best_scaled_error = scaled_error
                    stalled_steps = 0
                else:
                    stalled_steps += 1
                    if stalled_steps >= self._cartesian_stall_steps:
                        return ActionResult.failed(
                            FailureCategory.CONTROLLER_REJECTED,
                            "Cartesian IK stalled before reaching the requested pose",
                            executed=executed,
                        )

                data = self.__env.sim.data
                base_rotation, _ = self._base_transform()
                jacobian_position = np.asarray(
                    data.get_body_jacp("gripper0_eef"), dtype=np.float64
                )[:, self._joint_velocity_indexes]
                jacobian_rotation = np.asarray(
                    data.get_body_jacr("gripper0_eef"), dtype=np.float64
                )[:, self._joint_velocity_indexes]
                if position_only:
                    jacobian = base_rotation.T @ jacobian_position
                    error = position_error
                else:
                    jacobian = np.vstack([
                        base_rotation.T @ jacobian_position,
                        base_rotation.T @ jacobian_rotation,
                    ])
                    error = np.concatenate(
                        [position_error, orientation_error]
                    )
                regularized = (jacobian @ jacobian.T +
                               self._cartesian_damping ** 2 *
                               np.eye(3 if position_only else 6))
                joint_delta = jacobian.T @ np.linalg.solve(regularized, error)
                max_delta = float(np.max(np.abs(joint_delta)))
                if max_delta > self._cartesian_joint_step_rad:
                    joint_delta *= self._cartesian_joint_step_rad / max_delta
                current_joints = self._read_joints()
                lower = self._joint_limits[:, 0] + 1e-4
                upper = self._joint_limits[:, 1] - 1e-4
                joint_target = np.clip(current_joints + joint_delta, lower, upper)
                commanded_delta = joint_target - current_joints
                if float(np.linalg.norm(commanded_delta)) < 1e-9:
                    return ActionResult.failed(
                        FailureCategory.CONTROLLER_REJECTED,
                        "Cartesian IK stalled at a singularity or joint limit",
                        executed=executed,
                    )
                if not self._step(commanded_delta, self._gripper_command_fraction):
                    if self._last_step_rejection == "deadline_expired":
                        return self._deadline_rejection_result(executed)
                    if self._last_step_rejection == "motion_guard_rejected":
                        return ActionResult.failed(FailureCategory.CONTROLLER_REJECTED,
                            "motion_guard_rejected", executed=executed)
                    category = FailureCategory.TIMEOUT if executed else FailureCategory.NOT_EXECUTED
                    return ActionResult.failed(
                        category, "LIBERO horizon exhausted", executed=executed
                    )
                executed = True
        except (ValueError, np.linalg.LinAlgError) as exc:
            return ActionResult.failed(
                FailureCategory.CONTROLLER_REJECTED,
                f"Cartesian IK failed: {type(exc).__name__}: {exc}",
                executed=executed,
            )
        except Exception as exc:
            return ActionResult.failed(
                FailureCategory.BACKEND_ERROR, f"{type(exc).__name__}: {exc}", executed=executed
            )
        return ActionResult.failed(
            FailureCategory.TIMEOUT,
            f"Cartesian target did not converge in {self._cartesian_max_steps} steps",
            executed=executed,
        )

    def move_by(self, delta_xyz) -> ActionResult[None]:
        state = self.get_robot_state()
        if not state.success or state.value is None:
            return ActionResult.failed(state.failure_category, state.message, executed=state.executed)
        target = np.asarray(state.value.end_effector_position_m) + np.asarray(delta_xyz)
        return self.move_to_pose(target, state.value.end_effector_quaternion_wxyz)

    def pregrasp_pose_status(
        self, position: Sequence[float], quaternion: Sequence[float],
    ) -> ActionResult[dict[str, object]]:
        """Recheck the original approach pose with existing alignment tolerances."""
        try:
            position_error, orientation_error = self._pose_residuals(position, quaternion)
            return ActionResult.ok({
                "valid": bool(position_error <= _ALIGN_DESCEND_TOLERANCE_M
                    and orientation_error <= _ALIGN_ROTATE_ACCEPT_TOLERANCE_RAD),
                "position_residual_m": position_error,
                "orientation_residual_rad": orientation_error,
                "position_tolerance_m": _ALIGN_DESCEND_TOLERANCE_M,
                "orientation_tolerance_rad": _ALIGN_ROTATE_ACCEPT_TOLERANCE_RAD,
            })
        except Exception as exc:
            return ActionResult.failed(FailureCategory.BACKEND_ERROR, str(exc))

    def _osc_hold_action(self, target_site: np.ndarray) -> np.ndarray:
        """Hold one fixed control-site pose, rather than re-anchor every step.

        OSC consumes deltas relative to the measured pose. Repeated zero deltas
        therefore preserve accumulated drift, including during gripper closure.
        Use the same public pose feedback/scaling as a Cartesian move.
        """
        current = self._control_site_pose_world()
        action = np.zeros(self._action_dim, dtype=np.float64)
        action[:3] = np.clip(
            (target_site[:3, 3] - current[:3, 3]) / self._osc_output_max[:3], -1., 1.)
        if self._controller_action_dim >= 6:
            rotation_error = _rotation_matrix_to_axis_angle(
                target_site[:3, :3] @ current[:3, :3].T)
            action[3:6] = np.clip(rotation_error / self._osc_output_max[3:6], -1., 1.)
        action[-1] = 1.0 - 2.0 * self._gripper_command_fraction
        return action

    def hold_position(self, duration_s: float = 0.5) -> ActionResult[dict[str, object]]:
        """Advance the public dynamics clock without a new arm/gripper target."""
        if (isinstance(duration_s, bool) or not isinstance(duration_s, (int, float))
                or not math.isfinite(duration_s) or not 0 < duration_s <= 30.0):
            return ActionResult.failed(FailureCategory.INVALID_ARGUMENT,
                "duration_s must be finite and in (0, 30]")
        start_steps = self._sim_steps
        requested_steps = int(math.ceil(duration_s * self._control_frequency))
        hold_site = None
        hold_state = None
        def evidence():
            value = {"requested_duration_s": float(duration_s),
                    "elapsed_s": (self._sim_steps - start_steps) / self._control_frequency,
                    "start_time_s": start_steps / self._control_frequency,
                    "end_time_s": self._sim_steps / self._control_frequency,
                    "steps_executed": self._sim_steps - start_steps,
                    "gripper_command_fraction": self._gripper_command_fraction}
            if hold_state is not None:
                try:
                    position_error, orientation_error = self._pose_residuals(
                        hold_state.end_effector_position_m, hold_state.end_effector_quaternion_wxyz)
                    value["pose_hold"] = {
                        "reference": "end_effector",
                        "position_residual_m": position_error,
                        "orientation_residual_rad": orientation_error,
                        "position_tolerance_m": self._osc_position_tolerance_m,
                        "orientation_tolerance_rad": self._osc_orientation_tolerance_rad,
                        "within_tolerance": bool(position_error <= self._osc_position_tolerance_m
                            and orientation_error <= self._osc_orientation_tolerance_rad),
                    }
                except Exception:
                    value["pose_hold"] = {"reference": "end_effector", "within_tolerance": False}
            return value
        try:
            if self._controller == "OSC_POSE":
                hold_site = self._control_site_pose_world().copy()
                hold_state = self._robot_state()
            for _ in range(requested_steps):
                if self._controller == "OSC_POSE":
                    stepped = self._step_action(self._osc_hold_action(hold_site))
                else:
                    stepped = self._step(np.zeros(7), self._gripper_command_fraction)
                if not stepped:
                    executed = self._sim_steps > start_steps
                    if self._last_step_rejection == 'deadline_expired':
                        return self._deadline_rejection_result(executed, evidence())
                    if self._last_step_rejection == "motion_guard_rejected":
                        return ActionResult.failed(FailureCategory.CONTROLLER_REJECTED,
                            "motion_guard_rejected", executed=executed, value=evidence())
                    return ActionResult.failed(FailureCategory.TIMEOUT if executed else FailureCategory.NOT_EXECUTED,
                        "environment horizon exhausted during time advance", executed=executed, value=evidence())
            return ActionResult.ok(evidence(), "Requested dynamics time advanced; object stability is not asserted.")
        except Exception as exc:
            return ActionResult.failed(FailureCategory.BACKEND_ERROR,
                f"{type(exc).__name__}: {exc}", executed=self._sim_steps > start_steps, value=evidence())

    def _set_gripper(self, fraction: float) -> ActionResult[None]:
        executed = False
        previous_command = self._gripper_command_fraction
        try:
            self._gripper_command_fraction = fraction
            hold_site = (self._control_site_pose_world().copy()
                         if self._controller == "OSC_POSE" else None)
            previous_aperture, still = None, 0
            for _ in range(self._gripper_steps):
                if self._controller == "OSC_POSE":
                    stepped = self._step_action(self._osc_hold_action(hold_site))
                else:
                    stepped = self._step(np.zeros(7), fraction)
                if not stepped:
                    if not executed:
                        self._gripper_command_fraction = previous_command
                    if self._last_step_rejection == "deadline_expired":
                        return self._deadline_rejection_result(executed)
                    if self._last_step_rejection == "motion_guard_rejected":
                        return ActionResult.failed(FailureCategory.CONTROLLER_REJECTED,
                            "motion_guard_rejected", executed=executed)
                    category = FailureCategory.TIMEOUT if executed else FailureCategory.NOT_EXECUTED
                    return ActionResult.failed(category, "LIBERO horizon exhausted", executed=executed)
                executed = True
                if self._gripper_settle_min_steps is not None:
                    aperture = self.get_robot_state().value.gripper_open_fraction
                    still = still + 1 if previous_aperture is not None and abs(aperture - previous_aperture) < 2e-3 else 0
                    previous_aperture = aperture
                    near = abs(aperture - fraction) <= getattr(self, '_gripper_near_target', 0.0)
                    if _ + 1 >= self._gripper_settle_min_steps and (still >= getattr(self, '_gripper_still_needed', 3) or near):
                        break
            self._gripper_open_fraction = fraction
            return ActionResult.ok(None)
        except Exception as exc:
            if not executed:
                self._gripper_command_fraction = previous_command
            return ActionResult.failed(
                FailureCategory.BACKEND_ERROR, f"{type(exc).__name__}: {exc}", executed=executed
            )

    def _bounded_gripper_opening(self, max_open_fraction_increase):
        from .gripper_target import bounded_opening_commands
        return bounded_opening_commands(self.__env.robots[0].gripper, max_open_fraction_increase)

    def release_opening(self, max_duration_s, max_open_fraction_increase):
        """Native opening cycles only; caller installs the constrained load guard."""
        initial = self.get_robot_state().value
        aperture = initial.gripper_open_fraction
        if aperture is None or not math.isfinite(aperture):
            return ActionResult.failed(FailureCategory.NOT_EXECUTED, "gripper_measurement_invalid")
        if aperture >= 1. - 1e-6:
            return ActionResult.failed(FailureCategory.NOT_EXECUTED, "release_already_open")
        target = min(1., aperture + max_open_fraction_increase)
        start = self._sim_steps
        baseline = self._eef_force_norm()
        previous = self._gripper_command_fraction
        hold_site = self._control_site_pose_world().copy() if self._controller == "OSC_POSE" else None
        reason = "release_duration_limit"
        self._gripper_command_fraction = 1.
        try:
            with self._bounded_gripper_opening(max_open_fraction_increase) as command_limit:
                for _ in range(int(max_duration_s*self._control_frequency+1e-9)):
                    stepped = (self._step_action(self._osc_hold_action(hold_site)) if hold_site is not None
                               else self._step(np.zeros(7), 1.))
                    if not stepped:
                        reason = self._last_step_rejection or "release_step_rejected"
                        break
                    measured = self.get_robot_state().value.gripper_open_fraction
                    if measured >= target:
                        reason = "release_opening_limit"
                        break
            force = self._eef_force_norm()
            unloaded = force is not None and baseline is not None and force < 18. and force < baseline
            value = {"gripper_command_limit": command_limit,
                     "opening_limit_reached": reason == "release_opening_limit",
                     "requested_open_fraction_limit": target,
                     "measured_open_fraction": self.get_robot_state().value.gripper_open_fraction,
                     "opening_increase": self.get_robot_state().value.gripper_open_fraction-aperture,
                     "unloaded": bool(unloaded), "stopped_reason": reason,
                     "endpoint_reached": reason == "release_opening_limit"}
            if unloaded and reason in {"release_opening_limit", "release_duration_limit"}:
                return ActionResult.ok(value)
            return ActionResult.failed(FailureCategory.CONTROLLER_REJECTED, reason,
                executed=self._sim_steps > start, value=value)
        finally:
            if self._sim_steps == start:
                self._gripper_command_fraction = previous

    def open_gripper(self) -> ActionResult[None]:
        return self._set_gripper(1.0)

    def close_gripper(self) -> ActionResult[None]:
        return self._set_gripper(0.0)

    def home(self) -> ActionResult[None]:
        if self._controller == "OSC_POSE":
            return self._move_to_pose_osc(
                self._home_eef_position,
                self._home_eef_quaternion,
                position_only=False,
                max_steps=self._home_max_steps,
            )
        return self._move_to_joints(self._home_joints, max_steps=self._home_max_steps)
