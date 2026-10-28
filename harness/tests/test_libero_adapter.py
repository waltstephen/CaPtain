from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from captain_harness.adapters.libero.backend import LiberoActionBackend, reset_to_seeded_init_state
from captain_harness.ports.capability import FailureCategory


class FakeModel:
    cam_fovy = np.array([45.0, 45.0])
    nv = 7
    jnt_range = np.tile(np.array([-0.5, 0.5]), (7, 1))
    jnt_limited = np.ones(7, dtype=np.uint8)

    @staticmethod
    def camera_name2id(name): return {"agentview": 0, "robot0_eye_in_hand": 1}[name]

    @staticmethod
    def body_name2id(name): return {"robot0_base": 0, "gripper0_eef": 1}[name]

    @staticmethod
    def joint_name2id(name): return int(name.removeprefix("robot0_joint")) - 1

    @staticmethod
    def site_name2id(name): return 0


class FakeData:
    def __init__(self, env):
        self.env = env

    @property
    def xpos(self):
        return np.array([[0.0, 0.0, 0.0],
                         [0.4 + self.env.joints[0], 0.1 + self.env.joints[1],
                          0.5 + self.env.joints[2]]])

    @property
    def xmat(self):
        return np.tile(np.eye(3).reshape(1, 9), (2, 1))

    @property
    def site_xpos(self):
        return np.asarray([self.xpos[1]], dtype=np.float64)

    @property
    def site_xmat(self):
        return np.asarray([np.eye(3).reshape(9)], dtype=np.float64)

    def get_body_jacp(self, name):
        jacobian = np.zeros((3, 7))
        jacobian[:, :3] = np.eye(3) * self.env.jacobian_scale
        return jacobian

    @staticmethod
    def get_body_jacr(name):
        jacobian = np.zeros((3, 7))
        jacobian[:, 3:6] = np.eye(3)
        return jacobian

    @staticmethod
    def get_camera_xpos(name): return np.array([0.5, 0.0, 1.0 if name == "agentview" else 0.7])

    @staticmethod
    def get_camera_xmat(name): return np.eye(3)


class FakeEnv:
    def __init__(self, *, jacobian_scale=1.0):
        self.joints, self.gripper = np.zeros(7), 0.04
        self.jacobian_scale = jacobian_scale
        self.sim = type("Sim", (), {"model": FakeModel(), "data": FakeData(self)})()
        robot_model = type("RobotModel", (), {
            "joints": [f"robot0_joint{index}" for index in range(1, 8)]
        })()
        self.robots = [type("Robot", (), {
            "_ref_joint_vel_indexes": list(range(7)), "robot_model": robot_model
        })()]
        self.seed_calls, self.init_state_calls, self.actions = [], [], []

    def seed(self, seed): self.seed_calls.append(seed)

    def reset(self):
        self.joints = np.zeros(7)
        return self._get_observations()

    def set_init_state(self, state):
        self.init_state_calls.append(np.array(state, copy=True))
        self.joints = np.array(state[:7], dtype=float)
        return self._get_observations()

    def _get_observations(self):
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        return {"robot0_joint_pos": self.joints.copy(),
                "robot0_gripper_qpos": np.array([self.gripper, -self.gripper]),
                "agentview_image": image, "robot0_eye_in_hand_image": image,
                "agentview_depth": np.ones((512, 512, 1), dtype=np.float32),
                "robot0_eye_in_hand_depth": np.ones((512, 512, 1), dtype=np.float32)}

    def step(self, action):
        action = np.asarray(action, dtype=float)
        self.actions.append(action.copy())
        self.joints += action[:7] / 20.0
        self.gripper = 0.04 if action[-1] < 0 else 0.0
        return self._get_observations(), 1.0, True, {"hidden": "discarded"}


class FakeOscEnv(FakeEnv):
    def __init__(self):
        super().__init__()
        controller = type(
            "Controller",
            (),
            {
                "name": "OSC_POSE",
                "update": lambda self, force=False: None,
                "reset_goal": lambda self: None,
                "control_dim": 6,
                "output_max": np.asarray(
                    [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                    dtype=np.float64,
                ),
            },
        )()
        gripper = type(
            "Gripper",
            (),
            {"important_sites": {"grip_site": "gripper0_grip_site"}},
        )()
        recent_ee_vel = type(
            "RecentVelocity",
            (),
            {"current": np.zeros(6, dtype=np.float64)},
        )()
        self.robots[0].controller = controller
        self.robots[0].gripper = gripper
        self.robots[0].recent_ee_vel = recent_ee_vel
        self.robots[0].ee_force = np.zeros(3, dtype=np.float64)
        self.robots[0].action_dim = 7

    # Single native-step fixture. Multi-substep latency is tested separately.
    model_timestep=.05

    def _pre_action(self, action, policy_step=False):
        self._native_action=np.asarray(action,dtype=float)

    def _update_observables(self, force=False):
        pass

    def step(self, action):
        self._pre_action(action,policy_step=True)
        action = np.asarray(self._native_action, dtype=float)
        assert action.shape == (7,)
        self.actions.append(action.copy())
        self.joints[:3] += action[:3] * 0.05
        self.gripper = 0.04 if action[-1] < 0 else 0.0
        self._update_observables()
        return self._get_observations(), 1.0, True, {"hidden": "discarded"}


def make_backend(env, path, **kwargs):
    return LiberoActionBackend(env, observation_dir=path,
                               initial_observation=env.reset(), horizon=20, **kwargs)


def test_reset_uses_seed_as_stable_init_state_index():
    env = FakeEnv()
    states = np.stack([np.full(7, value) for value in (0.1, 0.2, 0.3)])
    first = reset_to_seeded_init_state(env, states, seed=4)
    second = reset_to_seeded_init_state(env, states, seed=4)
    assert first.init_state_index == second.init_state_index == 1
    assert env.seed_calls == [4, 4]
    assert np.array_equal(first.observation["robot0_joint_pos"], states[1])
    assert len(env.actions) == 10


def test_reset_settles_physics_before_returning_public_observation():
    class SettlingEnv(FakeEnv):
        def __init__(self):
            super().__init__()
            self.height = 0.1

        def reset(self):
            self.height = 0.1
            observation = super().reset()
            observation["object_height"] = self.height
            return observation

        def set_init_state(self, state):
            observation = super().set_init_state(state)
            observation["object_height"] = self.height
            return observation

        def step(self, action):
            observation, reward, done, info = super().step(action)
            self.height = max(0.0, self.height - 0.02)
            observation["object_height"] = self.height
            return observation, reward, done, info

    env = SettlingEnv()

    result = reset_to_seeded_init_state(
        env,
        [np.zeros(7)],
        seed=0,
    )

    assert result.observation["object_height"] == 0.0
    assert len(env.actions) == 5


def test_reset_uses_runtime_action_dimension_for_osc_pose() -> None:
    env = FakeOscEnv()

    reset_to_seeded_init_state(
        env,
        [np.zeros(7)],
        seed=0,
        settling_steps=2,
    )

    assert len(env.actions) == 2
    assert all(action.shape == (7,) for action in env.actions)


def test_joint_and_gripper_actions_use_libero_contract(tmp_path: Path):
    env = FakeEnv()
    backend = make_backend(env, tmp_path, motion_max_steps=4, gripper_steps=2)
    assert backend.move_to_joints((0.1,) * 7).success
    assert np.allclose(env.actions[0][0:7], np.full(7, 2.0))
    assert env.actions[0][-1] == -1.0
    assert backend.open_gripper().success and backend.close_gripper().success
    assert [action[-1] for action in env.actions[-4:]] == [-1.0, -1.0, 1.0, 1.0]


def test_default_gripper_duration_matches_pinned_capx(tmp_path: Path):
    env = FakeEnv()
    backend = LiberoActionBackend(
        env,
        observation_dir=tmp_path,
        initial_observation=env.reset(),
        horizon=100,
    )
    assert backend.open_gripper().success
    assert len(env.actions) == 40


def test_backend_ignores_online_done_signal(
    tmp_path: Path,
) -> None:
    class DoneSequenceEnv(FakeEnv):
        def __init__(self):
            super().__init__()
            self.done_values = iter((False, True, False))

        def step(self, action):
            observation, reward, _, info = super().step(action)
            return observation, reward, next(self.done_values), info

    env = DoneSequenceEnv()
    backend = LiberoActionBackend(
        env,
        observation_dir=tmp_path,
        initial_observation=env.reset(),
        horizon=10,
        gripper_steps=3,
    )

    assert backend.open_gripper().success
    assert not hasattr(backend, "host_completion")


def test_recorded_motion_limit_allows_slow_converging_joint(tmp_path: Path):
    class SlowLastJointEnv(FakeEnv):
        def step(self, action):
            action = np.asarray(action, dtype=float)
            self.actions.append(action.copy())
            self.joints[:6] += action[:6] / 20.0
            self.joints[6] += action[6] / 800.0
            self.gripper = 0.04 if action[-1] < 0 else 0.0
            return self._get_observations(), 1.0, True, {"hidden": "discarded"}

    env = SlowLastJointEnv()
    backend = LiberoActionBackend(
        env,
        observation_dir=tmp_path,
        initial_observation=env.reset(),
        horizon=300,
        motion_max_steps=360,
    )

    result = backend.move_to_joints((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8))

    assert result.success
    assert 120 < len(env.actions) <= 360


def test_joint_timeout_reports_residuals(tmp_path: Path):
    class StuckEnv(FakeEnv):
        def step(self, action):
            action = np.asarray(action, dtype=float)
            self.actions.append(action.copy())
            return self._get_observations(), 1.0, True, {"hidden": "discarded"}

    env = StuckEnv()
    backend = LiberoActionBackend(
        env,
        observation_dir=tmp_path,
        initial_observation=env.reset(),
        horizon=10,
        motion_max_steps=2,
    )

    result = backend.move_to_joints((0.4,) * 7)

    assert result.failure_category is FailureCategory.TIMEOUT
    assert "joint 1" in result.message
    assert "max residual" in result.message
    assert "L2 residual" in result.message


def test_default_cartesian_ik_moves_pose_and_delta_in_base_frame(tmp_path: Path):
    env = FakeEnv()
    backend = make_backend(env, tmp_path, cartesian_max_steps=8)
    result = backend.move_to_pose((0.44, 0.08, 0.51), (1, 0, 0, 0))
    assert result.success and result.executed
    assert np.allclose(backend.get_robot_state().value.end_effector_position_m,
                       (0.44, 0.08, 0.51), atol=0.002)
    result = backend.move_by((0.01, 0.01, -0.01))
    assert result.success and result.executed
    assert np.allclose(backend.get_robot_state().value.end_effector_position_m,
                       (0.45, 0.09, 0.50), atol=0.002)
    assert np.all(np.abs(env.joints) <= 0.5)


def test_osc_pose_moves_in_cartesian_space_and_keeps_joint_api_explicit(
    tmp_path: Path,
) -> None:
    env = FakeOscEnv()
    backend = LiberoActionBackend(
        env,
        observation_dir=tmp_path,
        initial_observation=env.reset(),
        horizon=100,
        controller="OSC_POSE",
        cartesian_max_steps=20,
        gripper_steps=2,
    )

    moved = backend.move_to_pose(
        (0.44, 0.08, 0.51),
        (1.0, 0.0, 0.0, 0.0),
    )

    assert moved.success and moved.executed
    assert np.allclose(
        backend.get_robot_state().value.end_effector_position_m,
        (0.44, 0.08, 0.51),
        atol=0.004,
    )
    rejected = backend.move_to_joints((0.0,) * 7)
    assert rejected.failure_category is FailureCategory.CONTROLLER_REJECTED
    assert "OSC_POSE" in rejected.message
    assert all(action.shape == (7,) for action in env.actions)


def test_osc_pose_stall_is_reported_as_failure(tmp_path: Path) -> None:
    class StalledOscEnv(FakeOscEnv):
        def step(self, action):
            action = np.asarray(action, dtype=float)
            self.actions.append(action.copy())
            return self._get_observations(), 1.0, True, {
                "hidden": "discarded"
            }

    env = StalledOscEnv()
    backend = LiberoActionBackend(
        env,
        observation_dir=tmp_path,
        initial_observation=env.reset(),
        horizon=100,
        controller="OSC_POSE",
        cartesian_max_steps=20,
        cartesian_stall_steps=2,
    )

    result = backend.move_to_pose(
        (0.50, 0.10, 0.50),
        (1.0, 0.0, 0.0, 0.0),
    )

    assert not result.success
    assert result.executed
    assert result.failure_category is FailureCategory.TIMEOUT
    assert "OSC_POSE stalled before the requested pose" in result.message
    assert "position residual" in result.message
    assert "orientation residual" in result.message


def test_osc_approach_and_close_reports_gripper_detection_first(
    tmp_path: Path,
) -> None:
    env = FakeOscEnv()
    backend = LiberoActionBackend(
        env,
        observation_dir=tmp_path,
        initial_observation=env.reset(),
        horizon=200,
        controller="OSC_POSE",
        cartesian_max_steps=30,
        gripper_steps=2,
    )

    result = backend.approach_and_close(
        (0.42, 0.1, 0.48),
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        approach_distance_m=0.03,
    )

    assert result.success
    assert result.value["gripper_open_fraction"] == 0.0
    assert result.value["object_detected"] is False
    assert backend.gripper_state().value["object_detected"] is False
    motion_z = [float(action[2]) for action in env.actions[2:]]
    assert any(value > 0 for value in motion_z)
    assert any(value < 0 for value in motion_z)


def test_descent_obstruction_preserves_open_gripper_and_reports_residual(tmp_path, monkeypatch):
    env = FakeOscEnv()
    backend = LiberoActionBackend(env, observation_dir=tmp_path,
        initial_observation=env.reset(), horizon=200, controller="OSC_POSE",
        cartesian_max_steps=30, gripper_steps=2)
    forces = iter([0.0, 20.0])
    monkeypatch.setattr(backend, "_eef_force_norm", lambda: next(forces, 20.0))
    closes = []
    original = backend._set_gripper
    monkeypatch.setattr(backend, "_set_gripper", lambda value: (closes.append(value), original(value))[1])
    result = backend.descend_and_close((0.4, 0.1, 0.35), (1, 0, 0, 0))
    assert not result.success and result.executed
    assert result.value["closure_executed"] is False
    assert result.value["position_residual_m"] > 0.003
    assert result.value["stopped_reason"] == "contact_force"
    assert result.value["force_limit_n"] == 18.0
    assert not closes


def test_descent_at_target_can_close_and_labels_empty_result(tmp_path):
    env = FakeOscEnv()
    backend = LiberoActionBackend(env, observation_dir=tmp_path,
        initial_observation=env.reset(), horizon=200, controller="OSC_POSE",
        cartesian_max_steps=30, gripper_steps=2)
    result = backend.descend_and_close((0.4, 0.1, 0.5), (1, 0, 0, 0))
    assert result.success and result.value["closure_executed"] is True
    assert result.value["grasp_outcome"] == "empty_close"


def test_osc_three_phase_pregrasp_reports_phase_and_residuals(
    tmp_path: Path,
) -> None:
    env = FakeOscEnv()
    backend = LiberoActionBackend(
        env,
        observation_dir=tmp_path,
        initial_observation=env.reset(),
        horizon=200,
        controller="OSC_POSE",
        cartesian_max_steps=30,
        gripper_steps=2,
    )

    result = backend.align_pregrasp(
        (0.42, 0.1, 0.48),
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        approach_distance_m=0.03,
    )

    assert result.success
    assert result.value["converged"] is True
    assert result.value["phase_reached"] == "pregrasp"
    assert result.value["failed_phase"] is None
    assert result.value["position_residual_m"] <= 0.004


class NearMissOscEnv(FakeOscEnv):
    """Freezes any axis whose commanded step is below 16 mm.

    Leaves a persistent ~15 mm residual on axes that start closer than the
    freeze threshold, which the public 30 mm contract accepts but the old
    4 mm internal default declared a stall.
    """

    def step(self, action):
        action = np.asarray(action, dtype=float)
        assert action.shape == (7,)
        self.actions.append(action.copy())
        delta = action[:3] * 0.05
        delta[np.abs(delta) < 0.016] = 0.0
        self.joints[:3] += delta
        self.gripper = 0.04 if action[-1] < 0 else 0.0
        return self._get_observations(), 1.0, True, {"hidden": "discarded"}


def test_align_pregrasp_accepts_near_miss_within_phase_tolerance(
    tmp_path: Path,
) -> None:
    env = NearMissOscEnv()
    backend = LiberoActionBackend(
        env,
        observation_dir=tmp_path,
        initial_observation=env.reset(),
        horizon=400,
        controller="OSC_POSE",
        cartesian_max_steps=30,
        gripper_steps=2,
    )

    # Start eef (0.4, 0.1, 0.5); hover z = pregrasp_z + 0.09 = 0.515, so the
    # translate phase carries a 15 mm z error the env refuses to close.
    result = backend.align_pregrasp(
        (0.42, 0.1, 0.485),
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        approach_distance_m=0.06,
    )

    assert result.success, result.message
    assert result.value["converged"] is True
    assert result.value["phase_reached"] == "pregrasp"


class RotationLockedAxisOscEnv(FakeOscEnv):
    """Yaw-capable env that freezes x actuation while a rotation is commanded.

    The first rotation step also kicks x by 70 mm, so the rotate phase can
    never satisfy a coupled position check; only the position-only settle
    phase can recover x once rotation commands cease.
    """

    def __init__(self):
        super().__init__()
        self.yaw = 0.0
        self.kicked = False
        env = self

        class YawData(FakeData):
            @property
            def xmat(self):
                cos, sin = math.cos(env.yaw), math.sin(env.yaw)
                rotation = np.array(
                    [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]],
                    dtype=np.float64,
                )
                return np.stack(
                    [np.eye(3).reshape(9), rotation.reshape(9)]
                )

            @property
            def site_xmat(self):
                return np.asarray([self.xmat[1]], dtype=np.float64)

        self.sim.data = YawData(self)

    def step(self, action):
        action = np.asarray(action, dtype=float)
        assert action.shape == (7,)
        self.actions.append(action.copy())
        rotating = float(np.linalg.norm(action[3:6])) > 1e-8
        delta = action[:3] * 0.05
        if rotating:
            if not self.kicked:
                self.joints[0] += 0.07
                self.kicked = True
            delta[0] = 0.0
        self.joints[:3] += delta
        self.yaw += float(action[5]) * 0.1
        self.gripper = 0.04 if action[-1] < 0 else 0.0
        return self._get_observations(), 1.0, True, {"hidden": "discarded"}


def test_align_pregrasp_rotate_drift_recovered_by_settle_phase(
    tmp_path: Path,
) -> None:
    env = RotationLockedAxisOscEnv()
    backend = LiberoActionBackend(
        env,
        observation_dir=tmp_path,
        initial_observation=env.reset(),
        horizon=2000,
        controller="OSC_POSE",
        cartesian_max_steps=40,
        gripper_steps=2,
    )

    yaw = 0.3
    quaternion = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))
    result = backend.align_pregrasp(
        (0.42, 0.1, 0.485),
        quaternion,
        (0.0, 0.0, 1.0),
        approach_distance_m=0.06,
    )

    assert result.success, result.message
    assert result.value["converged"] is True
    assert env.kicked
    assert result.value["position_residual_m"] <= 0.015
    assert result.value["orientation_residual_rad"] <= 0.087


class FrozenOscEnv(FakeOscEnv):
    def step(self, action):
        action = np.asarray(action, dtype=float)
        assert action.shape == (7,)
        self.actions.append(action.copy())
        self.gripper = 0.04 if action[-1] < 0 else 0.0
        return self._get_observations(), 1.0, True, {"hidden": "discarded"}


def test_align_pregrasp_stall_beyond_tolerance_names_failed_phase(
    tmp_path: Path,
) -> None:
    env = FrozenOscEnv()
    backend = LiberoActionBackend(
        env,
        observation_dir=tmp_path,
        initial_observation=env.reset(),
        horizon=200,
        controller="OSC_POSE",
        cartesian_max_steps=30,
        gripper_steps=2,
    )

    result = backend.align_pregrasp(
        (0.5, 0.2, 0.4),
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        approach_distance_m=0.06,
    )

    assert not result.success
    assert "align_pregrasp translate failed" in result.message
    assert result.value["converged"] is False
    assert result.value["failed_phase"] == "translate"
    assert result.value["phase_reached"] == "none"
    assert result.value["position_residual_m"] > 0.02


def test_position_priority_cartesian_correction_ignores_orientation_gate(
    tmp_path: Path,
) -> None:
    env = FakeEnv()
    backend = make_backend(env, tmp_path, cartesian_max_steps=20)

    result = backend.move_to_pose(
        (0.44, 0.08, 0.51),
        (0.0, 1.0, 0.0, 0.0),
        position_only=True,
    )

    assert result.success
    assert np.allclose(
        backend.get_robot_state().value.end_effector_position_m,
        (0.44, 0.08, 0.51),
        atol=0.006,
    )


def test_external_ik_target_converts_libero_eef_pose_to_pyroki_hand(
    tmp_path: Path,
) -> None:
    backend = make_backend(FakeEnv(), tmp_path)

    position, quaternion = backend.external_ik_target_from_eef_pose(
        (0.4, 0.1, 0.5),
        (1.0, 0.0, 0.0, 0.0),
    )

    assert np.allclose(position, (0.4, 0.1, 0.4035), atol=1e-8)
    assert abs(quaternion[0] - 2**-0.5) < 5e-4
    assert abs(quaternion[3] - 2**-0.5) < 5e-4
    assert abs(quaternion[1]) < 1e-8
    assert abs(quaternion[2]) < 1e-8


def test_external_ik_target_and_libero_eef_pose_are_inverse(
    tmp_path: Path,
) -> None:
    backend = make_backend(FakeEnv(), tmp_path)
    eef_position = (0.4, 0.1, 0.5)
    eef_quaternion = (0.5, 0.5, 0.5, 0.5)

    hand_position, hand_quaternion = (
        backend.external_ik_target_from_eef_pose(
            eef_position,
            eef_quaternion,
        )
    )
    recovered_position, recovered_quaternion = (
        backend.eef_pose_from_external_ik_target(
            hand_position,
            hand_quaternion,
        )
    )

    assert np.allclose(recovered_position, eef_position, atol=1e-8)
    assert abs(
        np.dot(
            np.asarray(recovered_quaternion),
            np.asarray(eef_quaternion),
        )
    ) > 1 - 1e-8


def test_cartesian_ik_reports_unreachable_without_claiming_execution(tmp_path: Path):
    env = FakeEnv(jacobian_scale=0.0)
    result = make_backend(env, tmp_path, cartesian_stall_steps=2).move_by((0.02, 0.0, 0.0))
    assert result.failure_category is FailureCategory.CONTROLLER_REJECTED
    assert not result.executed
    assert "stalled" in result.message


def test_pose_motion_preserves_explicit_solver_override(tmp_path: Path):
    env = FakeEnv()
    backend = make_backend(env, tmp_path / "reject", inverse_kinematics=lambda p, q, j: None)
    result = backend.move_to_pose((0.4, 0.1, 0.5), (1, 0, 0, 0))
    assert result.failure_category is FailureCategory.CONTROLLER_REJECTED


def test_observation_writes_two_512_images_and_calibration(tmp_path: Path):
    result = make_backend(FakeEnv(), tmp_path).get_observation_data()
    assert result.success and result.value is not None
    assert [camera.camera_name for camera in result.value.cameras] == ["agentview", "robot0_eye_in_hand"]
    for camera in result.value.cameras:
        assert Path(camera.rgb_path).is_file() and Path(camera.depth_path or "").is_file()
        assert len(camera.intrinsics_3x3) == 9 and len(camera.extrinsics_4x4) == 16
    assert result.value.robot_state.end_effector_position_m == (0.4, 0.1, 0.5)


def test_horizon_failures_are_explicit(tmp_path: Path):
    env = FakeEnv()
    backend = LiberoActionBackend(env, observation_dir=tmp_path,
                                   initial_observation=env.reset(), horizon=1, gripper_steps=2)
    first, second = backend.open_gripper(), backend.close_gripper()
    assert first.failure_category is FailureCategory.TIMEOUT and first.executed
    assert second.failure_category is FailureCategory.NOT_EXECUTED and not second.executed


def test_pregrasp_leaves_low_pose_vertically_before_lateral_motion(tmp_path):
    class LowObstacleEnv(FakeOscEnv):
        def step(self, action):
            # Publicly simulated clearance constraint; a diagonal shortcut would hit it.
            if self.sim.data.xpos[1, 2] < 0.57 and self.sim.data.xpos[1, 0] < 0.43:
                assert np.linalg.norm(np.asarray(action)[:2]) < 1e-8
            return super().step(action)
    env = LowObstacleEnv()
    backend = LiberoActionBackend(env, observation_dir=tmp_path,
        initial_observation=env.reset(), horizon=1000, controller='OSC_POSE',
        cartesian_max_steps=40, gripper_steps=2)
    # pregrasp=.52 and existing default clearance=.09 -> hover=.61
    result = backend.align_pregrasp((.55,.2,.55), (1,0,0,0), (0,0,1), approach_distance_m=.03)
    assert result.success, result.message
    assert result.value['phase_reached'] == 'pregrasp'


def test_failed_vertical_clearance_stops_before_translate(tmp_path):
    env = FrozenOscEnv()
    backend = LiberoActionBackend(env, observation_dir=tmp_path,
        initial_observation=env.reset(), horizon=1000, controller='OSC_POSE',
        cartesian_max_steps=30, gripper_steps=2)
    result = backend.align_pregrasp((.55,.2,.55), (1,0,0,0), (0,0,1), approach_distance_m=.03)
    assert not result.success
    assert result.value['failed_phase'] == 'clearance'
    assert result.value['phase_reached'] == 'none'
    assert all(np.linalg.norm(a[:2]) < 1e-8 for a in env.actions)
