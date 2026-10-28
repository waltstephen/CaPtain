"""Measured poses must round-trip through commands with the same reference."""
import json
from pathlib import Path

import numpy as np
import pytest

from captain_harness.adapters.capx.backend import _quaternion_wxyz_to_matrix
from captain_harness.adapters.capx.manifest import ToolApiLevel
from captain_harness.adapters.libero.backend import LiberoActionBackend
from captain_harness.adapters.libero.types import ObservationData, RobotState
from test_tool_backend import _fixture, _invoke


@pytest.mark.parametrize('reference', ['end_effector', 'tool', 'tool_default'])
@pytest.mark.parametrize('yaw', [0., .8, 2.1])
@pytest.mark.parametrize('command', ['goto_pose', 'servo_to'])
def test_observed_pose_vertical_translation_preserves_actual_orientation(tmp_path, reference, yaw, command):
    omit_reference = reference == 'tool_default'
    reference = 'tool' if omit_reference else reference
    f = _fixture(tmp_path, dehardcoded=True, completion_evidence=True, v15_protocol=True)
    calibration = json.loads(Path('configs/calibration/libero_franka_v3.json').read_text())
    transforms = object.__new__(LiberoActionBackend)
    transforms._pyroki_hand_to_libero_eef = np.asarray(calibration['pyroki_hand_to_libero_eef_4x4']).reshape(4, 4)
    f.robot.external_ik_target_from_eef_pose = transforms.external_ik_target_from_eef_pose
    f.robot.eef_pose_from_external_ik_target = transforms.eef_pose_from_external_ik_target
    f.robot.controller = 'OSC_POSE'
    start = np.array([.45, -.09, .15])
    quat = [np.cos(yaw/2), 0., np.sin(yaw/2), 0.]
    f.robot.observation = ObservationData(f.robot.observation.cameras,
        RobotState((0.,)*7, tuple(start), tuple(quat), 1.))
    options = dict(service_aligned=True, dehardcoded=True, completion_evidence=True, v15_protocol=True)
    observation = _invoke(f, ToolApiLevel.S3_LOW_LEVEL, 'observe', {}, **options)
    assert observation.success, observation.message
    state = observation.value['robot_state']
    assert state[reference+'_pose_reference'] == reference
    requested = np.asarray(state[reference+'_position_m']) + [0, 0, .12]
    arguments = {
        'position': requested.tolist(), 'quaternion': state[reference+'_quaternion_wxyz'],
        'position_reference': state[reference+'_pose_reference'], 'approach_frame': 'world'}
    if command == 'servo_to':
        arguments = {'target_pose': {'position_m': requested.tolist(),
            'quaternion_wxyz': state[reference+'_quaternion_wxyz']},
            'position_reference': reference, 'camera': 'robot0_eye_in_hand',
            'max_duration_s': 2., 'tolerance_m': .015}
    if omit_reference:
        arguments.pop('position_reference')
    result = _invoke(f, ToolApiLevel.S3_LOW_LEVEL, command, arguments, **options)
    assert result.success, result.message
    actual = f.robot.observation.robot_state
    if command == 'servo_to':
        # Fake servo records its commanded endpoint instead of simulating motion.
        actual = RobotState((0.,)*7, f.robot.pending_eef_position,
            f.robot.pending_eef_quaternion, 1.)
    np.testing.assert_allclose(actual.end_effector_position_m, start + [0, 0, .12], atol=1e-10)
    np.testing.assert_allclose(_quaternion_wxyz_to_matrix(actual.end_effector_quaternion_wxyz),
        _quaternion_wxyz_to_matrix(quat), atol=1e-10)
    assert result.value['input_pose_reference'] == reference
    np.testing.assert_allclose(result.value['input_position_m'], requested)
