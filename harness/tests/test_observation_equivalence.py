from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest

from captain_harness.adapters.capx.observation_equivalence import same_public_snapshot


def snapshot():
    return SimpleNamespace(time_s=12.3, robot_state=SimpleNamespace(
        joint_positions_rad=(0.,) * 7, gripper_open_fraction=1.,
        end_effector_position_m=(0., 0., .3), end_effector_quaternion_wxyz=(1., 0., 0., 0.)),
        cameras=(SimpleNamespace(camera_name='head', image=np.zeros((4,4,3), np.uint8),
            depth=np.ones((4,4)), intrinsics=np.eye(3), extrinsics=np.eye(4)),))


def test_distinct_handles_may_represent_exactly_same_paused_public_snapshot():
    first = snapshot()
    assert same_public_snapshot(first, deepcopy(first))


@pytest.mark.parametrize('change', ['time', 'missing_time', 'image', 'depth', 'extrinsics', 'intrinsics', 'camera', 'joints', 'gripper', 'pose', 'nan_depth'])
def test_snapshot_aliasing_rejects_any_observable_change_or_missing_time(change):
    first = snapshot()
    second = deepcopy(first)
    if change == 'time': second.time_s += .05
    elif change == 'missing_time': first.time_s = second.time_s = None
    elif change == 'camera': second.cameras[0].camera_name = 'wrist'
    elif change == 'joints': second.robot_state.joint_positions_rad = (.001,) * 7
    elif change == 'gripper': second.robot_state.gripper_open_fraction = .95
    elif change == 'pose': second.robot_state.end_effector_position_m = (.01, 0., .3)
    elif change == 'nan_depth': second.cameras[0].depth[0,0] = float('nan')
    else:
        value = getattr(second.cameras[0], change)
        value.flat[0] += 1
    assert not same_public_snapshot(first, second)
