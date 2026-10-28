import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from captain_harness.adapters.robotics.cartesian_sweeps import cartesian_sweep


def test_pivoted_rotation_follows_a_circle_with_bounded_full_pose_increments():
    start = np.eye(4)
    start[:3, 3] = [.4, -.2, .1]
    pivot = [.3, -.2, .1]
    path = cartesian_sweep(start, axis=[0, 0, 7], pivot_m=pivot, angle_rad=np.pi/2)
    np.testing.assert_allclose(path[-1, :3, 3], [.3, -.1, .1], atol=1e-12)
    np.testing.assert_allclose(path[-1, :3, :3], Rotation.from_euler('z', 90, degrees=True).as_matrix(), atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(path[:, :3, 3] - pivot, axis=1), .1, atol=1e-12)
    assert np.max(np.linalg.norm(np.diff(path[:, :3, 3], axis=0), axis=1)) <= .004
    turns = [Rotation.from_matrix(b[:3, :3] @ a[:3, :3].T).magnitude() for a, b in zip(path, path[1:])]
    assert max(turns) <= .02 + 1e-12


def test_linear_sweep_preserves_orientation_and_input_and_reverses_without_task_knowledge():
    start = np.eye(4)
    start[:3, :3] = Rotation.from_euler('xyz', [.2, .4, .1]).as_matrix()
    saved = start.copy()
    path = cartesian_sweep(start, displacement_m=[.07, .02, -.03])
    np.testing.assert_array_equal(start, saved)
    np.testing.assert_allclose(path[-1, :3, 3], [.07, .02, -.03])
    for pose in path:
        np.testing.assert_allclose(pose[:3, :3], start[:3, :3])
    assert np.max(np.linalg.norm(np.diff(path[:, :3, 3], axis=0), axis=1)) <= .004 + 1e-12
    reverse = cartesian_sweep(path[-1], displacement_m=[-.07, -.02, .03])
    np.testing.assert_allclose(reverse[-1], start, atol=1e-12)


@pytest.mark.parametrize('options', [
    {'axis': [0, 0, 0], 'pivot_m': [0, 0, 0], 'angle_rad': .2},
    {'angle_rad': .2}, {'axis': [0, 0, 1], 'angle_rad': .2},
    {'displacement_m': [0, np.nan, 0]}, {'max_position_step_m': 0},
    {'displacement_m': [1, 0, 0], 'max_position_step_m': 1e-9}])
def test_invalid_or_unbounded_sweep_is_rejected(options):
    with pytest.raises(ValueError):
        cartesian_sweep(np.eye(4), **options)
