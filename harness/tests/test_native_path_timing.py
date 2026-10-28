import numpy as np

from captain_harness.adapters.libero.native_path_timing import advance_reference
from test_native_joint_sweep import Model


def test_dense_waypoints_share_one_control_cycle_without_shortcutting_path_length():
    model = Model()
    path = np.zeros((251, 7)); path[:, 0] = np.linspace(0, .005, 251)
    command, index, record = advance_reference(model, path, path[0], 1, 1.,
        translation_budget=.0005, rotation_budget=.01, joint_budget=.05)
    assert index == 26
    assert abs(command[0]-.0005) < 1e-12
    assert record['traversed_waypoint_count'] == 25
    assert record['reference_translation_m'] <= .0005+1e-12


def test_small_backtracking_segments_consume_their_full_distance():
    model = Model()
    path = np.zeros((4, 7)); path[:, 0] = [0, .0004, 0, .0004]
    command, index, record = advance_reference(model, path, path[0], 1, 1.,
        translation_budget=.0005, rotation_budget=.01, joint_budget=.05)
    assert index == 2
    assert abs(command[0]-.0003) < 1e-12
    assert abs(record['reference_translation_m']-.0005) < 1e-12


def test_rotation_budget_and_joint_budget_limit_reference_progress():
    model = Model()
    path = np.zeros((3, 7)); path[:, 5] = [0, .001, .1]
    command, index, record = advance_reference(model, path, path[0], 1, 1.,
        translation_budget=.01, rotation_budget=.005, joint_budget=.05)
    assert abs(command[5]-.005) < 1e-12
    assert record['reference_rotation_rad'] <= .005+1e-12
    command, _, record = advance_reference(model, path, path[0], 1, 1.,
        translation_budget=.01, rotation_budget=.05, joint_budget=.002)
    assert abs(command[5]-.002) < 1e-12
    assert record['reference_joint_travel_rad'] <= .002+1e-12
