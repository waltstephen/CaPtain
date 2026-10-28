import numpy as np

from captain_harness.adapters.robotics.shortcutting import shortcut_path


def test_shortcuts_reduce_detour_without_crossing_obstacle_or_changing_endpoints():
    original = np.array([[-.8, 0], [-.9, .7], [-.5, .9], [.5, .9], [.9, .7], [.8, 0]])
    valid = lambda q: abs(q[0]) >= .2 or q[1] >= .5
    path, report = shortcut_path(original, valid)
    assert report['shortened_length_rad'] < report['original_length_rad']
    np.testing.assert_array_equal(path[[0, -1]], original[[0, -1]])
    assert all(valid(q) for q in path)
    assert np.max(np.abs(np.diff(path, axis=0))) <= .025 + 1e-12


def test_rejects_invalid_input_edge_even_with_valid_endpoints():
    path, report = shortcut_path([[-1, 0], [1, 0]], lambda q: abs(q[0]) > .3)
    assert path is None and report['reason'] == 'input_path_invalid'


def test_budget_exhaustion_retains_only_previously_validated_path():
    original = np.array([[-.1, 0], [0, .1], [.1, 0]])
    calls = []
    def valid(q):
        calls.append(q.copy())
        return True
    path, report = shortcut_path(original, valid, max_checks=12)
    assert len(calls) <= 12
    assert report['reason'] == 'shortcut_check_budget_exhausted'
    np.testing.assert_array_equal(path[[0, -1]], original[[0, -1]])
