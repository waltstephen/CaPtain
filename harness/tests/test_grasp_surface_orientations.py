"""Surface-derived approaches are geometrical and frame equivariant."""
import numpy as np
from scipy.spatial.transform import Rotation

from captain_harness.adapters.robotics.grasp_proposals import surface_orientations


def test_side_approaches_exist_for_both_visible_horizontal_axes():
    points = np.array([[x, y, z] for x in [-.02, .02] for y in [-.04, .04] for z in [0., .15]])
    normal = np.array([0., 0., 1.])
    orientations = surface_orientations(np.diag([1., -1., -1.]), points, normal)
    sides = [r for name, r in orientations if name.startswith('surface_tangent')]
    assert len(sides) == 8
    for r in sides:
        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-9)
        assert np.linalg.det(r) > .999
        assert abs(r[:, 2] @ normal) < 1e-9
        assert abs(r[:, 1] @ normal) < 1e-9
        assert any(np.allclose(other[:, 2], -r[:, 2]) for other in sides)
        assert any(np.allclose(other, r @ np.diag([-1., -1., 1.])) for other in sides)


def test_rotating_the_scene_rotates_the_proposals_without_world_axis_rules():
    rng = np.random.default_rng(20)
    target = rng.normal(size=(100, 3)) * [.02, .04, .08]
    normal = np.array([0., 0., 1.])
    original = np.diag([1., -1., -1.])
    rotation = Rotation.from_rotvec([.5, .2, -.7]).as_matrix()
    first = surface_orientations(original, target, normal)
    second = surface_orientations(rotation @ original, target @ rotation.T, rotation @ normal)
    # PCA signs may change; both approach directions must remain represented.
    first_sides = [rotation @ r for n, r in first if n.startswith('surface_tangent')]
    second_sides = [r for n, r in second if n.startswith('surface_tangent')]
    assert len(first_sides) == len(second_sides) == 8
    for r in first_sides:
        assert any(np.allclose(r, other) for other in second_sides)
