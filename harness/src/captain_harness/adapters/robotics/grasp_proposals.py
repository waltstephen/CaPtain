"""Gripper orientations from a measured support normal and target surfaces."""
import numpy as np


def surface_orientations(original_rotation, target, normal):
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    centered = np.asarray(target, dtype=float) - np.mean(target, axis=0)
    projected = centered - (centered @ normal)[:, None] * normal
    _, axes = np.linalg.eigh(np.cov(projected.T))
    result = []
    for name, heading in [('surface_normal_same_heading', original_rotation[:, 1]),
                          ('surface_normal_narrow_axis', axes[:, 1])]:
        closing = heading - (heading @ normal) * normal
        if np.linalg.norm(closing) < 1e-8:
            continue
        closing /= np.linalg.norm(closing)
        approach = -normal
        result.append((name, np.column_stack([np.cross(closing, approach), closing, approach])))
    # A tall object may obstruct the palm during top entry while exposing a
    # usable lateral surface. Both signs avoid choosing an approach from a
    # task label or a fixed world direction. Contact/path/IK checks decide
    # whether any proposal is usable; these are not force-closure claims.
    for axis in (1, 2):
        tangent = axes[:, axis] - (axes[:, axis] @ normal) * normal
        if np.linalg.norm(tangent) < 1e-8:
            continue
        tangent /= np.linalg.norm(tangent)
        for sign in (-1, 1):
            approach = sign * tangent
            closing = np.cross(normal, approach)
            closing /= np.linalg.norm(closing)
            result.append((f'surface_tangent_axis_{axis}_{sign:+d}',
                np.column_stack([np.cross(closing, approach), closing, approach])))
    # Parallel jaws can exchange sides without changing the closing line.
    # The wrist orientation still changes, so IK and robot geometry must check
    # both variants rather than treating one joint-limit failure as conclusive.
    return [option for name, rotation in result for option in (
        (name, rotation), (name + '_jaw_swap', rotation @ np.diag([-1., -1., 1.]))) ]
