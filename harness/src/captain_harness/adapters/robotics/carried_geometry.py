"""Unregistered candidate for carrying observed geometry under rigid attachment.

Callers must supply already validated, contemporaneous public observations and
attachment provenance. This geometry helper does not establish that provenance.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree


def _points(value, *, allow_empty=False):
    points = np.asarray(value, dtype=float)
    if allow_empty and points.size == 0:
        return np.empty((0, 3))
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('observed points must be a finite N by 3 array')
    return points


def _pose(value):
    pose = np.asarray(value, dtype=float)
    if (pose.shape != (4,4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0,0,0,1], rtol=0, atol=1e-8)
            or not np.allclose(pose[:3,:3].T @ pose[:3,:3], np.eye(3), rtol=0, atol=1e-6)
            or not np.isclose(np.linalg.det(pose[:3,:3]), 1., rtol=0, atol=1e-6)):
        raise ValueError('end-effector transform must be finite and rigid')
    return pose


def partition_measured_target(scene_points, target_points):
    """Remove only identical selected samples; never erase an enclosing region."""
    scene = _points(scene_points, allow_empty=True)
    target = _points(target_points)
    if not len(target):
        raise ValueError('selected target has no measured samples')
    same = cKDTree(target).query(scene)[0] <= 1e-7
    return scene[~same].copy(), scene[same].copy()


@dataclass(frozen=True)
class ObservedCarriedGeometry:
    _points_local: np.ndarray
    _planes_local: np.ndarray

    @classmethod
    def from_observation(cls, points_world, eef_pose):
        points, pose = _points(points_world), _pose(eef_pose)
        if len(points) < 4:
            raise ValueError('at least four noncoplanar observed points are required')
        local = (points-pose[:3,3]) @ pose[:3,:3]
        try:
            hull = ConvexHull(local)
        except QhullError as exc:
            raise ValueError('observed geometry does not define a 3D hull') from exc
        if hull.volume <= 1e-12:
            raise ValueError('observed hull volume is numerically insufficient')
        planes = np.unique(hull.equations, axis=0)
        local.setflags(write=False)
        planes.setflags(write=False)
        return cls(local, planes)

    def world_points(self, eef_pose):
        pose = _pose(eef_pose)
        return self._points_local @ pose[:3,:3].T + pose[:3,3]

    def scene_collision(self, eef_pose, scene_points, margin_m):
        if not np.isfinite(margin_m) or margin_m < 0:
            raise ValueError('collision margin must be finite and nonnegative')
        pose, scene = _pose(eef_pose), _points(scene_points, allow_empty=True)
        count = 0
        # No axis-aligned broad-phase truncation: inflating acute hull planes
        # can extend beyond a simple original-bounds-plus-margin box.
        for offset in range(0, len(scene), 1024):
            local = (scene[offset:offset+1024]-pose[:3,3]) @ pose[:3,:3]
            distances = local @ self._planes_local[:,:3].T + self._planes_local[:,3]
            count += int(np.sum(np.max(distances, axis=1) <= margin_m))
        return {'collision': bool(count), 'observed_scene_points_in_carried_hull': count,
            'geometry_scope': 'partial_visible_convex_hull',
            'margin_m': float(margin_m),
            'limitation': 'Rigid attachment is assumed, not proven. Unobserved target and scene surfaces remain unknown.'}


def check_carried_configuration(model, joints, aperture, scene_points, geometry, margin_m):
    """Combine existing robot checks with this additional observed payload proxy."""
    robot = model.collision(joints, aperture, scene_points, margin=margin_m)
    pose = model.eef_pose()
    carried = geometry.scene_collision(pose, scene_points, margin_m)
    _, arm_hits = model.points_in_robot(geometry.world_points(pose), margin=margin_m,
                                       exclude_gripper=True)
    return {'collision': bool(robot['collision'] or carried['collision'] or arm_hits),
        'robot': robot, 'carried': carried, 'carried_points_in_non_gripper_robot': arm_hits}
