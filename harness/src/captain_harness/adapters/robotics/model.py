"""Detached robot-only kinematics and convex collision geometry.

The description is the robot hardware model, not the environment XML. This
module never receives an environment instance or reads simulated object state.
MuJoCo is used for FK/Jacobians and robot self collision, without stepping.
"""
from __future__ import annotations

from captain_harness.ports.deadline import check_deadline

from dataclasses import dataclass
from collections.abc import Mapping
from numbers import Real
from xml.etree import ElementTree as ET
import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation


def _convex_membership(local, planes, margin, *, first_hit_only=False):
    """Same halfspace test with bounded native work and optional early hit."""
    selected = np.zeros(len(local), dtype=bool)
    # A ball centered at the local origin with this radius lies inside every
    # inflated halfspace by Cauchy-Schwarz. It proves interior membership only;
    # all other points still use the exact halfspace test below.
    normal_bound = float(np.max(np.linalg.norm(planes[:, :3], axis=1)))
    interior_radius = (margin - float(np.max(planes[:, 3]))) / normal_bound
    for begin in range(0, len(local), 256):
        check_deadline()
        block = local[begin:begin + 256]
        known_inside = ((np.sum(block * block, axis=1) <= interior_radius ** 2)
                        if interior_radius >= 0 else np.zeros(len(block), dtype=bool))
        check_deadline()
        if first_hit_only and known_inside.any():
            selected[begin:begin + len(block)] = known_inside
            return selected
        retained = np.ones(len(block), dtype=bool)
        uncertain = np.flatnonzero(~known_inside)
        if len(uncertain):
            # Reject only by an existing halfspace. Ordering by the batch
            # centroid is a speed heuristic; survivors still check every face.
            center = np.mean(block[uncertain], axis=0)
            face = planes[int(np.argmax(planes[:, :3] @ center + planes[:, 3]))]
            separated = block[uncertain] @ face[:3] + face[3] > margin
            retained[uncertain[separated]] = False
            check_deadline()
        face_batch = max(32, 8192 // max(1, len(block)))
        for first in range(0, len(planes), face_batch):
            check_deadline()
            indices = np.flatnonzero(retained & ~known_inside)
            if not len(indices):
                break
            faces = planes[first:first + face_batch]
            # Avoid an unbounded dense BLAS query. Full public calls still
            # evaluate every point; planning certificates may stop on a hit.
            distance = np.einsum('ij,kj->ik', block[indices], faces[:, :3], optimize=False) + faces[:, 3]
            check_deadline()
            retained[indices[~(np.max(distance, axis=1) <= margin)]] = False
        selected[begin:begin + len(block)] = retained
        if first_hit_only and retained.any():
            break
    return selected


@dataclass(frozen=True)
class ConvexGeometry:
    name: str
    geom_id: int
    planes: np.ndarray
    radius: float
    moving: bool
    collidable: bool
    vertices: np.ndarray
    primitive: str | None = None
    size: tuple[float, ...] = ()


def box_signed_separation(center_a, rotation_a, half_a, center_b, rotation_b, half_b):
    """Exact OBB separating-axis test; negative means overlap, in meters."""
    delta = np.asarray(center_b)-center_a
    axes = [*np.asarray(rotation_a).T, *np.asarray(rotation_b).T]
    axes += [np.cross(a,b) for a in np.asarray(rotation_a).T for b in np.asarray(rotation_b).T]
    separation = -float("inf")
    for axis in axes:
        length = float(np.linalg.norm(axis))
        if length < 1e-10:
            continue
        axis = axis/length
        radius_a = float(np.abs(np.asarray(rotation_a).T@axis)@half_a)
        radius_b = float(np.abs(np.asarray(rotation_b).T@axis)@half_b)
        separation = max(separation, abs(float(delta@axis))-radius_a-radius_b)
    return separation


class RobotCollisionModel:
    def __init__(self, xml: str, *, joints: list[str], gripper_joints: list[str],
                 base: str, end_effector: str, gripper_prefix: str, angle_unit: str | None = None):
        import mujoco
        if angle_unit is not None:
            if angle_unit not in {'radian', 'degree'}:
                raise ValueError('robot description angle unit must be radian or degree')
            root = ET.fromstring(xml)
            compiler = root.find('compiler')
            if compiler is None:
                compiler = ET.SubElement(root, 'compiler')
            compiler.set('angle', angle_unit)
            xml = ET.tostring(root, encoding='unicode')
        self._description_xml = xml
        self._description_options = {'joints': list(joints), 'gripper_joints': list(gripper_joints),
            'base': base, 'end_effector': end_effector, 'gripper_prefix': gripper_prefix,
            'angle_unit': angle_unit}
        self.carried_geometry_name = None
        self._rigid_eef_geometry_names = set()
        self.mj = mujoco
        self.model = m = mujoco.MjModel.from_xml_string(xml)
        self.data = d = mujoco.MjData(m)
        self.joints = [m.joint(name).id for name in joints]
        self.qadr = m.jnt_qposadr[self.joints]
        self.vadr = m.jnt_dofadr[self.joints]
        self.gripper = [m.joint(name).id for name in gripper_joints]
        self.limits = m.jnt_range[self.joints].copy()
        self.eef_id, self.base_id = m.body(end_effector).id, m.body(base).id
        self.gripper_prefix = gripper_prefix
        self.geometries = []
        arm_joint_bodies = set(int(v) for v in m.jnt_bodyid[self.joints])
        for i in range(m.ngeom):
            collidable = bool(m.geom_contype[i] or m.geom_conaffinity[i])
            name = m.geom(i).name
            typ, size = int(m.geom_type[i]), m.geom_size[i]
            primitive = None
            if typ == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = int(m.geom_dataid[i])
                begin, count = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
                vertices = np.asarray(m.mesh_vert[begin:begin + count], dtype=float)
                planes = np.unique(np.round(ConvexHull(vertices).equations, 10), axis=0)
                radius = float(np.linalg.norm(vertices, axis=1).max())
            elif typ == mujoco.mjtGeom.mjGEOM_BOX:
                normals = np.concatenate([np.eye(3), -np.eye(3)])
                planes = np.column_stack([normals, -np.tile(size, 2)])
                radius = float(np.linalg.norm(size))
                vertices = np.asarray([[x,y,z] for x in (-size[0],size[0])
                                       for y in (-size[1],size[1]) for z in (-size[2],size[2])])
            elif typ in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CYLINDER,
                         mujoco.mjtGeom.mjGEOM_CAPSULE):
                primitive = {int(mujoco.mjtGeom.mjGEOM_SPHERE): 'sphere',
                             int(mujoco.mjtGeom.mjGEOM_CYLINDER): 'cylinder',
                             int(mujoco.mjtGeom.mjGEOM_CAPSULE): 'capsule'}[typ]
                radius = float(size[0] if primitive == 'sphere' else
                               np.hypot(size[0], size[1]) if primitive == 'cylinder' else
                               size[0] + size[1])
                # Round shapes use analytic membership and support, not a box
                # approximation that could erase nearby scene points.
                planes, vertices = np.empty((0,4)), np.empty((0,3))
            else:
                if collidable:
                    raise ValueError(f'unsupported robot collision geometry {name}: {typ}')
                continue
            body, moving = int(m.geom_bodyid[i]), False
            while body:
                moving |= body in arm_joint_bodies
                body = int(m.body_parentid[body])
            self.geometries.append(ConvexGeometry(name, i, planes, radius, moving, collidable,
                                                  vertices, primitive, tuple(float(x) for x in size)))
        self._closure_allowances = self._hardware_closure_allowances()
        self.set_state(np.mean(self.limits, axis=1), 1.)

    def _hardware_closure_allowances(self):
        # Detached hardware only: opposing sliding fingers at their declared
        # closed endpoint. Never classify palm, arm or observed payload contacts.
        m = self.model
        branches = {}
        for jid in self.gripper:
            if (m.jnt_type[jid] != self.mj.mjtJoint.mjJNT_SLIDE
                    or not m.jnt_limited[jid]
                    or not m.jnt_range[jid, 0] <= 0 <= m.jnt_range[jid, 1]):
                continue
            branches[int(m.jnt_bodyid[jid])] = jid
        def branch(geom):
            body = int(m.geom_bodyid[geom])
            while body:
                if body in branches:
                    return branches[body]
                body = int(m.body_parentid[body])
            return None
        self.set_state(np.mean(self.limits, axis=1), 0.)
        allowances = {}
        for contact in self.data.contact[:self.data.ncon]:
            a, b = int(contact.geom1), int(contact.geom2)
            ja, jb = branch(a), branch(b)
            if ja is None or jb is None or ja == jb or contact.dist >= -1e-5:
                continue
            # Check that opening these branches separates along opposing axes.
            axes = []
            for jid in (ja, jb):
                body = int(m.jnt_bodyid[jid])
                axis = self.data.xmat[body].reshape(3, 3) @ m.jnt_axis[jid]
                low, high = m.jnt_range[jid]
                endpoint = low if abs(low) > abs(high) else high
                axes.append(axis * np.sign(endpoint))
            if np.dot(*axes) > -.99:
                continue
            pair = tuple(sorted((a, b)))
            allowances[pair] = min(allowances.get(pair, 0.), float(contact.dist))
        return allowances

    def with_observed_carried_geometry(self, points_eef):
        """Return a detached planning model with a partial observed convex payload.

        The caller must establish current public attachment/observation binding.
        This operation cannot change the environment or physical robot model.
        """
        from .observed_payload_model import augment_robot_model
        return augment_robot_model(self, points_eef)

    def gripper_pose_error_padding(self, joints, aperture, *, position_tolerance_m,
                                   orientation_tolerance_rad, point_padding_m=0.):
        """Bound rigid gripper-envelope travel under a declared EEF pose error.

        Fixed aperture is required. This describes a geometric allowance, not a
        controller tracking guarantee; EEF error gives no bound for arm posture.
        """
        check_deadline()
        values = (position_tolerance_m, orientation_tolerance_rad, point_padding_m)
        if any(isinstance(v, bool) or not isinstance(v, Real) or not np.isfinite(v) or v < 0 for v in values):
            raise ValueError('pose-error allowances must be finite nonnegative distances and angles')
        if orientation_tolerance_rad > np.pi:
            raise ValueError('pose-error angle must be at most pi radians')
        eef = self.eef_pose(joints, aperture)
        padding = {}
        for geometry in self.geometries:
            check_deadline()
            if not geometry.collidable or not geometry.moving:
                continue
            value = 0.
            if (geometry.name.startswith(self.gripper_prefix)
                    or geometry.name in self._rigid_eef_geometry_names):
                center, _ = self.geom_pose(geometry)
                radius = float(np.linalg.norm(center - eef[:3, 3])) + geometry.radius + point_padding_m
                value = position_tolerance_m + 2. * np.sin(orientation_tolerance_rad / 2.) * radius
            padding[geometry.name] = float(value)
        return padding

    def max_point_motion(self, joint_delta, *, point_padding_m=0.):
        """Scalar upper bound retained for callers using uniform inflation."""
        return max(self.geometry_point_motion(joint_delta,
            point_padding_m=point_padding_m).values(), default=0.)

    def geometry_point_motion(self, joint_delta, *, point_padding_m=0.):
        """Upper bound on each collidable geometry's joint-linear point travel.

        For each geometry, sum |dq| times a conservative distance to each
        ancestor revolute axis (or one for a slide). Descendant link lengths,
        pivot offsets and bounded slide ranges give a configuration-independent
        radius; triangle inequality covers simultaneous articulated motion.
        Gripper configuration stays fixed, with its full slide range bounded.
        point_padding_m includes the collision envelope, not just physical
        vertices, when certifying a clearance policy under rotation.
        """
        delta = np.asarray(joint_delta, dtype=float)
        if delta.shape != self.qadr.shape or not np.isfinite(delta).all():
            raise ValueError('joint motion must be finite and match the robot')
        if not np.isfinite(point_padding_m) or point_padding_m < 0:
            raise ValueError('point padding must be finite and nonnegative')
        cache = getattr(self, '_point_motion_coefficients', {})
        coefficients = cache.get(float(point_padding_m))
        if coefficients is None:
            m, mj = self.model, self.mj
            controlled = {jid: i for i, jid in enumerate(self.joints)}
            rows = []
            for geom in self.geometries:
                if not geom.collidable or not geom.moving:
                    continue
                row = np.zeros(len(self.joints))
                radius = float(np.linalg.norm(m.geom_pos[geom.geom_id])) + geom.radius + point_padding_m
                body = int(m.geom_bodyid[geom.geom_id])
                while body and body != self.base_id:
                    first, count = int(m.body_jntadr[body]), int(m.body_jntnum[body])
                    for jid in reversed(range(first, first + count)):
                        kind = int(m.jnt_type[jid])
                        pivot = float(np.linalg.norm(m.jnt_pos[jid]))
                        if kind == mj.mjtJoint.mjJNT_HINGE:
                            if jid in controlled:
                                row[controlled[jid]] = radius + pivot
                            # Rotation about an offset pivot can translate the
                            # body origin by at most twice that pivot radius.
                            radius += 2. * pivot
                        elif kind == mj.mjtJoint.mjJNT_SLIDE:
                            if not m.jnt_limited[jid]:
                                raise ValueError('swept geometry requires bounded slide joints')
                            if jid in controlled:
                                row[controlled[jid]] = 1.
                            radius += float(np.max(np.abs(m.jnt_range[jid])))
                        else:
                            raise ValueError('swept geometry supports hinge and slide chains only')
                    radius += float(np.linalg.norm(m.body_pos[body]))
                    body = int(m.body_parentid[body])
                rows.append(row)
            coefficients = np.asarray(rows).reshape(-1, len(self.joints))
            cache[float(point_padding_m)] = coefficients
            self._point_motion_coefficients = cache
        names = [geom.name for geom in self.geometries if geom.collidable and geom.moving]
        return dict(zip(names, map(float, coefficients @ np.abs(delta))))

    def set_state(self, joints, aperture: float):
        q = np.asarray(joints, dtype=float)
        if q.shape != self.qadr.shape or not np.isfinite(q).all():
            raise ValueError('joint configuration must be finite and match the robot')
        self.data.qpos[self.qadr] = q
        for jid in self.gripper:
            low, high = self.model.jnt_range[jid]
            # Open is the endpoint farther from zero for either finger sign.
            self.data.qpos[self.model.jnt_qposadr[jid]] = (
                low if abs(low) > abs(high) else high) * aperture
        self.mj.mj_forward(self.model, self.data)

    def _base(self):
        return (self.data.xpos[self.base_id], self.data.xmat[self.base_id].reshape(3, 3))

    def eef_pose(self, joints=None, aperture=1.):
        if joints is not None:
            self.set_state(joints, aperture)
        b, r = self._base()
        pose = np.eye(4)
        pose[:3, 3] = (self.data.xpos[self.eef_id] - b) @ r
        pose[:3, :3] = r.T @ self.data.xmat[self.eef_id].reshape(3, 3)
        return pose

    def geom_pose(self, geom):
        b, r = self._base()
        i = geom.geom_id
        return ((self.data.geom_xpos[i] - b) @ r,
                r.T @ self.data.geom_xmat[i].reshape(3, 3))

    def points_in_robot(self, points, *, margin=0.003, moving_only=False,
                        gripper_only=False, eef_override=None, include_visual=False,
                        exclude_gripper=False, first_hit_only=False, geometry_padding_m=None):
        """Conservative convex inclusion; returns a per-point mask and counts."""
        raw = points
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        cached = getattr(self, '_point_trees', [])
        tree = next((tree for owner, tree in cached if owner is raw), None)
        if tree is None:
            tree = cKDTree(points)
            self._point_trees = [*cached[-3:], (raw, tree)]
        inside = np.zeros(len(points), dtype=bool)
        hits = {}
        original = self.eef_pose()
        transform = None if eef_override is None else np.asarray(eef_override) @ np.linalg.inv(original)
        for geom in self.geometries:
            if not geom.collidable and not include_visual:
                continue
            if moving_only and not geom.moving:
                continue
            if gripper_only and not geom.name.startswith(self.gripper_prefix):
                continue
            if exclude_gripper and geom.name.startswith(self.gripper_prefix):
                continue
            center, rotation = self.geom_pose(geom)
            if transform is not None:
                center = transform[:3, :3] @ center + transform[:3, 3]
                rotation = transform[:3, :3] @ rotation
            local_margin = margin + (0. if geometry_padding_m is None else geometry_padding_m.get(geom.name, 0.))
            indices = np.asarray(tree.query_ball_point(center, geom.radius + local_margin * 2), dtype=int)
            if not len(indices):
                continue
            for begin in range(0, len(indices), 256):
                check_deadline()
                batch = indices[begin:begin + 256]
                local = (points[batch] - center) @ rotation
                if geom.primitive == 'sphere':
                    selected = np.linalg.norm(local, axis=1) <= geom.size[0] + local_margin
                elif geom.primitive == 'capsule':
                    radial = np.linalg.norm(local[:, :2], axis=1)
                    axial = np.maximum(np.abs(local[:, 2]) - geom.size[1], 0.)
                    selected = np.hypot(radial, axial) <= geom.size[0] + local_margin
                elif geom.primitive == 'cylinder':
                    radial = np.maximum(np.linalg.norm(local[:, :2], axis=1) - geom.size[0], 0.)
                    axial = np.maximum(np.abs(local[:, 2]) - geom.size[1], 0.)
                    selected = np.hypot(radial, axial) <= local_margin
                else:
                    # Convex hull plane inflation is conservative at edges/corners.
                    selected = _convex_membership(local, geom.planes, local_margin,
                        first_hit_only=first_hit_only)
                inside[batch[selected]] = True
                if selected.any():
                    hits[geom.name] = hits.get(geom.name, 0) + int(selected.sum())
                    if first_hit_only:
                        return inside, hits
        return inside, hits

    def collision(self, joints, aperture, points, margin=0.003, *,
                  contact_points=None, contact_margin=0.003, first_hit_only=False, geometry_padding_m=None):
        if geometry_padding_m is not None:
            expected = {g.name for g in self.geometries if g.collidable and g.moving}
            if not isinstance(geometry_padding_m, Mapping) or set(geometry_padding_m) != expected:
                raise ValueError('geometry padding must name every collidable moving geometry exactly once')
            if any(isinstance(v, bool) or not isinstance(v, Real) or not np.isfinite(v) or v < 0
                   for v in geometry_padding_m.values()):
                raise ValueError('geometry padding must contain finite nonnegative distances')
        self.set_state(joints, aperture)
        self_contacts, allowed_contacts, self_details, contact_checks = [], [], [], []
        for contact in self.data.contact[:self.data.ncon]:
            distance = float(contact.dist)
            a, b = int(contact.geom1), int(contact.geom2)
            if (self.model.geom_type[a] == self.mj.mjtGeom.mjGEOM_BOX
                and self.model.geom_type[b] == self.mj.mjtGeom.mjGEOM_BOX):
                distance = box_signed_separation(self.data.geom_xpos[a],
                    self.data.geom_xmat[a].reshape(3,3), self.model.geom_size[a],
                    self.data.geom_xpos[b], self.data.geom_xmat[b].reshape(3,3), self.model.geom_size[b])
                if abs(distance-float(contact.dist)) > 1e-6:
                    contact_checks.append({"geometries":[self.model.geom(a).name,self.model.geom(b).name],
                        "provider_distance_m":float(contact.dist),"verified_box_separation_m":distance,
                        "basis":"independent 15-axis oriented-box geometry"})
            if distance < -1e-5:
                pair = tuple(sorted((int(contact.geom1), int(contact.geom2))))
                names = [self.model.geom(i).name for i in pair]
                bound = self._closure_allowances.get(pair)
                if bound is not None and distance >= bound - 1e-6:
                    allowed_contacts.append({"geometries": names, "distance_m": distance,
                        "closed_hardware_distance_m": bound, "numeric_tolerance_m": 1e-6})
                else:
                    self_contacts.append(names)
                    self_details.append({"geometries":names,"penetration_m":-distance})
        if first_hit_only and self_contacts:
            return {'collision': True, 'observed_collisions': {}, 'self_collisions': self_contacts, 'allowed_closure_contacts': allowed_contacts,
                    'self_collision_details':self_details,'contact_geometry_checks':contact_checks}
        _, hits = self.points_in_robot(points, margin=margin, moving_only=True,
            first_hit_only=first_hit_only, geometry_padding_m=geometry_padding_m)
        if contact_points is not None and len(contact_points) and not (first_hit_only and hits):
            _, arm_hits = self.points_in_robot(contact_points, margin=margin,
                moving_only=True, exclude_gripper=True, first_hit_only=first_hit_only, geometry_padding_m=geometry_padding_m)
            hand_hits = {}
            if not (first_hit_only and arm_hits):
                _, hand_hits = self.points_in_robot(contact_points, margin=contact_margin,
                    moving_only=True, gripper_only=True, first_hit_only=first_hit_only, geometry_padding_m=geometry_padding_m)
            for name, count in {**arm_hits, **hand_hits}.items():
                hits[name] = hits.get(name, 0) + count
        return {'collision': bool(hits or self_contacts), 'observed_collisions': hits,
                'self_collisions': self_contacts, 'allowed_closure_contacts': allowed_contacts,
                    'self_collision_details':self_details,'contact_geometry_checks':contact_checks}

    def gripper_support_distance(self, eef_pose, normal, offset):
        transform = np.asarray(eef_pose) @ np.linalg.inv(self.eef_pose())
        lowest = float('inf')
        for geom in self.geometries:
            if not geom.collidable or not geom.name.startswith(self.gripper_prefix): continue
            center, rotation = self.geom_pose(geom)
            if geom.primitive is not None:
                local_normal = (transform[:3,:3] @ rotation).T @ normal
                radial = geom.size[0] * float(np.linalg.norm(
                    local_normal[:2] if geom.primitive == 'cylinder' else local_normal))
                axial = (0. if geom.primitive == 'sphere' else
                         geom.size[1] * abs(float(local_normal[2])))
                transformed_center = transform[:3,:3] @ center + transform[:3,3]
                lowest = min(lowest, float(transformed_center @ normal + offset) - radial - axial)
                continue
            vertices = geom.vertices @ rotation.T + center
            vertices = vertices @ transform[:3,:3].T + transform[:3,3]
            lowest = min(lowest, float(np.min(vertices @ normal + offset)))
        return lowest

    def solve_ik(self, target, seed, *, aperture=1., max_iterations=180,
                 position_tolerance=0.002, orientation_tolerance=0.02, valid=None):
        """Bounded deterministic restarts on the hardware model; no scene access.

        Try the measured configuration first. A failed local solve is not a
        reachability proof, so also try configurations derived solely from
        joint limits. Every accepted solution must satisfy the same tolerances.
        """
        self.last_ik = {"iterations": 0, "status": "started"}
        check_deadline()
        initial = np.asarray(seed, dtype=float)
        if initial.shape != self.qadr.shape or not np.isfinite(initial).all():
            raise ValueError('IK seed must be finite and match the robot joints')
        center = np.mean(self.limits, axis=1)
        span = self.limits[:, 1] - self.limits[:, 0]
        signs = np.where(np.arange(len(center)) % 2, -1., 1.)
        starts = [initial, center, center + .2 * span * signs,
                  center - .2 * span * signs, center + .15 * span]
        used, attempts = [], []
        best, best_score, best_index = None, float('inf'), 0
        accepted, pose_solution_count = None, 0
        for start in starts:
            check_deadline()
            start = np.clip(start, self.limits[:, 0], self.limits[:, 1])
            if any(np.allclose(start, prior, rtol=0, atol=1e-9) for prior in used):
                continue
            used.append(start)
            solved = self._solve_ik_seed(target, start, aperture=aperture,
                max_iterations=max_iterations, position_tolerance=position_tolerance,
                orientation_tolerance=orientation_tolerance)
            check_deadline()
            diagnostic = dict(self.last_ik)
            if solved is not None:
                pose_solution_count += 1
            admissible = solved is not None and (valid is None or bool(valid(solved)))
            check_deadline()
            attempts.append({k: diagnostic[k] for k in
                ('iterations', 'position_residual_m', 'orientation_residual_rad')})
            score = (diagnostic['position_residual_m'] / position_tolerance
                     + diagnostic['orientation_residual_rad'] / orientation_tolerance)
            if admissible or score < best_score:
                best, best_score, best_index = diagnostic, score, len(attempts) - 1
            if admissible:
                accepted = solved
                break
        self.last_ik = {**best, 'seed_attempts': len(attempts),
            'total_iterations': sum(a['iterations'] for a in attempts),
            'best_seed_index': best_index, 'attempt_residuals': attempts,
            'pose_solution_count': pose_solution_count, 'valid_solution': accepted is not None}
        if accepted is None:
            self.set_state(best['joints'], aperture)
        return accepted

    def _solve_ik_seed(self, target, seed, *, aperture, max_iterations,
                       position_tolerance, orientation_tolerance):
        q = np.clip(np.asarray(seed, dtype=float), self.limits[:, 0], self.limits[:, 1])
        target = np.asarray(target, dtype=float).reshape(4, 4)
        for _ in range(max_iterations):
            check_deadline()
            self.set_state(q, aperture)
            actual = self.eef_pose()
            dp = target[:3, 3] - actual[:3, 3]
            dr = Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec()
            self.last_ik = {'iterations': _ + 1, 'position_residual_m': float(np.linalg.norm(dp)),
                            'orientation_residual_rad': float(np.linalg.norm(dr)), 'joints': q.tolist()}
            check_deadline()
            if np.linalg.norm(dp) <= position_tolerance and np.linalg.norm(dr) <= orientation_tolerance:
                return q.copy()
            jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            self.mj.mj_jacBody(self.model, self.data, jp, jr, self.eef_id)
            _, base_r = self._base()
            jac = np.vstack([base_r.T @ jp[:, self.vadr], base_r.T @ jr[:, self.vadr]])
            error = np.r_[dp, dr]
            step = jac.T @ np.linalg.solve(jac @ jac.T + .0025 * np.eye(6), error)
            step *= min(1., .12 / max(np.max(np.abs(step)), 1e-12))
            q = np.clip(q + step, self.limits[:, 0] + 1e-5, self.limits[:, 1] - 1e-5)
        return None
