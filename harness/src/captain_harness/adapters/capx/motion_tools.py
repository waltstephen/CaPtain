"""Public-sensor motion capabilities mixed into the CaP-X adapter.

Large geometry stays in the episode registry; planned and measured joint paths
also enter full execution logs with on-demand decision feedback. These tools do
not access the simulator; their robot dependency supplies a hardware model and
measured execution. Semantic target choice remains with the caller.
"""
from __future__ import annotations

from dataclasses import replace
from copy import deepcopy
from collections.abc import Mapping
import time
from captain_harness.ports.deadline import check_deadline, deadline_scope, DeadlineExceeded
import numpy as np
from scipy.spatial.transform import Rotation
from captain_harness.ports.capability import CapabilityResult, FailureCategory
from captain_harness.adapters.robotics.planning import SensorScene, MotionPlan, plan_joint_path
from captain_harness.adapters.robotics.motion_precision import motion_precision
from .contact_tools import ContactToolMixin


def pose_matrix(position, quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-10:
        raise ValueError('orientation must be a finite nonzero quaternion')
    p = np.asarray(position, dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all():
        raise ValueError('position must be a finite three-vector')
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(np.roll(q / np.linalg.norm(q), -1)).as_matrix()
    result[:3, 3] = p
    return result


def quaternion(pose):
    return np.roll(Rotation.from_matrix(pose[:3, :3]).as_quat(), 1).tolist()


class MotionToolMixin(ContactToolMixin):
    def _motion_model(self):
        return self.robot.planning_model()

    def _scene(self, handle, *, fresh=False):
        scene = self.registry.get(handle, expected_kind='sensor_scene')
        if fresh:
            state = self.robot.get_robot_state()
            if not state.success or state.value is None:
                raise ValueError('robot state is unavailable')
            now = state.value
            if (not np.allclose(now.joint_positions_rad, scene.joints, rtol=0, atol=1e-6)
                    or abs(now.gripper_open_fraction - scene.aperture) > 1e-6
                    or (scene.time_s is not None and abs(self.robot.time_s - scene.time_s) > 1e-9)):
                raise ValueError('stale_scene: observe and build_scene after movement or time advance')
        return scene

    def build_scene(self, observation_handle):
        from .backend import _depth_to_base_points
        obs = self.registry.get(observation_handle, expected_kind='observation')
        clouds = [_depth_to_base_points(c.depth, c.intrinsics, c.extrinsics)
                  for c in obs.cameras if c.depth is not None]
        if not clouds or not sum(map(len, clouds)):
            return CapabilityResult.failed(FailureCategory.PERCEPTION_UNCERTAIN, 'no valid measured depth')
        points = np.concatenate(clouds)
        model = self._motion_model()
        state = obs.robot_state
        model.set_state(state.joint_positions_rad, state.gripper_open_fraction)
        own, _ = model.points_in_robot(points, margin=.003, include_visual=True)
        points = points[~own]
        # Retain measured points, not voxel centroids that may lie inside surfaces.
        _, unique = np.unique(np.floor(points / .004).astype(np.int64), axis=0, return_index=True)
        points = points[np.sort(unique)]
        scene = SensorScene(points, observation_handle, tuple(state.joint_positions_rad),
                            float(state.gripper_open_fraction), obs.time_s, .003 + np.sqrt(3) * .004,
                            int(own.sum()))
        handle = self.registry.put('sensor_scene', scene)
        return CapabilityResult.ok({'scene_handle': handle, 'point_count': len(points),
            'source_observation_handle': observation_handle, 'time_s': obs.time_s,
            'robot_points_removed': int(own.sum()), 'voxel_size_m': .004,
            'collision_margin_m': scene.margin_m,
            'coverage': 'measured surfaces only; occluded/unobserved space is unknown',
            'robot_geometry_source': 'detached robot hardware description; no environment bodies'})

    def _candidate_scene_target(self, candidate, scene_handle):
        return self._selected_scene_target(candidate["detection_id"], scene_handle)

    def _selected_scene_target(self, detection_id, scene_handle):
        from .backend import _depth_to_base_points
        scene = self._scene(scene_handle, fresh=True)
        detection = self._require_selected_detection(detection_id)
        obs = self.registry.get(scene.source_observation_handle, expected_kind='observation')
        if detection.observation_handle != scene.source_observation_handle:
            from .observation_equivalence import same_public_snapshot
            source = self.registry.get(detection.observation_handle, expected_kind='observation')
            if not same_public_snapshot(source, obs):
                raise ValueError('selected target and current scene observations differ; acquire fresh target geometry')
        camera = self._camera_by_name(obs, detection.camera_name)
        if camera.depth is None:
            raise ValueError('target has no measured depth')
        target = _depth_to_base_points(camera.depth, camera.intrinsics, camera.extrinsics, mask=detection.mask)
        if len(target) < 3:
            raise ValueError('target geometry is insufficient')
        return scene, target

    def _grasp_scene_handle(self, candidate, scene_handle):
        if scene_handle is not None:
            return scene_handle
        from .backend import ToolOperationError
        detection = self._require_selected_detection(candidate['detection_id'])
        result = self.build_scene(detection.observation_handle)
        if not result.success:
            raise ToolOperationError(result.failure_category, result.message)
        return result.value['scene_handle']

    def evaluate_grasp(self, candidate, scene_handle=None):
        candidate = self._resolve_projected_grasp(candidate)
        scene_handle = self._grasp_scene_handle(candidate, scene_handle)
        scene, target = self._candidate_scene_target(candidate, scene_handle)
        position, q, direction, ep, eq = self._candidate_execution_pose(candidate)
        direction = np.asarray(direction, dtype=float)
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            raise ValueError('approach direction must be nonzero')
        direction /= norm
        contact = self._grasp_contact_evidence(pose_matrix(position, q), target)
        _, local = self._contact_local_points(pose_matrix(position, q), target)
        if 'pad_sweep_min_local_m' in contact:
            inside = np.all((local >= np.asarray(contact['pad_sweep_min_local_m'])-.003) &
                            (local <= np.asarray(contact['pad_sweep_max_local_m'])+.003), axis=1)
            contact['visible_point_count_in_pad_sweep'] = int(inside.sum())
            interior = np.all((local >= np.asarray(contact['pad_sweep_min_local_m'])) &
                              (local <= np.asarray(contact['pad_sweep_max_local_m'])), axis=1)
            contact['visible_point_count_in_pad_interior'] = int(interior.sum())
            contact['pad_interior_basis'] = 'Actual calibrated pad sweep without the contact-position tolerance dilation.'
        model = self._motion_model()
        model.set_state(scene.joints, 1.)
        goal = pose_matrix(ep, eq)
        hits, first = {}, None
        approach_distance = self._grasp_approach_distance(candidate)
        samples = max(2, int(np.ceil(approach_distance / .003)) + 1)
        for distance in np.linspace(approach_distance, 0., samples):
            pose = goal.copy()
            pose[:3, 3] -= direction * distance
            # Contact approach uses the existing 3 mm contact tolerance. Free
            # transport planning separately includes a larger voxel-coverage buffer.
            _, collision = model.points_in_robot(scene.points, margin=.003,
                                                  gripper_only=True, eef_override=pose)
            if collision:
                if first is None:
                    first = float(distance)
                for name, count in collision.items():
                    hits[name] = max(hits.get(name, 0), count)
        return CapabilityResult.ok({'contact': contact,
            'approach': {'observed_collision': bool(hits), 'geometry_hits': hits,
                         'first_collision_distance_before_contact_m': first, 'samples': samples,
                         'approach_distance_m': approach_distance,
                         'collision_margin_m': .003,
                         'full_arm_path': 'not_checked', 'coverage': 'visible surfaces only',
                         'tracking_reserve_checked': False,
                         'scope': 'nominal gripper approach only; refine_grasp reports tracking-aware endpoint feasibility'},
            'source_observation_handle': scene.source_observation_handle,
            'scene_handle': scene_handle,
            'grasp_verified': False,
            'execution_feasibility': 'not established by nominal approach; inspect refined pregrasp collision policy and plan_motion',
            'limitation': 'Open-hand collision and pad overlap do not prove force closure; partial target geometry.'})

    def _refresh_grasp_seed(self, candidate, scene_handle, target_detection_id):
        # Caller selects the current target. Never match identities across time.
        if not candidate.get('grasp_handle'):
            raise ValueError('fresh-target refinement requires an issued prior grasp')
        candidate = self._resolve_projected_grasp(candidate)
        scene, target = self._candidate_scene_target(
            {'detection_id': target_detection_id}, scene_handle)
        envelope = self.robot.gripper_contact_envelope()
        if not envelope:
            raise ValueError('finger-pad calibration is unavailable')
        position, q, direction, _, _ = self._candidate_execution_pose(candidate)
        pose = pose_matrix(position, q)
        _, local = self._contact_local_points(pose, target)
        lower, upper = np.percentile(local, [5., 95.], axis=0)
        pad_center = (np.asarray(envelope['pad_min_local_m']) +
                      np.asarray(envelope['pad_max_local_m'])) / 2.
        pad_center[1] = 0.
        pose[:3, 3] += pose[:3, :3] @ ((lower + upper)/2. - pad_center)
        detection = self._require_selected_detection(target_detection_id)
        from .backend import _depth_to_base_points
        obs = self.registry.get(scene.source_observation_handle, expected_kind='observation')
        camera = self._camera_by_name(obs, detection.camera_name)
        full_points = _depth_to_base_points(camera.depth, camera.intrinsics, camera.extrinsics)
        track = self._track_selected_detection(target_detection_id, detection, full_points, target)
        # Construct from public geometry, rather than copy old pose authority,
        # native payload, scores, attachment or feasibility into the new track.
        seed = {'position_m': pose[:3, 3].tolist(), 'quaternion_wxyz': list(q),
                'approach_direction_world': list(direction),
                'approach_distance_m': self._grasp_approach_distance(candidate),
                'detection_id': target_detection_id, 'tracking_handle': track,
                'source_observation_handle': scene.source_observation_handle,
                'backend': 'public_geometry_refinement', 'backend_candidate': None,
                'model_score': None}
        provenance = {'prior_grasp_handle': candidate['grasp_handle'],
            'prior_detection_id': candidate['detection_id'],
            'target_detection_id': target_detection_id,
            'source_observation_handle': scene.source_observation_handle,
            'cross_observation_identity_verified': False,
            'seed_translation_from_prior_m': (pose[:3, 3] - position).tolist(),
            'before_after_scope': 'Freshly centered seed versus refined proposals, measured on the same current cloud; old target geometry is not reused.',
            'basis': 'Caller-selected fresh visible target; prior orientation and approach retained as a proposal, all geometry checks rerun.'}
        return seed, provenance

    def refine_grasp(self, candidate, scene_handle=None, target_detection_id=None):
        from captain_harness.adapters.robotics.surfaces import support_plane
        from captain_harness.adapters.robotics.grasp_proposals import surface_orientations
        candidate = self._resolve_projected_grasp(candidate)
        target_refresh = None
        if target_detection_id is not None:
            scene_handle = self._grasp_scene_handle({'detection_id': target_detection_id}, scene_handle)
            candidate, target_refresh = self._refresh_grasp_seed(candidate, scene_handle, target_detection_id)
        scene_handle = self._grasp_scene_handle(candidate, scene_handle)
        scene, target = self._candidate_scene_target(candidate, scene_handle)
        position, q, _, _, _ = self._candidate_execution_pose(candidate)
        original_pose = pose_matrix(position, q)
        envelope = self.robot.gripper_contact_envelope()
        if not envelope:
            return CapabilityResult.failed(FailureCategory.NOT_EXECUTED, 'finger-pad calibration is unavailable')
        pad_center = (np.asarray(envelope['pad_min_local_m']) + np.asarray(envelope['pad_max_local_m'])) / 2.
        pad_center[1] = 0.
        rotations = [('preserved_orientation', original_pose[:3,:3])]
        plane = support_plane(scene.points, target,
            self.gripper_profile.max_width_m + self.gripper_profile.fingertip_depth_m)
        if plane is not None:
            normal, offset, _ = plane
            rotations.extend(surface_orientations(original_pose[:3, :3], target, normal))
        before = self.evaluate_grasp(candidate, scene_handle)
        options = [(dict(candidate), before.value, 'original')]
        model = self._motion_model()
        collision, collision_policy = self._grasp_planning_collision_checker(model, scene, target)
        valid_pregrasp = lambda joints: not collision(joints)['collision']
        for name, rotation in rotations:
            proposed = original_pose.copy(); proposed[:3,:3] = rotation
            _, local = self._contact_local_points(proposed, target)
            lower, upper = np.percentile(local,[5.,95.],axis=0)
            proposed[:3,3] += rotation @ ((lower+upper)/2. - pad_center)
            proposal = {**candidate,'position_m':proposed[:3,3].tolist(),
                        'quaternion_wxyz':quaternion(proposed)}
            proposal.pop('grasp_handle', None)
            # This explicit geometry proposal cannot borrow a projected native payload.
            proposal['backend_candidate'] = None
            if name != 'preserved_orientation': proposal['approach_direction_world'] = rotation[:, 2].tolist()
            if plane is not None:
                _,_,_,ep,eq = self._candidate_execution_pose(proposal)
                model.set_state(scene.joints,1.)
                gap = model.gripper_support_distance(pose_matrix(ep,eq),plane[0],plane[1])
                proposal['position_m'] = (np.asarray(proposal['position_m']) + plane[0]*max(0.,.003-gap)).tolist()
            evidence = self.evaluate_grasp(proposal,scene_handle).value
            options.append((proposal,evidence,name))
        shortened = []
        for proposal, evidence, name in options:
            evidence['reachability'] = self._grasp_endpoint_reachability(
                proposal, scene.joints, pregrasp_valid=valid_pregrasp)
            checks = evidence['reachability']['checks']
            if checks['contact']['solution_returned'] and not checks['pregrasp']['solution_returned']:
                distance = self._grasp_approach_distance(proposal)
                # Bounded geometric search, independent of task labels or
                # success feedback. Never waive endpoint or collision checks.
                for attempt in range(3):
                    distance /= 2.
                    if distance < .005:
                        break
                    short = {**proposal, 'approach_distance_m': distance}
                    short.pop('grasp_handle', None)
                    short['backend_candidate'] = None
                    measured = self.evaluate_grasp(short, scene_handle).value
                    measured['reachability'] = self._grasp_endpoint_reachability(
                        short, scene.joints, pregrasp_valid=valid_pregrasp)
                    shortened.append((short, measured, name + f'_shorter_pregrasp_{attempt + 1}'))
                    if measured['reachability']['reachable']:
                        break
        options.extend(shortened)
        for _, evidence, _ in options:
            joints = evidence['reachability']['joints']
            evidence['pregrasp_collision'] = (
                {'checked': True, **collision(joints)} if joints is not None else
                {'checked': False, 'collision': None, 'reason': 'endpoint_ik_unavailable'})
            evidence['collision_policy'] = collision_policy
        def rank(option):
            proposed,evidence,_ = option
            collisions = sum(evidence['approach']['geometry_hits'].values())
            support = evidence['contact'].get('visible_point_count_in_pad_sweep',0)
            return (not evidence['reachability']['reachable'] or evidence['pregrasp_collision']['collision'] is not False,
                    not bool(support),bool(collisions),collisions,-support,
                    float(np.linalg.norm(np.asarray(proposed['position_m'])-position)))
        selected_index = min(range(len(options)),key=lambda i:rank(options[i]))
        refined, after, choice = options[selected_index]
        delta = np.asarray(refined['position_m'])-position
        alternatives = [{'kind':name,'position_m':p['position_m'],'quaternion_wxyz':p['quaternion_wxyz'],
                         'approach_distance_m':self._grasp_approach_distance(p),
                         'approach':e['approach'],'contact':e['contact'],
                         'reachability':e['reachability'],
                         'pregrasp_collision':e['pregrasp_collision']} for p,e,name in options]
        if selected_index == 0 and target_refresh is None:
            return CapabilityResult.ok({'candidate':refined,'translation_m':delta.tolist(),
                'before':before.value,'after':after,'alternatives':alternatives,'selected_index':0,
                'pose_changed':False,'reason':'no_geometry_proposal_improved_the_measured_checks',
                'requires_decision':True})
        refined.update(backend='public_geometry_refinement', model_score=None,
                       refinement_basis=choice+'; public support plane and calibrated finger-pad sweep',
                       source_observation_handle=scene.source_observation_handle)
        # Native network pose/score/placement binding no longer describes this pose.
        refined['backend_candidate'] = None
        refined.pop('feasibility_evidence', None)
        width = self._predicted_grasp_width(pose_matrix(refined['position_m'],refined['quaternion_wxyz']), target)
        refined['predicted_grasp_width_m'] = width
        refined['width_feasible'] = width is None or width <= self.gripper_profile.max_width_m
        refined['width_known'] = width is not None
        refined['convergence_risk'] = 'unassessed'
        direction = np.asarray(refined['approach_direction_world'], dtype=float)
        refined['top_down_alignment'] = float(np.clip(-direction[2] / np.linalg.norm(direction), 0., 1.))
        # The candidate does not retain the original request/preference style.
        # A new pose cannot inherit either request matching or ranking scores.
        refined['style_preference_score'] = None
        refined['request_style_match'] = 'unassessed'
        reachability = after['reachability']
        refined['reachable'] = refined['ik_feasible'] = reachability['reachable']
        refined['predicted_joint_travel_rad'] = (None if not reachability['reachable'] else
            float(np.sum(np.abs(np.asarray(reachability['joints']) - scene.joints))))
        refined['quality_flags'] = ['geometry_refinement_unscored']
        refined['target_contact_evidence'] = after['contact']
        refined = self._issue_grasp_candidate(refined)
        return CapabilityResult.ok({'candidate': refined, 'translation_m': delta.tolist(),
            'before': before.value, 'after': after,'alternatives':alternatives,'selected_index':selected_index,
            'pose_changed':bool(np.linalg.norm(delta) > 1e-9 or not np.allclose(
                pose_matrix(refined['position_m'], refined['quaternion_wxyz'])[:3, :3], original_pose[:3, :3])),
            'approach_changed':self._grasp_approach_distance(refined) != self._grasp_approach_distance(candidate),
            'target_refresh': target_refresh,
            'score_status': 'unscored geometry proposal; native score not transferred',
            'requires_decision': True})

    def capture_carried_geometry(self, tracking_handle, current_point_handle, use_multiview=True):
        from .carried_planning import capture_carried_geometry
        return capture_carried_geometry(self, tracking_handle, current_point_handle, use_multiview)

    def plan_motion(self, target_pose, position_reference, scene_handle, timeout_s=5., carried_geometry_handle=None):
        return self._plan_motion(target_pose, position_reference, scene_handle, timeout_s,
            **({'carried_geometry_handle': carried_geometry_handle} if carried_geometry_handle is not None else {}))

    def _scene_collision_checker(self, model, scene, target_points=None, *, carried=None):
        if carried is not None:
            if target_points is not None:
                raise ValueError('carried transport cannot borrow a grasp contact exception')
            from captain_harness.adapters.robotics.carried_geometry import partition_measured_target
            check_deadline()
            other, selected = partition_measured_target(scene.points, carried.points_world)
            check_deadline()
            contact = None
            policy = {'kind': 'robot_and_observed_carried_geometry',
                'selected_target_sample_count': len(selected), 'other_scene_point_count': len(other),
                'observed_surface_margin_m': scene.margin_m,
                'carried_geometry': deepcopy(carried.receipt),
                'scope': 'Only exact selected public target samples are excluded from the static scene; the detached carried mesh moves with the EEF.'}
        elif target_points is None:
            other, contact = scene.points, None
            policy = {'kind': 'free_transport', 'observed_surface_margin_m': scene.margin_m}
        else:
            from scipy.spatial import cKDTree
            # Only exact measured samples from the selected detection may use
            # contact clearance. Per-geometry motion bounds do not widen this set.
            selected = cKDTree(target_points).query(scene.points)[0] <= 1e-7
            other, contact = scene.points[~selected], scene.points[selected]
            policy = {'kind': 'selected_public_target_contact', 'gripper_target_margin_m': .003,
                'other_surface_and_arm_margin_m': scene.margin_m,
                'selected_target_sample_count': int(selected.sum()),
                'scope': 'Only selected measured samples versus the gripper use contact clearance; physical penetration remains a collision.'}
        def collision(joints, extra_margin=0., first_hit_only=False):
            per_geometry = isinstance(extra_margin, Mapping)
            uniform = 0. if per_geometry else extra_margin
            extra = {'geometry_padding_m': extra_margin} if per_geometry else {}
            inflated = any(v > 0 for v in extra_margin.values()) if per_geometry else uniform > 0
            return model.collision(joints, scene.aperture, other, scene.margin_m + uniform,
                contact_points=contact, contact_margin=.003 + uniform,
                first_hit_only=first_hit_only or inflated, **extra)
        return collision, policy

    def _tracking_reserved_checker(self, model, scene, nominal_collision, *, carried=None):
        """One tracking allowance for candidate endpoints and motion planning."""
        collision = nominal_collision
        tracking_reserve = None
        reserve_padding = {}
        tracking_policy = getattr(self.robot, 'planned_path_tracking_policy', lambda: None)()
        if tracking_policy is not None:
            check_deadline()
            reserve_padding = model.gripper_pose_error_padding(scene.joints, scene.aperture,
                position_tolerance_m=tracking_policy['position_tolerance_m'],
                orientation_tolerance_rad=tracking_policy['orientation_tolerance_rad'],
                point_padding_m=2. * scene.margin_m)
            check_deadline()
            tracking_reserve = {**deepcopy(tracking_policy), 'geometry_padding_m': dict(reserve_padding),
                'applies_to': 'IK and nominal/swept path planning only; execution retains original measured collision margins',
                'limitation': 'Rigid gripper allowance at fixed aperture, not guaranteed tracking accuracy; arm posture and unseen geometry are not covered.'}
            if carried is not None:
                tracking_reserve.update(scope='rigid_gripper_and_observed_carried_geometry',
                    limitation='Rigid gripper and captured payload allowance at fixed aperture; slip, unseen geometry and actual tracking remain unproven.')
            def collision(joints, extra_margin=0., first_hit_only=False):
                if isinstance(extra_margin, Mapping) and set(extra_margin) != set(reserve_padding):
                    raise ValueError('swept padding must match every reserved geometry')
                combined = {name: value + (extra_margin[name] if isinstance(extra_margin, Mapping) else extra_margin)
                    for name, value in reserve_padding.items()}
                return nominal_collision(joints, extra_margin=combined, first_hit_only=first_hit_only)
        return collision, tracking_reserve

    def _grasp_planning_collision_checker(self, model, scene, target):
        nominal, policy = self._scene_collision_checker(model, scene, target)
        collision, reserve = self._tracking_reserved_checker(model, scene, nominal)
        return collision, {**policy, 'planning_tracking_reserve': reserve,
            'check_scope': 'tracking-aware pregrasp endpoint; complete motion path not checked'}

    def _plan_motion(self, target_pose, position_reference, scene_handle, timeout_s=5., *, target_points=None, carried_geometry_handle=None):
        if not np.isfinite(timeout_s) or not .1 <= timeout_s <= 30.:
            raise ValueError('planning timeout must be in [.1, 30] seconds')
        began = time.monotonic()
        progress = {'planning_stage': 'entry'}
        with deadline_scope('planning', timeout_s=timeout_s):
            try:
                check_deadline()
                result = self._plan_motion_within_deadline(target_pose, position_reference,
                    scene_handle, target_points=target_points, progress=progress,
                    carried_geometry_handle=carried_geometry_handle)
                check_deadline()
                return result
            except DeadlineExceeded as exc:
                model = progress.pop('_model', None)
                if model is not None and progress.get('planning_stage') == 'ik':
                    progress['ik_last_iteration'] = deepcopy(getattr(model, 'last_ik', {}))
                return replace(CapabilityResult.failed(FailureCategory.TIMEOUT, 'deadline_expired',
                    value={**progress, 'reason': 'deadline_expired',
                    'planning_time_s': time.monotonic() - began,
                    'limitation': 'Interrupted planning is not a reachability or collision proof; no plan issued.'}),
                    diagnostics={'deadline': dict(exc.deadline_receipt)})

    def _plan_motion_within_deadline(self, target_pose, position_reference, scene_handle, *, target_points, progress, carried_geometry_handle=None):
        if self._active_attachment_handle and carried_geometry_handle is None:
            return CapabilityResult.failed(FailureCategory.NOT_EXECUTED,
                'held_object_planning_unavailable: capture_carried_geometry from the current accepted attachment mark, then pass its handle')
        if position_reference not in {'tool', 'end_effector'}:
            raise ValueError('position_reference must be tool or end_effector')
        progress['planning_stage'] = 'scene'
        check_deadline()
        scene = self._scene(scene_handle, fresh=True)
        check_deadline()
        p, q = self._pose_mapping(target_pose)
        if position_reference == 'tool':
            hand = p + pose_matrix(p, q)[:3, :3] @ self._tcp_offset_local_m
            p, q = self.robot.eef_pose_from_external_ik_target(hand, q)
        target = pose_matrix(p, q)
        progress['planning_stage'] = 'model'
        check_deadline()
        carried = None
        if carried_geometry_handle is not None:
            from .carried_planning import carried_context
            try:
                carried = carried_context(self, carried_geometry_handle, scene)
            except ValueError as exc:
                return CapabilityResult.failed(FailureCategory.NOT_EXECUTED, str(exc), value={'reason': str(exc)})
        model = self._motion_model() if carried is None else carried.model
        progress['_model'] = model
        check_deadline()
        nominal_collision, collision_policy = self._scene_collision_checker(model, scene, target_points,
            **({'carried': carried} if carried is not None else {}))
        progress['planning_stage'] = 'tracking_reserve'
        collision, tracking_reserve = self._tracking_reserved_checker(
            model, scene, nominal_collision, carried=carried)
        if tracking_reserve is not None:
            progress['planning_tracking_reserve'] = tracking_reserve
        reserve_padding = (tracking_reserve or {}).get('geometry_padding_m', {})
        progress.update(collision_policy=collision_policy,
            source_observation_handle=scene.source_observation_handle,
            input_pose_reference=position_reference)
        # This closure is confined to one unchanged public scene/aperture.
        # Execution later uses a fresh measured checker, never these caches.
        state_cache = {}
        cache_info = {'state_hits': 0, 'state_geometry_queries': 0}
        progress['collision_cache'] = cache_info
        def valid(joints):
            check_deadline()
            state = np.asarray(joints, dtype=np.float64)
            key = (state.shape, state.tobytes())
            if key in state_cache:
                cache_info['state_hits'] += 1
                check_deadline()
                return state_cache[key]
            cache_info['state_geometry_queries'] += 1
            measured = collision(joints, first_hit_only=True)
            check_deadline()
            state_cache[key] = not measured['collision']
            if measured['collision'] and 'first_invalid_state' not in progress:
                progress['first_invalid_state'] = {
                    'planning_stage': progress['planning_stage'], 'joints': np.asarray(joints).tolist(),
                    'collision_receipt': deepcopy(measured),
                    'scope': 'First rejected sampled configuration; counts may stop at first hit, not a global infeasibility proof.'}
            return not measured['collision']
        progress['planning_stage'] = 'start_validation'
        if not valid(scene.joints):
            progress['start_validation'] = {'status': 'rejected', 'ik_called': False,
                'path_search_called': False, 'diagnostics_complete': False}
            progress['start_collision'] = collision(scene.joints)
            check_deadline()
            progress['start_nominal_collision'] = nominal_collision(scene.joints)
            check_deadline()
            nominal_blocked = progress['start_nominal_collision']['collision']
            reason = 'start_in_observed_collision' if nominal_blocked else 'tracking_start_clearance_insufficient'
            progress['recovery_guidance'] = (
                'The same start is already rejected by the observed geometry checks. A longer search or '
                'a different goal alone cannot repair it. Inspect the source RGB-D and collision_evidence '
                'pixels when available: a partial selection, contaminated depth, conservative hull, or real '
                'obstacle can cause this conflict. Repair the public geometry/selection or choose a '
                'separately monitored recovery when justified, then rebuild current evidence and replan. '
                'No motion or contact-free path has been established.' if nominal_blocked else
                'The same start clears nominal geometry but lacks the declared tracking allowance. '
                'A longer search cannot certify this reserve; inspect current public geometry and choose '
                'a separately monitored recovery before replanning.')
            if nominal_blocked and carried is not None:
                from .planning_failure_evidence import carried_start_evidence
                progress['planning_stage'] = 'start_collision_evidence'
                progress['collision_evidence'] = carried_start_evidence(self, model, scene, carried)
                check_deadline()
            progress['start_validation']['diagnostics_complete'] = True
            return CapabilityResult.failed(FailureCategory.CONTROLLER_REJECTED, reason,
                value={**{k: v for k, v in progress.items() if k != '_model'}, 'reason': reason})
        start_pose = model.eef_pose(scene.joints, scene.aperture)
        precision = motion_precision(start_pose, target,
            min(.003, getattr(self, '_motion_position_tolerance_m', .003)),
            min(.03, getattr(self, '_motion_orientation_tolerance_rad', .03)))
        progress['planning_stage'] = 'ik'
        check_deadline()
        goal = model.solve_ik(target, scene.joints, aperture=scene.aperture, valid=valid,
            position_tolerance=min(.002, .5 * precision['position_tolerance_m']),
            orientation_tolerance=min(.02, .5 * precision['orientation_tolerance_rad']))
        check_deadline()
        progress['ik'] = deepcopy(getattr(model, 'last_ik', {}))
        if goal is None:
            collided = model.last_ik.get('pose_solution_count', 0) > 0
            reason = 'no_collision_free_ik_returned' if collided else 'ik_not_converged'
            endpoint_collision = collision(model.last_ik['joints'])
            check_deadline()
            details = {'reason': reason, 'ik': model.last_ik, 'collision_policy': collision_policy,
                'best_endpoint_collision': endpoint_collision}
            if tracking_reserve is not None:
                original_endpoint = nominal_collision(model.last_ik['joints'])
                check_deadline()
                details.update(planning_tracking_reserve=tracking_reserve,
                    best_endpoint_nominal_collision=original_endpoint)
                accurate = (model.last_ik.get('position_residual_m', float('inf')) <= min(.002, .5 * precision['position_tolerance_m'])
                    and model.last_ik.get('orientation_residual_rad', float('inf')) <= min(.02, .5 * precision['orientation_tolerance_rad']))
                if collided and accurate and not original_endpoint['collision'] and endpoint_collision['collision']:
                    reason = 'tracking_clearance_insufficient'
                    details.update(reason=reason,
                        recovery_guidance='The best measured-pose IK candidate clears nominal geometry but lacks the declared gripper tracking allowance. Choose a different public grasp/approach candidate or target pose and replan; repeating this exact target with a longer search does not resolve this endpoint witness. Other poses/configurations are not ruled out.')
            return CapabilityResult.failed(FailureCategory.CONTROLLER_REJECTED if reason == 'tracking_clearance_insufficient'
                else FailureCategory.IK_FAILURE, reason, value=details)
        progress['planning_stage'] = 'path_search'
        remaining = check_deadline().remaining()
        from captain_harness.adapters.robotics.swept_collision import SweptSceneValidator
        planning_collision = lambda joints, extra_margin=0.: collision(
            joints, extra_margin=extra_margin, first_hit_only=True)
        # Every point admitted by the original collision broad phase lies
        # within geometry radius +2*margin. Include that inflated envelope in
        # rotational travel bounds, rather than bounding physical vertices only.
        motion_bounds = getattr(model, 'geometry_point_motion', model.max_point_motion)
        swept = SweptSceneValidator(planning_collision,
            lambda delta: motion_bounds(delta, point_padding_m=2. * (scene.margin_m + max(reserve_padding.values(), default=0.))),
            cache_validated_edges=True)
        path, info = plan_joint_path(scene.joints, goal, model.limits, valid,
            timeout_s=remaining, edge_valid=swept)
        if tracking_reserve is not None:
            info['planning_tracking_reserve'] = tracking_reserve
        info['swept_scene_validation'] = swept.receipt()
        if tracking_reserve is not None:
            info['swept_scene_validation']['nominal_query_scope'] = 'original observed geometry plus the declared rigid-gripper tracking allowance'
        info['collision_cache'] = {**cache_info, 'cached_states': len(state_cache),
            'scope': 'exact configurations and clear edges within this unchanged planning scene only; execution checks fresh measurements'}
        progress['path_search'] = info
        check_deadline()
        if path is None:
            details = {**info, 'collision_policy': collision_policy,
                'start_collision': collision(scene.joints), 'goal_collision': collision(goal)}
            check_deadline()
            if tracking_reserve is not None and info['reason'] == 'start_in_observed_collision':
                original_start = nominal_collision(scene.joints)
                check_deadline()
                details['start_nominal_collision'] = original_start
                if not original_start['collision']:
                    details.update(reason='tracking_start_clearance_insufficient',
                        recovery_guidance='The current measured start clears nominal geometry but lacks the declared gripper tracking allowance. A longer search from the same start cannot certify this reserve; inspect current public geometry and choose a separately monitored recovery before replanning.')
            return CapabilityResult.failed(FailureCategory.CONTROLLER_REJECTED, details['reason'], value=details)
        progress['planning_stage'] = 'plan_issuance'
        check_deadline()
        plan = MotionPlan(scene_handle, path, scene.joints, scene.aperture, self.robot.controller,
                          target, info['planning_time_s'], info['checks'],
                          None if target_points is None else np.asarray(target_points).copy(),
                          deepcopy(precision), deepcopy(tracking_reserve), carried_geometry_handle)
        return CapabilityResult.ok({**info, 'endpoint_precision': deepcopy(precision),
            'plan_handle': self.registry.put('motion_plan', plan),
            'planned_joint_path': path.tolist(),
            'waypoint_count': len(path), 'path_joint_length_rad': float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
            'source_observation_handle': scene.source_observation_handle,
            'collision_policy': collision_policy,
            'collision_scope': ('robot and observed carried mesh versus self/observed surfaces; rigid attachment and unseen geometry unproven'
                if carried is not None else 'robot self collision and inflated observed surfaces; unseen geometry unknown'),
            **({'carried_geometry_handle': carried_geometry_handle} if carried is not None else {}),
            'input_pose_reference': position_reference, 'execution_controller': self.robot.controller})

    def execute_motion(self, plan_handle):
        plan = self.registry.get(plan_handle, expected_kind='motion_plan')
        scene = self._scene(plan.scene_handle, fresh=True)
        carried_handle = getattr(plan, 'carried_geometry_handle', None)
        if plan.controller != self.robot.controller or (self._active_attachment_handle and carried_handle is None):
            return CapabilityResult.failed(FailureCategory.NOT_EXECUTED, 'plan controller or attachment state changed')
        carried = None
        if carried_handle is not None:
            from .carried_planning import carried_context
            try:
                carried = carried_context(self, carried_handle, scene)
            except ValueError as exc:
                return CapabilityResult.failed(FailureCategory.NOT_EXECUTED, str(exc), value={'reason': str(exc)})
        model = self._motion_model() if carried is None else carried.model
        valid, collision_policy = self._scene_collision_checker(model, scene, plan.contact_target_points,
            **({'carried': carried} if carried is not None else {}))
        endpoint = ({'endpoint_target': plan.target, 'endpoint_precision': plan.endpoint_precision}
                    if plan.endpoint_precision is not None else {})
        result = self.robot.execute_joint_path(plan.waypoints, valid, **endpoint)
        if result.executed:
            self._aligned_pregrasp = None
        value = dict(result.value or {})
        if carried_handle is not None:
            value['carried_geometry_handle'] = carried_handle
        trace = value.get('trajectory', [])
        value.update(planning_tracking_reserve=deepcopy(plan.tracking_reserve),
                     trajectory_handle=self.registry.put('motion_trajectory', trace),
                     planned_joint_path=plan.waypoints.tolist(),
                     plan_handle=plan_handle, source_observation_handle=scene.source_observation_handle,
                     collision_policy=collision_policy)
        return replace(result, value=value)

    def _grasp_endpoint_reachability(self, candidate, seed, *, pregrasp_valid=None):
        goal, pregrasp = self._grasp_motion_poses(candidate)
        model = self._motion_model()
        checks, solutions = {}, {}
        for name, target in (("contact", goal), ("pregrasp", pregrasp)):
            solved = model.solve_ik(target, seed, aperture=1.,
                **({'valid': pregrasp_valid} if name == 'pregrasp' and pregrasp_valid is not None else {}))
            solutions[name] = solved
            checks[name] = {
                "solution_returned": solved is not None,
                "position_m": target[:3, 3].tolist(),
                "quaternion_wxyz": quaternion(target),
                "ik": deepcopy(model.last_ik),
            }
        reachable = all(value is not None for value in solutions.values())
        return {
            "reachable": reachable,
            "reason": "endpoint_solutions_returned" if reachable else "endpoint_ik_unavailable",
            "joints": solutions["pregrasp"].tolist() if reachable else None,
            "check_scope": "grasp_contact_and_pregrasp_endpoints",
            "position_reference": "end_effector", "solver": "robot_planning_model",
            "checks": checks, "motion_path": "not_checked",
            "approach_distance_m": self._grasp_approach_distance(candidate),
            "pregrasp_validity_checked": pregrasp_valid is not None,
            "limitation": "Endpoint IK does not prove a collision-free approach; use a fresh scene for planned alignment.",
        }

    def _grasp_motion_poses(self, candidate):
        """Exact EEF targets shared by grasp diagnostics and planned alignment."""
        candidate = self._resolve_projected_grasp(candidate)
        _, _, direction, ep, eq = self._candidate_execution_pose(candidate)
        direction = np.asarray(direction, dtype=float)
        if np.linalg.norm(direction) < 1e-10:
            raise ValueError('approach direction must be nonzero')
        goal = pose_matrix(ep, eq)
        pregrasp = goal.copy()
        pregrasp[:3, 3] -= direction / np.linalg.norm(direction) * self._grasp_approach_distance(candidate)
        return goal, pregrasp

    @staticmethod
    def _grasp_approach_distance(candidate):
        distance = candidate.get('approach_distance_m', .06)
        if isinstance(distance, bool) or not np.isfinite(distance) or not .005 <= distance <= .20:
            raise ValueError('approach_distance_m must be finite and in [0.005, 0.20]')
        return float(distance)

    def _planned_alignment(self, candidate, scene_handle):
        scene, target_points = self._candidate_scene_target(candidate, scene_handle)
        from .grasp_preparation import PLANNED_GRASP_MIN_OPEN_FRACTION
        if scene.aperture < PLANNED_GRASP_MIN_OPEN_FRACTION:
            return CapabilityResult.failed(FailureCategory.NOT_EXECUTED,
                'open_gripper_then_reobserve: planned alignment requires an already open gripper')
        _, pregrasp = self._grasp_motion_poses(candidate)
        target, eq = pregrasp[:3, 3], quaternion(pregrasp)
        planned = self._plan_motion({'position_m': target.tolist(), 'quaternion_wxyz': list(eq)},
                                   'end_effector', scene_handle, target_points=target_points)
        if not planned.success:
            return planned
        moved = self.execute_motion(planned.value['plan_handle'])
        if not moved.success:
            return moved
        status = self.robot.pregrasp_pose_status(target, eq)
        if not status.success or not status.value.get('valid'):
            return CapabilityResult.failed(FailureCategory.CONTROLLER_REJECTED,
                'planned_pregrasp_not_converged', executed=moved.executed, value=moved.value)
        return replace(moved, value={**moved.value, 'converged': True, 'phase_reached': 'pregrasp',
            'failed_phase': None, 'convergence_reason': 'planned_path_and_measured_pregrasp',
            'pregrasp_position_m': target.tolist(), 'hover_position_m': target.tolist(),
            'position_residual_m': status.value['position_residual_m'],
            'orientation_residual_rad': status.value['orientation_residual_rad'],
            'planning': planned.value})

    def track_target(self, detection_id, observation_handle):
        from .visual_tracking import track_rgbd
        detection = self._require_selected_detection(detection_id)
        old = self.registry.get(detection.observation_handle, expected_kind='observation')
        new = self.registry.get(observation_handle, expected_kind='observation')
        first = self._camera_by_name(old, detection.camera_name)
        latest = self._camera_by_name(new, detection.camera_name)
        result = track_rgbd(first, latest, detection.mask)
        result.update(source_observation_handle=detection.observation_handle,
                      observation_handle=observation_handle, source_detection_id=detection_id,
                      source_time_s=old.time_s, time_s=new.time_s,
                      identity_basis='image feature continuity; not semantic identity verification')
        result['tracking_evidence_handle'] = self.registry.put('target_motion', dict(result))
        return CapabilityResult.ok(result)

    def measure_goal_evidence(self, observation_handle, subject_detection_ids, claims,
                              previous_evidence_handle=None, supporting_evidence_handles=()):
        from .public_goal_evidence import collect_goal_evidence
        return collect_goal_evidence(self, observation_handle, subject_detection_ids, claims,
                                     previous_evidence_handle, supporting_evidence_handles)
