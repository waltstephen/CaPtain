"""Explicit endpoint pose search for the sensor-motion capability profile."""
import math
import numpy as np
from scipy.spatial.transform import Rotation

from captain_harness.adapters.robotics.pose_candidates import solve_pose_candidates
from captain_harness.ports.capability import CapabilityResult, CapabilitySpec, FailureCategory, SideEffect

from .motion_tools import pose_matrix, quaternion


class PoseCandidateToolMixin:
    def propose_pose_candidates(self, position, quaternions=None, position_reference=None,
                                tracking_handle=None, base_quaternion=None,
                                rotation_axis=None, angle_offsets_rad=None):
        if position_reference not in {'tool', 'end_effector', 'held_target'}:
            raise ValueError('position_reference must be tool, end_effector or held_target')
        offset_diagnostics = {}
        axis_inputs = (base_quaternion, rotation_axis, angle_offsets_rad)
        if any(value is not None for value in axis_inputs):
            if quaternions is not None or not all(value is not None for value in axis_inputs):
                raise ValueError('supply either quaternions or all three explicit axis-offset inputs')
            axis = np.asarray(rotation_axis, dtype=float)
            angles = np.asarray(angle_offsets_rad, dtype=float)
            if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-10:
                raise ValueError('rotation_axis must be a finite nonzero robot-base vector')
            if angles.ndim != 1 or not 1 <= len(angles) <= 12 or not np.isfinite(angles).all():
                raise ValueError('provide 1-12 finite angle offsets in radians')
            normalized_angles = [math.remainder(float(angle), 2 * math.pi) for angle in angles]
            offset_diagnostics = {'requested_angle_offsets_rad': angles.tolist(),
                'normalized_angle_offsets_rad': normalized_angles,
                'angle_normalization': 'equivalent endpoint orientation modulo 2*pi; no physical rotation path'}
            base = pose_matrix([0, 0, 0], base_quaternion)
            quaternions = []
            for angle in normalized_angles:
                rotated = base.copy()
                rotated[:3, :3] = Rotation.from_rotvec(axis / np.linalg.norm(axis) * angle).as_matrix() @ base[:3, :3]
                quaternions.append(quaternion(rotated))
        if quaternions is None:
            raise ValueError('explicit allowed orientations are required')
        if not 1 <= len(quaternions) <= 12:
            raise ValueError('provide between 1 and 12 explicitly allowed orientations')
        poses = [pose_matrix(position, q) for q in quaternions]
        normalized = [quaternion(pose) for pose in poses]
        reference_offset = np.zeros(3)
        if position_reference == 'held_target':
            if tracking_handle is None or tracking_handle != self._active_attachment_handle:
                return CapabilityResult.failed(FailureCategory.PERCEPTION_UNCERTAIN,
                    'held_target_pose_unknown: use the currently verified attachment handle',
                    value={'reason': 'held_target_pose_unknown'})
            tracked = self.registry.get(tracking_handle, expected_kind='entity_track')
            if not tracked.attachment_verified or tracked.eef_to_object_transform is None:
                return CapabilityResult.failed(FailureCategory.PERCEPTION_UNCERTAIN,
                    'held_target_pose_unknown: verify the held target from public observations',
                    value={'reason': 'held_target_pose_unknown'})
            reference_offset = np.asarray(tracked.eef_to_object_transform)[:3, 3].copy()
        elif tracking_handle is not None:
            raise ValueError('tracking_handle applies only to held_target position_reference')

        rotations = []
        for pose, quat in zip(poses, normalized):
            if position_reference == 'end_effector':
                rotation = pose[:3, :3]
            else:
                hand_position = pose[:3, :3] @ self._tcp_offset_local_m
                ep, eq = self.robot.eef_pose_from_external_ik_target(hand_position, quat)
                rotation = pose_matrix(ep, eq)[:3, :3]
                if position_reference == 'tool':
                    reference_offset = -rotation.T @ np.asarray(ep)
            rotations.append(rotation)

        state = self.robot.get_robot_state()
        if not state.success or state.value is None:
            return CapabilityResult.failed(FailureCategory.PERCEPTION_UNCERTAIN,
                'fresh robot state unavailable', value={'reason': 'robot_state_unavailable'})
        measured = state.value
        model = self.robot.planning_model()
        observed = pose_matrix(measured.end_effector_position_m, measured.end_effector_quaternion_wxyz)
        fk = model.eef_pose(measured.joint_positions_rad, aperture=measured.gripper_open_fraction)
        position_error = float(np.linalg.norm(fk[:3, 3] - observed[:3, 3]))
        orientation_error = float(Rotation.from_matrix(fk[:3, :3] @ observed[:3, :3].T).magnitude())
        if (not np.isfinite(position_error) or not np.isfinite(orientation_error)
                or position_error > .002 or orientation_error > .002):
            return CapabilityResult.failed(FailureCategory.PERCEPTION_UNCERTAIN,
                'robot_model_state_mismatch', value={'reason': 'robot_model_state_mismatch',
                    'current_fk_error_m': position_error, 'current_fk_error_rad': orientation_error})
        result = solve_pose_candidates(model, reference_position_m=position,
            eef_rotations=rotations, reference_offset_in_eef_m=reference_offset,
            seed=measured.joint_positions_rad, aperture=measured.gripper_open_fraction,
            position_tolerance_m=self._motion_position_tolerance_m,
            orientation_tolerance_rad=self._motion_orientation_tolerance_rad)
        for entry, quat in zip(result['candidates'], normalized):
            entry['goto_arguments'] = {'position': poses[0][:3, 3].tolist(),
                'quaternion': quat, 'position_reference': position_reference,
                **({'tracking_handle': tracking_handle} if position_reference == 'held_target' else {})}
        result.update(offset_diagnostics)
        result.update(input_pose_reference=position_reference,
            current_fk_error_m=position_error, current_fk_error_rad=orientation_error,
            seed_joint_positions_rad=list(measured.joint_positions_rad),
            tracking_handle=tracking_handle,
            attachment_scope='active_verified_estimate' if tracking_handle else 'not_used')
        return CapabilityResult.ok(result)


def pose_candidate_capability():
    vector = {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}
    quat = {'type': 'array', 'items': {'type': 'number'}, 'minItems': 4, 'maxItems': 4}
    properties = {'position': vector,
        'quaternions': {'type': ['array', 'null'], 'items': quat, 'minItems': 1, 'maxItems': 12, 'default': None},
        'position_reference': {'enum': ['tool', 'end_effector', 'held_target']},
        'tracking_handle': {'type': ['string', 'null'], 'pattern': '^artifact://v1/', 'default': None},
        'base_quaternion': {**quat, 'type': ['array', 'null'], 'default': None},
        'rotation_axis': {**vector, 'type': ['array', 'null'], 'default': None},
        'angle_offsets_rad': {'type': ['array', 'null'], 'items': {'type': 'number'},
                              'minItems': 1, 'maxItems': 12, 'default': None}}
    return CapabilitySpec('propose_pose_candidates',
        'Evaluate 1-12 explicitly allowed orientations at the same reference-point position using '
        'current proprioception and the independent robot model. '
        'Orientations may be supplied as quaternions, or as base_quaternion plus a robot-base '
        'rotation_axis and explicit angle_offsets_rad (left-applied about that axis); these input forms '
        'are mutually exclusive. Finite offsets are reduced modulo 2*pi for equivalent endpoint orientations; '
        'original and normalized angles are returned. This does not request a physical multi-turn rotation. '
        'No extra orientations are inferred. '
        'Quaternion conventions and returned goto_arguments match goto_pose: hand orientation for tool/held_target, EEF orientation for '
        'end_effector. held_target requires the active verified attachment; rotating it preserves the '
        'measured target reference point, not the EEF position. Inspect ranked_candidate_indices and '
        'each failed endpoint. No numerical solution is not a reachability proof. '
        'This computes endpoint IK only: no path, carried-object collision, retention or task-orientation '
        'check and no executable motion plan. Supply only orientations compatible with the task and '
        'grasp. Execution must use the existing motion tools and their current preconditions.',
        {'type': 'object', 'properties': properties,
         'required': ['position', 'position_reference'], 'additionalProperties': False,
         'oneOf': [
             {'required': ['quaternions'], 'properties': {'quaternions': {'type': 'array'},
                **{key: {'type': 'null'} for key in ('base_quaternion', 'rotation_axis', 'angle_offsets_rad')}}},
             {'required': ['base_quaternion', 'rotation_axis', 'angle_offsets_rad'], 'properties': {
                'quaternions': {'type': 'null'},
                **{key: {'type': 'array'} for key in ('base_quaternion', 'rotation_axis', 'angle_offsets_rad')}}}]},
        {'type': 'object', 'properties': {
            'candidates': {'type': 'array', 'minItems': 1, 'maxItems': 12,
                'items': {'type': 'object', 'properties': {
                    'candidate_index': {'type': 'integer', 'minimum': 0},
                    'endpoint_valid': {'type': 'boolean'},
                    'goto_arguments': {'type': 'object'},
                    'ik': {'type': 'object', 'x-decision-feedback': 'on_demand'},
                    'target_eef_pose': {'type': 'array', 'x-decision-feedback': 'on_demand'}},
                    'required': ['candidate_index', 'endpoint_valid', 'goto_arguments'],
                    'additionalProperties': True}},
            'ranked_candidate_indices': {'type': 'array', 'items': {'type': 'integer', 'minimum': 0}, 'maxItems': 12},
            **{name: {'const': 'not_checked'} for name in
               ('motion_path', 'carried_object_collision', 'object_retention', 'task_semantics')}},
         'required': ['candidates', 'ranked_candidate_indices', 'motion_path',
                      'carried_object_collision', 'object_retention', 'task_semantics'],
         'additionalProperties': True},
        {'position': 'meter', 'joints': 'radian', 'quaternions': 'wxyz'},
        {'position': 'robot_base with explicit reference', 'quaternions': 'same as goto_pose'},
        180., SideEffect.COMPUTE, argument_order=tuple(properties),
        real_robot_equivalent='Measured robot state, calibration, public attachment estimate and robot hardware model')
