"""Public contact-motion wrappers and contracts, independently of task labels."""
from dataclasses import replace
from copy import deepcopy

from captain_harness.ports.capability import CapabilitySpec, CapabilityResult, SideEffect, ControlSemantic


class ContactToolMixin:
    def read_motion_trajectory(self, trajectory_handle, offset=0, limit=20):
        if (isinstance(offset, bool) or not isinstance(offset, int) or offset < 0
                or isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100):
            raise ValueError('offset must be a nonnegative integer and limit must be in [1,100]')
        trajectory = self.registry.get(trajectory_handle, expected_kind='motion_trajectory')
        end = min(len(trajectory), offset + limit)
        return CapabilityResult.ok({'trajectory_handle': trajectory_handle, 'offset': offset,
            'total_samples': len(trajectory), 'next_offset': end if end < len(trajectory) else None,
            'samples': deepcopy(trajectory[offset:end])})

    @staticmethod
    def _validate_motion_intent(motion_intent, target_detection_id):
        if motion_intent not in {"free_motion", "target_contact"}:
            raise ValueError("motion_intent must be free_motion or target_contact")
        if motion_intent == "target_contact" and not target_detection_id:
            raise ValueError("target_contact requires target_detection_id selected from the current public view")

    def _contact_result(self, result, target_detection_id=None):
        if result.executed:
            self._aligned_pregrasp = None
        if not isinstance(result.value, dict):
            return result
        value = dict(result.value)
        if target_detection_id is not None:
            history = {item['detection_id']: item for item in getattr(self, '_selection_history', ())}
            target_ref = target_detection_id
            seen = set()
            while target_ref in history and target_ref not in seen:
                seen.add(target_ref)
                item = history[target_ref]
                if item.get('identity_update') != 'same_target' or not item.get('previous_detection_id'):
                    break
                target_ref = item['previous_detection_id']
            progress = value.get('target_progress', {})
            status = {'progress_observed': 'progress_observed',
                      'no_observed_target_progress': 'no_progress',
                      'opposite_target_progress': 'contradicted'}.get(progress.get('status'), 'unknown')
            value['public_effect'] = {'status': status, 'target_ref': target_ref,
                'basis': 'public_rgbd_target_progress; target association is a policy claim',
                'evidence': deepcopy(progress)}
            value['motion_intent'] = 'target_contact'
        else:
            value['motion_intent'] = 'free_motion'
            value['public_effect'] = {'status': 'unknown',
                'basis': 'robot_pose_only; no target motion was measured'}
        # Keep full telemetry in the persisted execution result. Projection
        # hides it from routine model feedback; handles remain readable online.
        trajectory = value.get('trajectory')
        if trajectory is not None:
            value['trajectory_handle'] = self.registry.put('motion_trajectory', trajectory)
        return replace(result, value=value)

    def move_linear(self, direction, distance_m, max_speed_m_s=.02, max_force_n=18., max_duration_s=10.,
                    target_detection_id=None, motion_intent="free_motion"):
        self._validate_motion_intent(motion_intent, target_detection_id)
        if target_detection_id is not None:
            from .observed_contact import observed_contact
            return self._contact_result(observed_contact(self,target_detection_id,direction=direction,
                distance_m=distance_m,max_speed_m_s=max_speed_m_s,max_force_n=max_force_n,max_duration_s=max_duration_s), target_detection_id)
        return self._contact_result(self.robot.move_linear(direction, distance_m,
            max_speed_m_s=max_speed_m_s, max_force_n=max_force_n, max_duration_s=max_duration_s))

    def rotate_about_axis(self, axis, pivot_m, angle_rad, max_speed_m_s=.02,
                          max_angular_speed_rad_s=.2, max_force_n=18., max_duration_s=10.,target_detection_id=None,
                          motion_intent="free_motion"):
        self._validate_motion_intent(motion_intent, target_detection_id)
        if target_detection_id is not None:
            from .observed_contact import observed_contact
            return self._contact_result(observed_contact(self,target_detection_id,axis=axis,pivot_m=pivot_m,
                angle_rad=angle_rad,max_speed_m_s=max_speed_m_s,max_angular_speed_rad_s=max_angular_speed_rad_s,
                max_force_n=max_force_n,max_duration_s=max_duration_s), target_detection_id)
        return self._contact_result(self.robot.rotate_about_axis(axis, pivot_m, angle_rad,
            max_speed_m_s=max_speed_m_s, max_angular_speed_rad_s=max_angular_speed_rad_s,
            max_force_n=max_force_n, max_duration_s=max_duration_s))

    def retreat_last_motion(self, max_distance_m=.05, max_duration_s=10.):
        return self._contact_result(self.robot.retreat_last_motion(max_distance_m=max_distance_m,
                                                                 max_duration_s=max_duration_s))


def contact_motion_capabilities():
    vector = {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}
    speed = {'type': 'number', 'exclusiveMinimum': 0., 'maximum': .1, 'default': .02}
    force = {'type': 'number', 'exclusiveMinimum': 0., 'maximum': 18., 'default': 18.}
    duration = {'type': 'number', 'exclusiveMinimum': 0., 'maximum': 30., 'default': 10.}
    common = {'max_speed_m_s': speed, 'max_force_n': force, 'max_duration_s': duration}
    target = {'target_detection_id': {'type':['string','null'], 'default':None},
              'motion_intent': {'type':'string', 'enum':['free_motion','target_contact'], 'default':'free_motion'}}
    result = {'type': 'object', 'required': ['reached', 'stopped_reason', 'control_steps', 'contact_identity'],
        'properties': {'reached': {'type': 'boolean'}, 'stopped_reason': {'type': 'string'},
            'control_steps': {'type': 'integer', 'minimum': 0}, 'contact_identity': {'const': 'unknown'},
            'requested_displacement_m': vector, 'requested_rotation_vector_rad': vector,
            'actual_rotation_vector_rad': vector,
            'position_tolerance_m': {'type': 'number', 'exclusiveMinimum': 0},
            'orientation_tolerance_rad': {'type': 'number', 'exclusiveMinimum': 0},
            'orientation_reference': {'type': 'string', 'enum': ['commanded_change', 'recorded_pose']},
            'actual_displacement_m': vector, 'position_residual_m': {'type': 'number', 'minimum': 0},
            'orientation_residual_rad': {'type': 'number', 'minimum': 0},
            'force_available': {'type': 'boolean'}, 'observed_force_n': {'type': ['number', 'null']},
            'max_observed_force_n': {'type': 'number'},
            'force_guard': {'type': 'object'}, 'operation_force_guard': {'type': 'object'},
            'trajectory': {'type': 'array', 'items': {'type': 'object'}, 'x-decision-feedback': 'on_demand'},
            'observed_contact_segments': {'type':'array','items':{'type':'object'},'x-decision-feedback':'on_demand'},
            'target_progress': {'type':'object'},
            'motion_intent': {'type':'string'}, 'public_effect': {'type':'object'},
            'robot_goal_reached': {'type':'boolean'},
            'trajectory_handle': {'type': 'string', 'pattern': '^artifact://v1/'}},
        'additionalProperties': True}
    scope = (' Uses measured robot pose and wrist force; requires a functioning force sensor. '
        'The gripper command is preserved. Stop on the force threshold, a 5 N increase, tracking error, '
        'stall or duration limit. These are sampled stops, not hard bounds on inter-sample force peaks. '
        'No collision-free path, target contact identity, attachment or task success is asserted. '
        'Endpoint residual must fit the absolute tolerance (2 mm / 0.01 rad for contact primitives) '
        'and 10% of each nonzero requested pose change; recorded retreat uses the absolute orientation '
        'tolerance because historical orientation drift is not an explicit relative rotation command. '
        'This is a stopping criterion, not guaranteed '
        'hardware precision: unachievable fine commands return a stop instead of being enlarged. '
        'Bounded paths also require two real low-motion samples at the endpoint within the same duration budget. '
        'Inspect force_guard for the sampled reference, enabled limits and triggered comparisons. '
        'Inspect requested and actual displacement/rotation, effective tolerances, force, stop reason '
        'and fresh images before the next action. '
        'Full public control records are behind trajectory_handle.')
    def spec(name, description, properties, required, example):
        if name in {'move_linear','rotate_about_axis'}:
            properties={**properties,**target}
            description += (' Supply target_detection_id selected from the current snapshot for '
                'segmented contact with public RGB-D target-progress feedback. Each segment is at most '
                '1 cm of nominal travel and 0.1 rad rotation; the original total time and +5 N force '
                'increase budgets are preserved. Unknown, absent or opposite progress stops the probe '
                'and returns its evidence. Full segments are recorded, with only the last eligible '
                'for retreat. robot_goal_reached/reached describe the full robot waypoint; they can be true '
                'while target_progress remains unresolved and the operation returns a perception failure. '
                'Declare motion_intent=target_contact when intending to move a visible object or mechanism; '
                'this requires target_detection_id. Use free_motion for approach, transport or retreat '
                'that only specifies the hand waypoint. Without a target, this remains robot-pose control only.')
        return CapabilitySpec(name=name, description=description + scope,
            arguments_schema={'type': 'object', 'properties': properties, 'required': required, 'additionalProperties': False,
                **({'allOf': [{'if': {'properties': {'motion_intent': {'const': 'target_contact'}},
                                      'required': ['motion_intent']},
                               'then': {'required': ['target_detection_id'],
                                        'properties': {'target_detection_id': {'type': 'string', 'minLength': 1}}}}]}
                   if name in {'move_linear', 'rotate_about_axis'} else {})},
            result_schema=result, argument_order=tuple(properties), return_type='ContactMotionEvidence',
            units={'distance_m': 'meter', 'pivot_m': 'meter', 'angle_rad': 'radian',
                   'max_speed_m_s': 'meter/second command rate', 'max_force_n': 'newton stop threshold'},
            coordinate_frames={'direction': 'robot_base', 'axis': 'robot_base',
                               'pivot_m': 'robot_base; rotation of the complete end-effector pose'},
            timeout_s=120., side_effect=SideEffect.ROBOT_MOTION, control_semantic=ControlSemantic.MOTION,
            examples=(example,), real_robot_equivalent='bounded Cartesian impedance setpoints and measured wrist force')
    return (
        CapabilitySpec(name='read_motion_trajectory',
            description='Read a bounded page of an issued public robot-motion trajectory. '
                'Records contain measured robot pose/joints/force and controller diagnostics, never hidden task state. '
                'This works for both planned motion and guarded contact-motion handles.',
            arguments_schema={'type': 'object', 'properties': {
                'trajectory_handle': {'type': 'string', 'pattern': '^artifact://v1/'},
                'offset': {'type': 'integer', 'minimum': 0, 'default': 0},
                'limit': {'type': 'integer', 'minimum': 1, 'maximum': 100, 'default': 20}},
                'required': ['trajectory_handle'], 'additionalProperties': False},
            result_schema={'type': 'object', 'properties': {
                'samples': {'type': 'array', 'maxItems': 100, 'items': {'type': 'object'}},
                'next_offset': {'type': ['integer', 'null']}, 'total_samples': {'type': 'integer'}},
                'required': ['samples', 'next_offset', 'total_samples'], 'additionalProperties': True},
            argument_order=('trajectory_handle', 'offset', 'limit'), return_type='MotionTrajectoryPage',
            units={}, coordinate_frames={}, timeout_s=10., side_effect=SideEffect.READ_ONLY,
            examples=('read_motion_trajectory(trajectory_handle, offset=0, limit=20)',),
            real_robot_equivalent='read retained public controller telemetry'),
        spec('move_linear', 'Translate the end effector along an explicit nonzero direction, preserving orientation. '
             'Choose direction and distance from public observations for a bounded push, pull, approach or retreat. '
             'This tool does not infer the mechanism or select a task solution.',
             {'direction': vector, 'distance_m': {'type': 'number', 'exclusiveMinimum': 0., 'maximum': .25}, **common},
             ['direction', 'distance_m'], 'move_linear([0, 0, 1], 0.03, motion_intent="free_motion"); move_linear(direction, 0.02, target_detection_id=selected, motion_intent="target_contact")'),
        spec('rotate_about_axis', 'Rotate the complete end-effector pose by a signed angle about an explicit base-frame axis and pivot. '
             'The pivot must be within 0.5 m of the end effector. Use observed/calibrated geometry; no hidden joint axis is available. '
             'This moves the hand around the specified axis and does not claim the contacted mechanism follows it.',
             {'axis': vector, 'pivot_m': vector, 'angle_rad': {'type': 'number', 'minimum': -3.141592653589793, 'maximum': 3.141592653589793},
              'max_speed_m_s': speed,
              'max_angular_speed_rad_s': {'type': 'number', 'exclusiveMinimum': 0., 'maximum': 1., 'default': .2},
              'max_force_n': force, 'max_duration_s': duration},
             ['axis', 'pivot_m', 'angle_rad'], 'rotate_about_axis(axis, measured_pivot, 0.2, target_detection_id=selected, motion_intent="target_contact")'),
        spec('retreat_last_motion', 'Reverse a bounded portion of the most recent measured guarded-motion path. '
             'Any intervening physical action/time advance or endpoint change invalidates this record. '
             'An existing contact load is permitted only on this recorded reverse path, with a stop if force rises by 5 N. '
             'Follow the recorded orientation targets within 0.01 rad; retain raw rotation residuals in the receipt. '
             'The returned route is historical evidence, not a new collision-free plan. A retreat cannot be reversed into the obstacle.',
             {'max_distance_m': {'type': 'number', 'exclusiveMinimum': 0., 'maximum': .25, 'default': .05},
              'max_duration_s': duration}, [], 'retreat_last_motion(max_distance_m=0.05)'),
    )
