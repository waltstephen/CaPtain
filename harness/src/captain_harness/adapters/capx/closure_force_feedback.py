"""Typed publication of the adapter's measured closure-force evidence."""
from copy import deepcopy
from dataclasses import replace


def with_closure_force_contract(spec):
    changed = False
    pose = {'type': ['object', 'null'], 'properties': {
        'eef_position_m': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3},
        'eef_quaternion_wxyz': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 4, 'maxItems': 4},
        'gripper_open_fraction': {'type': 'number'}},
        'required': ['eef_position_m', 'eef_quaternion_wxyz', 'gripper_open_fraction'],
        'additionalProperties': False}
    motion = {'type': 'object', 'x-attempt-feedback': 'priority', 'properties': {
        'status': {'enum': ['unavailable', 'partial', 'measured']},
        'valid_samples': {'type': 'integer', 'minimum': 0},
        'total_samples': {'type': 'integer', 'minimum': 0},
        'initial_pose': pose, 'final_pose': pose,
        **{k: {'type': ['number', 'null'], 'minimum': 0} for k in (
            'final_translation_m', 'max_sampled_translation_m',
            'final_rotation_rad', 'max_sampled_rotation_rad')},
        'physical_retention_verified': {'const': False}, 'scope': {'type': 'string'}},
        'required': ['status', 'valid_samples', 'total_samples', 'initial_pose', 'final_pose',
            'final_translation_m', 'max_sampled_translation_m', 'final_rotation_rad',
            'max_sampled_rotation_rad', 'physical_retention_verified', 'scope'],
        'additionalProperties': False}
    def expand(schema):
        nonlocal changed
        if isinstance(schema, list):
            return [expand(item) for item in schema]
        if not isinstance(schema, dict):
            return schema
        result = {key: expand(value) for key, value in schema.items()}
        properties = result.get('properties', {})
        if {'closure_executed', 'gripper_open_fraction'} <= properties.keys():
            changed = True
            if 'contacted' in properties:
                properties['force_measurement_phase'] = {'const': 'after_descend_and_close_return'}
            properties['requested_position_tolerance_m'] = {'type':'number', 'minimum':.0005, 'maximum':.003}
            properties['contact_precision'] = {'type':'object', 'x-attempt-feedback':'priority', 'properties':{
                'requested_position_tolerance_m':{'type':'number','minimum':.0005,'maximum':.003},
                'preclosure_endpoint_within_tolerance':{'type':'boolean'},
                'postclosure_endpoint_within_tolerance':{'type':'boolean'},
                'scope':{'type':'string'}}, 'additionalProperties':False}
            properties['closure_force_monitor'] = {'type': 'object', 'x-attempt-feedback': 'priority', 'properties': {
                'closure_motion': motion,
                'reason': {'type': ['string', 'null']}, 'initial_force_n': {'type': ['number', 'null']},
                'force_limit_n': {'type': 'number', 'exclusiveMinimum': 0, 'maximum': 18},
                'force_increase_limit_n': {'const': 5.}, 'control_steps': {'type': 'integer', 'minimum': 0},
                'scope': {'type': 'string'}, 'diagnostic': {'type': 'string'},
                'post_stop_gripper_command': {'type': 'object', 'properties': {
                    'native_action': {'const': 0.}, 'effect': {'enum': ['hold_current_actuator_target', 'hold_measured_joint_target',
                        'physics_advance_inhibited_until_explicit_open']},
                    'target_replacement': {'type': 'object', 'properties': {
                        'status': {'enum': ['replaced', 'unavailable']}, 'error': {'type': 'string'},
                        'joint_names': {'type': 'array', 'items': {'type': 'string'}},
                        'actuator_names': {'type': 'array', 'items': {'type': 'string'}},
                        **{k: {'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2}
                           for k in ('measured_joint_positions_m', 'previous_target_m', 'target_m',
                                     'previous_normalized_target', 'normalized_target')},
                        'basis': {'type': 'string'}, 'physics_steps': {'const': 0}},
                        'required': ['status'], 'additionalProperties': False},
                    'reason': {'const': 'monitored_closure_interrupted'}, 'limitation': {'type': 'string'}},
                    'required': ['native_action', 'effect', 'reason', 'limitation'], 'additionalProperties': False},
                'samples': {'type': 'array', 'x-decision-feedback': 'on_demand', 'items': {
                    'type': 'object', 'properties': {'step': {'type': 'integer', 'minimum': 0},
                        'force_n': {'type': ['number', 'null']}, 'pose': pose}, 'required': ['step', 'force_n'],
                    'additionalProperties': False}}},
                'required': ['reason', 'initial_force_n', 'force_limit_n', 'force_increase_limit_n',
                             'control_steps', 'samples', 'scope'], 'additionalProperties': False}
        return result
    result_schema = expand(deepcopy(dict(spec.result_schema)))
    if not changed:
        return spec
    extra = {}
    if spec.name == 'descend_and_close':
        args = deepcopy(dict(spec.arguments_schema))
        args.setdefault('properties', {})['position_tolerance_m'] = {
            'type': ['number', 'null'], 'minimum': .0005, 'maximum': .003, 'default': None}
        extra = {'arguments_schema': args, 'argument_order': (*spec.argument_order, 'position_tolerance_m')}
    return replace(spec, result_schema=result_schema, **extra,
        description=spec.description+' Closing also samples wrist force at each control cycle and after '
        'the final cycle, with the same force and force-increase limits. closure_motion separately reports '
        'measured EEF drift during closure, sampled maxima and coverage; it does not certify target retention. '
        'observed_force_n is measured '
        'after the entire operation. descend_and_close accepts an optional position_tolerance_m (0.0005 to 0.003 m, '
        'default 0.003 m) for measured pre-closure endpoint convergence; tighter requests keep the same force and step limits '
        'and can fail without closing. Post-closure error and target effect remain separate. A partial closure stopped by force is an executed failure, '
        'not a verified grasp. After an executed closing-guard stop, later motions use neutral gripper '
        'input after replacing its accumulated target with measured finger joint positions through the calibrated '
        'position servo. Inspect target_replacement: if unavailable, physics advance is inhibited until an explicit '
        'opening request. Replacement does not guarantee stationary fingers or released load. '
        'Inspect current force and views before recovery. An explicit opening request remains available.')
