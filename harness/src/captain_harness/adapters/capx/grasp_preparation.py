"""Public gripper preparation for the existing planned-approach contract."""
from copy import deepcopy
from dataclasses import replace
import math

PLANNED_GRASP_MIN_OPEN_FRACTION = .95
_PREPARATION_TOOLS = frozenset({'gripper_state', 'grasp_pose_estimate', 'build_scene'})


def planned_grasp_preparation(open_fraction):
    known = (isinstance(open_fraction, (int, float)) and not isinstance(open_fraction, bool)
             and math.isfinite(open_fraction) and 0 <= open_fraction <= 1)
    prepared = open_fraction >= PLANNED_GRASP_MIN_OPEN_FRACTION if known else None
    return {'observed_open_fraction': float(open_fraction) if known else None,
            'minimum_open_fraction': PLANNED_GRASP_MIN_OPEN_FRACTION,
            'gripper_prepared': prepared,
            'next_step': ('open_gripper_then_reobserve' if prepared is False else
                          'obtain_public_gripper_state' if prepared is None else 'preserve_opening_and_use_fresh_geometry'),
            'motion_path': 'not_checked',
            'scope': 'Gripper preparation only. Opening changes robot state and invalidates old scene/target geometry; '
                     'obtain fresh observations after opening. Endpoint/path checks and target verification remain required.'}


def add_grasp_preparation(backend, name, result):
    if name not in _PREPARATION_TOOLS or not result.success or not isinstance(result.value, dict):
        return result
    if name == 'gripper_state':
        fraction = result.value.get('open_fraction')
    else:
        try:
            state = backend.robot.get_robot_state()
            fraction = state.value.gripper_open_fraction if state.success and state.value is not None else None
        except Exception:
            fraction = None
    return replace(result, value={**result.value,
        'planned_grasp_preparation': planned_grasp_preparation(fraction)})


def with_grasp_preparation_contract(spec):
    if spec.name not in _PREPARATION_TOOLS:
        return spec
    schema = deepcopy(dict(spec.result_schema))
    schema.setdefault('properties', {})['planned_grasp_preparation'] = {
        'type':'object', 'properties': {
            'observed_open_fraction': {'type':['number','null'], 'minimum':0, 'maximum':1},
            'minimum_open_fraction': {'const':PLANNED_GRASP_MIN_OPEN_FRACTION},
            'gripper_prepared': {'type':['boolean','null']},
            'next_step': {'enum':['open_gripper_then_reobserve','obtain_public_gripper_state',
                                 'preserve_opening_and_use_fresh_geometry']},
            'motion_path': {'const':'not_checked'}, 'scope':{'type':'string'}},
        'required':['observed_open_fraction','minimum_open_fraction','gripper_prepared','next_step','motion_path','scope'],
        'additionalProperties':False}
    return replace(spec,result_schema=schema,description=spec.description+
        ' Read planned_grasp_preparation before choosing planned alignment: a partly open hand '
        'can still need open_gripper followed by a fresh observation. This preparation signal '
        'does not certify the chosen grasp or motion path.')
