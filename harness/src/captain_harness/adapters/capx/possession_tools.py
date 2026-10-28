"""Adapter boundary for sampled possession and evidence expiry."""
from copy import deepcopy
from dataclasses import replace

from .artifacts import ArtifactHandleError


_PRESERVE_POSSESSION = frozenset({'goto_pose', 'move_linear', 'rotate_about_axis',
    'servo_to', 'hold_position', 'lift_probe', 'home', 'execute_motion'})


def with_possession_contract(spec):
    if spec.name not in _PRESERVE_POSSESSION:
        return spec
    schema = deepcopy(dict(spec.result_schema))
    schema['x-attachment-continuity-field'] = ['possession_monitor', 'object_detected']
    schema.setdefault('properties', {})['possession_monitor'] = {
        'type': 'object', 'properties': {
            'status': {'enum': ['maintained', 'lost', 'unavailable', 'not_present']},
            'object_detected': {'type': ['boolean', 'null']},
            'initial_sample': {'type': 'object'}, 'stop_sample': {'type': 'object'},
            'aperture_samples': {'type': 'array', 'items': {'type': 'object'}, 'x-decision-feedback': 'on_demand'},
            'aperture_change_fraction': {'type': 'number'},
            'sampled_duration_s': {'type': 'number', 'minimum': 0},
            'scope': {'type': 'string'}, 'samples': {'type': 'integer'}},
        'required': ['status', 'object_detected'], 'additionalProperties': True}
    return replace(spec, result_schema=schema, description=spec.description +
        ' When a grasp is pending/verified or the public register detects possession, sample that register during '
        'this operation. Loss or unavailable evidence stops motion and invalidates prior attachment '
        'and placement permissions. Maintained possession alone does not prove target identity or '
        'a stable grasp. For the LIBERO adapter this register infers occupancy from a close command and measured jaw opening, '
        'not target contact; slipping can precede the loss threshold. Inspect aperture_change_fraction and fresh images; continuing closure during '
        'a stationary hold can indicate slipping or settling and needs further observation before transport. '
        'Release and recorded retreat remain separate operations.')


def invoke_with_possession_evidence(backend, spec, arguments):
    pending_handle = backend._pending_attachment_handle or backend._active_attachment_handle
    handles = {backend._active_attachment_handle, backend._pending_attachment_handle} - {None}
    guard = getattr(backend.robot, 'with_possession_guard', None)
    if 'x-attachment-continuity-field' not in spec.result_schema:
        result = backend.invoke(spec.name, arguments)
        value = result.value if isinstance(result.value, dict) else {}
        if spec.name == 'open_gripper' and result.executed:
            backend._untracked_possession_expected = False
        elif spec.name in {'close_gripper', 'descend_and_close', 'approach_and_close'}:
            if result.executed and value.get('closure_executed') is not False:
                backend._untracked_possession_expected = value.get('object_detected') is not False
        elif spec.name == 'gripper_state' and result.success:
            if value.get('object_detected') is True:
                backend._untracked_possession_expected = True
        return result
    expected = bool(handles) or getattr(backend, '_untracked_possession_expected', False)
    if not expected:
        read = getattr(backend.robot, 'gripper_state', None)
        try:
            observed = read() if callable(read) else None
        except Exception:
            observed = None
        expected = bool(observed and observed.success and isinstance(observed.value, dict)
                        and observed.value.get('object_detected') is True)
    if not expected:
        return backend.invoke(spec.name, arguments)
    backend._untracked_possession_expected = True
    if not callable(guard):
        raise RuntimeError('declared possession monitor is unavailable on this robot backend')
    result = guard(lambda: backend.invoke(spec.name, arguments))
    value = dict(result.value or {})
    monitor = dict(value['possession_monitor'])
    status = monitor['status']
    if status in {'lost', 'not_present'}:
        backend._untracked_possession_expected = False
    monitor['object_detected'] = (True if status == 'maintained' else
                                  False if status in {'lost', 'not_present'} else None)
    if status != 'maintained':
        for handle in handles:
            try:
                tracked = backend.registry.get(handle, expected_kind='entity_track')
            except ArtifactHandleError:
                continue
            tracked.attachment_verified = False
            tracked.eef_to_object_transform = None
            tracked.last_completion_observation_handle = None
            tracked.last_completion_centroid = None
            tracked.last_completion_time_s = None
            tracked.last_completion_released = False
        backend._active_attachment_handle = None
        backend._pending_attachment_handle = pending_handle if status == 'unavailable' else None
        if status != 'unavailable':
            backend._latest_lift_evidence = None
        monitor['attachment_evidence_invalidated'] = True
    value['possession_monitor'] = monitor
    return replace(result, value=value)
