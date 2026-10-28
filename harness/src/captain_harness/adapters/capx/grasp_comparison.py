"""Bounded, diverse candidate comparison using measured public geometry."""
import math
from copy import deepcopy
import numpy as np


def _distance(a, b):
    # Pose diversity selects what to MEASURE, never predicts grasp success.
    position = np.linalg.norm(np.asarray(a.get('position_m', [0,0,0]))-b.get('position_m', [0,0,0]))
    direction = np.linalg.norm(np.asarray(a.get('approach_direction_world', [0,0,0]))-b.get('approach_direction_world', [0,0,0]))
    qa, qb = (np.asarray(c.get('quaternion_wxyz', [1,0,0,0]), float) for c in (a,b))
    orientation = min(np.linalg.norm(qa-qb), np.linalg.norm(qa+qb))
    return float(position/.05 + direction + orientation)


def compare_grasps(backend, candidates, limit=4):
    pending = list({c['grasp_handle']: c for c in candidates if c.get('grasp_handle')}.values())
    chosen=[]
    while pending and len(chosen)<limit:
        index = 0 if not chosen else max(range(len(pending)),
            key=lambda i:min(_distance(pending[i], c) for c in chosen))
        chosen.append(pending.pop(index))
    records=[]
    for candidate in chosen:
        record={'grasp_handle':candidate['grasp_handle'], 'native_model_score':candidate.get('model_score'),
            'requested_style_match':candidate.get('request_style_match','unassessed'),
            'approach_direction_world':candidate.get('approach_direction_world'),
            'width_feasible':candidate.get('width_feasible'), 'endpoint_ik_feasible':candidate.get('ik_feasible'),
            'motion_path':'not_checked', 'stable_grasp_verified':False,
            'contact_support':'unknown', 'observed_approach_collision':None}
        try:
            measured=backend.evaluate_grasp(candidate)
            if measured.success and isinstance(measured.value,dict):
                contact=measured.value.get('contact',{})
                record['surface_quality'] = deepcopy(contact.get('contact_region_support', {}).get(
                    'surface_quality', {'opposing_surface_status': 'unverified', 'force_closure_verified': False}))
                record['pose_sensitivity'] = deepcopy(contact.get('contact_region_support', {}).get('pose_sensitivity'))
                inside=contact.get('visible_point_count_in_pad_interior')
                sweep=contact.get('visible_point_count_in_pad_sweep')
                valid=lambda x:isinstance(x,(int,float)) and math.isfinite(x) and x>=0
                record.update(interior_points=inside, tolerance_sweep_points=sweep,
                    contact_support=('interior_observed' if valid(inside) and inside>0 else
                        'tolerance_only' if valid(inside) and inside==0 and valid(sweep) and sweep>0 else
                        'no_visible_support' if valid(sweep) and sweep==0 else 'unknown'),
                    observed_approach_collision=measured.value.get('approach',{}).get('observed_collision'),
                    source_observation_handle=measured.value.get('source_observation_handle'))
            else:
                record['measurement_error']=measured.message
        except (ValueError, RuntimeError) as exc:
            record['measurement_error']=f'{type(exc).__name__}: {exc}'
        record['inspect_before_grazing_candidate']=bool(
            record['contact_support']=='interior_observed'
            and record['observed_approach_collision'] is False
            and record['width_feasible'] is True and record['endpoint_ik_feasible'] is True)
        records.append(record)
    return {'candidates':records,
        'preferred_for_inspection':[r['grasp_handle'] for r in records if r['inspect_before_grazing_candidate']],
        'unmeasured_candidate_count':len(pending), 'max_measured_candidates':limit,
        'guidance':'Compare interior support and observed obstructions before native scores. '
            'Tolerance-only overlap can graze an edge. Visible interior points do not prove bilateral '
            'force closure or retention. An explicit style change remains your decision. '
            'Surface quality reports observed depth, closing-axis offset and opposing normal patches; '
            'occluded surfaces remain unknown. Choose contact, probe and recovery from current evidence.'}
