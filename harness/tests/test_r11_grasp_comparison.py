from copy import deepcopy
from types import SimpleNamespace

from captain_harness.ports.capability import CapabilityResult


def test_comparison_exposes_weak_first_grasp_and_preserves_all_scores():
    from captain_harness.adapters.capx.grasp_comparison import compare_grasps
    candidates = [{'grasp_handle': str(i), 'position_m': [i*.01,0,0],
        'approach_direction_world': direction, 'quaternion_wxyz':[1,0,0,0],
        'model_score': score, 'width_feasible': True, 'ik_feasible': True}
        for i,direction,score in [(0,[0,0,-1],.99),(1,[1,0,0],.51),(2,[0,1,0],.45)]]
    def evaluate(candidate):
        return CapabilityResult.ok({'contact': {'visible_point_count_in_pad_sweep': 20,
            'visible_point_count_in_pad_interior': 0 if candidate['grasp_handle']=='0' else 15},
            'approach': {'observed_collision':False}})
    original = deepcopy(candidates)
    value = compare_grasps(SimpleNamespace(evaluate_grasp=evaluate), candidates)
    assert candidates == original
    assert value['candidates'][0]['contact_support'] == 'tolerance_only'
    assert value['preferred_for_inspection'][0] != '0'
    assert {r['grasp_handle']:r['native_model_score'] for r in value['candidates']} == {'0':.99,'1':.51,'2':.45}
    assert all(r['stable_grasp_verified'] is False for r in value['candidates'])


def test_missing_geometry_is_unknown_and_comparison_has_fixed_work_bound():
    from captain_harness.adapters.capx.grasp_comparison import compare_grasps
    calls=[]
    def evaluate(c):
        calls.append(c['grasp_handle'])
        return CapabilityResult.ok({})
    candidates=[{'grasp_handle':str(i),'position_m':[i*.01,0,0]} for i in range(20)]
    value=compare_grasps(SimpleNamespace(evaluate_grasp=evaluate),candidates)
    assert len(calls)<=4
    assert value['preferred_for_inspection']==[]
    assert all(r['contact_support']=='unknown' for r in value['candidates'])


def test_surface_quality_survives_candidate_comparison_without_grasp_claim():
    from captain_harness.adapters.capx.grasp_comparison import compare_grasps
    quality={'opposing_surface_status':'unverified','closing_axis_offset_m':.004,
             'force_closure_verified':False}
    backend=SimpleNamespace(evaluate_grasp=lambda c: CapabilityResult.ok({
        'contact':{'contact_region_support':{'surface_quality':quality}},'approach':{}}))
    result=compare_grasps(backend,[{'grasp_handle':'A'}])
    record=result['candidates'][0]
    assert record['surface_quality']==quality
    assert not record['stable_grasp_verified']
    assert result['preferred_for_inspection']==[]
