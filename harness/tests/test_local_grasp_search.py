import numpy as np
from copy import deepcopy
from types import SimpleNamespace
import pytest

from captain_harness.adapters.capx.local_grasp_search import local_translations, expand_grasp_refinement
from captain_harness.ports.capability import CapabilityResult


def test_local_search_offsets_follow_robot_axes_and_contact_resolution():
    rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    offsets = list(local_translations(rotation, .003))
    assert len(offsets) == 6
    np.testing.assert_allclose(offsets[0][1], [0, -.003, 0])
    np.testing.assert_allclose(offsets[1][1], [0, .003, 0])
    np.testing.assert_allclose([np.linalg.norm(delta) for _, delta in offsets], .003)


def test_local_fallback_does_not_perturb_a_new_coarse_grasp_orientation():
    result = CapabilityResult.ok({'selected_index': 3, 'candidate': {'grasp_handle': 'coarse', 'width_feasible':True},
        'after':{'approach':{'geometry_hits':{}},'reachability':{'reachable':True},
                 'pregrasp_collision':{'checked':True,'collision':False},
                 'contact':{'visible_point_count_in_pad_sweep':20}}})
    expanded = expand_grasp_refinement(SimpleNamespace(), {'grasp_handle': 'native'}, result)
    assert expanded.value['candidate'] == result.value['candidate']
    assert expanded.value['local_search']['attempts'] == []


@pytest.mark.parametrize('reachable,collision,improved', [(True, False, True), (False, False, False), (True, True, False)])
@pytest.mark.parametrize('quaternion', [[1.,0.,0.,0.], [0., .8847455765838064, .4660743124388942, 0.]])
def test_local_search_keeps_collision_ik_and_native_payload_boundaries(reachable, collision, improved, quaternion):
    original = {'position_m': [0.,0.,0.], 'quaternion_wxyz': quaternion,
        'approach_direction_world': [0.,0.,1.], 'backend_candidate': {'native': True},
        'grasp_handle': 'old', 'model_score': .9}
    before = {'contact': {'visible_point_count_in_pad_sweep': 5, 'contact_position_tolerance_m': .003},
        'approach': {'geometry_hits': {}}, 'reachability': {'reachable': True},
        'pregrasp_collision': {'checked': True, 'collision': False}}
    base = {**original, **before, 'kind': 'original', 'approach_distance_m': .06}
    refinement = CapabilityResult.ok({'candidate': {**original, 'width_feasible':True}, 'before': before,
        'after': before, 'alternatives': [base]})
    unchanged = deepcopy(refinement.value)
    inspected = []
    issued = []
    def evaluate(candidate, scene):
        inspected.append(deepcopy(candidate))
        return CapabilityResult.ok({'approach': {'geometry_hits': {'hand': 1} if collision else {}},
            'contact': {'visible_point_count_in_pad_sweep': 10}})
    def issue(candidate):
        issued.append(candidate)
        return {**candidate, 'grasp_handle': 'new'}
    backend = SimpleNamespace(_grasp_scene_handle=lambda *a: 'scene',
        _candidate_scene_target=lambda *a: (SimpleNamespace(source_observation_handle='snapshot', joints=np.zeros(7)), np.zeros((3,3))),
        _motion_model=lambda: None, _grasp_planning_collision_checker=lambda *a: (lambda j: {'collision': False}, {}),
        evaluate_grasp=evaluate, _predicted_grasp_width=lambda *a: .02,
        gripper_profile=SimpleNamespace(max_width_m=.08),
        _grasp_endpoint_reachability=lambda *a, **kw: {'reachable': reachable, 'joints': [0.]*7},
        _issue_grasp_candidate=issue)
    result = expand_grasp_refinement(backend, original, refinement)
    assert refinement.value == unchanged
    assert len(inspected) == 12
    assert result.value['local_search']['improved'] is improved
    if improved:
        assert len(issued) == 1
        assert result.value['candidate']['backend_candidate'] is None
        assert result.value['candidate']['model_score'] is None
        assert result.value['candidate']['grasp_handle'] == 'new'
        assert 0. <= result.value['candidate']['top_down_alignment'] <= 1.
    else:
        assert not issued
        assert result.value['candidate'] == unchanged['candidate']


@pytest.mark.parametrize('experimental', [False, True])
@pytest.mark.parametrize('blocked', ['none', 'collision', 'ik'])
def test_intermediate_centering_can_recover_a_pose_missed_by_small_axial_offsets(blocked, experimental):
    original={'position_m':[0.,0.,0.], 'quaternion_wxyz':[1.,0.,0.,0.],
        'approach_direction_world':[0.,0.,1.], 'backend_candidate':{'native':True},
        'grasp_handle':'original', 'model_score':.9, 'width_feasible':True}
    evidence={'contact':{'visible_point_count_in_pad_sweep':5, 'visible_point_count_in_pad_interior':2,
        'contact_position_tolerance_m':.003}, 'approach':{'geometry_hits':{}},
        'reachability':{'reachable':True}, 'pregrasp_collision':{'checked':True,'collision':False}}
    base={**original, **deepcopy(evidence), 'kind':'original','approach_distance_m':.06}
    center={**deepcopy(base), 'kind':'preserved_orientation','position_m':[0.,0.,.06],
        'approach':{'geometry_hits':{'hand':1}}}
    initial=CapabilityResult.ok({'candidate':original,'before':deepcopy(evidence),
        'after':deepcopy(evidence),'alternatives':[base,center], 'selected_index':0})
    unchanged=deepcopy(initial.value);seen=[];issued=[]
    def evaluate(candidate, scene):
        position=np.asarray(candidate['position_m']);seen.append(position.copy())
        promising=np.allclose(position, [0.,0.,.009], rtol=0, atol=1e-12)
        return CapabilityResult.ok({'approach':{'geometry_hits':{'hand':1} if promising and blocked=='collision' else {}},
            'contact':{'visible_point_count_in_pad_sweep':10 if promising else 5,
                       'visible_point_count_in_pad_interior':8 if promising else 2}})
    def issue(candidate):
        issued.append(deepcopy(candidate));return {**candidate,'grasp_handle':'new'}
    backend=SimpleNamespace(_grasp_scene_handle=lambda *a:'scene',
        _candidate_scene_target=lambda *a:(SimpleNamespace(source_observation_handle='snapshot',joints=np.zeros(7)),np.zeros((3,3))),
        _motion_model=lambda:None,_grasp_planning_collision_checker=lambda *a:(lambda q:{'collision':False},{}),
        evaluate_grasp=evaluate,_predicted_grasp_width=lambda *a:.02,
        gripper_profile=SimpleNamespace(max_width_m=.08),
        _grasp_endpoint_reachability=lambda *a,**kw:{'reachable':blocked!='ik','joints':[0.]*7},
        _issue_grasp_candidate=issue)
    result=(expand_grasp_refinement(backend,original,initial,experimental_centering=True) if experimental
            else expand_grasp_refinement(backend,original,initial))
    assert initial.value==unchanged
    assert any(np.allclose(p,[0,0,.009],rtol=0,atol=1e-12) for p in seen) is experimental
    assert len(seen)<=32
    assert len({tuple(np.round(p,12)) for p in seen})==len(seen)
    assert result.value['local_search']['improved'] is (blocked=='none' and experimental)
    if blocked=='none' and experimental:
        np.testing.assert_allclose(result.value['candidate']['position_m'],[0,0,.009],atol=1e-12)
        assert len(issued)==1 and issued[0]['backend_candidate'] is None and issued[0]['model_score'] is None
    else:assert not issued


def test_expired_local_refinement_does_not_start_another_geometry_query():
    from captain_harness.ports.deadline import deadline_scope, DeadlineExceeded
    import time
    with deadline_scope('episode',expires_s=time.monotonic()-1):
        with pytest.raises(DeadlineExceeded):
            expand_grasp_refinement(SimpleNamespace(),{},CapabilityResult.ok({}))
