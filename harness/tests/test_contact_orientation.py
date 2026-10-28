"""Fixed contact-region proposals keep provenance and every measured gate."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from captain_harness.ports.capability import CapabilityResult


def fixture(monkeypatch, *, collision=False, reachable=True, width=.04, support=10):
    import captain_harness.adapters.capx.contact_orientation as module
    original_rotation=Rotation.from_euler('xyz',[.1,.2,.3]).as_matrix()
    changed=Rotation.from_euler('x',.3).as_matrix()@original_rotation
    # Same approach and closing axis with jaws exchanged isn't a new strategy.
    swapped=original_rotation@Rotation.from_euler('z',np.pi).as_matrix()
    monkeypatch.setattr(module,'support_plane',lambda *a:(np.array([0,0,1]),0,None))
    monkeypatch.setattr(module,'surface_orientations',lambda *a:[('same',original_rotation),
        ('swapped',swapped),('changed',changed)])
    candidate={'position_m':[.4,.2,.1], 'quaternion_wxyz':module.quaternion(
        np.block([[original_rotation,np.zeros((3,1))],[np.zeros((1,3)),np.ones((1,1))]])),
        'grasp_handle':'old','backend_candidate':{'native':'payload'},'model_score':.99,
        'feasibility_evidence':{'old':True},'tracking_handle':'track','detection_id':'detection'}
    scene=SimpleNamespace(points=np.ones((20,3)),joints=np.zeros(7),source_observation_handle='fresh')
    backend=SimpleNamespace(
        _candidate_scene_target=lambda *a:(scene,np.ones((20,3))),
        robot=SimpleNamespace(gripper_contact_envelope=lambda:{'pad_min_local_m':[-.01,-.02,.08],
            'pad_max_local_m':[.01,.02,.12]}),
        gripper_profile=SimpleNamespace(max_width_m=.08,fingertip_depth_m=.04),
        _contact_local_points=lambda pose,points:(pose[:3,3]+pose[:3,:3]@np.array([0,0,.012]),points),
        _motion_model=lambda:None,
        _grasp_planning_collision_checker=lambda *a:(lambda q:{'collision':collision},{}),
        evaluate_grasp=lambda *a:CapabilityResult.ok({'approach':{'geometry_hits':{}},
            'contact':{'visible_point_count_in_pad_sweep':support}}),
        _predicted_grasp_width=lambda *a:width,
        _grasp_endpoint_reachability=lambda *a,**kw:{'joints':np.zeros(7) if reachable else None,'reachable':reachable})
    return module,backend,candidate


def test_proposals_preserve_calibrated_contact_point_and_clear_native_provenance(monkeypatch):
    module,backend,candidate=fixture(monkeypatch)
    before=deepcopy(candidate)
    result=module.inspect_orientations(backend,candidate,'scene').value
    selected=result['selected_candidate']
    assert result['selected_kind']=='changed'
    assert selected['backend_candidate'] is None and selected['model_score'] is None
    assert 'grasp_handle' not in selected and 'feasibility_evidence' not in selected
    assert selected['source_observation_handle']=='fresh'
    assert selected['tracking_handle']=='track' and selected['detection_id']=='detection'
    assert all(r['contact_reference_error_m'] < 1e-12 for r in result['alternatives'])
    assert next(r for r in result['alternatives'] if r['kind']=='swapped')['distinct_contact_orientation'] is False
    assert candidate==before


@pytest.mark.parametrize('options',[{'collision':True},{'reachable':False},{'width':None},{'width':.12},{'support':0}])
def test_failed_measured_constraint_never_produces_a_selected_orientation(monkeypatch,options):
    module,backend,candidate=fixture(monkeypatch,**options)
    result=module.inspect_orientations(backend,candidate,'scene').value
    assert result['selected_candidate'] is None
    assert len(result['alternatives'])==2
    assert all(r['admissible'] is False for r in result['alternatives'])


def test_last_refinement_orientation_pass_issues_new_reference_and_keeps_input_checks(monkeypatch):
    import captain_harness.adapters.capx.contact_orientation as module
    base={'grasp_handle':'coarse','position_m':[.4,.2,.1],'quaternion_wxyz':[1,0,0,0]}
    chosen={**base,'position_m':[.4,.2,.11],'quaternion_wxyz':[.99,.1,0,0]}
    chosen.pop('grasp_handle')
    after={'contact':{'visible_point_count_in_pad_sweep':8},'approach':{'geometry_hits':{}},
           'reachability':{'reachable':True},'pregrasp_collision':{'collision':False},'collision_policy':{}}
    measurements={'selected_candidate':chosen,'selected_kind':'changed','reason':'candidate_found',
                  'alternatives':[{'kind':'changed','admissible':True,'candidate':chosen,**after}]}
    calls=[]
    def inspect(backend,candidate,scene):
        calls.append((deepcopy(candidate),scene)); return CapabilityResult.ok(deepcopy(measurements))
    monkeypatch.setattr(module,'inspect_orientations',inspect)
    backend=SimpleNamespace(_grasp_scene_handle=lambda *a:'current-scene',
                            _issue_grasp_candidate=lambda candidate:{**candidate,'grasp_handle':'issued'})
    previous=CapabilityResult.ok({'candidate':base,'after':{'coarse':True},'selected_index':2,
                                 'local_search':{'attempts':[]}})
    result=module.expand_contact_orientation(backend,previous)
    assert calls==[(base,'current-scene')]
    assert result.value['candidate']['grasp_handle']=='issued'
    assert result.value['after']['pregrasp_collision']=={'checked':True,'collision':False}
    assert result.value['contact_orientation']['input_candidate']==base
    assert result.value['contact_orientation']['input_checks']=={'coarse':True}
    assert result.value['local_search']==previous.value['local_search']
    assert previous.value['candidate']==base
