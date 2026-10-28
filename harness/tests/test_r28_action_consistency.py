from types import SimpleNamespace
import numpy as np
import pytest
from captain_harness.ports.capability import CapabilityResult
from test_tracking_clearance_contract import tool_with_tracking
from test_partial_attachment_feedback import setup_case

def test_candidate_pregrasp_and_planner_reject_the_same_reserved_endpoint():
    tool,model,queries=tool_with_tracking()
    scene=tool._scene("scene",fresh=True)
    check,policy=tool._grasp_planning_collision_checker(model,scene,np.empty((0,3)))
    assert check(np.array([.2]))["collision"]
    assert not check(np.array([.05]))["collision"]
    assert policy["planning_tracking_reserve"]["geometry_padding_m"]["hand"]==.006

@pytest.mark.parametrize("partial,cross_view",[(False,True),(True,False)])
def test_large_apparent_motion_from_incomparable_surfaces_cannot_verify(tmp_path,partial,cross_view):
    f,track,handle,mark=setup_case(tmp_path,partial=partial,cross_view=cross_view)
    current=f.backend.registry.get(mark,expected_kind="marked_point")
    delta=np.array([0.,0.,.09])
    current.centroid+=delta;current.bounds_min+=delta;current.bounds_max+=delta
    result=f.backend.assert_entity_attached(handle,current_point_handle=mark)
    assert result.success and not result.value["verified"]
    assert result.value["reason"]=="partial_geometry_motion_unresolved"
    assert track.eef_to_object_transform is None and not track.attachment_verified

def bound_fixture(tmp_path):
    from test_v18_grasp_references import issued
    from captain_harness.adapters.capx.environment import CapXLiberoEnvironmentAdapter
    f,broker,candidate,_=issued(tmp_path)
    adapter=CapXLiberoEnvironmentAdapter.__new__(CapXLiberoEnvironmentAdapter)
    adapter._tools=f.backend;adapter._motion_tools=True
    adapter.capabilities=lambda:broker.manifest
    adapter._require_robot=lambda:SimpleNamespace(time_s=0)
    return f,adapter,candidate

def test_explicit_primitive_closure_and_later_loss_share_exact_attempt(tmp_path):
    f,adapter,candidate=bound_fixture(tmp_path)
    r=adapter.execute("close_gripper",{"tracking_handle":candidate["tracking_handle"]})
    assert r.success,r.message
    assert r.attempt_identity["attempt_reference"]
    assert r.value["associated_tracking_handle"]==candidate["tracking_handle"]
    assert r.attempt_identity["reference"]==candidate["detection_id"]
    assert f.backend._pending_attachment_handle==candidate["tracking_handle"]
    lost=adapter._attempt_evidence_binding.annotate(f.backend,SimpleNamespace(name="move_linear"),{},
        CapabilityResult.ok({"possession_monitor":{"status":"lost"}}),r.attempt_identity)
    assert lost.attempt_evidence["attempt_reference"]==r.attempt_identity["attempt_reference"]
    assert lost.attempt_evidence["status"]=="contradicted"
    assert adapter._attempt_evidence_binding.annotate(f.backend,SimpleNamespace(name="assert_entity_attached"),
        {"tracking_handle":candidate["tracking_handle"]},CapabilityResult.ok({"verified":True}),None).attempt_evidence is None

def test_unbound_primitive_closure_has_receipt_but_never_borrows_target(tmp_path):
    f,adapter,candidate=bound_fixture(tmp_path)
    r=adapter.execute("close_gripper",{})
    assert r.success,r.message
    assert r.attempt_identity["attempt_reference"]
    assert r.attempt_identity["basis"]=="unbound_closure"
    assert r.value["associated_tracking_handle"] is None
    assert f.backend._pending_attachment_handle is None
    binding=adapter._attempt_evidence_binding
    query=binding.annotate(f.backend,SimpleNamespace(name="assert_entity_attached"),
        {"tracking_handle":candidate["tracking_handle"]},CapabilityResult.ok({"verified":True}),None)
    assert query.attempt_evidence is None
    lost=binding.annotate(f.backend,SimpleNamespace(name="move_linear"),{},
        CapabilityResult.ok({"possession_monitor":{"status":"lost"}}),None)
    assert lost.attempt_evidence["attempt_reference"]==r.attempt_identity["attempt_reference"]
    assert lost.attempt_evidence["status"]=="contradicted"
    assert lost.attempt_evidence["evidence"]["target_identity_verified"] is False

def test_invalid_explicit_closure_target_does_not_move(tmp_path):
    f,adapter,candidate=bound_fixture(tmp_path)
    before=list(f.robot.calls)
    r=adapter.execute("close_gripper",{"tracking_handle":"artifact://invalid"})
    assert not r.success and not r.executed
    assert f.robot.calls==before

def test_alignment_phase_receipts_use_absolute_phase_tolerance(tmp_path):
    from test_grasp_final_measurement import robot_fixture
    _,robot=robot_fixture(tmp_path)
    robot._guarded_cartesian_motion=True
    state=robot.get_robot_state().value
    result=robot.align_pregrasp(
        np.asarray(state.end_effector_position_m)+[.02,0,-.06],
        state.end_effector_quaternion_wxyz,[0,0,-1],approach_distance_m=.06)
    phases=result.value["phase_executions"]
    assert phases and any(isinstance(p['value'],dict) and 'precision_basis' in p['value'] for p in phases)
    for p in phases:
        v=p["value"]
        if isinstance(v,dict) and "precision_basis" in v:
            assert v["precision_reference"]=="pregrasp_phase"
            assert v["position_tolerance_m"]==v["configured_position_tolerance_m"]
