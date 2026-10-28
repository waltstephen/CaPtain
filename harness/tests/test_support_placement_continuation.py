"""Harness-issued geometry grasps must have a truthful support-placement continuation."""
from dataclasses import replace
import numpy as np
import pytest
from test_tool_backend import _fixture, _invoke, _v9_selected_detection
from captain_harness.adapters.capx.manifest import ToolApiLevel
from captain_harness.adapters.libero.types import ObservationData, RobotState


def setup_attachment(tmp_path, backend_name):
    f=_fixture(tmp_path, dehardcoded=True, v15_protocol=True)
    obs,target=_v9_selected_detection(f,v15_protocol=True)
    def invoke(name,args):
        return _invoke(f,ToolApiLevel.S3_LOW_LEVEL,name,args,motion_tools=True,
            service_aligned=True,dehardcoded=True,v15_protocol=True)
    segmented=invoke("segment",{"prompt":"support", "observation_handle":obs["observation_handle"],
        "camera_name":"agentview"})
    destination=invoke("select_detection",{"detection_id":segmented.value["detections"][0]["detection_id"]}).value
    hand=invoke("gripper_state",{}).value
    grasps=invoke("grasp_pose_estimate",{"observation_handle":obs["observation_handle"],
        "detection_id":target["detection_id"],"gripper_profile":hand["profile"],
        "backend":"auto","depth_source":"sensor","grasp_style":"auto","top_k":1}).value
    candidate=dict(grasps["candidates"][0])
    candidate.update(backend=backend_name,backend_candidate=None,model_score=None,convergence_risk="unassessed")
    candidate.pop("grasp_handle",None)
    candidate=f.backend._issue_grasp_candidate(candidate)
    assert invoke("align_pregrasp",{"candidate":candidate}).success
    assert invoke("descend_and_close",{}).success
    track=f.backend.registry.get(grasps["tracking_handle"],expected_kind="entity_track")
    delta=np.array([0.,0.,.1])
    f.backend._object_clouds=lambda *_args:(track.full_points,track.object_points+delta)
    state=f.robot.observation.robot_state
    f.robot.observation=ObservationData(f.robot.observation.cameras,RobotState(
        state.joint_positions_rad,tuple(track.centroid+delta),state.end_effector_quaternion_wxyz,.05))
    attached=invoke("assert_entity_attached",{"tracking_handle":grasps["tracking_handle"],
        "post_object_name":"target","use_multiview":True})
    assert attached.success and attached.value["verified"]
    fresh=f.backend.observe().value["observation_handle"]
    old=f.backend.registry.get(destination["detection_id"],expected_kind="detection")
    destination_id=f.backend.registry.put("detection",replace(old,observation_handle=fresh))
    def forbidden(*args,**kwargs):
        pytest.fail("public geometry must not invent native AnyPlace input or resegment an old view")
    f.services.predict_placements=forbidden
    f.backend._align_detection_to_observation_camera=forbidden
    args={"tracking_handle":grasps["tracking_handle"],"destination_detection_id":destination_id,
        "relation_type":"on","reference_tolerance_m":.02}
    return f,invoke,candidate,track,args


@pytest.mark.parametrize("backend_name",["public_geometry_refinement","mask_topdown_fallback"])
@pytest.mark.parametrize("explicit_handle",[False,True])
def test_geometry_grasp_can_propose_support_placement(tmp_path,backend_name,explicit_handle):
    f,invoke,candidate,track,args=setup_attachment(tmp_path,backend_name)
    if explicit_handle:args["selected_grasp"]={"grasp_handle":candidate["grasp_handle"]}
    result=invoke("placement_estimate",args)
    assert result.success,result.message
    value=result.value
    assert value["placement_backend"]==value["placement_pose_basis"]=="public_support_geometry"
    assert "visible_support_geometry_is_advisory" in value["quality_flags"]
    selected=value["placement_candidates"][0]
    assert selected["backend"]=="public_support_geometry"
    assert selected["backend_candidate"] is None and selected["model_score"] is None
    assert selected["position_m"]==pytest.approx(value["selected_placement_pose"]["position_m"])
    assert selected["quaternion_wxyz"]==pytest.approx(value["selected_placement_pose"]["quaternion_wxyz"])
    destination=f.backend.registry.get(args["destination_detection_id"],expected_kind="detection")
    predicted=np.array(value["predicted_target_centroid_m"])
    extent=track.bounds_max-track.bounds_min
    assert predicted[:2]==pytest.approx(destination.centroid[:2])
    assert predicted[2]==pytest.approx(destination.bounds_max[2]+extent[2]/2+.01)
    assert value["predicted_support_gap_m"]==pytest.approx(.01)
    assert value["geometry_refinement_m"]==pytest.approx([0,0,0])
    assert value["raw_selected_placement_pose"]==value["selected_placement_pose"]
    assert value["requested_destination_detection_id"]==args["destination_detection_id"]


def test_support_proposal_still_requires_verified_attachment(tmp_path):
    f,invoke,candidate,track,args=setup_attachment(tmp_path,"public_geometry_refinement")
    track.attachment_verified=False
    result=invoke("placement_estimate",args)
    assert not result.success and not result.executed
    assert result.value["reason"]=="attachment_pose_unknown"


def test_support_pose_does_not_claim_a_learned_rotation_refinement(tmp_path):
    f,invoke,candidate,track,args=setup_attachment(tmp_path,"public_geometry_refinement")
    state=f.robot.observation.robot_state
    f.robot.observation=replace(f.robot.observation,robot_state=replace(state,
        end_effector_quaternion_wxyz=(float(np.cos(.15)),0.,0.,float(np.sin(.15)))))
    result=invoke("placement_estimate",args)
    assert result.success,result.message
    assert result.value["raw_selected_placement_pose"]==result.value["selected_placement_pose"]
    assert result.value["raw_to_selected_orientation_delta_rad"]==pytest.approx(0.)
