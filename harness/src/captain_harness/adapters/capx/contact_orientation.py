"""Public orientation proposals that preserve one calibrated contact region.

Endpoint and visible geometry checks do not prove force closure or retention.
All alternatives remain auditable, including infeasible and jaw-swap proposals.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from captain_harness.adapters.capx.motion_tools import pose_matrix, quaternion
from captain_harness.adapters.robotics.surfaces import support_plane
from captain_harness.adapters.robotics.grasp_proposals import surface_orientations
from captain_harness.ports.capability import CapabilityResult, FailureCategory


def inspect_orientations(backend, candidate, scene_handle):
    scene, target = backend._candidate_scene_target(candidate, scene_handle)
    original = pose_matrix(candidate['position_m'],candidate['quaternion_wxyz'])
    calibration = backend.robot.gripper_contact_envelope()
    if not calibration:
        return CapabilityResult.failed(FailureCategory.NOT_EXECUTED, 'finger-pad calibration is unavailable')
    pad = (np.asarray(calibration['pad_min_local_m']) + calibration['pad_max_local_m'])/2
    pad[1] = 0
    center, _ = backend._contact_local_points(original,np.empty((0,3)))
    reference = center + original[:3,:3] @ pad
    plane = support_plane(scene.points,target,backend.gripper_profile.max_width_m+backend.gripper_profile.fingertip_depth_m)
    if plane is None:
        return CapabilityResult.ok({'alternatives':[],'selected_candidate':None,'reason':'support_plane_unavailable'})
    model=backend._motion_model()
    collision,policy=backend._grasp_planning_collision_checker(model,scene,target)
    alternatives=[]
    for name,rotation in surface_orientations(original[:3,:3],target,plane[0]):
        angle=float(Rotation.from_matrix(rotation@original[:3,:3].T).magnitude())
        if angle < 1e-6: continue
        pose=np.eye(4);pose[:3,:3]=rotation
        offset,_=backend._contact_local_points(pose,np.empty((0,3)))
        pose[:3,3]=reference-rotation@pad-offset
        proposal={**candidate,'position_m':pose[:3,3].tolist(),'quaternion_wxyz':quaternion(pose),
            'approach_direction_world':rotation[:,2].tolist(),'backend_candidate':None,
            'backend':'public_geometry_refinement','model_score':None,'style_preference_score':None,
            'request_style_match':'unassessed','convergence_risk':'unassessed',
            'predicted_joint_travel_rad':None,
            'refinement_basis':'public support plane orientation, fixed calibrated pad-center point'}
        proposal.pop('grasp_handle',None);proposal.pop('feasibility_evidence',None)
        measured=backend.evaluate_grasp(proposal,scene_handle)
        if not measured.success:
            alternatives.append({'kind':name,'angle_rad':angle,'admissible':False,'error':measured.message});continue
        evidence=measured.value
        width=backend._predicted_grasp_width(pose,target)
        reach=backend._grasp_endpoint_reachability(proposal,scene.joints,
            pregrasp_valid=lambda q:not collision(q)['collision'])
        pre=(collision(reach['joints']) if reach['joints'] is not None else {'collision':None})
        known_width=width is not None and width <= backend.gripper_profile.max_width_m
        proposal.update(predicted_grasp_width_m=width,width_known=width is not None,width_feasible=known_width,
            reachable=reach['reachable'],ik_feasible=reach['reachable'],source_observation_handle=scene.source_observation_handle,
            top_down_alignment=float(np.clip(-rotation[2,2],0,1)),quality_flags=['geometry_refinement_unscored'],
            target_contact_evidence=evidence['contact'])
        if reach['joints'] is not None:
            proposal['predicted_joint_travel_rad'] = float(np.sum(np.abs(np.asarray(reach['joints']) - scene.joints)))
        moved_center,_=backend._contact_local_points(pose,np.empty((0,3)))
        reference_error=float(np.linalg.norm(moved_center+rotation@pad-reference))
        admissible=bool(known_width and reach['reachable'] and pre['collision'] is False
            and all(count == 0 for count in evidence['approach']['geometry_hits'].values()) and evidence['contact'].get('visible_point_count_in_pad_sweep',0)>0)
        distinct = not (float(rotation[:,2] @ original[:3,2]) > 1-1e-6
                        and abs(float(rotation[:,1] @ original[:3,1])) > 1-1e-6)
        alternatives.append({'kind':name,'angle_rad':angle,'admissible':admissible,'distinct_contact_orientation':distinct,'candidate':proposal,
            'contact_reference_error_m':reference_error,'contact':evidence['contact'],'approach':evidence['approach'],
            'reachability':reach,'pregrasp_collision':pre,'collision_policy':policy})
    selected=min((r for r in alternatives if r['admissible'] and r['distinct_contact_orientation']),key=lambda r:r['angle_rad'],default=None)
    return CapabilityResult.ok({'alternatives':alternatives,'selected_candidate':selected['candidate'] if selected else None,
        'selected_kind':selected['kind'] if selected else None,'reference_world_m':reference.tolist(),
        'selection_basis':'smallest changed orientation angle among unchanged measured constraints; no force-closure guarantee',
        'reason':'candidate_found' if selected else 'no_admissible_changed_orientation'})


def expand_contact_orientation(backend, refinement):
    """Measure around the final coarse/local choice before style adjudication."""
    from copy import deepcopy
    from dataclasses import replace
    from collections.abc import Mapping
    if not refinement.success or not isinstance(refinement.value, Mapping):
        return refinement
    value = deepcopy(dict(refinement.value))
    candidate = value.get('candidate')
    if not isinstance(candidate, Mapping):
        return refinement
    try:
        scene_handle = backend._grasp_scene_handle(candidate, None)
        inspected = inspect_orientations(backend, candidate, scene_handle)
    except (ValueError, RuntimeError) as exc:
        value['contact_orientation'] = {'status':'unavailable', 'reason':f'{type(exc).__name__}: {exc}'}
        return replace(refinement, value=value)
    report = deepcopy(dict(inspected.value or {}))
    report.update(status='measured' if inspected.success else 'unavailable',
        input_candidate=deepcopy(candidate), input_checks=deepcopy(value.get('after', {})))
    value['contact_orientation'] = report
    if not inspected.success:
        report['reason'] = inspected.message
        return replace(refinement, value=value)
    proposed = report.get('selected_candidate')
    if proposed is None:
        return replace(refinement, value=value)
    index = next(i for i,r in enumerate(report['alternatives']) if r.get('kind')==report['selected_kind'])
    chosen = report['alternatives'][index]
    issued = backend._issue_grasp_candidate(proposed)
    report['selected_candidate'] = issued
    value.update(candidate=issued, after={
        key:deepcopy(chosen[key]) for key in ('contact','approach','reachability','collision_policy')},
        coarse_selected_index=value.get('selected_index'), selected_index=index,
        selection_stage='contact_orientation', pose_changed=True,
        translation_m=(np.asarray(issued['position_m'])-candidate['position_m']).tolist(),
        translation_reference='contact_orientation input_candidate',
        reason='contact_region_preserving_orientation_passed_measured_checks', requires_decision=True)
    value['after']['pregrasp_collision'] = {'checked':True, **chosen['pregrasp_collision']}
    return replace(refinement, value=value)
