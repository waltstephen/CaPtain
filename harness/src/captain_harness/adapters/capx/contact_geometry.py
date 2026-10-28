"""Public pad geometry evidence; overlap never certifies contact or attachment."""
from collections.abc import Mapping
import numpy as np


def surface_quality(points, lower, upper, margin):
    """Bounded observed-surface geometry, explicitly not physical grasp truth."""
    from scipy.spatial import cKDTree
    voxel = max(float(margin) / 3., .0005)
    selected = points[np.all((points >= lower) & (points <= upper), axis=1)]
    if len(selected):
        _, indices = np.unique(np.floor(selected / voxel).astype(np.int64), axis=0, return_index=True)
        selected = selected[indices]
        if len(selected) > 1024:
            selected = selected[np.linspace(0, len(selected)-1, 1024, dtype=int)]
    result = {'sampled_points': len(selected), 'voxel_size_m': voxel,
        'closing_axis_span_m': None, 'closing_axis_offset_m': None,
        'observed_depth_span_m': None, 'depth_edge_reserve_m': None,
        'negative_side_normal_points': 0, 'positive_side_normal_points': 0,
        'normal_correspondence_cells': 0, 'opposing_surface_status': 'unverified',
        'force_closure_verified': False, 'normal_alignment_cos_min': float(np.cos(np.pi/6)),
        'scope': 'Visible samples inside calibrated pad sweep; local PCA normals are unsigned. Opposing patches do not prove physical contact, friction, semantic identity or retention. Missing support can be occlusion.'}
    if not len(selected):
        return result
    lo, hi = np.min(selected, axis=0), np.max(selected, axis=0)
    span = hi[1] - lo[1]
    result.update(closing_axis_span_m=float(span), closing_axis_offset_m=float((lo[1]+hi[1])/2),
        observed_depth_span_m=float(hi[2]-lo[2]), depth_edge_reserve_m=float(min(lo[2]-lower[2],upper[2]-hi[2])))
    if len(selected) < 12 or span <= 2*max(margin, voxel):
        return result
    distances, neighbors = cKDTree(selected).query(selected, k=12)
    local = selected[neighbors]
    centered = local-local.mean(axis=1,keepdims=True)
    covariance = np.einsum('nki,nkj->nij',centered,centered)/12
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    reliable = ((distances[:,-1] <= max(3*margin, 4*voxel)) & (eigenvalues[:,1] > voxel**2*.01)
        & (eigenvalues[:,0] <= .2*eigenvalues[:,1]))
    aligned = reliable & (np.abs(eigenvectors[:,1,0]) >= result['normal_alignment_cos_min'])
    band = max(margin, voxel)
    negative = aligned & (selected[:,1] <= lo[1]+band)
    positive = aligned & (selected[:,1] >= hi[1]-band)
    result['negative_side_normal_points'] = int(negative.sum())
    result['positive_side_normal_points'] = int(positive.sum())
    cell_size = max(2*margin, 2*voxel)
    cells = np.floor((selected[:,[0,2]]-lower[[0,2]])/cell_size).astype(int)
    overlap = set(map(tuple,cells[negative])) & set(map(tuple,cells[positive]))
    result['normal_correspondence_cells'] = len(overlap)
    result['normal_correspondence_cell_size_m'] = cell_size
    if negative.sum() >= 6 and positive.sum() >= 6 and overlap:
        result['opposing_surface_status'] = 'opposing_visible_patches'
    return result


def pose_sensitivity(points, lower, upper, translation_m, rotation_rad=np.pi/36):
    """Thirteen bounded pose perturbations on the same public source cloud."""
    from scipy.spatial.transform import Rotation
    if len(points) > 2048:
        points = points[np.linspace(0, len(points)-1, 2048, dtype=int)]
    def count(cloud):
        return int(np.all((cloud >= lower) & (cloud <= upper), axis=1).sum())
    nominal = count(points)
    trials = [{'kind': 'nominal', 'axis': None, 'magnitude': 0., 'interior_points': nominal}]
    for axis in range(3):
        for sign in (-1, 1):
            delta = np.zeros(3); delta[axis] = sign*translation_m
            trials.append({'kind': 'translation_m', 'axis': axis, 'magnitude': float(delta[axis]),
                'interior_points': count(points-delta)})
            angle = np.zeros(3); angle[axis] = sign*rotation_rad
            trials.append({'kind': 'rotation_rad', 'axis': axis, 'magnitude': float(angle[axis]),
                'interior_points': count(points @ Rotation.from_rotvec(angle).as_matrix())})
    worst = min(trials, key=lambda trial: trial['interior_points'])
    return {'status': 'sampled' if len(points) else 'unknown',
        'sampled_source_points': len(points), 'evaluated_poses': len(trials),
        'translation_perturbation_m': float(translation_m), 'rotation_perturbation_rad': float(rotation_rad),
        'nominal_interior_points': nominal, 'minimum_sampled_interior_points': worst['interior_points'],
        'minimum_retained_fraction_of_nominal': worst['interior_points']/nominal if nominal else None,
        'some_sampled_poses_lose_all_visible_overlap': (worst['interior_points'] == 0) if nominal else None,
        'worst_sample': worst, 'force_closure_verified': False,
        'scope': 'Independent plus/minus XYZ translations and rotations about the EEF on the same source cloud. Samples do not cover combined errors, unseen surfaces, sensor error or physical force closure; perturbations are not calibrated uncertainty estimates.'}


def pad_support(local_points, lower, upper, position_tolerance_m):
    points=np.asarray(local_points,dtype=float).reshape(-1,3)
    points=points[np.isfinite(points).all(axis=1)]
    lo=np.asarray(lower,dtype=float).reshape(3);hi=np.asarray(upper,dtype=float).reshape(3)
    margin=float(position_tolerance_m)
    if not np.isfinite(margin) or margin<0 or not np.isfinite([lo,hi]).all() or np.any(lo>hi):
        raise ValueError('invalid pad envelope or positional tolerance')
    inside=np.all((points>=lo)&(points<=hi),axis=1)
    dilated=np.all((points>=lo-margin)&(points<=hi+margin),axis=1)
    resilient=np.all((points>=lo+margin)&(points<=hi-margin),axis=1)
    gap=np.maximum(np.maximum(lo-points,points-hi),0.)
    return {'pose_sensitivity':pose_sensitivity(points,lo,hi,margin),
        'surface_quality':surface_quality(points,lo,hi,margin),'finite_points':len(points),'interior_points':int(inside.sum()),
        'tolerance_sweep_points':int(dilated.sum()),'translation_resilient_points':int(resilient.sum()),
        'position_tolerance_m':margin,
        'support_status':('unknown' if not len(points) else
            'interior_with_translation_reserve' if resilient.any() else
            'edge_sensitive' if inside.any() else
            'tolerance_only' if dilated.any() else 'no_visible_overlap'),
        'minimum_visible_point_distance_to_sweep_m':float(np.linalg.norm(gap,axis=1).min()) if len(points) else None,
        'bilateral_contact_status':'unverified',
        'limitation':'Eroded-box support covers positional translation only. It does not cover rotation, depth error, unseen surfaces, opposing surface normals, friction or target retention.'}


def pose_support_comparison(points, requested_center, requested_hand_rotation,
                            measured_center, measured_hand_rotation, lower, upper, tolerance_m):
    points=np.asarray(points,dtype=float).reshape(-1,3)
    target=np.asarray(requested_center,dtype=float).reshape(3)
    actual=np.asarray(measured_center,dtype=float).reshape(3)
    planned=np.asarray(requested_hand_rotation,dtype=float).reshape(3,3)
    measured=np.asarray(measured_hand_rotation,dtype=float).reshape(3,3)
    if not all(np.isfinite(a).all() for a in [target,actual,planned,measured]):
        raise ValueError('nonfinite contact pose')
    nominal=pad_support((points-target)@planned,lower,upper,tolerance_m)
    observed=pad_support((points-actual)@measured,lower,upper,tolerance_m)
    relation=('unknown' if not nominal['finite_points'] else
        'nominal_overlap_lost_at_measured_pose' if nominal['interior_points']>0 and observed['interior_points']==0 else
        'overlap_at_both_poses' if nominal['interior_points']>0 and observed['interior_points']>0 else
        'nominal_has_no_interior_overlap')
    return {'nominal':nominal,'measured_pose_on_source_geometry':observed,'relation':relation,
        'requested_eef_position_m':target.tolist(),'measured_eef_position_m':actual.tolist(),
        'position_residual_m':float(np.linalg.norm(actual-target)),
        'measured_minus_requested_hand_m':((actual-target)@planned).tolist(),
        'orientation_residual_rad':float(np.arccos(np.clip((np.trace(planned.T@measured)-1)/2,-1,1))),
        'fresh_target_contact_verified':False,
        'basis':'Two robot poses applied to the SAME source target cloud; not a fresh target observation.',
        'guidance':'Separate a bad nominal contact region from failure to reach it. Inspect current target and finger images before retrying; source-cloud overlap does not establish a grasp.'}


def measured_grasp_geometry(backend, aligned, requested_position, centering_delta):
    """Read-only post-action diagnostic; failure must not erase physical receipts."""
    try:
        hardware=getattr(backend.robot,'gripper_contact_envelope',None)
        envelope=hardware() if callable(hardware) else None
        if not isinstance(envelope,Mapping) or not aligned.tracking_handle:
            return {'status':'unknown','reason':'target_cloud_or_pad_calibration_unavailable'}
        tracked=backend.registry.get(aligned.tracking_handle,expected_kind='entity_track')
        state=backend.robot.get_robot_state()
        if not state.success or state.value is None:
            return {'status':'unknown','reason':'post_action_robot_state_unavailable'}
        from .backend import _quaternion_wxyz_to_matrix
        # Use the calibrated public conversion for both orientations. EEF and
        # hand frames are not interchangeable on Panda (roughly a quarter turn).
        _,pq=backend.robot.external_ik_target_from_eef_pose(requested_position,aligned.eef_quaternion)
        _,mq=backend.robot.external_ik_target_from_eef_pose(state.value.end_effector_position_m,state.value.end_effector_quaternion_wxyz)
        lo=np.asarray(envelope['pad_min_local_m'],float).copy();hi=np.asarray(envelope['pad_max_local_m'],float).copy()
        lo[1],hi[1]=-backend.gripper_profile.max_width_m/2,backend.gripper_profile.max_width_m/2
        result=pose_support_comparison(np.asarray(tracked.object_points)+centering_delta,
            requested_position,_quaternion_wxyz_to_matrix(pq),state.value.end_effector_position_m,
            _quaternion_wxyz_to_matrix(mq),lo,hi,.003)
        return {**result,'status':'measured_robot_pose_on_source_cloud',
            'source_observation_handle':aligned.candidate.get('source_observation_handle'),
            'source_detection_id':aligned.detection_id,
            'source_translation_correction_m':np.asarray(centering_delta).tolist(),
            'measurement_phase':'after_descent_and_close_return',
            'aperture_scope':'maximum-width pad sweep, not the occupied closed finger boxes'}
    except Exception as exc:
        # This optional evidence is collected AFTER the physical action. Keep
        # its original success/failure and execution provenance on sensor errors.
        return {'status':'unknown','reason':'contact_geometry_measurement_unavailable',
                'measurement_error_type':type(exc).__name__}
