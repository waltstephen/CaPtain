"""Optional public geometry and temporal measurements; never selects actions or scores tasks."""
from itertools import product
import hashlib
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from .backend import _depth_to_base_points, _filter_point_cloud_noise
from .motion_tools import pose_matrix

def cloud(tools, handle, *, fresh=True):
    detection=tools._require_selected_detection(handle)
    obs=tools.registry.get(detection.observation_handle,expected_kind="observation")
    if fresh:
        current=tools.robot.get_robot_state()
        if not current.success or current.value is None:raise ValueError("robot state unavailable")
        state=current.value
        if (abs(tools.robot.time_s-obs.time_s)>1e-9 or
            not np.allclose(state.joint_positions_rad,obs.robot_state.joint_positions_rad,rtol=0,atol=1e-6) or
            abs(state.gripper_open_fraction-obs.robot_state.gripper_open_fraction)>1e-6):
            raise ValueError("stale detection: observe, reselect the same target and compare before reusing geometry")
    camera=tools._camera_by_name(obs,detection.camera_name)
    if camera.depth is None:raise ValueError("public metric depth unavailable")
    points=_filter_point_cloud_noise(_depth_to_base_points(camera.depth,camera.intrinsics,camera.extrinsics,mask=detection.mask))
    if len(points)<6:raise ValueError("insufficient measured selected surface")
    return detection,obs,camera,points

def provenance(handle,detection,obs,camera):
    return {"detection_id":handle,"observation_handle":detection.observation_handle,
        "camera":detection.camera_name,"time_s":obs.time_s,"coordinate_frame":"robot_base",
        "rgb_sha256":hashlib.sha256(camera.image.tobytes()).hexdigest(),
        "depth_sha256":hashlib.sha256(camera.depth.tobytes()).hexdigest(),
        "identity":"policy-selected hypothesis; not independently verified"}

def project(camera,points):
    local=(np.asarray(points)-camera.extrinsics[:3,3])@camera.extrinsics[:3,:3]
    pixels=local@camera.intrinsics.T
    h,w=camera.image.shape[:2]
    return [[float(p[0]/p[2]/w),float(p[1]/p[2]/h)] if p[2]>1e-6 else None for p in pixels]

def pad_sweep(model,joints):
    model.set_state(joints,1.)
    eef=model.eef_pose()
    pads=[g for g in model.geometries if g.collidable and
        g.name.startswith(model.gripper_prefix) and "pad" in g.name and len(g.vertices)==8]
    if len(pads)!=2:raise ValueError("two box finger pads unavailable; no inferred contact envelope")
    corners=[];centers=[]
    for pad in pads:
        center,rotation=model.geom_pose(pad)
        points=pad.vertices@rotation.T+center
        corners.append((points-eef[:3,3])@eef[:3,:3])
        centers.append((center-eef[:3,3])@eef[:3,:3])
    axis=centers[1]-centers[0];axis=axis/np.linalg.norm(axis)
    reference=np.eye(3)[np.argmin(abs(axis))]
    cross=np.cross(axis,reference);cross/=np.linalg.norm(cross)
    basis=np.column_stack((cross,axis,np.cross(cross,axis)))
    a,b=(x@basis for x in corners)
    lower=np.maximum(a.min(axis=0),b.min(axis=0))
    upper=np.minimum(a.max(axis=0),b.max(axis=0))
    lower[1],upper[1]=a[:,1].max(),b[:,1].min()
    if np.any(upper<=lower):raise ValueError("finger-pad free closing volume unavailable")
    return basis,lower,upper,corners

def contact_tests(model,joints,aperture,points,pose,camera,closing_axis,push_direction=None):
    """Separate empty jaw volume from actual pads and partial visible surface normals."""
    from scipy.spatial import cKDTree
    model.set_state(joints,aperture)
    current=model.eef_pose()
    pads=[g for g in model.geometries if g.collidable and g.name.startswith(model.gripper_prefix)
        and "pad" in g.name and len(g.vertices)==8]
    evidence=[]
    near_any=np.zeros(len(points),dtype=bool)
    for pad in pads:
        center,rotation=model.geom_pose(pad)
        local_center=current[:3,:3].T@(center-current[:3,3])
        local_rotation=current[:3,:3].T@rotation
        planned_center=pose[:3,:3]@local_center+pose[:3,3]
        planned_rotation=pose[:3,:3]@local_rotation
        local=(points-planned_center)@planned_rotation
        lower,upper=pad.vertices.min(axis=0),pad.vertices.max(axis=0)
        distances=np.linalg.norm(np.maximum(lower-local,0)+np.maximum(local-upper,0),axis=1)
        near=distances<=.003;near_any|=near
        evidence.append({"pad":pad.name,"nearest_visible_surface_m":float(distances.min()),
            "visible_points_within_3mm":int(near.sum())})
    selected=np.linspace(0,len(points)-1,min(128,len(points))).astype(int)
    tree=cKDTree(points);normal_rows=[]
    for i in selected:
        distances,neighbors=tree.query(points[i],k=min(16,len(points)))
        sample=points[np.asarray(neighbors)[np.asarray(distances)<=.015]]
        if len(sample)<6:continue
        cov=np.cov(sample.T);values,vectors=np.linalg.eigh(cov)
        if values[1]<=1e-8 or values[0]/max(values.sum(),1e-12)>.1:continue
        normal=vectors[:,0]
        if normal@(camera.extrinsics[:3,3]-points[i])<0:normal=-normal
        normal_rows.append({"point_index":int(i),"point_robot_base_m":points[i].tolist(),
            "view_facing_normal":normal.tolist(),"closing_axis_dot":float(normal@closing_axis),
            "near_actual_pad":bool(near_any[i])})
    result={"assessed_open_fraction":float(aperture),"actual_pad_proximity":evidence,
        "visible_local_normals":normal_rows,
        "two_sided_grasp_contact":"unverified; view-facing normals do not reveal hidden opposing surfaces or friction",
        "closure_volume_is_physical_contact":False,
        "scope":"partial local surface geometry at the specified pose/aperture; no grasp score"}
    if push_direction is not None:
        direction=np.asarray(push_direction,dtype=float)
        if direction.shape!=(3,) or not np.isfinite(direction).all() or np.linalg.norm(direction)<1e-9:
            raise ValueError("push_direction must be a finite nonzero vector in robot_base")
        direction/=np.linalg.norm(direction)
        aligned=[r for r in normal_rows if r["near_actual_pad"] and np.dot(r["view_facing_normal"],direction)<-.5]
        result["push_test"]={"direction_robot_base":direction.tolist(),
            "visible_near_pad_normals_opposing_motion":len(aligned),
            "status":"local_contact_hypothesis_supported" if aligned else "effective_contact_unverified",
            "limitation":"A local normal/proximity match is only a test hypothesis; verify object movement after a bounded push."}
    return result

def contact(tools, target_detection_id, position, quaternion, push_direction=None):
    detection,obs,camera,points=cloud(tools,target_detection_id)
    pose=pose_matrix(position,quaternion)
    model=tools._motion_model()
    basis,lower,upper,corners=pad_sweep(model,obs.robot_state.joint_positions_rad)
    local=(points-pose[:3,3])@pose[:3,:3]@basis
    inside=np.all((local>=lower)&(local<=upper),axis=1)
    nearest=np.linalg.norm(np.maximum(lower-local,0)+np.maximum(local-upper,0),axis=1)
    selected=local[inside]
    margin=(np.minimum(selected-lower,upper-selected).min(axis=1) if len(selected) else np.array([]))
    world_corners=[x@pose[:3,:3].T+pose[:3,3] for x in corners]
    source=provenance(target_detection_id,detection,obs,camera)
    tests=contact_tests(model,obs.robot_state.joint_positions_rad,obs.robot_state.gripper_open_fraction,
            points,pose,camera,pose[:3,:3]@basis[:,1],push_direction)
    candidate_evidence={"query_status":"assessed","source":source,
        "requested_eef_position_m":list(position),"requested_quaternion_wxyz":list(quaternion),
        "closing_axis_robot_base":(pose[:3,:3]@basis[:,1]).tolist(),
        "visible_closing_volume_support":"observed" if inside.any() else "not_observed",
        "visible_points_in_closing_volume":int(inside.sum()),
        "nearest_visible_surface_to_closing_volume_m":float(nearest.min()),
        "open_pad_proximity":tests["actual_pad_proximity"],
        "opposing_contact":"unverified","object_retention":"unverified",
        "decision_scope":"Visible overlap and unsigned open-pad distance are not a closure/contact score. Missing overlap leaves this contact hypothesis unsupported by visible points; occlusion remains unknown. A failed query is unassessed. Choose evidence, revised geometry or an explicitly uncertain contact test; no action permission is inferred."}
    return {"source":source,"candidate_evidence":candidate_evidence,
        "requested_eef_position_m":list(position),"requested_quaternion_wxyz":list(quaternion),
        "closing_axis_eef":basis[:,1].tolist(),
        "closing_axis_robot_base":(pose[:3,:3]@basis[:,1]).tolist(),
        "pad_sweep_min_m":lower.tolist(),"pad_sweep_max_m":upper.tolist(),
        "target_bounds_in_pad_axes_m":[local.min(axis=0).tolist(),local.max(axis=0).tolist()],
        "visible_points_in_closing_volume":int(inside.sum()),"visible_points_total":len(points),
        "nearest_visible_surface_to_closing_volume_m":float(nearest.min()),
        "median_interior_margin_m":float(np.median(margin)) if len(margin) else None,
        "nominal_position_tolerance_m":.003,
        "two_sided_contact":"unverified; enclosed visible points do not prove opposing contact normals",
        "contact_tests":tests,
        "pad_corners_robot_base":[x.tolist() for x in world_corners],
        "pad_corner_pixels_normalized":[project(camera,x) for x in world_corners],
        "physical_action_executed":False,
        "limitation":"Diagnostic geometry for your exact pose, not a grasp proposal or permission. Occluded surfaces, contact normals, friction and object retention remain unknown. Small margin relative to execution error requires reconsideration."}

def compare(tools, before_detection_id, after_detection_id, action_ids=None):
    first,old,a,_=cloud(tools,before_detection_id,fresh=False)
    last,new,b,_=cloud(tools,after_detection_id)
    if old.time_s>=new.time_s:raise ValueError("comparison requires an earlier observation")
    if first.camera_name!=last.camera_name:raise ValueError("use the same calibrated camera for temporal correspondence")
    from .visual_tracking import track_rgbd
    tracked=track_rgbd(a,b,first.mask,target_mask=last.mask)
    eef=np.asarray(new.robot_state.end_effector_position_m)-old.robot_state.end_effector_position_m
    translation=tracked.get("translation_m")
    resolution=max(.003,3*float(tracked.get("motion_spread_m") or 0.))
    comparable=tracked.get("status")=="supported" and tracked.get("association_status")=="supported"
    stalled=(comparable and np.linalg.norm(eef)>.01 and np.linalg.norm(translation)<=resolution)
    return {"before":provenance(before_detection_id,first,old,a),"after":provenance(after_detection_id,last,new,b),
        "tracking":tracked,"measured_eef_displacement_m":eef.tolist(),
        "measured_object_displacement_m":translation if comparable else None,
        "object_motion_resolution_m":resolution if comparable else None,
        "effect":"robot_moved_without_resolved_target_motion" if stalled else "measured_motion" if comparable else "unknown",
        "object_to_eef_displacement_difference_m":(np.asarray(translation)-eef).tolist() if comparable else None,
        "physical_action_executed":False,"retention":"not_proven",
        "limitation":"Correspondence is fallible. Metrics compensate camera motion using public calibration; different visible surfaces or lost identity remain unknown. Displacement alone does not establish attachment or the final task relation."}

def release_projection(points,current_pose,release_pose,surface_points,margin=.003):
    local=(points-current_pose[:3,3])@current_pose[:3,:3]
    predicted=local@release_pose[:3,:3].T+release_pose[:3,3]
    if len(surface_points)<6:raise ValueError("insufficient visible selected support plane")
    try:
        hull=ConvexHull(surface_points[:,:2])
    except QhullError as exc:
        raise ValueError("visible support points do not span a surface") from exc
    equations=hull.equations
    clearance=-(predicted[:,:2]@equations[:,:2].T+equations[:,2])
    return {"predicted_visible_object_bounds_m":[predicted.min(axis=0).tolist(),predicted.max(axis=0).tolist()],
        "support_polygon_xy_m":surface_points[hull.vertices,:2].tolist(),
        "minimum_projected_boundary_clearance_m":float(clearance.min()),
        "visible_object_inside_observed_support_polygon":bool(np.all(clearance>=margin)),
        "predicted_visible_bottom_above_surface_m":float(predicted[:,2].min()-np.median(surface_points[:,2])),
        "boundary_margin_m":margin}

def release(tools,target_detection_id,destination_detection_id,position,quaternion,surface_height_m):
    if not np.isfinite(surface_height_m):raise ValueError("finite declared surface height required")
    target,obs,a,points=cloud(tools,target_detection_id)
    dest,other,b,support=cloud(tools,destination_detection_id)
    if abs(obs.time_s-other.time_s)>1e-9:raise ValueError("target and support must share a current observation time")
    selected=support[abs(support[:,2]-surface_height_m)<=.005]
    current=pose_matrix(obs.robot_state.end_effector_position_m,obs.robot_state.end_effector_quaternion_wxyz)
    value=release_projection(points,current,pose_matrix(position,quaternion),selected)
    return {**value,"target_source":provenance(target_detection_id,target,obs,a),
        "destination_source":provenance(destination_detection_id,dest,other,b),
        "declared_surface_height_m":surface_height_m,"surface_band_half_width_m":.005,
        "physical_action_executed":False,"support_relation":"unverified",
        "assumptions":"Selected object remains rigidly attached with its CURRENT EEF-relative pose; gravity projection to a horizontal support plane.",
        "limitation":"The model chooses the destination and height from images. Convex visible support hull can span holes and occlusion, and object bounds omit hidden parts. A negative margin exposes a mismatch; a positive margin is only partial geometry, never release authorization or task success."}
