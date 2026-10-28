"""Register public read-only measurements with source pixels, selection and calibration."""
from copy import deepcopy
import hashlib
import numpy as np
from captain_harness.ports.measurement import make_measurement

def source_record(tools,handle,*,kind="detection"):
    detection=(tools._require_selected_detection(handle) if kind=="detection"
        else tools.registry.get(handle,expected_kind=kind))
    obs=tools.registry.get(detection.observation_handle,expected_kind="observation")
    camera=tools._camera_by_name(obs,detection.camera_name)
    history=tools._public_observation_history[detection.observation_handle]
    mask=np.asarray(detection.mask,dtype=bool)
    flat=mask.reshape(-1).astype(np.uint8)
    edges=np.r_[0,np.flatnonzero(flat[1:]!=flat[:-1])+1,len(flat)]
    ys,xs=np.nonzero(mask)
    choice=np.linspace(0,len(xs)-1,min(24,len(xs))).astype(int)
    samples=[]
    for index in choice:
        x,y=int(xs[index]),int(ys[index])
        z=float(camera.depth[y,x]) if camera.depth is not None else float("nan")
        if np.isfinite(z) and z>0:
            optical=np.linalg.solve(camera.intrinsics,[x,y,1.])*z
            point=camera.extrinsics[:3,:3]@optical+camera.extrinsics[:3,3]
            samples.append({"pixel_xy":[x,y],"depth_m":z,"robot_base_point_m":point.tolist()})
    return {"camera":detection.camera_name,"step":history.step,"sample_time_s":obs.time_s,
        "observation_handle":detection.observation_handle,"selection_handle":handle,"selection_kind":kind,
        "detection_id":handle if kind=="detection" else None,
        "rgb_pixels_sha256":hashlib.sha256(camera.image.astype(np.uint8).tobytes()).hexdigest(),
        "rgb_shape":list(camera.image.shape),
        "depth_values_sha256":hashlib.sha256(np.asarray(camera.depth,dtype=np.float64).tobytes()).hexdigest(),
        "mask_sha256":hashlib.sha256(mask.tobytes()).hexdigest(),"mask_shape":list(mask.shape),
        "mask_rle":{"initial":int(flat[0]),"counts":np.diff(edges).tolist()},
        "selected_public_samples":samples,"intrinsics_3x3":camera.intrinsics.tolist(),
        "camera_to_robot_base_4x4":camera.extrinsics.tolist(),
        "identity_scope":"model-selected mask, not verified task identity"}

def register(direct,name,args,value):
    if name=="inspect_contact_geometry":
        handles=[args["target_detection_id"]]
        fields=("requested_eef_position_m","requested_quaternion_wxyz","closing_axis_robot_base",
            "visible_points_in_closing_volume","nearest_visible_surface_to_closing_volume_m",
            "median_interior_margin_m","two_sided_contact","contact_tests","candidate_evidence","closing_axis_eef")
        kind="hypothetical_contact_geometry"
    elif name=="measure_object_geometry":
        handles=[args["target_detection_id"]]+([args["destination_detection_id"]] if args.get("destination_detection_id") else [])
        fields=("object","destination","object_to_destination_xy_offset_m","xy_distance_m","xy_offset_uncertainty_m")
        kind="measured_object_geometry"
    elif name=="inspect_release_geometry":
        handles=[args["target_detection_id"],args["destination_detection_id"]]
        fields=("predicted_visible_object_bounds_m","support_polygon_xy_m",
            "minimum_projected_boundary_clearance_m","visible_object_inside_observed_support_polygon",
            "predicted_visible_bottom_above_surface_m","declared_surface_height_m","assumptions")
        kind="hypothetical_release_geometry"
    else:
        handles=[args["before_detection_id"],args["after_detection_id"]]
        fields=("tracking","measured_eef_displacement_m","measured_object_displacement_m",
            "object_motion_resolution_m","effect","object_to_eef_displacement_difference_m","retention")
        kind="observed_effect_comparison"
    sources=[source_record(direct.tools,h) for h in handles]
    body={"kind":kind,"sources":sources,"result":{k:deepcopy(value[k]) for k in fields if k in value},
        "uncertainty":value.get("limitation",""),"physical_action_executed":False}
    receipt=make_measurement(body)
    direct._measurement_receipts[receipt["measurement_id"]]=receipt
    answer={**value,
        "measurement_reference":{"measurement_id":receipt["measurement_id"],"kind":kind,
            "source_observations":[{k:r.get(k) for k in ("camera","step","observation_handle","selection_handle")} for r in sources],
            "scope":"Cite this measurement ID; the host resolves source images and checks RGB-D/time/calibration dependencies within the image budget. Full body is durably stored."}}
    if kind=="observed_effect_comparison":
        start,end=sources[0]["step"],sources[1]["step"]
        actual=[r for r in direct._issued_action_receipts.values() if r["executed"] and
            start<=r["start_step"] and r["end_step"]<=end and r["end_step"]>r["start_step"]]
        issued=[r["action_id"] for r in actual]
        requested=args.get("action_ids",[])
        if not requested:
            requested=issued
        bound=bool(requested) and len(requested)==len(set(requested)) and set(requested)==set(issued)
        answer["effect_evidence"]={"measurement_id":receipt["measurement_id"],
            "action_ids":requested,"issued_action_ids_in_interval":issued,
            "source_interval":{"start_step":start,"end_step":end},
            "association_status":"explicit_action_span" if bound else "unbound_action_span",
            "association_source":"host_resolved_observation_interval" if not args.get("action_ids") else "policy_ids_validated_against_interval",
            "binding_guidance":"The host resolves ALL executed physical actions between the selected observations; explicit action_ids, if supplied, must match. An interval cannot isolate one action's contribution.",
            "identity_hypothesis":{"before_detection_id":handles[0],"after_detection_id":handles[1],
                "semantic_identity":"model_claim","visual_correspondence":value.get("tracking",{}).get("association_status","unknown")},
            "observed_effect":{k:v for k,v in body["result"].items() if k!="tracking"},"uncertainty":body["uncertainty"],
            "attribution_scope":"interval_aggregate_not_per_action"}
    return answer


def register_observation(direct,name,result):
    value=result.value
    if not result.success or not isinstance(value,dict):
        return result
    handle=value.get("point_handle") if name in {"point","mark_point"} else value.get("detection_id")
    if not handle:return result
    from dataclasses import replace
    try:
        source=source_record(direct.tools,handle,kind="marked_point" if name in {"point","mark_point"} else "detection")
        fields=("pixel_xy","position_m","point_centroid_m","point_bounds_min_m","point_bounds_max_m",
            "centroid_m","extent_m","quality_flags","grounding_status")
        record=make_measurement({"kind":"selected_public_surface","sources":[source],
            "result":{k:deepcopy(value[k]) for k in fields if k in value},
            "uncertainty":"Visible selected surface only; selection and labels do not prove task identity or final relation.",
            "physical_action_executed":False})
        direct._measurement_receipts[record["measurement_id"]]=record
        return replace(result,value={**value,
            "measurement_reference":{"measurement_id":record["measurement_id"],"kind":"selected_public_surface",
                "source_observations":[{k:source.get(k) for k in ("camera","step","observation_handle","selection_handle")}]}})
    except (KeyError,ValueError,TypeError,AttributeError) as exc:
        # Preserve the real perception result; expose why it cannot serve as bound evidence.
        return replace(result,value={**value,"measurement_registration":{
            "status":"unavailable","reason":str(exc)}})
