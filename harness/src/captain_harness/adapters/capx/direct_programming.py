"""R30 direct Cartesian programming surface; no grasp proposal or task policy."""
from dataclasses import asdict, replace
from types import SimpleNamespace
import uuid
import numpy as np
from scipy.spatial.transform import Rotation
from captain_harness.ports.capability import CapabilityManifest, CapabilitySpec, CapabilityResult, FailureCategory, SideEffect, ControlSemantic
from captain_harness.ports.deadline import check_deadline, DeadlineExceeded
from .motion_tools import pose_matrix, quaternion
from .direct_geometry import detection_handles, selected_geometry

PERCEPTION = frozenset({"observe", "segment", "select_detection", "point", "mark_point",
    "get_robot_state", "gripper_state", "track_target", "read_motion_trajectory"})
USAGE = "Control this robot by writing Python from public images and measurements.\nmove_to(position=..., quaternion=...) commands the calibrated EEF in robot_base, wxyz.\nThe EEF site is near the finger-pad midpoint, not the wrist or palm. Use the\npublished end_effector_geometry; do not add an assumed finger length. Use its closing_axis_eef rotated by YOUR requested EEF quaternion for jaw direction, never an assumed EEF Y axis.\nmove_to(gripper='close') or 'open' changes only the jaws. Jaw commands persist.\nhold(duration_s) keeps the current measured EEF pose and jaw command while time advances.\nA physical request suspends Python and supplies fresh images to YOU under the host suspension rules (failure, rejection, decision needed, batch limit or checkpoint()).\nAfter reviewing them, emit only resume('current continuation id') to continue the saved stack,\nor write replacement Python to cancel the old stack in the same persistent namespace.\nFunctions, variables and loop locals persist. last_action_result is the physical receipt.\nYou choose contacts, approaches, lifts, holds, transports, release and recovery.\nNo target detection, carrying declaration or grasp macro is required for physical commands.\nThe driver checks bounded robot IK/self collision and monitors force and tracking.\nIt follows your straight Cartesian/shortest-rotation path without detours or pose corrections.\nAbsolute pose tolerances default to 0.003 m / 0.03 rad; set position_tolerance_m and\norientation_tolerance_rad explicitly for tighter corrections. Unattainable precision fails visibly.\nScene contact is part of your plan; use current images and the actual controller feedback.\ncheck_motion is OPTIONAL pure geometry for your specified goal and explicit public target\nhypotheses. Its partial scene/proxy assessment is not a physical permission, collision-free\ncertificate, attachment verdict or task decision. A failed query does not disable move_to.\nUse perception and transform_pose when useful. No candidate/refine/grasp/place macros exist.\nInspect for missed grasp, slip and loss of identity after each action. Neither aperture,\ntool success nor a completed hold proves object retention. Explain the expected visual effect\nbefore each physical continuation and compare it with the newest images."

USAGE += "\nA perception label is a hypothesis: check the task-specified entity against initial/current images and visible distinguishing features. centroid_m/extent_m are visible-surface statistics, not object centers; call measure_object_geometry(target_detection_id, destination_detection_id) for the symmetry-completed center, top/bottom height, radius or extents, destination center and object_to_destination_xy_offset_m with uncertainties. Check a query's success before using its value. You may choose coordinates directly from public evidence; the tool does not choose contact depth.\nBefore a pose request, estimate max(distance/max_speed, angle/max_angular_speed); this is nominal command time, not a completion guarantee. Allow time for tracking and settling within the declared action limits. Receipts report minimum_command_duration_s and effective_duration_limit_s. Inspect the recorded commanded and measured trajectory to separate an unissued endpoint from tracking error or a contact stop.\nAfter unexpected contact or loss of retention, use the images and measured motion to revise the relevant pose, orientation, timing or gripping strategy in your code. A repeated command or a changed number alone is not recovery. You choose when additional views or a timed hold can resolve uncertainty; no fixed sequence or grasp depth is prescribed. Finish only when public evidence supports every clause of the original task, including the correct entity and required final relation."

USAGE += (
    "\nKeep the ORIGINAL task wording available in your Python workspace and revisable notes. "
    "Check every entity/instance qualifier, destination and conjunction against visible evidence; "
    "do not replace the task by a description of your recent action. Before claiming completion, "
    "explain the observed final relation and any unresolved competing interpretation. Tool labels "
    "remain hypotheses, especially for similar containers and destination compartments."
    "\nrecover(max_distance_m=..., max_duration_s=...) reverses only the immediately preceding "
    "measured pose trace, at bounded speed, with current self-collision and non-worsening joint-limit "
    "checks. It may unload existing wrist force but stops if force rises or fails to unload. "
    "An intervening physical action invalidates the trace. It cannot undo an object slip or "
    "guarantee task recovery. When a guarded move fails, inspect its receipt and images before "
    "choosing recovery; do not blindly retry a blocked ordinary move. After recovery, re-localize "
    "the object and revise the actual failed subgoal in your code. If no safe executable "
    "correction remains, you may end incomplete with the unresolved cause; do not claim completion."
)

USAGE += "\nUse preview_motion(action='move_to' or 'recover', arguments={...}) before deciding duration. " \
    "It consumes no physical request. Full move_to and recover refuse impossible command budgets without extending them. " \
    "recover uses 0.01 m/s and 0.2 rad/s. Default mode='full'; mode='partial' explicitly permits a bounded exit. " \
    "To continue the same exit, pass its continuation_id; only the remaining outward path is allowed. " \
    "Read-only observation preserves it; other executed physical actions invalidate it. " \
    "Inspect endpoint_reached, force_decreased, force_still_over_limit and remaining_path separately. " \
    "Exit success does not prove object retention or restoration of your failed goal."

USAGE += "\nrelease(max_duration_s=.5, max_open_fraction_increase=.25) explicitly allows bounded opening under existing load, holding the measured EEF pose. It stops on force rise, lack of unloading, pose drift or robot guards. It may drop an object; inspect images and re-localize before retrying the original goal. It invalidates previous exit traces and never authorizes arbitrary arm motion."


USAGE += "\ntracking_feedback reports requested, commanded and measured displacement, remaining pose error and blocked directions in robot_base coordinates. Directional stalling includes persistent command/reference accumulation despite small measured motion. motion_stalled preserves the stop category; blocked_directions gives the measured basis. A cause such as wrong target, obstacle identity or held-object loss is not inferred. The controller and these receipts are shared across reflection conditions."

USAGE += (
    "\nBounded pose motion paces its reference when it gets ahead of the measured EEF. "
    "command_lead_limit_m/rad and command_pacing_steps are reported separately from endpoint tolerances. "
    "Force is monitored at native simulator substeps, with the original controller gains and physics. "
    "substep_force_monitor records the trigger and internal peak; final force can already have fallen. "
    "Contact detection uses a per-episode calibrated force residual (guarded mode stops at first contact; manipulation mode tolerates load along the motion); a 60 N emergency limit is the only absolute stop. Use public scene geometry "
    "to plan clearance and choose an approach speed before contact; a force stop is not a path-clearance certificate."
)

USAGE += (
    "\nAfter an empty close or ineffective push, compare_object_observations(before_detection_id, after_detection_id) "
    "can measure target motion in robot_base while accounting for camera motion. Keep the old detection handle "
    "in your workspace, then observe and select the same target again; do not compare changing surface centroids as if they were object centers. "
    "inspect_contact_geometry(target_detection_id, position, quaternion) checks visible overlap and margins "
    "for YOUR exact pose and publishes projected finger-pad corners, actual pad proximity and partial visible normals. "
    "Optionally give push_direction in robot_base to inspect normals opposing that motion. Never rank empty-jaw-volume "
    "point counts as a grasp-success score: a gap overlap is not pad contact. Choose a short contact test and bind its measured "
    "object effect before treating the geometric hypothesis as effective. "
    "Before releasing, inspect_release_geometry with current target/destination detections, your pose and selected surface_height_m "
    "can expose an off-edge footprint under the explicit rigid-retention assumption. Check actual retention and destination identity in images. "
    "Geometry queries never execute a robot action. A positive query is not a grasp, placement, or task-success verdict. "
    "Read result.success before result.value. segment returns value.detections; select_detection returns value.centroid_m; "
    "point/mark_point return value.position_m, not centroid_m. These are visible surface measurements, not volume centers. "
    "Read-only diagnostics issue measurement_reference.measurement_id. Cite it in control.measurement_evidence_ids "
    "when requesting completion review; the host resolves source dependencies. Reusing a new ID or changing a hypothetical pose "
    "does not provide new observed evidence. compare_object_observations resolves action_ids from issued receipts "
    "for the full before/after interval when omitted; supplied action_ids must match. The effect appears alongside those attempts. "
    "Use the explicit bounded release for predicted separable self-contact; ordinary open/hold/move still require collision-free robot geometry."
)

USAGE += (
    "\nGeometry fields bind to the requested EEF position/quaternion and selected observation. "
    "Use closing_axis_eef with the proposed EEF quaternion, or inspect_contact_geometry.closing_axis_robot_base for that exact candidate. "
    "candidate_evidence separates visible closing-volume overlap, nearest gap and open-pad proximity; opposing contact and retention remain unverified. "
    "An unsigned distance from one OPEN pad (for example about1cm) does not locate material between closing pads and is not a closure objective. "
    "Zero visible points means no visible overlap supporting that hypothesis; occluded surfaces remain unknown. "
    "If no candidate has support for your stated contact hypothesis, do not silently choose a default or label a skipped test as passed. "
    "Revise the candidate, obtain relevant evidence, or explicitly choose an informative bounded test under uncertainty. "
    "On failed/ambiguous select_detection or geometry queries, handle the error and keep that proposed measurement unassessed. "
    "If you instead act from direct public images, state that separate basis and uncertainty; failed remeasurement supplies no new geometry. "
    "A later motion success cannot validate a previous unsupported identity/contact assumption. "
    "\npreview_motion.command_time_feasible concerns only ideal setpoint travel. "
    "recent_controller_timing contains same-episode measured tracking/settling outcomes; incomplete traces are lower bounds, not completion estimates. "
    "Choose duration with this evidence and room for settling within max_duration_s. "
    "motion_progress distinguishes path_not_fully_commanded, endpoint_tracking, endpoint_settling and endpoint_settled. "
    "Reaching tolerance without settling is still incomplete. Inspect the phase and current images before choosing a correction or hold."
)

_MOTION_VALIDATORS: dict = {}   # CAPTAIN_FAST_LOOP=1: action -> (arguments schema, validator)


def direct_manifest(original):
    vector={"type":"array","items":{"type":"number"},"minItems":3,"maxItems":3}
    quat={"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4}
    handle={"type":"string","pattern":"^artifact://v1/"}
    handles={"type":"array","items":handle,"minItems":1,"maxItems":8,"uniqueItems":True}
    def spec(name,desc,props,required=(),effect=SideEffect.COMPUTE,semantic=None,extra=None):
        return CapabilitySpec(name,desc,{"type":"object","properties":props,"required":list(required),
            "additionalProperties":False,**(extra or {})},
            {"type":"object","additionalProperties":True,"properties":{
                "trajectory":{"x-decision-feedback":"on_demand"},"control_trace":{"x-decision-feedback":"on_demand"},
                "effect_evidence":{"x-decision-feedback":"on_demand"},
                "candidate_evidence":{"type":"object","x-attempt-feedback":"priority"},
                "motion_progress":{"type":"object","x-attempt-feedback":"priority"},
                "contact_tests":{"type":"object","properties":{"visible_local_normals":{"x-decision-feedback":"on_demand"}}},
                "pad_corners_robot_base":{"x-decision-feedback":"on_demand"},
                "pad_corner_pixels_normalized":{"x-decision-feedback":"on_demand"}}},
            {"position":"meter","rotation":"radian","time":"second"},
            {"pose":"robot_base; calibrated end_effector; quaternion wxyz"},180.,effect,
            argument_order=tuple(props),control_semantic=semantic)
    move=spec("move_to",
        "Execute your absolute EEF pose OR one gripper command. Sampled Cartesian IK and robot "
        "self collision are checked; live force/tracking limits and per-step self collision remain. "
        "Scene contact is model-planned; optional check_motion reports partial public geometry. "
        "No segmentation, carrying/empty declaration or grasp workflow is required. No automatic "
        "detour, pose correction or regrasp. Every request suspends for new images. "
        "Absolute pose residual tolerances default to 0.003 m and 0.03 rad; you may explicitly tighten them. "
        "Small corrections need explicit tolerances if finer accuracy matters; no relative-motion accuracy is inferred. "
        "Execution success does not prove holding or task completion. "
        "contact_mode='guarded' (default) stops at the first detected contact using a per-episode calibrated force "
        "residual; use it for free-space and approach moves. contact_mode='manipulation' keeps commanding while the EEF "
        "still progresses under load along the commanded direction (pull, push, turn) and stops on stall, on lateral "
        "contact, or when the load exceeds load_budget_n (default 25 N, at most 60 N); you choose the mode and the budget, "
        "not detection thresholds.",
        {"position":vector,"quaternion":quat,"gripper":{"enum":["open","close"]},
         "max_speed_m_s":{"type":"number","exclusiveMinimum":0,"maximum":.1,"default":.05},
         "max_angular_speed_rad_s":{"type":"number","exclusiveMinimum":0,"maximum":1.,"default":.5},
         "max_duration_s":{"type":"number","exclusiveMinimum":0,"maximum":30.,"default":20.},
         "position_tolerance_m":{"type":"number","exclusiveMinimum":0,"maximum":.003,"default":.003},
         "orientation_tolerance_rad":{"type":"number","exclusiveMinimum":0,"maximum":.03,"default":.03},
         "contact_mode":{"enum":["guarded","manipulation"],"default":"guarded"},
         "load_budget_n":{"type":"number","exclusiveMinimum":0,"maximum":60.}},
        effect=SideEffect.ROBOT_MOTION,semantic=ControlSemantic.CARTESIAN_MOTION,
        extra={"x-execution-goal-fields": ["position", "quaternion", "gripper",
                "position_tolerance_m", "orientation_tolerance_rad"],
            "oneOf":[{"required":["position","quaternion"],"not":{"required":["gripper"]}},
            {"required":["gripper"],"not":{"anyOf":[{"required":[name]} for name in
                ("position","quaternion","max_speed_m_s","max_angular_speed_rad_s","max_duration_s",
                 "position_tolerance_m","orientation_tolerance_rad","contact_mode","load_budget_n")]}}]})
    check=spec("check_motion",
        "OPTIONAL computation only: assess your unchanged requested Cartesian goal against robot IK, "
        "self collision and current measured surfaces. Explicit target/carrying detection handles may "
        "describe one target across current views; you select them and own that identity/attachment "
        "hypothesis. Geometry uses the same noise filter as public perception. "
        "separate_from_contact tests bounded departure from existing clearance/proxy uncertainty, "
        "not an automatic retreat. Returns query validity and scoped geometry assessment; it never "
        "moves, selects a grasp or controls whether move_to may run. Unseen geometry and retention remain unknown.",
        {"position":vector,"quaternion":quat,"target_detection_id":handle,"target_detection_ids":handles,
         "carrying_detection_id":handle,"carrying_detection_ids":handles,"gripper_empty":{"type":"boolean"},
         "separate_from_contact":{"type":"boolean","default":False}},("position","quaternion"),
        extra={"allOf":[{"not":{"required":[single,plural]}} for single,plural in
            (("target_detection_id","target_detection_ids"),("carrying_detection_id","carrying_detection_ids"))]})
    hold=spec("hold","Maintain the measured EEF pose and last gripper command while advancing dynamics, "
        "then suspend for visual review. Completion is not an object-stability verdict.",
        {"duration_s":{"type":"number","exclusiveMinimum":0,"maximum":30.}},("duration_s",),
        SideEffect.ROBOT_MOTION,ControlSemantic.TIME_ADVANCE)
    recover=spec("recover", "Bounded reversal of the latest measured pose trace. No arbitrary new "
        "goal, automatic grasp or scene inference. A physical request with fresh images even on rejection. "
        "Preserves self-collision guards; permits existing joint-limit excess only while it does not worsen. "
        "Existing wrist load must unload; recovery success requires the pose endpoint and normal limits. "
        "A stale trace is rejected; object restoration remains your responsibility.",
        {"max_distance_m":{"type":"number","exclusiveMinimum":0,"maximum":.25,"default":.05},
         "max_duration_s":{"type":"number","exclusiveMinimum":0,"maximum":30.,"default":10.},
         "mode":{"enum":["full","partial"],"default":"full"},
         "continuation_id":{"type":"string","minLength":1}},
        effect=SideEffect.ROBOT_MOTION,semantic=ControlSemantic.CARTESIAN_MOTION)
    release=spec("release", "Explicit bounded opening to unload existing wrist force, with fixed measured EEF target. "
        "No arm direction or closing is allowed. Stops on sampled force rise >2 N from best, no 1 N unloading "
        "over .5 s while overloaded, pose drift >.003 m/.03 rad, self collision or joint limits. "
        "Opening increase is a sampled stop threshold; one-cycle physical overshoot is possible. "
        "Opening can lose an object. Success means measured unloading, never object or task recovery.",
        {"max_duration_s":{"type":"number","exclusiveMinimum":0,"maximum":2.,"default":.5},
         "max_open_fraction_increase":{"type":"number","exclusiveMinimum":0,"maximum":.25,"default":.25}},
        effect=SideEffect.ROBOT_MOTION,semantic=ControlSemantic.CARTESIAN_MOTION)
    measure=spec("measure_image_points", "Read public metric depth and calibrated robot-base points at your "
        "chosen image pixels. No object labels or hidden state. Cite source image hash, camera and pixels; "
        "invalid depth remains unknown. Does not advance simulation.",
        {"camera":{"type":"string"}, "pixels":{"type":"array", "minItems":1,"maxItems":64,
            "items":{"type":"array","minItems":2,"maxItems":2,
                "items":{"type":"number","minimum":0,"maximum":1}}}}, ("camera","pixels"))
    preview=spec("preview_motion", "Read-only timing for move_to and recover (jaw requests report fixed control-cycle time). Uses current measured "
        "state and executor path/rates; no simulation or physical budget. Reports the theoretical command lower "
        "bound and request-budget path position, excluding tracking/contact/settling and collision certification.",
        {"action":{"enum":["move_to","recover"]},"arguments":{"type":"object"}}, ("action","arguments"))
    transform=spec("transform_pose","Pure geometry: apply explicit translation and rotation vector about "
        "a pivot (default EEF position), in robot_base or input EEF local axes. Does not choose or move.",
        {"position":vector,"quaternion":quat,"translation":{**vector,"default":[0.,0.,0.]},
         "rotation_vector":{**vector,"default":[0.,0.,0.]},
         "frame":{"enum":["robot_base","end_effector"],"default":"robot_base"},"pivot":vector},
        ("position","quaternion"))
    geometry=spec("robot_geometry","Read calibrated EEF/pad reference, pad envelope and gripper limits; "
        "these describe robot hardware, not an object's state.",{})
    contact_geometry=spec("inspect_contact_geometry",
        "Read-only: project your EXACT EEF pose and open finger-pad closing volume against an explicitly selected current RGB-D target. Reports gap, margins, axis and image coordinates; no candidate selection, motion or grasp verdict.",
        {"target_detection_id":handle,"position":vector,"quaternion":quat,"push_direction":vector},
        ("target_detection_id","position","quaternion"))
    object_comparison=spec("compare_object_observations",
        "Read-only: compare two explicitly selected detections across time in the SAME calibrated camera. Uses RGB-D feature correspondence to separate robot motion from target motion. Lost correspondence stays unknown; not an attachment oracle.",
        {"before_detection_id":handle,"after_detection_id":handle,
            "action_ids":{"type":"array","items":{"type":"string"},"minItems":1,"maxItems":60,"uniqueItems":True}},
        ("before_detection_id","after_detection_id"))
    object_geometry=spec("measure_object_geometry",
        "Read-only measurement of a selected object's geometry from the current RGB-D views: symmetry-completed "
        "center (circle axis or rectangle center, not the visible-surface median), radius or XY extents with yaw, "
        "top and bottom heights from the fitted support plane, per-axis center uncertainty and a quality flag. "
        "With destination_detection_id also returns the destination center, rim radius, support surface height and "
        "object_to_destination_xy_offset_m (object minus destination). With before_detection_id (the same part, e.g. a "
        "handle or door front, detected in an EARLIER observation from the same camera) also returns part_displacement_m, "
        "part_rotation_rad and part_motion_significant: the state evidence for open/close tasks. Use these numbers for "
        "closing height/width and for centering a placement; they are measurements of visible surfaces, not identity or task verdicts.",
        {"target_detection_id":handle,"destination_detection_id":handle,"before_detection_id":handle,
         "shape":{"enum":["auto","cylinder","box","disc"],"default":"auto"},
         "cameras":{"enum":["all","primary"],"default":"all"}},
        ("target_detection_id",))
    release_geometry=spec("inspect_release_geometry",
        "Read-only: project the current selected object's visible footprint at your proposed EEF release pose onto your selected horizontal destination surface height. Reports signed boundary clearance. Assumes current rigid retention; positive geometry never proves support or authorizes release.",
        {"target_detection_id":handle,"destination_detection_id":handle,"position":vector,
            "quaternion":quat,"surface_height_m":{"type":"number"}},
        ("target_detection_id","destination_detection_id","position","quaternion","surface_height_m"))
    kept=tuple(s for s in original.capabilities if s.name in PERCEPTION and s.name!="gripper_state")
    if not any(s.name=="get_robot_state" for s in kept):
        kept+=(spec("get_robot_state","Read measured EEF position/quaternion, joint angles and aperture.",{}),)
    kept=tuple(replace(s,description=s.description+" granularity='part' selects the smallest SAM3 mask containing the pixel (a drawer front, handle or knob "
        "rather than the whole cabinet); 'object' (default) the best-scoring one; 'whole' the largest. The result also carries detection_id: a "
        "selected detection of that mask, usable directly in measure_object_geometry / compare_object_observations.",
        arguments_schema={**s.arguments_schema,"properties":{**s.arguments_schema.get("properties",{}),
            "granularity":{"type":"string","enum":["object","part","whole"],"default":"object"}}},
        argument_order=(*s.argument_order,"granularity")) if s.name=="mark_point" else s for s in kept)
    kept=tuple(replace(s,result_schema={**s.result_schema,"properties":{
        **s.result_schema.get("properties",{}),
        
        "measurement_reference":{"type":"object"},"measurement_registration":{"type":"object"},
        "detection_id":{"type":["string","null"]},"detection_selected":{"type":"boolean"}}})
        if s.name in {"point","mark_point","select_detection"} else s for s in kept)
    kept+=(spec("gripper_state","Read measured aperture, commanded opening and wrist force. "
        "These do not identify contact or prove retention.",{}),)
    return CapabilityManifest("captain-capabilities/r30-direct-v8",original.provider+"-r30-direct-v8",
        (*kept,move,hold,recover,release,preview,measure,transform,geometry,check,contact_geometry,object_comparison,release_geometry,object_geometry),USAGE)


def transform_pose(position, quaternion, translation=(0.,0.,0.), rotation_vector=(0.,0.,0.),
                   frame="robot_base", pivot=None):
    p = pose_matrix(position, quaternion)
    t, r = np.asarray(translation,dtype=float), np.asarray(rotation_vector,dtype=float)
    if t.shape != (3,) or r.shape != (3,) or not np.isfinite([t,r]).all():
        raise ValueError("finite translation and rotation three-vectors required")
    if frame == "end_effector":
        t, r = p[:3,:3] @ t, p[:3,:3] @ r
    elif frame != "robot_base":
        raise ValueError("unknown frame")
    center = p[:3,3] if pivot is None else np.asarray(pivot,dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("finite robot_base pivot required")
    rot = Rotation.from_rotvec(r).as_matrix()
    p[:3,3] = center + rot @ (p[:3,3]-center) + t
    p[:3,:3] = rot @ p[:3,:3]
    return CapabilityResult.ok({"position":p[:3,3].tolist(),"quaternion":quaternion_from_pose(p)})

quaternion_from_pose = quaternion

class DirectProgramming:
    def __init__(self, tools):
        self.tools = tools
        self.robot = tools.robot
        self.attempt_id = None
        self.target_id = None
        self._attempt_active = False
        self._unresolved_failure_id = None
        self._measurement_receipts = {}
        self._issued_action_receipts = {}
        self._recent_motion_timing = []

    def geometry(self):
        from captain_harness.adapters.robotics.public_gripper_geometry import envelope_in_eef
        raw = self.robot.gripper_contact_envelope()
        envelope = envelope_in_eef(raw)
        center = envelope["center_local_m"] if envelope else None
        axis = envelope["closing_axis_eef"] if envelope else None
        state = self.robot.get_robot_state()
        pose = state.value if state.success else None
        current_axis = (Rotation.from_quat(np.roll(pose.end_effector_quaternion_wxyz,-1)).apply(axis).tolist()
                        if pose is not None and axis is not None else None)
        return {"geometry_contract":"eef-pad-frames-v2",
                "geometry_status":"available" if envelope is not None else "frame_contract_unavailable",
                "eef_frame":"end_effector", "coordinate_frame":"robot_base",
                "eef_reference":"gripper0_eef site, near finger-pad midpoint; not wrist or palm",
                "eef_to_pad_center_local_m":center,
                "finger_pad_envelope":envelope,
                "closing_axis_eef":axis,"closing_axis_robot_base":current_axis,
                "closing_axis_robot_base_pose":asdict(pose) if pose is not None else None,
                "axes_transform":{"source_axes":raw.get("axes_frame") if raw else None,
                    "origin_frame":"end_effector",
                    "rotation_end_effector_from_source_axes":envelope["rotation_end_effector_from_source_axes"] if envelope else None,
                    "rule":"v_eef = rotation_end_effector_from_source_axes @ v_source; both vectors originate at the EEF"},
                "reference_rule":"move_to uses measured EEF pose. For YOUR proposed quaternion, rotate closing_axis_eef into robot_base; do not assume EEF Y is jaw closure. closing_axis_robot_base refers only to closing_axis_robot_base_pose. Rotate the declared EEF pad offset when choosing a pad-center target; do not add an assumed hand length. Unknown frame metadata remains unavailable.",
                "gripper_profile":self.tools.gripper_profile.public(),
                "gripper_commands":["open","close"],"variable_grip_force_supported":False}

    def invoke(self, name, args):
        if name == "get_robot_state":
            result = self.robot.get_robot_state()
            return replace(result, value=asdict(result.value) if result.value is not None else None)
        if name == "gripper_state":
            state = self.robot.get_robot_state()
            if not state.success or state.value is None:
                return state
            return CapabilityResult.ok({
                "measured_open_fraction": state.value.gripper_open_fraction,
                "commanded_open_fraction": self.robot._gripper_command_fraction,
                "wrist_force_norm_n": self.robot._eef_force_norm(),
                "object_retention": "not_verified", "target_identity": "unknown"})
        if name in {"inspect_contact_geometry","compare_object_observations","inspect_release_geometry","measure_object_geometry"}:
            from . import effect_geometry, object_geometry
            from .backend import ToolOperationError
            try:
                function={"inspect_contact_geometry":effect_geometry.contact,
                    "compare_object_observations":effect_geometry.compare,
                    "inspect_release_geometry":effect_geometry.release,
                    "measure_object_geometry":object_geometry.measure}[name]
                value=function(self.tools,**args)
                from .measurement_records import register
                return CapabilityResult.ok(register(self,name,args,value))
            except ToolOperationError as exc:
                return CapabilityResult.failed(exc.category,str(exc),executed=False,
                    value={"query_status":"unavailable","candidate_arguments":dict(args),
                        "contact_assessment":"not_assessed","physical_action_executed":False})
            except (ValueError,TypeError,KeyError) as exc:
                return CapabilityResult.failed(FailureCategory.PERCEPTION_UNCERTAIN,
                    "public_geometry_unavailable: "+str(exc),executed=False,
                    value={"query_status":"unavailable","candidate_arguments":dict(args),
                        "contact_assessment":"not_assessed","physical_action_executed":False})
        if name == "transform_pose":
            return transform_pose(**args)
        if name == "robot_geometry":
            return CapabilityResult.ok(self.geometry())
        if name == "measure_image_points":
            try:
                return self._measure_image_points(**args)
            except (ValueError, TypeError, OSError, IndexError) as exc:
                return CapabilityResult.failed(FailureCategory.PERCEPTION_UNCERTAIN, str(exc))
        if name == "preview_motion":
            try:
                action, arguments = args['action'], args['arguments']
                self._validate_motion_arguments(action, arguments)
                if action == 'recover':
                    _, value, _ = self.robot.preview_retreat(**arguments)
                elif 'gripper' in arguments:
                    duration = self.robot._gripper_steps/self.robot._control_frequency
                    value = {'command_kind':'gripper', 'minimum_command_duration_s':duration,
                        'effective_duration_limit_s':duration, 'requested_duration_limit_s':duration,
                        'max_speed_m_s':None, 'max_angular_speed_rad_s':None,
                        'command_time_feasible':True, 'physical_action_executed':False,
                        'timing_scope':'fixed jaw-command control cycles; jaw speed, achieved aperture, contact and object retention are not guaranteed'}
                else:
                    _, value = self._prepare_move(**arguments)
                return CapabilityResult.ok(value)
            except (ValueError, TypeError, KeyError) as exc:
                return CapabilityResult.failed(FailureCategory.INVALID_ARGUMENT, str(exc))
        if name == "check_motion":
            try:
                assessed=self._check_motion(**args)
            except (ValueError,DeadlineExceeded) as exc:
                return CapabilityResult.failed(FailureCategory.PERCEPTION_UNCERTAIN,
                    f"geometry_query_unavailable: {type(exc).__name__}: {exc}",executed=False)
            return CapabilityResult.ok({"query_completed":True,"geometry_check_passed":assessed.success,
                "assessment_message":assessed.message,"assessment_category":assessed.failure_category.value,
                "assessment":assessed.value,"physical_action_executed":False,"object_retention":"not_verified",
                "scope":"optional partial geometry assessment; does not gate or authorize move_to"})
        if name not in {"move_to","hold","recover","release"}:
            result=self.tools.invoke(name,args)
            if name in {"point","mark_point","select_detection"}:
                from .measurement_records import register_observation
                result=register_observation(self,name,result)
                if not result.success:
                    result=replace(result,value={**(dict(result.value) if isinstance(result.value,dict) else {}),
                        "query_status":"unavailable","selection_arguments":dict(args),
                        "target_identity":"unresolved","contact_assessment":"not_assessed",
                        "decision_guidance":"This selection/measurement failed. Resolve identity with current public evidence or state a separate direct-image hypothesis; do not count this as a passed geometry check."})
            return result
        start = self.robot.simulation_steps
        action_id = uuid.uuid4().hex
        previous_attempt = (self.attempt_id, self.target_id, self._attempt_active)
        before = self._read_state()
        force_reader = getattr(self.robot, '_eef_force_norm', None)
        initial_force = force_reader() if callable(force_reader) else None
        try:
            if not before.success or before.value is None:
                result = before
            elif name == "recover":
                result = self._recover(**args)
            elif name == "release":
                result = self._release(**args)
            elif name == "hold":
                result, _ = self._guarded_execution(lambda: self.robot.hold_position(**args),
                    model=self.tools._motion_model(), stationary=True)
            elif "gripper" in args:
                target_id = None
                if args["gripper"] == "close":
                    self.attempt_id, self.target_id = uuid.uuid4().hex, target_id
                    self._attempt_active = True
                result, _ = self._guarded_execution(
                    lambda: self.robot.close_gripper() if args["gripper"] == "close" else self.robot.open_gripper(),
                    model=self.tools._motion_model(), stationary=True, gripper=True)
            else:
                result = self._move(**args)
        except (ValueError, DeadlineExceeded) as exc:
            advanced = self.robot.simulation_steps > start
            result = CapabilityResult.failed(
                FailureCategory.TIMEOUT if isinstance(exc, DeadlineExceeded) else
                (FailureCategory.BACKEND_ERROR if advanced else FailureCategory.NOT_EXECUTED),
                f"{type(exc).__name__}: {exc}", executed=advanced)
        except Exception as exc:
            result = CapabilityResult.failed(FailureCategory.BACKEND_ERROR,
                f"{type(exc).__name__}: {exc}", executed=self.robot.simulation_steps > start)
        if args.get("gripper") == "close" and not result.executed and self.robot.simulation_steps == start:
            self.attempt_id, self.target_id, self._attempt_active = previous_attempt
        stop_targets = None
        if (not result.success or name == "release") and self.robot.simulation_steps > start:
            cancel = getattr(self.robot, 'cancel_direct_targets', None)
            if callable(cancel):
                stop_targets = cancel(cancel_gripper=(name in {"hold", "release"} or "gripper" in args))
        if getattr(self.robot, '_physical_execution_unknown', False):
            result = replace(result, success=False, failure_category=FailureCategory.BACKEND_ERROR,
                diagnostics={**dict(result.diagnostics or {}), 'execution_state':'unknown', 'episode_must_stop':True})
        after = self._read_state()
        value = dict(result.value) if isinstance(result.value,dict) else {}
        final_force = force_reader() if callable(force_reader) else None
        value.setdefault('stopped_reason', result.message)
        value.setdefault('executed_distance_m', 0. if self.robot.simulation_steps == start else None)
        value.setdefault('endpoint_reached', False if not result.executed else None)
        from captain_harness.adapters.robotics.force_receipt import finalize_force_receipt
        finalize_force_receipt(value, initial_force_n=initial_force, final_force_n=final_force,
            step=self.robot.simulation_steps,
            sample_time_s=self.robot.simulation_steps/self.robot._control_frequency)
        value['original_goal_restored'] = 'pending_visual_confirmation'
        if before.value is not None and after.value is not None and 'tracking_feedback' not in value:
            from captain_harness.adapters.robotics.tracking_feedback import tracking_feedback
            def pose(state):
                out = np.eye(4)
                out[:3,3] = state.end_effector_position_m
                out[:3,:3] = Rotation.from_quat(np.roll(state.end_effector_quaternion_wxyz,-1)).as_matrix()
                return out
            start_pose, final_pose = pose(before.value), pose(after.value)
            requested_pose = start_pose.copy()
            if isinstance(args.get('position'), (list, tuple)) and len(args['position']) == 3 and all(isinstance(x,(int,float)) and np.isfinite(x) for x in args['position']):
                requested_pose[:3,3] = args['position']
            if isinstance(args.get('quaternion'), (list, tuple)) and len(args['quaternion']) == 4 and all(isinstance(x,(int,float)) and np.isfinite(x) for x in args['quaternion']) and np.linalg.norm(args['quaternion']) > 0:
                requested_pose[:3,:3] = Rotation.from_quat(np.roll(args['quaternion'],-1)).as_matrix()
            value['tracking_feedback'] = tracking_feedback(start_pose, final_pose, requested_pose,
                start_pose, stop_reason=value['stopped_reason'])
            if name == 'recover' or (name == 'move_to' and 'gripper' not in args and 'position' not in args):
                for field in ('requested_displacement_m','requested_rotation_vector_rad',
                              'remaining_displacement_m','remaining_rotation_vector_rad'):
                    value['tracking_feedback'][field] = None
                value['tracking_feedback']['requested_path_status'] = 'unavailable; see refusal and preview details'
            value['tracking_feedback']['command_reference_scope'] = 'stationary reference or unexecuted request; see full trajectory'
        # Drivers may already supply pose tracking; aperture measurements still
        # belong in the same compact feedback for motion and actuator requests.
        if before.value is not None and after.value is not None:
            grip_before = before.value.gripper_open_fraction
            grip_after = after.value.gripper_open_fraction
            value['tracking_feedback']['gripper_before'] = grip_before
            value['tracking_feedback']['gripper_after'] = grip_after
            value['tracking_feedback']['requested_gripper'] = args.get('gripper')
            value['tracking_feedback']['measured_gripper_change'] = (
                float(grip_after - grip_before)
                if all(isinstance(x, (int, float)) and np.isfinite(x) for x in (grip_before, grip_after))
                else None)
        tf = value.get("tracking_feedback", {})
        def magnitude(field):
            raw = tf.get(field)
            return float(np.linalg.norm(raw)) if (isinstance(raw,(list,tuple)) and len(raw)==3
                and all(type(x) in (int,float) and np.isfinite(x) for x in raw)) else 0.
        changed = ((magnitude("requested_displacement_m") > .0001 and magnitude("measured_displacement_m") > .0001)
            or (magnitude("requested_rotation_vector_rad") > .001 and magnitude("measured_rotation_vector_rad") > .001)
            or ((args.get("gripper") is not None or name == "release")
                and abs(tf.get("measured_gripper_change") or 0.) > .01))
        value["public_state_change"] = {
            "commanded_change_observed": bool(result.executed and changed),
            "basis": "commanded robot change measured; excludes stationary drift and rejected requests",
            "task_effect": "not_inferred"}
        if name=="move_to" and "gripper" not in args:
            from captain_harness.adapters.robotics.motion_progress import motion_progress
            value["motion_progress"]=motion_progress(value)
            if result.executed:
                self._recent_motion_timing.append({"action_id":action_id,
                    "requested_max_speed_m_s":args.get("max_speed_m_s",.05),
                    "requested_max_angular_speed_rad_s":args.get("max_angular_speed_rad_s",.5),
                    "request_success":result.success,**value["motion_progress"]})
                self._recent_motion_timing=self._recent_motion_timing[-6:]
        value["driver_success"] = result.success
        if stop_targets is not None:
            value["post_stop_targets"] = stop_targets
        if not after.success or after.value is None:
            value["robot_after_error"] = after.message or "robot state unavailable"
            advanced = result.executed or self.robot.simulation_steps > start
            result = replace(CapabilityResult.failed(FailureCategory.BACKEND_ERROR,
                "post_action_measurement_failed: " + value["robot_after_error"], executed=advanced),
                diagnostics={**dict(result.diagnostics or {}),
                    'execution_state':'unknown' if advanced else 'confirmed', 'episode_must_stop':bool(advanced)})
        value.update(action_id=action_id, requested=dict(args), robot_before=asdict(before.value) if before.value else None,
            robot_after=asdict(after.value) if after.value else None,
            start_step=start,end_step=self.robot.simulation_steps,
            attempt_id=self.attempt_id, target_detection_id=self.target_id,
            target_identity="policy_association_only" if self.target_id else "unknown",
            object_retention="not_verified", source="public_robot_measurements",
            gripper_command_fraction=self.robot._gripper_command_fraction)
        value['related_failure_action_id'] = self._unresolved_failure_id
        if not result.success:
            self._unresolved_failure_id = action_id
        if result.executed or self.robot.simulation_steps > start:
            if name != 'recover':
                self.robot._retreat_continuation = None
                if name in {'hold', 'release'} or 'gripper' in args:
                    self.robot._last_guarded_motion = None
            saved = getattr(self.robot, '_last_guarded_motion', None)
            if saved is not None:
                saved['action_id'] = action_id
            self.tools._aligned_pregrasp = None
        identity, assessment = None, None
        if self._attempt_active:
            reference = "attempt://" + self.attempt_id
            origin = args.get("gripper") == "close" and result.executed
            identity = {
                "reference": self.target_id,
                "attempt_reference": reference,
                "attempt_role": "origin" if origin else "followup",
                "basis": "explicit_policy_target" if self.target_id else "unbound_closure",
                "semantic_identity_status": "unverified",
                "association_status": "policy_associated" if self.target_id else "unbound"}
            if not origin:
                assessment = {
                    "attempt_reference": reference, "status":"unknown",
                    "evidence":{"scope":("association_ended" if (args.get("gripper") == "open" or name == "release") and result.executed else "action_followup"),
                        "action_id":action_id, "capability":name, "requested":dict(args),
                        "executed":result.executed, "execution_success":result.success,
                        "start_step":start, "end_step":self.robot.simulation_steps,
                        "stop_reason":result.message,
                        "object_retention":"not_verified",
                        "release_executed":(args.get("gripper") == "open" or name == "release") and result.executed}}
        if (args.get("gripper") == "open" or name == "release") and result.executed:
            self._attempt_active = False
        continuation = getattr(self.robot, '_retreat_continuation', None)
        if continuation is not None and stop_targets is not None:
            if stop_targets['status'] == 'cancelled':
                continuation['gripper_command'] = self.robot._gripper_command_fraction
            else:
                self.robot._retreat_continuation = None
                value.update(continuation_id=None, continuation_unavailable_reason='target_cancellation_unconfirmed')
        value["attempt_active_after"] = self._attempt_active
        value["execution_interval"]={"start_step":start,"end_step":self.robot.simulation_steps}
        self._issued_action_receipts[action_id]={"action_id":action_id,"executed":result.executed,
            "start_step":start,"end_step":self.robot.simulation_steps}
        return replace(result,value=value,attempt_identity=identity,attempt_evidence=assessment)

    def _measure_image_points(self, camera, pixels):
        import hashlib, json
        from pathlib import Path
        observed = self.robot.get_observation_data()
        if not observed.success:
            return observed
        source = next((c for c in observed.value.cameras if c.camera_name == camera), None)
        if source is None or source.depth_path is None:
            return CapabilityResult.failed(FailureCategory.PERCEPTION_UNCERTAIN, 'declared_camera_depth_unavailable')
        depth = np.load(source.depth_path, allow_pickle=False).squeeze()
        k = np.asarray(source.intrinsics_3x3).reshape(3,3)
        transform = np.asarray(source.extrinsics_4x4).reshape(4,4)
        points = []
        for x,y in pixels:
            if not (np.isfinite([x,y]).all() and 0<=x<=1 and 0<=y<=1):
                raise ValueError('pixels must be normalized [x,y] pairs')
            u,v = int(round(x*(depth.shape[1]-1))), int(round(y*(depth.shape[0]-1)))
            z = float(depth[v,u])
            point = None
            if np.isfinite(z) and z>0:
                optical = np.linalg.solve(k, [u,v,1.])*z
                point = (transform[:3,:3]@optical+transform[:3,3]).tolist()
            points.append({'pixel_xy':[u,v], 'depth_m':z if point is not None else None,
                'robot_base_point_m':point})
        content = {'camera':camera,'points':points,'intrinsics_3x3':k.tolist(),
            'camera_to_robot_base_4x4':transform.tolist()}
        digest = hashlib.sha256(json.dumps(content,sort_keys=True,allow_nan=False).encode()).hexdigest()
        receipt = {**content, 'evidence_content_sha256':digest,
            'rgb_sha256':hashlib.sha256(Path(source.rgb_path).read_bytes()).hexdigest(),
            'raw_depth_sha256':hashlib.sha256(Path(source.depth_path).read_bytes()).hexdigest(),
            'step':self.robot.simulation_steps, 'sample_time_s':observed.value.time_s,
            'source':'public_rgbd_pixel_measurement', 'depth_status':'attached_public_samples',
            'physical_action_executed':False, 'limitation':'sampled visible surface only; identity, support and occluded geometry unverified'}
        self._measurement_receipts[digest] = receipt
        return CapabilityResult.ok(receipt)

    def _read_state(self):
        try:
            return self.robot.get_robot_state()
        except Exception as exc:
            return CapabilityResult.failed(FailureCategory.BACKEND_ERROR,
                f"{type(exc).__name__}: {exc}")

    def _check_motion(self, position, quaternion, target_detection_id=None,
              carrying_detection_id=None, gripper_empty=False,
              target_detection_ids=None, carrying_detection_ids=None, separate_from_contact=False,
              max_speed_m_s=.05, max_angular_speed_rad_s=.5, max_duration_s=20.):
        from captain_harness.adapters.libero.contact_motion import _pose
        from captain_harness.adapters.robotics.cartesian_sweeps import cartesian_sweep
        target_handles=detection_handles(target_detection_id,target_detection_ids)
        carrying_handles=detection_handles(carrying_detection_id,carrying_detection_ids)
        if carrying_handles and (target_handles or gripper_empty):
            raise ValueError("carrying geometry cannot combine with target contact allowance or empty-jaw declaration")
        start = _pose(self.robot.get_robot_state().value)
        goal = pose_matrix(position,quaternion)
        turn = Rotation.from_matrix(goal[:3,:3] @ start[:3,:3].T).as_rotvec()
        angle = float(np.linalg.norm(turn))
        path = cartesian_sweep(start, displacement_m=goal[:3,3]-start[:3,3],
            **({"axis":turn/angle,"angle_rad":angle,"pivot_m":start[:3,3]} if angle>1e-10 else {}))
        observed = self.tools.observe()
        if not observed.success:
            return observed
        scene_result = self.tools.build_scene(observed.value["observation_handle"])
        if not scene_result.success:
            return scene_result
        scene_handle = scene_result.value["scene_handle"]
        scene = self.tools._scene(scene_handle,fresh=True)
        target = None
        geometry_receipt = None
        if target_handles:
            scene,target,geometry_receipt = selected_geometry(self.tools,scene_handle,target_handles)
        model = self.tools._motion_model()
        carried = None
        if carrying_handles:
            scene, points,geometry_receipt = selected_geometry(self.tools,scene_handle,carrying_handles)
            model.set_state(scene.joints, scene.aperture)
            own, _ = model.points_in_robot(points, margin=.003, include_visual=True)
            selected = points[~own]
            local = (selected - start[:3,3]) @ start[:3,:3]
            model = model.with_observed_carried_geometry(local)
            carried = SimpleNamespace(points_world=selected, receipt={
                **dict(model.carried_geometry_receipt),
                "detection_id":carrying_handles[0] if len(carrying_handles)==1 else None,
                "detection_ids":list(carrying_handles),
                "public_geometry":geometry_receipt,
                "attachment":"model_hypothesis_only",
                "source":"current measured target surfaces",
                "limitation":"rigid attachment, segmentation identity and unseen geometry remain unverified"})
        nominal, policy = self.tools._scene_collision_checker(model,scene,target,carried=carried)
        checked, reserve = self.tools._tracking_reserved_checker(model,scene,nominal,carried=carried)
        policy={**policy,"selected_public_geometry":geometry_receipt,
                "separate_from_contact_requested":bool(separate_from_contact)}
        separation=None
        if separate_from_contact:
            from functools import partial
            from scipy.spatial import cKDTree
            from captain_harness.adapters.robotics.carried_geometry import partition_measured_target
            from captain_harness.adapters.robotics.contact_separation import ContactSeparationChecker
            static_points=scene.points
            target_mask=None
            if carried is not None:
                static_points,_=partition_measured_target(scene.points,carried.points_world)
            elif target is not None:
                target_mask=cKDTree(target).query(scene.points)[0]<=1e-7
            separation=ContactSeparationChecker(model,scene.joints,scene.aperture,
                static_points,scene.margin_m,target_mask=target_mask,
                reserve_padding=(reserve or {}).get("geometry_padding_m",{}))
            planned_nominal=partial(separation.check,stage="planning")
            checked,reserve=self.tools._tracking_reserved_checker(model,scene,planned_nominal,carried=carried)
            nominal=partial(separation.check,stage="execution")
            policy["separation_constraints"]={
                "initial_sample_pairs":separation.initial_sample_pairs,
                "hardware_penetration":"rejected","payload_proxy_overlap_bound_m":scene.margin_m,
                "new_collisions":"rejected","cleared_contact_reentry":"rejected",
                "direction":"explicit policy command; unchanged",
                "scope":"sampled public-surface separation and per-step guard; unseen geometry unknown"}
        checks=[]
        seed=np.asarray(scene.joints)
        initial = separation.check(seed,stage="initial") if separation is not None else nominal(seed)
        if initial["collision"]:
            return CapabilityResult.failed(FailureCategory.NOT_EXECUTED,
                "observed_start_collision",value={"collision":initial,"collision_policy":policy})
        for index,pose in enumerate(path[1:],1):
            check_deadline()
            solved=model.solve_ik(pose,seed,aperture=scene.aperture,
                position_tolerance=.001, orientation_tolerance=.01)
            joints=solved
            if joints is None:
                return CapabilityResult.failed(FailureCategory.IK_FAILURE,
                    "requested_cartesian_path_ik_unavailable",value={"sample_index":index,"ik":model.last_ik})
            seed=np.asarray(joints)
            collision=checked(seed)
            checks.append({"sample_index":index,"collision":collision})
            if collision["collision"]:
                return CapabilityResult.failed(FailureCategory.NOT_EXECUTED,
                    "requested_cartesian_path_obstructed",value={"sample_index":index,
                        "collision":collision,"collision_policy":policy,"tracking_reserve":reserve})
        return CapabilityResult.ok({"scene":scene_result.value,"collision_policy":policy,
            "tracking_reserve":reserve,"path_samples":len(path),
            "payload_assumption":"explicit_public_geometry" if carried else "model_declared_empty" if gripper_empty else "unspecified",
            "scope":"sampled IK and partial measured scene geometry only; no motor/retention certificate"})

    @staticmethod
    def _validate_motion_arguments(action, arguments):
        from jsonschema import validate, ValidationError
        from captain_harness.ports import interface_audit
        if action not in {'move_to', 'recover'}:
            interface_audit.record(validation_source="motion_argument_validator", phase="input", outcome="rejected",
                                   rule_id="enum.preview_action", argument_path=["action"], tool="preview_motion",
                                   detail=f"action {action!r} not in move_to/recover")
            raise ValueError('preview supports move_to and recover only')
        import os
        fast = os.environ.get("CAPTAIN_FAST_LOOP", "") == "1"
        cached = _MOTION_VALIDATORS.get(action) if fast else None
        if cached is None:
            schema = next(s.arguments_schema for s in direct_manifest(SimpleNamespace(
                capabilities=(), provider='validation')).capabilities if s.name == action)
        else:
            schema = cached[0]
        try:
            if fast:
                # = jsonschema.validate(arguments, schema) minus the per-call check_schema (done once, below) and the
                # per-call manifest rebuild: same validator class, same best_match error
                from jsonschema.exceptions import best_match
                from jsonschema.validators import validator_for
                if cached is None:
                    cls = validator_for(schema)
                    cls.check_schema(schema)
                    cached = _MOTION_VALIDATORS[action] = (schema, cls(schema))
                error = best_match(cached[1].iter_errors(arguments))
                if error is not None:
                    raise error
            else:
                validate(arguments, schema)
        except ValidationError as exc:
            interface_audit.record_validation_error(exc, validation_source="motion_argument_validator", tool="preview_motion",
                                                    schema=schema, validated_action=action, path_prefix=["arguments"])
            raise ValueError(exc.message) from exc
        interface_audit.record(validation_source="motion_argument_validator", phase="input", outcome="accepted",
                               tool="preview_motion", validated_action=action, schema=schema)

    def _prepare_move(self, position, quaternion, max_speed_m_s=.05,
              max_angular_speed_rad_s=.5, max_duration_s=20.,
              position_tolerance_m=.003, orientation_tolerance_rad=.03,
              contact_mode='guarded', load_budget_n=None):
        from captain_harness.adapters.robotics.cartesian_path_timing import motion_preview
        self._validate_motion_arguments('move_to', dict(position=position, quaternion=quaternion,
            max_speed_m_s=max_speed_m_s, max_angular_speed_rad_s=max_angular_speed_rad_s,
            max_duration_s=max_duration_s, position_tolerance_m=position_tolerance_m,
            orientation_tolerance_rad=orientation_tolerance_rad))
        from captain_harness.adapters.libero.contact_motion import _pose
        from captain_harness.adapters.robotics.cartesian_sweeps import cartesian_sweep
        state=self.robot.get_robot_state()
        if not state.success or state.value is None:raise ValueError("robot state unavailable")
        start=_pose(state.value)
        goal=pose_matrix(position,quaternion)
        turn=Rotation.from_matrix(goal[:3,:3]@start[:3,:3].T).as_rotvec()
        angle=float(np.linalg.norm(turn))
        path=cartesian_sweep(start,displacement_m=goal[:3,3]-start[:3,3],
            **({"axis":turn/angle,"angle_rad":angle,"pivot_m":start[:3,3]} if angle>1e-10 else {}))
        timing=motion_preview(path, max_speed_m_s, max_angular_speed_rad_s,
            max_duration_s, self.robot._control_frequency, self.robot._cartesian_max_steps)
        timing["recent_controller_timing"]=list(self._recent_motion_timing)
        timing["completion_time_status"]="unknown; ideal command feasibility excludes measured tracking waits and settling"
        return path, timing

    def _move(self, position, quaternion, max_speed_m_s=.05,
              max_angular_speed_rad_s=.5, max_duration_s=20.,
              position_tolerance_m=.003, orientation_tolerance_rad=.03,
              contact_mode='guarded', load_budget_n=None):
        if contact_mode not in ('guarded', 'manipulation'):
            return CapabilityResult.failed(FailureCategory.INVALID_ARGUMENT, 'contact_mode must be guarded or manipulation')
        make_detector = getattr(self.robot, 'make_contact_detector', None)
        detector = make_detector(contact_mode, load_budget_n) if callable(make_detector) else None
        from captain_harness.adapters.robotics.contact_limits import EMERGENCY_LIMIT_N
        path, timing = self._prepare_move(position, quaternion, max_speed_m_s,
            max_angular_speed_rad_s, max_duration_s, position_tolerance_m, orientation_tolerance_rad)
        if not timing['command_time_feasible']:
            # v4: extend a too-short command budget to the controller's own minimum (capped at 30 s)
            # instead of rejecting; the receipt records the extension.
            minimum = float(timing.get('minimum_command_duration_s') or 0.)
            extended = min(30., minimum * 1.25 + .5)
            if 0. < minimum <= extended and extended > max_duration_s:
                path, timing = self._prepare_move(position, quaternion, max_speed_m_s,
                    max_angular_speed_rad_s, extended, position_tolerance_m, orientation_tolerance_rad)
                timing['duration_auto_extended_from_s'] = max_duration_s
                max_duration_s = extended
        if not timing['command_time_feasible']:
            return CapabilityResult.failed(FailureCategory.NOT_EXECUTED, 'insufficient_command_duration',
                value={**timing, 'stopped_reason':'insufficient_command_duration',
                    'executed_distance_m':0., 'endpoint_reached':False})
        state=self.robot.get_robot_state()
        model=self.tools._motion_model()
        seed=np.asarray(state.value.joint_positions_rad)
        aperture=state.value.gripper_open_fraction
        empty=np.empty((0,3))
        if np.any(seed<model.limits[:,0]) or np.any(seed>model.limits[:,1]):
            return CapabilityResult.failed(FailureCategory.NOT_EXECUTED,"measured_robot_joint_limits_exceeded")
        initial=model.collision(seed,aperture,empty)
        if initial["collision"]:
            return CapabilityResult.failed(FailureCategory.NOT_EXECUTED,"robot_start_self_collision",
                value={"robot_collision":initial})
        start_seed=seed.copy()
        for index,pose in enumerate(path[1:],1):
            check_deadline()
            solved=model.solve_ik(pose,seed,aperture=aperture,
                position_tolerance=.001,orientation_tolerance=.01)
            if solved is None:
                return CapabilityResult.failed(FailureCategory.IK_FAILURE,
                    "requested_cartesian_path_ik_unavailable",value={"sample_index":index,"ik":model.last_ik,
                    **_reach_hint(model,path,index,aperture,start_seed)})
            # A restart that lands on another IK branch is not a continuous Cartesian path (the executor
            # cannot follow it); report it as unavailable IK, not as the far branch's self-collision.
            if IK_BRANCH_CHECK and np.abs(np.asarray(solved)-seed).max()>1.0:
                return CapabilityResult.failed(FailureCategory.IK_FAILURE,
                    "requested_cartesian_path_ik_unavailable",value={"sample_index":index,"ik":model.last_ik,
                    "reason":"discontinuous_ik_branch",**_reach_hint(model,path,index,aperture,start_seed)})
            seed=np.asarray(solved)
            hit=model.collision(seed,aperture,empty)
            if hit["collision"]:
                return CapabilityResult.failed(FailureCategory.NOT_EXECUTED,
                    "requested_cartesian_path_self_collision",value={"sample_index":index,"robot_collision":hit})
        result, stops = self._guarded_execution(lambda: self.robot._execute_contact_sweep(
            path,max_speed_m_s=max_speed_m_s,
            max_angular_speed_rad_s=max_angular_speed_rad_s,max_force_n=(EMERGENCY_LIMIT_N if detector is not None else 18.),
            max_duration_s=max_duration_s,max_control_steps=self.robot._cartesian_max_steps,
            contact_mode=contact_mode,detector=detector,
            position_tolerance_m=position_tolerance_m,orientation_tolerance_rad=orientation_tolerance_rad,
            precision_reference="absolute_pose_goal"), model=model)
        return replace(result,value={**dict(result.value or {}),"execution_stop":stops,
            "path_samples":len(path),"requested_position":list(position),"requested_quaternion":list(quaternion),
            "control_scope":"explicit model EEF command; sampled robot IK/self collision and live force/tracking guards",
            "scene_contact":"model-planned; optional check_motion is diagnostic, not an execution prerequisite"})

    def _guarded_execution(self, execute, *, model, recovery=False, stationary=False, release=False, release_increase=None, gripper=False):
        old_guard = getattr(self.robot, "_motion_guard", None)
        stops = {}
        best_excess = None
        samples = []
        baseline = self.robot._eef_force_norm() if stationary else None
        reference = self.robot.get_robot_state().value if stationary else None
        release_best = baseline
        release_window_force = baseline
        release_window_step = self.robot.simulation_steps
        release_aperture = reference.gripper_open_fraction if reference else None
        collision_exit = None
        collision_initial_checked = False
        residual_calibration = (getattr(self.robot, '_force_calibration', None)
            if getattr(self.robot, '_contact_detector_mode', 'legacy') == 'residual' else None)
        stationary_state = {'over': 0}
        def guard():
            nonlocal best_excess, release_best, release_window_force, release_window_step, release_aperture
            nonlocal collision_exit, collision_initial_checked
            now = self.robot.get_robot_state()
            if not now.success or now.value is None:
                stops["reason"] = "robot_state_unavailable"
                return False
            if stationary:
                force = self.robot._eef_force_norm()
                p = np.asarray(now.value.end_effector_position_m)
                q = np.asarray(now.value.end_effector_quaternion_wxyz)
                pe = float(np.linalg.norm(p - reference.end_effector_position_m))
                ae = float((Rotation.from_quat(np.roll(q,-1)) * Rotation.from_quat(
                    np.roll(reference.end_effector_quaternion_wxyz,-1)).inv()).magnitude())
                sample = {'step':self.robot.simulation_steps,
                    'sample_time_s':self.robot.simulation_steps/self.robot._control_frequency,
                    'eef_position_m':p.tolist(), 'eef_quaternion_wxyz':q.tolist(),
                    'commanded_eef_position_m':list(reference.end_effector_position_m),
                    'commanded_eef_quaternion_wxyz':list(reference.end_effector_quaternion_wxyz),
                    'position_error_m':pe, 'orientation_error_rad':ae, 'force_n':force,
                    'gripper_open_fraction':now.value.gripper_open_fraction,
                    'gripper_command_fraction':self.robot._gripper_command_fraction}
                # Sensor refresh can change values without advancing physics.
                if not samples or samples[-1] != sample:
                    samples.append(sample)
                if force is None or baseline is None or not np.isfinite(force):
                    stops['reason'] = 'force_unavailable'
                    return False
                if release:
                    aperture = now.value.gripper_open_fraction
                    if aperture is None or not np.isfinite(aperture):
                        stops['reason'] = 'gripper_measurement_invalid'
                        return False
                    if aperture < release_aperture - .01:
                        stops['reason'] = 'release_aperture_reversed'
                        return False
                    release_aperture = max(release_aperture, aperture)
                    if force > release_best + 2.:
                        stops['reason'] = 'release_force_rise'
                        return False
                    release_best = min(release_best, force)
                    if (self.robot.simulation_steps-release_window_step)/self.robot._control_frequency >= .5:
                        if force >= 18. and force > release_window_force - 1.:
                            stops['reason'] = 'release_not_unloading'
                            return False
                        release_window_step, release_window_force = self.robot.simulation_steps, force
                elif residual_calibration is not None and not gripper:
                    # v5: stationary contact from the calibrated residual, sustained; emergency limit absolute.
                    # Jaw commands are exempt (squeezing an object is contact by design; the closure monitor covers them).
                    from captain_harness.adapters.robotics.contact_limits import EMERGENCY_LIMIT_N, SUSTAIN_S
                    nonlocal_over = None
                    world = self.robot._eef_force_world()
                    if world is None:
                        stops['reason'] = 'force_unavailable'
                        return False
                    _tare = getattr(self.robot, '_rd_tare', None)   # r9 (opt-in): re-tared offset after a guard stop
                    contact = float(np.linalg.norm(np.asarray(world) - np.asarray(residual_calibration.bias_world_n)
                                                   - (np.asarray(_tare) if _tare is not None else 0.)))
                    sample['contact_n'] = contact
                    if contact > EMERGENCY_LIMIT_N:
                        stops['reason'] = 'emergency_limit'
                        stops['contact_n'] = contact
                        return False
                    stationary_state['over'] = stationary_state['over'] + 1 if contact > residual_calibration.stationary_threshold_n else 0
                    if stationary_state['over'] >= max(1, int(round(SUSTAIN_S * self.robot._control_frequency))):
                        stops['reason'] = 'contact_detected'
                        stops['contact_n'] = contact
                        stops['threshold_n'] = residual_calibration.stationary_threshold_n
                        return False
                elif residual_calibration is not None and gripper:
                    from captain_harness.adapters.robotics.contact_limits import EMERGENCY_LIMIT_N
                    if force >= EMERGENCY_LIMIT_N:
                        stops['reason'] = 'emergency_limit'
                        return False
                elif force >= 18. or force-baseline >= 5.:
                    stops['reason'] = 'contact_force'
                    return False
                if pe > (.003 if release else .02) or ae > (.03 if release else .1):
                    stops['reason'] = 'tracking_error_limit'
                    return False
            joints = np.asarray(now.value.joint_positions_rad, dtype=float)
            if joints.shape != model.limits[:, 0].shape or not np.isfinite(joints).all():
                stops["reason"] = "robot_joint_measurement_invalid"
                return False
            # Separate signed boundaries: an opposite-boundary crossing cannot
            # reuse the old excess. No global hardware limit is relaxed.
            excess = np.maximum(np.stack((model.limits[:, 0]-joints,
                                          joints-model.limits[:, 1]), axis=1), 0.)
            if best_excess is None:
                best_excess = excess.copy() if recovery else np.zeros_like(excess)
            if np.any(excess > best_excess + (1e-6 if recovery else 0.)):
                stops.update(reason="robot_joint_limit_worsened" if recovery else
                    "robot_joint_limits_during_execution", joint_limit_excess_rad=excess.tolist())
                return False
            best_excess = np.minimum(best_excess, excess)
            hit = model.collision(joints, now.value.gripper_open_fraction, np.empty((0,3)))
            initial_collision_check = not collision_initial_checked
            collision_initial_checked = True
            if initial_collision_check and hit["collision"] and release and release_increase is not None:
                try:
                    from captain_harness.adapters.robotics.collision_release import MonotonicCollisionRelease
                    collision_exit = MonotonicCollisionRelease(model,joints,
                        now.value.gripper_open_fraction,release_increase)
                    stops["collision_release"] = collision_exit.receipt()
                except ValueError as exc:
                    stops.update(reason="bounded_release_geometry_unavailable",robot_collision=hit,
                        recovery_detail=str(exc))
                    return False
            if collision_exit is not None:
                try:
                    collision_exit.observe(hit,now.value.gripper_open_fraction)
                    stops["collision_release"] = collision_exit.receipt()
                except ValueError as exc:
                    stops.update(reason="bounded_release_collision_worsened",robot_collision=hit,
                        recovery_detail=str(exc))
                    return False
            elif hit["collision"]:
                stops.update(reason="robot_self_collision_during_execution", robot_collision=hit,
                    recovery_guidance="Ordinary commands cannot start in self-collision. Explicit release checks a bounded opening path; no safe path is guaranteed.")
                return False
            if old_guard and not old_guard():
                stops["reason"] = "inherited_motion_guard"
                return False
            return True
        self.robot._motion_guard = guard
        initial_step = self.robot.simulation_steps
        try:
            if not guard():
                return CapabilityResult.failed(FailureCategory.NOT_EXECUTED, stops["reason"],
                    value={"trajectory":samples, "stopped_reason":stops["reason"]}), stops
            result = execute()
            # Include the final sample even if the driver reports success.
            if not guard():
                result = CapabilityResult.failed(FailureCategory.CONTROLLER_REJECTED,
                    stops["reason"], executed=self.robot.simulation_steps > initial_step,
                    value={**dict(result.value or {}), "reached": False})
            elif collision_exit is not None and collision_exit.remaining and result.success:
                stops.update(reason="bounded_release_collision_unresolved",collision_release=collision_exit.receipt())
                result = CapabilityResult.failed(FailureCategory.CONTROLLER_REJECTED,stops["reason"],
                    executed=self.robot.simulation_steps > initial_step,value={**dict(result.value or {}),"reached":False})
            elif recovery and result.success and np.any(best_excess > 0.):
                stops.update(reason="recovery_joint_limits_unresolved",
                    joint_limit_excess_rad=best_excess.tolist())
                result = CapabilityResult.failed(FailureCategory.CONTROLLER_REJECTED, stops["reason"],
                    executed=self.robot.simulation_steps > initial_step,
                    value={**dict(result.value or {}), "reached": False})
            if stationary:
                result = replace(result, value={**dict(result.value or {}), 'trajectory':samples,
                    'stopped_reason':stops.get('reason', (result.value or {}).get('stopped_reason', result.message or 'completed')),
                    'max_observed_force_n':max((x['force_n'] for x in samples if x['force_n'] is not None and np.isfinite(x['force_n'])), default=None),
                    'executed_distance_m':float(sum(np.linalg.norm(np.asarray(b['eef_position_m'])-a['eef_position_m']) for a,b in zip(samples[:-1],samples[1:]))),
                    'control_steps':self.robot.simulation_steps-initial_step,
                    'elapsed_s':(self.robot.simulation_steps-initial_step)/self.robot._control_frequency,
                    'force_scope':'18 N sampled trigger; inter-sample peaks are not bounded'})
            return result, stops
        finally:
            self.robot._motion_guard = old_guard

    def _recover(self, max_distance_m=.05, max_duration_s=10., mode="full", continuation_id=None):
        result, stops = self._guarded_execution(
            lambda: self.robot.retreat_last_motion(max_distance_m=max_distance_m,
                max_duration_s=max_duration_s, mode=mode, continuation_id=continuation_id), model=self.tools._motion_model(), recovery=True)
        return replace(result, value={**dict(result.value or {}), "execution_stop": stops,
            "recovery_scope": "bounded measured-trace reversal; no object-level recovery inferred"})

    def _release(self, max_duration_s=.5, max_open_fraction_increase=.25):
        for value, maximum in ((max_duration_s, 2.), (max_open_fraction_increase, .25)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 < value <= maximum:
                raise ValueError("release duration must be in (0,2], opening increase in (0,.25]")
        if int(max_duration_s*self.robot._control_frequency+1e-9) < 1:
            raise ValueError("release duration is below one control cycle")
        def execute():
            current = self.robot.get_robot_state()
            if not current.success or current.value is None:
                return CapabilityResult.failed(FailureCategory.NOT_EXECUTED, "robot_state_unavailable")
            if current.value.gripper_open_fraction >= 1. - 1e-6:
                return CapabilityResult.failed(FailureCategory.NOT_EXECUTED, "release_already_open")
            reset = self.robot.cancel_direct_targets(cancel_gripper=True,
                max_open_fraction_correction=max_open_fraction_increase)
            if reset.get("status") != "cancelled":
                return CapabilityResult.failed(FailureCategory.BACKEND_ERROR,
                    "release_target_reset_unconfirmed", value={"pre_release_targets":reset})
            projection = float((reset.get("gripper_target") or {}).get("opening_projection_fraction", 0.)) if isinstance(reset.get("gripper_target"), dict) else 0.
            result = self.robot.release_opening(max_duration_s, max_open_fraction_increase-projection)
            return replace(result, value={**dict(result.value or {}), "pre_release_targets":reset})
        result, stops = self._guarded_execution(execute,
            model=self.tools._motion_model(), stationary=True, release=True,
            release_increase=max_open_fraction_increase)
        return replace(result, value={**dict(result.value or {}), "execution_stop":stops,
            "release_scope":"bounded opening only; measured unload is not object/task recovery",
            "object_released":"pending_visual_confirmation"})


USAGE += "\nMeasurement references are selected by measurement_id; the Harness resolves sensor images, depth/calibration and source time as a complete evidence group. You do not need to copy source frame IDs for that measurement. A group that cannot fit the unchanged image budget is explicitly reported. compare_object_observations derives the full physical action interval when action_ids is omitted; supplied IDs must still match. This associates effects with an interval and never proves single-action causation or task identity."


REACH_HINT = False  # opt-in (RoboDojo X5): add robot-kinematics reach facts to IK-unavailable move_to receipts
IK_BRANCH_CHECK = False  # opt-in (RoboDojo X5, always on there): a path sample whose IK jumps to another branch is unavailable IK


def _reach_hint(model, path, index, aperture, start_seed):
    """Robot-kinematics-only facts for an unreachable move_to: the last reachable sample of the requested straight path
    and the highest reachable EEF z at the goal xy with the requested orientation (planner IK + self-collision only;
    no scene, contact or task information)."""
    if not REACH_HINT:
        return {}
    goal = np.asarray(path[-1], float)
    best = None
    for z in np.arange(goal[2, 3], -0.005, -0.01)[:40]:
        check_deadline()
        target = goal.copy()
        target[2, 3] = z
        q = model.solve_ik(target, start_seed, aperture=aperture, position_tolerance=.002, orientation_tolerance=.02)
        if q is not None and not model.collision(q, aperture, np.empty((0, 3)))["collision"]:
            best = round(float(z), 3)
            break
    return {"reach_hint": {
        "last_reachable_path_point_m": [round(float(v), 4) for v in np.asarray(path[max(0, index - 1)])[:3, 3]],
        "goal_xy_max_reachable_z_m": best,
        "basis": "this arm's kinematics only (planner IK and self-collision) at the requested orientation; "
                 "None = no reachable height at this xy with this orientation; not a scene, path or grasp check"}}
