"""Adapter-owned workflow notes for the expanded public robot interface."""

SENSOR_MOTION_USAGE = """Public robot environment and contracts (sensor-motion-v11):
Read the available RGB/RGB-D cameras, public robot state and declared calibration.
A successful capability operation can require no physical movement. Inspect
result.diagnostics['physical_execution']: control_steps is the measured count,
0 means no control step, and null/status=unknown means that count is unavailable.
The count alone proves neither reaching a pose nor holding an object nor task completion.
Check the returned endpoint/attachment evidence and fresh post-action images.
Coordinates are meters in robot_base unless a tool explicitly states otherwise;
quaternions use wxyz. observe().value['robot_state'] distinguishes tool and end_effector poses.
Do not substitute an object's centroid for the tool or end-effector position.
Placement recommended_eef_delta_m is a nominal partial-geometry calculation, not an
accuracy guarantee or a required command. Read correction_resolution when supplied;
a delta below observed geometry change does not justify chasing that exact number.
Inspect existing views and measured action residuals; do not silently enlarge commands.
Fine-motion stopping criteria can be tighter than the controller achieves within the
requested duration. A timeout preserves actual movement and does not mean zero motion.


A point-service no_point result is a localization failure, not proof the target is absent.
If you can identify it in the current image, use mark_point with measured image coordinates;
otherwise inspect the last reliable before/after view and resolve the actual occlusion.
Selection failures return selection_recovery with source/camera and any geometrically
matching candidates. Inspect and explicitly choose; never repeat the same rejected pair.
select_detection binds explicit identity points even for one candidate. Multiple
distinct candidates require an identity point in the same camera/public snapshot.
Its default selection_relation='identity' does not turn a relational description
into a distance formula. Use selection_relation='nearest_xy' with a reference
point only when you explicitly need the minimum observed XY centroid distance;
that measurement alone does not establish the task's semantic relation.
When relocalizing, pass previous_detection_id to select_detection. Match the
original task target using initial/current images and surrounding parts; a new
label or artifact ID is not identity proof. An explicit corrected_target and
identity_update_reason can revise the earlier interpretation. Inspect recent_selections
and visual_correspondence, including contradictory or insufficient measurements.

For a planned grasp approach, open_gripper BEFORE the observation used for
segmentation, grasp estimation and build_scene. Read planned_grasp_preparation:
this path requires open_fraction >= 0.95; a partially open hand may still need
an opening command. A prepared gripper needs no extra opening command. Once motion or dynamics time advances, obtain fresh
geometry. Two handles may alias only an exactly identical timed public snapshot;
this does not make a pose from before movement current.

grasp_pose_estimate screens its first native candidate with bounded public
geometry alternatives, including a final orientation pass that preserves the
calibrated finger-pad center in world coordinates. It preserves original candidates in sensor_screening and
may return a new, unscored geometry candidate. A requested explicit grasp_style
is not silently changed. sensor_screening.geometry_choice exposes a checked
alternative's grasp_handle, measured_checks_passed and requested_style_match.
If it requires a style-change decision, explicitly choose that handle only after
reviewing its geometry and the current image; no new estimate is required just
to select an already issued alternative. Inspect original versus selected evidence. Endpoint IK,
observed collision checks and finger-pad overlap do not guarantee a stable grasp.
sensor_screening.candidate_comparison measures up to four diverse candidates.
Inspect tolerance_only contact versus interior_observed support and obstructions;
surface_quality reports closing-axis span/offset, visible depth and edge reserve,
and whether local normal evidence supports opposing visible patches. One visible
face or unobserved surfaces remain unverified; these fields cannot prove friction,
physical contact or retention. Native scores are not comparable across backends. preferred_for_inspection is a
geometry shortlist, not a retention guarantee. pose_sensitivity measures whether
visible pad overlap disappears under sampled translation/rotation perturbations.
These are declared test magnitudes, not calibrated sensor uncertainty or a guarantee
against combined pose errors. Compare this evidence with the contact precision you request. After a lost grasp, use the measured
evidence to choose whether and how to revise the approach/contact region.
Detailed alternative/search trials remain available through read_execution_result
at the declared on-demand paths; before/after and selected checks remain in feedback.
When a result says requires_decision=true, the reflective executor ends that code
cell at the measurement call. Its full public result is in the returned tool
feedback; assignment or remember_record statements after the call will not run.
Read that feedback and choose the next physical action in the following model turn. Do not append an
unconditional align_pregrasp or open_gripper after such a measurement.

Pass an issued grasp_handle string, {'grasp_handle': handle}, or its unchanged
full descriptor to grasp tools. The registry owns its pose and native binding;
never change its coordinates or copy a different native payload into it.
align_pregrasp(candidate=handle, scene_handle=scene) and
align_pregrasp(grasp_handle=handle, scene_handle=scene) are equivalent; supply
exactly one candidate reference, never both keyword forms.
refine_grasp issues a new reference. To reuse an orientation after movement, you
may explicitly choose a fresh target_detection_id and pass it with the prior
issued candidate and current scene. This constructs a new unscored candidate
and track from current visible geometry; target identity across observations is
your decision, not automatically verified. It clears old native placement and
attachment bindings. Without explicit refresh, stale-source rejection remains.
align_pregrasp reaches the chosen pregrasp;
descend_and_close then approaches contact in a later reflected action.
Its optional position_tolerance_m requests a measured pre-closure endpoint tolerance
between 0.0005 and 0.003 m (default 0.003 m). You choose the tolerance from the
contact region and available evidence. Force/step limits remain in force; a tighter
request may stop without closing. contact_precision separates pre-closure
compliance from post-closure drift; neither proves the target is attached. A stopped
descent may execute no closure: inspect its remaining displacement and images.
closure_force_monitor.closure_motion reports EEF translation/rotation relative to
the pre-closure pose, including sampled maxima and coverage. It excludes descent
error. Missing initial/final pose stays unknown; low drift does not prove retention.
controller_execution.retreat_available reports whether that unclosed descent
retained its measured route for an immediate retreat_last_motion. Gripper closure
or later physical actions/time invalidate it; inspect the retreat's own result.
Automatic XY correction uses corresponding RGB-D feature motion, not a change
in segmentation centroid. Excessive measured drift requires a fresh grasp.

For freely carried objects, inspect the lift and verify target identity/motion
with public attachment evidence before transport. A jaw opening alone is not an
attachment proof. lift_probe preserves the OSC pose-goal response with sampled
force stops; it does not guarantee command rate. A reached robot endpoint still
requires evidence that the target followed.
Choose probe type, direction and amplitude from current geometry, uncertainty and force limits.
attachment_evidence_gaps names unmet and unknown public checks, source/current
camera and unchanged numerical thresholds. Unknown motion checks on uncomparable
visible surfaces do not mean the target physically failed to move. They identify
missing evidence for your next decision, not a prescribed recovery motion.
Attachment thresholds describe what the verifier can certify; they do not prescribe a motion.
Sub-threshold measured motion is not proof of stationarity. A verified observation
is not proof of sustained retention across later motion.
During holds and transport, possession_monitor samples the public gripper register; loss or unavailable
readings stop motion and invalidate old attachment/placement evidence.
Maintained possession does not prove identity or a stable grasp; inspect fresh
images and the aperture trend. On unknown evidence, reobserve and verify again;
reclosing is not mandatory. On loss, reacquire before further carried-object motion.
An attachment mark must come from the current robot state and sensor time.
partial_geometry_motion_unresolved means the pre/post visible regions are not
comparable: inspect target_motion_evidence, then obtain a clearer current mark
in next_evidence_camera. These supplemental feature correspondences do not
independently authorize attachment or define an object pose. Re-observation can
resolve uncertainty without a new grasp attempt.
For articulated fixtures or switches, choose motions suited
to their observed mechanism; lifting a drawer or declaring it rigidly carried
is not a required intermediate goal. Inspect public motion/force results and
the raw before/after views. plan_motion accepts fresh capture_carried_geometry evidence
for held-object collision checking when available. Its partial measured mesh leaves
unobserved surfaces unknown; read the current tool contract and source binding.

For an unladen free-space waypoint, build_scene from a fresh observation and
pass scene_handle to goto_pose with z_approach=0 and an explicit pose reference.
This plans and executes in one call; a preliminary solve_ik is unnecessary.
Planning rejection executes nothing and is not proof of physical unreachability.
Without a scene, goto_pose uses bounded Cartesian commands and sampled force/
tracking stops but does not certify a collision-free route. solve_ik validates
the proposed joints with independent FK and the declared endpoint tolerances;
an empty, malformed or inaccurate external reply gets one bounded robot-model
recovery before failure, using the same independent endpoint checks.
An endpoint solution never certifies the path. propose_pose_candidates checks
only the explicit task-compatible orientations you provide, at one reference
position. Use a list of quaternions or base_quaternion/rotation_axis/angle_offsets_rad.
Inspect all failed endpoints and ranked_candidate_indices, then choose a returned
goto_arguments entry; it is not a path or retention certificate.

For intentional contact, move_linear specifies a base-frame direction/distance;
rotate_about_axis specifies the base-frame axis, pivot and signed angle for the
whole end-effector pose. Choose these from observed geometry. Both require
measured wrist force and preserve the gripper command. The default 18 N force
threshold is also the maximum accepted value. It is a sampled stop, not a bound on physical force peaks. Inspect the
actual displacement, stop reason, force and current images after each motion.
For a manipulation probe, observe and select the same visible part after the
last motion, then pass target_detection_id and motion_intent="target_contact" to
move_linear or rotate_about_axis. This declaration requires a public target binding.
For an approach, free transport or retreat that only commands the hand waypoint,
use motion_intent="free_motion". Free motion does not report object progress.
The adapter executes bounded segments and checks visible target progress, keeping
the total duration and original force-increase limit. target_motion_unresolved
requires better observation; no_observed_target_progress or opposite_target_progress
requires inspecting the contact point/direction/axis. A stopped probe retains only
its last measured segment for retreat. Do not repeat the same loaded command or
increase the force threshold without new evidence. Progress is not task completion.
After a guarded stop, retreat_last_motion can reverse the most recent measured
route even under an existing load, stopping if force increases by 5 N. Any
intervening physical action or time advance invalidates that route; observe and
read-only evidence queries do not. A retreat cannot be reversed into the obstacle.
read_motion_trajectory pages public samples from an issued trajectory_handle.
Full controller_execution/phase_executions records are retained in the execution
log and accessible with read_execution_result at their on-demand feedback paths.
planned_joint_path is also saved with plan and execution results, so it can be
compared with the actual joint trajectory. Retained logs do not refresh a plan.

placement_estimate may omit selected_grasp and target_detection_id: the adapter
reuses the grasp and target actually used for closure on tracking_handle.
For example: placement_estimate(track, destination_detection_id=destination_id,
relation_type='in', reference_tolerance_m=your_declared_tolerance).
An explicit different executed-grasp binding is rejected before service work.
Use fresh destination geometry. Native AnyPlace needs compatible camera inputs;
public container geometry does not require re-segmenting a visible destination
into an old grasp view. A container-mouth fallback is a hover proposal, not a
release permission or a task-success claim. Inspect its quality and current views.
selected_placement_pose is a tool pose; use goto_pose with position_reference='tool'.
For a desired held-object center, use position_reference='held_target' and the
verified tracking_handle. After transport reobserve; safe-release advice is
public evidence and requires_decision may require another model turn.

Completion must address every clause of the original task. Existing on/in
placement assertions remain available for those relations. For articulated,
device-state, directional or compound goals, use measure_goal_evidence with
selected visible subject handles and concise natural-language claims. Do not
coerce a different goal into 'on'. The tool reports visible geometry and
acquisition requirements; it does not decide open/closed, on/off, or task truth.
Retain the first evidence_handle, hold_position to advance dynamics, observe
and select again, then measure with previous_evidence_handle. These separated
samples must follow the last configuration-changing action. If the earlier handle
is omitted, the tool can reuse an eligible unloaded sample for the same exact
claims and supported public subject correspondences; inspect previous_evidence_resolution.
A new observation handle at the same paused timestamp does not establish stability. Verify that
the gripper is unloaded. Any historical supporting_evidence_handles remain
historical: assess whether intervening actions invalidated their claims.
The separate visual verifier must assess the entire original task and every
declared claim from supplied current and relevant historical images. Missing
evidence calls for better observations, not a gratuitous physical correction.
Cite owned episode frame IDs in Working State evidence_frame_ids when intermediate
events matter. Completion verification prioritizes current views, initial identity,
release before/after, then these cited images under the same total image budget.
Its coverage report distinguishes exported citations from images actually supplied;
history proves its recorded time, not the present state.
"""

SENSOR_MOTION_USAGE += """
Attempt comparisons retain original effect_status and later bound derived_effect_status.
Association ended means prior possession is historical; releasing does not erase an
earlier failed grasp. execution_parameters contains adapter-issued selected geometry,
so a new opaque handle alone is not a new contact strategy. The same-reference
comparison does not prove semantic identity. Parameter changes do not establish
recovery; check the resulting object effect and remaining original task yourself.
"""
