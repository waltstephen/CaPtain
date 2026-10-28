"""Hardware-specific capability publication and sensor-planning tools."""
from dataclasses import replace
from captain_harness.ports.capability import CapabilitySpec, SideEffect, ControlSemantic


def runtime_manifest(manifest, *, controller, motion_tools=False):
    unavailable = ({'move_to_joints'} if controller == 'OSC_POSE' else
                   {'servo_to', 'align_pregrasp', 'descend_and_close', 'approach_and_close'})
    specs = [s for s in manifest.capabilities if s.name not in unavailable]
    if controller == 'OSC_POSE':
        specs = [replace(s, description='Return to the initial end-effector pose using OSC; '
                 'the original joint configuration is not guaranteed.') if s.name == 'home' else s for s in specs]
    if motion_tools:
        specs = [replace(s, description=s.description +
            ' In the LIBERO sensor-motion adapter, object_detected is an aperture proxy: '
            'a closing command with measured opening at or above 2 percent. It is not a target contact sensor. '
            'Slipping can precede the loss threshold; inspect the aperture trend and fresh RGB. '
            'This register alone proves neither target identity nor sustained retention.')
            if s.name == 'gripper_state' else s for s in specs]
        specs = [replace(s,
            arguments_schema={**s.arguments_schema, 'properties': {
                **s.arguments_schema.get('properties', {}),
                'tracking_handle': {'type': 'string', 'description':
                    'Optional explicit issued target track for this new closure attempt; does not verify possession.'}}},
            result_schema={**s.result_schema, 'properties': {
                **s.result_schema.get('properties', {}),
                'associated_tracking_handle': {'type': ['string', 'null']}}},
            argument_order=(*s.argument_order, 'tracking_handle'),
            description=s.description + ' To associate a primitive regrasp with its target, supply the issued tracking_handle. '
                'Without it the closure has an unbound attempt receipt; later target evidence cannot be borrowed from an old grasp.')
            if s.name == 'close_gripper' else s for s in specs]
        def expand(schema):
            if isinstance(schema, list):
                return [expand(v) for v in schema]
            if not isinstance(schema, dict):
                return schema
            result = {k: expand(v) for k, v in schema.items()}
            props = result.get('properties', {})
            if 'pre_descent_centering' in props:
                props['pre_descent_centering']['additionalProperties'] = True
            if 'backend' in props and 'detection_id' in props and 'position_m' in props:
                enum = props['backend'].get('enum')
                if enum is not None and 'public_geometry_refinement' not in enum:
                    enum.append('public_geometry_refinement')
                props.update(refinement_basis={'type': 'string'}, source_observation_handle={'type': 'string'},
                    approach_distance_m={'type': 'number', 'minimum': .005, 'maximum': .20})
                if 'model_score' in props:
                    props['model_score'] = {'type': ['number', 'null']}
                if 'style_preference_score' in props:
                    props['style_preference_score']['type'] = ['number', 'null']
                match = props.get('request_style_match', {})
                if 'enum' in match and 'unassessed' not in match['enum']:
                    match['enum'].append('unassessed')
            for name, risk in props.items():
                if name.endswith('convergence_risk') and 'enum' in risk and 'unassessed' not in risk['enum']:
                    risk['enum'].append('unassessed')
                if name == 'selected_grasp_backend' and 'enum' in risk and 'public_geometry_refinement' not in risk['enum']:
                    risk['enum'].append('public_geometry_refinement')
            return result
        specs = [replace(s, arguments_schema=expand(dict(s.arguments_schema)),
                         result_schema=expand(dict(s.result_schema))) for s in specs]
        for i, s in enumerate(specs):
            if s.name == 'select_detection':
                properties = {**s.arguments_schema['properties'],
                    'previous_detection_id': {'type': ['string', 'null'], 'default': None},
                    'identity_update': {'type': 'string', 'enum': ['same_target', 'corrected_target'], 'default': 'same_target'},
                    'identity_update_reason': {'type': 'string', 'maxLength': 512, 'default': ''}}
                s = replace(s, arguments_schema={**s.arguments_schema, 'properties': properties},
                    argument_order=tuple(properties),
                    result_schema={**s.result_schema, 'properties': {**s.result_schema.get('properties', {}),
                        'target_continuity': {'type': 'object'}, 'recent_selections': {'type': 'array', 'items': {'type': 'object'}}}},
                    description=s.description + ' For relocalization, pass previous_detection_id to retain '
                    'the initial target anchor. Use identity_update=corrected_target with an evidence-based '
                    'reason when revising the original choice. Recent selections expose unlinked changes too. '
                    'These links and labels do not prove identity; inspect initial/current images and track_target.')
                specs[i] = s
            if s.control_semantic is ControlSemantic.GRASP_PLANNING:
                candidates_schema = s.result_schema.get('properties', {}).get('candidates', {})
                candidate_schema = candidates_schema.get('items', {})
                result = {**s.result_schema, 'properties':{
                    **s.result_schema.get('properties', {}),
                    'sensor_screening':{'type':'object', 'properties':{
                        'original_candidates':candidates_schema,
                        'refinement':{'type':'object','properties':{
                            'candidate':candidate_schema,
                            'alternatives':{'type':'array','x-decision-feedback':'on_demand'},
                            'contact_orientation':{'type':'object','properties':{
                                'alternatives':{'type':'array','x-decision-feedback':'on_demand'}},
                                'additionalProperties':True},
                            'local_search':{'type':'object','properties':{
                                'attempts':{'type':'array','x-decision-feedback':'on_demand'}},
                                'additionalProperties':True}},'additionalProperties':True}},
                        'additionalProperties':True}}}
                if 'selection_feedback' in result['properties']:
                    result['properties']['selection_feedback']['additionalProperties'] = True
                s = replace(s, result_schema=result,
                    description=s.description + ' With sensor-motion tools, the first candidate is screened against '
                    'bounded public geometry alternatives. Inspect sensor_screening and original_candidates; '
                    'requires_decision=true ends this code cell before any physical action.')
                specs[i] = s
            if s.name == 'align_pregrasp':
                arguments = dict(s.arguments_schema)
                arguments['properties'] = {**arguments['properties'],
                    'scene_handle': {'type': ['string', 'null'], 'default': None},
                    'grasp_handle': {'type': 'string', 'pattern': '^artifact://v1/'}}
                arguments['required'] = [name for name in arguments.get('required', ()) if name != 'candidate']
                arguments['oneOf'] = [{'required':['candidate'], 'not':{'required':['grasp_handle']}},
                                      {'required':['grasp_handle'], 'not':{'required':['candidate']}}]
                result = {**s.result_schema, 'additionalProperties': True}
                evidence = result.get('properties', {}).get('candidate_evidence', {}).get('properties', {})
                if 'motion_path' in evidence:
                    evidence['motion_path'] = {'type':'string', 'enum':['not_checked', 'planned_observed_surfaces', 'planning_or_execution_failed']}
                specs[i] = replace(s, arguments_schema=arguments,
                    argument_order=(*s.argument_order, 'scene_handle', 'grasp_handle'),
                    result_schema=result,
                    description=s.description + ' Supply exactly one of candidate or grasp_handle; '
                    'grasp_handle is an alias for an issued candidate reference, with identical registry checks. '
                    'With scene_handle, use OMPL and monitored execution '
                    'to this candidate pregrasp; first open the gripper, observe and build_scene. '
                    'Planning failure executes nothing and keeps previous alignment. '
                    'Only the selected target\'s measured samples versus the gripper use the declared '
                    '3 mm contact clearance during this grasp approach; the arm and other surfaces retain '
                    'transport clearance. Physical penetration is never exempted. '
                    'The default staged controller remains available when scene_handle is omitted.')
        specs.extend(motion_capabilities())
        from .contact_tools import contact_motion_capabilities
        specs.extend(s for s in contact_motion_capabilities()
                     if controller == 'OSC_POSE' or s.name == 'read_motion_trajectory')
        specs.append(goal_evidence_capability())
        from .pose_tools import pose_candidate_capability
        specs.append(pose_candidate_capability())
        reference = {'type': 'string', 'pattern': '^artifact://v1/'}
        for i, s in enumerate(specs):
            arguments = dict(s.arguments_schema)
            properties = dict(arguments.get('properties', {}))
            uses_grasp = False
            for key in ('candidate', 'selected_grasp'):
                if key not in properties:
                    continue
                uses_grasp = True
                descriptor = dict(properties[key])
                descriptor['type'] = ['object', 'string']
                descriptor['pattern'] = reference['pattern']
                required = descriptor.pop('required', [])
                descriptor['properties'] = {**descriptor.get('properties', {}), 'grasp_handle': reference}
                descriptor['anyOf'] = [{'required': required},
                    {'required': ['grasp_handle'], 'maxProperties': 1}]
                properties[key] = descriptor
            if uses_grasp:
                arguments['properties'] = properties
                description = s.description + (
                    " Pass candidate=g['grasp_handle'] or {'grasp_handle': g['grasp_handle']} to reuse an immutable issued grasp. "
                    "The reference restores its target, source observation, pose and native payload; "
                    "do not reconstruct or edit its fields. refine_grasp issues a new reference when it changes the pose or approach. "
                    "A retained reference does not make old geometry fresh.")
                if s.name == 'placement_estimate':
                    arguments['required'] = [k for k in arguments.get('required', [])
                                             if k not in {'selected_grasp','target_detection_id'}]
                    description += (' Omit selected_grasp and target_detection_id to use the grasp and target actually executed for tracking_handle. '
                                    'Target identity and public attachment checks still apply.')
                if s.name in {'evaluate_grasp', 'refine_grasp'}:
                    properties['scene_handle'] = {'type': ['string', 'null'], 'default': None}
                    arguments['required'] = [k for k in arguments['required'] if k != 'scene_handle']
                    description += (' Omit scene_handle to build from the explicit fresh target_detection_id when supplied, '
                        'otherwise from the candidate source observation. Robot/time freshness is still checked; '
                        'after movement obtain a fresh selected target.' if s.name == 'refine_grasp' else
                        ' Omit scene_handle to build from the candidate source observation. Robot/time freshness is still checked.')
                specs[i] = replace(s, arguments_schema=arguments, description=description)
            elif s.name == 'check_reachable':
                properties['pose'] = {'anyOf': [properties['pose'], {
                    'type': 'object', 'properties': {'grasp_handle': reference},
                    'required': ['grasp_handle'], 'additionalProperties': False}, reference]}
                arguments['properties'] = properties
                specs[i] = replace(s, arguments_schema=arguments,
                    result_schema={**s.result_schema, 'additionalProperties': True},
                    description='For an issued grasp reference or descriptor, check contact and pregrasp '
                    'EEF endpoints using the same robot model and conversion as planned alignment. '
                    'The approach distance is part of the candidate, default 6 cm. '
                    'reachable=true means both endpoint IK solutions were returned; no collision or path check. '
                    'A plain pose without grasp approach metadata retains the tool/TCP PyRoKi endpoint contract. '
                    'Read check_scope and motion_path before acting.',
                    examples=("check_reachable({'grasp_handle': g['grasp_handle']})",))
        trajectory_schema = {'type': 'array', 'items': {'type': 'object'}, 'x-decision-feedback': 'on_demand'}
        planned_path_schema = {'type': 'array', 'items': {'type': 'array', 'items': {'type': 'number'}},
                               'x-decision-feedback': 'on_demand'}
        controller_schema = {'type': ['object', 'null'], 'properties': {'trajectory': trajectory_schema},
                             'additionalProperties': True}
        for i, s in enumerate(specs):
            if s.side_effect in {SideEffect.ROBOT_MOTION, SideEffect.ACTUATOR}:
                result = {**s.result_schema, 'properties': {**s.result_schema.get('properties', {}),
                    'trajectory': trajectory_schema, 'controller_execution': controller_schema,
                    'phase_executions': s.result_schema.get('properties', {}).get('phase_executions',
                        {'type': 'array', 'x-decision-feedback': 'on_demand'}),
                    'stop_evidence': {'type': 'object', 'properties': {'controller_execution': controller_schema},
                                      'additionalProperties': True}}}
                if s.name == 'home':
                    result.update(type=['object', 'null'], additionalProperties=True)
                s = replace(s, result_schema=result)
            if s.name in {'plan_motion', 'execute_motion', 'goto_pose', 'align_pregrasp'}:
                result = {**s.result_schema, 'properties': {**s.result_schema.get('properties', {}),
                    'planned_joint_path': planned_path_schema,
                    'planning': {'type': 'object', 'properties': {'planned_joint_path': planned_path_schema},
                                 'additionalProperties': True}}}
                s = replace(s, result_schema=result,
                    description=s.description + ' The complete planned_joint_path remains in the saved execution result '
                    'and is available through read_execution_result; normal feedback projects the array on demand.')
            if s.name == 'assert_entity_attached':
                properties = dict(s.result_schema['properties'])
                properties['reason'] = {**properties['reason'],
                    'enum': [*properties['reason']['enum'], 'partial_geometry_motion_unresolved',
                             'measured_motion_below_attachment_threshold', 'measured_lift_below_attachment_threshold']}
                properties['attachment_evidence_basis'] = {
                    **properties['attachment_evidence_basis'],
                    'enum': [*properties['attachment_evidence_basis']['enum'],
                             'fresh_target_motion_and_gripper_register']}
                properties.update(motion_diagnosis={'type':'object', 'additionalProperties':True, 'x-attempt-feedback':'priority'},
                    target_motion_evidence={'type': ['object', 'null'], 'additionalProperties': True},
                    next_evidence_camera={'type': ['string', 'null']}, recovery_guidance={'type': 'string'},
                    lift_execution={'type': ['object', 'null'], 'additionalProperties': True},
                    motion_authorized={'const': False}, motion_guidance={'type': 'string'})
                s = replace(s, result_schema={**s.result_schema, 'properties': properties, 'additionalProperties': True},
                    description='Validate possession using a current same-target observation and gripper evidence. '
                    'current_point_handle must come from the current robot state and sensor time. '
                    'Fresh same-view geometry, appearance and target-motion checks can confirm possession after '
                    'a correctly bound inconclusive lift; the failed lift remains failed in lift_execution. '
                    'verified and pose_valid describe evidence and pose availability only, not motion authorization. '
                    'Resolve contact stops before transport; motion force and collision guards still apply. '
                    'With changed visible geometry, centroid differences are unresolved motion evidence; inspect '
                    'target_motion_evidence and reobserve in next_evidence_camera. These feature measurements '
                    'do not independently authorize attachment or an object pose. A sub-threshold displacement '
                     'is measured motion with insufficient amplitude, not proof that the object stayed still. '
                     'motion_diagnosis separates measurements, verifier thresholds and unverified sustained retention.')
            if s.name == 'goto_pose':
                arguments = {**s.arguments_schema, 'properties': {**s.arguments_schema['properties'],
                    'scene_handle': {'type': ['string', 'null'], 'default': None}}}
                clearance_order = ()
                if controller == 'OSC_POSE':
                    arguments['properties']['clearance_first'] = {'type':'boolean', 'default':False,
                        'description':'With world approach and no scene_handle, rise at current XY first; '
                            'then traverse at max(current tool Z, target tool Z + z_approach), then descend. '
                            'All phases share one controller step budget and deadline.'}
                    clearance_order = ('clearance_first',)
                    properties = {**s.result_schema.get('properties', {}),
                        'clearance_first':{'type':'boolean'},
                        'clearance_tool_z_m':{'type':'number'},
                        'route_scope':{'type':'string'},
                        'control_budget':{'type':'object', 'x-attempt-feedback':'priority',
                            'properties':{name:{'type':'integer','minimum':0}
                                for name in ('limit_steps','used_steps','remaining_steps')},
                            'required':['limit_steps','used_steps','remaining_steps'],
                            'additionalProperties':False}}
                    s = replace(s, result_schema={**s.result_schema,'properties':properties})
                s = replace(s, arguments_schema=arguments,
                    argument_order=(*s.argument_order, 'scene_handle', *clearance_order),
                    result_schema={**s.result_schema, 'additionalProperties': True},
                    description=s.description + ' With a fresh scene_handle and z_approach=0, plan and execute this '
                    'free-space waypoint with the public robot/scene model; no separate solve_ik call is needed. '
                    'Held-object planning remains unavailable. Without a scene, the driver uses bounded Cartesian '
                    'setpoints and sampled force/tracking stops, but does not check a collision-free path. '
                    'Use move_linear or rotate_about_axis for an intentional bounded contact operation.' +
                    (' clearance_first=True explicitly raises at current XY before travel, holding the initial '
                    'orientation during the raise. Use approach_frame=world and no scene_handle. '
                    'All phases retain the original rotation request and share one configured control-step '
                    'budget/deadline. Choose height from visible evidence; this route does not certify '
                    'collision freedom. Read completed_phases, failed_phase and control_budget before recovery.'
                    if controller == 'OSC_POSE' else ''))
            elif s.name == 'solve_ik':
                s = replace(s, description=s.description + ' The returned configuration is checked with independent '
                    'robot FK and joint limits against the declared pose tolerances. A numerical proposal that '
                    'fails these checks is not returned as success. This does not check the motion path. '
                    'goto_pose performs its own endpoint check, so an identical preliminary solve_ik is optional.')
            if s.name == 'plan_motion':
                s = replace(s, result_schema={**s.result_schema, 'x-attempt-history': 'failed_requests'})
            if s.side_effect in {SideEffect.ROBOT_MOTION, SideEffect.ACTUATOR} or s.name == 'plan_motion':
                # Same-episode history keeps these causal diagnostics when
                # bulky provenance cannot fit; current-result delivery is unchanged.
                properties = dict(s.result_schema.get('properties', {}))
                policy_schema = properties.get('collision_policy', {})
                properties['collision_policy'] = {**policy_schema, 'type': 'object',
                    'properties': {**policy_schema.get('properties', {}),
                        'carried_geometry': {'type': 'object', 'x-decision-feedback': 'on_demand'}},
                    'additionalProperties': True}
                for key in ('stop_evidence', 'controller_execution', 'endpoint_precision', 'collision_policy',
                            'public_effect', 'verification_requirements', 'target_motion_evidence',
                            'swept_scene_validation', 'planning_tracking_reserve', 'best_endpoint_nominal_collision',
                            'best_endpoint_collision', 'recovery_guidance', 'path_search', 'first_invalid_state',
                            'start_nominal_collision', 'start_validation', 'collision_evidence'):
                    properties[key] = {**properties.get(key, {}), 'x-attempt-feedback': 'priority'}
                s = replace(s, result_schema={**s.result_schema, 'properties': properties})
            specs[i] = s
        from .possession_tools import with_possession_contract
        from .grasp_preparation import with_grasp_preparation_contract
        from .closure_force_feedback import with_closure_force_contract
        specs = [with_closure_force_contract(with_grasp_preparation_contract(with_possession_contract(s)))
                 for s in specs]
    from .usage_instructions import SENSOR_MOTION_USAGE
    return replace(manifest, provider=manifest.provider + ('/sensor-motion-v10' if motion_tools else ''),
                   capabilities=tuple(specs),
                   usage_instructions=SENSOR_MOTION_USAGE if motion_tools else manifest.usage_instructions)


def goal_evidence_capability():
    handle = {'type':'string', 'pattern':'^artifact://v1/'}
    handles = {'type':'array', 'items':handle, 'maxItems':8, 'uniqueItems':True}
    properties = {'observation_handle':handle,
        'subject_detection_ids':{**handles, 'minItems':1},
        'claims':{'type':'array', 'items':{'type':'string','minLength':1,'maxLength':512},
                  'minItems':1,'maxItems':8,'uniqueItems':True},
        'previous_evidence_handle':{'type':['string','null'], 'pattern':'^artifact://v1/', 'default':None},
        'supporting_evidence_handles':{**handles, 'default':[]}}
    return CapabilitySpec('measure_goal_evidence',
        'Collect selected visible geometry for general, directional, device-state and compound task claims. '
        'Reports acquisition requirements and semantic verification questions, never task truth. '
        'Use two observations at separate times after the last configuration-changing action; '
        'pass the first evidence_handle as previous_evidence_handle. When omitted, an eligible earlier '
        'unloaded sample can be reused for the same exact claims and issued subjects with supported '
        'visual correspondence to the current selected masks. Inspect previous_evidence_resolution; '
        'an absent eligible sample leaves the acquisition requirement unmet. Earlier supporting handles carry '
        'historical measurements and raw image sources, not current confirmation. Release the gripper '
        'before final verification. The full original task must be visually checked independently of the claim list.',
        {'type':'object','properties':properties,
         'required':['observation_handle','subject_detection_ids','claims'], 'additionalProperties':False},
        {'type':'object','x-completion-evidence-contract':'generic_acquisition_v1',
         'properties':{'evidence_contract':{'const':'generic_acquisition_v1'},
             'evidence_handle':handle,'source_observation_handle':handle,
             'verification_requirements':{'type':'array','minItems':1,'items':{'type':'object'}},
             'verification_questions':{'type':'array','minItems':1,'items':{'type':'object'}},
             'visual_evidence_sources':{'type':'array','items':{'type':'object'}}},
         'required':['evidence_contract','evidence_handle','source_observation_handle',
                     'verification_requirements','verification_questions'], 'additionalProperties':True},
        {'positions':'meter','time':'second'}, {'geometry':'robot_base'}, 30., SideEffect.COMPUTE,
        argument_order=tuple(properties), control_semantic=ControlSemantic.COMPLETION_ASSERTION,
        real_robot_equivalent='Public images, measured depth, calibrated robot state and recorded sensor times')


def motion_capabilities():
    handle = {'type': 'string', 'pattern': '^artifact://v1/'}
    vector = {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}
    quat = {'type': 'array', 'items': {'type': 'number'}, 'minItems': 4, 'maxItems': 4}
    pose = {'type': 'object', 'properties': {'position_m': vector, 'quaternion_wxyz': quat},
            'required': ['position_m', 'quaternion_wxyz'], 'additionalProperties': True}
    candidate = {**pose, 'properties': {**pose['properties'], 'detection_id': handle,
                 'approach_direction_world': vector},
                 'required': [*pose['required'], 'detection_id', 'approach_direction_world']}
    def spec(name, description, properties, required, result, effect=SideEffect.COMPUTE, semantic=None):
        return CapabilitySpec(name=name, description=description,
            arguments_schema={'type': 'object', 'properties': properties, 'required': required,
                              'additionalProperties': False},
            argument_order=tuple(properties),
            result_schema={'type': 'object', 'properties': result, 'additionalProperties': True},
            return_type='SensorMotionEvidence', units={'positions': 'meter', 'joints': 'radian'},
            coordinate_frames={'poses': 'robot_base, explicit local end_effector or tool reference'},
            timeout_s=180., side_effect=effect, control_semantic=semantic,
            examples=(f'{name}(...)',), real_robot_equivalent='RGB-D, robot description, OMPL and measured controller execution')
    diagnostic = {'type': 'object', 'additionalProperties': True}
    return (
        spec('build_scene', 'Requires at least one calibrated depth camera; RGB-only is unavailable. '
             'Build a scene from available RGB-D cameras in one immutable observation. '
             'Remove only the calibrated robot geometry, voxelize measured surfaces, and retain '
             'source/time. Unobserved space is unknown; this is not a complete collision map.',
             {'observation_handle': handle}, ['observation_handle'],
             {'scene_handle': handle, 'point_count': {'type': 'integer'}, 'coverage': {'type': 'string'}}),
        spec('capture_carried_geometry', 'Capture a detached planning mesh from the current accepted '
             'attachment mark and calibrated public depth. Additional views use robot-filtered visible primary '
             'samples with measured depth agreement, then segmentation and primary-surface overlap checks; '
             'an off-image centroid does not discard an otherwise visible part. Inspect view_reconstruction. '
             'Requires an already verified attachment and the exact accepted current_point_handle. '
             'This query does not move, weld, open or close anything. It binds partial visible geometry '
             'to the present observation, joints, gripper and attachment evidence. Use build_scene from '
             'source_observation_handle, then pass carried_geometry_handle to plan_motion. Any movement, '
             'time advance, release, loss or new attachment assertion invalidates the captured binding. '
             'The partial convex hull is conservative over measured points; unseen surfaces and semantic '
             'segmentation mistakes remain unknown. Query completion does not authorize motion.',
             {'tracking_handle': handle, 'current_point_handle': handle,
              'use_multiview': {'type': 'boolean', 'default': True}},
             ['tracking_handle', 'current_point_handle'],
             {'carried_geometry_handle': handle, 'source_observation_handle': handle,
              'motion_authorized': {'const': False}, 'selected_point_count': {'type': 'integer'}}),
        spec('evaluate_grasp', 'Evaluate one candidate against a fresh scene and its selected target: '
             'visible finger-pad support and sampled open-hand approach collisions. '
             'No movement, no force-closure proof, no full-arm path check. Inspect the evidence before aligning.',
             {'candidate': candidate, 'scene_handle': handle}, ['candidate', 'scene_handle'],
             {'contact': diagnostic, 'approach': diagnostic}),
        spec('refine_grasp', 'Compare the original candidate with visible target centering and '
             'public support-plane aligned top and lateral approaches along both visible principal axes. '
             'Keep robot fingertips above the measured plane. Compare before/after contact and approach '
             'evidence and contact/pregrasp endpoint IK; none of these is a full-arm path or force-closure proof. '
             'Pregrasp posture collision is also checked using the same declared contact-target rule as planned alignment. '
             'If contact IK succeeds but pregrasp IK fails, evaluate at most three distance halvings '
             'down to the controller minimum; every proposal retains the same checks. '
             'This is an explicit new geometry proposal with no transferred native backend score '
             'or native placement binding; the model decides whether to execute it. '
             'To refresh a stale issued candidate, explicitly supply a newly selected target_detection_id '
             'and a current scene (or omit scene to build from that detection). It re-centers on the '
             'new visible target and creates a new track; identity across observations is not inferred. '
             'For relation in, placement_estimate can use a measured above-rim hover after '
             'public attachment verification. For on, measured public support geometry is also available ' 
             'after verified attachment; neither fallback certifies final placement or transfers a native score.',
             {'candidate': candidate, 'scene_handle': handle, 'target_detection_id': handle}, ['candidate', 'scene_handle'],
             {'candidate': diagnostic, 'translation_m': vector, 'before': diagnostic, 'after': diagnostic}),
        spec('plan_motion', 'Plan a full-arm free-space path using robot-only IK, self collision and '
             'public RGB-D surface collision with OMPL RRTConnect. The goal reference is explicit. '
             'A rejected measured start stops before IK/search. Inspect start_validation, collision_evidence '
             'and recovery_guidance; pixel witnesses locate geometric conflicts, not proven object identities. '
             'Only exact, densely checked paths produce plan handles. RRT detours may be shortened using ' 
             'the same state and swept observed-surface checks within the remaining planning budget; inspect path_optimization. '
             'Tree growth and final validation check between joint samples using bounded hardware geometry. '
             'When the robot declares a tracking allowance, IK and path checks reserve gripper clearance ' 
             'for that allowance; inspect planning_tracking_reserve. tracking_clearance_insufficient means ' 
             'a nominally clear pose lacks this allowance: choose another public pose or approach candidate. ' 
             'Self collision is still sampled; actual tracking and unobserved surfaces are not certified. ' 
             'endpoint_precision applies to the complete requested pose change (at most 10% residual ' 
             'for nonzero translation and rotation, capped at 3 mm and 0.03 rad). IK uses tighter residuals. ' 
             'timeout_s covers scene/model setup, IK, search, dense validation and shortening together, ' 
             'and an earlier episode/policy/tool deadline wins. Checks are cooperative between operations; ' 
             'late computation yields timeout evidence with no executable plan. Unknown geometry remains unknown. '
             'For held-object transport, first capture current verified public carried geometry and supply its '
             'handle. The carried mesh participates in self collision, scene collision, swept bounds and '
             'tracking reserves. Without fresh geometry, held-object planning remains unavailable.',
             {'target_pose': pose, 'position_reference': {'type': 'string', 'enum': ['tool','end_effector']},
              'scene_handle': handle, 'timeout_s': {'type': 'number', 'minimum': .1, 'maximum': 30., 'default': 5.},
              'carried_geometry_handle': handle},
             ['target_pose', 'position_reference', 'scene_handle'],
             {'plan_handle': handle, 'reason': {'type': 'string'}, 'waypoint_count': {'type': 'integer'},
              'endpoint_precision': diagnostic, 'path_optimization': diagnostic}),
        spec('execute_motion', 'Execute a plan only while its start state, gripper, controller and source '
             'scene and any carried-geometry/attachment binding are still current. Record every control-step joint state and stop on observed collision '
             'or path deviation. OSC uses Cartesian waypoints with a monitored joint posture, not a '
             'controller switch. Internal waypoints use 3 mm/0.03 rad path-following tolerances; ' 
             'the final measured pose must meet the complete command endpoint_precision and OSC settling ' 
             'within the original control budget. A joint-tolerance pass alone is insufficient. ' 
             'Inspect residuals and stop_evidence on failure. Execution does not certify unseen obstacles or task success.',
             {'plan_handle': handle}, ['plan_handle'],
             {'trajectory_handle': handle, 'reason': {'type': 'string'}, 'control_steps': {'type': 'integer'},
              'endpoint_precision': diagnostic, 'position_residual_m': {'type': 'number'},
              'orientation_residual_rad': {'type': 'number'}, 'reached': {'type': 'boolean'},
              'endpoint_settled': {'type': ['boolean','null']}, 'stop_evidence': diagnostic},
             SideEffect.ROBOT_MOTION, ControlSemantic.CARTESIAN_MOTION),
        spec('track_target', 'Track image features from a selected detection into a newer observation '
             'using forward/backward optical flow and measured depth when available. With RGB only, '
             'report image displacement including possible camera motion, never metric target translation. '
             'Report target displacement or '
             'insufficient evidence; never assert attachment or silently replace the target. '
             'Use the result to reobserve, relocalize or propose a corrected pose.',
             {'detection_id': handle, 'observation_handle': handle}, ['detection_id', 'observation_handle'],
             {'status': {'type': 'string'}, 'translation_m': {'type': ['array','null'], 'items': {'type': 'number'}},
              'tracking_evidence_handle': {'type': 'string'}}),
    )
