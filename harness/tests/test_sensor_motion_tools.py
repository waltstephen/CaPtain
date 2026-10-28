"""Geometry/trajectory counterexamples, including stale evidence and visual loss."""
from types import SimpleNamespace
import numpy as np
import pytest
from jsonschema import Draft202012Validator

from captain_harness.adapters.robotics.model import RobotCollisionModel
from captain_harness.adapters.robotics.planning import plan_joint_path, SensorScene, MotionPlan
from captain_harness.adapters.capx.runtime_manifest import runtime_manifest
from captain_harness.adapters.capx.manifest import build_capx_tool_manifest
from captain_harness.adapters.capx.visual_tracking import track_rgbd
from test_tool_backend import _fixture


XML = '''<mujoco><worldbody><body name="base"><body name="arm">
<joint name="x" type="slide" axis="1 0 0" range="-1 1" limited="true"/>
<geom name="arm_box" type="box" size=".05 .05 .05"/>
<body name="eef" pos="0 0 .15"><joint name="y" type="slide" axis="0 1 0" range="-1 1" limited="true"/>
<geom name="hand_box" type="box" size=".02 .02 .02"/></body>
</body></body></worldbody></mujoco>'''


def model():
    return RobotCollisionModel(XML, joints=['x','y'], gripper_joints=[],
        base='base', end_effector='eef', gripper_prefix='hand_')


def test_robot_description_fk_ik_and_convex_collision():
    robot = model()
    np.testing.assert_allclose(robot.eef_pose([.3,.2])[:3,3], [.3,.2,.15])
    target = np.eye(4); target[:3,3] = [-.2,.4,.15]
    solved = robot.solve_ik(target, [.3,.2])
    np.testing.assert_allclose(solved, [-.2,.4], atol=.002)
    assert robot.collision([0,0], 1., [[.01,.01,.15]])['collision']
    assert not robot.collision([.3,0], 1., [[.01,.01,.15]])['collision']
    # A line can collide even though its endpoints do not.
    assert not robot.collision([-.3,0], 1., [[0,0,.15]])['collision']


def test_detached_description_preserves_explicit_radian_joint_limits():
    xml = XML.replace('type="slide" axis="1 0 0" range="-1 1"',
                      'type="hinge" axis="0 0 1" range="-2.9 2.9"')
    robot = RobotCollisionModel(xml,joints=['x','y'],gripper_joints=[],base='base',
        end_effector='eef',gripper_prefix='hand_',angle_unit='radian')
    np.testing.assert_allclose(robot.limits[0], [-2.9,2.9], atol=1e-10)


def test_rrt_finds_checked_detour_and_never_accepts_approximate_solution():
    valid = lambda q: not (abs(q[0]) < .2 and q[1] < .5)
    path, info = plan_joint_path([-.8,0], [.8,0], [[-1,1],[-1,1]], valid, timeout_s=1.)
    assert path is not None, info
    assert all(valid(q) for q in path)
    assert path[:,1].max() >= .5
    assert np.max(np.abs(np.diff(path, axis=0))) <= .025 + 1e-9
    np.testing.assert_allclose(path[-1], [.8,0], atol=1e-7)
    blocked, detail = plan_joint_path([-.8,0], [.8,0], [[-1,1],[-1,1]],
        lambda q: abs(q[0]) >= .2, timeout_s=.05)
    assert blocked is None and detail['reason'] in {'no_exact_path_within_budget','deadline_expired'}
    if detail['reason']=='deadline_expired':assert detail['deadline']['scope']=='path_planning'


def test_start_collision_produces_no_plan():
    path, info = plan_joint_path([0,0], [.8,0], [[-1,1],[-1,1]], lambda q: q[0] > .1)
    assert path is None and info['reason'] == 'start_in_observed_collision'


@pytest.mark.parametrize('change', ['joints','aperture','time','controller'])
def test_stale_plan_never_reaches_actuator_or_clears_alignment(tmp_path, change):
    from dataclasses import replace
    f = _fixture(tmp_path, dehardcoded=True, v15_protocol=True)
    f.robot.time_s = 0.
    state = f.robot.observation.robot_state
    scene = SensorScene(np.zeros((1,3)), 'observation', state.joint_positions_rad, 1., 0., .01, 0)
    sh = f.backend.registry.put('sensor_scene', scene)
    plan = MotionPlan(sh, np.array([[0.]*7,[.1]*7]), scene.joints, 1., f.robot.controller, np.eye(4), 0., 1)
    ph = f.backend.registry.put('motion_plan', plan)
    previous = object(); f.backend._aligned_pregrasp = previous
    f.robot.execute_joint_path = lambda *a: pytest.fail('stale plan reached the robot')
    if change == 'time': f.robot.time_s = .05
    elif change == 'controller': f.robot.controller = 'OSC_POSE'
    else:
        updated = replace(state, **({'joint_positions_rad': (.01,)*7} if change == 'joints' else {'gripper_open_fraction': .5}))
        f.robot.observation = replace(f.robot.observation, robot_state=updated)
    result = f.backend.invoke('execute_motion', {'plan_handle': ph})
    assert not result.success and not result.executed
    assert f.backend._aligned_pregrasp is previous


def test_expanded_manifest_retains_typed_pose_contract_and_accepts_new_candidate_origin():
    base = build_capx_tool_manifest('S3', joint_count=7, service_aligned=True, dehardcoded=True, v15_protocol=True)
    enhanced = runtime_manifest(base, controller='OSC_POSE', motion_tools=True)
    assert len(enhanced.names) == len(base.names) + 12
    assert set(('build_scene','evaluate_grasp','refine_grasp','plan_motion','execute_motion','track_target',
                'measure_goal_evidence','capture_carried_geometry')) <= set(enhanced.names)
    args = enhanced.require('align_pregrasp').arguments_schema['properties']
    assert 'scene_handle' in args
    assert 'public_geometry_refinement' in args['candidate']['properties']['backend']['enum']
    # Expanding a profile must not mutate the base interface or another experiment.
    assert 'public_geometry_refinement' not in base.require('align_pregrasp').arguments_schema['properties']['candidate']['properties']['backend']['enum']
    for spec in enhanced.capabilities:
        Draft202012Validator.check_schema(spec.arguments_schema)
        Draft202012Validator.check_schema(spec.result_schema)


def camera(image, depth=None):
    return SimpleNamespace(image=image, depth=np.full(image.shape[:2], .5) if depth is None else depth,
        intrinsics=np.array([[100.,0,64],[0,100.,64],[0,0,1.]]), extrinsics=np.eye(4))


def test_refined_candidate_close_result_survives_broker_validation(tmp_path):
    from test_tool_backend import _v9_selected_detection
    from captain_harness.ports.broker import ValidatedCapabilityBroker
    f = _fixture(tmp_path, dehardcoded=True, v15_protocol=True)
    observation, selected = _v9_selected_detection(f, v15_protocol=True)
    manifest = runtime_manifest(build_capx_tool_manifest('S3', joint_count=7,
        service_aligned=True, dehardcoded=True, v15_protocol=True),
        controller='OSC_POSE', motion_tools=True)
    broker = ValidatedCapabilityBroker(manifest, f.backend.invoke)
    hand = broker.invoke('gripper_state', {})
    grasps = broker.invoke('grasp_pose_estimate', {
        'observation_handle': observation['observation_handle'],
        'detection_id': selected['detection_id'], 'gripper_profile': hand.value['profile'],
        'backend': 'auto', 'depth_source': 'sensor', 'grasp_style': 'auto', 'top_k': 5})
    assert grasps.success, grasps.message
    # Geometry-generated poses have no calibrated native convergence score.
    candidate = {**grasps.value['candidates'][0], 'backend': 'public_geometry_refinement',
                 'backend_candidate': None, 'model_score': None, 'convergence_risk': 'unassessed',
                 'style_preference_score': None, 'request_style_match': 'unassessed'}
    # This fixture constructs a NEW geometry proposal, as refine_grasp does;
    # it must not claim the immutable native candidate's reference.
    candidate.pop('grasp_handle', None)
    candidate = f.backend._issue_grasp_candidate(candidate)
    aligned = broker.invoke('align_pregrasp', {'candidate': candidate})
    assert aligned.success, aligned.message
    closed = broker.invoke('descend_and_close', {})
    assert closed.success and closed.executed, closed.message
    assert closed.value['candidate_convergence_risk'] == 'unassessed'
    assert f.backend._attachment_attempt_id == 1


def test_target_tracking_measures_motion_and_reports_texture_loss():
    cv2 = pytest.importorskip('cv2')
    rng = np.random.default_rng(9)
    image = np.zeros((128,128,3), np.uint8)
    image[30:90,30:90] = rng.integers(0,255,(60,60,3),dtype=np.uint8)
    shifted = cv2.warpAffine(image, np.float32([[1,0,4],[0,1,0]]),(128,128))
    mask = np.zeros((128,128),bool); mask[30:90,30:90] = True
    result = track_rgbd(camera(image), camera(shifted), mask)
    assert result['status'] == 'supported'
    np.testing.assert_allclose(result['translation_m'], [.02,0,0], atol=.001)
    rgb_first, rgb_latest = camera(image), camera(shifted)
    rgb_first.depth = rgb_latest.depth = None
    rgb = track_rgbd(rgb_first, rgb_latest, mask)
    assert rgb['status'] == 'image_only' and rgb['translation_m'] is None
    np.testing.assert_allclose(rgb['image_displacement_px'], [4,0], atol=.1)
    blank = np.zeros_like(image)
    lost = track_rgbd(camera(blank), camera(blank), mask)
    assert lost['status'] == 'insufficient' and lost['translation_m'] is None
