"""Explicit whole motion and internal interpolation have different precision roles."""
from types import SimpleNamespace
import numpy as np
import pytest
from captain_harness.adapters.capx.artifacts import EpisodeArtifactRegistry
from captain_harness.adapters.capx.motion_tools import MotionToolMixin
from captain_harness.adapters.robotics.planning import SensorScene
from captain_harness.ports.capability import CapabilityResult


@pytest.mark.parametrize('distance',[-.001,.001,-.000409,.000409])
def test_planned_ik_does_not_absorb_a_small_public_translation(distance):
    from test_sensor_motion_tools import model
    robot_model=model()
    tool=MotionToolMixin();tool.registry=EpisodeArtifactRegistry('fine-plan',ttl_s=None)
    tool._active_attachment_handle=None
    scene=SensorScene(np.empty((0,3)),'public-view',(0.,0.),1.,0.,.01,0)
    tool._scene=lambda *a,**kw:scene
    tool._scene_collision_checker=lambda *a,**kw:(lambda q, extra_margin=0., first_hit_only=False:{'collision':False},{})
    tool._pose_mapping=lambda pose:(pose['position_m'],pose['quaternion_wxyz'])
    tool.robot=SimpleNamespace(controller='OSC_POSE',planning_model=lambda:robot_model)
    start=robot_model.eef_pose(scene.joints,scene.aperture)
    target=start[:3,3]+[distance,0,0]
    result=tool.plan_motion({'position_m':target.tolist(),'quaternion_wxyz':[1,0,0,0]},'end_effector','scene')
    assert result.success,result.message
    path=np.array(result.value['planned_joint_path'])
    actual=robot_model.eef_pose(path[-1],scene.aperture)[:3,3]
    assert np.linalg.norm(actual-target)<=.1*abs(distance)
    assert not np.array_equal(path[0],path[-1])


@pytest.mark.parametrize('distance',[-.001,.001])
def test_planned_goto_no_movement_cannot_claim_small_translation(distance,tmp_path):
    from test_tool_backend import _fixture
    f=_fixture(tmp_path,dehardcoded=True,v15_protocol=True)
    state=f.backend.observe().value['robot_state']
    target=np.array(state['tool_position_m'])+[distance,0,0]
    f.backend._plan_motion=lambda *a:CapabilityResult.ok({'plan_handle':'artifact://v1/test/plan'})
    f.backend.execute_motion=lambda *a:CapabilityResult.ok({'control_steps':0})
    result=f.backend.goto_pose(target,state['tool_quaternion_wxyz'],position_reference='tool',scene_handle='scene')
    assert not result.success
    assert result.value['position_residual_m']>.1*abs(distance)
    assert result.value['position_tolerance_m']<=.1*abs(distance)+1e-12


def test_internal_path_waypoints_do_not_tighten_incidental_orientation_drift(tmp_path):
    from test_libero_adapter import FakeOscEnv,RotationLockedAxisOscEnv
    from captain_harness.adapters.libero import LiberoActionBackend
    class Drift(RotationLockedAxisOscEnv):
        def step(self,action):
            result=FakeOscEnv.step(self,action)
            self.yaw=.003
            return result
    env=Drift()
    robot=LiberoActionBackend(env,observation_dir=tmp_path,initial_observation=env.reset(),
        horizon=1000,controller='OSC_POSE',guarded_cartesian_motion=True)
    start=np.array(robot.get_robot_state().value.end_effector_position_m)
    def fk(q,aperture):
        pose=np.eye(4);pose[:3,3]=start+q[:3];return pose
    robot.planning_model=lambda:SimpleNamespace(eef_pose=fk)
    path=np.zeros((4,7));path[:,0]=[0,.01,.02,.03]
    result=robot.execute_joint_path(path,lambda q, extra_margin=0., first_hit_only=False:{'collision':False})
    assert result.success,result.message
    final=np.array(robot.get_robot_state().value.end_effector_position_m)
    assert np.linalg.norm(final-(start+[.03,0,0]))<=.003
    assert result.value['orientation_residual_rad']==pytest.approx(.003)


@pytest.mark.parametrize('angle',[-.004,.004])
def test_planned_ik_resolves_explicit_small_rotation(angle):
    from scipy.spatial.transform import Rotation
    from captain_harness.adapters.robotics.model import RobotCollisionModel
    xml = """<mujoco><worldbody><body name="base"><body name="eef">
      <joint name="yaw" type="hinge" axis="0 0 1" range="-1 1" limited="true"/>
      <geom name="hand" type="box" size=".02 .02 .02"/>
      </body></body></worldbody></mujoco>"""
    model=RobotCollisionModel(xml,joints=['yaw'],gripper_joints=[],base='base',
        end_effector='eef',gripper_prefix='hand',angle_unit='radian')
    tool=MotionToolMixin();tool.registry=EpisodeArtifactRegistry('rotate-plan',ttl_s=None)
    tool._active_attachment_handle=None
    scene=SensorScene(np.empty((0,3)),'public-view',(0.,),1.,0.,.01,0)
    tool._scene=lambda *a,**kw:scene
    tool._scene_collision_checker=lambda *a,**kw:(lambda q, extra_margin=0., first_hit_only=False:{'collision':False},{})
    tool._pose_mapping=lambda pose:(pose['position_m'],pose['quaternion_wxyz'])
    tool.robot=SimpleNamespace(controller='OSC_POSE',planning_model=lambda:model)
    target=Rotation.from_rotvec([0,0,angle])
    result=tool.plan_motion({'position_m':[0,0,0],
        'quaternion_wxyz':np.roll(target.as_quat(),1).tolist()},'end_effector','scene')
    assert result.success,result.message
    path=np.array(result.value['planned_joint_path'])
    achieved=Rotation.from_matrix(model.eef_pose(path[-1])[:3,:3])
    assert (achieved*target.inv()).magnitude()<=.1*abs(angle)
    assert abs(path[-1,0])>=.9*abs(angle)


@pytest.mark.parametrize('angle',[-.004,.004])
def test_planned_goto_checks_measured_rotation_even_if_executor_claims_success(angle,tmp_path):
    from scipy.spatial.transform import Rotation
    from test_tool_backend import _fixture
    f=_fixture(tmp_path,dehardcoded=True,v15_protocol=True)
    state=f.backend.observe().value['robot_state']
    start=Rotation.from_quat(np.roll(state['tool_quaternion_wxyz'],-1))
    target=Rotation.from_rotvec([0,0,angle])*start
    f.backend._plan_motion=lambda *a:CapabilityResult.ok({'plan_handle':'artifact://v1/test/plan'})
    f.backend.execute_motion=lambda *a:CapabilityResult.ok({'control_steps':0,
        'orientation_residual_rad':0.,'orientation_tolerance_rad':.03,'converged':True})
    result=f.backend.goto_pose(state['tool_position_m'],np.roll(target.as_quat(),1).tolist(),
        position_reference='tool',scene_handle='scene')
    assert not result.success
    assert result.value['orientation_residual_rad']==pytest.approx(abs(angle))
    assert result.value['orientation_tolerance_rad']==pytest.approx(.1*abs(angle))
    assert result.value['converged'] is False


def test_joint_position_completion_checks_actual_cartesian_endpoint(tmp_path):
    from test_libero_adapter import FakeEnv
    from captain_harness.adapters.libero import LiberoActionBackend
    class Locked(FakeEnv):
        def step(self,action):
            self.actions.append(np.asarray(action).copy())
            return self._get_observations(),0.,False,{}
    env=Locked();env.robots[0].controller=SimpleNamespace()
    robot=LiberoActionBackend(env,observation_dir=tmp_path,initial_observation=env.reset(),
        horizon=1000,controller='JOINT_POSITION')
    start=np.array(robot.get_robot_state().value.end_effector_position_m)
    def fk(q,aperture):
        pose=np.eye(4);pose[:3,3]=start+q[:3];return pose
    robot.planning_model=lambda:SimpleNamespace(eef_pose=fk)
    path=np.zeros((2,7));path[-1,0]=.001
    result=robot.execute_joint_path(path,lambda q, extra_margin=0., first_hit_only=False:{'collision':False})
    assert not result.success
    assert result.value['reason']=='planned_endpoint_not_reached'
    assert result.value['position_residual_m']>result.value['position_tolerance_m']
    assert result.value['reached'] is False
    assert result.value['control_steps']==len(env.actions)
    assert result.executed==bool(env.actions)


@pytest.mark.parametrize('angle',[-.004,.004])
def test_osc_planned_endpoint_does_not_accept_locked_small_rotation(angle,tmp_path):
    from scipy.spatial.transform import Rotation
    from test_libero_adapter import FakeOscEnv
    from captain_harness.adapters.libero import LiberoActionBackend
    env=FakeOscEnv()
    robot=LiberoActionBackend(env,observation_dir=tmp_path,initial_observation=env.reset(),
        horizon=1000,controller='OSC_POSE',guarded_cartesian_motion=True)
    start=np.array(robot.get_robot_state().value.end_effector_position_m)
    def fk(q,aperture):
        pose=np.eye(4);pose[:3,3]=start
        pose[:3,:3]=Rotation.from_rotvec([0,0,q[3]]).as_matrix();return pose
    robot.planning_model=lambda:SimpleNamespace(eef_pose=fk)
    path=np.zeros((2,7));path[-1,3]=angle
    result=robot.execute_joint_path(path,lambda q, extra_margin=0., first_hit_only=False:{'collision':False},endpoint_target=fk(path[-1],1.))
    assert not result.success and result.executed
    assert result.value['orientation_residual_rad']==pytest.approx(abs(angle))
    assert result.value['orientation_tolerance_rad']==pytest.approx(.1*abs(angle))
    assert result.value['reached'] is False
    assert 0<result.value['control_steps']<=40
    assert len(env.actions)==result.value['control_steps']


@pytest.mark.parametrize('deadline_s',[0.,.25])
def test_planned_path_keeps_deadline_category_and_partial_receipt(tmp_path,monkeypatch,deadline_s):
    from captain_harness.ports.deadline import deadline_scope
    from captain_harness.ports.capability import FailureCategory
    from test_libero_adapter import FakeOscEnv
    from captain_harness.adapters.libero import LiberoActionBackend
    import captain_harness.ports.deadline as deadlines
    clock=[100.];monkeypatch.setattr(deadlines.time,'monotonic',lambda:clock[0])
    class Clocked(FakeOscEnv):
        def step(self,action):
            result=super().step(action);clock[0]+=.1;return result
    env=Clocked()
    robot=LiberoActionBackend(env,observation_dir=tmp_path,initial_observation=env.reset(),
        horizon=1000,controller='OSC_POSE',guarded_cartesian_motion=True)
    start=np.array(robot.get_robot_state().value.end_effector_position_m)
    def fk(q,aperture):
        pose=np.eye(4);pose[:3,3]=start+q[:3];return pose
    robot.planning_model=lambda:SimpleNamespace(eef_pose=fk)
    path=np.zeros((2,7));path[-1,0]=.04
    with deadline_scope('episode',expires_s=clock[0]+deadline_s):
        result=robot.execute_joint_path(path,lambda q, extra_margin=0., first_hit_only=False:{'collision':False})
    assert not result.success
    assert result.failure_category is FailureCategory.TIMEOUT
    assert result.diagnostics['deadline']['scope']=='episode'
    assert result.value['reason']=='deadline_expired'
    assert result.value['control_steps']==len(env.actions)==(3 if deadline_s else 0)
    assert result.executed==bool(env.actions)
    assert result.value['position_residual_m']>0


@pytest.mark.parametrize('distance',[0.,.001])
def test_planned_goto_preserves_nested_deadline_receipt(tmp_path,distance):
    from dataclasses import replace
    from captain_harness.ports.capability import FailureCategory
    from test_tool_backend import _fixture
    f=_fixture(tmp_path,dehardcoded=True,v15_protocol=True)
    state=f.backend.observe().value['robot_state']
    f.backend._plan_motion=lambda *a:CapabilityResult.ok({'plan_handle':'artifact://v1/test/plan'})
    timeout=replace(CapabilityResult.failed(FailureCategory.TIMEOUT,'deadline_expired',
        executed=True,value={'control_steps':3}),diagnostics={'deadline':{'scope':'episode','remaining_s':0.}})
    f.backend.execute_motion=lambda *a:timeout
    result=f.backend.goto_pose(np.array(state['tool_position_m'])+[distance,0,0],
        state['tool_quaternion_wxyz'],position_reference='tool',scene_handle='scene')
    assert not result.success and result.executed
    assert result.failure_category is FailureCategory.TIMEOUT
    assert result.diagnostics['deadline']==timeout.diagnostics['deadline']
    assert result.value['control_steps']==3
