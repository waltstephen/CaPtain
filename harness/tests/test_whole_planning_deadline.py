"""No executable plan may hide IK or dense validation outside its deadline."""
from types import SimpleNamespace
import numpy as np
import pytest
from captain_harness.ports import deadline
from captain_harness.adapters.robotics import planning
from captain_harness.adapters.capx.artifacts import EpisodeArtifactRegistry
from captain_harness.adapters.capx.motion_tools import MotionToolMixin
from captain_harness.adapters.robotics.planning import SensorScene
from captain_harness.ports.capability import FailureCategory


def clock_at(monkeypatch):
    now=[0.];clock=SimpleNamespace(monotonic=lambda:now[0])
    monkeypatch.setattr(deadline,'time',clock);monkeypatch.setattr(planning,'time',clock)
    # Public planning uses the same monotonic clock as its inherited deadline.
    import captain_harness.adapters.capx.motion_tools as motion
    monkeypatch.setattr(motion,'time',clock,raising=False)
    return now


def test_direct_collision_checks_stop_when_whole_path_budget_expires(monkeypatch):
    now=clock_at(monkeypatch);seen=[]
    def valid(q):seen.append(q.copy());now[0]+=.1;return True
    path,info=planning.plan_joint_path([0.],[.5],[[-1,1]],valid,timeout_s=.25)
    assert path is None
    assert info['reason']=='deadline_expired'
    assert len(seen)==3
    assert info['checks']==3
    assert info['deadline']['scope']=='path_planning'


def test_parent_deadline_caps_solver_and_rejects_its_late_return(monkeypatch):
    now=clock_at(monkeypatch);seen=[];solver=[]
    def valid(q):seen.append(q.copy());now[0]+=.01;return abs(q[0])>=.2
    def candidate(*args,**kw):
        solver.append(kw['timeout_s']);now[0]=.3
        return np.array([[-.3],[.3]])
    monkeypatch.setattr(planning,'_rrt_candidate',candidate)
    with deadline.deadline_scope('episode',expires_s=.2):
        path,info=planning.plan_joint_path([-.3],[.3],[[-1,1]],valid,timeout_s=5.)
    assert path is None and info['reason']=='deadline_expired'
    assert len(solver)==1 and 0<solver[0]<=.2
    assert info['deadline']['scope']=='episode'
    assert len(seen)<30


def test_ik_checks_between_iterations(monkeypatch):
    from test_sensor_motion_tools import model
    robot=model();now=clock_at(monkeypatch);original=robot.eef_pose;calls=[]
    def measured(*a,**kw):calls.append(1);value=original(*a,**kw);now[0]+=.1;return value
    robot.eef_pose=measured
    goal=np.eye(4);goal[:3,3]=[.8,.8,.15]
    with deadline.deadline_scope('policy',expires_s=.25):
        with pytest.raises(deadline.DeadlineExceeded):robot.solve_ik(goal,[0.,0.])
    assert len(calls)==3
    assert robot.last_ik['position_residual_m']>0


def test_late_ik_collision_callback_cannot_accept_solution(monkeypatch):
    from test_sensor_motion_tools import model
    robot=model();target=robot.eef_pose([0.,0.]);now=clock_at(monkeypatch)
    def valid(q):now[0]=.5;return True
    with deadline.deadline_scope('tool',expires_s=.1):
        with pytest.raises(deadline.DeadlineExceeded):robot.solve_ik(target,[0.,0.],valid=valid)


def make_tool():
    tool=MotionToolMixin();tool.registry=EpisodeArtifactRegistry('planning-deadline',ttl_s=None)
    tool._active_attachment_handle=None
    scene=SensorScene(np.empty((0,3)),'public-view',(0.,),1.,0.,.01,0)
    tool._scene=lambda *a,**kw:scene
    tool._scene_collision_checker=lambda *a,**kw:(lambda q, extra_margin=0., first_hit_only=False:{'collision':False},{})
    tool._pose_mapping=lambda pose:(pose['position_m'],pose['quaternion_wxyz'])
    model=SimpleNamespace(limits=np.array([[-1.,1.]]),eef_pose=lambda *a:np.eye(4),last_ik={},
        max_point_motion=lambda d, **kw:float(np.abs(d).sum()))
    tool.robot=SimpleNamespace(controller='OSC_POSE',planning_model=lambda:model)
    return tool,model


def test_public_planning_does_not_start_path_search_after_late_ik(monkeypatch):
    now=clock_at(monkeypatch);tool,model=make_tool()
    def ik(*a,**kw):now[0]=.2;model.last_ik={'iterations':1};return np.array([.2])
    model.solve_ik=ik
    monkeypatch.setattr('captain_harness.adapters.capx.motion_tools.plan_joint_path',
        lambda *a,**kw:pytest.fail('path search started after IK exhausted the shared budget'))
    result=tool.plan_motion({'position_m':[.2,0,0],'quaternion_wxyz':[1,0,0,0]},'end_effector','scene',timeout_s=.1)
    assert not result.success and not result.executed
    assert result.failure_category is FailureCategory.TIMEOUT
    assert result.diagnostics['deadline']['scope']=='planning'
    assert result.value['planning_stage']=='ik'
    assert 'plan_handle' not in result.value


def test_ik_and_rrt_share_one_budget(monkeypatch):
    now=clock_at(monkeypatch);tool,model=make_tool();remaining=[]
    def ik(*a,**kw):now[0]=.07;return np.array([.2])
    def path(*a,**kw):
        remaining.append(kw['timeout_s']);return np.array([[0.],[.2]]),{'checks':2,'planning_time_s':.01,'reason':'direct_path'}
    model.solve_ik=ik
    monkeypatch.setattr('captain_harness.adapters.capx.motion_tools.plan_joint_path',path)
    result=tool.plan_motion({'position_m':[.2,0,0],'quaternion_wxyz':[1,0,0,0]},'end_effector','scene',timeout_s=.1)
    assert result.success
    assert len(remaining)==1 and 0<remaining[0]<=.030000001
    assert deadline.current_deadline() is None


def test_expired_parent_dispatches_no_scene_or_model_work(monkeypatch):
    clock_at(monkeypatch);tool,model=make_tool()
    tool._scene=lambda *a,**kw:pytest.fail('expired planning request read scene')
    with deadline.deadline_scope('episode',expires_s=-1.):
        result=tool.plan_motion({'position_m':[.2,0,0],'quaternion_wxyz':[1,0,0,0]},'end_effector','scene')
    assert not result.success and not result.executed
    assert result.diagnostics['deadline']['scope']=='episode'



def test_public_path_is_not_returned_after_shortening_expires(monkeypatch):
    from test_validated_path_shortening import START,GOAL,RAW,valid
    now=clock_at(monkeypatch);count=[0];started=[False]
    dense=planning.interpolate_path(RAW)
    def candidate(*a,**kw):started[0]=True;return RAW.copy()
    def measured(q):
        if started[0]:
            count[0]+=1
            if count[0]>len(dense):now[0]=2.
        return valid(q)
    monkeypatch.setattr(planning,'_rrt_candidate',candidate)
    path,info=planning.plan_joint_path(START,GOAL,[[-1,1],[-1,1]],measured,timeout_s=1.)
    assert path is None and info['reason']=='deadline_expired'
    assert info['planning_stage']=='path_shortening'
    assert info['validated_path_waypoints']==len(dense)
    assert count[0]==len(dense)+1


def test_optional_shortening_does_not_spend_the_whole_return_budget(monkeypatch):
    from test_validated_path_shortening import START,GOAL,RAW,valid,length
    now=clock_at(monkeypatch);count=[0];started=[False]
    dense=planning.interpolate_path(RAW)
    def candidate(*a,**kw):started[0]=True;return RAW.copy()
    def measured(q):
        if started[0]:
            count[0]+=1
            if count[0]==len(dense):now[0]=.4
            elif count[0]>len(dense):now[0]+=.1
        return valid(q)
    monkeypatch.setattr(planning,'_rrt_candidate',candidate)
    path,info=planning.plan_joint_path(START,GOAL,[[-1,1],[-1,1]],measured,timeout_s=1.)
    assert path is not None, 'optional optimization consumed a valid plan and the whole request budget'
    assert now[0]<1.
    assert info['reason']=='rrt_connect'
    assert info['path_optimization']['budget_exhausted']
    np.testing.assert_allclose(path[0],START);np.testing.assert_allclose(path[-1],GOAL)
    assert all(valid(q) for q in path)
    assert length(path)<=length(dense)+1e-10
    assert np.max(np.abs(np.diff(path,axis=0)))<=.025+1e-10
