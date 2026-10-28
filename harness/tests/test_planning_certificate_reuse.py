import numpy as np
import pytest
from types import SimpleNamespace
from captain_harness.adapters.robotics.swept_collision import SweptSceneValidator
from captain_harness.adapters.robotics import planning
from captain_harness.adapters.robotics.planning import SensorScene
from captain_harness.ports.deadline import deadline_scope,DeadlineExceeded
from test_whole_planning_deadline import clock_at,make_tool


def test_interpolation_retains_original_endpoint_bits():
    start=np.array([.9,-.7]);end=np.array([.1,.13])
    path=planning.interpolate_path([start,end])
    np.testing.assert_array_equal(path[0],start)
    np.testing.assert_array_equal(path[-1],end)
    assert np.max(np.abs(np.diff(path,axis=0)))<=.025+1e-12


def test_only_fully_clear_exact_edges_are_reused_and_expiry_still_applies(monkeypatch):
    now=clock_at(monkeypatch);queries=[]
    def collision(q,extra_margin=0.):queries.append(q.copy());return {'collision':False}
    edge=SweptSceneValidator(collision,lambda d:float(np.abs(d).sum()),cache_validated_edges=True)
    assert edge([0.],[.1]);count=len(queries)
    assert edge([.1],[0.]) and len(queries)==count
    assert edge([0.],[.10000001]) and len(queries)>count
    with deadline_scope('expired',expires_s=-1.):
        with pytest.raises(DeadlineExceeded):edge([0.],[.1])
    assert edge.receipt()['clear_edge_cache_hits']==1


def test_rejected_edges_and_other_validator_instances_have_no_clear_certificate():
    queries=[]
    def collision(q,extra_margin=0.):queries.append(1);return {'collision':True}
    edge=SweptSceneValidator(collision,lambda d:float(np.abs(d).sum()),cache_validated_edges=True)
    assert not edge([0.],[.1]);count=len(queries)
    assert not edge([0.],[.1]) and len(queries)>count
    assert edge.receipt()['clear_edge_cache_hits']==0
    clear=SweptSceneValidator(lambda *a,**kw:{'collision':False},lambda d:0.,cache_validated_edges=True)
    assert clear([0.],[.1])
    assert not edge([0.],[.1])


def test_public_planner_reuses_rtt_checks_before_shared_deadline(monkeypatch):
    now=clock_at(monkeypatch);tool,model=make_tool()
    start=np.array([0.,0.]);goal=np.array([.02,0.]);raw=np.array([start,[0.,.02],[.02,.02],goal])
    scene=SensorScene(np.empty((0,3)),'public-view',tuple(start),1.,0.,.003,0)
    tool._scene=lambda *a,**kw:scene
    model.limits=np.array([[-.04,.04],[-.04,.04]])
    model.solve_ik=lambda *a,**kw:goal.copy()
    def collision(q,extra_margin=0.,first_hit_only=False):
        now[0]+=.01
        return {'collision':abs(q[0]-.005)<=.001+extra_margin and abs(q[1])<=.001+extra_margin}
    tool._scene_collision_checker=lambda *a,**kw:(collision,{'kind':'static_fixture'})
    def candidate(a,b,limits,checked,**kw):
        assert all(checked(q) for q in raw)
        assert all(kw['edge_valid'](a,b) for a,b in zip(raw[:-1],raw[1:]))
        now[0]=.99
        return raw.copy()
    monkeypatch.setattr(planning,'_rrt_candidate',candidate)
    monkeypatch.setattr(planning,'_shorten_validated_path',lambda path,*a,**kw:(path,{}))
    result=tool.plan_motion({'position_m':[.02,0,0],'quaternion_wxyz':[1,0,0,0]},'end_effector','scene',timeout_s=1.)
    assert result.success, result.value
    assert now[0]<1.
    assert result.value['collision_cache']['state_hits']>=4
    assert result.value['swept_scene_validation']['clear_edge_cache_hits']>=3


def test_generic_mutable_callbacks_do_not_reuse_edges_by_default():
    blocked=[False]
    edge=SweptSceneValidator(lambda *a,**kw:{'collision':blocked[0]},lambda d:0.)
    assert edge([0.],[.1])
    blocked[0]=True
    assert not edge([0.],[.1])


def test_public_planning_state_cache_is_new_for_each_request():
    tool,model=make_tool();blocked=[False];queries=[]
    model.solve_ik=lambda *a,**kw:np.array([.2])
    def collision(q,extra_margin=0.,first_hit_only=False):
        queries.append((np.asarray(q).copy(),blocked[0]));return {'collision':blocked[0]}
    tool._scene_collision_checker=lambda *a,**kw:(collision,{})
    target={'position_m':[.2,0,0],'quaternion_wxyz':[1,0,0,0]}
    assert tool.plan_motion(target,'end_effector','scene').success
    before=len(queries);blocked[0]=True
    result=tool.plan_motion(target,'end_effector','scene')
    assert not result.success and result.value['reason']=='start_in_observed_collision'
    assert len(queries)>before and queries[before][1] is True
