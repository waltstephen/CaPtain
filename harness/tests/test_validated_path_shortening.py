"""Shorter planning routes must retain sampled collision and budget evidence."""
from types import SimpleNamespace
import numpy as np
from captain_harness.adapters.robotics import planning

START=np.array([-.8,0.]);GOAL=np.array([.8,0.])
RAW=np.array([START,[-.8,.95],[-.6,.95],[-.6,.8],[.7,.8],[.7,.95],[.8,.95],GOAL])


def valid(q):return not(abs(q[0])<.2 and q[1]<.5)


def length(path):return np.linalg.norm(np.diff(path,axis=0),axis=1).sum()


def test_validated_rrt_detour_shortens_without_crossing_obstacle(monkeypatch):
    calls=[]
    def candidate(*args,**kwargs):calls.append(kwargs['timeout_s']);return RAW.copy()
    monkeypatch.setattr(planning,'_rrt_candidate',candidate)
    path,info=planning.plan_joint_path(START,GOAL,[[-1,1],[-1,1]],valid,timeout_s=1.)
    assert len(calls)==1
    assert length(path)<.85*length(RAW)
    np.testing.assert_allclose(path[0],START);np.testing.assert_allclose(path[-1],GOAL)
    assert all(valid(q) for q in path)
    assert np.max(np.abs(np.diff(path,axis=0)))<=.025+1e-10
    evidence=info['path_optimization']
    assert evidence['accepted_shortcuts']>0
    assert evidence['original_joint_length_rad']>=evidence['final_joint_length_rad']
    assert evidence['collision_policy']=='same_sampled_checker'


def test_expiring_shortcut_never_replaces_a_partially_checked_edge(monkeypatch):
    now=[0.];count=[0];dense=planning.interpolate_path(RAW)
    monkeypatch.setattr(planning,'time',SimpleNamespace(monotonic=lambda:now[0]))
    def checked(q):count[0]+=1;now[0]=2.;return valid(q)
    # Local shortening retains a validated fallback. Public deadline admission
    # is checked separately by test_whole_planning_deadline.
    path,info=planning._shorten_validated_path(dense,checked,expires_s=1.,max_joint_step=.025)
    np.testing.assert_allclose(path,dense)
    assert count[0]==1
    assert info['budget_exhausted'] and info['accepted_shortcuts']==0


def test_shortening_obeys_earlier_inherited_deadline(monkeypatch):
    from captain_harness.ports import deadline
    now=[0.];count=[0];dense=planning.interpolate_path(RAW)
    clock=SimpleNamespace(monotonic=lambda:now[0])
    monkeypatch.setattr(planning,'time',clock);monkeypatch.setattr(deadline,'time',clock)
    def checked(q):count[0]+=1;now[0]=.5;return valid(q)
    with deadline.deadline_scope('episode',timeout_s=.25):
        path,info=planning._shorten_validated_path(dense,checked,expires_s=10.,max_joint_step=.025)
    np.testing.assert_allclose(path,dense)
    assert count[0]==1
    assert info['budget_exhausted'] and info['accepted_shortcuts']==0
