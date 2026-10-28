import numpy as np
import pytest
from captain_harness.adapters.robotics import planning


def test_colliding_new_endpoint_stops_before_interior_work_or_sweep():
    seen=[]
    def checked(q):seen.append(q.copy());return q[0] < .5
    result=planning._sampled_motion_clear(np.array([0.]),np.array([1.]),checked,
        max_joint_step=.025,edge_valid=lambda *a:pytest.fail('nominal rejection must not sweep'))
    assert not result
    assert len(seen)==1 and seen[0][0]==1.


def test_clear_endpoint_then_middle_detects_interior_obstacle_early():
    seen=[]
    def checked(q):seen.append(q.copy());return abs(q[0]-.5)>.01
    assert not planning._sampled_motion_clear(np.array([0.]),np.array([1.]),checked,
        max_joint_step=.025,edge_valid=lambda *a:pytest.fail('nominal rejection must not sweep'))
    assert [q[0] for q in seen]==[1.,.5]


def test_clear_edge_checks_every_original_sample_and_requires_sweep():
    start=np.array([-.7,.2]);end=np.array([.4,-.5]);seen=[];sweeps=[]
    def checked(q):seen.append(q.copy());return True
    def sweep(a,b):sweeps.append((a.copy(),b.copy()));return False
    assert not planning._sampled_motion_clear(start,end,checked,max_joint_step=.025,edge_valid=sweep)
    expected=planning.interpolate_path([start,end],.025)[1:]
    assert len(seen)==len(expected)
    assert {tuple(q) for q in seen}=={tuple(q) for q in expected}
    assert len(sweeps)==1
    np.testing.assert_array_equal(sweeps[0][0],start)
    np.testing.assert_array_equal(sweeps[0][1],end)


def test_short_clear_edge_is_still_validated():
    seen=[]
    assert planning._sampled_motion_clear(np.array([0.]),np.array([.001]),
        lambda q:seen.append(q.copy()) or True,max_joint_step=.025,edge_valid=lambda *a:True)
    assert len(seen)==1 and seen[0][0]==pytest.approx(.001)
