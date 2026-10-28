import numpy as np
import pytest
from captain_harness.adapters.robotics import model as geometry
from captain_harness.ports import deadline


def test_batched_convex_membership_matches_halfspace_definition():
    from scipy.spatial import ConvexHull
    rng=np.random.default_rng(31)
    planes=ConvexHull(rng.normal(size=(80,3))).equations
    points=rng.normal(size=(2100,3))*2
    for margin in [0.,.003,.3]:
        reference=np.max(points@planes[:,:3].T+planes[:,3],axis=1)<=margin
        actual=geometry._convex_membership(points,planes,margin)
        np.testing.assert_array_equal(actual,reference)
        first=geometry._convex_membership(points,planes,margin,first_hit_only=True)
        assert bool(first.any())==bool(reference.any())
        assert not np.any(first & ~reference)


def test_large_convex_query_checks_deadline_between_bounded_batches(monkeypatch):
    now=[0.];calls=[]
    monkeypatch.setattr(deadline.time,'monotonic',lambda:now[0])
    original=np.einsum
    def dot(*a,**kw):
        calls.append(a[1].shape);result=original(*a,**kw);now[0]+=.03;return result
    monkeypatch.setattr(geometry.np,'einsum',dot)
    planes=np.column_stack([np.r_[np.eye(3),-np.eye(3)],np.full(6,-1.)])
    with deadline.deadline_scope('planning',timeout_s=.07):
        with pytest.raises(deadline.DeadlineExceeded):
            geometry._convex_membership(np.full((10000,3),.99),planes,0.)
    assert len(calls)==3 and max(s[0] for s in calls)<=256


def test_first_hit_collision_preserves_boolean_and_full_public_receipts():
    from test_sensor_motion_tools import model
    robot=model();points=np.array([[0.,0.,0.],[0.,0.,.15],[.5,0.,0.]])
    full=robot.collision([0.,0.],1.,points)
    quick=robot.collision([0.,0.],1.,points,first_hit_only=True)
    assert full['collision']==quick['collision']==True
    assert len(full['observed_collisions'])==2
    assert len(quick['observed_collisions'])==1
    assert robot.collision([0.,0.],1.,points)==full


def test_nonfinite_coordinates_cannot_be_classified_inside():
    planes=np.column_stack([np.r_[np.eye(3),-np.eye(3)],np.full(6,-1.)])
    points=np.array([[np.nan,0,0],[np.inf,0,0],[0,0,0]])
    with np.errstate(invalid='ignore'):
        np.testing.assert_array_equal(geometry._convex_membership(points,planes,0.),[False,False,True])


def test_first_hit_query_transforms_only_a_bounded_candidate_batch(monkeypatch):
    from test_sensor_motion_tools import model
    robot=model();sizes=[];original=geometry._convex_membership
    def membership(local,*a,**kw):sizes.append(len(local));return original(local,*a,**kw)
    monkeypatch.setattr(geometry,'_convex_membership',membership)
    result=robot.collision([0.,0.],1.,np.zeros((10000,3)),first_hit_only=True)
    assert result['collision'] and len(sizes)==1 and sizes[0]<=256


def test_public_planning_uses_boolean_queries_and_nominal_checker_keeps_full_receipts():
    from types import SimpleNamespace
    from test_sensor_motion_tools import model
    from captain_harness.adapters.capx.motion_tools import MotionToolMixin
    from captain_harness.adapters.capx.artifacts import EpisodeArtifactRegistry
    from captain_harness.adapters.robotics.planning import SensorScene
    robot=model();tool=MotionToolMixin();tool._active_attachment_handle=None
    tool.registry=EpisodeArtifactRegistry('boolean-planning',ttl_s=None)
    scene=SensorScene(np.empty((0,3)),'public',(0.,0.),1.,0.,.01,0)
    tool._scene=lambda *a,**kw:scene
    tool._pose_mapping=lambda pose:(pose['position_m'],pose['quaternion_wxyz'])
    tool.robot=SimpleNamespace(controller='OSC_POSE',planning_model=lambda:robot)
    flags=[];original=robot.collision
    def collision(*a,**kw):flags.append(kw.get('first_hit_only',False));return original(*a,**kw)
    robot.collision=collision
    result=tool.plan_motion({'position_m':[.1,0.,.15],'quaternion_wxyz':[1,0,0,0]},'end_effector','scene')
    assert result.success and flags and all(flags)
    check,_=tool._scene_collision_checker(robot,scene)
    check([0.,0.])
    assert flags[-1] is False


def test_inflated_convex_interior_can_prove_first_hit_without_plane_matrix(monkeypatch):
    planes=np.column_stack([np.r_[np.eye(3),-np.eye(3)],np.full(6,-.01)])
    monkeypatch.setattr(geometry.np,'einsum',lambda *a,**kw:pytest.fail('known interior hit performed plane matrix'))
    result=geometry._convex_membership(np.array([[.1,.1,.1],[1.,0.,0.]]),planes,.2,first_hit_only=True)
    assert result[0] and not result[1]


def test_coherent_outside_batch_rejects_on_existing_separating_plane(monkeypatch):
    planes=np.column_stack([np.r_[np.eye(3),-np.eye(3)],np.full(6,-.1)])
    monkeypatch.setattr(geometry.np,'einsum',lambda *a,**kw:pytest.fail('separated points entered full face matrix'))
    points=np.tile([.21,.02,.03],(900,1))
    result=geometry._convex_membership(points,planes,.003)
    assert not result.any()
