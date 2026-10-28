import numpy as np
import pytest
from types import SimpleNamespace
from captain_harness.adapters.robotics.collision_release import MonotonicCollisionRelease

def hit(depth=0.,pair=('left','right')):
    return dict(collision=depth>0,observed_collisions={},
        self_collision_details=[dict(geometries=list(pair),penetration_m=depth)] if depth>0 else [])
def model(fn):return SimpleNamespace(collision=lambda q,a,p:fn(a))

def test_predicted_opening_monotonic_and_cleared_pair_cannot_reenter():
    guard=MonotonicCollisionRelease(model(lambda a:hit(max(.002-.01*a,0.))),[0],0.,.25)
    guard.observe(hit(.001),.1)
    with pytest.raises(ValueError,match='worsened'):guard.observe(hit(.0015),.15)
    guard.observe(hit(),.2)
    with pytest.raises(ValueError,match='reentered'):guard.observe(hit(.0001),.21)

@pytest.mark.parametrize('fn',[lambda a:hit(.001),lambda a:hit(.001+a*.01),
    lambda a:hit(.001,('new','pair') if a>.1 else ('left','right'))])
def test_no_path_when_opening_does_not_improve_or_adds_collision(fn):
    with pytest.raises(ValueError):MonotonicCollisionRelease(model(fn),[0],0.,.25)

def test_box_separation_is_rotation_invariant_and_preserves_real_overlap():
    from captain_harness.adapters.robotics.model import box_signed_separation
    from scipy.spatial.transform import Rotation
    half=np.array([.01,.02,.03])
    for rotation in [np.eye(3),Rotation.from_rotvec([.7,.8,-.2]).as_matrix()]:
        center=np.array([.4,.2,.7])
        for offset,expected in [(.02,0.),(.019,-.001),(.022,.002)]:
            other=center+rotation@np.array([offset,0,0])
            assert box_signed_separation(center,rotation,half,other,rotation,half)==pytest.approx(expected,abs=1e-12)
