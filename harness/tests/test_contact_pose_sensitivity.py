import numpy as np
from captain_harness.adapters.capx.contact_geometry import pad_support


def test_nominal_overlap_can_disappear_under_declared_translation_error():
    points=np.array([[0,0,.0549],[.001,0,.0548]])
    r=pad_support(points,[-.01,-.04,.025],[.01,.04,.055],.003)['pose_sensitivity']
    assert r['nominal_interior_points']==2
    assert r['minimum_sampled_interior_points']==0
    assert r['some_sampled_poses_lose_all_visible_overlap'] is True
    assert r['force_closure_verified'] is False
    assert r['evaluated_poses']==13


def test_center_support_and_missing_geometry_are_distinct():
    a=pad_support([[0,0,.04]],[-.01,-.04,.025],[.01,.04,.055],.003)['pose_sensitivity']
    assert a['minimum_sampled_interior_points']==1
    b=pad_support([],[-.01,-.04,.025],[.01,.04,.055],.003)['pose_sensitivity']
    assert b['status']=='unknown'
    assert b['some_sampled_poses_lose_all_visible_overlap'] is None
