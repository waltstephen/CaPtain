import numpy as np
import pytest
from types import SimpleNamespace
from captain_harness.adapters.capx.direct_geometry import detection_handles, selected_geometry
from captain_harness.adapters.capx.backend import _filter_point_cloud_noise


def surface():
    values=np.linspace(-.004,.004,7)
    return np.array([[x,y,z] for x in values for y in values for z in [-.002,.002]])


def test_execution_geometry_matches_public_filtered_geometry_and_rejects_outlier():
    dense=surface();raw=np.concatenate([dense,[[.3,.3,.3]]])
    scene=object()
    tools=SimpleNamespace(_selected_scene_target=lambda handle,source:(scene,raw))
    actual,points,receipt=selected_geometry(tools,"scene",("target",))
    assert actual is scene
    np.testing.assert_array_equal(points,np.unique(_filter_point_cloud_noise(raw),axis=0))
    assert points.max()<.01
    assert receipt["detections"][0]["raw_points"]==len(raw)
    assert receipt["detections"][0]["filtered_points"]==len(dense)


def test_only_explicit_views_are_unioned_and_each_is_freshness_checked():
    a=surface();b=surface()+[.015,0,0];calls=[]
    def resolve(handle,source):
        calls.append((handle,source))
        if handle=="stale":raise ValueError("selected target and current scene differ")
        return "scene",{"head":a,"wrist":b}[handle]
    tools=SimpleNamespace(_selected_scene_target=resolve)
    _,points,_=selected_geometry(tools,"snapshot",("head","wrist"))
    np.testing.assert_array_equal(points,np.unique(np.concatenate([a,b]),axis=0))
    assert calls==[("head","snapshot"),("wrist","snapshot")]
    with pytest.raises(ValueError,match="scene differ"):
        selected_geometry(tools,"snapshot",("head","stale"))


@pytest.mark.parametrize("single,multiple",[("one",["two"]),(None,[]),(None,["one","one"]),(None,[3]),(None,"one")])
def test_conflicting_or_ambiguous_handle_arguments_reject(single,multiple):
    with pytest.raises(ValueError):detection_handles(single,multiple)


def test_geometry_failure_does_not_fall_back_to_unfiltered_samples():
    tools=SimpleNamespace(_selected_scene_target=lambda *args:("scene",np.array([[0,0,0],[1,0,0],[0,1,0]])))
    with pytest.raises(ValueError,match="insufficient filtered"):
        selected_geometry(tools,"scene",("target",))
