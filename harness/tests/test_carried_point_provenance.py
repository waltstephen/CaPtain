"""A held-object mesh may not switch to a higher-score disjoint camera cloud."""
from types import SimpleNamespace
import numpy as np
import pytest
from captain_harness.adapters.capx.backend import ToolCapabilityBackend, MaskCandidate, _MarkedPoint
from captain_harness.adapters.capx.artifacts import EpisodeArtifactRegistry


def fixture(primary_index=0, secondary_depth=.56):
    def camera(name,depth):
        return SimpleNamespace(camera_name=name,image=np.zeros((12,12,3),dtype=np.uint8),
            depth=np.full((12,12),depth),intrinsics=np.array([[1000.,0,5.5],[0,1000.,5.5],[0,0,1.]]),
            extrinsics=np.eye(4))
    cameras=[camera('head',.5),camera('wrist',secondary_depth)]
    mask=np.ones((12,12),bool)
    tool=object.__new__(ToolCapabilityBackend);tool.registry=EpisodeArtifactRegistry('cloud',ttl_s=None)
    tool.services=SimpleNamespace(segment_point=lambda *a:[MaskCandidate(mask=mask,score=.9)])
    obs=SimpleNamespace(cameras=tuple(cameras))
    oh=tool.registry.put('observation',obs)
    mark=_MarkedPoint(observation_handle=oh,camera_name=cameras[primary_index].camera_name,
        rgb_handle='fixture-rgb',depth_handle='fixture-depth',x_norm=.5,y_norm=.5,pixel_xy=(6.,6.),
        position=np.array([0.,0.,.5]),label=None,mask=mask,segmentation_score=.5,
        mask_area_fraction=1.,depth_continuity=1.,geometry_diagonal_m=.01,
        appearance_rgb_mean=np.zeros(3),centroid=np.array([0.,0.,.5]),
        bounds_min=np.array([-.003,-.003,.5]),bounds_max=np.array([.003,.003,.5]),
        point_count=144,grounding_status='grounded')
    return tool,tool.registry.put('marked_point',mark)


def test_primary_replacement_is_rejected_only_for_bound_carried_capture():
    tool,point=fixture()
    # Existing unbound fusion prefers the higher-score disjoint secondary cloud.
    _,_,points,_=tool._points_from_mark(point,use_multiview=True)
    np.testing.assert_allclose(points[:,2],.56)
    mark,_,points,_=tool._points_from_mark(point,use_multiview=True,require_primary=True)
    np.testing.assert_allclose(points[:,2],.5)
    assert mark.views_used==1
    assert mark.carried_view_reconstruction[0]['status']=='insufficient'


def test_consistent_multiview_cloud_keeps_primary_samples():
    tool,point=fixture(secondary_depth=.502)
    _,_,points,_=tool._points_from_mark(point,use_multiview=True,require_primary=True)
    assert np.count_nonzero(np.isclose(points[:,2],.5))==144
    assert np.count_nonzero(np.isclose(points[:,2],.502))==144


def test_single_view_capture_respects_a_wrist_primary_mark():
    tool,point=fixture(primary_index=1)
    tool.services.segment_point=lambda *a:pytest.fail('single view invoked segmentation')
    _,_,points,_=tool._points_from_mark(point,use_multiview=False,require_primary=True)
    np.testing.assert_allclose(points[:,2],.56)


def test_unexpected_primary_replacement_still_rejects_before_capture():
    tool,point=fixture()
    tool._clouds_from_masks=lambda *a,**kw:(np.empty((0,3)),np.array([[1.,0,1.],[1.,1.,1.],[0.,1.,1.]]))
    with pytest.raises(ValueError,match='carried_geometry_primary_view_not_retained'):
        tool._points_from_mark(point,use_multiview=True,require_primary=True)


def test_robot_only_projected_samples_do_not_seed_a_secondary_segmentation():
    tool,point=fixture(secondary_depth=.502)
    tool.services.segment_point=lambda *a:pytest.fail('excluded robot points seeded another view')
    mark,_,points,_=tool._points_from_mark(point,use_multiview=True,require_primary=True,
        projection_filter=lambda p:np.empty((0,3)))
    assert mark.views_used==1 and len(points)==144
    assert mark.carried_view_reconstruction[0]['reason']=='no_finite_primary_points'


def test_bound_selection_pixel_provenance_is_immutable_and_separate_from_mesh():
    tool,point=fixture(secondary_depth=.502)
    mark=tool.registry.get(point,expected_kind='marked_point')
    original_mask=mark.mask.copy()
    mark,_,points,_=tool._points_from_mark(point,use_multiview=True,require_primary=True)
    saved=dict(mark.carried_selected_masks)
    np.testing.assert_array_equal(saved['head'],original_mask)
    assert not saved['head'].flags.writeable and not saved['wrist'].flags.writeable
    mark.mask[:]=False
    np.testing.assert_array_equal(saved['head'],original_mask)
    assert len(points)==288
