"""Metric target correspondence separates object motion from camera motion."""
import numpy as np
from types import SimpleNamespace
from captain_harness.adapters.capx.visual_tracking import track_rgbd
from test_sensor_motion_tools import camera


def test_rigid_transform_maps_source_world_points_and_compensates_camera_motion():
    import cv2
    rng=np.random.default_rng(4);image=np.zeros((128,128,3),np.uint8)
    image[30:90,30:90]=rng.integers(0,255,(60,60,3),dtype=np.uint8)
    shifted=cv2.warpAffine(image,np.float32([[1,0,4],[0,1,0]]),(128,128))
    mask=np.zeros((128,128),bool);mask[30:90,30:90]=True
    first,latest=camera(image),camera(shifted)
    moving=track_rgbd(first,latest,mask)
    transform=np.asarray(moving['rigid_transform_source_to_current_world'])
    np.testing.assert_allclose(transform[:3,3],[.02,0,0],atol=.001)
    np.testing.assert_allclose(transform[:3,:3],np.eye(3),atol=.001)
    latest.extrinsics=latest.extrinsics.copy();latest.extrinsics[0,3]=-.02
    stationary=track_rgbd(first,latest,mask)
    np.testing.assert_allclose(stationary['rigid_transform_source_to_current_world'],np.eye(4),atol=.001)


def fixture():
    shape=(32,32);intrinsics=np.array([[100.,0,16],[0,100.,16],[0,0,1.]])
    def cam():return SimpleNamespace(image=np.zeros((*shape,3),np.uint8),depth=np.full(shape,.5),
        intrinsics=intrinsics.copy(),extrinsics=np.eye(4))
    first,latest=cam(),cam();mask=np.zeros(shape,bool);mask[8:18,8:18]=True
    forward=np.zeros((*shape,2),np.float32);forward[...,0]=4.
    backward=-forward.copy();transform=np.eye(4);transform[0,3]=.02
    fit={'status':'supported','rigid_transform_source_to_current_world':transform.tolist()}
    return first,latest,mask,fit,forward,backward


def apply(data):
    from captain_harness.adapters.capx.temporal_target_surface import corresponded_surface
    return corresponded_surface(*data)


def test_only_source_labelled_moving_pixels_are_recovered():
    data=fixture();mask,receipt=apply(data)
    expected=np.zeros((32,32),bool);expected[8:18,12:22]=True
    np.testing.assert_array_equal(mask,expected)
    assert receipt['accepted_current_pixels']==100
    # A nearby current surface outside the tracked target cannot be erased.
    assert not mask[10,22] and not mask[7,12]


import pytest
@pytest.mark.parametrize('defect',['occluding_depth','source_depth','round_trip','nan_flow','missing_depth',
    'different_identity','camera_only','out_of_frame'])
def test_unresolved_correspondence_cannot_extend_target_labels(defect):
    first,latest,mask,fit,forward,backward=fixture()
    if defect=='occluding_depth':latest.depth[:]=.54
    if defect=='source_depth':first.depth[:]=0.
    if defect=='round_trip':forward[...,0]+=2.
    if defect=='nan_flow':backward[:]=np.nan
    if defect=='missing_depth':latest.depth=None
    if defect=='different_identity':fit['association_status']='contradicted'
    if defect=='camera_only':
        latest.extrinsics[0,3]=-.02;fit['rigid_transform_source_to_current_world']=np.eye(4).tolist()
    if defect=='out_of_frame':backward[:]=-100.
    accepted,receipt=apply((first,latest,mask,fit,forward,backward))
    assert not accepted.any() and receipt['status']=='insufficient'


def test_one_occluded_pixel_does_not_erase_other_valid_surface_matches():
    data=fixture();data[1].depth[12,16]=.6
    accepted,receipt=apply(data)
    assert not accepted[12,16] and accepted.sum()==99
    assert receipt['rejected_depth_or_metric']==1


def test_full_flow_pipeline_recovers_a_translated_textured_target():
    import cv2
    from captain_harness.adapters.capx.temporal_target_surface import track_target_surface
    rng=np.random.default_rng(4);image=np.zeros((128,128,3),np.uint8)
    image[30:90,30:90]=rng.integers(0,255,(60,60,3),dtype=np.uint8)
    shifted=cv2.warpAffine(image,np.float32([[1,0,4],[0,1,0]]),(128,128))
    old=np.zeros((128,128),bool);old[30:90,30:90]=True
    allowed=np.zeros_like(old);allowed[30:90,34:94]=True
    recovered,receipt=track_target_surface(camera(image),camera(shifted),old)
    assert receipt['status']=='supported' and recovered.sum()>2500
    assert not (recovered & ~allowed).any()


def test_surface_query_honors_earlier_parent_deadline_before_work(monkeypatch):
    from captain_harness.ports import deadline as deadlines
    from captain_harness.adapters.capx.temporal_target_surface import track_target_surface
    now=[10.];monkeypatch.setattr(deadlines.time,'monotonic',lambda:now[0])
    first,latest,mask,*_=fixture()
    with deadlines.deadline_scope('episode',timeout_s=1.):
        now[0]+=2.
        with pytest.raises(deadlines.DeadlineExceeded):track_target_surface(first,latest,mask)
