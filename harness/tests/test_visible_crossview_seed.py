"""A partial view may contain the target even when its centroid is outside."""
from types import SimpleNamespace
import numpy as np
import pytest


def fixture():
    camera=SimpleNamespace(image=np.zeros((32,32,3),np.uint8),depth=np.full((32,32),.5),
        intrinsics=np.array([[100.,0,16],[0,100.,16],[0,0,1.]]),extrinsics=np.eye(4))
    pixels=np.array([[x,y] for x in range(12,16) for y in range(12,16)])
    visible=np.column_stack([(pixels-16)*.005,np.full(len(pixels),.5)])
    points=np.vstack([visible,np.tile([3.,0,.5],(100,1))])
    return camera,points,pixels


def test_off_image_centroid_does_not_discard_visible_target_samples():
    from captain_harness.adapters.capx.visible_crossview import visible_seed
    camera,points,pixels=fixture()
    seed,receipt=visible_seed(points,camera)
    assert seed is not None and any(np.array_equal(seed,p) for p in pixels)
    assert receipt['distinct_supported_pixels']==16 and receipt['measured_overlap_samples']==16


@pytest.mark.parametrize('defect',['occluded','missing_depth','behind','duplicates','nonfinite'])
def test_unsupported_projection_never_requests_a_segmentation_seed(defect):
    from captain_harness.adapters.capx.visible_crossview import visible_seed
    camera,points,_=fixture()
    if defect=='occluded':camera.depth[:]=.4
    if defect=='missing_depth':camera.depth=None
    if defect=='behind':points[:,2]=-.5
    if defect=='duplicates':points=np.tile(points[0],(100,1))
    if defect=='nonfinite':points[:]=np.nan
    seed,receipt=visible_seed(points,camera)
    assert seed is None and receipt['status']=='insufficient'


def test_overlap_accepts_partial_view_but_rejects_disjoint_centroid_neighbor():
    from captain_harness.adapters.capx.visible_crossview import overlap_receipt
    _,points,_=fixture();candidate=points[:16]
    assert np.linalg.norm(points.mean(axis=0)-candidate.mean(axis=0))>.08
    assert overlap_receipt(points,candidate)['supported']
    assert not overlap_receipt(candidate,candidate+[0,0,.06])['supported']
