import numpy as np
import pytest
from captain_harness.adapters.capx.contact_geometry import pad_support

LO=np.array([-.008,-.04,-.012]); HI=np.array([.008,.04,.004])

def walls(one=False):
    x,z=np.meshgrid(np.linspace(-.006,.006,13),np.linspace(-.009,.001,11))
    a=np.c_[x.ravel(),np.full(x.size,-.018),z.ravel()]
    return a if one else np.r_[a,a+[0,.036,0]]

def test_dense_top_overlap_is_not_opposing_side_surface_support():
    x,y=np.meshgrid(np.linspace(-.006,.006,13),np.linspace(-.025,.025,51))
    a=np.c_[x.ravel(),y.ravel(),np.full(x.size,-.004)]
    result=pad_support(a,LO,HI,.003)
    assert result['interior_points']>100
    q=result['surface_quality']
    assert q['opposing_surface_status']=='unverified'
    assert q['negative_side_normal_points']==q['positive_side_normal_points']==0
    assert q['force_closure_verified'] is False

def test_opposing_visible_walls_report_measured_span_and_depth_without_grasp_claim():
    q=pad_support(walls(),LO,HI,.003)['surface_quality']
    assert q['opposing_surface_status']=='opposing_visible_patches'
    assert q['closing_axis_span_m']==pytest.approx(.036)
    assert q['closing_axis_offset_m']==pytest.approx(0,abs=1e-9)
    assert q['normal_correspondence_cells']>0
    assert q['observed_depth_span_m']==pytest.approx(.01)
    assert q['force_closure_verified'] is False

@pytest.mark.parametrize('points',[walls(True),np.zeros((30,3)),np.full((10,3),np.nan),np.empty((0,3))])
def test_missing_or_degenerate_geometry_stays_unverified(points):
    q=pad_support(points,LO,HI,.003)['surface_quality']
    assert q['opposing_surface_status']=='unverified'
    assert q['force_closure_verified'] is False

def test_density_duplication_does_not_manufacture_surface_support():
    a=walls(True)
    q=pad_support(np.repeat(a,20,axis=0),LO,HI,.003)['surface_quality']
    assert q['opposing_surface_status']=='unverified'
    assert q['sampled_points']<=1024

def test_edge_and_offset_measurements_change_with_contact_region():
    q=pad_support(walls()+[0,.008,.002],LO,HI,.003)['surface_quality']
    assert q['closing_axis_offset_m']==pytest.approx(.008)
    assert q['depth_edge_reserve_m']==pytest.approx(.001)
