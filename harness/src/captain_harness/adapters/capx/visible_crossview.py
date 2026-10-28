"""Select only depth-supported visible samples for a partial secondary view."""
import numpy as np
from scipy.spatial import cKDTree
from captain_harness.ports.deadline import check_deadline


def _finite_points(points):
    points=np.asarray(points,dtype=float)
    if points.size==0:return np.empty((0,3))
    if points.ndim!=2 or points.shape[1]!=3:raise ValueError('points must be N by 3')
    return points[np.isfinite(points).all(axis=1)]


def visible_seed(primary_points,camera):
    check_deadline()
    receipt={'status':'insufficient','metric_tolerance_m':.003,'required_distinct_pixels':6,
        'seed_basis':'actual measured depth-supported pixel nearest the median visible projection',
        'scope':'calibrated visible surface correspondence; semantic identity unverified'}
    if camera.depth is None:return None,{**receipt,'reason':'secondary_depth_unavailable'}
    points=_finite_points(primary_points)
    if not len(points):return None,{**receipt,'reason':'no_finite_primary_points'}
    depth=np.asarray(camera.depth).squeeze();h,w=depth.shape
    ext=np.asarray(camera.extrinsics);intrinsics=np.asarray(camera.intrinsics)
    local=(points-ext[:3,3])@ext[:3,:3]
    good=local[:,2]>1e-6;points=points[good];local=local[good]
    uv=local@intrinsics.T;xy=uv[:,:2]/uv[:,2:3]
    good=np.isfinite(xy).all(axis=1)&(xy[:,0]>=0)&(xy[:,0]<=w-1)&(xy[:,1]>=0)&(xy[:,1]<=h-1)
    points=points[good];pixels=np.rint(xy[good]).astype(int)
    z=depth[pixels[:,1],pixels[:,0]]
    rays=np.linalg.solve(intrinsics,np.column_stack([pixels,np.ones(len(pixels))]).T).T
    measured=(rays*z[:,None])@ext[:3,:3].T+ext[:3,3]
    error=np.linalg.norm(measured-points,axis=1)
    supported=np.isfinite(z)&(z>0)&np.isfinite(error)&(error<=.003)
    unique=np.unique(pixels[supported],axis=0)
    receipt.update(projected_primary_samples=len(points),measured_overlap_samples=int(supported.sum()),
        distinct_supported_pixels=len(unique))
    check_deadline()
    if len(unique)<6:return None,{**receipt,'reason':'insufficient_visible_depth_correspondence'}
    median=np.median(unique,axis=0)
    seed=unique[np.argmin(np.linalg.norm(unique-median,axis=1))].tolist()
    return seed,{**receipt,'status':'supported','seed_pixel_xy':seed}


def overlap_receipt(primary_points,candidate_points):
    check_deadline()
    primary=_finite_points(primary_points);candidate=np.unique(_finite_points(candidate_points),axis=0)
    count=0
    if len(primary) and len(candidate):count=int((cKDTree(primary).query(candidate)[0]<=.003).sum())
    check_deadline()
    return {'supported':count>=6,'matched_candidate_samples':count,'candidate_samples':len(candidate),
        'metric_tolerance_m':.003,'required_samples':6,
        'scope':'partial visible surface overlap, not semantic identity proof'}
