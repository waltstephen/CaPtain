"""Experimental positive RGB-D target associations; never a free-space oracle."""
import numpy as np
from captain_harness.ports.deadline import check_deadline


def corresponded_surface(first, latest, source_mask, fit, forward, backward):
    """Filter independently computed dense flow using sparse rigid target motion.

    No hull/box carve or mask dilation. Returned pixels are CURRENT measurements
    positively associated with previously selected pixels, under uncertain flow.
    """
    import cv2
    check_deadline()
    shape=latest.image.shape[:2]
    empty=np.zeros(shape,dtype=bool)
    base={'status':'insufficient','accepted_current_pixels':0,'metric_tolerance_m':.003,
        'forward_backward_tolerance_px':1.,'minimum_predicted_motion_m':.009,
        'limitation':'Public image/depth correspondence, not semantic identity or free-space proof.'}
    transform=fit.get('rigid_transform_source_to_current_world')
    if (fit.get('status')!='supported' or transform is None
            or fit.get('association_status')=='contradicted'):
        return empty,{**base,'reason':'rigid_target_correspondence_unavailable'}
    t=np.asarray(transform,dtype=float)
    if (t.shape!=(4,4) or not np.isfinite(t).all() or
            not np.allclose(t[3],[0,0,0,1],atol=1e-8,rtol=0) or
            not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-6,rtol=0) or
            not np.isclose(np.linalg.det(t[:3,:3]),1.,atol=1e-6)):
        raise ValueError('target transform must be finite and rigid')
    if first.depth is None or latest.depth is None:
        return empty,{**base,'reason':'measured_depth_unavailable'}
    if (first.image.shape[:2]!=shape or np.asarray(source_mask).shape!=shape
            or np.asarray(forward).shape!=(*shape,2) or np.asarray(backward).shape!=(*shape,2)
            or np.asarray(first.depth).squeeze().shape!=shape or np.asarray(latest.depth).squeeze().shape!=shape):
        raise ValueError('source/current pixel geometry must match')
    h,w=shape;yy,xx=np.indices(shape);grid=np.stack([xx,yy],axis=-1).astype(np.float32)
    source_xy=grid+np.asarray(backward,dtype=np.float32)
    finite=np.isfinite(source_xy).all(axis=-1)
    source_xy=np.where(np.isfinite(source_xy),source_xy,0.)
    inside=finite & (source_xy[...,0]>=0)&(source_xy[...,0]<=w-1)&(source_xy[...,1]>=0)&(source_xy[...,1]<=h-1)
    pixels=np.rint(source_xy).astype(int);pixels[...,0]=pixels[...,0].clip(0,w-1);pixels[...,1]=pixels[...,1].clip(0,h-1)
    forward_at_source=cv2.remap(np.asarray(forward,dtype=np.float32),source_xy[...,0],source_xy[...,1],cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,borderValue=float('nan'))
    round_trip=np.linalg.norm(np.asarray(backward)+forward_at_source,axis=-1)
    consistent=inside & np.isfinite(round_trip) & (round_trip<=1.)
    selected=consistent & np.asarray(source_mask,dtype=bool)[pixels[...,1],pixels[...,0]]
    check_deadline()
    def world(camera,xy,z):
        rays=np.linalg.solve(camera.intrinsics,np.column_stack([xy.reshape(-1,2),np.ones(h*w)]).T).T
        return (rays*z.reshape(-1,1))@camera.extrinsics[:3,:3].T+camera.extrinsics[:3,3]
    old_depth=np.asarray(first.depth).squeeze()[pixels[...,1],pixels[...,0]]
    new_depth=np.asarray(latest.depth).squeeze()
    valid_depth=np.isfinite(old_depth)&np.isfinite(new_depth)&(old_depth>0)&(new_depth>0)
    old=world(first,pixels,old_depth);new=world(latest,grid,new_depth)
    predicted=old@t[:3,:3].T+t[:3,3]
    residual=np.linalg.norm(predicted-new,axis=1).reshape(shape)
    movement=np.linalg.norm(predicted-old,axis=1).reshape(shape)
    accepted=selected & valid_depth & np.isfinite(residual)&(residual<=.003)&(movement>.009)
    check_deadline()
    return accepted,{**base,'status':'supported' if accepted.any() else 'insufficient',
        'reason':'positive_temporal_pixel_and_rigid_depth_correspondence' if accepted.any() else 'no_resolved_moving_target_samples',
        'consistent_selected_source_pixels':int(selected.sum()),'accepted_current_pixels':int(accepted.sum()),
        'rejected_depth_or_metric':int((selected & (~valid_depth|~np.isfinite(residual)|(residual>.003))).sum()),
        'rejected_unresolved_target_motion':int((selected & valid_depth & (movement<=.009)).sum()),
        'accepted_median_residual_m':float(np.median(residual[accepted])) if accepted.any() else None,
        'rigid_target_fit':fit}


def track_target_surface(first, latest, source_mask, *, current_mask=None):
    import cv2
    from .visual_tracking import track_rgbd
    check_deadline()
    fit=track_rgbd(first,latest,source_mask,target_mask=current_mask)
    check_deadline()
    if fit.get('status')!='supported' or fit.get('rigid_transform_source_to_current_world') is None:
        return np.zeros(latest.image.shape[:2],dtype=bool),{'status':'insufficient','reason':'rigid_target_correspondence_unavailable','rigid_target_fit':fit}
    a,b=(cv2.cvtColor(c.image,cv2.COLOR_RGB2GRAY) for c in (first,latest))
    forward=cv2.calcOpticalFlowFarneback(a,b,None,.5,3,21,3,5,1.2,0)
    check_deadline()
    backward=cv2.calcOpticalFlowFarneback(b,a,None,.5,3,21,3,5,1.2,0)
    check_deadline()
    return corresponded_surface(first,latest,source_mask,fit,forward,backward)


def reproject_target_surface(first, latest, source_mask, fit):
    """Geometric candidates only: inverse rigid pose, selected pixel and 3D depth.

    This does not establish RGB correspondence or authorize scene removal.
    It supplies a calibration-based initialization for large image displacement.
    """
    check_deadline()
    shape=latest.image.shape[:2];empty=np.zeros(shape,bool)
    transform=fit.get('rigid_transform_source_to_current_world')
    base={'status':'insufficient','scope':'geometric_correspondence_candidates_only',
        'metric_tolerance_m':.003,'minimum_predicted_motion_m':.009,'scene_removal_authorized':False}
    if fit.get('status')!='supported' or transform is None or fit.get('association_status')=='contradicted':
        return empty,None,{**base,'reason':'rigid_target_correspondence_unavailable'}
    if first.depth is None or latest.depth is None:
        return empty,None,{**base,'reason':'measured_depth_unavailable'}
    from captain_harness.adapters.robotics.carried_geometry import _pose
    t=_pose(transform)
    h,w=shape;yy,xx=np.indices(shape);xy=np.stack([xx,yy],axis=-1).reshape(-1,2)
    def lift(camera,pixels,depth):
        rays=np.linalg.solve(camera.intrinsics,np.column_stack([pixels,np.ones(len(pixels))]).T).T
        return (rays*depth.reshape(-1,1))@camera.extrinsics[:3,:3].T+camera.extrinsics[:3,3]
    depth=np.asarray(latest.depth).squeeze()
    current=lift(latest,xy,depth)
    predicted_source=(current-t[:3,3])@t[:3,:3]
    cam=(predicted_source-first.extrinsics[:3,3])@first.extrinsics[:3,:3]
    image=cam@first.intrinsics.T
    valid=np.isfinite(image).all(axis=1)&(cam[:,2]>1e-6)&np.isfinite(depth.ravel())&(depth.ravel()>0)
    source_xy=np.divide(image[:,:2],image[:,2:3],out=np.zeros_like(image[:,:2]),where=image[:,2:3]!=0)
    source_xy=np.where(np.isfinite(source_xy),source_xy,0.)
    sh,sw=first.image.shape[:2]
    valid&=(source_xy[:,0]>=0)&(source_xy[:,0]<=sw-1)&(source_xy[:,1]>=0)&(source_xy[:,1]<=sh-1)
    pixels=np.rint(source_xy).astype(int);pixels[:,0]=pixels[:,0].clip(0,sw-1);pixels[:,1]=pixels[:,1].clip(0,sh-1)
    source_depth=np.asarray(first.depth).squeeze()[pixels[:,1],pixels[:,0]]
    selected=valid&np.asarray(source_mask,bool)[pixels[:,1],pixels[:,0]]
    measured=lift(first,pixels,source_depth)
    current_prediction=measured@t[:3,:3].T+t[:3,3]
    residual=np.linalg.norm(current_prediction-current,axis=1)
    motion=np.linalg.norm(current_prediction-measured,axis=1)
    accepted=selected&np.isfinite(source_depth)&(source_depth>0)&np.isfinite(residual)&(residual<=.003)&(motion>.009)
    appearance=np.linalg.norm(first.image[pixels[:,1],pixels[:,0]].astype(float)-latest.image.reshape(-1,3),axis=1)
    check_deadline()
    return accepted.reshape(shape),source_xy.reshape(*shape,2),{**base,
        'status':'candidates' if accepted.any() else 'insufficient','selected_source_pixels':int(selected.sum()),
        'accepted_geometric_pixels':int(accepted.sum()),
        'median_metric_residual_m':float(np.median(residual[accepted])) if accepted.any() else None,
        'appearance_distance_quantiles_rgb':np.quantile(appearance[accepted],[0,.5,.9,1]).tolist() if accepted.any() else None}
