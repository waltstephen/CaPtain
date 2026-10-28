"""Locate observed collision witnesses without asserting semantic identity."""
import numpy as np
from captain_harness.ports.deadline import check_deadline
from captain_harness.adapters.robotics.carried_geometry import partition_measured_target
from .artifacts import ArtifactHandleError


def project_witnesses(points, cameras, selected_masks=()):
    """Bounded exact RGB-D source locations; projection alone is insufficient."""
    points=np.asarray(points,dtype=float).reshape(-1,3)
    if not np.isfinite(points).all():raise ValueError('witness points must be finite')
    mapped=np.zeros(len(points),bool);views=[];masks=dict(selected_masks)
    for camera in cameras:
        check_deadline()
        depth=getattr(camera,'depth',None)
        if depth is None:
            views.append({'camera_name':camera.camera_name,'status':'depth_unavailable'})
            continue
        depth=np.asarray(depth).squeeze();h,w=depth.shape
        ext=np.asarray(camera.extrinsics);intrinsics=np.asarray(camera.intrinsics)
        local=(points-ext[:3,3])@ext[:3,:3]
        image=local@intrinsics.T
        xy=np.divide(image[:,:2],image[:,2:3],out=np.zeros_like(image[:,:2]),where=image[:,2:3]!=0)
        inside=np.isfinite(xy).all(axis=1)&(local[:,2]>1e-6)
        inside&=(xy[:,0]>=0)&(xy[:,0]<=w-1)&(xy[:,1]>=0)&(xy[:,1]<=h-1)
        indices=np.flatnonzero(inside);pixels=np.rint(xy[inside]).astype(int)
        z=depth[pixels[:,1],pixels[:,0]]
        rays=np.linalg.solve(intrinsics,np.column_stack([pixels,np.ones(len(pixels))]).T).T
        reconstructed=(rays*z[:,None])@ext[:3,:3].T+ext[:3,3]
        exact=np.isfinite(z)&(z>0)&(np.linalg.norm(reconstructed-points[indices],axis=1)<=1e-7)
        mapped[indices[exact]]=True;pixels=pixels[exact]
        unique=np.unique(pixels,axis=0)
        sample=(unique[np.linspace(0,len(unique)-1,min(8,len(unique)),dtype=int)]
                if len(unique) else np.empty((0,2),int))
        view={'camera_name':camera.camera_name,'status':'mapped' if len(unique) else 'no_exact_source_samples',
            'image_size_wh':[w,h],'exact_source_samples':int(exact.sum()),
            'distinct_pixels':len(unique),'sample_pixels_xy':sample.tolist(),
            'pixel_bounds_xy':[unique.min(axis=0).tolist(),unique.max(axis=0).tolist()] if len(unique) else None}
        if camera.camera_name in masks:
            mask=np.asarray(masks[camera.camera_name],bool)
            if mask.shape!=depth.shape:raise ValueError('selected mask and depth shapes differ')
            selected=mask[pixels[:,1],pixels[:,0]]
            view.update(inside_selected_mask=int(selected.sum()),outside_selected_mask=int((~selected).sum()))
        views.append(view)
    check_deadline()
    return {'witness_count':len(points),'mapped_witness_count':int(mapped.sum()),
        'unmapped_witness_count':int((~mapped).sum()),'cameras':views,
        'scope':'Exact measured source pixels of nominal collision samples, not physical contacts or object identities. Camera counts may overlap; selection membership does not authorize scene removal.'}


def carried_start_evidence(tool,model,scene,carried):
    check_deadline()
    other,_=partition_measured_target(scene.points,carried.points_world)
    model.set_state(scene.joints,scene.aperture)
    inside,hits=model.points_in_robot(other,margin=scene.margin_m,moving_only=True)
    check_deadline()
    base={'source_observation_handle':scene.source_observation_handle,'geometry_hits':hits,
        'observed_surface_margin_m':scene.margin_m,'scene_mutated':False,
        'motion_authorized':False,'evidence_frame':'robot_base / original camera pixels'}
    try:obs=tool.registry.get(scene.source_observation_handle,expected_kind='observation')
    except ArtifactHandleError as exc:return {**base,'status':'source_unavailable','reason':str(exc)}
    projected=project_witnesses(other[inside],getattr(obs,'cameras',()),carried.selected_masks)
    return {**base,'status':'measured',**projected}
