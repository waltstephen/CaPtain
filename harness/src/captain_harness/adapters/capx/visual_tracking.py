"""Image-feature continuity with RGB-D motion evidence, never an attachment oracle."""
import numpy as np


def track_rgbd(first, latest, mask, *, target_mask=None):
    import cv2
    missing = {'status': 'insufficient', 'translation_m': None, 'feature_count': 0}
    a, b = (cv2.cvtColor(c.image, cv2.COLOR_RGB2GRAY) for c in (first, latest))
    if a.shape != b.shape:
        return {**missing, 'reason': 'image_geometry_changed'}
    if np.asarray(mask).shape != a.shape:
        return {**missing, 'reason': 'source_mask_geometry_mismatch'}
    interior = cv2.erode(np.asarray(mask, dtype=np.uint8), np.ones((3, 3), np.uint8))
    points = cv2.goodFeaturesToTrack(a, maxCorners=120, qualityLevel=.01, minDistance=4, mask=interior)
    if points is None or len(points) < 6:
        return {**missing, 'reason': 'insufficient_target_texture'}
    moved, forward, _ = cv2.calcOpticalFlowPyrLK(a, b, points, None)
    if moved is None:
        return {**missing, 'reason': 'forward_tracking_failed'}
    back, backward, _ = cv2.calcOpticalFlowPyrLK(b, a, moved, None)
    if back is None:
        return {**missing, 'reason': 'backward_tracking_failed'}
    consistent = (forward.ravel() > 0) & (backward.ravel() > 0)
    consistent &= np.linalg.norm((back - points).reshape(-1, 2), axis=1) <= 1.
    association = {}
    if target_mask is not None:
        target_mask = np.asarray(target_mask, dtype=bool)
        xy = moved.reshape(-1, 2)
        finite = np.isfinite(xy).all(axis=1)
        pixels = np.rint(np.where(np.isfinite(xy), xy, 0)).astype(int)
        h, w = b.shape
        valid = consistent & finite & (pixels[:,0]>=0) & (pixels[:,0]<w) & (pixels[:,1]>=0) & (pixels[:,1]<h)
        count = int(valid.sum())
        if target_mask.shape == b.shape and count >= 6:
            ratio = float(np.mean(target_mask[pixels[valid,1], pixels[valid,0]]))
            association = {'association_status': 'supported' if ratio >= .6 else 'contradicted' if ratio <= .2 else 'insufficient',
                'tracked_features_in_current_selection_ratio': ratio, 'association_feature_count': count}
        else:
            association = {'association_status':'insufficient', 'association_feature_count':count}
        missing.update(association)
    if first.depth is None or latest.depth is None:
        if consistent.sum() < 6:
            return {**missing, 'reason': 'correspondence_lost', 'feature_count': int(consistent.sum())}
        displacement = (moved - points).reshape(-1, 2)[consistent]
        return {**association, 'status': 'image_only', 'translation_m': None,
                'image_displacement_px': np.median(displacement, axis=0).tolist(),
                'feature_count': int(consistent.sum()), 'reason': 'measured_depth_unavailable',
                'limitation': 'Image displacement includes camera motion; no metric object motion or attachment claim.'}
    def lift(camera, xy):
        xy = xy.reshape(-1, 2)
        h, w = camera.depth.squeeze().shape
        finite = np.isfinite(xy).all(axis=1)
        clean = np.where(np.isfinite(xy), xy, 0.)
        pixel = np.rint(clean).astype(int)
        valid = finite & (pixel[:, 0] >= 0) & (pixel[:, 0] < w) & (pixel[:, 1] >= 0) & (pixel[:, 1] < h)
        pixel[:, 0] = np.clip(pixel[:, 0], 0, w - 1); pixel[:, 1] = np.clip(pixel[:, 1], 0, h - 1)
        depth = camera.depth.squeeze()[pixel[:, 1], pixel[:, 0]]
        valid &= np.isfinite(depth) & (depth > 0)
        rays = np.linalg.solve(camera.intrinsics, np.c_[clean, np.ones(len(clean))].T).T
        world = (rays * depth[:, None]) @ camera.extrinsics[:3, :3].T + camera.extrinsics[:3, 3]
        return world, valid
    old, va = lift(first, points)
    new, vb = lift(latest, moved)
    valid = consistent & va & vb
    if valid.sum() < 6:
        return {**missing, 'reason': 'correspondence_or_depth_lost', 'feature_count': int(valid.sum())}
    deltas = new[valid] - old[valid]
    center = np.median(deltas, axis=0)
    deviations = np.linalg.norm(deltas - center, axis=1)
    inliers = deviations <= max(.003, 3. * float(np.median(deviations)))
    if inliers.sum() < 6 or float(np.mean(inliers)) < .6:
        return {**missing, 'reason': 'inconsistent_target_motion', 'feature_count': int(valid.sum())}
    # Rigid feature motion is optional; collinear or inconsistent visible
    # points cannot support a mechanism rotation estimate.
    from scipy.spatial.transform import Rotation
    aa, bb = old[valid][inliers], new[valid][inliers]
    ca, cb = aa.mean(axis=0), bb.mean(axis=0)
    u, singular, vt = np.linalg.svd((aa-ca).T @ (bb-cb))
    fix = np.diag([1.,1.,float(np.linalg.det(vt.T @ u.T))])
    rotation = vt.T @ fix @ u.T
    residual = float(np.median(np.linalg.norm((aa-ca) @ rotation.T - (bb-cb), axis=1)))
    radius = float(np.sqrt(np.mean(np.sum((aa-ca)**2, axis=1))))
    rigid = bool(singular[1] > 1e-8 and radius > .005 and residual <= .003)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = cb - rotation @ ca
    indices=np.flatnonzero(valid)[inliers]
    shown=indices[np.linspace(0,len(indices)-1,min(24,len(indices))).astype(int)]
    correspondences=[{"source_pixel_xy":points.reshape(-1,2)[i].tolist(),
        "current_pixel_xy":moved.reshape(-1,2)[i].tolist(),
        "source_robot_base_m":old[i].tolist(),"current_robot_base_m":new[i].tolist()} for i in shown]
    return {**association, 'status': 'supported',
            'matched_public_features':correspondences,
            'rigid_transform_source_to_current_world': transform.tolist() if rigid else None,
            'rigid_fit_centers_world_m': {'source': ca.tolist(), 'current': cb.tolist()} if rigid else None,
            'translation_m': np.median(deltas[inliers], axis=0).tolist(),
            'feature_count': int(inliers.sum()), 'motion_spread_m': float(np.median(deviations[inliers])),
            'rotation_vector_rad': Rotation.from_matrix(rotation).as_rotvec().tolist() if rigid else None,
            'rigid_fit_residual_m': residual,
            'rotation_resolution_rad': max(.025,3*residual/radius) if rigid else None,
            'reason': 'forward_backward_features_with_measured_depth',
            'limitation': 'Partial feature motion; occlusion, deformation and similar appearance can invalidate identity.'}
