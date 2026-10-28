"""Local support-plane estimation from measured scene points."""
import numpy as np
from scipy.spatial import cKDTree


def support_plane(scene_points, target_points, neighborhood_m):
    center = np.median(target_points, axis=0)
    extent = np.linalg.norm(np.ptp(target_points, axis=0)) / 2
    nearby = scene_points[np.linalg.norm(scene_points - center, axis=1) <= extent + neighborhood_m]
    # Exclude measured target samples only, not a bounding-box hole in the scene.
    if len(nearby):
        distances, _ = cKDTree(target_points).query(nearby)
        nearby = nearby[distances > .003]
    if len(nearby) < 30:
        return None
    if len(nearby) > 5000:
        nearby = nearby[np.linspace(0,len(nearby)-1,5000,dtype=int)]
    rng = np.random.default_rng(0)
    best = None
    for _ in range(80):
        a,b,c = nearby[rng.choice(len(nearby),3,replace=False)]
        normal = np.cross(b-a,c-a)
        length = np.linalg.norm(normal)
        if length < 1e-10: continue
        normal /= length
        offset = -float(a @ normal)
        mask = np.abs(nearby @ normal + offset) <= .003
        if best is None or mask.sum() > best.sum(): best = mask
    if best is None or best.sum() < 30: return None
    selected = nearby[best]
    point = selected.mean(axis=0)
    _,_,vh = np.linalg.svd(selected-point,full_matrices=False)
    normal = vh[-1]
    if (center-point) @ normal < 0: normal = -normal
    return normal, -float(point @ normal), int(best.sum())
