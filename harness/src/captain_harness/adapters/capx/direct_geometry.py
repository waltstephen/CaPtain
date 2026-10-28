"""Public target geometry for model-specified direct commands."""
import numpy as np
from .backend import _filter_point_cloud_noise


def detection_handles(single=None, multiple=None):
    if single is not None and multiple is not None:
        raise ValueError("use singular or plural detection handles, not both")
    if multiple is None:
        return () if single is None else (single,)
    if not isinstance(multiple, (list, tuple)) or not multiple:
        raise ValueError("plural detection handles must be a nonempty sequence")
    if any(not isinstance(item, str) or not item for item in multiple):
        raise ValueError("detection handles must be nonempty strings")
    if len(set(multiple)) != len(multiple):
        raise ValueError("duplicate detection handles")
    return tuple(multiple)


def selected_geometry(tools, scene_handle, handles):
    """Do not select a mask or infer identity; every handle is model supplied."""
    if not handles:
        raise ValueError("selected geometry needs explicit detection handles")
    clouds = []
    counts = []
    scene = None
    for handle in handles:
        scene, raw = tools._selected_scene_target(handle, scene_handle)
        # Exactly the filter used by public segment centroid/bounds. Raw depth
        # outliers must not silently enlarge the execution payload.
        filtered = _filter_point_cloud_noise(raw)
        if len(filtered) < 3:
            raise ValueError("selected_public_geometry_unavailable: insufficient filtered target samples")
        clouds.append(filtered)
        counts.append({"detection_id": handle, "raw_points": len(raw), "filtered_points": len(filtered)})
    points = np.unique(np.concatenate(clouds), axis=0)
    return scene, points, {
        "source": "explicit model-selected contemporaneous public detections",
        "identity": "model_declared_same_object; not independently verified",
        "filter": "same neighborhood filter as public segment geometry",
        "detections": counts, "point_count": len(points)}
