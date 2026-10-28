"""Public robot envelope conversion; never infer a missing coordinate frame."""
from itertools import product
import numpy as np

def envelope_in_eef(envelope):
    if not envelope:
        return None
    if (envelope.get("origin_frame")!="end_effector" or
        envelope.get("axes_frame") not in {"hand","end_effector"} or
        envelope.get("units")!="meter" or
        "rotation_end_effector_from_axes" not in envelope):
        return None
    rotation=np.asarray(envelope["rotation_end_effector_from_axes"],dtype=float)
    if (rotation.shape!=(3,3) or not np.isfinite(rotation).all() or
        not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6) or
        not np.isclose(np.linalg.det(rotation),1.,atol=1e-6)):
        raise ValueError("invalid declared pad-frame rotation")
    lower=np.asarray(envelope["pad_min_local_m"],dtype=float)
    upper=np.asarray(envelope["pad_max_local_m"],dtype=float)
    if lower.shape!=(3,) or upper.shape!=(3,) or not np.isfinite([lower,upper]).all() or np.any(lower>upper):
        raise ValueError("invalid declared pad envelope bounds")
    corners=np.array(list(product(*zip(lower,upper))))@rotation.T
    raw_axis=envelope.get("closing_axis_local")
    axis=None
    if raw_axis is not None:
        axis=np.asarray(raw_axis,dtype=float)
        if axis.shape!=(3,) or not np.isfinite(axis).all() or np.linalg.norm(axis)<1e-9:
            raise ValueError("invalid declared closing axis")
        axis=rotation@(axis/np.linalg.norm(axis))
    return {"origin_frame":"end_effector","axes_frame":"end_effector","units":"meter",
        "pad_min_local_m":corners.min(axis=0).tolist(),"pad_max_local_m":corners.max(axis=0).tolist(),
        "corners_local_m":corners.tolist(),"center_local_m":(rotation@((lower+upper)/2)).tolist(),
        "closing_axis_eef":axis.tolist() if axis is not None else None,
        "source_axes_frame":envelope["axes_frame"],"rotation_end_effector_from_source_axes":rotation.tolist(),
        "bounds_semantics":"axis-aligned bounds in EEF axes; corners retain the rotated source envelope",
        "basis":envelope.get("basis"),"scope":"robot pad envelope; not object contact or grasp success"}
