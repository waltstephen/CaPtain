"""Public pose-command stopping accuracy, independent of task or simulator state."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def motion_precision(start, goal, position_tolerance_m, orientation_tolerance_rad, *,
                     orientation_reference="commanded_change", orientation_start_rotation=None):
    """Cap residual at 10% of each nonzero requested change, without changing goals.

    Zero-change axes retain the configured hold tolerance. This is a stopping
    criterion, not a guarantee that a physical controller can converge to it.
    A recorded retreat follows historical absolute orientations; incidental
    measured drift is not a new precision rotation command.
    """
    if orientation_reference not in {"commanded_change", "recorded_pose", "compound_request"}:
        raise ValueError("unknown orientation reference")
    translation = np.asarray(goal[:3, 3]) - start[:3, 3]
    rotation = Rotation.from_matrix(goal[:3, :3] @ start[:3, :3].T).as_rotvec()
    reference_rotation = rotation
    if orientation_reference == "compound_request":
        reference = np.asarray(orientation_start_rotation, dtype=float)
        if reference.shape != (3, 3) or not np.isfinite(reference).all():
            raise ValueError("compound orientation reference must be a finite rotation matrix")
        reference_rotation = Rotation.from_matrix(goal[:3, :3] @ reference.T).as_rotvec()
    distance, angle = float(np.linalg.norm(translation)), float(np.linalg.norm(reference_rotation))
    result = {
        "requested_displacement_m": translation.tolist(),
        "requested_rotation_vector_rad": rotation.tolist(),
        "configured_position_tolerance_m": float(position_tolerance_m),
        "configured_orientation_tolerance_rad": float(orientation_tolerance_rad),
        "position_tolerance_m": min(float(position_tolerance_m), .1 * distance)
            if distance > 1e-10 else float(position_tolerance_m),
        "orientation_tolerance_rad": min(float(orientation_tolerance_rad), .1 * angle)
            if angle > 1e-10 and orientation_reference in {"commanded_change", "compound_request"}
            else float(orientation_tolerance_rad),
        "orientation_reference": orientation_reference,
        "precision_basis": ("configured absolute tolerance capped at 10% of each nonzero requested pose change"
            if orientation_reference == "commanded_change" else
            "translation capped at 10% of recorded displacement; recorded orientation uses configured absolute tolerance"),
    }
    if orientation_reference == "compound_request":
        result.update(precision_reference_rotation_vector_rad=reference_rotation.tolist(),
            precision_basis="translation capped at 10% of phase displacement; rotation capped at 10% of original compound request")
    return result
