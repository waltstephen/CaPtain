"""Advisory footprint corrections from partial, public surface bounds."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def minimum_support_translation(center, extent, support_min, support_max, *, coverage=0.5):
    """Project a rectangular footprint onto the required overlap region.

    Return None when even the maximum visible overlap is insufficient. This
    describes observed bounding rectangles, not physical support or hidden shape.
    For overlap lengths u,v with u*v >= required area, the valid center intervals
    are [support_min - extent/2 + (u,v), support_max + extent/2 - (u,v)].
    Optimizing log(u) makes both squared projection distances convex.
    """
    center, extent, lower, upper = (np.asarray(x, dtype=float)[:2]
                                     for x in (center, extent, support_min, support_max))
    if (not all(np.all(np.isfinite(x)) for x in (center, extent, lower, upper))
            or np.any(extent <= 0) or np.any(upper <= lower) or not 0 < coverage <= 1):
        return None
    required = float(np.prod(extent) * coverage)
    maximum = np.minimum(extent, upper - lower)
    if float(np.prod(maximum)) < required:
        return None
    overlap = np.maximum(0, np.minimum(center + extent/2, upper)
                            - np.maximum(center - extent/2, lower))
    if float(np.prod(overlap)) >= required:
        return np.zeros(2)
    u_min, u_max = required / maximum[1], maximum[0]

    def correction(u):
        lengths = np.array([u, required/u])
        return np.clip(center, lower - extent/2 + lengths,
                       upper + extent/2 - lengths) - center

    candidates = [u_min, u_max]
    if u_min < u_max:
        solution = minimize_scalar(
            lambda log_u: float(np.sum(correction(np.exp(log_u))**2)),
            bounds=(np.log(u_min), np.log(u_max)), method='bounded',
            options={'xatol': 1e-13},
        )
        if not solution.success:
            return None
        candidates.append(float(np.exp(solution.x)))
    # Include zero-axis-displacement breakpoints, avoiding numerical drift.
    candidates.append(float(np.clip(overlap[0], u_min, u_max)))
    if overlap[1] > 0:
        candidates.append(float(np.clip(required/overlap[1], u_min, u_max)))
    return min((correction(u) for u in candidates), key=lambda d: float(d @ d))
