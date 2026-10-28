"""Retain distinct observation times when a paused scene is measured again."""
import math

import numpy as np


def compare_completion_samples(samples, *, observation_handle, centroid, time_s, released):
    """Return a two-time window and public measurement evidence, never success.

    Re-localizing a paused scene replaces only the latest sample. Older clocks
    cannot roll the window back; unknown clocks cannot prove temporal coverage.
    The owner must reset the window when attachment/release evidence expires.
    """
    point = np.asarray(centroid, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError('completion centroid must be a finite three-vector')
    time = float(time_s) if time_s is not None and math.isfinite(time_s) else None
    current = {'observation_handle': observation_handle, 'centroid': point.tolist(),
               'time_s': time, 'released': released is True}
    last = samples[-1] if samples else None
    reference = last
    if last is None:
        updated = [current]
    elif time is None or last['time_s'] is None:
        updated = [current]
    elif time < last['time_s']:
        updated = list(samples)
    elif time == last['time_s']:
        if current['released'] != last['released']:
            reference = None
            updated = [current]
        else:
            reference = samples[-2] if len(samples) > 1 else last
            updated = [*samples[:-1], current]
    else:
        updated = [last, current]
    delta = (time-reference['time_s'] if reference is not None
             and time is not None and reference['time_s'] is not None else None)
    displacement = (float(np.linalg.norm(point-reference['centroid'])) if reference is not None
                    and reference['observation_handle'] != observation_handle else None)
    return updated, {
        'comparison_time_delta_s': delta,
        'temporal_comparison_available': bool(delta is not None and delta > 0
            and current['released'] and reference['released'] and displacement is not None),
        'target_static_after_release_m': displacement,
        'comparison_source_observation_handle': reference['observation_handle'] if reference else None,
        'comparison_observation_time_s': reference['time_s'] if reference else None,
    }
