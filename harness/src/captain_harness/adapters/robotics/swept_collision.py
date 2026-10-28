"""Adaptive observed-surface edge checks; independent of environment and policy."""
from copy import deepcopy
from collections.abc import Mapping
from numbers import Real
import numpy as np
from captain_harness.ports.deadline import check_deadline


class SweptSceneValidator:
    """Inflate midpoint geometry by a hardware point-motion upper bound.

    Clear inflated geometry covers the whole joint-linear interval for observed
    surfaces. Self collisions are sampled only; actual tracking and unseen space
    are not certified. Unresolved near-contact intervals conservatively reject.
    """
    def __init__(self, collision, motion_bound, *, minimum_bound_m=1e-5, maximum_inflation_m=None, cache_validated_edges=False):
        self.collision, self.motion_bound = collision, motion_bound
        if not np.isfinite(minimum_bound_m) or minimum_bound_m <= 0 or (maximum_inflation_m is not None and (
                not np.isfinite(maximum_inflation_m) or maximum_inflation_m < minimum_bound_m)):
            raise ValueError('swept check bounds must be finite, positive and ordered')
        # Opt in only for one immutable scene/aperture/model query scope.
        self._clear_edges = set() if cache_validated_edges else None
        self._clear_edge_cache_hits = 0
        self._first_rejection = None
        self._active_edge = None
        self.minimum_bound_m = minimum_bound_m
        self.maximum_inflation_m = maximum_inflation_m
        self.stats = {'intervals': 0, 'collision_checks': 0, 'certified_intervals': 0,
                      'colliding_midpoints': 0, 'unresolved_intervals': 0}

    def _collision(self, q, extra_margin=0.):
        check_deadline()
        self.stats['collision_checks'] += 1
        result = self.collision(q, extra_margin=extra_margin)
        check_deadline()
        return result

    def receipt(self):
        return {**self.stats, 'first_rejection': deepcopy(self._first_rejection),
            'clear_edge_cache_hits': self._clear_edge_cache_hits,
            'cached_clear_edges': len(self._clear_edges) if self._clear_edges is not None else 0,
            'scope': 'bounded robot sweep versus observed surfaces; self collision sampled; tracking and unseen geometry unproven'}

    def _remember_rejection(self, kind, middle, start, end, bound, collision):
        if self._first_rejection is not None:
            return
        self._first_rejection = {
            'kind': kind, 'joints': middle.tolist(),
            'edge_start_joints': self._active_edge[0].tolist(),
            'edge_end_joints': self._active_edge[1].tolist(),
            'interval_start_joints': start.tolist(), 'interval_end_joints': end.tolist(),
            'remaining_motion_bound_m': bound, 'collision_receipt': deepcopy(collision),
            'receipt_scope': 'Nominal query; planning can stop at first hit, so collision counts are not exhaustive.',
            'limitation': 'Rejects this proposed interval under the existing clearance policy; does not prove physical penetration or global infeasibility.'}

    def __call__(self, start, end):
        check_deadline()
        self._active_edge = (np.asarray(start, dtype=float), np.asarray(end, dtype=float))
        key = None
        if self._clear_edges is not None:
            key = tuple(sorted((q.shape, q.tobytes()) for q in self._active_edge))
            if key in self._clear_edges:
                self._clear_edge_cache_hits += 1
                check_deadline()
                return True
        clear = self._interval(*self._active_edge, 0)
        check_deadline()
        if clear and self._clear_edges is not None:
            self._clear_edges.add(key)
        return clear

    def _interval(self, start, end, depth):
        check_deadline()
        self.stats['intervals'] += 1
        middle = (start + end) / 2.
        inflation = self.motion_bound((end - start) / 2.)
        values = list(inflation.values()) if isinstance(inflation, Mapping) else [inflation]
        if any(isinstance(v, bool) or not isinstance(v, Real) or not np.isfinite(v) or v < 0 for v in values):
            raise ValueError('hardware point-motion bounds must be finite and nonnegative')
        bound = float(max(values, default=0.))
        # Callers with unbounded native geometry may request an inflation cap.
        # Production uses bounded batches and early boolean-hit queries instead.
        if (self.maximum_inflation_m is None or bound <= self.maximum_inflation_m) and not self._collision(middle, extra_margin=inflation)['collision']:
            self.stats['certified_intervals'] += 1
            return True
        nominal = self._collision(middle)
        if nominal['collision']:
            self.stats['colliding_midpoints'] += 1
            self._remember_rejection('nominal_collision', middle, start, end, bound, nominal)
            return False
        if bound <= self.minimum_bound_m or depth >= 32:
            self.stats['unresolved_intervals'] += 1
            self._remember_rejection('interval_unresolved', middle, start, end, bound, nominal)
            return False
        return self._interval(start, middle, depth + 1) and self._interval(middle, end, depth + 1)
