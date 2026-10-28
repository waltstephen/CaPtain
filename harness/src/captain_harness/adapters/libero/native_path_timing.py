"""Traverse a measured joint polyline within one control-cycle travel budget."""
import numpy as np

from .joint_cartesian import _errors


def advance_reference(model, path, desired, index, aperture, *, translation_budget,
                      rotation_budget, joint_budget):
    command = np.asarray(desired, dtype=float).copy()
    limits = np.asarray([translation_budget, rotation_budget, joint_budget], dtype=float)
    if not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError('reference travel budgets must be finite and positive')
    remaining = limits.copy()
    crossed = 0
    while index < len(path):
        delta = path[index]-command
        if np.max(np.abs(delta)) < 1e-12:
            command = path[index].copy()
            index += 1
            crossed += 1
            continue
        pose = model.eef_pose(command, aperture)
        dp, da = _errors(pose, model.eef_pose(path[index], aperture))
        cost = np.asarray([dp, da, float(np.max(np.abs(delta)))])
        fractions = remaining/np.maximum(cost, 1e-15)
        fraction = min(1., float(np.min(fractions)))
        if fraction <= 1e-12:
            break
        # FK can curve along a joint interpolation. Recheck the actual partial
        # reference step rather than assuming proportional Cartesian distance.
        for _ in range(30):
            proposed = command+delta*fraction
            pe, ae = _errors(pose, model.eef_pose(proposed, aperture))
            used = np.asarray([pe, ae, float(np.max(np.abs(proposed-command)))])
            if np.all(used <= remaining+1e-12):
                break
            fraction *= .5
        else:
            raise ValueError('reference interpolation could not respect travel budgets')
        command = proposed
        remaining = np.maximum(0., remaining-used)
        if fraction >= 1.-1e-12:
            command = path[index].copy()
            index += 1
            crossed += 1
        else:
            break
    consumed = limits-remaining
    return command, index, {'traversed_waypoint_count': crossed,
        'reference_waypoint_index': index, 'reference_translation_m': float(consumed[0]),
        'reference_rotation_rad': float(consumed[1]),
        'reference_joint_travel_rad': float(consumed[2])}
