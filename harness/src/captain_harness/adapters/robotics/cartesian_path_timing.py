"""Advance a Cartesian polyline using one shared control-cycle time budget."""
import math
import numpy as np
from scipy.spatial.transform import Rotation


def minimum_command_duration(path, max_speed_m_s, max_angular_speed_rad_s):
    """Nominal setpoint travel only: tracking, settling and contacts can add time."""
    if not all(math.isfinite(x) and x > 0 for x in (max_speed_m_s, max_angular_speed_rad_s)):
        raise ValueError("command speeds must be finite and positive")
    total = 0.
    for start, end in zip(path[:-1], path[1:]):
        distance = np.linalg.norm(end[:3, 3] - start[:3, 3])
        angle = Rotation.from_matrix(end[:3, :3] @ start[:3, :3].T).magnitude()
        total += max(distance / max_speed_m_s, angle / max_angular_speed_rad_s)
    return float(total)


def advance_cartesian_reference(path, desired, index, *, duration_s,
                                max_speed_m_s, max_angular_speed_rad_s):
    if not all(math.isfinite(x) and x > 0 for x in
               (duration_s, max_speed_m_s, max_angular_speed_rad_s)):
        raise ValueError("cycle duration and speeds must be finite and positive")
    command = np.asarray(desired, dtype=float).copy()
    remaining = float(duration_s)
    while index < len(path):
        target = path[index]
        delta = target[:3, 3] - command[:3, 3]
        turn = Rotation.from_matrix(target[:3, :3] @ command[:3, :3].T).as_rotvec()
        required = max(np.linalg.norm(delta) / max_speed_m_s,
                       np.linalg.norm(turn) / max_angular_speed_rad_s)
        if required <= 1e-12:
            command = target.copy()
            index += 1
            continue
        if remaining <= 1e-12:
            break
        fraction = min(1., remaining / required)
        command[:3, 3] += fraction * delta
        command[:3, :3] = Rotation.from_rotvec(fraction * turn).as_matrix() @ command[:3, :3]
        remaining = max(0., remaining - fraction * required)
        if fraction >= 1. - 1e-12:
            command = target.copy()
            index += 1
        else:
            break
    return command, index


def interpolate_pose(start, end, fraction):
    pose = start.copy()
    pose[:3, 3] += fraction * (end[:3, 3] - start[:3, 3])
    turn = Rotation.from_matrix(end[:3, :3] @ start[:3, :3].T).as_rotvec()
    pose[:3, :3] = Rotation.from_rotvec(fraction * turn).as_matrix() @ start[:3, :3]
    return pose


def truncate_translation(path, distance):
    """Keep rotations and interpolate the crossing segment at the exact boundary."""
    result, travelled = [path[0]], 0.
    for start, end in zip(path[:-1], path[1:]):
        length = float(np.linalg.norm(end[:3, 3] - start[:3, 3]))
        if length > 0 and travelled + length > distance:
            result.append(interpolate_pose(start, end, (distance - travelled) / length))
            break
        result.append(end)
        travelled += length
    return np.asarray(result)


def motion_preview(path, speed, angular_speed, duration, frequency, max_steps=None):
    for value, upper, name in ((speed, .1, 'max_speed_m_s'),
            (angular_speed, 1., 'max_angular_speed_rad_s'), (duration, 30., 'max_duration_s')):
        if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= upper:
            raise ValueError(f'{name} must be in (0, {upper}]')
    steps = int(math.floor(duration * frequency + 1e-9))
    if max_steps is not None:
        steps = min(steps, max_steps)
    effective = steps / frequency
    minimum = minimum_command_duration(path, speed, angular_speed)
    pose, index = (advance_cartesian_reference(path, path[0], 1, duration_s=effective,
        max_speed_m_s=speed, max_angular_speed_rad_s=angular_speed) if steps else (path[0], 1))
    return {'max_speed_m_s': speed, 'max_angular_speed_rad_s': angular_speed,
        'minimum_command_duration_s': minimum, 'requested_duration_limit_s': duration,
        'effective_duration_limit_s': effective, 'command_time_feasible': minimum <= effective + 1e-9,
        'budget_command_pose': pose.tolist(), 'budget_next_path_index': index,
        'path_translation_m': float(sum(np.linalg.norm(b[:3,3]-a[:3,3]) for a,b in zip(path[:-1],path[1:]))),
        'path_samples': len(path), 'physical_action_executed': False,
        'timing_scope': 'command travel only; tracking, contact and endpoint settling remain unguaranteed',
        'scope': 'read-only timing; no collision, retention or task certification; execution revalidates'}


def remaining_exit_path(path, current, command, command_index):
    """Project measured progress only onto the issued prefix; never reverse a retreat.

    Return no permission if the measured pose cannot be bound to that prefix.
    The connector begins at the measured pose and heads toward the next original
    waypoint; it never walks back to an earlier measured point.
    """
    prefix = list(path[:command_index])
    if not prefix or not np.allclose(prefix[-1], command):
        prefix.append(command)
    candidates = []
    for index, (a, b) in enumerate(zip(prefix[:-1], prefix[1:])):
        delta = b[:3,3]-a[:3,3]
        turn = Rotation.from_matrix(b[:3,:3]@a[:3,:3].T).as_rotvec()
        observed = Rotation.from_matrix(current[:3,:3]@a[:3,:3].T).as_rotvec()
        denom = np.dot(delta, delta) + .01 * np.dot(turn, turn)
        fraction = np.clip((np.dot(current[:3,3]-a[:3,3], delta) + .01*np.dot(observed, turn)) / denom, 0, 1) if denom > 1e-20 else 1.
        projected = interpolate_pose(a, b, fraction)
        pe = np.linalg.norm(current[:3,3]-projected[:3,3])
        ae = Rotation.from_matrix(current[:3,:3]@projected[:3,:3].T).magnitude()
        next_pose = path[index+1]
        remaining_delta = next_pose[:3,3] - current[:3,3]
        remaining_turn = Rotation.from_matrix(next_pose[:3,:3]@current[:3,:3].T).as_rotvec()
        follows_exit = (np.dot(remaining_delta,delta) >= -1e-12
            and np.dot(remaining_turn,turn) >= -1e-12)
        if pe <= .002 and ae <= .01 and follows_exit:
            candidates.append((pe + .1*ae, -index, index))
    if not candidates:
        return None
    index = min(candidates)[2]
    # prefix segment index maps to the same segment in the original path.
    suffix = [current, *path[index+1:]]
    return np.asarray(suffix)
