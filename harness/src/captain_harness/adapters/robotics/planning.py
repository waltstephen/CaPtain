"""OMPL RRTConnect with explicit sampled-edge checking and public scenes."""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import time
import numpy as np

from captain_harness.ports.deadline import current_deadline, check_deadline, deadline_scope, DeadlineExceeded


@dataclass(frozen=True)
class SensorScene:
    points: np.ndarray
    source_observation_handle: str
    joints: tuple[float, ...]
    aperture: float
    time_s: float | None
    margin_m: float
    removed_robot_points: int


@dataclass(frozen=True)
class MotionPlan:
    scene_handle: str
    waypoints: np.ndarray
    start_joints: tuple[float, ...]
    aperture: float
    controller: str
    target: np.ndarray
    elapsed_s: float
    checks: int
    contact_target_points: np.ndarray | None = None
    endpoint_precision: dict | None = None
    tracking_reserve: dict | None = None
    carried_geometry_handle: str | None = None


def interpolate_path(path, max_joint_step=.025):
    path = np.asarray(path, dtype=float)
    output = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        count = max(1, int(np.ceil(np.max(np.abs(b - a)) / max_joint_step)))
        output.extend(a + (b - a) * f for f in np.linspace(0., 1., count + 1)[1:-1])
        output.append(b.copy())
    return np.asarray(output)


def _shorten_validated_path(path, checked, *, expires_s, max_joint_step, edge_valid=None):
    """Keep only fully sampled shorter edges; fall back to the validated remainder."""
    parent = current_deadline()
    if parent is not None:
        expires_s = min(expires_s, parent.expires_s)
    path = np.asarray(path, dtype=float)
    lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    output = [path[0]]
    index, attempted, accepted, expired = 0, 0, 0, False
    while index < len(path) - 1:
        successor = index + 1
        shortcut = None
        for candidate in range(len(path) - 1, index + 1, -1):
            if time.monotonic() >= expires_s:
                expired = True
                break
            # Preserve direct segments and never exchange length for fewer samples.
            if np.linalg.norm(path[candidate] - path[index]) >= lengths[candidate] - lengths[index] - 1e-10:
                continue
            edge = interpolate_path([path[index], path[candidate]], max_joint_step)
            attempted += 1
            clear = True
            for point in edge:
                if time.monotonic() >= expires_s:
                    expired, clear = True, False
                    break
                point_clear = checked(point)
                if time.monotonic() >= expires_s:
                    expired, clear = True, False
                    break
                if not point_clear:
                    clear = False
                    break
            if clear and edge_valid is not None:
                try:
                    with deadline_scope('optional_shortening', expires_s=expires_s):
                        clear = all(edge_valid(a, b) for a, b in zip(edge[:-1], edge[1:]))
                except DeadlineExceeded:
                    check_deadline()  # Earlier public deadline remains a real timeout.
                    expired, clear = True, False
            if expired:
                break
            if clear:
                successor, shortcut = candidate, edge
                accepted += 1
                break
        if expired:
            output.extend(path[index + 1:])
            break
        output.extend(shortcut[1:] if shortcut is not None else [path[successor]])
        index = successor
    shortened = np.asarray(output)
    return shortened, {'algorithm': 'deterministic_validated_shortcuts',
        'collision_policy': 'same_state_and_edge_checker' if edge_valid is not None else 'same_sampled_checker', 'max_joint_step_rad': max_joint_step,
        'original_waypoint_count': len(path), 'final_waypoint_count': len(shortened),
        'original_joint_length_rad': float(lengths[-1]),
        'final_joint_length_rad': float(np.linalg.norm(np.diff(shortened, axis=0), axis=1).sum()),
        'attempted_shortcuts': attempted, 'accepted_shortcuts': accepted,
        'budget_exhausted': expired,
        'limitation': 'Sampled observed-scene clearance only; no continuous or unseen-space guarantee.'}


def plan_joint_path(start, goal, limits, valid, *, timeout_s=5., max_joint_step=.025, edge_valid=None):
    """Validate and plan cooperatively within one inherited/local time budget."""
    began = time.monotonic()
    progress = {'checks': 0, 'edge_checks': 0, 'planning_stage': 'entry'}
    with deadline_scope('path_planning', timeout_s=timeout_s):
        try:
            return _plan_joint_path_within_deadline(start, goal, limits, valid,
                max_joint_step=max_joint_step, began=began, progress=progress, edge_valid=edge_valid)
        except DeadlineExceeded as exc:
            return None, {**progress, 'reason': 'deadline_expired',
                'planning_time_s': time.monotonic() - began, 'deadline': exc.deadline_receipt,
                'limitation': 'Interrupted search is not evidence of infeasibility; no executable path returned.'}


def _plan_joint_path_within_deadline(start, goal, limits, valid, *, max_joint_step, began, progress, edge_valid=None):
    check_deadline()
    start, goal, limits = np.asarray(start), np.asarray(goal), np.asarray(limits)
    def checked(q):
        check_deadline()
        progress['checks'] += 1
        result = bool(valid(np.asarray(q, dtype=float)))
        check_deadline()
        return result
    def checked_edge(a, b):
        check_deadline()
        progress['edge_checks'] += 1
        clear = bool(edge_valid(a, b))
        check_deadline()
        return clear
    def path_clear(path):
        return (all(checked(q) for q in path) and (edge_valid is None or
                all(checked_edge(a, b) for a, b in zip(path[:-1], path[1:]))))
    progress['planning_stage'] = 'start_validation'
    if not checked(start):
        return None, {'reason': 'start_in_observed_collision', 'checks': progress['checks']}
    progress['planning_stage'] = 'goal_validation'
    if not checked(goal):
        return None, {'reason': 'goal_in_observed_collision', 'checks': progress['checks']}
    progress['planning_stage'] = 'direct_validation'
    direct = interpolate_path([start, goal], max_joint_step)
    if path_clear(direct):
        return direct, {'reason': 'direct_path', 'checks': progress['checks'],
                        'planning_time_s': time.monotonic() - began, 'edge_checks': progress['edge_checks']}
    rejected = 0
    reason = 'no_exact_path_within_budget'
    for _ in range(3):
        progress.update(planning_stage='rrt_solve', rejected_candidate_paths=rejected)
        remaining = check_deadline().remaining()
        edge_options = {} if edge_valid is None else {'edge_valid': checked_edge}
        raw = _rrt_candidate(start, goal, limits, checked, timeout_s=remaining,
                             max_joint_step=max_joint_step, **edge_options)
        check_deadline()
        if raw is None:
            reason = 'no_exact_path_within_budget'
            break
        progress['planning_stage'] = 'dense_validation'
        path = interpolate_path(raw, max_joint_step)
        if path_clear(path):
            progress.update(planning_stage='path_shortening', validated_path_waypoints=len(path))
            # Optional optimization must leave time to return the already
            # validated path. Its local expiry is not a new public budget.
            shortening_start = time.monotonic()
            shortening_budget = .5 * check_deadline().remaining()
            path, optimization = _shorten_validated_path(path, checked,
                expires_s=shortening_start + shortening_budget, max_joint_step=max_joint_step,
                edge_valid=checked_edge if edge_valid is not None else None)
            optimization.update(time_budget_s=shortening_budget,
                budget_basis='half of the remaining shared planning time; no added budget')
            check_deadline()
            return path, {'reason': 'rrt_connect', 'checks': progress['checks'],
                          'rejected_candidate_paths': rejected,
                          'path_optimization': optimization,
                          'planning_time_s': time.monotonic() - began, 'edge_checks': progress['edge_checks']}
        rejected += 1
        reason = 'dense_path_validation_failed'
    return None, {'reason': reason, 'checks': progress['checks'], 'rejected_candidate_paths': rejected,
                  'planning_time_s': time.monotonic() - began, 'edge_checks': progress['edge_checks']}


def _sampled_motion_clear(start, end, checked, *, max_joint_step, edge_valid):
    """Check the same samples, visiting the new endpoint and broad interiors first."""
    check_deadline()
    sampled = interpolate_path([start, end], max_joint_step)
    if not checked(sampled[-1]):
        return False
    pending = deque([(1, len(sampled) - 2)])
    while pending:
        check_deadline()
        first, last = pending.popleft()
        if first > last:
            continue
        middle = (first + last) // 2
        if not checked(sampled[middle]):
            return False
        pending.extend(((first, middle - 1), (middle + 1, last)))
    # Every original nominal sample still passed; no observed-surface sweep
    # may be bypassed merely because its endpoints/interiors appear clear.
    return edge_valid(start, end)


def _rrt_candidate(start, goal, limits, checked, *, timeout_s, max_joint_step, edge_valid=None):
    """Return one exact OMPL candidate; caller owns final sampled validation."""
    from ompl import base as ob, geometric as og
    space = ob.RealVectorStateSpace(len(start))
    bounds = ob.RealVectorBounds(len(start))
    for i, (low, high) in enumerate(limits):
        bounds.setLow(i, float(low)); bounds.setHigh(i, float(high))
    space.setBounds(bounds)
    setup = og.SimpleSetup(space)
    setup.setStateValidityChecker(lambda state: checked([state[i] for i in range(len(start))]))
    # OMPL divides edges by state-space extent; dense validation below is final.
    setup.getSpaceInformation().setStateValidityCheckingResolution(
        max_joint_step / float(np.linalg.norm(limits[:, 1] - limits[:, 0])))
    if edge_valid is not None:
        class ObservedMotionValidator(ob.MotionValidator):
            def __init__(self, information):
                super().__init__(information)

            def checkMotion(self, first, last):
                start = np.asarray([first[i] for i in range(len(limits))])
                end = np.asarray([last[i] for i in range(len(limits))])
                return _sampled_motion_clear(start, end, checked,
                    max_joint_step=max_joint_step, edge_valid=edge_valid)
        setup.getSpaceInformation().setMotionValidator(ObservedMotionValidator(setup.getSpaceInformation()))
    a, b = space.allocState(), space.allocState()
    for i in range(len(start)):
        a[i], b[i] = float(start[i]), float(goal[i])
    setup.setStartAndGoalStates(a, b)
    planner = og.RRTConnect(setup.getSpaceInformation())
    # Tie tree growth to the declared path resolution rather than the full
    # joint-limit box. L2<=step implies every joint stays within that scale.
    planner.setRange(float(max_joint_step))
    setup.setPlanner(planner)
    status = setup.solve(float(timeout_s))
    if not status or not setup.haveExactSolutionPath():
        return None
    return np.asarray([[state[i] for i in range(len(start))]
                       for state in setup.getSolutionPath().getStates()])
