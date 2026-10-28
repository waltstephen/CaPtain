"""Small hardware-scaled translations around geometrically plausible grasps.

The search changes the pose and rechecks collision; it never reduces clearance
or IK tolerances. This is measured-surface proposal search, not grasp success.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np
from itertools import product
from captain_harness.ports.deadline import check_deadline

from .motion_tools import pose_matrix


def local_translations(rotation, step_m):
    for axis in range(3):
        for sign in (-1, 1):
            yield f"local_axis_{axis}_{sign:+d}", sign * step_m * rotation[:, axis]


def local_diagonals(rotation, step_m):
    for direction in product((-1,0,1), repeat=3):
        if sum(v != 0 for v in direction) >= 2:
            yield 'local_diagonal_' + '_'.join(f'{v:+d}' for v in direction), rotation @ (step_m * np.asarray(direction))


def centered_translations(base, alternatives, step_m, existing, capacity):
    """Sample toward an already measured center without changing its heading.

    These are individually checked grasp proposals, not a clear motion path.
    Keep the declared total proposal bound even for a distant center.
    """
    pose = pose_matrix(base['position_m'], base['quaternion_wxyz'])
    center = next((item for item in alternatives if item.get('kind') == 'preserved_orientation'
        and np.allclose(pose_matrix(item['position_m'], item['quaternion_wxyz'])[:3, :3],
                        pose[:3, :3], rtol=0, atol=1e-10)), None)
    if center is None or capacity <= 0:
        return []
    delta = np.asarray(center['position_m']) - pose[:3, 3]
    distance = float(np.linalg.norm(delta))
    if not np.isfinite(distance) or distance <= step_m:
        return []
    offsets = []
    for index in range(1, int(np.ceil(distance / step_m))):
        offset = delta * (index * step_m / distance)
        if any(np.allclose(offset, old, rtol=0, atol=1e-12) for _, old in [*existing, *offsets]):
            continue
        offsets.append((f'toward_measured_center_step_{index}', offset))
        if len(offsets) >= capacity:
            break
    return offsets


def expand_grasp_refinement(backend, original, refinement, scene_handle=None, *, experimental_centering=False):
    if not refinement.success:
        return refinement
    check_deadline()
    value = deepcopy(refinement.value)
    # Keep a newly selected coarse orientation intact. Increasing point count
    # after that change can pull the hand into contact and undermine execution.
    # Otherwise refine the selected, still-unaccepted proposal locally. This
    # also covers a coarse pose that still collides with a measured surface.
    from .grasp_screening import _admissible
    if value.get('selected_index', 0) != 0 and _admissible(value):
        from dataclasses import replace
        value['local_search'] = {'improved': False, 'attempts': [],
            'reason': 'coarse_geometry_proposal_retained'}
        return replace(refinement, value=value)
    scene_handle = backend._grasp_scene_handle(original, scene_handle)
    scene, target = backend._candidate_scene_target(original, scene_handle)
    model = backend._motion_model()
    collision, policy = backend._grasp_planning_collision_checker(model, scene, target)
    chosen = value['alternatives'][value.get('selected_index', 0)]
    viable = [chosen] if chosen['contact'].get('visible_point_count_in_pad_sweep', 0) > 0 else []
    # If the coarse proposal is still obstructed, cover a small number of
    # distinct approach/closing directions instead of repeatedly translating
    # only one tilted pose. Jaw swaps do not consume another direction slot.
    if not _admissible(value):
        remaining = sorted(value['alternatives'], key=lambda a:
            (sum(a['approach']['geometry_hits'].values()), -a['contact'].get('visible_point_count_in_pad_sweep', 0)))
        for alternative in remaining:
            if len(viable) >= 3:
                break
            if alternative['contact'].get('visible_point_count_in_pad_sweep', 0) <= 0:
                continue
            r = pose_matrix(alternative['position_m'], alternative['quaternion_wxyz'])[:3,:3]
            rotations = [pose_matrix(a['position_m'], a['quaternion_wxyz'])[:3,:3] for a in viable]
            if any(float(r[:,2] @ old[:,2]) > .999 and abs(float(r[:,1] @ old[:,1])) > .995 for old in rotations):
                continue
            viable.append(alternative)
    step = float(value['before']['contact']['contact_position_tolerance_m'])
    if not np.isfinite(step) or step <= 0:
        raise ValueError('invalid public contact resolution')
    def rank(evidence):
        return (bool(evidence['approach']['geometry_hits']),
                sum(evidence['approach']['geometry_hits'].values()),
                -evidence['contact'].get('visible_point_count_in_pad_interior',
                                         evidence['contact'].get('visible_point_count_in_pad_sweep', 0)),
                -evidence['contact'].get('visible_point_count_in_pad_sweep', 0))
    best = rank(value['after'])
    diagnostics = []
    winner = None
    for base in viable[:3]:
        rotation = pose_matrix(base['position_m'], base['quaternion_wxyz'])[:3, :3]
        offsets = [(f'{name}_step_{multiple}', delta)
                   for multiple in (1, 2)
                   for name, delta in local_translations(rotation, multiple * step)]
        if not _admissible(value):
            offsets.extend(local_diagonals(rotation, step))
        elif experimental_centering and value.get('selected_index', 0) == 0:
            # Not qualified for normal screening: v3 exposed a closure stop.
            # The full center can collide while an intermediate contact region
            # is usable. Spend unused slots within the existing 32-offset bound.
            offsets.extend(centered_translations(base, value['alternatives'], step,
                                                  offsets, 32 - len(offsets)))
        for name, delta in offsets:
            check_deadline()
            candidate = {**original, 'position_m': (np.asarray(base['position_m']) + delta).tolist(),
                'quaternion_wxyz': base['quaternion_wxyz'],
                'approach_direction_world': rotation[:, 2].tolist(),
                'approach_distance_m': base['approach_distance_m'], 'backend_candidate': None}
            candidate.pop('grasp_handle', None)
            measured = backend.evaluate_grasp(candidate, scene_handle)
            check_deadline()
            if not measured.success:
                diagnostics.append({'base': base['kind'], 'offset': name, 'reason': measured.message})
                continue
            evidence = measured.value
            record = {'base': base['kind'], 'offset': name, 'translation_m': delta.tolist(),
                'position_m': candidate['position_m'], 'approach': evidence['approach'],
                'contact': evidence['contact']}
            diagnostics.append(record)
            if rank(evidence) >= best or evidence['approach']['geometry_hits']:
                continue
            width = backend._predicted_grasp_width(pose_matrix(candidate['position_m'], candidate['quaternion_wxyz']), target)
            if width is not None and width > backend.gripper_profile.max_width_m:
                record['reason'] = 'width_outside_hardware_limit'
                continue
            reach = backend._grasp_endpoint_reachability(candidate, scene.joints,
                pregrasp_valid=lambda joints: not collision(joints)['collision'])
            record['reachability'] = reach
            check_deadline()
            if not reach['reachable']:
                continue
            evidence.update(reachability=reach,
                pregrasp_collision={'checked': True, **collision(reach['joints'])}, collision_policy=policy)
            if evidence['pregrasp_collision']['collision']:
                continue
            candidate.update(backend='public_geometry_refinement', model_score=None,
                refinement_basis=base['kind']+'; '+name+' at public contact resolution',
                source_observation_handle=scene.source_observation_handle,
                predicted_grasp_width_m=width, width_feasible=True, width_known=width is not None,
                convergence_risk='unassessed', style_preference_score=None, request_style_match='unassessed',
                reachable=True, ik_feasible=True,
                predicted_joint_travel_rad=float(np.abs(np.asarray(reach['joints'])-scene.joints).sum()),
                quality_flags=['geometry_refinement_unscored'], target_contact_evidence=evidence['contact'],
                top_down_alignment=float(np.clip(-rotation[2, 2], 0., 1.)))
            candidate.pop('feasibility_evidence', None)
            best = rank(evidence)
            winner = candidate, evidence
    value['local_search'] = {'step_m': step, 'max_bases': 3, 'max_offsets_per_base': 32,
        'attempts': diagnostics, 'improved': winner is not None,
        'limitation': 'More observed finger-pad support does not prove force closure or stable attachment.'}
    if winner is not None:
        candidate, evidence = winner
        check_deadline()
        candidate = backend._issue_grasp_candidate(candidate)
        value.update(candidate=candidate, after=evidence, pose_changed=True,
            translation_m=(np.asarray(candidate['position_m'])-original['position_m']).tolist(),
            requires_decision=True, reason='local_geometry_search_improved_measured_support',
            score_status='unscored geometry proposal; native score not transferred')
    from dataclasses import replace
    return replace(refinement, value=value)
