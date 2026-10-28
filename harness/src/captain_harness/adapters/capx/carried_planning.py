"""Current public attachment/geometry binding for detached carried planning."""
from dataclasses import dataclass
from copy import deepcopy
import math
import numpy as np
from captain_harness.ports.capability import CapabilityResult, FailureCategory
from captain_harness.ports.deadline import check_deadline
from .artifacts import ArtifactHandleError


@dataclass(frozen=True)
class CarriedObservation:
    tracking_handle: str
    point_handle: str
    epoch: int
    source_observation_handle: str
    joints: tuple
    aperture: float
    time_s: float
    points_world: np.ndarray
    model: object
    receipt: dict
    selected_masks: tuple = ()


def _tracked(tool,tracking_handle,point_handle,epoch=None):
    if tool._active_attachment_handle!=tracking_handle:
        raise ValueError('carried_geometry_requires_current_verified_attachment')
    tracked=tool.registry.get(tracking_handle,expected_kind='entity_track')
    if (not tracked.attachment_verified or tracked.eef_to_object_transform is None
            or getattr(tracked,'attachment_current_point_handle',None)!=point_handle
            or (epoch is not None and getattr(tracked,'carried_geometry_epoch',None)!=epoch)):
        raise ValueError('carried_geometry_attachment_evidence_changed_or_unaccepted')
    return tracked


def _fresh(tool,joints,aperture,time_s):
    state=tool.robot.get_robot_state()
    now=getattr(tool.robot,'time_s',None)
    if not state.success or state.value is None:
        raise ValueError('carried_geometry_robot_state_unavailable')
    if (time_s is None or now is None or not math.isfinite(time_s) or not math.isfinite(now)
            or abs(time_s-now)>1e-9
            or not np.allclose(joints,state.value.joint_positions_rad,rtol=0,atol=1e-6)
            or abs(aperture-state.value.gripper_open_fraction)>1e-6):
        raise ValueError('stale_carried_geometry: capture from a current accepted attachment mark')
    return state.value


def capture_carried_geometry(tool,tracking_handle,current_point_handle,use_multiview=True):
    from .motion_tools import pose_matrix
    try:
        check_deadline()
        tracked=_tracked(tool,tracking_handle,current_point_handle)
        epoch=tracked.carried_geometry_epoch
        marked=tool.registry.get(current_point_handle,expected_kind='marked_point')
        obs=tool.registry.get(marked.observation_handle,expected_kind='observation')
        state=_fresh(tool,obs.robot_state.joint_positions_rad,obs.robot_state.gripper_open_fraction,obs.time_s)
        original=tool._motion_model()
        original.set_state(state.joint_positions_rad,state.gripper_open_fraction)
        def seed_points(points):
            own,_=original.points_in_robot(points,margin=.003,include_visual=True)
            return np.asarray(points)[~own]
        mark,_,points,_=tool._points_from_mark(current_point_handle,use_multiview=use_multiview,
            require_primary=True,projection_filter=seed_points)
        check_deadline()
        own,_=original.points_in_robot(points,margin=.003,include_visual=True)
        selected=np.asarray(points,dtype=float)[~own].copy()
        pose=pose_matrix(state.end_effector_position_m,state.end_effector_quaternion_wxyz)
        local=(selected-pose[:3,3])@pose[:3,:3]
        model=original.with_observed_carried_geometry(local)
        check_deadline()
        _fresh(tool,state.joint_positions_rad,state.gripper_open_fraction,obs.time_s)
        _tracked(tool,tracking_handle,current_point_handle,epoch)
        selected.setflags(write=False)
        receipt={**deepcopy(model.carried_geometry_receipt),'tracking_handle':tracking_handle,
            'current_point_handle':current_point_handle,'attachment_epoch':epoch,
            'source_observation_handle':marked.observation_handle,'time_s':obs.time_s,
            'selected_point_count':len(selected),'robot_points_removed':int(own.sum()),
            'primary_view_retained':True,
            'view_reconstruction':deepcopy(getattr(mark,'carried_view_reconstruction',[])),
            'views_used':mark.views_used,'cross_view_consistent':mark.cross_view_consistent,
            'points_eef_bounds_min_m':local.min(axis=0).tolist(),
            'points_eef_bounds_max_m':local.max(axis=0).tolist(),
            'perception_scope':'accepted public attachment mark plus existing calibrated cross-view segmentation; semantic identity and unseen coverage remain uncertain',
            'motion_authorized':False}
        bundle=CarriedObservation(tracking_handle,current_point_handle,epoch,
            marked.observation_handle,tuple(state.joint_positions_rad),float(state.gripper_open_fraction),
            float(obs.time_s),selected,model,receipt,getattr(mark,'carried_selected_masks',()))
        handle=tool.registry.put('carried_geometry',bundle)
        return CapabilityResult.ok({**receipt,'carried_geometry_handle':handle})
    except (ValueError,ArtifactHandleError) as exc:
        return CapabilityResult.failed(FailureCategory.NOT_EXECUTED,str(exc),
            value={'reason':str(exc),'motion_authorized':False})


def carried_context(tool,handle,scene):
    check_deadline()
    bundle=tool.registry.get(handle,expected_kind='carried_geometry')
    _tracked(tool,bundle.tracking_handle,bundle.point_handle,bundle.epoch)
    _fresh(tool,bundle.joints,bundle.aperture,bundle.time_s)
    if scene.source_observation_handle!=bundle.source_observation_handle:
        from .observation_equivalence import same_public_snapshot
        source=tool.registry.get(bundle.source_observation_handle,expected_kind='observation')
        current=tool.registry.get(scene.source_observation_handle,expected_kind='observation')
        if not same_public_snapshot(source,current):
            raise ValueError('stale_carried_geometry_scene: rebuild from the bound public observation')
    check_deadline()
    return bundle
