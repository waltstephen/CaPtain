"""Bounded contact segments with measured target progress and explicit unknowns."""
from copy import deepcopy
from dataclasses import replace
import math
import numpy as np

from captain_harness.ports.capability import CapabilityResult, FailureCategory


def assess_progress(measurement, *, previous=None, requested_direction=None,
                    rotation_axis=None, rotation_sign=1,requested_segment_motion=None):
    unknown={'status':'target_motion_unresolved', 'measurement':deepcopy(measurement),
        'guidance':'Stop this bounded probe and obtain a clearer view or reselect the same target. '
            'Missing visual evidence does not prove that the mechanism failed to move.'}
    if measurement.get('status')!='supported':
        return unknown
    if rotation_axis is not None:
        vector=measurement.get('rotation_vector_rad')
        old=(previous or {}).get('rotation_vector_rad',[0,0,0])
        direction=np.asarray(rotation_axis,float)*rotation_sign
        threshold=measurement.get('rotation_resolution_rad')
        if not isinstance(threshold,(int,float)) or not math.isfinite(threshold):
            return unknown
        units='radian'
    else:
        vector=measurement.get('translation_m')
        old=(previous or {}).get('translation_m',[0,0,0])
        direction=np.asarray(requested_direction,float)
        spread=measurement.get('motion_spread_m')
        if not isinstance(spread,(int,float)) or not math.isfinite(spread):
            return unknown
        threshold=max(.002,2*spread)
        units='meter'
    if vector is None or old is None:
        return unknown
    if requested_segment_motion is not None and threshold > abs(requested_segment_motion)/2:
        return {**unknown,'measurement_resolution_threshold':threshold,
            'requested_segment_motion':requested_segment_motion,
            'guidance':'The visual resolution is too coarse for this probe. Acquire a clearer view '
                'before treating this as absent mechanism motion or changing the contact strategy.'}
    if rotation_axis is not None:
        from scipy.spatial.transform import Rotation
        delta=(Rotation.from_rotvec(vector)*Rotation.from_rotvec(old).inv()).as_rotvec()
    else:
        delta=np.asarray(vector,float)-old
    if delta.shape!=(3,) or not np.isfinite(delta).all():
        return unknown
    signed=float(delta @ (direction/np.linalg.norm(direction)))
    status=('progress_observed' if signed>threshold else
            'opposite_target_progress' if signed < -threshold else 'no_observed_target_progress')
    return {'status':status,'incremental_projected_motion':signed,'units':units,
        'measurement_resolution_threshold':threshold,'measurement':deepcopy(measurement),
        'guidance':('Continue only within the declared motion and force budgets.' if status=='progress_observed' else
            'The measurement does not establish progress along the requested motion. Inspect target identity, '
            'contact point and direction/axis; retreat the recorded last segment if appropriate, then reobserve. '
            'Do not repeat the same loaded command or increase force without new evidence.')}


def run_segments(move, measure, *, segments, duration_s, initial_force_n, max_force_n,
                 time_s, progress_arguments):
    start=time_s(); previous=None; records=[]; trajectory=[]; displacement=np.zeros(3)
    steps=0; any_executed=False; peak=initial_force_n; result=None
    # Preserve the original operation's +5 N stop across segment boundaries.
    force_limit=min(max_force_n,initial_force_n+5.)
    for index in range(segments):
        remaining=duration_s-(time_s()-start)
        if remaining<=0:
            result=CapabilityResult.failed(FailureCategory.CONTROLLER_REJECTED,'contact_duration_limit',
                executed=any_executed,value={'stopped_reason':'duration_limit'})
            break
        result=move(1./segments,remaining,force_limit)
        any_executed |= result.executed
        value=result.value if isinstance(result.value,dict) else {}
        steps+=int(value.get('control_steps',0))
        displacement+=value.get('actual_displacement_m',[0,0,0])
        trajectory.extend(value.get('trajectory',[]))
        force=value.get('max_observed_force_n',value.get('observed_force_n'))
        if isinstance(force,(int,float)) and math.isfinite(force):
            peak=max(peak,force)
        if not result.executed:
            break
        try:
            measurement=measure()
        except (ValueError,RuntimeError) as exc:
            measurement={'status':'insufficient','reason':f'{type(exc).__name__}: {exc}'}
        progress=assess_progress(measurement,previous=previous,**progress_arguments)
        records.append({'segment':index,'executed':result.executed,'success':result.success,
            'stopped_reason':value.get('stopped_reason'), 'target_progress':progress,
            'force_guard':deepcopy(value.get('force_guard'))})
        if not result.success:
            break
        if progress['status']!='progress_observed':
            result=CapabilityResult.failed(FailureCategory.PERCEPTION_UNCERTAIN,progress['status'],
                executed=True,value={**value,'reached':False,'stopped_reason':progress['status']})
            break
        previous=measurement
    assert result is not None
    return replace(result,executed=any_executed,value={
        **(result.value if isinstance(result.value,dict) else {}),
        'reached': result.success,
        'control_steps':steps,'actual_displacement_m':displacement.tolist(),
        'max_observed_force_n':peak,'force_limit_n':force_limit,'trajectory':trajectory,
        'operation_force_guard':{'initial_force_n':initial_force_n,'requested_absolute_limit_n':max_force_n,
            'increase_limit_n':5.,'effective_segment_absolute_limit_n':force_limit,
            'scope':'original sampled +5 N budget is retained across all contact segments'},
        'contact_identity':'unknown','observed_contact_segments':records,
        'target_progress':records[-1]['target_progress'] if records else {'status':'not_measured'},
        'executed_segments':len(records),'requested_segments':segments,
        'retreat_scope':'Only the last recorded robot segment is eligible; inspect retreat_last_motion result.',
        'task_success':'not_assessed'})


def observed_contact(backend, target_detection_id, *, direction=None, distance_m=None,
                     axis=None, pivot_m=None, angle_rad=None, max_speed_m_s=.02,
                     max_angular_speed_rad_s=.2,max_force_n=18.,max_duration_s=10.):
    from .visual_tracking import track_rgbd
    from .observation_equivalence import same_public_snapshot
    detection=backend._require_selected_detection(target_detection_id)
    source=backend.registry.get(detection.observation_handle,expected_kind='observation')
    current=backend._capture_observation()
    if not same_public_snapshot(source,current):
        raise ValueError('contact target is stale: observe and select the same target before this probe')
    camera=backend._camera_by_name(source,detection.camera_name)
    # Validate the complete caller request before executing ANY segment.
    for name,value,maximum in [('speed',max_speed_m_s,.1),('force',max_force_n,18.),
                               ('duration',max_duration_s,30.)]:
        if not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<value<=maximum:
            raise ValueError(f'invalid contact {name}')
    rotation=axis is not None
    from captain_harness.adapters.robotics.contact_goals import (
        linear_waypoint,angular_waypoint,final_contact_goal,rotation as pose_rotation)
    origin=current.robot_state
    segment_index=0
    vector=np.asarray(axis if rotation else direction,float)
    if vector.shape!=(3,) or not np.isfinite(vector).all() or np.linalg.norm(vector)<1e-10:
        raise ValueError('contact direction/axis must be a finite nonzero three-vector')
    if rotation:
        pivot=np.asarray(pivot_m,float)
        radius=np.linalg.norm(pivot-np.asarray(current.robot_state.end_effector_position_m))
        if pivot.shape!=(3,) or not np.isfinite(pivot).all() or radius>.5:
            raise ValueError('invalid contact pivot')
        if (not math.isfinite(angle_rad) or not 0<abs(angle_rad)<=math.pi
                or not math.isfinite(max_angular_speed_rad_s) or not 0<max_angular_speed_rad_s<=1.):
            raise ValueError('invalid contact rotation')
        segments=max(1,math.ceil(abs(angle_rad)/.1),math.ceil(abs(angle_rad)*radius/.01))
        def move(fraction,remaining,force):
            nonlocal segment_index
            segment_index+=1
            measured=backend.robot.get_robot_state()
            if not measured.success or measured.value is None:
                return CapabilityResult.failed(FailureCategory.CONTROLLER_REJECTED,'robot_state_unavailable')
            _,increment=angular_waypoint(origin.end_effector_quaternion_wxyz,axis,angle_rad,
                segment_index*fraction,measured.value.end_effector_quaternion_wxyz)
            if abs(increment)<1e-7:
                return replace(CapabilityResult.ok({'stopped_reason':'waypoint_already_reached'}),executed=False)
            return backend.robot.rotate_about_axis(axis,pivot_m,increment,
                max_speed_m_s=max_speed_m_s,max_angular_speed_rad_s=max_angular_speed_rad_s,
                max_force_n=force,max_duration_s=remaining)
        arguments={'rotation_axis':axis,'rotation_sign':1 if angle_rad>0 else -1,
                   'requested_segment_motion':abs(angle_rad)/segments}
    else:
        if not math.isfinite(distance_m) or not 0<distance_m<=.25:
            raise ValueError('invalid contact distance')
        segments=max(1,math.ceil(distance_m/.01))
        def move(fraction,remaining,force):
            nonlocal segment_index
            segment_index+=1
            measured=backend.robot.get_robot_state()
            if not measured.success or measured.value is None:
                return CapabilityResult.failed(FailureCategory.CONTROLLER_REJECTED,'robot_state_unavailable')
            _,delta,remaining_distance=linear_waypoint(origin.end_effector_position_m,direction,
                distance_m,segment_index*fraction,measured.value.end_effector_position_m)
            if remaining_distance<1e-7:
                return replace(CapabilityResult.ok({'stopped_reason':'waypoint_already_reached'}),executed=False)
            return backend.robot.move_linear(delta.tolist(),remaining_distance,max_speed_m_s=max_speed_m_s,
                max_force_n=force,max_duration_s=remaining)
        arguments={'requested_direction':direction,'requested_segment_motion':distance_m/segments}
    initial=backend.robot._eef_force_norm()
    if initial is None or not math.isfinite(initial):
        return CapabilityResult.failed(FailureCategory.CONTROLLER_REJECTED,'force_unavailable',
            value={'stopped_reason':'force_unavailable','contact_identity':'unknown','control_steps':0,'reached':False})
    def measure():
        latest=backend._capture_observation()
        value=track_rgbd(camera,backend._camera_by_name(latest,detection.camera_name),detection.mask)
        value.update(source_detection_id=target_detection_id,source_observation_handle=detection.observation_handle,
            source_time_s=source.time_s,time_s=latest.time_s)
        return value
    result=run_segments(move,measure,segments=segments,duration_s=max_duration_s,
        initial_force_n=initial,max_force_n=max_force_n,time_s=lambda:backend.robot.time_s,
        progress_arguments=arguments)
    final=backend.robot.get_robot_state()
    if final.success and final.value is not None:
        goal_position,goal_orientation=final_contact_goal(origin,direction=direction,distance=distance_m,
            axis=axis,pivot=pivot_m,angle=angle_rad)
        position_error=float(np.linalg.norm(goal_position-final.value.end_effector_position_m))
        angle_error=float((goal_orientation*pose_rotation(final.value.end_effector_quaternion_wxyz).inv()).magnitude())
        from captain_harness.adapters.robotics.motion_precision import motion_precision
        start_pose, goal_pose = np.eye(4), np.eye(4)
        start_pose[:3, 3] = origin.end_effector_position_m
        start_pose[:3, :3] = pose_rotation(origin.end_effector_quaternion_wxyz).as_matrix()
        goal_pose[:3, 3], goal_pose[:3, :3] = goal_position, goal_orientation.as_matrix()
        precision = motion_precision(start_pose, goal_pose, .002, .01)
        robot_goal_reached=(position_error<=precision['position_tolerance_m']
            and angle_error<=precision['orientation_tolerance_rad'])
        actual_rotation = (pose_rotation(final.value.end_effector_quaternion_wxyz)
            * pose_rotation(origin.end_effector_quaternion_wxyz).inv()).as_rotvec().tolist()
        result=replace(result,value={**result.value,**precision,
            'actual_rotation_vector_rad':actual_rotation,'full_goal_position_residual_m':position_error,
            'full_goal_orientation_residual_rad':angle_error,'goal_reference':'original_measured_robot_pose',
            'reached':robot_goal_reached,'robot_goal_reached':robot_goal_reached})
        if result.success and not robot_goal_reached:
            return CapabilityResult.failed(FailureCategory.CONTROLLER_REJECTED,'contact_final_pose_residual',
                executed=result.executed,value={**result.value,'reached':False,'stopped_reason':'contact_final_pose_residual'})
    elif result.success:
        return CapabilityResult.failed(FailureCategory.CONTROLLER_REJECTED,'final_robot_state_unavailable',
            executed=result.executed,value={**result.value,'reached':False,'stopped_reason':'final_robot_state_unavailable'})
    return result
