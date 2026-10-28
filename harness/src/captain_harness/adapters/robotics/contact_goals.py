"""Absolute contact waypoints anchored to the original measured robot pose."""
import numpy as np
from scipy.spatial.transform import Rotation


def rotation(quaternion_wxyz):
    return Rotation.from_quat(np.roll(quaternion_wxyz,-1))


def linear_waypoint(origin,direction,distance,fraction,current):
    unit=np.asarray(direction,float);unit/=np.linalg.norm(unit)
    target=np.asarray(origin,float)+unit*distance*fraction
    delta=target-np.asarray(current,float)
    return target,delta,float(np.linalg.norm(delta))


def angular_waypoint(origin_quaternion,axis,angle,fraction,current_quaternion):
    unit=np.asarray(axis,float);unit/=np.linalg.norm(unit)
    target=Rotation.from_rotvec(unit*angle*fraction)*rotation(origin_quaternion)
    difference=target*rotation(current_quaternion).inv()
    # Projection gives the correction permitted by this one-axis primitive.
    # Orthogonal tracking errors remain visible in the final full-pose residual.
    correction=float(difference.as_rotvec() @ unit)
    return target,float(np.clip(correction,-.1,.1))


def final_contact_goal(state,*,direction=None,distance=None,axis=None,pivot=None,angle=None):
    position=np.asarray(state.end_effector_position_m,float)
    orientation=rotation(state.end_effector_quaternion_wxyz)
    if axis is None:
        unit=np.asarray(direction,float);unit/=np.linalg.norm(unit)
        return position+unit*distance,orientation
    unit=np.asarray(axis,float);unit/=np.linalg.norm(unit)
    change=Rotation.from_rotvec(unit*angle)
    return np.asarray(pivot,float)+change.apply(position-pivot),change*orientation
