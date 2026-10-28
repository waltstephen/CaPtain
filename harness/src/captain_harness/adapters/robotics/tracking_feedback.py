"""Public measured tracking in the robot world frame; shared across conditions."""
import numpy as np
from scipy.spatial.transform import Rotation

class DirectionalProgressGuard:
    def __init__(self, *, position_tolerance_m, orientation_tolerance_rad, window=10):
        self.pt=position_tolerance_m
        self.at=orientation_tolerance_rad
        self.window=window
        self.count=np.zeros(3,dtype=int)
        self.sign=np.zeros(3)
        self.angular_count=0
        from collections import deque
        self.history=deque(maxlen=window)

    def observe(self, before, after, target, *, position_only=False):
        residual=target[:3,3]-before[:3,3]
        direction=np.sign(residual)
        # Relative to the SAME issued reference: orthogonal motion cannot mask blockage.
        progress=np.abs(residual)-np.abs(target[:3,3]-after[:3,3])
        active=np.abs(target[:3,3]-after[:3,3]) > self.pt
        blocked=active & (progress <= .001*self.pt)
        self.count=np.where(blocked, np.where(direction==self.sign,self.count+1,1), 0)
        self.sign=direction
        old_angle=Rotation.from_matrix(target[:3,:3]@before[:3,:3].T).magnitude()
        angle=Rotation.from_matrix(target[:3,:3]@after[:3,:3].T).magnitude()
        self.angular_count=(self.angular_count+1 if not position_only and angle>self.at
            and old_angle-angle <= .001*self.at else 0)
        self.history.append((after.copy(), target.copy()))
        accumulating=np.zeros(3,dtype=bool)
        growth=None
        angular_accumulating=False
        if len(self.history)==self.window:
            first_pose, first_target=self.history[0]
            commanded=target[:3,3]-first_target[:3,3]
            measured=after[:3,3]-first_pose[:3,3]
            remaining=target[:3,3]-after[:3,3]
            old_remaining=first_target[:3,3]-first_pose[:3,3]
            same_direction=np.all([np.sign(t[:3,3]-p[:3,3])==np.sign(remaining)
                for p,t in self.history],axis=0)
            # Growing backlog needs evidence of poor tracking relative to the
            # issued increment, not merely motion smaller than an absolute scale.
            # This prevents slow but meaningful multi-axis progress being called blocked.
            accumulating=(same_direction & (np.abs(commanded)>self.pt)
                & (measured*np.sign(commanded)<np.minimum(self.pt,.2*np.abs(commanded)))
                & (np.abs(remaining)-np.abs(old_remaining)>self.pt))
            requested_turn=Rotation.from_matrix(target[:3,:3]@first_target[:3,:3].T).magnitude()
            measured_turn=Rotation.from_matrix(after[:3,:3]@first_pose[:3,:3].T).magnitude()
            first_error=Rotation.from_matrix(first_target[:3,:3]@first_pose[:3,:3].T).magnitude()
            angular_accumulating=(not position_only and requested_turn>self.at
                and measured_turn<min(self.at,.2*requested_turn) and angle-first_error>self.at)
            growth={"commanded_rotation_increment_rad":float(requested_turn),
                "measured_rotation_increment_rad":float(measured_turn),
                "rotation_error_growth_rad":float(angle-first_error),
                "commanded_increment_m":commanded.tolist(),
                "measured_increment_m":measured.tolist(),
                "residual_growth_m":(np.abs(remaining)-np.abs(old_remaining)).tolist()}
        axes=[name for name,n,g in zip("xyz",self.count,accumulating) if n>=self.window or g]
        if axes or self.angular_count>=self.window or angular_accumulating:
            return {"translation_axes":axes, "rotation_blocked":self.angular_count>=self.window or angular_accumulating,
                "consecutive_samples":self.window, "window_progress":growth,
                "command_minus_measured_m":(target[:3,3]-after[:3,3]).tolist(),
                "orientation_error_rad":float(angle),
                "basis":"directional residual stalled or commanded progress accumulated beyond measured progress; contact cause unknown"}
        return None

def tracking_feedback(start, final, requested, commanded, *, stop_reason, blocked=None):
    return {
        "source":"public robot pose measurements and issued reference",
        "frame":"robot_base", "requested_displacement_m":(requested[:3,3]-start[:3,3]).tolist(),
        "commanded_displacement_m":(commanded[:3,3]-start[:3,3]).tolist(),
        "measured_displacement_m":(final[:3,3]-start[:3,3]).tolist(),
        "remaining_displacement_m":(requested[:3,3]-final[:3,3]).tolist(),
        "command_minus_measured_m":(commanded[:3,3]-final[:3,3]).tolist(),
        "requested_rotation_vector_rad":Rotation.from_matrix(requested[:3,:3]@start[:3,:3].T).as_rotvec().tolist(),
        "measured_rotation_vector_rad":Rotation.from_matrix(final[:3,:3]@start[:3,:3].T).as_rotvec().tolist(),
        "remaining_rotation_vector_rad":Rotation.from_matrix(requested[:3,:3]@final[:3,:3].T).as_rotvec().tolist(),
        "stop_reason":stop_reason, "blocked_directions":blocked,
        "causal_scope":"measured tracking only; does not identify the obstacle or object effect"}
