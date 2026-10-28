"""Adapter mixin retaining measured, unclosed descent routes."""
from dataclasses import replace
from .contact_motion import _pose, _quaternion


class DescentRecordingMixin:
    def descend_and_close(self, *args, **kwargs):
        if getattr(self, '_descent_recording', None) is not None:
            raise RuntimeError('nested descent recording is unsupported')
        recording = {'poses': [_pose(self.get_robot_state().value).tolist()],
                     'start_step': self.simulation_steps, 'last_step': self.simulation_steps,
                     'closure_started': False, 'forces_n': [self._eef_force_norm()],
                     'steps': [self.simulation_steps]}
        self._descent_recording = recording
        try:
            result = super().descend_and_close(*args, **kwargs)
        finally:
            self._sample_descent_pose()
            self._descent_recording = None
            self._last_descent_recording = recording
            if self.simulation_steps > recording['start_step'] and not recording['closure_started']:
                self._last_guarded_motion = {'poses': recording['poses'], 'end_step': self.simulation_steps}
        if isinstance(result.value, dict):
            import numpy as np
            trajectory = [{'step': step, 'eef_position_m': np.asarray(pose)[:3,3].tolist(),
                           'eef_quaternion_wxyz': _quaternion(np.asarray(pose)).tolist(), 'force_n': force}
                          for pose, force, step in zip(recording['poses'], recording['forces_n'], recording['steps'])]
            available = (not recording['closure_started'] and len(trajectory) > 1)
            return replace(result, value={**result.value, 'controller_execution': {
                'trajectory': trajectory, 'retreat_available': available,
                'control_steps': recording['last_step']-recording['start_step'],
                'scope': 'Measured descent samples before gripper closure; later actions invalidate the retreat route.'}})
        return result

    def _sample_descent_pose(self):
        recording = getattr(self, '_descent_recording', None)
        if (recording is not None and not recording['closure_started']
                and recording['last_step'] != self.simulation_steps):
            recording['poses'].append(_pose(self.get_robot_state().value).tolist())
            recording['forces_n'].append(self._eef_force_norm())
            recording['steps'].append(self.simulation_steps)
            recording['last_step'] = self.simulation_steps

    def _move_to_pose_osc(self, *args, **kwargs):
        try:
            return super()._move_to_pose_osc(*args, **kwargs)
        finally:
            # The inherited descent loop calls this with max_steps=1.
            self._sample_descent_pose()

    def _set_gripper(self, *args, **kwargs):
        recording = getattr(self, '_descent_recording', None)
        if recording is not None:
            recording['closure_started'] = True
        return super()._set_gripper(*args, **kwargs)
