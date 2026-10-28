"""Candidate guard must survive a driver replacing the outer motion guard."""
import pytest

from captain_harness.adapters.libero.joint_path_guard import GuardedJointPathMixin
from captain_harness.ports.capability import CapabilityResult


class Driver:
    simulation_steps = 0
    _motion_guard = None

    def __init__(self, forces):
        self.forces = forces

    def _eef_force_norm(self):
        return self.forces[min(self.simulation_steps, len(self.forces)-1)]

    def _step_action(self, action):
        if not self._motion_guard():
            return False
        self.simulation_steps += 1
        return True

    def execute_joint_path(self, path, collision):
        outer = self._motion_guard
        self._motion_guard = lambda: not collision(self.simulation_steps)['collision']
        try:
            for q in path:
                if not self._step_action(q):
                    return CapabilityResult.failed('controller_rejected', 'driver_guard_stopped',
                                                    executed=self.simulation_steps > 0)
            return CapabilityResult.ok({'reason': 'path_executed'})
        finally:
            self._motion_guard = outer


class Robot(GuardedJointPathMixin, Driver):
    pass


def free(_):
    return {'collision': False}


def test_outer_guard_is_not_bypassed_when_path_installs_its_own_guard():
    robot = Robot([5.]*8)
    outer = lambda: robot.simulation_steps < 2
    robot._motion_guard = outer
    result = robot.execute_joint_path(range(7), free)
    assert not result.success and result.executed
    assert result.message == 'outer_motion_guard_rejected'
    assert robot.simulation_steps == 2 and robot._motion_guard is outer
    assert robot._joint_path_guard_context is None


@pytest.mark.parametrize('forces', [[5., 5., 18.], [5., 5., 10.], [5., 5., None]])
def test_final_sample_stop_has_priority_over_endpoint_success(forces):
    robot = Robot(forces)
    result = robot.execute_joint_path(range(2), free)
    assert not result.success and result.executed
    assert robot.simulation_steps == 2
    assert len(result.value['path_measurement_guard']['samples']) == 3


@pytest.mark.parametrize('force', [None, float('nan'), 18.])
def test_initial_force_rejection_executes_nothing(force):
    robot = Robot([force])
    result = robot.execute_joint_path(range(3), free)
    assert not result.success and not result.executed
    assert robot.simulation_steps == 0


def test_path_collision_still_stops_even_when_measurements_are_good():
    robot = Robot([5.]*8)
    result = robot.execute_joint_path(range(7), lambda step: {'collision': step >= 1})
    assert not result.success and robot.simulation_steps == 1
    assert result.message == 'driver_guard_stopped'


def test_good_path_preserves_success_and_all_cycle_samples():
    robot = Robot([5.]*8)
    result = robot.execute_joint_path(range(7), free)
    assert result.success and robot.simulation_steps == 7
    assert len(result.value['path_measurement_guard']['samples']) == 8
