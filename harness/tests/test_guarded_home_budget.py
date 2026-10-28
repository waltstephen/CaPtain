"""The driver's doubled home budget must not become an invalid public duration."""
from captain_harness.adapters.libero.backend import LiberoActionBackend
from test_libero_adapter import FakeOscEnv


def test_guarded_home_accepts_the_standard_480_step_driver_configuration(tmp_path):
    env = FakeOscEnv()
    robot = LiberoActionBackend(env, observation_dir=tmp_path,
        initial_observation=env.reset(), horizon=15000, controller='OSC_POSE',
        motion_max_steps=480, cartesian_max_steps=480, guarded_cartesian_motion=True)
    assert robot.move_linear([1, 0, 0], .02).success
    result = robot.home()
    assert result.success, result.message
    assert result.value['elapsed_s'] <= 30.


def test_public_contact_duration_limit_remains_strict(tmp_path):
    env = FakeOscEnv()
    robot = LiberoActionBackend(env, observation_dir=tmp_path,
        initial_observation=env.reset(), horizon=15000, controller='OSC_POSE',
        motion_max_steps=480, cartesian_max_steps=480, guarded_cartesian_motion=True)
    result = robot.move_linear([1, 0, 0], .02, max_duration_s=48.)
    assert not result.success and not result.executed
    assert not env.actions
