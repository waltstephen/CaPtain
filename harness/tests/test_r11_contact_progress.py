from types import SimpleNamespace
import numpy as np
import pytest

from captain_harness.ports.capability import CapabilityResult, FailureCategory


@pytest.mark.parametrize('measurement,expected', [
    ({'status':'insufficient'}, 'target_motion_unresolved'),
    ({'status':'image_only','translation_m':None}, 'target_motion_unresolved'),
    ({'status':'supported','translation_m':[0,0,0],'motion_spread_m':.0001}, 'no_observed_target_progress'),
    ({'status':'supported','translation_m':[.006,0,0],'motion_spread_m':.0001}, 'progress_observed'),
])
def test_visual_unknown_never_becomes_zero_motion(measurement,expected):
    from captain_harness.adapters.capx.observed_contact import assess_progress
    assert assess_progress(measurement, requested_direction=[1,0,0])['status']==expected


def test_opposite_motion_is_reported_instead_of_counted_as_forward_progress():
    from captain_harness.adapters.capx.observed_contact import assess_progress
    value=assess_progress({'status':'supported','translation_m':[-.01,0,0],'motion_spread_m':.0001},
        requested_direction=[1,0,0])
    assert value['status']=='opposite_target_progress'


def test_rotation_below_visual_resolution_is_unresolved_not_a_mechanism_stall():
    from captain_harness.adapters.capx.observed_contact import assess_progress
    value=assess_progress({'status':'supported','rotation_vector_rad':[0,0,.04],
        'rotation_resolution_rad':.067},rotation_axis=[0,0,1],rotation_sign=1,
        requested_segment_motion=.075)
    assert value['status']=='target_motion_unresolved'


def test_segmented_contact_stops_without_retry_and_preserves_real_driver_failure():
    from captain_harness.adapters.capx.observed_contact import run_segments
    calls=[]
    def move(fraction, remaining, force):
        calls.append((fraction,remaining,force))
        return CapabilityResult.failed(FailureCategory.CONTROLLER_REJECTED,'contact_force',executed=True,
            value={'control_steps':3,'stopped_reason':'contact_force','observed_force_n':18.2,
                   'actual_displacement_m':[.002,0,0], 'trajectory':[{'step':1}]})
    result=run_segments(move, lambda:{'status':'insufficient'},segments=3,
        duration_s=10, initial_force_n=5, max_force_n=18, time_s=lambda:0,
        progress_arguments={'requested_direction':[1,0,0]})
    assert len(calls)==1 and calls[0][2]==10
    assert result.failure_category is FailureCategory.CONTROLLER_REJECTED
    assert result.value['stopped_reason']=='contact_force'
    assert result.value['target_progress']['status']=='target_motion_unresolved'
    assert result.value['control_steps']==3


def test_segmented_contact_checks_incremental_progress_and_total_time_budget():
    from captain_harness.adapters.capx.observed_contact import run_segments
    clock=[0.]; calls=[]
    def move(fraction,remaining,force):
        calls.append((fraction,remaining)); clock[0]+=.5
        return CapabilityResult.ok({'control_steps':10,'actual_displacement_m':[.01,0,0],
            'stopped_reason':'target_reached','observed_force_n':6})
    # Both captures report the same accumulated translation: the second segment
    # moved the hand but made no additional visible target progress.
    result=run_segments(move,lambda:{'status':'supported','translation_m':[.007,0,0], 'motion_spread_m':.0001},
        segments=3,duration_s=10,initial_force_n=5,max_force_n=18,time_s=lambda:clock[0],
        progress_arguments={'requested_direction':[1,0,0]})
    assert len(calls)==2
    assert calls[1][1]==9.5
    assert result.executed and not result.success
    assert result.value['stopped_reason']=='no_observed_target_progress'
    assert result.value['control_steps']==20
    np.testing.assert_allclose(result.value['actual_displacement_m'],[.02,0,0])
