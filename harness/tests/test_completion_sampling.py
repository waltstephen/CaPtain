import pytest

from captain_harness.adapters.capx.completion_sampling import compare_completion_samples


def measure(history, handle, time, x=0., released=True):
    return compare_completion_samples(history, observation_handle=handle,
        centroid=[x, 0., 0.], time_s=time, released=released)


def test_paused_remeasurement_preserves_prior_time_and_recomputes_localization_change():
    history, _ = measure([], 'a', 43.25)
    history, result = measure(history, 'b', 43.75, .071)
    assert result['temporal_comparison_available']
    history, again = measure(history, 'c', 43.75, .002)
    assert again['temporal_comparison_available']
    assert again['comparison_time_delta_s'] == .5
    assert again['target_static_after_release_m'] == pytest.approx(.002)
    assert again['comparison_source_observation_handle'] == 'a'
    assert [s['observation_handle'] for s in history] == ['a', 'c']


def test_repeated_paused_views_never_create_elapsed_time():
    history, _ = measure([], 'a', 10.)
    for handle in ['b', 'c', 'd']:
        history, result = measure(history, handle, 10.)
        assert not result['temporal_comparison_available']
        assert result['comparison_time_delta_s'] == 0.
        assert len(history) == 1


def test_new_time_compares_latest_distinct_sample_without_accumulating_history():
    history, _ = measure([], 'a', 1.)
    history, _ = measure(history, 'b', 2., .01)
    history, result = measure(history, 'c', 3., .011)
    assert result['comparison_source_observation_handle'] == 'b'
    assert result['target_static_after_release_m'] == pytest.approx(.001)
    assert [s['time_s'] for s in history] == [2., 3.]


def test_attachment_then_release_requires_new_released_time_pair():
    history, _ = measure([], 'a', 1.)
    history, _ = measure(history, 'b', 2.)
    history, result = measure(history, 'held', 2., released=False)
    assert not result['temporal_comparison_available'] and len(history) == 1
    history, result = measure(history, 'released', 3.)
    assert not result['temporal_comparison_available']
    history, result = measure(history, 'settled', 4.)
    assert result['temporal_comparison_available']
    assert result['comparison_source_observation_handle'] == 'released'


def test_unknown_or_older_clock_cannot_prove_coverage_or_roll_back_history():
    history, _ = measure([], 'a', 2.)
    history, _ = measure(history, 'b', 3.)
    unchanged, result = measure(history, 'old', 1.)
    assert unchanged == history and not result['temporal_comparison_available']
    history, result = measure(history, 'unknown', None)
    assert not result['temporal_comparison_available'] and len(history) == 1
    history, result = measure(history, 'known', 4.)
    assert not result['temporal_comparison_available']


def test_repeated_original_handle_can_use_a_separate_earlier_time():
    history, _ = measure([], 'a', 1.)
    history, _ = measure(history, 'b', 2.)
    _, result = measure(history, 'b', 2.)
    assert result['temporal_comparison_available']
    assert result['comparison_source_observation_handle'] == 'a'
