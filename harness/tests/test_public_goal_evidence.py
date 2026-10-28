"""Public goal observations support semantic verification without a task oracle."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest

from captain_harness.adapters.capx.public_goal_evidence import measure_goal_observations, collect_goal_evidence


def subject(name, center):
    center = np.asarray(center, float)
    return SimpleNamespace(selected=True, observation_handle='current', label=name,
        camera_name='head', centroid=center, bounds_min=center-.02,
        bounds_max=center+.02, point_count=100)


def measure(**changes):
    args = dict(observation_handle='current', time_s=2., current_time_s=2.,
        last_configuration_motion_time_s=1.,
        subjects={'a':subject('visible moving part',[.1,0,.1]),
                  'b':subject('visible fixed part',[0,0,.1])},
        claims=['The moving part is open.', 'The other requested condition is satisfied.'],
        previous={'source_observation_handle':'previous', 'observation_time_s':1.5})
    args.update(changes)
    return measure_goal_observations(**args)


def test_multi_goal_probe_declares_every_claim_and_full_task_question():
    value = measure()
    assert all(r['satisfied'] for r in value['verification_requirements'])
    questions = value['verification_questions']
    assert len(questions) == 3
    assert questions[0]['name'] == 'full_task_satisfied'
    assert 'entire original task' in questions[0]['prompt']
    assert [q['name'] for q in questions[1:]] == ['goal_claim_1','goal_claim_2']
    assert value['relation_evidence']['status'] == 'unassessed'
    assert 'relation_ready' not in value and 'public_relation_satisfied' not in value


def test_directional_measurements_report_public_intervals_in_explicit_frame():
    value = measure()
    pair = value['public_evidence']['pairwise_geometry'][0]
    assert pair['frame'] == 'robot_base'
    np.testing.assert_allclose(pair['visible_centroid_delta_m'], [.1,0,0])
    np.testing.assert_allclose(pair['visible_separation_bounds_m'][0], [.06,-.04,-.04])
    np.testing.assert_allclose(pair['visible_separation_bounds_m'][1], [.14,.04,.04])
    assert 'left' not in pair and 'open' not in pair


@pytest.mark.parametrize('previous', [None, {'source_observation_handle':'other','observation_time_s':2.},
    {'source_observation_handle':'other','observation_time_s':.5}])
def test_new_handles_or_pre_motion_samples_do_not_establish_settling_evidence(previous):
    value = measure(previous=previous)
    assert any(not r['satisfied'] for r in value['verification_requirements'])


def test_stale_current_observation_and_unselected_or_cross_observation_subjects_are_rejected():
    with pytest.raises(ValueError, match='current'):
        measure(current_time_s=3.)
    for field,new in [('selected',False),('observation_handle','old')]:
        s=subject('part',[0,0,0]);setattr(s,field,new)
        with pytest.raises(ValueError): measure(subjects={'a':s})


def test_historical_evidence_is_labeled_and_does_not_replace_current_samples():
    earlier=measure()
    value=measure(previous=None, supporting_evidence=[earlier])
    assert value['public_evidence']['historical_support'][0]['scope']=='historical_only'
    assert any(not r['satisfied'] for r in value['verification_requirements'])


def test_missing_metric_geometry_is_reported_without_inventing_a_goal_verdict():
    s=subject('part',[0,0,0]);s.centroid[0]=np.nan
    value=measure(subjects={'a':s})
    assert value['public_evidence']['subjects'][0]['geometry_status']=='insufficient'
    assert value['public_evidence']['subjects'][0]['visible_centroid_m'] is None
    assert value['relation_evidence']['status']=='unassessed'


def test_collector_resolves_immutable_episode_evidence_and_rejects_stale_inputs(tmp_path):
    from dataclasses import replace
    from test_tool_backend import _fixture, _v9_selected_detection
    from captain_harness.adapters.capx.artifacts import ArtifactHandleError
    f = _fixture(tmp_path, dehardcoded=True, v15_protocol=True)
    f.robot.observation = replace(f.robot.observation, time_s=1.)
    f.robot.time_s = 1.
    obs, target = _v9_selected_detection(f, v15_protocol=True)
    first = collect_goal_evidence(f.backend, obs['observation_handle'], [target['detection_id']], ['The fixture is open.'])
    assert first.success
    handle = first.value['evidence_handle']
    first.value['observation_time_s'] = -100
    assert f.backend.registry.get(handle, expected_kind='goal_evidence')['observation_time_s'] == 1.
    f.robot.observation = replace(f.robot.observation, time_s=2.)
    f.robot.time_s = 2.
    with pytest.raises(ValueError, match='stale'):
        collect_goal_evidence(f.backend, obs['observation_handle'], [target['detection_id']], ['The fixture is open.'])
    obs, target = _v9_selected_detection(f, v15_protocol=True)
    second = collect_goal_evidence(f.backend, obs['observation_handle'], [target['detection_id']], ['The fixture is open.'], handle)
    assert next(r for r in second.value['verification_requirements'] if r['name']=='separated_post_motion_samples')['satisfied']
    with pytest.raises(ArtifactHandleError):
        collect_goal_evidence(f.backend, obs['observation_handle'], [target['detection_id']], ['The fixture is open.'], target['detection_id'])


def test_runtime_goal_collection_preserves_real_history_and_tracks_configuration_changes(tmp_path):
    from dataclasses import replace
    from test_tool_backend import _fixture, _v9_selected_detection
    from captain_harness.adapters.capx.environment import CapXLiberoEnvironmentAdapter
    from captain_harness.adapters.capx.manifest import build_capx_tool_manifest
    from captain_harness.adapters.capx.runtime_manifest import runtime_manifest
    from captain_harness.ports.capability import CapabilityResult
    f = _fixture(tmp_path, dehardcoded=True, v15_protocol=True)
    f.robot.time_s = 1.
    f.robot.observation = replace(f.robot.observation,time_s=1.)
    env = object.__new__(CapXLiberoEnvironmentAdapter)
    env._tools = f.backend; env._robot = f.robot
    env._motion_tools = True
    env._manifest = runtime_manifest(build_capx_tool_manifest('S3',joint_count=7,
        service_aligned=True,dehardcoded=True,v15_protocol=True),controller='OSC_POSE',motion_tools=True)
    obs, target = _v9_selected_detection(f,v15_protocol=True)
    f.backend.gripper_state = lambda:CapabilityResult.ok({'object_detected':False})
    first = env.execute('measure_goal_evidence',{'observation_handle':obs['observation_handle'],
        'subject_detection_ids':[target['detection_id']], 'claims':['The visible mechanism is open.']})
    assert first.success, first.message
    actual = env.public_evidence_observations((obs['observation_handle'],'unknown'))
    assert list(actual) == [obs['observation_handle']]
    assert actual[obs['observation_handle']].images[0].rgb_path == f.robot.observation.cameras[0].rgb_path
    # Use real environment dispatch while substituting only robot execution.
    old = f.backend.invoke
    f.backend.invoke = lambda *args:CapabilityResult.ok({})
    for name, expected in [('hold_position',0.),('open_gripper',3.),('hold_position',3.)]:
        f.robot.time_s = 3.
        assert env.execute(name,{} if name=='open_gripper' else {'duration_s':.2}).executed
        assert f.backend._last_configuration_motion_time_s == expected
    f.backend.invoke = old


def test_satisfied_noop_preserves_last_configuration_motion_time(tmp_path):
    from test_tool_backend import _fixture
    from captain_harness.adapters.capx.environment import CapXLiberoEnvironmentAdapter
    from captain_harness.adapters.capx.manifest import build_capx_tool_manifest
    from captain_harness.adapters.capx.runtime_manifest import runtime_manifest
    from captain_harness.ports.capability import CapabilityResult, FailureCategory
    f=_fixture(tmp_path,dehardcoded=True,v15_protocol=True)
    env=object.__new__(CapXLiberoEnvironmentAdapter)
    env._tools=f.backend;env._robot=f.robot;env._motion_tools=True
    env._manifest=runtime_manifest(build_capx_tool_manifest('S3',joint_count=7,
        service_aligned=True,dehardcoded=True,v15_protocol=True),controller='OSC_POSE',motion_tools=True)
    f.robot.simulation_steps=10;f.robot.time_s=2.;f.backend._last_configuration_motion_time_s=1.
    f.backend.invoke=lambda *args:CapabilityResult.ok(None)
    assert env.execute('open_gripper',{}).success
    assert f.backend._last_configuration_motion_time_s==1.
    def partial(*args):
        f.robot.simulation_steps+=3;f.robot.time_s=2.15
        return CapabilityResult.failed(FailureCategory.TIMEOUT,'partial release',executed=True)
    f.backend.invoke=partial
    assert env.execute('open_gripper',{}).executed
    assert f.backend._last_configuration_motion_time_s==2.15
