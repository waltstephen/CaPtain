"""Eligible own-episode evidence survives lost variables and repeated paused reads."""
from types import SimpleNamespace
from copy import deepcopy
import numpy as np
import pytest
from captain_harness.adapters.capx.artifacts import EpisodeArtifactRegistry, ArtifactHandleError
from captain_harness.adapters.capx.public_goal_evidence import collect_goal_evidence
from captain_harness.ports.capability import CapabilityResult


def backend():
    b=SimpleNamespace(registry=EpisodeArtifactRegistry('goal-cache',ttl_s=60.),
        robot=SimpleNamespace(time_s=1.),_last_configuration_motion_time_s=0.,
        _active_attachment_handle=None,_selection_history=[],_public_observation_history={})
    def selected(handle,observation_handle=None):
        result=b.registry.get(handle,expected_kind='detection')
        if observation_handle is not None and result.observation_handle!=observation_handle:raise ValueError('wrong source')
        return result
    b._require_selected_detection=selected
    b.gripper_state=lambda:CapabilityResult.ok({'object_detected':False,'open_fraction':1.})
    return b


def sample(b,t,previous=None,correspondence='supported',update='same_target',claims=None,association='supported'):
    b.robot.time_s=t
    obs=b.registry.put('observation',SimpleNamespace(time_s=t))
    b._public_observation_history[obs]=SimpleNamespace(step=int(t*20))
    detection=b.registry.put('detection',SimpleNamespace(selected=True,observation_handle=obs,
        label='public item',camera_name='head',centroid=np.zeros(3),bounds_min=np.full(3,-.02),
        bounds_max=np.full(3,.02),point_count=100))
    b._selection_history.append({'detection_id':detection,'previous_detection_id':previous,
        'identity_update':update if previous else 'initial_selection',
        'visual_correspondence':{'status':correspondence,'association_status':association},'initial_observation_handle':
            b._selection_history[0]['initial_observation_handle'] if previous else obs})
    result=collect_goal_evidence(b,obs,[detection],claims or ['The requested relation holds.'])
    return result.value,detection


def ready(value):return next(r['satisfied'] for r in value['verification_requirements']
    if r['name']=='separated_post_motion_samples')


def test_matching_prior_measurement_is_reused_with_its_real_frame():
    b=backend();first,a=sample(b,1.);second,_=sample(b,1.5,a)
    assert not ready(first) and ready(second)
    assert second['previous_evidence_resolution']['mode']=='automatic'
    assert second['previous_evidence_resolution']['evidence_handle']==first['evidence_handle']
    assert second['public_evidence']['previous_observation_handle']==first['source_observation_handle']
    assert any(s['source_id']==first['source_observation_handle'] for s in second['visual_evidence_sources'])
    assert second['relation_evidence']['status']=='unassessed'


@pytest.mark.parametrize('case',['unlinked','uncertain','corrected','motion','same_time','changed_claims','loaded','missing_frame','wrong_selection','unknown_selection'])
def test_ineligible_history_is_not_borrowed(case):
    b=backend()
    if case=='loaded':b.gripper_state=lambda:CapabilityResult.ok({'object_detected':True})
    first,a=sample(b,1.)
    b.gripper_state=lambda:CapabilityResult.ok({'object_detected':False})
    if case=='motion':b._last_configuration_motion_time_s=1.25
    if case=='missing_frame':b._public_observation_history.clear()
    second,_=sample(b,1. if case=='same_time' else 1.5,None if case=='unlinked' else a,
        correspondence='insufficient' if case=='uncertain' else 'supported',
        update='corrected_target' if case=='corrected' else 'same_target',
        claims=['A different claim.'] if case=='changed_claims' else None,
        association='contradicted' if case=='wrong_selection' else None if case=='unknown_selection' else 'supported')
    assert not ready(second)
    assert second['previous_evidence_resolution']['mode']=='unavailable'


def test_paused_remeasurement_does_not_evict_the_distinct_prior_sample():
    b=backend();first,a=sample(b,1.)
    for _ in range(70):
        result,a=sample(b,1.5,a)
        assert ready(result)
        assert result['public_evidence']['previous_observation_handle']==first['source_observation_handle']
    assert len(b._goal_sample_index)==2


def test_explicit_invalid_reference_is_not_replaced_from_cache():
    b=backend();first,a=sample(b,1.);second,c=sample(b,1.5,a)
    with pytest.raises(ArtifactHandleError):
        collect_goal_evidence(b,second['source_observation_handle'],[c],['The requested relation holds.'],a)
