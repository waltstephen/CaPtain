from types import SimpleNamespace as NS
import pytest
from captain_harness.adapters.capx.artifacts import EpisodeArtifactRegistry
from captain_harness.adapters.capx.attempt_identity import resolve_attempt_identity
from captain_harness.ports.capability import SideEffect


def fixture():
    backend=NS(registry=EpisodeArtifactRegistry("test",ttl_s=None),_selection_history=[])
    first=backend.registry.put("detection",NS(selected=True))
    backend._selection_history.append({"detection_id":first})
    second=backend.registry.put("detection",NS(selected=True))
    backend._selection_history.append({"detection_id":second,"previous_detection_id":first,
        "identity_update":"same_target","visual_correspondence":{"status":"supported"}})
    track=backend.registry.put("entity_track",NS(source_detection_id=second))
    backend._active_attachment_handle=track
    return backend,first,second,track


def spec(name="lift_probe",effect=SideEffect.ROBOT_MOTION):return NS(name=name,side_effect=effect)


def test_issued_grasp_track_and_same_target_correspondence_share_reference():
    b,first,second,track=fixture()
    grasp=b.registry.put("grasp_candidate",{"detection_id":second})
    for s,a in [(spec(),{}),(spec(),{"tracking_handle":track}),
            (spec("approach_and_close"),{"candidate":grasp})]:
        identity=resolve_attempt_identity(b,s,a)
        assert identity["reference"]==first and identity["semantic_identity_status"]=="unverified"


@pytest.mark.parametrize("change",[{"identity_update":"corrected_target"},
    {"visual_correspondence":{"status":"uncertain"}}, {"visual_correspondence":{}}])
def test_correction_or_missing_correspondence_starts_separate_history(change):
    b,first,second,track=fixture();b._selection_history[-1].update(change)
    assert resolve_attempt_identity(b,spec(),{})["reference"]==second


def test_invalid_explicit_reference_cannot_borrow_active_target():
    b,*_=fixture()
    assert resolve_attempt_identity(b,spec(),{"tracking_handle":"bad"}) is None
    assert resolve_attempt_identity(b,spec(),{"tracking_handle":""}) is None
    assert resolve_attempt_identity(b,spec(),{"target_detection_id":"bad"}) is None
    assert resolve_attempt_identity(b,spec(effect=SideEffect.READ_ONLY),{}) is None


def test_unissued_candidate_dictionary_cannot_assert_an_attempt_identity():
    b,_,second,_=fixture()
    assert resolve_attempt_identity(b,spec("approach_and_close"),{"candidate":{"detection_id":second}}) is None


def test_adapter_captures_reference_before_execution_invalidates_registry(monkeypatch):
    from captain_harness.adapters.capx.environment import CapXLiberoEnvironmentAdapter
    from captain_harness.adapters.capx import possession_tools,grasp_preparation
    from captain_harness.ports.capability import CapabilityManifest,CapabilitySpec,CapabilityResult
    b,first,_,_=fixture();b.invoke=lambda *args:None
    adapter=CapXLiberoEnvironmentAdapter.__new__(CapXLiberoEnvironmentAdapter)
    adapter._tools=b;adapter._motion_tools=True;adapter._coding_control=False
    manifest=CapabilityManifest("test/v1","test",(CapabilitySpec(name="lift_probe",description="test",
        arguments_schema={"type":"object","properties":{}},result_schema={},units={},coordinate_frames={},
        timeout_s=1,side_effect=SideEffect.ROBOT_MOTION),))
    adapter.capabilities=lambda:manifest
    adapter._require_robot=lambda:NS(time_s=1)
    def execute(*args):
        b.registry.close()
        return CapabilityResult.ok(None)
    monkeypatch.setattr(possession_tools,"invoke_with_possession_evidence",execute)
    monkeypatch.setattr(grasp_preparation,"add_grasp_preparation",lambda b,n,r:r)
    result=adapter.execute("lift_probe",{})
    assert result.success and result.attempt_identity["reference"]==first


def test_closure_comparison_contains_selected_geometry_without_handle_noise():
    b,first,second,track=fixture()
    candidate={'detection_id':second,'position_m':[.1,.2,.3],
        'quaternion_wxyz':[1.,0,0,0],'approach_direction_world':[0,0,-1],
        'grasp_handle':'opaque-change-only'}
    b._aligned_pregrasp=NS(detection_id=second,tracking_handle=track,candidate=candidate)
    identity=resolve_attempt_identity(b,spec('descend_and_close'),{})
    parameters=identity['execution_parameters']
    assert parameters['position_m']==candidate['position_m']
    assert parameters['quaternion_wxyz']==candidate['quaternion_wxyz']
    assert 'grasp_handle' not in parameters and 'detection_id' not in parameters
    assert identity['semantic_identity_status']=='unverified'
