from types import SimpleNamespace
import pytest
from jsonschema import Draft202012Validator, ValidationError
from captain_harness.adapters.capx.contact_tools import ContactToolMixin, contact_motion_capabilities
from captain_harness.ports.capability import CapabilityResult


def test_declared_target_contact_requires_public_binding_before_any_motion():
    backend = ContactToolMixin()
    backend.robot = SimpleNamespace(move_linear=lambda *a, **kw: pytest.fail("unbound contact executed"))
    with pytest.raises(ValueError, match="target_detection_id"):
        backend.move_linear([1, 0, 0], .02, motion_intent="target_contact")
    spec = next(s for s in contact_motion_capabilities() if s.name == "move_linear")
    with pytest.raises(ValidationError):
        Draft202012Validator(spec.arguments_schema).validate({
            "direction": [1, 0, 0], "distance_m": .02, "motion_intent": "target_contact"})


def test_free_motion_stays_unbound_and_does_not_claim_target_progress():
    backend = ContactToolMixin()
    backend.robot = SimpleNamespace(move_linear=lambda *a, **kw: CapabilityResult.ok({"reached": True}))
    result = backend.move_linear([1, 0, 0], .02, motion_intent="free_motion")
    assert result.value["motion_intent"] == "free_motion"
    assert result.value["public_effect"]["status"] == "unknown"


def test_contact_intent_routes_to_monitor_and_publishes_generic_effect(monkeypatch):
    from captain_harness.adapters.capx import observed_contact
    calls = []
    def observed(backend, target, **kwargs):
        calls.append(target)
        return CapabilityResult.ok({"target_progress": {"status": "progress_observed"}})
    monkeypatch.setattr(observed_contact, "observed_contact", observed)
    backend = ContactToolMixin()
    result = backend.move_linear([1, 0, 0], .02, target_detection_id="selected", motion_intent="target_contact")
    assert calls == ["selected"]
    assert result.value["public_effect"]["status"] == "progress_observed"
    assert result.value["motion_intent"] == "target_contact"


def test_public_target_comparison_follows_links_but_separates_corrected_identity():
    backend = ContactToolMixin()
    backend._selection_history = [
        {"detection_id": "a", "identity_update": "initial_selection", "previous_detection_id": None},
        {"detection_id": "b", "identity_update": "same_target", "previous_detection_id": "a"},
        {"detection_id": "c", "identity_update": "corrected_target", "previous_detection_id": "b"},
        {"detection_id": "d", "identity_update": "same_target", "previous_detection_id": "c"},
    ]
    result = CapabilityResult.ok({"target_progress": {"status": "no_observed_target_progress"}})
    assert backend._contact_result(result, "b").value["public_effect"]["target_ref"] == "a"
    assert backend._contact_result(result, "d").value["public_effect"]["target_ref"] == "c"
