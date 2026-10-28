"""Exact episode-local closure/effect association; no strategy or hidden state."""
from collections.abc import Mapping
from dataclasses import replace
from uuid import uuid4


class AttemptEvidenceBinding:
    def __init__(self):
        self.active = None

    def annotate(self, backend, spec, arguments, result, identity):
        value = result.value if isinstance(result.value, Mapping) else {}
        origin = spec.name in {"descend_and_close", "approach_and_close", "close_gripper"}
        if origin and result.executed and value.get("closure_executed") is not False:
            # Even a direct, unbound reclosure ends the previous association.
            self.active = None
            track = (value.get("associated_tracking_handle") if spec.name == "close_gripper" else
                value.get("tracking_handle") or getattr(backend, "_pending_attachment_handle", None))
            reference = "attempt://" + uuid4().hex
            if isinstance(track, str) and isinstance(identity, Mapping):
                identity = {**identity, "attempt_reference": reference, "attempt_role": "origin"}
                self.active = {"reference": reference, "tracking_handle": track}
            else:
                identity = {"reference": reference, "attempt_reference": reference, "attempt_role": "origin",
                    "basis": "unbound_closure", "semantic_identity_status": "unverified",
                    "association_status": "unbound",
                    "scope": "Executed closure recorded; no target association inferred."}
                self.active = {"reference": reference, "tracking_handle": None}
            return replace(result, attempt_identity=identity)
        assessment = None
        active = self.active
        if active is not None:
            monitor = value.get("possession_monitor", {})
            if isinstance(monitor, Mapping) and monitor.get("status") in {"lost", "not_present", "unavailable"}:
                assessment = {"attempt_reference": active["reference"],
                    "status": "unknown" if monitor["status"] == "unavailable" else "contradicted",
                    "evidence": {"scope": "continued_possession", "possession_monitor": {
                        k: v for k, v in monitor.items() if k != "aperture_samples"},
                        "target_identity_verified": False}}
                if monitor["status"] != "unavailable":
                    self.active = None
            elif (spec.name == "assert_entity_attached" and result.success
                    and isinstance(active["tracking_handle"], str)
                    and arguments.get("tracking_handle") == active["tracking_handle"]):
                status = ("progress_observed" if value.get("verified") is True else
                    "contradicted" if value.get("reason") in {"closed_on_nothing", "lift_probe_contradicted_attachment"}
                    else "unknown")
                assessment = {"attempt_reference": active["reference"], "status": status,
                    "evidence": {"scope": "target_attachment_at_observation", **{
                        k: value[k] for k in ("tracking_handle", "current_point_handle", "verified", "reason",
                            "target_attachment_status", "displacement_m", "vertical_displacement_m",
                            "min_displacement_m", "min_vertical_displacement_m", "motion_diagnosis",
                            "quality_flags", "attachment_evidence_gaps") if k in value}}}
            if spec.name == "open_gripper" and result.executed:
                assessment = {"attempt_reference": active["reference"], "status": "unknown",
                    "evidence": {"scope": "association_ended", "reason": "release_executed",
                        "prior_attachment_is_historical": True, "task_effect": "unassessed"}}
                self.active = None
        return replace(result, attempt_identity=identity, attempt_evidence=assessment)
