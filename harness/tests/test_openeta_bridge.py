from __future__ import annotations

from types import SimpleNamespace

import pytest

from captain_harness.adapters.capx.openeta_bridge import (
    dispatch_openeta_request,
    ensure_numpy_checkpoint_compatibility,
    openeta_health_payload,
)


class FakeAnyGrasp:
    def detect_grasps(self, **arguments: object):
        return {"success": True, "details": dict(arguments)}


def test_numpy_checkpoint_compatibility_installs_missing_private_alias() -> None:
    multiarray = object()
    private_core = SimpleNamespace()
    numpy_module = SimpleNamespace(
        _core=private_core,
        core=SimpleNamespace(multiarray=multiarray),
    )

    assert ensure_numpy_checkpoint_compatibility(numpy_module) is True
    assert private_core.multiarray is multiarray
    assert ensure_numpy_checkpoint_compatibility(numpy_module) is False


def test_openeta_bridge_dispatches_only_the_backend_path() -> None:
    result = dispatch_openeta_request(
        "anygrasp",
        FakeAnyGrasp(),
        "/detect_grasps",
        {"mode": "targeted"},
    )

    assert result["success"] is True
    assert result["details"]["mode"] == "targeted"
    with pytest.raises(ValueError, match="unsupported path"):
        dispatch_openeta_request(
            "anygrasp",
            FakeAnyGrasp(),
            "/predict_grasps",
            {},
        )


def test_openeta_bridge_health_records_pinned_revision() -> None:
    payload = openeta_health_payload(
        "anygrasp",
        FakeAnyGrasp(),
        openeta_revision="revision",
    )

    assert payload == {
        "ok": True,
        "backend": "anygrasp",
        "openeta_revision": "revision",
        "settings": {},
    }
