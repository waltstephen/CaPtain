from __future__ import annotations

from typing import Mapping, Protocol


class OpenEtaBackend(Protocol):
    def predict_grasps(self, **arguments: object) -> Mapping[str, object]: ...


def ensure_numpy_checkpoint_compatibility(
    numpy_module: object,
) -> bool:
    private_core = getattr(numpy_module, "_core", None)
    public_core = getattr(numpy_module, "core", None)
    public_multiarray = getattr(public_core, "multiarray", None)
    if (
        private_core is None
        or hasattr(private_core, "multiarray")
        or public_multiarray is None
    ):
        return False
    setattr(private_core, "multiarray", public_multiarray)
    return True


def dispatch_openeta_request(
    kind: str,
    backend: object,
    path: str,
    payload: Mapping[str, object],
) -> Mapping[str, object]:
    normalized_kind = str(kind).strip().casefold()
    normalized_path = "/" + path.strip().strip("/")
    expected = {
        "contact_graspnet": "/predict_grasps",
        "anygrasp": "/detect_grasps",
        "graspgenx": "/predict_grasps",
        "anyplace": "/predict_placement",
        "unidepth_v2": "/estimate_depth",
    }
    if normalized_kind not in expected:
        raise ValueError(f"unsupported OpenETA backend kind: {kind}")
    if normalized_path != expected[normalized_kind]:
        raise ValueError(
            f"unsupported path for {normalized_kind}: {normalized_path}"
        )
    method_name = {
        "contact_graspnet": "predict_grasps",
        "anygrasp": "detect_grasps",
        "graspgenx": "predict_grasps",
        "anyplace": "predict_placement",
        "unidepth_v2": "estimate_depth",
    }[normalized_kind]
    method = getattr(backend, method_name, None)
    if not callable(method):
        raise RuntimeError(
            f"configured backend has no callable {method_name}"
        )
    result = method(**dict(payload))
    if not isinstance(result, Mapping):
        raise RuntimeError("OpenETA core returned a non-object response")
    return result


def openeta_health_payload(
    kind: str,
    backend: object,
    *,
    openeta_revision: str,
) -> dict[str, object]:
    settings = {}
    for name in (
        "depth_truncation",
        "max_candidates",
        "model_id",
        "device_request",
        "resolution_level",
    ):
        value = getattr(backend, name, None)
        if isinstance(value, (str, int, float, bool)):
            settings[name] = value
    return {
        "ok": backend is not None,
        "backend": str(kind),
        "openeta_revision": str(openeta_revision),
        "settings": settings,
    }
