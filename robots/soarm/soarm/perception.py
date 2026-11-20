"""Perception mode of a SO-101 twin episode (SOARM_PERCEPTION) and its depth source (SOARM_DEPTH).

real (default): the LIBERO paper protocol. SAM3 (cap-x server: /segment, /segment_point) and Molmo2-8B (vLLM,
  OpenAI chat/completions "Point at <text>") are real model services reached over HTTP by the harness's own
  `CapXHttpToolServices` with its default urllib transport (attempts=1, as in env_server's build_libero_episode);
  nothing in the episode reads simulator segmentation.
privileged: perception_sim.SimPerceptionTransport answers from MuJoCo segmentation (tests and old data only).

Depth (SOARM_DEPTH; the real SO-ARM has RGB cameras only):
  moge2 (default, the only depth batches collect): the scene renders RGB only and the harness's real-arm monocular depth
    path estimates every camera's depth from the RGB frame -- `adapters/hardware/mono_depth.py` MonoDepthSource
    (MoGe-2 ViT-L via services/depth_server.py, intrinsics in, one support-plane scale per frame with the table top at
    robot_base z = 0), plugged into XR1FacadeEnv(depth_estimators=...) exactly as hardware/episode.py does for a
    camera with `"depth": "mono"`. No simulator depth reaches the episode.
  sensor: MuJoCo's rendered depth (LIBERO protocol; optional SOARM_DEPTH_NOISE sensor model). Only inside pytest or an
    explicit ablation run (SOARM_DEPTH_ABLATION=1); tools/batch.py refuses it.

Environment (real mode):
  SOARM_SAM3_URLS / SOARM_MOLMO_URLS  comma-separated replica base URLs (Molmo with /v1); one replica per episode,
                                      chosen by seed. Unset: env_server's --sam3-url / --molmo-url ($SAM3_URL/$MOLMO_URL).
  SOARM_SAM3_REVISION / SOARM_MOLMO_REVISION  ExperienceKey tool_revisions (defaults: the paper's pinned weights).
  SOARM_DEPTH_URLS    MoGe-2 depth service replicas (one per episode, by seed). Unset: $CAPTAIN_DEPTH_URL or :8130.
  SOARM_DEPTH_REVISION  recorded MoGe-2 weights (default: the harness's pinned real-robot checkpoint).
  SOARM_DEPTH_NOISE   sensor depth only: off (default) | axial sigma in metres at 1 m (e.g. 0.002): see depth_noise().
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Mapping

import numpy as np

MODES = ("real", "privileged")
DEPTH_MODES = ("moge2", "sensor")
# weights used for the CaPtain paper (harness scripts/models/download_*.sh, third_party/manifest.json)
SAM3_REVISION = "facebook/sam3:sam3.pt@sha256:9999e2341ceef5e1"
MOLMO_REVISION = "allenai/Molmo2-8B@e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"
CAPX_REVISION = "53e9966d7a8e2fa7494676772bccc35280f5c0ed"
# MoGe-2 checkpoint of the real-robot runs (harness scripts/models/download_moge.sh) and the MoGe code it needs
MOGE2_REVISION = "Ruicheng/moge-2-vitl-normal@cb0e8bbd6b1e243589717c78e750b1ba4c093acf:model.pt@sha256:280741fd09bc3f40"
MOGE_CODE_REVISION = "microsoft/MoGe@74fbce054ebed49800de42d0ad0e83495065719a"
MOGE2_MODEL = "moge2"             # depth_server.py MODEL_NAME
SUPPORT_PLANE_Z_M = 0.0           # table top in robot_base (scene.py): the one-time calibration of the plane alignment
# Calibrated table rectangle (robot_base m) for the plane fit: only pixels whose ray meets the table plane inside it may
# set the MoGe-2 scale (harness MonoDepthSource table_footprint_m). Every SO-ARM table covers it: center mount
# x -0.25..0.55 / |y| <= 0.45, edge x -0.06..0.45 / |y| <= 0.36, edge3 x -0.05..0.39 / |y| <= 0.33 (scene_random
# TABLE_RANGES). Without it the fit locked onto the floor or wall past the table edge in 12/160 div3 head frames
# (5-15 cm object errors); with it 0/160 (Div3Diagnose offline eval). The real arm measures the same rectangle once.
TABLE_FOOTPRINT_M = {"x": [-0.04, 0.39], "y": [-0.33, 0.33]}


def under_pytest() -> bool:
    return bool(os.environ.get("PYTEST_CURRENT_TEST"))


def mode() -> str:
    """real, or privileged ONLY inside pytest: any other process asking for privileged perception raises."""
    value = os.environ.get("SOARM_PERCEPTION", "real").strip() or "real"
    if value not in MODES:
        raise ValueError(f"SOARM_PERCEPTION must be one of {MODES}, got {value!r}")
    if value == "privileged" and not under_pytest():
        raise RuntimeError("SOARM_PERCEPTION=privileged (MuJoCo-segmentation perception) is allowed only inside pytest; "
                           "every collected episode uses the real SAM3/Molmo2 services")
    return value


def depth_mode() -> str:
    """moge2 (default), or sensor ONLY inside pytest / an explicit ablation (SOARM_DEPTH_ABLATION=1)."""
    value = os.environ.get("SOARM_DEPTH", "moge2").strip().lower() or "moge2"
    if value not in DEPTH_MODES:
        raise ValueError(f"SOARM_DEPTH must be one of {DEPTH_MODES}, got {value!r}")
    if value == "sensor" and not (under_pytest() or os.environ.get("SOARM_DEPTH_ABLATION") == "1"):
        raise RuntimeError("SOARM_DEPTH=sensor (MuJoCo rendered depth) is allowed only inside pytest or an explicit "
                           "ablation (SOARM_DEPTH_ABLATION=1); collected episodes estimate depth with MoGe-2")
    return value


def require_depth_healthy(url: str, timeout_s: float = 30.0) -> dict:
    """MoGe-2 depth: the episode's depth service must answer /health as the moge2 model before the episode starts."""
    import urllib.request
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=timeout_s) as resp:
            report = json.loads(resp.read().decode("utf-8"))
    except OSError as exc:
        raise RuntimeError(f"MoGe-2 depth service unavailable: {url}: {exc}") from exc
    if report.get("status") != "ok" or report.get("default_model") != MOGE2_MODEL:
        raise RuntimeError(f"depth service {url} is not a healthy {MOGE2_MODEL} service: {report}")
    return report



def require_healthy(services) -> dict:
    """Real mode: the SAM3 and Molmo2 services of this episode must answer before it starts (no silent fallback)."""
    report = services.health(names=("sam3", "molmo"))
    bad = {k: v for k, v in report.items() if not v.get("healthy")}
    if bad or set(report) != {"sam3", "molmo"}:
        raise RuntimeError(f"real perception services unavailable: {bad or report}")
    return report


def pick_url(csv: str | None, default: str, seed: int) -> str:
    """One replica per episode (stable in the seed, so a resumed episode keeps its replica)."""
    urls = [u.strip() for u in (csv or "").split(",") if u.strip()] or [default]
    url = urls[int(hashlib.sha1(str(seed).encode()).hexdigest(), 16) % len(urls)]
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"real perception needs an http(s) service URL, got {url!r}")
    return url


def depth_noise_sigma() -> float:
    value = os.environ.get("SOARM_DEPTH_NOISE", "off").strip().lower()
    if value in ("", "off", "0", "none"):
        return 0.0
    sigma = float(value)
    if sigma < 0:
        raise ValueError("SOARM_DEPTH_NOISE must be >= 0")
    return sigma


def depth_config(args, seed: int) -> dict:
    """Depth source of a real-perception episode (part of config())."""
    d = depth_mode()
    sigma = depth_noise_sigma()
    if d == "sensor":
        return {"depth": "sim_sensor", "depth_noise_sigma_1m": sigma}
    if sigma:
        raise ValueError("SOARM_DEPTH_NOISE models a depth sensor; MoGe-2 episodes have none (unset it)")
    default = getattr(args, "hardware_depth_url", None) or os.environ.get("CAPTAIN_DEPTH_URL") or "http://127.0.0.1:8130"
    return {"depth": "moge2", "depth_noise_sigma_1m": 0.0,
            "depth_url": pick_url(os.environ.get("SOARM_DEPTH_URLS"), default, seed),
            "depth_revision": os.environ.get("SOARM_DEPTH_REVISION") or MOGE2_REVISION,
            "depth_code_revision": MOGE_CODE_REVISION,
            "depth_estimator": "captain_harness.adapters.hardware.mono_depth.MonoDepthSource",
            "depth_align": "plane", "depth_support_plane_z_m": SUPPORT_PLANE_Z_M,
            "depth_table_footprint_m": TABLE_FOOTPRINT_M}


def depth_noise(depth_m: np.ndarray, sigma_1m: float, rng: np.random.Generator) -> np.ndarray:
    """Structured-light / stereo style sensor noise (off unless SOARM_DEPTH_NOISE is set): axial Gaussian noise
    growing with z^2 (sigma_1m at 1 m), 1 mm quantisation, and dropout of 1 % of valid pixels (0 = no reading, the
    harness's hole convention)."""
    out = depth_m + rng.normal(0.0, 1.0, depth_m.shape).astype(np.float32) * np.float32(sigma_1m) * depth_m * depth_m
    out = np.round(out * 1000.0) / 1000.0
    out[rng.random(depth_m.shape) < 0.01] = 0.0
    return np.where(depth_m > 0, out, depth_m).astype(np.float32)


def config(args, seed: int) -> dict:
    """Resolved perception settings of one episode (written to soarm/perception.json and the batch record)."""
    m = mode()
    if m == "privileged":
        sigma = depth_noise_sigma()
        return {"mode": m, "sam3_url": "sim://sam3", "molmo_url": "sim://molmo/v1",
                "tool_revisions": {"sam3": "privileged-sim", "molmo": "privileged-sim"},
                "depth": "sim_sensor", "depth_noise_sigma_1m": sigma}
    depth = depth_config(args, seed)
    revisions = {"sam3": os.environ.get("SOARM_SAM3_REVISION") or SAM3_REVISION,
                 "molmo": os.environ.get("SOARM_MOLMO_REVISION") or MOLMO_REVISION,
                 "cap_x": CAPX_REVISION}
    if depth["depth"] == "moge2":
        revisions["depth"] = depth["depth_revision"]
    return {"mode": m,
            "sam3_url": pick_url(os.environ.get("SOARM_SAM3_URLS"), args.sam3_url, seed),
            "molmo_url": pick_url(os.environ.get("SOARM_MOLMO_URLS"), args.molmo_url, seed),
            "tool_revisions": revisions, **depth}


def mono_depth_estimators(perc: Mapping, cameras, transport: "LoggedHttpTransport") -> dict:
    """MoGe-2 episodes: one real-arm MonoDepthSource per camera (both SO-ARM cameras are RGB-only), each estimate logged
    to perception.jsonl. Sensor-depth episodes: none (the frames carry rendered depth)."""
    if perc.get("depth") != "moge2":
        return {}
    from captain_harness.adapters.hardware.mono_depth import MonoDepthSource
    source = MonoDepthSource(perc["depth_url"], model=MOGE2_MODEL, align=perc["depth_align"],
                             support_plane_z_m=float(perc["depth_support_plane_z_m"]),
                             table_footprint_m=perc.get("depth_table_footprint_m"))
    logged = LoggedMonoDepth(source, transport)
    return {name: logged for name in cameras}


class LoggedMonoDepth:
    """The harness's MonoDepthSource, plus one perception.jsonl row per estimate (latency, plane-fit scale and inliers,
    cache hits; never the depth map)."""

    needs_sim_depth = False

    def __init__(self, source, transport: "LoggedHttpTransport") -> None:
        self.source = source
        self.transport = transport

    def __call__(self, *, camera, rgb, raw_sim_depth, intrinsics_3x3, camera_to_base_4x4):
        row: dict = {"t": time.time(), "privileged": False, "kind": "depth", "method": "POST",
                     "url": f"{self.source.url}/estimate_depth", "model": self.source.model, "camera": camera}
        started = time.monotonic()
        try:
            depth = self.source(camera=camera, rgb=rgb, raw_sim_depth=raw_sim_depth, intrinsics_3x3=intrinsics_3x3,
                                camera_to_base_4x4=camera_to_base_4x4)
        except BaseException as exc:
            row.update(ok=False, latency_s=round(time.monotonic() - started, 4), error=f"{type(exc).__name__}: {exc}"[:300])
            self.transport._write(row)
            raise
        rec = self.source.last_record or {}
        row.update(ok=True, latency_s=round(time.monotonic() - started, 4), cache_hit=bool(rec.get("cache_hit")))
        if not row["cache_hit"]:
            service = rec.get("service") or {}
            row.update(service_s=service.get("elapsed_s"), valid_ratio=service.get("valid_ratio"),
                       uses_intrinsics=service.get("uses_intrinsics"), align=rec.get("align"))
        self.transport._write(row)
        return depth


class LoggedHttpTransport:
    """The harness's UrllibJsonTransport, plus one perception.jsonl row per request (endpoint, latency, outcome and a
    compact answer summary; never images or masks)."""

    def __init__(self, log_path: Path) -> None:
        from captain_harness.adapters.capx.services import UrllibJsonTransport
        self.inner = UrllibJsonTransport()
        self.log_path = log_path
        self._lock = threading.Lock()

    def request(self, method: str, url: str, *, payload: Mapping[str, object] | None = None,
                timeout_s: float) -> Mapping[str, object]:
        row: dict = {"t": time.time(), "privileged": False, "method": method, "url": url}
        if payload is not None:
            if "text_prompt" in payload:
                row["prompt"] = payload["text_prompt"]
            if "point_coords" in payload:
                row["point"] = payload["point_coords"]
            if "messages" in payload:
                row["model"] = payload.get("model")
                row["prompt"] = next((c.get("text") for c in payload["messages"][0]["content"]
                                      if c.get("type") == "text"), None)
        started = time.monotonic()
        try:
            value = self.inner.request(method, url, payload=payload, timeout_s=timeout_s)
        except BaseException as exc:
            row.update(ok=False, latency_s=round(time.monotonic() - started, 4), error=f"{type(exc).__name__}: {exc}"[:300])
            self._write(row)
            raise
        row.update(ok=True, latency_s=round(time.monotonic() - started, 4))
        if isinstance(value.get("results"), list):
            row["results"] = [{"score": r.get("score"), "box": r.get("box"), "label": r.get("label")}
                              for r in value["results"][:8]]
            row["n_results"] = len(value["results"])
        if isinstance(value.get("scores"), list):
            row["scores"] = value["scores"][:8]
        if isinstance(value.get("choices"), list) and value["choices"]:
            row["answer"] = str((value["choices"][0].get("message") or {}).get("content"))[:300]
        self._write(row)
        return value

    def _write(self, row: dict) -> None:
        with self._lock, self.log_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
