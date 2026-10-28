"""Monocular metric depth + support-plane scale alignment (from the rgb-only harness branch, c9575c9).

Used for cameras without a depth sensor (the XR1 wrist camera): MoGe-2 estimates metric depth from RGB +
intrinsics, then one robust scale per frame puts the table (robot_base z = support_plane_z_m) where the
camera extrinsics say it is. The same path runs in sim and on the robot.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import time
import urllib.request
from collections import OrderedDict
from typing import Any, Callable, Mapping, Protocol

import numpy as np

DepthConverter = Callable[[np.ndarray], np.ndarray]
def _png_b64(rgb: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb).astype(np.uint8, copy=False)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def plane_depth(intrinsics_3x3: np.ndarray, camera_to_base_4x4: np.ndarray, height_m: float, shape: tuple[int, int]) -> np.ndarray:
    """Camera optical-z depth of the horizontal plane z_base = height_m for every pixel (NaN where not hit)."""
    h, w = shape
    k_inv = np.linalg.inv(np.asarray(intrinsics_3x3, dtype=np.float64).reshape(3, 3))
    t = np.asarray(camera_to_base_4x4, dtype=np.float64).reshape(4, 4)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    rays = k_inv @ np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)])  # z = 1 in camera frame
    dirs_base = t[:3, :3] @ rays
    origin_z = t[2, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (height_m - origin_z) / dirs_base[2]
    z = np.where(np.isfinite(z) & (z > 0), z, np.nan)
    return z.reshape(h, w)


def fit_plane_scale(mono: np.ndarray, plane_z: np.ndarray, *, tolerance: float = 0.06, min_pixels: int = 2000,
                    exclude: np.ndarray | None = None, min_cluster_fraction: float = 0.3) -> tuple[float, dict[str, Any]]:
    """Robust single scale s with s*mono ~= plane depth on pixels that look like the support plane.

    Pixels are support-plane candidates when the plane is hit and the ratio plane/mono agrees with the
    current scale estimate within ``tolerance`` (relative); objects standing on the table are in front of
    the plane (ratio > s) and are rejected. ``exclude`` masks pixels that are never support plane (the robot's
    own gripper in the wrist camera). Start: among log-ratio clusters holding at least ``min_cluster_fraction``
    of the densest cluster's mass, the one with the *smallest* ratio, i.e. the farthest large surface -- the
    support plane lies behind everything standing on it.
    """
    valid = np.isfinite(mono) & (mono > 0) & np.isfinite(plane_z) & (plane_z > 0)
    if exclude is not None:
        valid &= ~np.asarray(exclude, dtype=bool).reshape(valid.shape)
    if int(valid.sum()) < min_pixels:
        return 1.0, {"status": "too_few_plane_pixels", "pixels": int(valid.sum())}
    ratio = plane_z[valid] / mono[valid]
    # Support-plane pixels share one ratio; objects on the plane are in front of it (larger ratio).
    log_ratio = np.log(ratio[np.isfinite(ratio) & (ratio > 0)])
    bins = np.arange(np.log(0.05), np.log(20.0) + 0.01, 0.01)
    hist, edges = np.histogram(log_ratio, bins=bins)
    # cluster mass = pixels within +-tolerance of each bin centre
    width = max(1, int(round(np.log1p(tolerance) / 0.01)))
    mass = np.convolve(hist, np.ones(2 * width + 1), mode="same")
    peaks = [i for i in range(len(mass)) if mass[i] > 0 and mass[i] >= mass[max(0, i - width):i + width + 1].max()]
    strong = [i for i in peaks if mass[i] >= min_cluster_fraction * mass.max()]
    start = min(strong) if strong else int(np.argmax(mass))
    # A sharp cluster gives a flat-topped box of equal mass; move to the middle of that plateau, then start from
    # the median log-ratio inside the window (the plateau edge alone can sit a full tolerance off the cluster).
    end = start
    while end + 1 < len(mass) and mass[end + 1] == mass[start]:
        end += 1
    centre = edges[(start + end) // 2] + 0.005
    near = np.abs(log_ratio - centre) <= np.log1p(tolerance)
    s = float(np.exp(np.median(log_ratio[near]))) if near.any() else float(np.exp(centre))
    inliers = np.zeros_like(ratio, dtype=bool)
    for _ in range(10):
        inliers = np.abs(ratio / s - 1.0) < tolerance
        if inliers.sum() < min_pixels:
            break
        s_new = float(np.median(ratio[inliers]))
        if abs(s_new - s) < 1e-4 * s:
            s = s_new
            break
        s = s_new
    return s, {"status": "ok" if inliers.sum() >= min_pixels else "few_inliers",
               "inlier_pixels": int(inliers.sum()), "plane_pixels": int(valid.sum())}


class MonoDepthSource:
    """Monocular metric depth from the RGB frame via the depth HTTP service.

    align:
      none   use the network's metric depth as is.
      plane  rescale by one robust scalar per frame so the dominant support plane lands at a known height
             ``support_plane_z_m`` in the robot-base frame. Reads: RGB, intrinsics, camera extrinsics and that one
             height (a one-time scene calibration, e.g. measured once by touching the table).
    """

    needs_sim_depth = False

    def __init__(
        self,
        url: str,
        *,
        model: str | None = None,
        align: str = "none",
        support_plane_z_m: float | None = None,
        self_masks: Mapping[str, np.ndarray] | None = None,
        timeout_s: float = 60.0,
        cache_size: int = 16,
    ) -> None:
        if align not in ("none", "plane"):
            raise ValueError(f"unknown depth alignment {align!r}")
        if align == "plane" and support_plane_z_m is None:
            raise ValueError("align=plane needs support_plane_z_m")
        self.url = url.rstrip("/")
        self.model = model
        self.align = align
        self.support_plane_z_m = support_plane_z_m
        self.self_masks = {k: np.asarray(v, dtype=bool) for k, v in (self_masks or {}).items()}
        self.timeout_s = timeout_s
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_size = cache_size
        self.last_record: dict[str, Any] | None = None
        self.calls = 0
        self.cache_hits = 0
        self.total_s = 0.0

    @property
    def name(self) -> str:
        return "mono" if self.align == "none" else f"mono+{self.align}"

    def describe(self) -> dict[str, Any]:
        inputs = ["RGB frame", "camera intrinsics (estimator input)", "camera extrinsics (back-projection)"]
        if self.align == "plane":
            inputs.append(f"support plane height z={self.support_plane_z_m:.4f} m in robot base frame (scale alignment)")
            if self.self_masks:
                inputs.append("fixed robot self-occlusion mask (gripper region) excluded from the fit: "
                              + ", ".join(sorted(self.self_masks)))
        return {"depth_source": "mono", "service": self.url, "model": self.model or "server default",
                "align": self.align, "inputs": inputs, "simulator_depth_used": False}

    def _request(self, rgb: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        payload = {"rgb_png_base64": _png_b64(rgb), "intrinsics_3x3": [float(v) for v in np.asarray(intrinsics).ravel()]}
        if self.model:
            payload["model"] = self.model
        request = urllib.request.Request(f"{self.url}/estimate_depth", data=json.dumps(payload).encode("utf-8"),
                                         headers={"Content-Type": "application/json"}, method="POST")
        last: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    body = json.loads(response.read().decode("utf-8"))
                depth = np.load(io.BytesIO(base64.b64decode(body["depth_npy_base64"])), allow_pickle=False)
                return depth.astype(np.float64), {k: body.get(k) for k in ("model", "uses_intrinsics", "elapsed_s", "valid_ratio")}
            except Exception as exc:  # transient service errors: retry, then surface
                last = exc
                time.sleep(1.0 + attempt)
        raise RuntimeError(f"depth service unavailable: {type(last).__name__}: {last}")

    def __call__(self, *, camera, rgb, raw_sim_depth, intrinsics_3x3, camera_to_base_4x4):
        rgb = np.ascontiguousarray(rgb)
        intr = np.asarray(intrinsics_3x3, dtype=np.float64).reshape(3, 3)
        ext = np.asarray(camera_to_base_4x4, dtype=np.float64).reshape(4, 4)
        key = hashlib.sha1(camera.encode() + rgb.tobytes() + intr.tobytes() + ext.tobytes()).hexdigest()
        if key in self._cache:
            self.cache_hits += 1
            self.last_record = {"camera": camera, "cache_hit": True}
            self._cache.move_to_end(key)
            return self._cache[key]
        t0 = time.time()
        depth, meta = self._request(rgb, intr)
        if depth.shape != rgb.shape[:2]:
            raise ValueError(f"depth service returned {depth.shape} for image {rgb.shape[:2]}")
        record: dict[str, Any] = {"camera": camera, "service": meta}
        if self.align == "plane":
            plane = plane_depth(intr, ext, float(self.support_plane_z_m), depth.shape)
            mask = self.self_masks.get(camera)
            scale, info = fit_plane_scale(depth, plane, exclude=mask if mask is not None and mask.shape == depth.shape else None)
            if info["status"] == "ok" and 0.05 < scale < 20.0:
                depth = depth * scale
            else:
                scale = 1.0
            record["align"] = {"scale": scale, **info}
        self.calls += 1
        self.total_s += time.time() - t0
        record["latency_s"] = round(time.time() - t0, 4)
        self._cache[key] = depth
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        self.last_record = record
        return depth

