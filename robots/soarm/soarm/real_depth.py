"""MoGe-2 depth for the real SO-101 cameras, aligned to the calibrated table plane with a per-pixel affine scale.

The harness's plane alignment (adapters/hardware/mono_depth.py) fits one scale factor. On the real head camera
MoGe-2's table comes out tilted (after that one scale, table pixels read z = -40 .. +90 mm across the image and the
blue cube's top 55 mm instead of 30 mm), which one scale cannot remove. Here the scale is s(u, v) = a + b u + c v,
fitted robustly on the table-footprint pixels against the known plane depth, then applied to every pixel.
"""
from __future__ import annotations

import cv2
import numpy as np

from captain_harness.adapters.hardware.mono_depth import MonoDepthSource, footprint_mask, plane_depth

# depth edges ("flying pixels"): MoGe smooths the jump from an object's top edge to the table behind it, so the edge
# pixels land between the two surfaces and stretch an object's top slab away from the camera. On four 11-14 stacking
# frames a 3 cm cube's box fit came out 47-70 mm long and its centre 13 mm too near the base; without pixels whose 5x5
# depth range exceeds 1.5 cm: 30-33 mm and 3.8 mm. Dropped pixels read as no depth.
EDGE_JUMP_M = 0.015


class AffinePlaneDepth(MonoDepthSource):
    def __init__(self, url: str, *, model: str, support_plane_z_m: float, table_footprint_m, tolerance: float = 0.08):
        super().__init__(url, model=model, align="none")
        self.align = "plane_affine"
        self.support_plane_z_m = float(support_plane_z_m)
        self.table_footprint_m = table_footprint_m
        self.tolerance = float(tolerance)

    def __call__(self, *, camera, rgb, raw_sim_depth, intrinsics_3x3, camera_to_base_4x4):
        rgb = np.ascontiguousarray(rgb)
        intr = np.asarray(intrinsics_3x3, float).reshape(3, 3)
        ext = np.asarray(camera_to_base_4x4, float).reshape(4, 4)
        mono, meta = self._request(rgb, intr)
        plane = plane_depth(intr, ext, self.support_plane_z_m, mono.shape)
        table = footprint_mask(intr, ext, self.support_plane_z_m, mono.shape, self.table_footprint_m)
        ok = table & np.isfinite(mono) & (mono > 0) & np.isfinite(plane) & (plane > 0)
        h, w = mono.shape
        v, u = np.nonzero(ok)
        record = {"camera": camera, "service": meta}
        if len(u) < 2000:
            record["align"] = {"status": "too_few_table_pixels", "pixels": int(len(u))}
            self.last_record = record
            return mono
        ratio = plane[v, u] / mono[v, u]
        A = np.stack([np.ones(len(u)), u / w, v / h], axis=1)
        keep = np.ones(len(u), bool)
        for _ in range(4):   # objects on the table read nearer than the plane: drop them and refit
            coef, *_ = np.linalg.lstsq(A[keep], ratio[keep], rcond=None)
            resid = np.abs(mono[v, u] * (A @ coef) - plane[v, u])
            keep = resid < max(self.tolerance * np.median(plane[v, u]) * 0.25, 0.01)
        uu, vv = np.meshgrid(np.arange(w) / w, np.arange(h) / h)
        scale = coef[0] + coef[1] * uu + coef[2] * vv
        depth = mono * scale
        resid = depth[v, u][keep] - plane[v, u][keep]
        record["align"] = {"status": "ok", "coef": [round(float(c), 4) for c in coef], "inlier_pixels": int(keep.sum()),
                           "table_pixels": int(len(u)), "inlier_rms_mm": round(float(np.sqrt(np.mean(resid ** 2))) * 1000, 1)}
        d = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)
        k = np.ones((5, 5), np.uint8)
        edge = (cv2.dilate(d, k) - cv2.erode(d, k)) > EDGE_JUMP_M
        depth = np.where(edge, np.nan, depth)
        record["align"]["edge_pixels_dropped"] = int(edge.sum())
        self.last_record = record
        return depth
