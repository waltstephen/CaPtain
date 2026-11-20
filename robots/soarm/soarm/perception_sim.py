"""PRIVILEGED simulator-backed perception (SOARM_PERCEPTION=privileged: tests only, never data): answers the SAM3 /
Molmo HTTP JSON protocol of `captain_harness.adapters.capx.services.CapXHttpToolServices` in-process, from MuJoCo
segmentation of the very frame the agent saw (looked up by image hash). Plugged in as that class's `transport`; the
default real mode (soarm/perception.py) uses the real SAM3/Molmo2 services instead.

Grounding: the text prompt is matched against object attributes (colour, shape noun + synonyms). Every answer is logged
to `perception_sim.jsonl` with "privileged": true.
"""
from __future__ import annotations

import base64
import io
import json
import re
import time
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image

from .driver import SoarmDriver, image_key
from .objects import COLORS, ObjSpec

SYNONYMS = {
    "cube": {"cube", "cubes", "block", "blocks"}, "block": {"block", "blocks", "brick", "bricks", "bar", "cuboid"},
    "cylinder": {"cylinder", "cylinders", "can", "tube", "peg", "bottle", "rod"},
    "ball": {"ball", "balls", "sphere", "spheres", "marble"}, "tile": {"tile", "tiles", "slab", "slabs", "plate"},
    "bowl": {"bowl", "bowls", "basin", "dish"}, "box": {"box", "boxes", "tray", "bin", "container", "crate"},
    "cup": {"cup", "cups", "mug", "glass"}, "pad": {"mat", "mats", "pad", "square", "region", "area", "target", "zone"},
}
GENERIC = {"object", "objects", "item", "items", "thing", "toy", "piece"}
COLOR_SYNONYMS = {"grey": "gray", "violet": "purple", "magenta": "pink", "turquoise": "cyan", "teal": "cyan"}
MIN_PIXELS = 12


def _png_to_array(b64: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def score_object(prompt: str, spec: ObjSpec) -> float:
    """> 0 if the prompt can refer to the object; higher = better match. Colour mismatch or noun mismatch -> 0."""
    words = re.findall(r"[a-z]+", prompt.lower())
    words = [COLOR_SYNONYMS.get(w, w) for w in words]
    colors = [w for w in words if w in COLORS]
    nouns = {k for k, syn in SYNONYMS.items() if syn & set(words)}
    score = 0.1
    if colors:
        if spec.color not in colors:
            return 0.0
        score += 1.0 if colors[0] == spec.color else 0.5
    if nouns:
        if spec.kind not in nouns:
            return 0.0
        score += 2.0
    elif not (GENERIC & set(words)) and not colors:
        return 0.0
    if "small" in words or "smaller" in words or "large" in words or "big" in words or "larger" in words:
        score += 0.0   # size words never exclude; ranking among equals is left to the mask areas
    return score


class SimPerceptionTransport:
    """JsonTransport for CapXHttpToolServices (request(method, url, payload, timeout_s) -> Mapping)."""

    def __init__(self, driver: SoarmDriver, log_path: Path | None = None) -> None:
        from .perception import under_pytest
        if not under_pytest():
            raise RuntimeError("SimPerceptionTransport (privileged MuJoCo-segmentation perception) is test-only")
        self.driver = driver
        self.names = list(driver.sim.specs)
        self.specs = driver.sim.specs
        self.log_path = log_path

    # ------------------------------------------------------------------ frame lookup
    def _frame(self, image: np.ndarray):
        reg = self.driver.frame_registry
        f = reg.get(image_key(image))
        if f is not None:
            return f, False
        flipped = image[::-1]
        f = reg.get(image_key(np.ascontiguousarray(flipped)))
        if f is not None:
            return f, True
        best, best_err, best_flip = None, np.inf, False
        for cand in reversed(reg.values()):                  # nearest recent frame of the same shape
            if cand.rgb.shape != image.shape:
                continue
            for flip, img in ((False, image), (True, flipped)):
                err = float(np.abs(cand.rgb[::8, ::8].astype(np.int16) - img[::8, ::8].astype(np.int16)).mean())
                if err < best_err:
                    best, best_err, best_flip = cand, err, flip
        if best is None or best_err > 12.0:
            raise OSError("privileged perception: image does not match any rendered frame")
        return best, best_flip

    def _masks(self, image: np.ndarray):
        frame, flipped = self._frame(image)
        seg = frame.seg_object[::-1] if flipped else frame.seg_object
        return seg

    def _log(self, row: dict) -> None:
        if self.log_path is not None:
            with self.log_path.open("a") as fh:
                fh.write(json.dumps({"t": time.time(), "privileged": True, **row}) + "\n")

    # ------------------------------------------------------------------ protocol
    def request(self, method: str, url: str, payload: Mapping[str, object] | None = None, timeout_s: float = 0.0):
        if method == "GET":
            return {"ok": True, "backend": "soarm-privileged-sim"}
        assert payload is not None
        if url.endswith("/segment"):
            return self._segment_text(_png_to_array(str(payload["image_base64"])), str(payload["text_prompt"]))
        if url.endswith("/segment_point"):
            x, y = (float(v) for v in payload["point_coords"])
            return self._segment_point(_png_to_array(str(payload["image_base64"])), x, y)
        if url.endswith("/chat/completions"):
            content = payload["messages"][0]["content"]
            text = next(c["text"] for c in content if c.get("type") == "text")
            url_img = next(c["image_url"]["url"] for c in content if c.get("type") == "image_url")
            image = _png_to_array(url_img.split(",", 1)[1])
            return self._point(image, re.sub(r"^Point at\s+", "", text))
        raise OSError(f"privileged perception: unsupported endpoint {url}")

    def _candidates(self, seg: np.ndarray, prompt: str):
        out = []
        for i, name in enumerate(self.names):
            s = score_object(prompt, self.specs[name])
            if s <= 0:
                continue
            mask = seg == i
            n = int(mask.sum())
            if n < MIN_PIXELS:
                continue
            out.append((s + 1e-6 * n, name, mask))
        out.sort(key=lambda t: -t[0])
        return out

    def _segment_text(self, image: np.ndarray, prompt: str):
        seg = self._masks(image)
        cands = self._candidates(seg, prompt)
        top = cands[0][0] if cands else 1.0
        results = []
        for s, name, mask in cands:
            ys, xs = np.nonzero(mask)
            m8 = mask.astype(np.uint8)
            results.append({"mask_base64": _b64(m8), "shape": list(m8.shape), "score": round(0.95 * s / top, 4),
                            "box": [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
                            "label": prompt})
        self._log({"endpoint": "segment", "prompt": prompt, "objects": [c[1] for c in cands]})
        return {"results": results}

    def _segment_point(self, image: np.ndarray, x: float, y: float):
        seg = self._masks(image)
        h, w = seg.shape
        xi, yi = int(np.clip(round(x), 0, w - 1)), int(np.clip(round(y), 0, h - 1))
        idx = int(seg[yi, xi])
        if idx < 0:      # snap to the nearest object pixel within 6 px (SAM point prompts tolerate small misses)
            y0, y1, x0, x1 = max(0, yi - 6), min(h, yi + 7), max(0, xi - 6), min(w, xi + 7)
            win = seg[y0:y1, x0:x1]
            hits = np.argwhere(win >= 0)
            if len(hits):
                d = ((hits - [yi - y0, xi - x0]) ** 2).sum(1)
                idx = int(win[tuple(hits[int(np.argmin(d))])])
        self._log({"endpoint": "segment_point", "xy": [x, y], "object": self.names[idx] if idx >= 0 else None})
        if idx < 0:
            return {"scores": [], "masks_shape": [0, h, w], "masks_dtype": "uint8", "masks_base64": ""}
        mask = (seg == idx).astype(np.uint8)[None]
        return {"scores": [0.95], "masks_shape": list(mask.shape), "masks_dtype": "uint8", "masks_base64": _b64(mask)}

    def _point(self, image: np.ndarray, prompt: str):
        seg = self._masks(image)
        cands = self._candidates(seg, prompt)
        self._log({"endpoint": "point", "prompt": prompt, "object": cands[0][1] if cands else None})
        if not cands:
            return {"choices": [{"message": {"content": "There are no such objects in the image."}}]}
        mask = cands[0][2]
        ys, xs = np.nonzero(mask)
        # the mask pixel closest to the centroid (stays on the object for concave masks)
        cy, cx = ys.mean(), xs.mean()
        k = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
        h, w = mask.shape
        px, py = 100.0 * (xs[k] + 0.5) / w, 100.0 * (ys[k] + 0.5) / h
        return {"choices": [{"message": {"content": f'<point x="{px:.1f}" y="{py:.1f}" alt="{prompt}">{prompt}</point>'}}]}
