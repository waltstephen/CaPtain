"""Broad-object import: Google Scanned Objects (CC-BY-4.0) + the YCB object set (CC-BY-4.0) -> registry rows.

Runs in the assets venv (trimesh + coacd; not needed by the simulator):
    python -m assets.import_objects list   [--source gso,ycb]        # sizes only: catalogue + total download size
    python -m assets.import_objects fetch  [--source ...] [--limit N] # raw archives -> <out>/raw/<source>/
    python -m assets.import_objects process [--workers 12]            # filter, recentre, CoACD, rows -> registry.jsonl

Per object (process): visual mesh in metres, z up, origin at the bounding-box centre; filter: some scale inside the
category's realistic range gives a top-down grasp width (narrow horizontal extent) 1.5-6 cm, height <= 10 cm and a
footprint <= 15 cm (7 cm jaw, top-down reach); convex decomposition with CoACD (threshold 0.05, <= 12 hulls);
mass = mesh volume x category density (convex hull x 0.6 when the scan is not watertight), clipped 10-300 g;
colours = named colours of the texture sampled at the vertex UVs; split: 20 % of categories held out (stable hash).
Rows get graspable=None until tools/gate_assets.py runs. Archives are cached (fetch is resumable); process redoes every
fetched object and keeps the gate fields of unchanged ids.
"""
from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import re
import tarfile
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GSO_LIST = "https://fuel.gazebosim.org/1.0/GoogleResearch/models?per_page=100&page={page}"
GSO_ZIP = "https://fuel.gazebosim.org/1.0/GoogleResearch/models/{name}/1/{name}.zip"
YCB_LIST = "https://ycb-benchmarks.s3.amazonaws.com/data/objects.json"
YCB_TGZ = "https://ycb-benchmarks.s3.amazonaws.com/data/google/{name}_google_16k.tgz"
UA = {"User-Agent": "soarm-assets/1 (research; object import)"}
GRASP_W, MAX_H, MAX_FOOT = (0.015, 0.06), 0.10, 0.15
MASS = (0.010, 0.300)
HELDOUT_FRACTION = 0.2

# category -> (scale range relative to the scan, density kg/m3 of the solid mesh volume, friction range)
# scans are real-size; toys/figures exist in many sizes, manufactured goods keep +-5 %
CATEGORY = {
    "toy": ((0.5, 1.05), 350.0, (0.6, 1.0)), "figure": ((0.5, 1.05), 500.0, (0.6, 1.0)),
    "lego": ((0.8, 1.05), 600.0, (0.6, 0.9)), "stuffed toy": ((0.5, 1.05), 120.0, (0.9, 1.2)),
    "shoe": ((0.95, 1.05), 300.0, (0.9, 1.2)), "bottle": ((0.95, 1.05), 250.0, (0.5, 0.8)),
    "can": ((0.95, 1.05), 900.0, (0.5, 0.8)), "box": ((0.95, 1.05), 250.0, (0.6, 0.9)),
    "cup": ((0.95, 1.05), 500.0, (0.5, 0.9)), "fruit": ((0.95, 1.05), 900.0, (0.6, 1.0)),
    "tool": ((0.95, 1.05), 800.0, (0.6, 1.0)), "ball": ((0.95, 1.05), 400.0, (0.7, 1.1)),
    "kitchen": ((0.95, 1.05), 700.0, (0.5, 0.9)), "electronics": ((0.95, 1.05), 600.0, (0.5, 0.9)),
    "object": ((0.9, 1.05), 400.0, (0.6, 1.0)),
}
# GSO metadata category / name keywords -> category (first match wins)
KEYWORDS = (("stuffed", "stuffed toy"), ("plush", "stuffed toy"), ("lego", "lego"), ("duplo", "lego"),
            ("figure", "figure"), ("doll", "figure"), ("action", "figure"), ("shoe", "shoe"), ("sneaker", "shoe"),
            ("boot", "shoe"), ("slipper", "shoe"), ("bottle", "bottle"), ("can", "can"), ("cup", "cup"),
            ("mug", "cup"), ("bowl", "kitchen"), ("plate", "kitchen"), ("spoon", "kitchen"), ("fork", "kitchen"),
            ("pan", "kitchen"), ("pitcher", "kitchen"), ("box", "box"), ("media cases", "box"), ("board games", "box"),
            ("ball", "ball"), ("apple", "fruit"), ("banana", "fruit"), ("lemon", "fruit"), ("orange", "fruit"),
            ("pear", "fruit"), ("peach", "fruit"), ("plum", "fruit"), ("strawberry", "fruit"), ("drill", "tool"),
            ("hammer", "tool"), ("wrench", "tool"), ("screwdriver", "tool"), ("clamp", "tool"), ("scissors", "tool"),
            ("marker", "tool"), ("padlock", "tool"), ("mouse", "electronics"), ("camera", "electronics"),
            ("headphone", "electronics"), ("keyboard", "electronics"), ("toy", "toy"), ("toys", "toy"),
            ("car", "toy"), ("truck", "toy"), ("train", "toy"), ("animal", "toy"), ("dinosaur", "toy"))


def asset_root() -> Path:
    root = os.environ.get("SOARM_ASSETS") or str(Path(os.environ.get("SOARM_ROOT", "~/soarm_work")) / "assets")
    return Path(root).expanduser()


def _get(url: str, timeout: float = 120) -> bytes:
    for k in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            if exc.code < 500 or k == 3:          # 4xx is final (Fuel answers 404 past the last page)
                raise
            time.sleep(3 + 5 * k)
        except Exception:                       # noqa: BLE001 (network retry, re-raised after 4 tries)
            if k == 3:
                raise
            time.sleep(3 + 5 * k)
    raise AssertionError


# ---------------------------------------------------------------------------------------------------- catalogue
def list_gso() -> list[dict]:
    out, page = [], 1
    while True:
        try:
            items = json.loads(_get(GSO_LIST.format(page=page)))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return out
            raise
        if not items:
            return out
        out += [{"source": "gso", "name": it["name"], "bytes": int(it.get("filesize") or 0),
                 "licence": it.get("license_name", ""), "description": it.get("description", "")} for it in items]
        page += 1


def list_ycb() -> list[dict]:
    names = json.loads(_get(YCB_LIST))["objects"]
    return [{"source": "ycb", "name": n, "bytes": 0, "licence": "Creative Commons Attribution 4.0 International",
             "description": re.sub(r"^\d+-?[a-z]?_", "", n).replace("_", " ")} for n in names]


def catalogue(sources: list[str]) -> list[dict]:
    path = asset_root() / "raw" / "catalogue.json"
    cached = json.loads(path.read_text()) if path.exists() else {}
    for s in sources:
        if s not in cached:
            cached[s] = list_gso() if s == "gso" else list_ycb()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cached))
    return [it for s in sources for it in cached[s]]


# ---------------------------------------------------------------------------------------------------- fetch
def fetch_one(it: dict) -> str:
    dst = asset_root() / "raw" / it["source"] / it["name"]
    if (dst / "done").exists():
        return "cached"
    dst.mkdir(parents=True, exist_ok=True)
    if it["source"] == "gso":
        z = zipfile.ZipFile(io.BytesIO(_get(GSO_ZIP.format(name=it["name"]))))
        for member, name in (("meshes/model.obj", "model.obj"), ("materials/textures/texture.png", "texture.png"),
                             ("metadata.pbtxt", "metadata.pbtxt")):
            if member in z.namelist():
                (dst / name).write_bytes(z.read(member))
    else:
        try:
            t = tarfile.open(fileobj=io.BytesIO(_get(YCB_TGZ.format(name=it["name"]))))
        except Exception as exc:               # noqa: BLE001 (some YCB items have no google_16k scan)
            (dst / "done").write_text(f"missing: {exc}")
            return "missing"
        for m in t.getmembers():
            base = os.path.basename(m.name)
            if base in ("textured.obj", "texture_map.png"):
                (dst / ("model.obj" if base.endswith(".obj") else "texture.png")).write_bytes(t.extractfile(m).read())
    (dst / "done").write_text("ok")
    return "fetched"


# ---------------------------------------------------------------------------------------------------- process
def categorize(it: dict, raw: Path) -> str:
    meta = (raw / "metadata.pbtxt").read_text() if (raw / "metadata.pbtxt").exists() else ""
    cats = " ".join(re.findall(r'first: "([^"]+)"', meta)).lower()
    # name + metadata category only: GSO descriptions are marketing prose ("... can be used ..." -> can)
    text = f"{cats} {it['name']}".lower().replace("_", " ")
    for kw, cat in KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}s?\b", text):
            return cat
    return "object"


def display_name(it: dict, category: str) -> str:
    d = (it["description"] or it["name"]).split("\n")[0].replace("_", " ").strip()
    return re.sub(r"\s+", " ", d)[:60].lower() or category


def scale_window(ext: np.ndarray, category: str) -> tuple[float, float] | None:
    """Scales (relative to the scan) inside the category's realistic range that satisfy the gripper filter."""
    lo, hi = CATEGORY[category][0]
    w_narrow, w_wide, h = float(min(ext[:2])), float(max(ext[:2])), float(ext[2])
    lo = max(lo, GRASP_W[0] / w_narrow)
    hi = min(hi, GRASP_W[1] / w_narrow, MAX_H / h, MAX_FOOT / w_wide)
    return (lo, hi) if lo <= hi else None


def _colours(mesh, tex_path: Path) -> list[str]:
    from PIL import Image

    from assets.build_cubes import colour_names
    uv = getattr(mesh.visual, "uv", None)
    if uv is None or not tex_path.exists():
        return []
    img = Image.open(tex_path).convert("RGB")
    a = np.asarray(img)
    h, w = a.shape[:2]
    uv = np.asarray(uv)[:: max(1, len(uv) // 4000)] % 1.0
    px = a[((1 - uv[:, 1]) * (h - 1)).astype(int), (uv[:, 0] * (w - 1)).astype(int)]
    side = int(np.sqrt(len(px)))
    return colour_names(Image.fromarray(px[: side * side].reshape(side, side, 3)), min_share=0.2)


def process_one(it: dict) -> dict:
    import coacd
    import trimesh
    from PIL import Image
    raw = asset_root() / "raw" / it["source"] / it["name"]
    rid = f"{it['source']}_{re.sub(r'[^a-z0-9]+', '_', it['name'].lower()).strip('_')}"
    if not (raw / "model.obj").exists():
        return {"id": rid, "reject": "no mesh"}
    mesh = trimesh.load(raw / "model.obj", force="mesh", process=False)
    lo, hi = mesh.bounds
    ext = hi - lo
    category = categorize(it, raw)
    win = scale_window(ext, category)
    if win is None:
        return {"id": rid, "reject": f"size {np.round(ext * 100, 1).tolist()} cm ({category})"}
    nominal = float(np.clip(1.0, *win))
    out = asset_root() / "objects" / rid
    out.mkdir(parents=True, exist_ok=True)
    mesh.apply_translation(-(lo + hi) / 2)                          # origin = bbox centre (z: bottom at -h/2)
    if (raw / "texture.png").exists():
        Image.open(raw / "texture.png").convert("RGB").resize((1024, 1024), Image.LANCZOS).save(out / "texture.png")
    _write_obj(out / "visual.obj", mesh)
    cm = coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces))
    parts = coacd.run_coacd(cm, threshold=0.05, max_convex_hull=12, preprocess_mode="auto", seed=0)
    hulls, vols = [], []
    for k, (v, f) in enumerate(parts):
        hull = trimesh.Trimesh(v, f).convex_hull
        hull.export(out / f"collision_{k:02d}.obj")
        hulls.append(f"objects/{rid}/collision_{k:02d}.obj")
        vols.append(float(hull.volume))
    vol = float(mesh.volume) if mesh.is_watertight else 0.6 * float(mesh.convex_hull.volume)
    _, density, friction = CATEGORY[category]
    mass_at = lambda s: float(np.clip(density * vol * s ** 3, *MASS))      # noqa: E731
    name = display_name(it, category)
    return {
        "id": rid, "name": name, "category": category, "shape": "mesh", "synonyms": [category] if category != name else [],
        "colours": _colours(mesh, raw / "texture.png"), "dims_m": [round(float(v) * nominal, 4) for v in ext],
        "scan_dims_m": [round(float(v), 4) for v in ext], "scale": round(nominal, 4),
        "scale_range": [round(win[0], 4), round(win[1], 4)], "mass_kg": round(mass_at(nominal), 4),
        "mass_range_kg": [round(mass_at(win[0]), 4), round(mass_at(win[1]), 4)], "friction_range": list(friction),
        "visual": {"mesh": f"objects/{rid}/visual.obj",
                   "texture": f"objects/{rid}/texture.png" if (out / "texture.png").exists() else None},
        "collision": hulls, "collision_volumes_m3": [round(v, 9) for v in vols], "joints": [],
        "graspable": None, "scripted_grasp_success": None, "gate": None, "split": None,
        "source": {"gso": "Google Scanned Objects", "ycb": "YCB"}[it["source"]], "source_name": it["name"],
        "licence": "CC-BY-4.0",
    }


def _write_obj(path: Path, mesh) -> None:
    """v / vt / f (MuJoCo reads the UVs; no material file)."""
    uv = getattr(mesh.visual, "uv", None)
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in np.asarray(mesh.vertices)]
    if uv is not None and len(uv) == len(mesh.vertices):
        lines += [f"vt {u:.6f} {v:.6f}" for u, v in np.asarray(uv)]
        lines += [f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}" for a, b, c in np.asarray(mesh.faces)]
    else:
        lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in np.asarray(mesh.faces)]
    path.write_text("\n".join(lines) + "\n")


def _safe_process(it: dict) -> dict:
    try:
        return process_one(it)
    except Exception as exc:                   # noqa: BLE001 (one broken scan must not stop the batch; reported)
        return {"id": f"{it['source']}_{it['name']}", "reject": f"error {type(exc).__name__}: {exc}"}


def assign_splits(rows: list[dict]) -> list[str]:
    """Hold out whole categories (stable crc32 order) until >= HELDOUT_FRACTION of the objects; returns them."""
    count: dict[str, int] = {}
    for r in rows:
        count[r["category"]] = count.get(r["category"], 0) + 1
    held, n = [], 0
    for c in sorted(count, key=lambda c: zlib.crc32(c.encode())):
        if n >= HELDOUT_FRACTION * len(rows) or c == "object":    # the catch-all stays in train
            continue
        held.append(c)
        n += count[c]
    for r in rows:
        r["split"] = "heldout" if r["category"] in held else "train"
    return held


def _merge(rows: list[dict], registry: Path) -> None:
    old = [json.loads(line) for line in registry.read_text().splitlines() if line.strip()] if registry.exists() else []
    new = {r["id"]: r for r in rows}
    prev = {r["id"]: r for r in old}
    for rid, r in new.items():
        if rid in prev:
            for k in ("graspable", "scripted_grasp_success", "gate"):
                r[k] = prev[rid].get(k)
    keep = [r for r in old if r["id"] not in new]
    registry.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in keep + list(new.values())))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("list", "fetch", "process"))
    ap.add_argument("--source", default="gso,ycb")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--registry", default=str(HERE / "registry.jsonl"))
    a = ap.parse_args()
    items = catalogue(a.source.split(","))
    if a.limit:
        items = items[: a.limit]
    if a.cmd == "list":
        for s in a.source.split(","):
            sel = [it for it in items if it["source"] == s]
            print(f"{s}: {len(sel)} objects, {sum(it['bytes'] for it in sel) / 1e9:.2f} GB listed")
        return
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with mp.get_context("spawn").Pool(a.workers) as pool:      # fetch: network-bound, process: CPU (nice 15, <= 12)
        if a.cmd == "fetch":
            res = pool.map(fetch_one, items, chunksize=1)
            print({k: res.count(k) for k in set(res)})
            return
        raw = asset_root() / "raw"
        items = [it for it in items if (raw / it["source"] / it["name"] / "model.obj").exists()
                 and (raw / it["source"] / it["name"] / "done").exists()]
        rows, rejects = [], []
        for r in pool.imap_unordered(_safe_process, items, chunksize=1):
            (rejects if "reject" in r else rows).append(r)
    (asset_root() / "objects").mkdir(parents=True, exist_ok=True)
    (asset_root() / "objects" / "rejects.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rejects))
    held = assign_splits(rows)
    _merge(rows, Path(a.registry))
    print(f"held-out categories: {held}")
    cats = {}
    for r in rows:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    print(f"kept {len(rows)}, rejected {len(rejects)}; categories {dict(sorted(cats.items()))}")


if __name__ == "__main__":
    main()
