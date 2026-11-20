"""Procedural cube suite of the SO-101 twin (PIVOT 11-09: rigid cubes, side 3-5 cm, is the training domain).

    python -m assets.build_cubes [--out $SOARM_ASSETS] [--registry assets/registry.jsonl]

Writes, per cube, a beveled visual mesh (OBJ with UVs into a 3x2 face atlas) and its atlas texture (PNG) under
<out>/cubes/<id>/, and one registry row per cube (assets/registry.jsonl; rows of other sources are kept). Everything is
generated here from a fixed seed, so the suite is CC0 by construction (letter glyphs: DejaVu Sans Bold, Bitstream Vera
licence, free for any use) and identical on every machine; `tools/gate_assets.py` adds the physics gate fields.

Suite: 5 train size classes (3.0-5.0 cm) x {12 matte plastic colours x 2 tones, 4 wood tones} = 140 train;
held out (CONTRACT 2.1 + PIVOT): 6 letter blocks + 8 printed-pattern blocks x 5 classes, and 16 cubes (12 colours,
2 wood, 2 letter blocks) in each held-out size class 2.5 / 5.5 / 6.0 cm.
"""
from __future__ import annotations

import argparse
import colorsys
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
TILE = 256                                    # px per face
FACES = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))   # atlas order, tile k = (k % 3, k // 3)
TRAIN_CM = (3.0, 3.5, 4.0, 4.5, 5.0)
HELDOUT_CM = (2.5, 5.5, 6.0)
JITTER = 0.03                                 # +-3 % side jitter, applied per episode by the loader
MASS_RANGE = (0.0085, 0.150)                  # kg (PIVOT: 8.5-150 g)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# matte plastic colours: names = soarm.objects.COLORS keys (the language vocabulary), RGB tuned to toy-plastic tones
PLASTIC = {
    "red": (0.80, 0.10, 0.09), "green": (0.13, 0.58, 0.22), "blue": (0.10, 0.28, 0.80), "yellow": (0.96, 0.82, 0.10),
    "orange": (0.96, 0.46, 0.07), "purple": (0.48, 0.20, 0.66), "pink": (0.95, 0.50, 0.70), "cyan": (0.08, 0.72, 0.82),
    "white": (0.92, 0.92, 0.89), "black": (0.08, 0.08, 0.08), "brown": (0.45, 0.27, 0.12), "gray": (0.50, 0.50, 0.50),
}
WOOD = {"pine": (0.86, 0.70, 0.47), "beech": (0.80, 0.60, 0.42), "oak": (0.66, 0.50, 0.31), "walnut": (0.42, 0.28, 0.17)}
# material -> (density kg/m3, friction): toy plastic blocks are often hollow (250) up to solid ABS (1200)
MATERIAL = {"plastic": ((250.0, 1200.0), (0.5, 0.9)), "wood": ((380.0, 800.0), (0.6, 1.1))}
LETTER_SETS = ("ABCDEF", "GHIJKL", "MNOPQR", "STUVWX", "YZ1234", "56789A")
LETTER_BASES = (("wood", "pine", "red"), ("wood", "beech", "blue"), ("plastic", "yellow", "blue"),
                ("plastic", "white", "red"), ("plastic", "red", "white"), ("plastic", "blue", "yellow"))
PRINTS = (("stripes", "red", "white"), ("dots", "blue", "white"), ("checker", "black", "white"),
          ("diagonal", "yellow", "black"), ("grid", "white", "blue"), ("halves", "green", "orange"),
          ("squares", "purple", "yellow"), ("chevron", "orange", "cyan"))


# ---------------------------------------------------------------------------------------------------- geometry
def _grid(h: float, r: float, n_bevel: int = 4, n_flat: int = 4) -> np.ndarray:
    th = np.linspace(0.0, np.pi / 2, n_bevel + 1)
    edge = -h + r * (1.0 - np.cos(th))                         # dense across the bevel
    flat = np.linspace(-h + r, h - r, n_flat + 1)[1:-1]
    return np.concatenate([edge, flat, -edge[::-1]])


def face_frame(n: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(normal, right, up) of a face as seen from outside; letters are upright in this frame."""
    n = np.asarray(n, float)
    up = np.array([1.0, 0, 0]) if abs(n[2]) > 0.5 else np.array([0, 0, 1.0])
    return n, np.cross(up, n), up


def beveled_cube(side: float, bevel: float):
    """Vertices, normals, atlas UVs and triangles of a cube with rounded edges/corners (radius `bevel`)."""
    h, r = side / 2, bevel
    g = _grid(h, r)
    k = len(g)
    verts, norms, uvs, tris = [], [], [], []
    margin = 2.0 / TILE
    for f, n in enumerate(FACES):
        nrm, right, up = face_frame(n)
        col, row = f % 3, f // 3
        base = len(verts)
        for j, b in enumerate(g):            # up coordinate
            for i, a in enumerate(g):        # right coordinate
                p = nrm * h + right * a + up * b
                q = np.clip(p, -(h - r), h - r)
                d = p - q
                dn = d / np.linalg.norm(d)
                verts.append(q + r * dn)
                norms.append(dn)
                s, t = (a + h) / (2 * h), (b + h) / (2 * h)
                u = (col + margin + (1 - 2 * margin) * s) / 3
                v = 1.0 - (row + margin + (1 - 2 * margin) * (1 - t)) / 2      # image row 0 = top -> OBJ v = 1
                uvs.append((u, v))
        for j in range(k - 1):
            for i in range(k - 1):
                v00, v10 = base + j * k + i, base + j * k + i + 1
                v01, v11 = v00 + k, v10 + k
                tris += [(v00, v10, v11), (v00, v11, v01)]
    return np.array(verts), np.array(norms), np.array(uvs), np.array(tris)


def write_obj(path: Path, verts, norms, uvs, tris) -> None:
    lines = ["# beveled cube, generated by robots/soarm/assets/build_cubes.py (CC0)"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
    lines += [f"vt {u:.6f} {v:.6f}" for u, v in uvs]
    lines += [f"vn {x:.5f} {y:.5f} {z:.5f}" for x, y, z in norms]
    lines += ["f " + " ".join(f"{i + 1}/{i + 1}/{i + 1}" for i in t) for t in tris]
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------------------------------- textures
def _noise(rng: np.random.Generator, size: int, scale: int) -> np.ndarray:
    """Smooth value noise in [-1, 1] (bilinear upsampling of a coarse grid)."""
    coarse = rng.uniform(-1, 1, (scale + 1, scale + 1)).astype(np.float32)
    img = Image.fromarray(coarse).resize((size, size), Image.BILINEAR)
    return np.asarray(img, np.float32)


def _jitter_rgb(rng: np.random.Generator, rgb, dh=0.012, ds=0.06, dv=0.06):
    hh, s, v = colorsys.rgb_to_hsv(*rgb)
    return colorsys.hsv_to_rgb((hh + rng.uniform(-dh, dh)) % 1.0, float(np.clip(s * (1 + rng.uniform(-ds, ds)), 0, 1)),
                               float(np.clip(v * (1 + rng.uniform(-dv, dv)), 0, 1)))


def plastic_tile(rng, rgb) -> np.ndarray:
    base = np.asarray(rgb, np.float32)
    shade = 1.0 + 0.025 * _noise(rng, TILE, 6) + 0.012 * _noise(rng, TILE, 48)
    return np.clip(base[None, None] * shade[..., None], 0, 1)


def wood_tile(rng, rgb, end_grain: bool) -> np.ndarray:
    yy, xx = np.mgrid[0:TILE, 0:TILE].astype(np.float32) / TILE
    warp = 0.06 * _noise(rng, TILE, 4) + 0.01 * _noise(rng, TILE, 32)
    if end_grain:                                      # growth rings around an off-centre pith
        cx, cy = rng.uniform(-0.6, 1.6, 2)
        ring = np.hypot(xx - cx, yy - cy) * rng.uniform(14, 22) + 6 * warp
    else:                                              # long grain along the face's up axis
        ring = xx * rng.uniform(10, 18) + 8 * warp
    band = 0.5 + 0.5 * np.sin(2 * np.pi * ring)
    shade = 1.0 - 0.16 * band ** 3 + 0.03 * _noise(rng, TILE, 64)
    return np.clip(np.asarray(rgb, np.float32)[None, None] * shade[..., None], 0, 1)


def _font(px: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT, px)


def letter_tile(base_tile: np.ndarray, ink, char: str) -> np.ndarray:
    img = Image.fromarray((base_tile * 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    ink8 = tuple(int(255 * c) for c in ink)
    w = int(TILE * 0.07)
    d.rectangle([w, w, TILE - 1 - w, TILE - 1 - w], outline=ink8, width=max(3, TILE // 32))
    d.text((TILE / 2, TILE / 2), char, font=_font(int(TILE * 0.62)), fill=ink8, anchor="mm")
    return np.asarray(img, np.float32) / 255


def printed_tile(rng, pattern: str, a, b) -> np.ndarray:
    yy, xx = np.mgrid[0:TILE, 0:TILE].astype(np.float32) / TILE
    if pattern == "stripes":
        m = (np.floor(yy * 6) % 2) == 0
    elif pattern == "dots":
        m = np.hypot((xx * 4) % 1 - 0.5, (yy * 4) % 1 - 0.5) < 0.28
    elif pattern == "checker":
        m = (np.floor(xx * 4) + np.floor(yy * 4)) % 2 == 0
    elif pattern == "diagonal":
        m = (np.floor((xx + yy) * 5) % 2) == 0
    elif pattern == "grid":
        m = ((xx * 5) % 1 < 0.18) | ((yy * 5) % 1 < 0.18)
    elif pattern == "halves":
        m = xx < 0.5
    elif pattern == "squares":
        m = (np.floor(np.maximum(np.abs(xx - 0.5), np.abs(yy - 0.5)) * 8) % 2) == 0
    elif pattern == "chevron":
        m = (np.floor((yy + np.abs(xx - 0.5)) * 6) % 2) == 0
    else:
        raise ValueError(pattern)
    out = np.where(m[..., None], np.asarray(a, np.float32), np.asarray(b, np.float32))
    return np.clip(out * (1.0 + 0.02 * _noise(rng, TILE, 8))[..., None], 0, 1)


def atlas(tiles: list[np.ndarray]) -> Image.Image:
    img = np.zeros((2 * TILE, 3 * TILE, 3), np.float32)
    for f, t in enumerate(tiles):
        col, row = f % 3, f // 3
        img[row * TILE:(row + 1) * TILE, col * TILE:(col + 1) * TILE] = t
    return Image.fromarray((np.clip(img, 0, 1) * 255).round().astype(np.uint8))


# ---------------------------------------------------------------------------------------------------- colour naming
def colour_names(img: Image.Image, min_share: float = 0.15) -> list[str]:
    """Named colours covering >= min_share of the texture (most frequent first), nearest in a hue-aware RGB metric."""
    names = list(PLASTIC)
    ref = np.array([PLASTIC[n] for n in names], np.float32)
    px = np.asarray(img.resize((96, 64)), np.float32).reshape(-1, 3) / 255
    d = ((px[:, None, :] - ref[None]) ** 2 * np.array([0.30, 0.59, 0.11]) * 3).sum(-1)
    idx = d.argmin(1)
    share = np.bincount(idx, minlength=len(names)) / len(idx)
    order = np.argsort(-share)
    return [names[i] for i in order if share[i] >= min_share]


def nearest_colour(rgb) -> str:
    return colour_names(Image.new("RGB", (4, 4), tuple(int(255 * c) for c in rgb)), 0.5)[0]


# ---------------------------------------------------------------------------------------------------- suite
def _row(cid, name, synonyms, side, split, material, texture, extra, colours, color, rel_mesh, rel_tex):
    (dlo, dhi), fr = MATERIAL[material]
    vol = side ** 3
    mlo, mhi = float(np.clip(dlo * vol, *MASS_RANGE)), float(np.clip(dhi * vol, *MASS_RANGE))
    return {
        "id": cid, "name": name, "category": "cube", "shape": "cube", "synonyms": synonyms, "colours": colours,
        "color": color, "dims_m": [side, side, side], "size_class": round(side * 100, 1),
        "scale_jitter": JITTER, "bevel_m": extra.pop("bevel_m"), "material": material, "texture": texture,
        "mass_kg": round(float(np.clip(0.5 * (dlo + dhi) * vol, *MASS_RANGE)), 4),
        "mass_range_kg": [round(mlo, 4), round(mhi, 4)], "density_range_kg_m3": [dlo, dhi], "friction_range": list(fr),
        "visual": {"mesh": rel_mesh, "texture": rel_tex},
        "collision": [{"type": "box", "size_m": [side, side, side]}], "joints": [],
        "graspable": None, "scripted_grasp_success": None, "gate": None, "split": split,
        "source": "procedural (robots/soarm/assets/build_cubes.py)", "licence": "CC0-1.0", **extra,
    }


def build(out: Path, seed: int = 20261109) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    classes = [(cm, "train") for cm in TRAIN_CM] + [(cm, "heldout") for cm in HELDOUT_CM]
    for cm, size_split in classes:
        side = cm / 100
        bevel = float(np.clip(0.06 * side, 0.0015, 0.003))
        tag = f"{int(round(cm * 10)):02d}"
        kinds: list[tuple] = [("plastic", (c, v)) for c in PLASTIC for v in "ab"] + [("wood", w) for w in WOOD] + \
                             [("letter", k) for k in range(len(LETTER_SETS))] + [("printed", k) for k in range(len(PRINTS))]
        if size_split == "heldout":                    # held-out sizes: plain colours + 2 wood + 2 letter blocks
            kinds = [k for k in kinds if (k[0] == "plastic" and k[1][1] == "a")
                     or k in (("wood", "pine"), ("wood", "walnut"), ("letter", 0), ("letter", 2))]
        for kind, which in kinds:
            if kind == "plastic":                      # two tones per colour (a: catalogue tone, b: wider shift)
                which, tone = which
                for _ in range(50):                    # the shifted tone must still read as its colour name
                    rgb = _jitter_rgb(rng, PLASTIC[which]) if tone == "a" else \
                        _jitter_rgb(rng, PLASTIC[which], dh=0.02, ds=0.15, dv=0.15)
                    if nearest_colour(rgb) == which:
                        break
                tiles = [plastic_tile(rng, rgb) for _ in FACES]
                cid, material, tclass = f"cube_{tag}_{which}_{tone}", "plastic", "plain"
                name = f"{which} cube"
                syn = [f"{which} block", f"{which} plastic cube", f"{which} toy block"]
                extra = {}
            elif kind == "wood":
                rgb = _jitter_rgb(rng, WOOD[which], dh=0.005, ds=0.04, dv=0.05)
                tiles = [wood_tile(rng, rgb, end_grain=abs(n[2]) == 1) for n in FACES]
                cid, material, tclass = f"cube_{tag}_wood_{which}", "wood", "wood"
                name = "wooden cube"
                syn = ["wood cube", "wooden block", "wood block", f"{which} cube", "natural wood cube"]
                extra = {"wood": which}
            elif kind == "letter":
                mat, base, ink = LETTER_BASES[which]
                chars = LETTER_SETS[which]
                brgb = _jitter_rgb(rng, WOOD[base] if mat == "wood" else PLASTIC[base])
                irgb = _jitter_rgb(rng, PLASTIC[ink])
                tiles = [letter_tile(wood_tile(rng, brgb, abs(n[2]) == 1) if mat == "wood" else plastic_tile(rng, brgb),
                                     irgb, ch) for n, ch in zip(FACES, chars)]
                cid, material, tclass = f"cube_{tag}_letters_{chars.lower()}", mat, "letters"
                name = "letter block"
                syn = ["alphabet block", "ABC block", "letter cube", "toy block with letters"]
                # letter = the face up at spawn (+z; objects spawn upright with a random yaw)
                extra = {"letter": chars[4], "letters": {"+x": chars[0], "-x": chars[1], "+y": chars[2],
                                                         "-y": chars[3], "+z": chars[4], "-z": chars[5]}, "ink": ink}
            else:
                pattern, a, b = PRINTS[which]
                ca, cb = _jitter_rgb(rng, PLASTIC[a]), _jitter_rgb(rng, PLASTIC[b])
                tiles = [printed_tile(rng, pattern, ca, cb) for _ in FACES]
                cid, material, tclass = f"cube_{tag}_print_{pattern}", "plastic", "printed"
                name = f"{pattern} cube"
                syn = [f"{a} and {b} cube", f"{pattern} block", "printed cube", "patterned block"]
                extra = {"pattern": pattern}
            img = atlas(tiles)
            d = out / "cubes" / cid
            d.mkdir(parents=True, exist_ok=True)
            img.save(d / "texture.png", optimize=True)
            write_obj(d / "mesh.obj", *beveled_cube(side, bevel))
            colours = colour_names(img)
            if tclass == "plain":
                if colours[:1] != [which]:
                    raise ValueError(f"{cid}: texture reads as {colours}, not {which}")
                color = which
            elif tclass == "wood" or (tclass == "letters" and material == "wood"):
                color = "wood"
            else:
                color = colours[0]
            if color == "wood" and "wood" not in colours:
                colours = ["wood", *colours]
            # held out: sizes 2.5/5.5/6.0, letter blocks, printed patterns (CONTRACT 2.1 + PIVOT)
            split = "heldout" if size_split == "heldout" or tclass in ("letters", "printed") else "train"
            rows.append(_row(cid, name, syn, side, split, material, tclass, {"bevel_m": round(bevel, 4), **extra},
                             colours, color, f"cubes/{cid}/mesh.obj", f"cubes/{cid}/texture.png"))
    return rows


def merge_registry(path: Path, rows: list[dict]) -> None:
    """Replace this builder's rows (same ids, keeping gate fields of unchanged ids) and keep every other row."""
    old = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    by_id = {r["id"]: r for r in old}
    keep = [r for r in old if not r["id"].startswith("cube_")]
    for r in rows:
        prev = by_id.get(r["id"])
        if prev is not None:
            for k in ("graspable", "scripted_grasp_success", "gate"):
                r[k] = prev.get(k)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in keep + rows))


def main() -> None:
    root = os.environ.get("SOARM_ASSETS") or str(Path(os.environ.get("SOARM_ROOT", "~/soarm_work")) / "assets")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=root)
    ap.add_argument("--registry", default=str(HERE / "registry.jsonl"))
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    rows = build(out)
    merge_registry(Path(a.registry), rows)
    n = {s: sum(r["split"] == s for r in rows) for s in ("train", "heldout")}
    print(f"{len(rows)} cubes ({n}) -> {out / 'cubes'}; registry {a.registry}")


if __name__ == "__main__":
    main()
