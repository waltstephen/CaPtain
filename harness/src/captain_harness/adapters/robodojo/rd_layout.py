"""Reachable-layout variant of an official RoboDojo eval layout (ROUND 4, decided a priori, disclosed).

Rule (generic, same for every task; never looks at task semantics or results):
  * Every object resting on the table (relative_plane == "Table": task objects, fixtures, clutter) is pulled radially
    toward the NEAREST arm base until it lies in that arm's comfortable straight-down reach band:
    REACH_MIN_M <= xy distance <= REACH_MAX_M, and at least FORWARD_MIN_M in front of the bases.
  * Objects resting on another object (relative_plane = "<parent>/...") move with their support.
  * Objects on the ground (e.g. RoboDojo's dustbin beside the table) and the camera stand are left unchanged.
  * Overlaps created by the pull are resolved by pushing pairs apart along their centre line (footprint = asset
    aligned-bbox half-diagonal x scale + 1 cm), re-clamping to the band; residual overlaps are reported.
  * Heights and orientations are unchanged.
World frame of the layout: robots face +y; arm bases at the robot config's default_root_pos.
"""
from __future__ import annotations

import json
import math
import os
from copy import deepcopy

REACH_MIN_M = 0.17
REACH_MAX_M = 0.36
FORWARD_MIN_M = 0.12
CLEARANCE_M = 0.01
MOVABLE_TYPES = ("Rigid", "Geometry", "Articulation", "Dynamic")


def _footprint(assets_root, typ, inst):
    cat_dir = "Clutter" if inst.get("type") == "cluttered" else typ
    idx = inst.get("clutter_idx", inst.get("category_idx", 0)) if inst.get("type") == "cluttered" else inst.get("category_idx", 0)
    cat = inst.get("category") or inst.get("_category")
    path = os.path.join(assets_root, "Object", "RoboDojo", cat_dir, str(cat), f"{int(idx):05d}", "metadata.json")
    scale = inst.get("scale") or [1.0, 1.0, 1.0]
    try:
        ext = json.load(open(path))["geometry"]["aligned_bbox"]["extents"]
        return 0.5 * math.hypot(ext[0] * scale[0], ext[1] * scale[1])
    except Exception:  # noqa: BLE001
        return 0.05


def make_reachable(layout: dict, bases_xy, assets_root: str):
    """Return (new_layout, record). bases_xy: list of (x, y) world positions of the arm bases (robots face +y)."""
    lay = deepcopy(layout)
    base_y = min(b[1] for b in bases_xy)
    items = []   # (typ, cat, index, inst)
    for typ in MOVABLE_TYPES:
        for cat, insts in (lay.get(typ) or {}).items():
            if not isinstance(insts, list):
                continue
            for i, inst in enumerate(insts):
                if isinstance(inst, dict) and "default_pos" in inst:
                    inst.setdefault("_category", cat)
                    items.append((typ, cat, i, inst))
    table = [it for it in items if it[3].get("relative_plane") == "Table"]
    pos = {id(it[3]): list(it[3]["default_pos"]) for it in table}
    rad = {id(it[3]): _footprint(assets_root, it[0], it[3]) for it in table}

    def nearest(p):
        return min(bases_xy, key=lambda b: math.hypot(p[0] - b[0], p[1] - b[1]))

    def clamp(p):
        b = nearest(p)
        dx, dy = p[0] - b[0], p[1] - b[1]
        d = math.hypot(dx, dy) or 1e-9
        if d > REACH_MAX_M:
            dx, dy = dx * REACH_MAX_M / d, dy * REACH_MAX_M / d
        elif d < REACH_MIN_M:
            dx, dy = dx * REACH_MIN_M / d, dy * REACH_MIN_M / d
        x, y = b[0] + dx, b[1] + dy
        y = max(y, base_y + FORWARD_MIN_M)
        return [x, y, p[2]]

    for it in table:
        pos[id(it[3])] = clamp(pos[id(it[3])])
    keys = [id(it[3]) for it in table]
    for _ in range(80):
        moved = False
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                pa, pb = pos[keys[a]], pos[keys[b]]
                need = rad[keys[a]] + rad[keys[b]] + CLEARANCE_M
                dx, dy = pb[0] - pa[0], pb[1] - pa[1]
                d = math.hypot(dx, dy)
                if d < need:
                    if d < 1e-6:
                        dx, dy, d = 1.0, 0.0, 1.0
                    push = 0.5 * (need - d) + 1e-3
                    ux, uy = dx / d, dy / d
                    pos[keys[a]] = clamp([pa[0] - ux * push, pa[1] - uy * push, pa[2]])
                    pos[keys[b]] = clamp([pb[0] + ux * push, pb[1] + uy * push, pb[2]])
                    moved = True
        if not moved:
            break
    residual = []
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            pa, pb = pos[keys[a]], pos[keys[b]]
            if math.hypot(pb[0] - pa[0], pb[1] - pa[1]) < rad[keys[a]] + rad[keys[b]] - 1e-3:
                residual.append([table[a][3].get("label") or table[a][1], table[b][3].get("label") or table[b][1]])
    record = {"rule": {"reach_min_m": REACH_MIN_M, "reach_max_m": REACH_MAX_M, "forward_min_m": FORWARD_MIN_M,
                       "clearance_m": CLEARANCE_M, "bases_world_xy": [list(b) for b in bases_xy]},
              "moved": [], "residual_overlaps": residual}
    disp = {}
    for typ, cat, i, inst in table:
        old = list(inst["default_pos"])
        new = pos[id(inst)]
        disp[cat + "/" + cat + "/" + str(i)] = (new[0] - old[0], new[1] - old[1])
        disp[inst.get("label") or ""] = disp[cat + "/" + cat + "/" + str(i)]
        inst["default_pos"] = [round(new[0], 4), round(new[1], 4), old[2]]
        record["moved"].append({"type": typ, "category": cat, "label": inst.get("label"), "from": [round(v, 4) for v in old],
                                "to": inst["default_pos"], "shift_m": round(math.hypot(new[0] - old[0], new[1] - old[1]), 4),
                                "footprint_r_m": round(rad[id(inst)], 4)})
    # objects resting on another object follow their support
    for typ, cat, i, inst in items:
        rp = inst.get("relative_plane")
        if rp in ("Table", "Ground", None):
            continue
        key = rp if rp in disp else "/".join(str(rp).split("/")[:3])
        if key in disp:
            dx, dy = disp[key]
            old = list(inst["default_pos"])
            inst["default_pos"] = [round(old[0] + dx, 4), round(old[1] + dy, 4), old[2]]
            record["moved"].append({"type": typ, "category": cat, "label": inst.get("label"), "from": old,
                                    "to": inst["default_pos"], "follows": rp})
    for typ, cat, i, inst in items:
        inst.pop("_category", None)
    return lay, record
