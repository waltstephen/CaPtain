"""div3 scene distribution and training mix: wider table/light ranges, div3 layout styles (sub-bands, grid, two
clusters, receptacle placement, yaw modes, distractor stacks, 1-2 platforms), cube size bias, tray / big bowl / big mat
/ big primitive kinds, the div3 variants and the mix (groups, depth-precise down-weighting, per-task cap)."""
import dataclasses
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from soarm import mix, scene_random, taskspec
from soarm.tasks import SceneConfig

ROOT = Path(__file__).resolve().parent.parent
DIV3 = SceneConfig.from_json(json.loads((ROOT / "configs" / "div3_scene.json").read_text()))
needs_menagerie = pytest.mark.skipif(not os.environ.get("SOARM_MENAGERIE"), reason="SOARM_MENAGERIE not set")
DIV3_IDS = sorted(s["id"] for s in taskspec.registry().values() if "div3 variant" in s.get("notes", ""))


def test_div3_config_is_train_only_and_not_default():
    assert not DIV3.scene_is_default()
    assert DIV3.cube_split == "train" and DIV3.camera == "train" and DIV3.layout == "div3"
    assert DIV3.table_mount == "edge3" and DIV3.lights == "random3"
    assert len(DIV3.cube_size_weights) == len(DIV3.cube_sizes)


def test_edge3_table_covers_the_reach_and_varies_more_than_div2():
    rng = np.random.default_rng(3)
    t3 = [scene_random.sample_table(rng, "edge3") for _ in range(400)]
    for t in t3:
        assert t["back"] + t["depth"] >= 0.39                  # reach + clutter annulus end at r 0.29
        assert t["y"] - t["width"] / 2 <= -0.33 and t["y"] + t["width"] / 2 >= 0.33
        assert -0.92 <= t["floor_z"] <= -0.58
    t2 = [scene_random.sample_table(rng) for _ in range(400)]
    for key in ("depth", "width", "floor_z"):
        assert np.ptp([t[key] for t in t3]) > np.ptp([t[key] for t in t2])


def test_random3_lights_stay_in_range():
    rng = np.random.default_rng(4)
    fills = Counter()
    for _ in range(200):
        app = scene_random.sample_appearance(rng, table_textures="all", lights="random3", table_mount="edge3",
                                             views="train")
        el = np.rad2deg(np.arcsin(-app.key_dir[2]))
        assert 24.9 <= el <= 88.1 and 0.08 <= app.ambient <= 0.5
        assert app.view in scene_random.VIEW_SETS["train"]
        fills[len(app.lights)] += 1
    assert set(fills) == {0, 1, 2, 3}
    with pytest.raises(ValueError):
        scene_random.sample_appearance(rng, lights="bright")


def test_div3_layout_styles_stay_in_the_band():
    rng = np.random.default_rng(5)
    for style in taskspec.LAYOUT_STYLES3:
        f = taskspec.layout_sampler(rng, style)
        pts = [p for p in (f([]) for _ in range(300)) if p is not None]
        assert len(pts) > 30 and all(taskspec.in_band(p) for p in pts), style
    r = {s: np.mean([np.hypot(*p) for p in (taskspec.layout_sampler(np.random.default_rng(k), s)([])
                                            for k in range(200)) if p is not None]) for s in ("near", "far")}
    assert r["near"] + 0.02 < r["far"]


@pytest.mark.parametrize("sid", DIV3_IDS + ["cube_stack_two", "cube_v_in_bowl", "cube_v_sort_one_each", "gen_park_beside_tower_base"])
def test_div3_samples_are_deterministic_reachable_and_unambiguous(sid):
    spec = taskspec.get(sid)
    for seed in range(12):
        t = taskspec.sample(spec, 1_500_000 + seed, DIV3)
        assert t.to_json() == taskspec.sample(spec, 1_500_000 + seed, DIV3).to_json()
        role_names = {n for v in t.roles.values() for n in v.split(",")}
        clutter = [o for o in t.objects if o.name not in role_names]
        assert len(clutter) <= DIV3.clutter[1] and all(o.kind in DIV3.clutter_kinds for o in clutter)
        for o in t.objects:
            p = t.layout[o.name]
            if o.kind in ("cube", "cylinder", "block") and p.on is None and o not in clutter \
                    and not spec.get("workspace") and not any(c.get("type") == "region" for c in spec.get("layout", [])):
                assert taskspec.in_band(p.xy), (o.name, p.xy)
        assert 1 <= len(t.platforms) <= DIV3.platforms_max or not t.platforms


def test_pseudo_kinds_have_their_sizes_and_nouns():
    seen = {}
    for sid in DIV3_IDS:
        t = taskspec.sample(taskspec.get(sid), 1_500_100, DIV3)
        for o in t.objects:
            if o.attrs.get("variant"):
                seen.setdefault(o.attrs["variant"], []).append(o)
    assert set(seen) == set(taskspec.PSEUDO_KINDS)
    for kind, objs in seen.items():
        real, noun = taskspec.PSEUDO_KINDS[kind]
        assert all(o.kind == real and o.noun == noun for o in objs)
    assert all(0.14 <= o.size[0] <= 0.18 for o in seen["tray"]) and all(o.size[0] >= 0.12 for o in seen["big_mat"])
    assert all(o.size[0] >= 0.13 for o in seen["big_bowl"]) and all(o.size[0] >= 0.036 for o in seen["big_cylinder"])
    assert "tray" in taskspec.sample(taskspec.get("cube_v_in_tray"), 1_500_000, DIV3).language


def test_cube_sizes_are_biased_toward_large():
    spec = taskspec.get("cube_v_in_tray")
    sizes = Counter()
    for seed in range(300):
        t = taskspec.sample(spec, 1_600_000 + seed, DIV3)
        sizes.update(o.attrs["size_class"] for o in t.objects if o.kind == "cube")
    big = sum(v for k, v in sizes.items() if k >= 4.0) / sum(sizes.values())
    assert big > 0.68, sizes            # uniform: 0.60


def test_distractor_stacks_receptacle_bands_and_yaw_modes_occur():
    stacked, aligned = 0, 0
    spec = taskspec.get("cube_v_lift_dense")      # 3-4 distractor cubes
    for seed in range(200):
        t = taskspec.sample(spec, 1_700_000 + seed, DIV3)
        ups = [n for n, p in t.layout.items() if p.on and p.on.startswith("cube_")]
        if ups:
            stacked += 1
            below = {o.name: o for o in t.objects}
            assert all(below[n].size[0] <= below[t.layout[n].on].size[0] + 0.005 for n in ups)
            assert not set(ups) & {a for _, args, _ in t.atoms for a in args}
        yaws = [p.yaw for p in t.layout.values() if p.on is None]
        if all(min(abs((y + np.pi / 4) % (np.pi / 2) - np.pi / 4), 1) < 0.2 for y in yaws):
            aligned += 1
    assert 10 <= stacked <= 60 and aligned >= 20


def test_div2_draws_are_unchanged_by_div3_fields():
    div2 = mix.load("div2")["config"]
    assert div2.platforms_max == 1 and div2.distractor_stacks == 0 and not div2.cube_size_weights
    assert div2.layout == "mixed" and div2.table_mount == "edge" and div2.lights == "random"


def test_div3_mix_is_trainable_balanced_and_downweights_precise_tasks():
    m = mix.load("div3")
    w, raw = m["weights"], m["raw"]
    assert abs(sum(w.values()) - 1) < 1e-9 and len(w) >= 120
    assert max(w.values()) <= raw["max_share"] + 1e-4
    assert m["config"] == DIV3 and m["intro"] == "v4"
    groups = Counter()
    for t in raw["tasks"]:
        groups[t["group"]] += t["weight"]
        assert t["scripted"][0] >= 0.5 * t["scripted"][1]
    assert sum(v for g, v in groups.items() if g.startswith("cube")) >= 0.65 and groups["cube_div3"] >= 0.2
    prec = [t for t in raw["tasks"] if t.get("precise")]
    loose = [t for t in raw["tasks"] if not t.get("precise") and t["group"] == "cube_div3"]
    assert prec and all(t["weight"] > 0 for t in prec)
    assert np.mean([t["weight"] for t in prec]) < 0.6 * np.mean([t["weight"] for t in loose])
    for sid in ("cube_stack_two", "cube_v_slot_small", "cube_line_up"):
        assert sid in w, sid                   # precise tasks stay in the mix


@needs_menagerie
@pytest.mark.parametrize("sid,view", [("cube_v_two_in_tray", "front"), ("cube_v_sort_size_tray_mat", "side"),
                                      ("obj_big_cylinder_in_big_bowl", "low"), ("cube_v_lift_dense", "oblique")])
def test_div3_physics_scene_builds(sid, view):
    from soarm.sim import SoarmSim
    sim = SoarmSim(sid, 1_500_000, config=dataclasses.replace(DIV3, camera=view))
    app = sim.rand.appearance
    assert app.view == view and app.table is not None
    g = sim.model.geom("table")
    assert abs(sim.model.geom_size[g.id][0] - app.table["depth"] / 2) < 1e-9
    assert not any(sim.atom_values())
