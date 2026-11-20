"""div2 scene distribution and training mix: edge-mounted table, camera view styles (train vs held-out), layout
styles, clutter/platforms, parameterised variants and the mix file."""
import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from soarm import mix, scene_random, taskspec
from soarm.tasks import SceneConfig

ROOT = Path(__file__).resolve().parent.parent
DIV2 = mix.load("div2")["config"]
needs_menagerie = pytest.mark.skipif(not os.environ.get("SOARM_MENAGERIE"), reason="SOARM_MENAGERIE not set")


def _az_el(pos):
    d = np.asarray(pos) - np.array([0.20, 0.0, 0.015])
    return abs(np.rad2deg(np.arctan2(d[1], d[0]))), np.rad2deg(np.arctan2(d[2], np.hypot(d[0], d[1])))


def test_default_config_draws_nothing_new():
    assert SceneConfig().scene_is_default()
    assert not DIV2.scene_is_default()
    assert DIV2.table_mount == "edge" and DIV2.camera == "train" and DIV2.layout == "mixed"


@pytest.mark.parametrize("style", sorted(scene_random.VIEW_STYLES))
def test_every_view_style_sees_the_workspace(style):
    rng = np.random.default_rng(0)
    for _ in range(40):
        pos, quat, fovy, got = scene_random.sample_view(rng, style)
        assert got == style and scene_random.VIEW_FOVY[0] <= fovy <= scene_random.VIEW_FOVY[1]
        assert scene_random.sees_workspace(pos, quat, fovy) and pos[2] > 0.12
        az, el = _az_el(pos)        # look-at jitter moves az/el a little around the nominal centre
        s = scene_random.VIEW_STYLES[style]
        assert s["el"][0] - 6 <= el <= s["el"][1] + 6 and s["az"][0] - 12 <= az <= s["az"][1] + 12, (az, el)


def test_train_and_heldout_views_are_disjoint():
    train, held = scene_random.VIEW_SETS["train"], scene_random.VIEW_SETS["heldout"]
    assert not set(train) & set(held) and abs(sum(train.values()) - 1) < 1e-9
    st = scene_random.VIEW_STYLES
    for t in train:      # held-out = |az| > 130 or el > 75; no train style reaches either
        assert st[t]["az"][1] <= 100 and st[t]["el"][1] <= 72
    assert all(st[h]["az"][0] >= 130 or st[h]["el"][0] >= 75 for h in held)


def test_edge_table_covers_the_reach_behind_and_beside_the_base():
    rng = np.random.default_rng(1)
    for _ in range(200):
        t = scene_random.sample_table(rng)
        assert -0.10 <= t["back"] <= -0.06 and t["back"] + t["depth"] >= 0.45
        assert t["y"] - t["width"] / 2 <= -0.36 and t["y"] + t["width"] / 2 >= 0.36
        assert -0.80 <= t["floor_z"] <= -0.65


def test_layout_styles_stay_in_the_band_and_vary():
    rng = np.random.default_rng(2)
    for style in taskspec.LAYOUT_STYLES:
        f = taskspec.layout_sampler(rng, style)
        pts = [p for p in (f([]) for _ in range(300)) if p is not None]
        assert len(pts) > 30 and all(taskspec.in_band(p) for p in pts), style
    # cluster draws are tighter than uniform ones
    def spread(style, k):
        f = taskspec.layout_sampler(np.random.default_rng(k), style)
        return np.std([q for q in (f([]) for _ in range(50)) if q is not None], axis=0).sum()

    spreads = {s: np.mean([spread(s, k) for k in range(10)]) for s in ("cluster", "uniform")}
    assert spreads["cluster"] < 0.6 * spreads["uniform"]


@pytest.mark.parametrize("sid", ["cube_stack_two", "cube_v_in_bowl", "cube_v_sort_one_each", "obj_cylinder_lift",
                                 "cube_v_stack_on_mat_cube", "cube_v_empty_container"])
def test_div2_samples_keep_cubes_reachable_and_clutter_unreferenced(sid):
    spec = taskspec.get(sid)
    used = {k for o in spec["objects"] for k in ([o["kind"]] if isinstance(o["kind"], str) else o["kind"])}
    for seed in range(20):
        t = taskspec.sample(spec, 1_500_000 + seed, DIV2)
        assert t.to_json() == taskspec.sample(spec, 1_500_000 + seed, DIV2).to_json()
        role_names = {n for v in t.roles.values() for n in v.split(",")}
        clutter = [o for o in t.objects if o.name not in role_names]
        assert len(clutter) <= DIV2.clutter[1] and all(o.kind in DIV2.clutter_kinds and o.kind not in used
                                                       for o in clutter)
        args = {a for _, aa, p in t.atoms for a in aa}
        assert not args & {o.name for o in clutter}
        for o in t.objects:
            p = t.layout[o.name]
            if o.kind == "cube" and p.on is None and not spec.get("workspace"):
                assert taskspec.in_band(p.xy), (o.name, p.xy)
        for pl in t.platforms:
            assert 0.01 <= pl["size"][2] <= 0.04


def test_variants_are_generated_and_train_safe():
    out = subprocess.run([sys.executable, str(ROOT / "tools" / "make_variants.py"), "--check"], capture_output=True,
                         text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    var = [s for s in taskspec.registry().values() if 200 <= s["tid"] < 1000 or s["domain"] == "broad"]
    assert len(var) >= 60
    for s in var:
        assert s["split"] == "train" and mix.train_safe(s) is None, s["id"]


def test_heldout_family_structure_is_refused():
    for sid in ("cube_swap", "cube_gather", "cube_count_place", "cube_unstack"):
        s = dict(taskspec.get(sid), split="train")
        assert mix.train_safe(s), sid


def test_div2_mix_is_trainable_and_balanced():
    m = mix.load("div2")
    w = m["weights"]
    assert abs(sum(w.values()) - 1) < 1e-9 and len(w) >= 50
    assert max(w.values()) <= m["raw"].get("max_share", 0.025) + 1e-4
    groups = {}
    for t in m["raw"]["tasks"]:
        groups[t["group"]] = groups.get(t["group"], 0) + t["weight"]
        assert t["scripted"][0] >= 0.5 * t["scripted"][1]
    assert sum(v for g, v in groups.items() if g.startswith("cube")) >= 0.75
    tg = mix.targets(w, 1000)
    assert min(tg.values()) >= 1 and abs(sum(tg.values()) - 1000) <= len(tg)
    assert m["intro"] == "v4" and m["config"].camera == "train"


@needs_menagerie
@pytest.mark.parametrize("view", ["front", "side", "low", "rear"])
def test_div2_physics_scene_builds_with_edge_table_and_view(view):
    from soarm.sim import SoarmSim
    sim = SoarmSim("cube_v_in_bowl", 1_500_000, config=dataclasses.replace(DIV2, camera=view))
    app = sim.rand.appearance
    assert app.view == view and app.table is not None
    g = sim.model.geom("table")
    assert abs(sim.model.geom_size[g.id][0] - app.table["depth"] / 2) < 1e-9
    assert abs(sim.data.geom_xpos[g.id][0] - (app.table["back"] + app.table["depth"] / 2)) < 1e-9
    assert not any(sim.atom_values())
