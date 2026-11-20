"""GSO/YCB registry meshes as task objects: registry rows, curated nouns, the kind "registry" sampler (soarm.objects
mesh_rows / add_mesh) and the broad_mesh specs; with $SOARM_MENAGERIE and the mesh files under $SOARM_ASSETS, the
MuJoCo bodies and one scripted (privileged-pose) solve."""
import dataclasses
import os

import numpy as np
import pytest

from soarm import taskspec
from soarm.objects import KINDS, MESH_NOUNS, asset_root, describe, load_registry, registry_row, select_assets
from soarm.tasks import SceneConfig

MESHES = [r for r in load_registry() if r["shape"] == "mesh"]
MESH_IDS = sorted(s["id"] for s in taskspec.registry().values() if s["id"].startswith("mesh_"))
CONFIG = SceneConfig.from_json({"table_mount": "edge", "camera": "train", "layout": "mixed", "table_textures": "all",
                                "lights": "random", "clutter": [0, 2], "platforms": 0.2})
HELDOUT_CATEGORIES = {"box", "tool", "figure", "can", "ball", "electronics", "fruit", "cup", "stuffed toy"}
have_meshes = pytest.mark.skipif(
    not os.environ.get("SOARM_MENAGERIE") or not MESHES
    or not (asset_root() / MESHES[0]["visual"]["mesh"]).is_file(), reason="SOARM_MENAGERIE or mesh assets missing")


def test_mesh_rows_are_gate_passed_and_split_by_category():
    assert len(MESHES) == 48
    for r in MESHES:
        assert r["graspable"] and r["gate"]["grasp_ok"] >= 4, r["id"]
        assert r["split"] == ("heldout" if r["category"] in HELDOUT_CATEGORIES else "train"), r["id"]
        assert r["collision"] and all(p.startswith(f"objects/{r['id']}/") for p in r["collision"])
    train = select_assets(split="train")
    assert train and all(r["split"] == "train" for r in train)
    assert {r["category"] for r in train if r["shape"] == "mesh"}.isdisjoint(HELDOUT_CATEGORIES)


def test_curated_nouns_are_concrete_and_cover_only_gated_rows():
    vague = {"object", "toy", "thing", "item", "figure"} | {noun for _, noun in KINDS.values()}
    for asset, (noun, group) in MESH_NOUNS.items():
        row = registry_row(asset)
        assert row["shape"] == "mesh" and row["graspable"], asset
        assert noun not in vague and group, asset
    nouns_by_group = {}
    for noun, group in MESH_NOUNS.values():
        nouns_by_group.setdefault(noun, set()).add(group)
    assert all(len(g) == 1 for g in nouns_by_group.values())          # one noun never spans two groups


@pytest.mark.parametrize("sid", MESH_IDS)
def test_mesh_specs_draw_train_meshes_with_unique_nouns(sid):
    spec = taskspec.get(sid)
    seen = set()
    for seed in range(25):
        t = taskspec.sample(spec, 1_500_000 + seed, CONFIG)
        assert t.to_json() == taskspec.sample(spec, 1_500_000 + seed, CONFIG).to_json()
        meshes = [o for o in t.objects if o.kind == "registry"]
        assert meshes
        assert len({o.attrs["group"] for o in meshes}) == len(meshes)
        for o in meshes:
            row = registry_row(o.attrs["asset"])
            assert row["split"] == "train" and o.attrs["noun"] == MESH_NOUNS[row["id"]][0]
            assert row["scale_range"][0] - 1e-9 <= o.attrs["scale"] <= row["scale_range"][1] + 1e-9
            assert describe(o, t.objects) == f"the {o.noun}"
            seen.add(row["id"])
        target = t.objects[[o.name for o in t.objects].index(t.roles["target"])]
        assert target.kind == "registry"
        sel = spec["objects"][[o["role"] for o in spec["objects"]].index("target")]
        cap = (sel.get("where") or {}).get("footprint")
        if cap:
            assert np.hypot(*target.size[:2]) <= cap["max"] + 1e-9
    assert len(seen) >= 4, seen


def test_heldout_config_draws_heldout_meshes():
    held = dataclasses.replace(CONFIG, cube_split="heldout")
    for seed in range(10):
        t = taskspec.sample(taskspec.get("mesh_lift"), 1_500_000 + seed, held)
        assert all(registry_row(o.attrs["asset"])["split"] == "heldout" for o in t.objects if o.kind == "registry")


@have_meshes
def test_mesh_body_builds_and_settles():
    from soarm.sim import SoarmSim
    sim = SoarmSim("mesh_in_container_two_meshes", 1_500_000, config=CONFIG)
    for name, o in sim.specs.items():
        if o.kind != "registry":
            continue
        hulls = [g for g in range(sim.model.ngeom) if sim.model.geom(g).name.startswith(f"obj:{name}:")
                 and not sim.model.geom(g).name.endswith(":vis")]
        assert len(hulls) == len(registry_row(o.attrs["asset"])["collision"])
        assert abs(float(sim.model.body(name).mass[0]) - o.mass) < 1e-6
        assert sim.data.xpos[sim.body_of[name]][2] > 0.0
    assert not any(sim.atom_values())


@have_meshes
def test_scripted_solver_lifts_a_mesh():
    from taskgen.solve import run_instance
    from taskgen.verify import draw_layouts
    ok, _ = draw_layouts(taskspec.get("mesh_lift"), 3, 12, 1_600_000, CONFIG)
    solved = sum(run_instance(task, seed, CONFIG)["ok"] for seed, task, _ in ok)
    assert solved >= 2, solved
