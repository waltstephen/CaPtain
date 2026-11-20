"""Task specs: legacy specs reproduce the pre-spec sampler and scenes (golden hashes from commit 0c89183), cube specs
sample valid cube-domain scenes, the validator rejects broken specs."""
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from soarm import taskspec
from soarm.episode import FAMILY_ORDER
from soarm.objects import registry_row
from soarm.taskspec import R_MAX, R_MIN
from soarm.tasks import SceneConfig

GOLDEN = json.loads((Path(__file__).parent / "data" / "legacy_golden.json").read_text())
CUBE = taskspec.select("cube")
needs_menagerie = pytest.mark.skipif(not os.environ.get("SOARM_MENAGERIE"), reason="SOARM_MENAGERIE not set")


def o_kind(t, name: str) -> str:
    return next(o.kind for o in t.objects if o.name == name)


def _h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def test_legacy_tids_are_the_family_order():
    assert [s["id"] for s in taskspec.select("legacy")] == list(FAMILY_ORDER)
    assert [s["tid"] for s in taskspec.select("legacy")] == list(range(len(FAMILY_ORDER)))


@pytest.mark.parametrize("family", FAMILY_ORDER)
def test_legacy_spec_sampling_matches_pre_spec_golden(family):
    spec = taskspec.get(family)
    for key, want in GOLDEN["tasks"].items():
        f, seed = key.split("/")
        if f == family:
            assert _h(json.dumps(taskspec.sample(spec, int(seed)).to_json(), sort_keys=True)) == want, key


@needs_menagerie
@pytest.mark.parametrize("family", FAMILY_ORDER)
def test_legacy_spec_scene_matches_pre_spec_golden(family):
    from soarm.scene import build_scene
    from soarm.sim import SoarmSim
    for seed in (0, 200, 9000):
        sim = SoarmSim(taskspec.get(family)["tid"], seed)
        want = GOLDEN["scenes"][f"{family}/{seed}"]
        assert _h(json.dumps(sim.task.to_json(), sort_keys=True)) == want["task"]
        assert _h(build_scene(sim.task, sim.rand).to_xml()) == want["xml"]


@pytest.mark.parametrize("spec", CUBE, ids=[s["id"] for s in CUBE])
def test_cube_specs_sample_deterministic_cube_domain_scenes(spec):
    for seed in range(12):
        t = taskspec.sample(spec, seed)
        assert t.to_json() == taskspec.sample(spec, seed).to_json()
        cubes = [o for o in t.objects if o.attrs.get("asset")]
        assert 1 <= len(cubes) <= 5 and all(o.kind != "cube" or o in cubes for o in t.objects), len(cubes)
        assert sum(o.category == "container" and not o.attrs.get("slot") for o in t.objects) <= 1
        for o in cubes:
            row = registry_row(o.attrs["asset"])
            assert row["split"] == "train" and row["graspable"]
            assert o.attrs["size_class"] in SceneConfig().cube_sizes and o.attrs["texture"] in ("plain", "wood")
            assert abs(o.size[0] / row["dims_m"][0] - 1) <= row["scale_jitter"] + 1e-9
            assert row["friction_range"][0] <= o.attrs["friction"] <= row["friction_range"][1]
        names = {o.name for o in t.objects}
        assert set(t.layout) == names
        lo, hi = (spec.get("workspace") or {}).get("r", (R_MIN, R_MAX))
        for name, p in t.layout.items():
            if p.on is None:
                assert min(lo, R_MIN) - 1e-9 <= float(np.hypot(*p.xy)) <= max(hi, R_MAX) + 1e-9, (name, p.xy)
            if p.on is None and o_kind(t, name) == "cube":
                assert R_MIN - 1e-9 <= float(np.hypot(*p.xy)) <= R_MAX + 1e-9, (name, p.xy)    # cubes: reliable band
        for _, args, _ in t.atoms:
            assert all(a in names for a in args if isinstance(a, str) and "_" in a)
        assert "{" not in t.language and "}" not in t.language, t.language


def test_heldout_appearance_config_changes_only_the_cube_distribution():
    spec = taskspec.get("cube_pick_lift")
    cfg = taskspec.suite("cube_heldout_appearance")["config"]
    seen = set()
    for seed in range(30):
        t = taskspec.sample(spec, seed, cfg)
        for o in t.objects:
            row = registry_row(o.attrs["asset"])
            assert row["split"] == "heldout" and row["graspable"] and o.attrs["size_class"] <= 5.0
            seen.add(o.attrs["texture"] if o.attrs["size_class"] != 2.5 else 2.5)
    assert {2.5, "letters", "printed"} <= seen


def test_count_goal_counts_members():
    from soarm.objects import ObjSpec
    from soarm.predicates import Pose, World
    from soarm.tasks import evaluate_goal
    bowl = ObjSpec("bowl_0", "bowl", "red", (0.11, 0.11, 0.04), 0.12)
    cs = [ObjSpec(f"cube_{i}", "cube", "blue", (0.04, 0.04, 0.04), 0.05) for i in range(3)]
    eye = np.eye(3)
    poses = {"bowl_0": Pose(np.array([0.2, 0.0, 0.02]), eye)}
    inside = {"cube_0": True, "cube_1": True, "cube_2": False}
    for c in cs:
        poses[c.name] = Pose(np.array([0.2, 0.0, 0.024]) if inside[c.name] else np.array([0.2, 0.15, 0.02]), eye)
    world = World(specs={o.name: o for o in [bowl, *cs]}, poses=poses)
    atom = ("count", tuple(c.name for c in cs), {"pred": "inside", "args": ["$", "bowl_0"], "params": {}, "eq": 2})
    assert evaluate_goal(world, [atom]) == [True]
    assert evaluate_goal(world, [(atom[0], atom[1], {**atom[2], "eq": 3})]) == [False]
    at_least_one = {k: v for k, v in atom[2].items() if k != "eq"} | {"min": 1}
    assert evaluate_goal(world, [(atom[0], atom[1], at_least_one)]) == [True]


@pytest.mark.parametrize("mutate, message", [
    (lambda s: s["goal"].append({"pred": "levitating", "args": ["@target"]}), "unknown predicate"),
    (lambda s: s["goal"].append({"pred": "lifted", "args": ["@nobody"]}), "bad role reference"),
    (lambda s: s["instructions"].append("Lift {ghost}."), "not a role"),
    (lambda s: s.update(tid=5), "outside"),
    (lambda s: s["objects"][0].update(kind="spaceship"), "schema"),
])
def test_validator_rejects_broken_specs(mutate, message):
    spec = copy.deepcopy(taskspec.get("cube_pick_lift"))
    mutate(spec)
    with pytest.raises(ValueError, match=message):
        taskspec.validate(spec)


@needs_menagerie
@pytest.mark.parametrize("spec", CUBE, ids=[s["id"] for s in CUBE])
def test_cube_physics_start_has_no_goal_satisfied(spec):
    from soarm.sim import SoarmSim
    for seed in range(2):
        sim = SoarmSim(spec, seed)
        assert not any(sim.atom_values()), (spec["id"], seed)
        assert not sim.held()


@pytest.mark.parametrize("name", taskspec.suites())
def test_eval_suites_sample_on_eval_seeds_only(name):
    s = taskspec.suite(name)
    lo, hi = taskspec.SEED_RANGES["armhost"]["eval"]
    assert s["jobs"] and all(lo <= seed <= hi for _, seed in s["jobs"])
    for spec_id, seed in s["jobs"]:
        taskspec.sample(taskspec.get(spec_id), seed, s["config"])


def test_suites_split_families_as_contracted():
    ids = {n: {j for j, _ in taskspec.suite(n)["jobs"]} for n in taskspec.suites()}
    hand = {s["id"] for s in taskspec.select("cube", "train", tids=(100, 199))}
    held = {s["id"] for s in taskspec.select("cube", "heldout")}
    assert ids["cube_seen"] == hand and ids["cube_heldout_families"] == held
    assert ids["cube_heldout_appearance"] == hand and len(hand) == 15
    assert len(held) == 4 and not hand & held
    assert not held & ids["div2_seen"] and ids["div2_seen"] == ids["div2_heldout_views"]
    assert all(taskspec.get(j)["split"] == "train" for j in ids["div2_seen"])
    assert not held & ids["div3_seen"] and len(ids["div3_seen"]) == 169
    assert not any(taskspec.get(j)["domain"] == "broad" and taskspec.get(j)["split"] == "train" for j in ids["broad"])


@needs_menagerie
def test_heldout_appearance_physics_starts():
    from soarm.sim import SoarmSim
    s = taskspec.suite("cube_heldout_appearance")
    for spec_id in {j for j, _ in s["jobs"]}:
        sim = SoarmSim(spec_id, 9000, config=s["config"])
        assert not any(sim.atom_values()), spec_id
