"""Object registry (assets/registry.jsonl), its loader and the camera-visibility rule of scene_random (no rendering)."""
import numpy as np
import pytest

from soarm.objects import COLORS, asset_spec, load_registry, select_assets
from soarm.scene import AGENTVIEW_FOVY, AGENTVIEW_POS, AGENTVIEW_TARGET, look_at_quat
from soarm.scene_random import sample_camera, sees_workspace

CUBES = [r for r in load_registry() if r["category"] == "cube"]
TRAIN_SIZES = {3.0, 3.5, 4.0, 4.5, 5.0}


def test_registry_rows_carry_contract_fields():
    need = {"id", "name", "category", "synonyms", "colours", "dims_m", "mass_kg", "visual", "collision", "joints",
            "graspable", "scripted_grasp_success", "split", "source", "licence",
            "size_class", "color", "texture", "density_range_kg_m3", "friction_range"}
    assert CUBES
    for r in CUBES:
        assert need <= set(r), r["id"]
        assert r["color"] in COLORS or r["color"] == "wood", r["id"]


def test_split_rule_holds_out_sizes_letters_and_prints():
    for r in CUBES:
        heldout = r["size_class"] not in TRAIN_SIZES or r["texture"] in ("letters", "printed")
        assert r["split"] == ("heldout" if heldout else "train"), r["id"]


def test_train_cubes_passed_the_physics_gate():
    train = [r for r in CUBES if r["split"] == "train"]
    assert train and all(r["graspable"] for r in train), [r["id"] for r in train if not r["graspable"]]


def test_asset_spec_stays_inside_the_row_ranges():
    rng = np.random.default_rng(0)
    for r in CUBES[::7]:
        for _ in range(20):
            o = asset_spec(rng, r, "c")
            side = r["dims_m"][0]
            assert abs(o.size[0] / side - 1) <= r["scale_jitter"] + 1e-9
            assert r["mass_range_kg"][0] - 1e-9 <= o.mass <= r["mass_range_kg"][1] + 1e-9
            assert r["friction_range"][0] <= o.attrs["friction"] <= r["friction_range"][1]
            assert o.kind == "cube" and o.attrs["asset"] == r["id"]


def test_select_assets_matches_scalars_lists_and_value_sets():
    rows = [{"id": "a", "graspable": True, "colours": ["red", "white"], "size_class": 3.0},
            {"id": "b", "graspable": True, "colours": ["blue"], "size_class": 5.0},
            {"id": "c", "graspable": False, "colours": ["red"], "size_class": 3.0}]
    assert [r["id"] for r in select_assets(rows, colours="red")] == ["a"]
    assert [r["id"] for r in select_assets(rows, colours="red", graspable=None)] == ["a", "c"]
    assert [r["id"] for r in select_assets(rows, size_class=(3.0, 5.0))] == ["a", "b"]
    assert select_assets(rows, size_class=4.0) == []


def test_default_agentview_sees_the_workspace_and_a_turned_camera_does_not():
    q = look_at_quat(AGENTVIEW_POS, AGENTVIEW_TARGET)
    assert sees_workspace(AGENTVIEW_POS, q, AGENTVIEW_FOVY)
    turned = look_at_quat(AGENTVIEW_POS, AGENTVIEW_TARGET + np.array([0.0, 0.25, 0.0]))
    assert not sees_workspace(AGENTVIEW_POS, turned, AGENTVIEW_FOVY)


@pytest.mark.parametrize("jitter,rot", [(0.05, 5.0), (0.10, 10.0)])
def test_jittered_cameras_keep_the_workspace_in_view(jitter, rot):
    rng = np.random.default_rng(1)
    for _ in range(20):
        pos, q, fovy, ok = sample_camera(rng, jitter, rot)
        assert sees_workspace(pos, q, fovy)
