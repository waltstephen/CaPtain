"""Success predicates on synthetic object states (no MuJoCo)."""
import numpy as np
import pytest

from soarm.objects import ObjSpec
from soarm.predicates import World, Pose, evaluate, in_line, inside, on_top, in_region, relation, at_initial_of, \
    lifted, lying, off_region, grouped

CUBE = ObjSpec("cube_0", "cube", "red", (0.03, 0.03, 0.03))
CUBE2 = ObjSpec("cube_1", "cube", "blue", (0.03, 0.03, 0.03))
CUBE3 = ObjSpec("cube_2", "cube", "green", (0.03, 0.03, 0.03))
BOWL = ObjSpec("bowl_0", "bowl", "white", (0.11, 0.11, 0.04))
PAD = ObjSpec("pad_0", "pad", "yellow", (0.09, 0.09, 0.002))
CYL = ObjSpec("cylinder_0", "cylinder", "red", (0.03, 0.03, 0.05))


def pose(x, y, z, rot=None):
    return Pose(np.array([x, y, z], float), np.eye(3) if rot is None else rot)


def world(poses, specs, initial=None, held=()):
    return World(specs={s.name: s for s in specs}, poses=poses, initial=initial or {}, held=frozenset(held))


def test_on_top_requires_contact_height_and_overlap():
    specs = [CUBE, CUBE2]
    stacked = world({"cube_0": pose(0.2, 0, 0.045), "cube_1": pose(0.2, 0.005, 0.015)}, specs)
    assert on_top(stacked, "cube_0", "cube_1")
    hovering = world({"cube_0": pose(0.2, 0, 0.075), "cube_1": pose(0.2, 0, 0.015)}, specs)
    assert not on_top(hovering, "cube_0", "cube_1")
    beside = world({"cube_0": pose(0.2, 0.02, 0.045), "cube_1": pose(0.2, 0, 0.015)}, specs)
    assert not on_top(beside, "cube_0", "cube_1")
    held = world({"cube_0": pose(0.2, 0, 0.045), "cube_1": pose(0.2, 0, 0.015)}, specs, held=["cube_0"])
    assert not on_top(held, "cube_0", "cube_1")


def test_inside_container_vs_resting_on_rim_or_outside():
    specs = [CUBE, BOWL]
    assert inside(world({"cube_0": pose(0.2, 0.02, 0.019), "bowl_0": pose(0.2, 0, 0.02)}, specs), "cube_0", "bowl_0")
    # balanced on the rim: centre is outside the inner radius
    assert not inside(world({"cube_0": pose(0.2, 0.054, 0.055), "bowl_0": pose(0.2, 0, 0.02)}, specs), "cube_0", "bowl_0")
    # next to the bowl on the table
    assert not inside(world({"cube_0": pose(0.2, 0.08, 0.015), "bowl_0": pose(0.2, 0, 0.02)}, specs), "cube_0", "bowl_0")
    # held above the bowl does not count
    w = world({"cube_0": pose(0.2, 0, 0.03), "bowl_0": pose(0.2, 0, 0.02)}, specs, held=["cube_0"])
    assert not inside(w, "cube_0", "bowl_0")


def test_in_region_and_off_region():
    specs = [CUBE, PAD]
    on = world({"cube_0": pose(0.2, 0.03, 0.017), "pad_0": pose(0.2, 0, 0.001)}, specs)
    assert in_region(on, "cube_0", "pad_0") and not off_region(on, "cube_0", "pad_0")
    straddling = world({"cube_0": pose(0.2, 0.05, 0.017), "pad_0": pose(0.2, 0, 0.001)}, specs)
    assert not in_region(straddling, "cube_0", "pad_0") and not off_region(straddling, "cube_0", "pad_0")
    away = world({"cube_0": pose(0.2, 0.12, 0.015), "pad_0": pose(0.2, 0, 0.001)}, specs)
    assert off_region(away, "cube_0", "pad_0")
    fallen = world({"cube_0": pose(0.2, 0.9, -0.5), "pad_0": pose(0.2, 0, 0.001)}, specs)
    assert not off_region(fallen, "cube_0", "pad_0")


def test_relation_uses_camera_left_right_and_distance_band():
    specs = [CUBE, CUBE2]
    ref = pose(0.2, 0.0, 0.015)
    # LIBERO convention: right = +y, left = -y, front = +x
    assert relation(world({"cube_0": pose(0.2, 0.07, 0.015), "cube_1": ref}, specs), "cube_0", "cube_1", "right")
    assert relation(world({"cube_0": pose(0.2, -0.07, 0.015), "cube_1": ref}, specs), "cube_0", "cube_1", "left")
    assert relation(world({"cube_0": pose(0.27, 0.0, 0.015), "cube_1": ref}, specs), "cube_0", "cube_1", "front")
    assert not relation(world({"cube_0": pose(0.2, 0.07, 0.015), "cube_1": ref}, specs), "cube_0", "cube_1", "left")
    too_far = world({"cube_0": pose(0.2, 0.30, 0.015), "cube_1": ref}, specs)
    assert not relation(too_far, "cube_0", "cube_1", "right")
    diagonal = world({"cube_0": pose(0.28, 0.08, 0.015), "cube_1": ref}, specs)
    assert not relation(diagonal, "cube_0", "cube_1", "right")


def test_swap_lift_lying_group():
    specs = [CUBE, CUBE2]
    init = {"cube_0": pose(0.2, 0.1, 0.015), "cube_1": pose(0.2, -0.1, 0.015)}
    swapped = world({"cube_0": pose(0.21, -0.09, 0.015), "cube_1": pose(0.2, 0.1, 0.015)}, specs, initial=init)
    assert at_initial_of(swapped, "cube_0", "cube_1") and at_initial_of(swapped, "cube_1", "cube_0")
    assert not at_initial_of(world(init, specs, initial=init), "cube_0", "cube_1")
    assert lifted(world({"cube_0": pose(0.2, 0, 0.08), "cube_1": init["cube_1"]}, specs, held=["cube_0"]), "cube_0")
    assert not lifted(world({"cube_0": pose(0.2, 0, 0.08), "cube_1": init["cube_1"]}, specs), "cube_0")
    side = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)      # cylinder axis horizontal
    assert lying(world({"cylinder_0": pose(0.2, 0, 0.015, side)}, [CYL]), "cylinder_0")
    assert not lying(world({"cylinder_0": pose(0.2, 0, 0.025)}, [CYL]), "cylinder_0")
    near = world({"cube_0": pose(0.2, 0.0, 0.015), "cube_1": pose(0.2, 0.05, 0.015)}, specs)
    assert grouped(near, "cube_0", "cube_1")
    assert not grouped(world(init, specs), "cube_0", "cube_1")


@pytest.mark.parametrize("ys,order,expected", [
    ((-0.08, 0.0, 0.08), "", True),
    ((-0.08, 0.0, 0.08), "left_to_right", True),
    ((0.08, 0.0, -0.08), "left_to_right", False),
])
def test_in_line_order(ys, order, expected):
    specs = [CUBE, CUBE2, CUBE3]
    w = world({s.name: pose(0.2, y, 0.015) for s, y in zip(specs, ys)}, specs)
    assert in_line(w, "cube_0", "cube_1", "cube_2", order=order) is expected


def test_in_line_rejects_bent_rows_and_wide_gaps():
    specs = [CUBE, CUBE2, CUBE3]
    bent = world({"cube_0": pose(0.2, -0.08, 0.015), "cube_1": pose(0.24, 0.0, 0.015), "cube_2": pose(0.2, 0.08, 0.015)}, specs)
    assert not in_line(bent, "cube_0", "cube_1", "cube_2")
    spread = world({"cube_0": pose(0.2, -0.25, 0.015), "cube_1": pose(0.2, 0.0, 0.015), "cube_2": pose(0.2, 0.25, 0.015)}, specs)
    assert not in_line(spread, "cube_0", "cube_1", "cube_2")


def test_evaluate_conjunction_per_atom():
    specs = [CUBE, CUBE2, BOWL]
    w = world({"cube_0": pose(0.2, 0.0, 0.019), "bowl_0": pose(0.2, 0, 0.02), "cube_1": pose(0.3, 0.1, 0.015)}, specs)
    atoms = [("inside", ("cube_0", "bowl_0"), {}), ("inside", ("cube_1", "bowl_0"), {})]
    assert evaluate(w, atoms) == [True, False]
