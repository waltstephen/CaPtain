"""taskgen invariants: referring expressions denote exactly one object (ambiguity boundaries, camera left/right), reach
limits in the planner, planned final states satisfy the goal, dedupe signatures ignore renaming but not goal changes."""
import numpy as np
import pytest

from soarm.objects import ObjSpec
from soarm.tasks import Placement, TaskInstance
from taskgen import dedupe, refexp
from taskgen.planner import NoPlan, plan, reachable


def cube(name, color, side=0.03):
    return ObjSpec(name=name, kind="cube", color=color, size=(side, side, side), attrs={"size_class": side * 100})


def scene(objs, xy, support=None):
    return refexp.Scene(specs={o.name: o for o in objs}, xy={k: np.asarray(v, float) for k, v in xy.items()},
                        support=support or {o.name: None for o in objs})


def test_ordinal_needs_a_clear_gap():
    objs = [cube("a", "red"), cube("b", "red"), cube("c", "blue")]
    clear = scene(objs, {"a": (0.2, -0.10), "b": (0.2, 0.0), "c": (0.2, 0.10)})
    e = {"type": "ordinal", "noun": "cube", "axis": "left", "k": 0}
    assert refexp.denote(e, clear) == {"a"}                          # left = smaller y (camera convention)
    assert refexp.denote({**e, "axis": "right"}, clear) == {"c"}
    close = scene(objs, {"a": (0.2, -0.10), "b": (0.25, -0.09), "c": (0.2, 0.10)})
    assert refexp.denote(e, close) == set()                          # 1 cm apart in y: not clearly ordered


def test_candidates_are_unique_and_group_excludes_extra_members():
    objs = [cube("a", "red"), cube("b", "red"), cube("c", "red"), cube("d", "blue")]
    s = scene(objs, {"a": (0.2, -0.12), "b": (0.2, 0.0), "c": (0.2, 0.12), "d": (0.15, 0.0)})
    for name in "abcd":
        for exprs in refexp.candidates(name, s).values():
            for e in exprs:
                assert refexp.denote(e, s) == {name}, (name, e)
    assert not refexp.candidates("a", s)["attr"]                     # "the red cube" is ambiguous
    assert refexp.group_candidates(["a", "b"], s) == []              # "the red cubes" would include c
    assert refexp.group_candidates(["a", "b", "c"], s)


def test_reach_band():
    assert reachable((0.20, 0.0), 0.02) and reachable((0.30, 0.0), 0.02)
    assert not reachable((0.33, 0.0), 0.02)
    assert reachable((0.20, 0.0), 0.10) and not reachable((0.26, 0.0), 0.10)
    assert not reachable((0.20, 0.0), 0.12)


def _stack_task(r_base, sides):
    objs = [cube(f"c{i}", c, s) for i, (c, s) in enumerate(zip(("red", "green", "blue"), sides))]
    layout = {"c0": Placement((r_base, 0.0), 0.0), "c1": Placement((0.18, 0.12), 0.0), "c2": Placement((0.18, -0.12), 0.0)}
    return TaskInstance(family="t", seed=0, language="", objects=objs, layout=layout,
                        atoms=[("stacked", ("c0", "c1", "c2"), {})])


def test_tower_height_limits_the_plan():
    with pytest.raises(NoPlan):
        plan(_stack_task(0.27, (0.04, 0.035, 0.03)))                 # 3rd cube released ~9.8 cm high at r 0.27
    steps = plan(_stack_task(0.20, (0.03, 0.03, 0.03)))
    assert [s.obj for s in steps] == ["c1", "c2"]


def test_count_plan_meets_exactly_n():
    bowl = ObjSpec(name="bowl_0", kind="bowl", color="white", size=(0.11, 0.11, 0.04))
    objs = [bowl] + [cube(f"c{i}", "red", 0.035) for i in range(3)]
    layout = {"bowl_0": Placement((0.20, 0.0), 0.0), "c0": Placement((0.16, 0.13), 0.0),
              "c1": Placement((0.24, 0.12), 0.0), "c2": Placement((0.18, -0.15), 0.0)}
    atoms = [("count", ("c0", "c1", "c2"), {"pred": "inside", "args": ["$", "bowl_0"], "params": {}, "eq": 2})]
    task = TaskInstance(family="t", seed=0, language="", objects=objs, layout=layout, atoms=atoms)
    assert len(plan(task)) == 2


def _spec(order=None, colour="red"):
    g = {"pred": "in_line", "args": ["@items*"]}
    if order:
        g["params"] = {"order": order}
    return {"id": "x", "objects": [{"role": "items", "kind": "cube", "count": 3, "where": {"color": {"distinct": True}}},
                                   {"role": "noise", "kind": "cube", "count": 1, "where": {"color": colour}}],
            "goal": [g], "instructions": ["Line up {items}."]}


def test_signature_ignores_renaming_not_goal_changes():
    a = _spec()
    b = _spec(colour="blue")
    b["objects"][0]["role"] = "things"
    b["goal"][0]["args"] = ["@things*"]
    assert dedupe.signature(a) == dedupe.signature(b)
    assert dedupe.signature(a) != dedupe.signature(_spec(order="left_to_right"))

