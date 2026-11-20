"""Task families. Each generator gets a `tasks.Builder`, adds objects (targets first, then distractors), and returns
(language, atoms, roles, fixed_placements). References are built after all objects exist, so they are unique in the scene.
"""
from __future__ import annotations

import numpy as np

from .objects import GRASPABLE
from .tasks import ANG_MAX, R_MAX, R_MIN, Builder, Placement, Reject


def _pick(b: Builder, options):
    return options[int(b.rng.integers(len(options)))]


def _graspable(b: Builder, color: str | None = None, kinds=GRASPABLE, avoid_colors: tuple[str, ...] = ()):
    return b.add(str(b.rng.choice(kinds)), color, avoid_colors=avoid_colors)


def pick_lift(b: Builder):
    t = _graspable(b)
    b.distractors(int(b.rng.integers(1, 4)))
    x = b.ref(t)
    lang = _pick(b, [f"Pick up {x} and hold it above the table.", f"Lift {x} off the table.",
                     f"Grasp {x} and raise it a few centimetres."])
    return lang, [("lifted", (t.name,), {})], {"target": t.name}, {}


def pick_place_mat(b: Builder):
    mat = b.add("pad")
    t = _graspable(b, avoid_colors=(mat.color,))
    b.distractors(int(b.rng.integers(1, 4)))
    if b.rng.random() < 0.5:
        b.add("pad", avoid_colors=(mat.color,))          # a second mat as distractor
    x, m = b.ref(t), b.ref(mat)
    lang = _pick(b, [f"Put {x} on {m}.", f"Place {x} onto {m}.", f"Move {x} to {m}."])
    return lang, [("in_region", (t.name, mat.name), {})], {"target": t.name, "destination": mat.name}, {}


def put_in_container(b: Builder):
    # no cups: releasing into a 6.5 cm-high cup needs the grip point ~9 cm up, beyond top-down reach
    c = b.add(_pick(b, ["bowl", "box"]))
    t = _graspable(b)
    b.distractors(int(b.rng.integers(1, 4)))
    if b.rng.random() < 0.4:
        b.add(_pick(b, ["bowl", "box"]))
    x, d = b.ref(t), b.ref(c)
    lang = _pick(b, [f"Put {x} in {d}.", f"Place {x} into {d}.", f"Drop {x} into {d}."])
    return lang, [("inside", (t.name, c.name), {})], {"target": t.name, "destination": c.name}, {}


def stack_two(b: Builder):
    base = b.add(_pick(b, ["cube", "block", "cube"]))
    top = b.add(_pick(b, ["cube", "cube", "cylinder"]))
    if top.size[0] > base.size[0] + 0.005 and base.kind == "cube":
        raise Reject("top wider than base")
    b.distractors(int(b.rng.integers(1, 3)))
    x, y = b.ref(top), b.ref(base)
    lang = _pick(b, [f"Stack {x} on top of {y}.", f"Put {x} on {y}.", f"Place {x} on top of {y}."])
    return lang, [("on_top", (top.name, base.name), {})], {"target": top.name, "destination": base.name}, {}


def stack_three(b: Builder):
    cols = b.colors(3)
    cubes = sorted((b.add("cube", c) for c in cols), key=lambda o: -o.size[0])
    b.distractors(int(b.rng.integers(0, 2)), kinds=("cylinder", "ball", "block"))
    refs = [b.ref(c) for c in cubes]
    lang = _pick(b, [f"Build a tower: {refs[0]} at the bottom, then {refs[1]}, then {refs[2]} on top.",
                     f"Stack the three cubes with {refs[0]} at the bottom, {refs[1]} in the middle and {refs[2]} on top."])
    return lang, [("stacked", tuple(c.name for c in cubes), {})], {f"level_{i}": c.name for i, c in enumerate(cubes)}, {}


def sort_by_color(b: Builder):
    c1, c2 = b.colors(2)
    kind = _pick(b, ["pad", "bowl", "box"])
    d1, d2 = b.add(kind, c1), b.add(kind, c2)
    objs = [_graspable(b, c1), _graspable(b, c2)]
    if b.rng.random() < 0.6:
        objs.append(_graspable(b, _pick(b, [c1, c2])))
    b.distractors(int(b.rng.integers(0, 2)))
    if sum(o.color in (c1, c2) for o in b.objects if o.category == "object") != len(objs):
        raise Reject("distractor shares a sorting colour")
    noun = {"pad": "mat", "bowl": "bowl", "box": "box"}[kind]
    prep = "on" if kind == "pad" else "in"
    lang = _pick(b, [f"Sort the {c1} and {c2} objects: put each one {prep} the {noun} of the same colour.",
                     f"Put every {c1} object {prep} the {c1} {noun} and every {c2} object {prep} the {c2} {noun}."])
    atoms = [("placed_at", (o.name, d1.name if o.color == c1 else d2.name), {}) for o in objs]
    return lang, atoms, {"bins": f"{d1.name},{d2.name}", "items": ",".join(o.name for o in objs)}, {}


def sort_by_shape(b: Builder):
    m1, m2 = b.add("pad"), b.add("pad")
    if m1.color == m2.color:
        raise Reject("same mat colour")
    k1, k2 = [str(k) for k in b.rng.choice(["cube", "cylinder", "ball", "block"], size=2, replace=False)]
    objs = [b.add(k1), b.add(k2), b.add(_pick(b, [k1, k2]))]
    noun = {"cube": "cubes", "cylinder": "cylinders", "ball": "balls", "block": "blocks"}
    x1, x2 = b.ref(m1), b.ref(m2)
    lang = _pick(b, [f"Put all the {noun[k1]} on {x1} and all the {noun[k2]} on {x2}.",
                     f"Sort by shape: {noun[k1]} go on {x1}, {noun[k2]} go on {x2}."])
    atoms = [("in_region", (o.name, m1.name if o.kind == k1 else m2.name), {}) for o in objs]
    return lang, atoms, {"mats": f"{m1.name},{m2.name}", "items": ",".join(o.name for o in objs)}, {}


def line_up(b: Builder):
    cols = b.colors(3)
    kind = _pick(b, ["cube", "cylinder", "cube"])
    objs = [b.add(kind, c) for c in cols]
    b.distractors(int(b.rng.integers(0, 2)), kinds=("ball", "block"))
    refs = [b.ref(o) for o in objs]
    if b.rng.random() < 0.5:
        lang = _pick(b, [f"Line up {refs[0]}, {refs[1]} and {refs[2]} in a straight row.",
                         f"Arrange {refs[0]}, {refs[1]} and {refs[2]} in a straight line, a few centimetres apart."])
        atom = ("in_line", tuple(o.name for o in objs), {})
    else:
        lang = (f"Arrange {refs[0]}, {refs[1]} and {refs[2]} in a straight row from left to right, in that order "
                f"(left and right as seen in the front camera image).")
        atom = ("in_line", tuple(o.name for o in objs), {"order": "left_to_right"})
    return lang, [atom], {"items": ",".join(o.name for o in objs)}, {}


def push_to_region(b: Builder):
    mat = b.add("pad")
    tile = b.add("tile", avoid_colors=(mat.color,))
    b.distractors(int(b.rng.integers(0, 3)))
    x, m = b.ref(tile), b.ref(mat)
    lang = _pick(b, [f"Push {x} onto {m}.", f"Slide {x} along the table until it is on {m}.",
                     f"Move {x} onto {m} (it is wide and flat; pushing works)."])
    # short pushes (12-15 cm between centres): long straight pushes with jaw tips spin the tile off the line
    for _ in range(50):
        r, a = b.rng.uniform(R_MIN + 0.02, R_MAX - 0.02), b.rng.uniform(-ANG_MAX + 0.2, ANG_MAX - 0.2)
        mxy = np.array([r * np.cos(a), r * np.sin(a)])
        phi = b.rng.uniform(-np.pi, np.pi)
        txy = mxy + b.rng.uniform(0.12, 0.15) * np.array([np.cos(phi), np.sin(phi)])
        tr, ta = float(np.hypot(*txy)), float(np.arctan2(txy[1], txy[0]))
        if R_MIN <= tr <= R_MAX and abs(ta) <= ANG_MAX:
            break
    else:
        raise Reject("no short push layout")
    fixed = {mat.name: Placement(xy=(float(mxy[0]), float(mxy[1])), yaw=float(b.rng.uniform(-np.pi, np.pi))),
             tile.name: Placement(xy=(float(txy[0]), float(txy[1])), yaw=float(b.rng.uniform(-np.pi, np.pi)))}
    return lang, [("in_region", (tile.name, mat.name), {})], {"target": tile.name, "destination": mat.name}, fixed


def swap(b: Builder):
    o1, o2 = _graspable(b), _graspable(b)
    b.distractors(int(b.rng.integers(0, 2)))
    x, y = b.ref(o1), b.ref(o2)
    lang = _pick(b, [f"Swap the positions of {x} and {y}.", f"Exchange the places of {x} and {y}."])
    atoms = [("at_initial_of", (o1.name, o2.name), {}), ("at_initial_of", (o2.name, o1.name), {})]
    return lang, atoms, {"a": o1.name, "b": o2.name}, {}


def relational_place(b: Builder):
    ref = b.add(_pick(b, ["bowl", "box", "pad", "cube", "cylinder"]))
    t = _graspable(b)
    b.distractors(int(b.rng.integers(1, 3)))
    direction = _pick(b, ["left", "right", "front", "behind"])
    words = {"left": "to the left of", "right": "to the right of", "front": "in front of", "behind": "behind"}[direction]
    x, y = b.ref(t), b.ref(ref)
    lang = _pick(b, [f"Place {x} {words} {y}.", f"Put {x} on the table {words} {y}."])
    if direction in ("front", "behind"):
        lang += " (Front means toward the front camera, away from the robot.)"
    return lang, [("relation", (t.name, ref.name, direction), {})], {"target": t.name, "reference": ref.name}, {}


def clear_region(b: Builder):
    mat = b.add("pad")
    on = [_graspable(b, avoid_colors=(mat.color,)) for _ in range(int(b.rng.integers(1, 3)))]
    b.distractors(int(b.rng.integers(0, 2)))
    m = b.ref(mat)
    lang = _pick(b, [f"Clear {m}: move everything off it onto the table.", f"Remove all objects from {m}.",
                     f"Take every object off {m} and leave them on the table."])
    fixed = {o.name: Placement(xy=(0.0, 0.0), yaw=0.0, on=mat.name) for o in on}
    return lang, [("off_region", (o.name, mat.name), {}) for o in on], {"mat": mat.name}, fixed


def gather(b: Builder):
    color = b.colors(1)[0]
    objs = [_graspable(b, color) for _ in range(3)]
    b.distractors(int(b.rng.integers(1, 3)), kinds=("cube", "cylinder", "ball"))
    if sum(o.color == color for o in b.objects) != 3:
        raise Reject("distractor shares the colour")
    lang = _pick(b, [f"Gather all the {color} objects close together.",
                     f"Bring the three {color} objects together into one small group."])
    return lang, [("grouped", tuple(o.name for o in objs), {})], {"items": ",".join(o.name for o in objs)}, {}


def lay_down(b: Builder):
    t = b.add("cylinder")
    b.distractors(int(b.rng.integers(1, 3)), kinds=("cube", "ball", "block"))
    x = b.ref(t)
    lang = _pick(b, [f"Lay {x} on its side.", f"Tip {x} over so it lies on its side on the table.",
                     f"Put {x} down lying flat on its side."])
    return lang, [("lying", (t.name,), {})], {"target": t.name}, {}


def put_all_in_container(b: Builder):
    c = b.add(_pick(b, ["bowl", "box"]))
    kind = _pick(b, ["cube", "ball", "cylinder"])
    objs = [b.add(kind) for _ in range(2)]
    b.distractors(int(b.rng.integers(1, 3)), kinds=tuple(k for k in ("cube", "ball", "cylinder", "block") if k != kind))
    noun = {"cube": "cubes", "ball": "balls", "cylinder": "cylinders"}[kind]
    d = b.ref(c)
    lang = _pick(b, [f"Put all the {noun} in {d}.", f"Place both {noun} into {d}."])
    return lang, [("inside", (o.name, c.name), {}) for o in objs], {"items": ",".join(o.name for o in objs),
                                                                     "destination": c.name}, {}


def multi_step(b: Builder):
    c = b.add(_pick(b, ["bowl", "box"]))
    t1 = _graspable(b, kinds=("cube", "ball"))
    base = b.add("cube")
    top = b.add("cube")
    if top.size[0] > base.size[0] + 0.005:
        raise Reject("top wider than base")
    b.distractors(int(b.rng.integers(0, 2)), kinds=("cylinder", "block"))
    x, d, y, z = b.ref(t1), b.ref(c), b.ref(top), b.ref(base)
    lang = _pick(b, [f"First put {x} in {d}, then stack {y} on {z}.",
                     f"Put {x} into {d}. After that, place {y} on top of {z}."])
    atoms = [("inside", (t1.name, c.name), {}), ("on_top", (top.name, base.name), {})]
    return lang, atoms, {"step1": t1.name, "step2": top.name}, {}


FAMILIES = {
    "pick_lift": pick_lift, "pick_place_mat": pick_place_mat, "put_in_container": put_in_container,
    "stack_two": stack_two, "stack_three": stack_three, "sort_by_color": sort_by_color, "sort_by_shape": sort_by_shape,
    "line_up": line_up, "push_to_region": push_to_region, "swap": swap, "relational_place": relational_place,
    "clear_region": clear_region, "gather": gather, "lay_down": lay_down, "put_all_in_container": put_all_in_container,
    "multi_step": multi_step,
}
