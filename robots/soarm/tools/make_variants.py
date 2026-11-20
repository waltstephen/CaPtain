"""div2 parameterised task variants: cube domain -> soarm/specs/cube_var (tids 200+), non-cube primitives ->
soarm/specs/broad (domain broad, split train, tids 10000+); list order = tid.

    python tools/make_variants.py            # (re)write the files; --check: fail if a file would change

APPEND-ONLY: tids are list positions, and data on disk is keyed by tid, so new variants go at the end and existing
entries are never reordered or removed (retire a variant by dropping it from the mix, not from this list).
Variants only use train-safe goal structure: no grouped / at_initial_of / count goals and no stacked starts with a
resting_on_table goal (held-out families gather, swap, count_place, unstack).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SPECS = Path(__file__).resolve().parent.parent / "soarm" / "specs"
SMALL, LARGE = {"size_class": {"max": 3.5}}, {"size_class": {"min": 4.5}}


def obj(role, kind="cube", count=1, **where):
    o = {"role": role, "kind": kind, "count": count}
    if where:
        o["where"] = where
    return o


def dis(lo=0, hi=2, **where):
    return obj("distractors", "cube", [lo, hi], **where)


def atom(pred, *args, **params):
    a = {"pred": pred, "args": list(args)}
    if params:
        a["params"] = params
    return a


def each(pred, role, *rest):
    return {"pred": pred, "for_each": role, "args": ["$", *rest]}


def spec(vid, difficulty, objects, goal, instructions, layout=(), notes="", **extra):
    return {"vid": vid, "difficulty": difficulty, "objects": list(objects), "layout": list(layout),
            "goal": list(goal), "instructions": list(instructions), "notes": notes, **extra}


V: list[dict] = []

# ---- pick and lift (d1)
LIFT = ["Pick up {target} and hold it above the table.", "Lift {target} up off the table.",
        "Grab {target} and raise it a few centimetres.", "Take {target} in the gripper and hold it up."]
V += [
    spec("lift_dense", 1, [obj("target"), dis(3, 4, size_class={"max": 4.0})], [atom("lifted", "@target")], LIFT,
         notes="pick_lift among 3-4 distractor cubes"),
    spec("lift_largest", 1, [dis(1, 3), obj("target", size_class={"min_role": "distractors", "not_role": "distractors"})],
         [atom("lifted", "@target")],
         ["Pick up the largest cube and hold it in the air.", "Lift the biggest cube off the table.",
          "Find the largest cube and lift it up.", "Grasp the biggest of the cubes and raise it."]),
    spec("lift_smallest", 1, [dis(1, 3), obj("target", size_class={"max_role": "distractors", "not_role": "distractors"})],
         [atom("lifted", "@target")],
         ["Pick up the smallest cube and hold it in the air.", "Lift the smallest cube off the table.",
          "Find the smallest cube and lift it up.", "Grasp the tiniest of the cubes and raise it."]),
    spec("lift_wood", 1, [obj("target", texture="wood"), dis(1, 3, texture="plain")], [atom("lifted", "@target")],
         ["Pick up the wooden cube and hold it up.", "Lift the wooden cube off the table.",
          "Grab the cube made of wood and raise it.", "Take the wooden cube in the gripper and hold it above the table."]),
    spec("lift_small", 1, [obj("target", **SMALL), dis(0, 2)], [atom("lifted", "@target")], LIFT,
         notes="3.0-3.5 cm target"),
    spec("lift_large", 1, [obj("target", **LARGE), dis(0, 2)], [atom("lifted", "@target")], LIFT,
         notes="4.5-5.0 cm target (off-centre grasp)"),
]

# ---- put into a container / onto a mat (d1-d2)
PUT_IN = ["Put {target} in {destination}.", "Place {target} into {destination}.", "Drop {target} into {destination}.",
          "Move {target} into {destination}."]
PUT_ON = ["Put {target} on {destination}.", "Place {target} onto {destination}.", "Move {target} onto {destination}.",
          "Set {target} down on {destination}."]
V += [
    spec("in_bowl", 1, [obj("destination", "bowl"), obj("target"), dis(0, 3)], [atom("inside", "@target", "@destination")],
         PUT_IN),
    spec("in_box", 1, [obj("destination", "box"), obj("target"), dis(0, 3)], [atom("inside", "@target", "@destination")],
         PUT_IN),
    spec("in_container_largest", 1, [obj("destination", ["bowl", "box"]), dis(1, 2),
                                    obj("target", size_class={"min_role": "distractors", "not_role": "distractors"})],
         [atom("inside", "@target", "@destination")],
         ["Put the largest cube in {destination}.", "Place the biggest cube into {destination}.",
          "Find the largest cube and drop it into {destination}.", "Move the biggest cube into {destination}."]),
    spec("in_container_smallest", 1, [obj("destination", ["bowl", "box"]), dis(1, 2),
                                     obj("target", size_class={"max_role": "distractors", "not_role": "distractors"})],
         [atom("inside", "@target", "@destination")],
         ["Put the smallest cube in {destination}.", "Place the smallest cube into {destination}.",
          "Find the smallest cube and drop it into {destination}.", "Move the tiniest cube into {destination}."]),
    spec("in_container_wood", 1, [obj("destination", ["bowl", "box"]), obj("target", texture="wood"),
                                 dis(1, 2, texture="plain")], [atom("inside", "@target", "@destination")],
         ["Put the wooden cube in {destination}.", "Place the wooden cube into {destination}.",
          "Drop the cube made of wood into {destination}.", "Move the wooden cube into {destination}."]),
    spec("in_container_small", 1, [obj("destination", ["bowl", "box"]), obj("target", **SMALL), dis(0, 2)],
         [atom("inside", "@target", "@destination")], PUT_IN),
    spec("in_container_large", 2, [obj("destination", ["bowl", "box"]), obj("target", **LARGE), dis(0, 2)],
         [atom("inside", "@target", "@destination")], PUT_IN),
    spec("on_mat", 1, [obj("destination", "pad"), obj("target"), dis(0, 3)], [atom("in_region", "@target", "@destination")],
         PUT_ON),
    spec("on_mat_of_two", 1, [obj("destination", "pad"), obj("other_mat", "pad", color={"not_role": "destination"}),
                               obj("target"), dis(0, 2)],
         [atom("in_region", "@target", "@destination")], PUT_ON, notes="two mats of different colours"),
    spec("on_mat_small", 1, [obj("destination", "pad"), obj("target", **SMALL), dis(0, 2)],
         [atom("in_region", "@target", "@destination")], PUT_ON),
    spec("on_mat_large", 1, [obj("destination", "pad"), obj("target", **LARGE), dis(0, 2)],
         [atom("in_region", "@target", "@destination")], PUT_ON),
    spec("all_on_mat", 2, [obj("destination", "pad"), obj("items", "cube", 2, color={"distinct": True}, **SMALL), dis(0, 1)],
         [each("in_region", "items", "@destination")],
         ["Put {items} on {destination}.", "Place both {items} onto {destination}.",
          "Move {items} onto {destination}.", "Set {items} down on {destination}."]),
    spec("all_color_in_container", 2, [obj("destination", ["bowl", "box"]),
                                      obj("items", "cube", 2, color={"same_role": "items[0]"}, size_class={"max": 3.5}),
                                      dis(1, 2, color={"not_role": "items"})],
         [each("inside", "items", "@destination")],
         ["Put all the {items.color} cubes in {destination}.", "Place every {items.color} cube into {destination}.",
          "Move both {items.color} cubes into {destination}.", "Collect the {items.color} cubes in {destination}."]),
]

# ---- stacking two cubes (d2)
STACK = ["Stack {target} on top of {destination}.", "Put {target} on {destination}.",
         "Place {target} on top of {destination}.", "Set {target} down on top of {destination}."]
SMALLER = {"max_role": "destination", "not_role": "destination"}
V += [
    spec("stack_dense", 2, [obj("destination"), obj("target", size_class={"max_role": "destination"}), dis(2, 3)],
         [atom("on_top", "@target", "@destination")], STACK),
    spec("stack_small_on_large", 2, [obj("destination", size_class={"min": 4.0}), obj("target", size_class=SMALLER)],
         [atom("on_top", "@target", "@destination")],
         ["Put the smaller cube on top of the bigger one.", "Stack the small cube on the large cube.",
          "Place the smaller of the two cubes on the larger one.", "Build a two-cube tower with the big cube at the bottom."]),
    spec("stack_on_wood", 2, [obj("destination", texture="wood"), obj("target", size_class={"max_role": "destination"},
                                                                       texture="plain"), dis(0, 2, texture="plain")],
         [atom("on_top", "@target", "@destination")],
         ["Stack {target} on the wooden cube.", "Put {target} on top of the wood cube.",
          "Place {target} onto the cube made of wood.", "Set {target} down on top of the wooden cube."]),
    spec("stack_same_color", 2, [obj("destination", size_class={"min": 4.5}, texture="plain"),
                                 obj("target", color={"same_role": "destination"}, size_class={"max": 3.0}), dis(0, 2)],
         [atom("on_top", "@target", "@destination")], STACK, notes="two cubes of one colour: larger/smaller references"),
    spec("stack_on_mat_cube", 2, [obj("mat", "pad"), obj("destination"),
                                  obj("target", size_class={"max_role": "destination"}), dis(0, 2)],
         [atom("on_top", "@target", "@destination")],
         ["Stack {target} on {destination}, the cube on {mat}.", "Put {target} on top of the cube that is on {mat}.",
          "Place {target} onto {destination}.", "Set {target} down on top of {destination}, which sits on {mat}."],
         layout=[{"type": "on", "role": "destination", "support": "@mat"}]),
    spec("stack_large", 2, [obj("destination", **LARGE), obj("target", size_class={"max_role": "destination"}), dis(0, 2)],
         [atom("on_top", "@target", "@destination")], STACK),
]

# ---- place relative to a reference (d2): one spec per direction and reference group
DIRS = {"left": "to the left of", "right": "to the right of", "front": "in front of", "behind": "behind"}
# front/behind of a landmark: keep the landmark near/far so the target spot stays inside the reach band
REF_BAND = {"front": [0.15, 0.19], "behind": [0.22, 0.26]}
for ref_name, ref_kind in (("cube", "cube"), ("landmark", ["bowl", "box", "pad"])):
    for d, text in DIRS.items():
        lay = [{"type": "region", "role": "reference", "r": REF_BAND[d], "angle_deg": [-45, 45]}] \
            if ref_name == "landmark" and d in REF_BAND else []
        V.append(spec(f"rel_{d}_{ref_name}", 2, [obj("reference", ref_kind), obj("target"), dis(0, 2)],
                      [atom("relation", "@target", "@reference", "%dir")],
                      ["Place {target} {%dir} {reference}.", "Put {target} on the table {%dir} {reference}.",
                       "Move {target} so that it ends up {%dir} {reference}.",
                       "Set {target} down {%dir} {reference}, close to it."],
                      layout=lay, vars={"dir": {d: text}}))

# ---- lines (d3)
V += [
    spec("line_same_color", 3, [obj("items", "cube", 3, color={"same_role": "items[0]"}, size_class={"max": 4.0}),
                                dis(0, 1, color={"not_role": "items"})],
         [atom("in_line", "@items*")],
         ["Line up the three {items.color} cubes in a straight row.", "Arrange the {items.color} cubes in a straight line.",
          "Put the three {items.color} cubes in one straight row, a few centimetres apart.",
          "Make a straight line out of the {items.color} cubes."]),
    spec("line_front_to_back", 3, [obj("items", "cube", 3, color={"distinct": True}, size_class={"max": 4.0}), dis(0, 1)],
         [atom("in_line", "@items*", order="front_to_back")],
         ["Arrange {items} in a straight row from front to back, in that order (front = farthest from the robot).",
          "Line up {items} in this order, starting farthest from the robot and ending closest to it.",
          "Make a straight row pointing away from the robot: {items}, from the far end to the near end."]),
    spec("line_small", 3, [obj("items", "cube", 3, **SMALL), dis(0, 1)], [atom("in_line", "@items*")],
         ["Line up {items} in a straight row.", "Arrange {items} in a straight line, a few centimetres apart.",
          "Put {items} in one straight row."]),
]

# ---- clear / transfer (d1-d2)
V += [
    spec("clear_mat_one", 1, [obj("mat", "pad"), obj("target"), dis(0, 2)], [atom("off_region", "@target", "@mat")],
         ["Take {target} off {mat} and put it on the table.", "Move {target} off {mat}.",
          "Clear {mat}: put {target} somewhere else on the table.", "Remove {target} from {mat}."],
         layout=[{"type": "on", "role": "target", "support": "@mat"}]),
    spec("empty_container", 2, [obj("container", ["bowl", "box"]), obj("items", "cube", [1, 2], size_class={"max": 3.5}),
                                dis(0, 1)],
         [each("off_region", "items", "@container")],
         ["Take everything out of {container} and put it on the table.", "Empty {container} onto the table.",
          "Remove the cubes from {container} and set them down on the table.", "Unload {container}: every cube goes on the table."],
         layout=[{"type": "on", "role": "items", "support": "@container"}]),
    spec("mat_to_mat", 1, [obj("source", "pad"), obj("destination", "pad", color={"not_role": "source"}), obj("target"),
                           dis(0, 1)],
         [atom("in_region", "@target", "@destination")],
         ["Move {target} from {source} to {destination}.", "Take {target} off {source} and put it on {destination}.",
          "Transfer {target} from {source} onto {destination}.", "Put {target}, which is on {source}, onto {destination}."],
         layout=[{"type": "on", "role": "target", "support": "@source"},
                 {"type": "region", "role": "distractors", "r": [0.15, 0.26], "angle_deg": [-60, 60]}],
         workspace={"r": [0.13, 0.29], "angle_deg": 70}),
    spec("container_to_mat", 2, [obj("container", ["bowl", "box"]), obj("destination", "pad"),
                                 obj("target", size_class={"max": 4.0}), dis(0, 1)],
         [atom("in_region", "@target", "@destination")],
         ["Take {target} out of {container} and put it on {destination}.", "Move {target} from {container} to {destination}.",
          "Transfer {target} from {container} onto {destination}.", "Put the cube that is in {container} on {destination}."],
         layout=[{"type": "on", "role": "target", "support": "@container"}]),
    spec("mat_to_container", 1, [obj("mat", "pad"), obj("destination", ["bowl", "box"]), obj("target"), dis(0, 1)],
         [atom("inside", "@target", "@destination")],
         ["Move {target} from {mat} into {destination}.", "Take {target} off {mat} and drop it into {destination}.",
          "Put the cube that is on {mat} into {destination}.", "Transfer {target} from {mat} to {destination}."],
         layout=[{"type": "on", "role": "target", "support": "@mat"}]),
]

# ---- sorting (d2-d3)
V += [
    spec("sort_one_each", 2, [obj("bins", "pad", 2, color={"distinct": True}),
                              obj("item0", color={"same_role": "bins[0]"}), obj("item1", color={"same_role": "bins[1]"})],
         [atom("in_region", "@item0", "@bins[0]"), atom("in_region", "@item1", "@bins[1]")],
         ["Put each cube on the mat of the same colour.", "Sort the two cubes onto the mats that match their colours.",
          "Put {item0} on the {item0.color} mat and {item1} on the {item1.color} mat.",
          "Match the cubes to the mats by colour."], workspace={"r": [0.13, 0.29], "angle_deg": 70},
         layout=[{"type": "region", "role": "item0", "r": [0.15, 0.26], "angle_deg": [-60, 60]},
                 {"type": "region", "role": "item1", "r": [0.15, 0.26], "angle_deg": [-60, 60]}]),
    spec("sort_by_size", 3, [obj("big_mat", "pad"), obj("small_mat", "pad", color={"not_role": "big_mat"}),
                             obj("big", size_class={"min": 4.5}), obj("small", size_class={"max": 3.0})],
         [atom("in_region", "@big", "@big_mat"), atom("in_region", "@small", "@small_mat")],
         ["Put the large cube on {big_mat} and the small cube on {small_mat}.",
          "Sort by size: the big cube goes on {big_mat}, the small one on {small_mat}.",
          "Place the bigger cube onto {big_mat} and the smaller cube onto {small_mat}."],
         workspace={"r": [0.13, 0.29], "angle_deg": 70},
         layout=[{"type": "region", "role": "big", "r": [0.15, 0.26], "angle_deg": [-60, 60]},
                 {"type": "region", "role": "small", "r": [0.15, 0.26], "angle_deg": [-60, 60]}]),
]

# ---- two-step combinations (d2-d3)
V += [
    spec("in_and_on", 2, [obj("container", ["bowl", "box"]), obj("mat", "pad"), obj("first"), obj("second"), dis(0, 1)],
         [atom("inside", "@first", "@container"), atom("in_region", "@second", "@mat")],
         ["Put {first} in {container} and {second} on {mat}.", "Place {first} into {container}, then put {second} onto {mat}.",
          "First drop {first} into {container}; after that move {second} onto {mat}."]),
    spec("mat_then_stack", 3, [obj("mat", "pad"), obj("base"), obj("top", size_class={"max_role": "base"}), dis(0, 1)],
         [atom("in_region", "@base", "@mat"), atom("on_top", "@top", "@base")],
         ["Put {base} on {mat}, then stack {top} on top of it.", "Move {base} onto {mat} and place {top} on {base}.",
          "First set {base} down on {mat}; then put {top} on top of {base}."]),
    spec("place_then_lift", 2, [obj("container", ["bowl", "box"]), obj("first"), obj("second"), dis(0, 1)],
         [atom("inside", "@first", "@container"), atom("lifted", "@second")],
         ["Put {first} in {container}, then pick up {second} and hold it in the air.",
          "Drop {first} into {container}; afterwards lift {second} and keep holding it.",
          "First place {first} into {container}, then grab {second} and raise it."]),
    spec("container_and_relation", 2, [obj("container", ["bowl", "box"]), obj("first"), obj("second"), dis(0, 1)],
         [atom("inside", "@first", "@container"), atom("relation", "@second", "@container", "%dir")],
         ["Put {first} in {container} and place {second} {%dir} {container}.",
          "Drop {first} into {container}, then set {second} down {%dir} it.",
          "Place {first} into {container}; put {second} on the table {%dir} {container}."],
         vars={"dir": {"left": "to the left of", "right": "to the right of"}}),
    spec("stack_and_relation", 3, [obj("base"), obj("top", size_class={"max_role": "base"}), obj("third"), dis(0, 1)],
         [atom("on_top", "@top", "@base"), atom("relation", "@third", "@base", "%dir")],
         ["Stack {top} on {base}, then put {third} {%dir} {base}.",
          "Put {top} on top of {base} and place {third} on the table {%dir} the tower.",
          "Build a tower of {top} on {base}; set {third} down {%dir} it."],
         vars={"dir": {"left": "to the left of", "right": "to the right of"}}),
]

# ---- tight slot (d3)
V += [
    spec("slot_small", 3, [obj("target", **SMALL), obj("destination", "slot", size_class={"same_role": "target"}), dis(0, 2)],
         [atom("inside", "@target", "@destination")],
         ["Fit {target} into {destination}; the opening is only a few millimetres wider than the cube.",
          "Insert {target} into the tight {destination.color} holder.", "Put {target} into the snug square holder."],
         layout=[{"type": "region", "role": "destination", "r": [0.17, 0.25]}]),
]

# ================================================================ div3 cube variants (11-11): MoGe-2 depth tolerant
# Goals that tolerate 1-2 cm of position error (real-arm monocular depth): wide trays (14-18 cm), big bowls (13-15 cm)
# and big mats (12-15 cm) as targets, several cubes to one region, clearing, sorting into regions, two-step chains of
# easy steps. Cube sizes follow the scene config (div3 biases draws toward 4-5 cm).
BIG = {"size_class": {"min": 4.0}}


def spec3(*args, **kw):
    return spec(*args, gen="div3", **kw)


TWO_IN = ["Put {items} in {destination}.", "Place both {items} into {destination}.", "Move {items} into {destination}.",
          "Collect {items} in {destination}."]
TWO_ON = ["Put {items} on {destination}.", "Place both {items} onto {destination}.", "Move {items} onto {destination}.",
          "Set {items} down on {destination}."]
WIDE = {"r": [0.13, 0.29], "angle_deg": 70}     # big receptacles may use the wide annulus ...


def band(*roles):  # ... while the objects to grasp stay in the reliable band (regions, as cube_v_sort_one_each)
    return [{"type": "region", "role": r, "r": [0.15, 0.26], "angle_deg": [-60, 60]} for r in roles]


V += [
    spec3("in_tray", 1, [obj("destination", "tray"), obj("target"), dis(0, 3)], [atom("inside", "@target", "@destination")],
          PUT_IN, notes="14-18 cm shallow tray"),
    spec3("in_tray_large", 1, [obj("destination", "tray"), obj("target", **BIG), dis(0, 2)],
          [atom("inside", "@target", "@destination")], PUT_IN, notes="4-5 cm cube into a tray"),
    spec3("two_in_tray", 2, [obj("destination", "tray"), obj("items", "cube", 2, color={"distinct": True}), dis(0, 2)],
          [each("inside", "items", "@destination")], TWO_IN),
    spec3("three_in_tray", 2, [obj("destination", "tray"), obj("items", "cube", 3, color={"distinct": True},
                                                               size_class={"max": 4.0})],
          [each("inside", "items", "@destination")],
          ["Put {items} in {destination}.", "Place all three cubes into {destination}.",
           "Move {items} into {destination}.", "Collect every cube in {destination}."]),
    spec3("color_in_tray", 2, [obj("destination", "tray"), obj("items", "cube", 2, color={"same_role": "items[0]"},
                                                               size_class={"max": 4.5}),
                               dis(1, 1, color={"not_role": "items"})],
          [each("inside", "items", "@destination")],
          ["Put all the {items.color} cubes in {destination}.", "Place every {items.color} cube into {destination}.",
           "Move both {items.color} cubes into {destination}.", "Collect the {items.color} cubes in {destination}."]),
    spec3("in_big_bowl", 1, [obj("destination", "big_bowl"), obj("target"), dis(0, 3)],
          [atom("inside", "@target", "@destination")], PUT_IN, notes="13-15 cm bowl"),
    spec3("two_in_big_bowl", 2, [obj("destination", "big_bowl"), obj("items", "cube", 2, color={"distinct": True},
                                                                     size_class={"max": 4.5}), dis(0, 1)],
          [each("inside", "items", "@destination")], TWO_IN),
    spec3("in_bin_large", 1, [obj("destination", "bin"), obj("target", **BIG), dis(0, 2)],
          [atom("inside", "@target", "@destination")], PUT_IN, notes="low-walled box, 4-5 cm cube"),
    spec3("on_big_mat", 1, [obj("destination", "big_mat"), obj("target"), dis(0, 3)],
          [atom("in_region", "@target", "@destination")], PUT_ON, notes="12-15 cm mat"),
    spec3("two_on_big_mat", 2, [obj("destination", "big_mat"), obj("items", "cube", 2, color={"distinct": True},
                                                                   size_class={"max": 4.5}), dis(0, 2)],
          [each("in_region", "items", "@destination")], TWO_ON),
    spec3("three_on_big_mat", 2, [obj("destination", "big_mat"), obj("items", "cube", 3, color={"distinct": True},
                                                                     size_class={"max": 3.5})],
          [each("in_region", "items", "@destination")],
          ["Put {items} on {destination}.", "Place all three cubes onto {destination}.",
           "Move {items} onto {destination}.", "Gather every cube on {destination}."]),
    spec3("clear_big_mat", 2, [obj("mat", "big_mat"), obj("items", "cube", [1, 2], size_class={"max": 4.5}), dis(0, 1)],
          [each("off_region", "items", "@mat")],
          ["Clear {mat}: move every cube on it to the table.", "Take the cubes off {mat} and put them on the table.",
           "Remove everything from {mat}.", "Move the cubes that are on {mat} somewhere else on the table."],
          layout=[{"type": "on", "role": "items", "support": "@mat"}]),
    spec3("empty_tray", 2, [obj("container", "tray"), obj("items", "cube", [1, 2], size_class={"max": 4.5}), dis(0, 1)],
          [each("off_region", "items", "@container")],
          ["Take everything out of {container} and put it on the table.", "Empty {container} onto the table.",
           "Remove the cubes from {container} and set them down on the table.",
           "Unload {container}: every cube goes on the table."],
          layout=[{"type": "on", "role": "items", "support": "@container"}]),
    spec3("tray_to_mat", 2, [obj("container", "tray"), obj("destination", "big_mat"), obj("target"), dis(0, 1)],
          [atom("in_region", "@target", "@destination")],
          ["Take {target} out of {container} and put it on {destination}.", "Move {target} from {container} to {destination}.",
           "Transfer {target} from {container} onto {destination}.", "Put the cube that is in {container} on {destination}."],
          layout=[{"type": "on", "role": "target", "support": "@container"}, *band("distractors")], workspace=WIDE),
    spec3("mat_to_tray", 1, [obj("mat", "pad"), obj("destination", "tray"), obj("target"), dis(0, 1)],
          [atom("inside", "@target", "@destination")],
          ["Move {target} from {mat} into {destination}.", "Take {target} off {mat} and put it in {destination}.",
           "Put the cube that is on {mat} into {destination}.", "Transfer {target} from {mat} to {destination}."],
          layout=[{"type": "on", "role": "target", "support": "@mat"}, *band("distractors")], workspace=WIDE),
    spec3("sort_mat_and_tray", 2, [obj("destination", "tray"), obj("mat", "pad"), obj("first"),
                                   obj("second", color={"not_role": "first"}), dis(0, 1)],
          [atom("inside", "@first", "@destination"), atom("in_region", "@second", "@mat")],
          ["Put {first} in {destination} and {second} on {mat}.", "Place {first} into {destination}, then {second} onto {mat}.",
           "Sort the two cubes: {first} goes in {destination}, {second} goes on {mat}.",
           "Drop {first} into {destination}; set {second} down on {mat}."], layout=band("first", "second", "distractors"),
          workspace=WIDE),
    spec3("sort_colors_two_mats", 3, [obj("mats", "big_mat", 2, color={"distinct": True}),
                                      obj("group0", "cube", [1, 2], color={"same_role": "mats[0]"}, size_class={"max": 4.0}),
                                      obj("group1", "cube", 1, color={"same_role": "mats[1]"}, size_class={"max": 4.0})],
          [each("in_region", "group0", "@mats[0]"), each("in_region", "group1", "@mats[1]")],
          ["Put each cube on the mat of the same colour.", "Sort the cubes onto the mats that match their colours.",
           "Match every cube to the mat of its colour.", "Move the cubes so each one sits on the mat with its colour."],
          layout=band("group0", "group1"), workspace=WIDE),
    spec3("sort_size_tray_mat", 2, [obj("destination", "tray"), obj("mat", "big_mat"), obj("big", size_class={"min": 4.5}),
                                    obj("small", size_class={"max": 3.5})],
          [atom("inside", "@big", "@destination"), atom("in_region", "@small", "@mat")],
          ["Put the large cube in {destination} and the small cube on {mat}.",
           "Sort by size: the big cube goes into {destination}, the small one onto {mat}.",
           "Place the bigger cube in {destination} and the smaller cube on {mat}.",
           "The larger cube belongs in {destination}; the smaller one belongs on {mat}."], layout=band("big", "small"),
          workspace=WIDE),
    spec3("sort_one_each_big", 2, [obj("bins", "big_mat", 2, color={"distinct": True}),
                                   obj("item0", color={"same_role": "bins[0]"}), obj("item1", color={"same_role": "bins[1]"})],
          [atom("in_region", "@item0", "@bins[0]"), atom("in_region", "@item1", "@bins[1]")],
          ["Put each cube on the mat of the same colour.", "Sort the two cubes onto the mats that match their colours.",
           "Put {item0} on the {item0.color} mat and {item1} on the {item1.color} mat.",
           "Match the cubes to the mats by colour."], layout=band("item0", "item1"), workspace=WIDE),
    spec3("tray_then_lift", 2, [obj("container", "tray"), obj("first"), obj("second", **BIG), dis(0, 1)],
          [atom("inside", "@first", "@container"), atom("lifted", "@second")],
          ["Put {first} in {container}, then pick up {second} and hold it in the air.",
           "Drop {first} into {container}; afterwards lift {second} and keep holding it.",
           "First place {first} into {container}, then grab {second} and raise it.",
           "Store {first} in {container} and finish by holding {second} up."]),
    spec3("mat_then_tray", 2, [obj("container", "tray"), obj("mat", "pad"), obj("first"), obj("second"), dis(0, 1)],
          [atom("in_region", "@first", "@mat"), atom("inside", "@second", "@container")],
          ["Put {first} on {mat}, then put {second} in {container}.", "Move {first} onto {mat} and {second} into {container}.",
           "First set {first} down on {mat}; then drop {second} into {container}.",
           "Place {first} on {mat} and store {second} in {container}."], layout=band("first", "second", "distractors"),
          workspace=WIDE),
    spec3("clear_then_store", 2, [obj("container", "tray"), obj("mat", "pad"), obj("first"), obj("second"), dis(0, 1)],
          [atom("off_region", "@first", "@mat"), atom("inside", "@second", "@container")],
          ["Take {first} off {mat} and put {second} in {container}.",
           "Clear {mat} by moving {first} to the table, then drop {second} into {container}.",
           "Move {first} off {mat}; afterwards store {second} in {container}.",
           "First remove {first} from {mat}, then place {second} into {container}."],
          layout=[{"type": "on", "role": "first", "support": "@mat"}, *band("second", "distractors")], workspace=WIDE),
]

# ================================================================ non-cube primitives (domain broad, split train)
# cylinders (2.5-3.4 cm wide, 3.5-5.5 cm tall) and blocks (4.5-6 x 2.2-2.8 x 2.2-3 cm), with cups/bowls/boxes/mats.
PRIM = ["cylinder", "block"]


def odis(lo=0, hi=2, kinds=("cylinder", "block", "cube")):
    return obj("distractors", list(kinds), [lo, hi])


B: list[dict] = [
    spec("cylinder_lift", 1, [obj("target", "cylinder"), odis(0, 2, ("cube", "block"))], [atom("lifted", "@target")], LIFT),
    spec("block_lift", 1, [obj("target", "block"), odis(0, 2, ("cube", "cylinder"))], [atom("lifted", "@target")], LIFT),
    spec("cylinder_in_container", 1, [obj("destination", ["bowl", "box"]), obj("target", "cylinder"),
                                      odis(0, 2, ("cube", "block"))], [atom("inside", "@target", "@destination")], PUT_IN),
    spec("block_in_container", 1, [obj("destination", ["bowl", "box"]), obj("target", "block"),
                                   odis(0, 2, ("cube", "cylinder"))], [atom("inside", "@target", "@destination")], PUT_IN),
    spec("cylinder_on_mat", 1, [obj("destination", "pad"), obj("target", "cylinder"), odis(0, 2, ("cube", "block"))],
         [atom("in_region", "@target", "@destination")], PUT_ON),
    spec("block_on_mat", 1, [obj("destination", "pad"), obj("target", "block"), odis(0, 2, ("cube", "cylinder"))],
         [atom("in_region", "@target", "@destination")], PUT_ON),
    spec("shape_in_container", 1, [obj("destination", ["bowl", "box"]), obj("target", PRIM),
                                   obj("distractors", "cube", [1, 2])], [atom("inside", "@target", "@destination")],
         PUT_IN, notes="the one non-cube object among cubes"),
    spec("cube_on_block", 2, [obj("destination", "block"), obj("target", "cube", size_class={"max": 3.5}),
                              odis(0, 1, ("cylinder",))], [atom("on_top", "@target", "@destination")], STACK),
    spec("block_on_block", 2, [obj("destination", "block"), obj("target", "block", color={"not_role": "destination"}),
                               odis(0, 1, ("cube", "cylinder"))], [atom("on_top", "@target", "@destination")], STACK),
    spec("cylinder_relation", 2, [obj("reference", ["bowl", "box", "pad", "cube"]), obj("target", "cylinder"),
                                  odis(0, 1, ("block",))], [atom("relation", "@target", "@reference", "%dir")],
         ["Place {target} {%dir} {reference}.", "Put {target} on the table {%dir} {reference}.",
          "Move {target} so that it ends up {%dir} {reference}."], vars={"dir": DIRS}),
    spec("block_relation", 2, [obj("reference", ["bowl", "box", "pad", "cylinder"]), obj("target", "block"),
                               odis(0, 1, ("cube",))], [atom("relation", "@target", "@reference", "%dir")],
         ["Place {target} {%dir} {reference}.", "Put {target} on the table {%dir} {reference}.",
          "Move {target} so that it ends up {%dir} {reference}."], vars={"dir": DIRS}),
    spec("mixed_all_in_box", 2, [obj("destination", "box"), obj("a", "cylinder"), obj("b", "block"),
                                 obj("distractors", "cube", [0, 1])],
         [atom("inside", "@a", "@destination"), atom("inside", "@b", "@destination")],
         ["Put {a} and {b} in {destination}.", "Place both {a} and {b} into {destination}.",
          "Pack {a} and {b} into {destination}."]),
    spec("cube_in_cup", 2, [obj("destination", "cup"), obj("target", "cube", size_class={"max": 3.5}),
                            odis(0, 2, ("cylinder", "block"))], [atom("inside", "@target", "@destination")], PUT_IN),
    spec("clear_mat_shape", 1, [obj("mat", "pad"), obj("target", PRIM), odis(0, 2, ("cube",))],
         [atom("off_region", "@target", "@mat")],
         ["Take {target} off {mat} and put it on the table.", "Move {target} off {mat}.", "Remove {target} from {mat}."],
         layout=[{"type": "on", "role": "target", "support": "@mat"}]),
    spec("shapes_to_mats", 3, [obj("cyl_mat", "pad"), obj("block_mat", "pad", color={"not_role": "cyl_mat"}),
                               obj("cyl", "cylinder"), obj("blk", "block")],
         [atom("in_region", "@cyl", "@cyl_mat"), atom("in_region", "@blk", "@block_mat")],
         ["Put {cyl} on {cyl_mat} and {blk} on {block_mat}.", "Place {cyl} onto {cyl_mat}, then {blk} onto {block_mat}.",
          "Sort the two objects: {cyl} goes on {cyl_mat}, {blk} goes on {block_mat}."],
         workspace={"r": [0.13, 0.29], "angle_deg": 70},
         layout=[{"type": "region", "role": "cyl", "r": [0.15, 0.26], "angle_deg": [-60, 60]},
                 {"type": "region", "role": "blk", "r": [0.15, 0.26], "angle_deg": [-60, 60]}]),
    spec("line_mixed", 3, [obj("a", "cube", size_class={"max": 4.0}), obj("b", "cylinder"), obj("c", "block")],
         [atom("in_line", "@a", "@b", "@c")],
         ["Line up {a}, {b} and {c} in a straight row.", "Arrange {a}, {b} and {c} in a straight line.",
          "Put {a}, {b} and {c} in one straight row, a few centimetres apart."]),
    # ---- div3 (11-11): trays, big bowls/mats, bigger cylinders (3.6-4.2 x 5-7 cm) and blocks (6-7 x 3-3.5 cm)
    spec3("cylinder_in_tray", 1, [obj("destination", "tray"), obj("target", "cylinder"), odis(0, 2, ("cube", "block"))],
          [atom("inside", "@target", "@destination")], PUT_IN),
    spec3("block_in_tray", 1, [obj("destination", "tray"), obj("target", "block"), odis(0, 2, ("cube", "cylinder"))],
          [atom("inside", "@target", "@destination")], PUT_IN),
    spec3("shapes_in_tray", 2, [obj("destination", "tray"), obj("a", "cylinder"), obj("b", "block"),
                                obj("distractors", "cube", [0, 1])],
          [atom("inside", "@a", "@destination"), atom("inside", "@b", "@destination")],
          ["Put {a} and {b} in {destination}.", "Place both {a} and {b} into {destination}.",
           "Pack {a} and {b} into {destination}.", "Collect {a} and {b} in {destination}."]),
    spec3("big_cylinder_lift", 1, [obj("target", "big_cylinder"), odis(0, 2, ("cube", "block"))],
          [atom("lifted", "@target")], LIFT),
    spec3("big_block_lift", 1, [obj("target", "big_block"), odis(0, 2, ("cube", "cylinder"))],
          [atom("lifted", "@target")], LIFT),
    spec3("big_cylinder_in_big_bowl", 1, [obj("destination", "big_bowl"), obj("target", "big_cylinder"),
                                          odis(0, 2, ("cube", "block"))], [atom("inside", "@target", "@destination")], PUT_IN),
    spec3("block_in_big_bowl", 1, [obj("destination", "big_bowl"), obj("target", "block"), odis(0, 2, ("cube", "cylinder"))],
          [atom("inside", "@target", "@destination")], PUT_IN),
    spec3("big_block_on_big_mat", 1, [obj("destination", "big_mat"), obj("target", "big_block"),
                                      odis(0, 2, ("cube", "cylinder"))], [atom("in_region", "@target", "@destination")],
          PUT_ON),
    spec3("clear_mat_two_shapes", 2, [obj("mat", "big_mat"), obj("a", "cylinder"), obj("b", "block"), odis(0, 1, ("cube",))],
          [atom("off_region", "@a", "@mat"), atom("off_region", "@b", "@mat")],
          ["Clear {mat}: move {a} and {b} to the table.", "Take {a} and {b} off {mat}.",
           "Remove both objects from {mat}.", "Move {a} and {b} off {mat} onto the table."],
          layout=[{"type": "on", "role": "a", "support": "@mat"}, {"type": "on", "role": "b", "support": "@mat"}]),
    spec3("sort_shapes_tray_mat", 2, [obj("destination", "tray"), obj("mat", "big_mat"), obj("cyl", "cylinder"),
                                      obj("blk", "block")],
          [atom("inside", "@cyl", "@destination"), atom("in_region", "@blk", "@mat")],
          ["Put {cyl} in {destination} and {blk} on {mat}.", "Place {cyl} into {destination}, then {blk} onto {mat}.",
           "Sort the two objects: {cyl} goes in {destination}, {blk} goes on {mat}.",
           "The cylinder belongs in {destination} and the block on {mat}."], layout=band("cyl", "blk"), workspace=WIDE),
    spec3("cube_and_cylinder_on_big_mat", 2, [obj("destination", "big_mat"), obj("a", "cube", size_class={"max": 4.0}),
                                              obj("b", "cylinder"), odis(0, 1, ("block",))],
          [atom("in_region", "@a", "@destination"), atom("in_region", "@b", "@destination")],
          ["Put {a} and {b} on {destination}.", "Place both {a} and {b} onto {destination}.",
           "Move {a} and {b} onto {destination}.", "Set {a} and {b} down on {destination}."]),
]

# ================================================================ GSO/YCB registry meshes (domain broad, split train)
# kind "registry" = one gate-passed mesh (soarm.objects.mesh_rows: curated noun, one per noun group per scene); where
# "footprint" caps the diagonal of its horizontal bounding box so it fits a bowl/box/mat (starting on a mat: smaller).
FITS, FITS_ON_MAT = {"footprint": {"max": 0.09}}, {"footprint": {"max": 0.08}}
M: list[dict] = [
    spec("lift", 1, [obj("target", "registry"), odis(0, 2)], [atom("lifted", "@target")], LIFT),
    spec("in_container", 1, [obj("destination", ["bowl", "box"]), obj("target", "registry", **FITS),
                             odis(0, 2)], [atom("inside", "@target", "@destination")], PUT_IN),
    spec("on_mat", 1, [obj("destination", "pad"), obj("target", "registry", **FITS), odis(0, 2)],
         [atom("in_region", "@target", "@destination")], PUT_ON),
    spec("off_mat", 1, [obj("mat", "pad"), obj("target", "registry", **FITS_ON_MAT), odis(0, 2, ("cube", "cylinder"))],
         [atom("off_region", "@target", "@mat")],
         ["Take {target} off {mat} and put it on the table.", "Move {target} off {mat}.", "Remove {target} from {mat}."],
         layout=[{"type": "on", "role": "target", "support": "@mat"}]),
    spec("in_container_among_cubes", 2, [obj("destination", ["bowl", "box"]), obj("target", "registry", **FITS),
                                         obj("distractors", "cube", [1, 2])], [atom("inside", "@target", "@destination")],
         PUT_IN, notes="the one mesh object among cubes (named by its noun: scene clutter may add other non-cubes)"),
    spec("in_container_two_meshes", 2, [obj("destination", ["bowl", "box"]),
                                        obj("target", "registry", **FITS), obj("other", "registry"),
                                        odis(0, 1)], [atom("inside", "@target", "@destination")], PUT_IN,
         notes="a second mesh object of another noun group as distractor"),
]

# (domain, list, first tid, specs subdir, id prefix); never reorder
GROUPS = [("cube", V, 200, "cube_var", "cube_v_"), ("broad", B, 10000, "broad", "obj_"),
          ("broad", M, 11000, "broad_mesh", "mesh_")]


def build() -> dict[str, tuple[Path, dict]]:
    """spec id -> (file, spec) for every group."""
    out = {}
    for domain, items, tid0, sub, prefix in GROUPS:
        for k, v in enumerate(items):
            sid = prefix + v["vid"]
            d = {"id": sid, "tid": tid0 + k, "version": 1, "domain": domain, "split": "train", "provenance": "hand",
                 "difficulty": v["difficulty"], "builder": "declarative", "objects": v["objects"],
                 "layout": v["layout"], "goal": v["goal"], "instructions": v["instructions"]}
            for key in ("vars", "workspace"):
                if key in v:
                    d[key] = v[key]
            gen = v.get("gen") or ("div3" if prefix == "mesh_" else "div2")
            d["notes"] = (f"{gen} variant (tools/make_variants.py). " + v["notes"]).strip()
            if sid in out:
                raise SystemExit(f"duplicate variant id {sid}")
            out[sid] = (SPECS / sub / f"{v['vid']}.json", d)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    changed = []
    for path, d in build().values():
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(d, indent=1) + "\n"
        if not path.exists() or path.read_text() != text:
            changed.append(path.name)
            if not a.check:
                path.write_text(text)
    for domain, items, tid0, sub, _ in GROUPS:
        print(f"{domain}: {len(items)} variants in specs/{sub}, tids {tid0}-{tid0 + len(items) - 1}")
    print(f"changed: {changed or 'none'}")
    if a.check and changed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
