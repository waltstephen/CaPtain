"""LLM proposals of task-family specs (CONTRACT 2 JSON), conditioned on the predicate registry, the object kinds, the
measured reach limits and every family that already exists, asked for novelty along named axes.
Specs only: every field is interpreted by soarm.taskspec (no generated code, unlike GenSim/RoboGen).
"""
from __future__ import annotations

import inspect
import json

from soarm import taskspec
from soarm.predicates import PREDICATES

from .llm import LLM, extract_json

AXES = {
    "spatial": "spatial relations (left/right/front/behind of a reference, between two objects via two relation atoms, "
               "next to, in a row along a direction)",
    "counting": "quantities through roles (all the cubes of a colour, both / all three cubes of a group with for_each, "
                "every cube on a mat), never a count goal",
    "ordinal": "ordinals and orderings (bottom-to-top colour order, left-to-right order, size order, 'the second "
               "cube from the left' style references in the instructions)",
    "functional": "functional references and roles (the container, the base of a tower, the cube that is alone on a "
                  "mat, the mat that has a cube on it)",
    "multi_step": "multi-step combinations (2-3 sub-goals of different kinds in one task: e.g. empty a mat then "
                  "stack, sort then line up)",
    "conditional": "conditional structure expressed with optional objects: a role with count [0, 1] and goal atoms "
                   "that only make sense when it is present or absent",
    "tolerant": "goals that still succeed with 1-2 cm of position error (the robot measures objects with monocular "
                "depth): put cubes into a tray / big bowl / bin or onto a big mat, several cubes to one region, clear "
                "a mat or tray, sort into regions by colour or size, short chains of such easy steps; avoid stacking, "
                "straight lines, tight slots and relations to a small cube",
}

CUBE_RULES = """Cube domain (the training domain):
- objects: kind "cube" (size_class 3.0/3.5/4.0/4.5/5.0 cm, 12 colours red green blue yellow orange purple pink cyan
  white black brown gray), at most ONE container (kind "bowl" d 10-12 cm or "box" 10-12 x 8.5-10 cm, rim 3-4.5 cm, or
  "tray" 14-18 x 11-14 cm with a 2-3 cm rim, or "big_bowl" d 13-15 cm), and at most 3 flat mats (kind "pad",
  8.5-10 cm square, or "big_mat" 12-15 cm square; "bin" = low box for sorting counts as the container). 1-5 cubes in
  total including distractors.
- the arm grasps top-down only, its grip point reaches at most ~10 cm above the table: any tower of three cubes must use
  cubes with size_class max 3.5 and its base in layout region r [0.18, 0.24]; two cubes in one container need
  size_class max 4.0; containers that receive cubes go in layout region r [0.17, 0.24], angle_deg [-50, 50].
- a mat holds one 5 cm cube or two cubes up to 3.5 cm; prefer small cubes when several go on one mat.
- every goal atom must be FALSE in the start layout (the sampler rejects starts where one holds) and achievable with
  pick-and-place / push; nothing is hidden or articulated.
- reserved for held-out evaluation, never in a training family: goals with grouped, at_initial_of or a count
  combinator, and a stacked start whose goal puts the cubes back on the table (unstack).
- keep it moderate: at most 3 pick-and-place steps; a teacher model must succeed in most episodes."""

BROAD_RULES = """Broad domain (held-out evaluation only, never trained on): kinds cube, block (4.5-6 x 2.2-2.8 cm bar),
cylinder (d 2.5-3.4, h 3.5-5.5 cm, can lie down), ball (d 2.8-3.5, rolls), tile (6.5-8 cm flat square, too wide to grasp:
push only), bowl, box, cup (d 7-8 cm, 5.5-6.5 cm high: too high to drop into), pad. Up to 6 objects, one container.
Same reach limits (top-down, grip <= ~10 cm high)."""

SPEC_FORMAT = """Spec JSON (CONTRACT 2; any other key is an error):
{"id": "gen_<snake_case>", "tid": 0, "version": 1, "domain": "cube"|"broad", "split": "train", "provenance": "llm",
 "difficulty": 1-5, "builder": "declarative",
 "objects": [{"role": "<snake>", "kind": "cube"|"bowl"|"box"|"pad"|"bin"|..., "count": 1 | [lo, hi],
              "where": {"size_class": 4.0 | [3.0, 3.5] | {"min": x, "max": x, "max_role": "r", "min_role": "r"},
                        "color": "red" | ["red", "blue"] | {"not": [...], "same_role": "r" | "r[0]", "not_role": "r",
                                                          "distinct": true}}}],
 "layout": [{"type": "on", "role": "r", "support": "@mat"} | {"type": "stack", "roles": ["@a", "@b"]} |
            {"type": "distance", "a": "@x", "b": "@y", "min": 0.10, "max": 0.15} |
            {"type": "region", "role": "r", "r": [0.17, 0.24], "angle_deg": [-50, 50]}],
 "goal": [{"pred": "<predicate>", "args": ["@role" | "@role[i]" | "@role*" (all members, variadic preds only) | "$"
          (the member, with "for_each": "role") | "%var" | literal], "params": {...}}
          | {"count": {"pred": "...", "args": ["$", "@dest"]}, "over": "role", "eq": N | "min": N | "max": N}],
 "vars": {"dir": {"left": "to the left of", "right": "to the right of"}},   (optional: %dir in args, {%dir} in text)
 "instructions": [>= 6 templates; slots {role} = generated unique referring expression of the role's object(s),
                  {role.color}, {role.size_class}, {count} (the count's N), {%var}],
 "notes": "one sentence: what is new compared to the existing families"}
Roles are unique; a role with several members needs for_each / count / "@role*" / "@role[i]" in goal args.
Distractor roles are normal roles that no goal atom mentions; give them colours that cannot be confused with the
goal's colour constraints (where.color.not_role)."""


def predicate_doc() -> str:
    lines = []
    for name, fn in PREDICATES.items():
        sig = str(inspect.signature(fn)).replace("world: 'World', ", "").replace("world, ", "")
        doc = " ".join((inspect.getdoc(fn) or "").split())
        lines.append(f"- {name}{sig}: {doc}")
    return "\n".join(lines)


def existing_summary(extra: list[dict] = ()) -> str:
    rows = []
    for spec in list(taskspec.registry().values()) + list(extra):
        if spec.get("domain") == "broad" and spec.get("provenance") == "hand":
            continue
        preds = sorted({g.get("pred") or "count:" + g["count"]["pred"] for g in spec["goal"]})
        rows.append(f"- {spec['id']} ({spec['domain']}): {spec['instructions'][0]} | goal {','.join(preds)}")
    return "\n".join(rows)


def examples() -> str:
    out = []
    for sid in ("cube_sort_by_color", "cube_multi_step", "cube_v_stack_on_mat_cube"):
        s = dict(taskspec.get(sid))
        s.pop("notes", None)
        out.append(json.dumps(s))
    return "\n".join(out)


def prompt(domain: str, axes: list[str], n: int, extra: list[dict], rejected: list[str]) -> list[dict]:
    sys = ("You design task families for a small 5-DoF robot arm (SO-101) on a table, in simulation. A family is a JSON "
           "spec that a sampler turns into many concrete scenes with random colours, sizes and layouts; success is "
           "checked by geometric predicates. You only write JSON specs, never code.")
    user = f"""{SPEC_FORMAT}

Predicate registry (name(signature): meaning). Use only these:
{predicate_doc()}

{CUBE_RULES if domain == "cube" else BROAD_RULES}

Existing families (do NOT duplicate them or make trivial variants such as a renamed colour or one more distractor):
{existing_summary(extra)}
{"Recently rejected proposals and why (avoid these mistakes):" + chr(10) + chr(10).join(rejected[-12:]) if rejected else ""}

Three valid example specs:
{examples()}

Write {n} NEW {domain}-domain families. Focus on: {"; ".join(AXES[a] for a in axes)}.
Each family must be clearly different from every existing one in goal structure or in the reasoning the instruction
requires. Instructions: >= 6 templates, natural and varied (imperative, goal description, step-by-step), each one
complete and unambiguous on its own. Return ONLY a JSON array of {n} specs."""
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def propose(llm: LLM, domain: str, axes: list[str], n: int, extra: list[dict], rejected: list[str]) -> list[dict]:
    text = llm.chat(prompt(domain, axes, n, extra, rejected), tag=f"propose:{domain}:{'+'.join(axes)}")
    try:
        specs = extract_json(text)
    except ValueError:
        return []
    return [s for s in (specs if isinstance(specs, list) else [specs]) if isinstance(s, dict)]
