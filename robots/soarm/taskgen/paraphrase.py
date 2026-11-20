"""Paraphrase templates per family + a resolver test.

1. The LLM writes N paraphrase templates of a family's instructions using the same slots ({role}, {role.color}, {count},
   {%var}); a template is kept only if it mentions exactly the roles of one of the family's own templates.
2. Rendering: on sampled layouts every {role} slot is filled with a referring expression of a randomly chosen kind
   (attribute / function / ordinal / relation / only / on, taskgen.refexp) that denotes exactly the intended objects.
3. Resolver test: a separate LLM call sees only the scene (object ids, kinds, colours, sizes, positions) and the
   rendered instructions and writes the goal as predicate atoms over object ids; a paraphrase passes when the atoms match
   the sampled task's goal on >= `need` of its renders. The family's own templates are scored the same way (baseline).
"""
from __future__ import annotations

import json
import re

import numpy as np

from . import refexp
from .llm import LLM, extract_json
from .propose import predicate_doc

SLOT = re.compile(r"\{(%?[a-z][a-z0-9_]*)(?:\.([a-z_]+))?\}")
KIND_WEIGHTS = {"attr": 1.0, "function": 1.0, "only": 0.7, "on": 1.0, "ordinal": 1.2, "relation": 1.2}
NUM = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def role_set(template: str) -> frozenset:
    return frozenset(r for r, _ in SLOT.findall(template) if not r.startswith("%") and r != "count")


def slot_set(template: str) -> frozenset:
    return frozenset(f"{r}.{a}" if a else r for r, a in SLOT.findall(template))


def generate(llm: LLM, spec: dict, n: int = 36) -> list[str]:
    roles = [{k: o[k] for k in ("role", "kind", "count", "where") if k in o} for o in spec["objects"]]
    user = f"""A robot task family (JSON goal over object roles) and its instruction templates:
roles: {json.dumps(roles)}
goal: {json.dumps(spec["goal"])}
vars: {json.dumps(spec.get("vars", {}))}
templates:
{chr(10).join("- " + t for t in spec["instructions"])}

Slots: {{role}} is replaced by a referring expression for that role's object(s) (e.g. "the red cube", "the leftmost cube",
"the cube next to the bowl"), so never add your own article, colour or noun around it; {{role.color}} is the colour word,
{{role.size_class}} e.g. "4 cm", {{count}} a number word, {{%var}} a phrase such as "to the left of".
Write {n} NEW paraphrase templates with exactly the same meaning. Each one must use the slots of one of the templates
above (same roles, same {{count}}/{{%var}} if used), be a complete instruction a person could follow, and keep every
constraint (order, exact counts, which object goes where). Vary the wording strongly: imperative, polite request, goal
description ("... should end up ..."), step-by-step, terse command, sentence order, synonyms (cube/block for cubes only).
Return ONLY a JSON array of strings."""
    text = llm.chat([{"role": "user", "content": user}], tag=f"paraphrase:{spec['id']}", temperature=1.0)
    try:
        out = extract_json(text)
    except ValueError:
        return []
    allowed = {role_set(t) for t in spec["instructions"]}
    slots = set().union(*(slot_set(t) for t in spec["instructions"]))
    keep = []
    for t in out if isinstance(out, list) else []:
        if isinstance(t, str) and role_set(t) in allowed and slot_set(t) <= slots and t not in keep \
                and t not in spec["instructions"]:
            keep.append(t.strip())
    return keep


# ------------------------------------------------------------------------------------------------------- rendering
def scene_of(task) -> refexp.Scene:
    return refexp.Scene(specs={o.name: o for o in task.objects},
                        xy={n: np.asarray(p.xy, dtype=float) for n, p in task.layout.items()},
                        support={n: p.on for n, p in task.layout.items()})


def _var_value(spec: dict, var: str, task) -> str:
    opts = spec["vars"][var]
    for _, args, _ in task.atoms:
        for a in args:
            if isinstance(a, str) and a in opts:
                return opts[a]
    raise KeyError(var)


def _count(spec: dict) -> int | None:
    return next((g.get("eq", g.get("min", g.get("max"))) for g in spec["goal"] if "count" in g), None)


def render(spec: dict, task, template: str, rng: np.random.Generator, weights=None) -> tuple[str, dict]:
    scene = scene_of(task)
    members = {r: [n for n in v.split(",") if n] for r, v in task.roles.items()}
    used = {}

    def fill(m):
        role, attr = m.group(1), m.group(2)
        if role.startswith("%"):
            return _var_value(spec, role[1:], task)
        if role == "count":
            c = _count(spec)
            return NUM.get(c, str(c))
        names = members.get(role, [])
        if not names:
            raise KeyError(role)
        if attr == "color":
            return " and ".join(dict.fromkeys(refexp.color_word(scene.specs[n].color) for n in names))
        if attr == "size_class":
            return " and ".join(dict.fromkeys(f"{scene.specs[n].attrs.get('size_class', 0):g} cm" for n in names))
        if attr:
            return " and ".join(dict.fromkeys(str(scene.specs[n].attrs.get(attr)) for n in names))
        if len(names) == 1:
            e = refexp.pick(names[0], scene, rng, weights=weights or KIND_WEIGHTS)
            used[role] = e["type"]
            return refexp.realize(e, scene)
        cands = refexp.group_candidates(names, scene)
        if cands:
            e = cands[int(rng.integers(len(cands)))]
            used[role] = "group"
            return refexp.realize(e, scene)
        used[role] = "list"
        return refexp.describe_list(names, scene, rng)

    text = SLOT.sub(fill, template)
    text = re.sub(r"\b([Aa]) (?=[aeiou])", lambda m: m.group(1) + "n ", text)
    return text[0].upper() + text[1:], used


# ------------------------------------------------------------------------------------------------------- resolver
def scene_listing(task) -> str:
    lines = []
    for o in task.objects:
        p = task.layout[o.name]
        size = f"{o.size[0] * 100:.1f} x {o.size[1] * 100:.1f} x {o.size[2] * 100:.1f} cm"
        on = f", starts on {p.on}" if p.on else ""
        lines.append(f"- {o.name}: {o.noun} ({o.category}), colour {o.color}, {size}, at x={p.xy[0]:.3f} m, "
                     f"y={p.xy[1]:.3f} m{on}")
    return "\n".join(lines)


def resolver_prompt(task, instructions: list[str]) -> str:
    return f"""Robot table scene (robot base at the origin, x = distance away from the robot, y to the RIGHT as seen by a
person standing in front of the robot and facing it; so "left" = smaller y, "right" = larger y, "front" = larger x, "behind" =
smaller x, "closest to the robot" = smallest sqrt(x^2+y^2)):
{scene_listing(task)}

Goal predicates:
{predicate_doc()}
plus the counting form {{"count": {{"pred": P, "args": ["$", dest]}}, "over": [ids], "eq"|"min"|"max": N}}: exactly / at
least / at most N of the listed objects satisfy P.

For EACH numbered instruction below independently, write the goal it asks for as a list of atoms over the object ids
above: ["pred", [args...], {{params}}] or a counting form. Use the most specific predicate (inside for containers,
in_region for mats, on_top / stacked for towers, relation with "left"/"right"/"front"/"behind", in_line with
{{"order": "left_to_right"}} only when an order is required, resting_on_table to put something back on the table).
Instructions:
{chr(10).join(f"{i + 1}. {t}" for i, t in enumerate(instructions))}

Return ONLY a JSON object {{"1": [atoms], "2": [atoms], ...}}."""


def _norm_atom(a) -> str | None:
    """Canonical string of an atom in task form (name, args, params) or resolver form."""
    if isinstance(a, dict) and "count" in a:
        c = a["count"]
        dest = [x for x in c.get("args", []) if x != "$"]
        q = {k: a[k] for k in ("eq", "min", "max") if k in a}
        return f"count:{c.get('pred')}:{dest}:{sorted(a.get('over', []))}:{sorted(q.items())}"
    if isinstance(a, (list, tuple)) and a and a[0] == "count" and isinstance(a[2], dict):   # task form
        p = a[2]
        dest = [x for x in p["args"] if x != "$"]
        q = {k: p[k] for k in ("eq", "min", "max") if k in p}
        return f"count:{p['pred']}:{dest}:{sorted(a[1])}:{sorted(q.items())}"
    if not isinstance(a, (list, tuple)) or len(a) < 2:
        return None
    name, args = a[0], list(a[1]) if isinstance(a[1], (list, tuple)) else [a[1]]
    params = dict(a[2]) if len(a) > 2 and isinstance(a[2], dict) else {}
    if name == "placed_at":
        name = "placed"
    if name in ("inside", "in_region"):
        name = "placed"
    if name in ("grouped",) or (name == "in_line" and not params.get("order")):
        args = sorted(args)
    params = {k: v for k, v in params.items() if k == "order"}
    return f"{name}:{args}:{sorted(params.items())}"


def goal_key(atoms) -> frozenset:
    return frozenset(x for x in (_norm_atom(a) for a in atoms) if x)


def resolve(llm: LLM, task, instructions: list[str], tag: str) -> list[frozenset | None]:
    text = llm.chat([{"role": "user", "content": resolver_prompt(task, instructions)}], tag=tag, temperature=0.0,
                    max_tokens=8000)
    try:
        out = extract_json(text)
    except ValueError:
        return [None] * len(instructions)
    res = []
    for i in range(len(instructions)):
        atoms = out.get(str(i + 1)) if isinstance(out, dict) else None
        res.append(goal_key(atoms) if isinstance(atoms, list) else None)
    return res
