"""Referring expressions with explicit semantics: every expression is a dict that `denote` evaluates on the start layout
to a set of object names; an expression is valid for an object only if it denotes exactly that object (groups: exactly
the member set). Kinds: attribute (colour/size/noun), function (category noun: "the container"), ordinal (left/right/
front/back/robot distance/size rank), relation (side of / nearest an anchor), exclusivity ("the only red object"),
support ("the cube on the mat").

Directions follow predicates.DIRECTIONS (LIBERO camera convention): right = +y, left = -y, front = +x (toward the camera).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from soarm.objects import ObjSpec

ORD_WORDS = ["", "second ", "third ", "fourth ", "fifth "]
NUM_WORDS = {2: "two", 3: "three", 4: "four", 5: "five"}
FUNCTION_NOUNS = {"container": ["the container", "the receptacle"], "region": ["the mat", "the flat pad"]}
# ordinal axes: key(obj) to sort ascending, phrase for rank k (k = 0 first)
AXES = {
    "left": (lambda s, o: s.xy[o][1], "the {o}{n} from the left", "the leftmost {n}"),
    "right": (lambda s, o: -s.xy[o][1], "the {o}{n} from the right", "the rightmost {n}"),
    "robot": (lambda s, o: float(np.hypot(*s.xy[o])), "the {o}closest {n} to the robot", "the {n} closest to the robot"),
    "far": (lambda s, o: -float(np.hypot(*s.xy[o])), "the {o}farthest {n} from the robot",
            "the {n} farthest from the robot"),
    "large": (lambda s, o: -s.specs[o].size[0], "the {o}largest {n}", "the largest {n}"),
    "small": (lambda s, o: s.specs[o].size[0], "the {o}smallest {n}", "the smallest {n}"),
}
POS_GAP = 0.025          # neighbours in an ordinal must differ by this much (m) to count as clearly ordered
SIZE_GAP = 0.004         # size ranks: >= 4 mm side difference
SIDE = {"left": np.array([0.0, -1.0]), "right": np.array([0.0, 1.0]), "front": np.array([1.0, 0.0]),
        "behind": np.array([-1.0, 0.0])}
SIDE_TEXT = {"left": "to the left of", "right": "to the right of", "front": "in front of", "behind": "behind"}


@dataclass
class Scene:
    specs: dict[str, ObjSpec]
    xy: dict[str, np.ndarray]
    support: dict[str, str | None]       # object -> what it starts on (None = table)

    def objects(self) -> list[str]:
        return [n for n, s in self.specs.items() if s.category == "object"]


def size_word(spec: ObjSpec) -> str | None:
    cm = spec.attrs.get("size_class")
    if cm is None:
        return None
    return "small" if cm <= 3.5 else "large" if cm >= 4.5 else None


def color_word(c: str | None) -> str | None:
    return "wooden" if c == "wood" else c


def _match(scene: Scene, noun: str | None, color: str | None = None, size: str | None = None,
           category: str | None = None) -> list[str]:
    out = []
    for n, s in scene.specs.items():
        if noun and s.noun != noun and not (noun == "object" and s.category == "object"):
            continue
        if category and s.category != category:
            continue
        if color and s.color != color:
            continue
        if size and size_word(s) != size:
            continue
        out.append(n)
    return out


def denote(e: dict, scene: Scene) -> set[str]:
    t = e["type"]
    if t in ("attr", "group"):
        return set(_match(scene, e.get("noun"), e.get("color"), e.get("size")))
    if t == "function":
        return set(_match(scene, None, category=e["category"]))
    if t == "only":
        return set(_match(scene, "object", e["color"]))
    if t == "on":
        region = denote(e["region"], scene)
        if len(region) != 1:
            return set()
        (r,) = region
        return {n for n in _match(scene, e["noun"], e.get("color")) if scene.support.get(n) == r}
    if t == "ordinal":
        cand = _match(scene, e["noun"], e.get("color"))
        key = AXES[e["axis"]][0]
        vals = sorted((key(scene, n), n) for n in cand)
        k = e["k"]
        if k >= len(vals):
            return set()
        gap = SIZE_GAP if e["axis"] in ("large", "small") else POS_GAP
        if (k > 0 and vals[k][0] - vals[k - 1][0] < gap) or (k + 1 < len(vals) and vals[k + 1][0] - vals[k][0] < gap):
            return set()
        return {vals[k][1]}
    if t == "relation":
        anchor = denote(e["anchor"], scene)
        if len(anchor) != 1:
            return set()
        (a,) = anchor
        cand = [n for n in _match(scene, e["noun"], e.get("color")) if n != a]
        d = SIDE[e["side"]]
        out = set()
        for n in cand:
            off = scene.xy[n] - scene.xy[a]
            along, perp = float(off @ d), float(abs(off @ np.array([-d[1], d[0]])))
            if along >= 0.03 and along >= 1.5 * perp:
                out.add(n)
        return out
    if t == "nearest":
        anchor = denote(e["anchor"], scene)
        if len(anchor) != 1:
            return set()
        (a,) = anchor
        cand = [n for n in _match(scene, e["noun"], e.get("color")) if n != a]
        dist = sorted((float(np.linalg.norm(scene.xy[n] - scene.xy[a])), n) for n in cand)
        if not dist or (len(dist) > 1 and dist[1][0] - dist[0][0] < POS_GAP):
            return set()
        return {dist[0][1]}
    raise ValueError(t)


def realize(e: dict, scene: Scene) -> str:
    t = e["type"]
    adj = " ".join(w for w in (e.get("size"), color_word(e.get("color"))) if w)
    noun = e.get("noun")
    np_ = f"{adj} {noun}".strip() if noun else ""
    if t == "attr":
        return f"the {np_}"
    if t == "group":
        n = e.get("n")
        plural = np_ + ("es" if np_.endswith("box") else "s")
        if e.get("all"):
            return f"all the {plural}"
        return f"both {plural}" if n == 2 and e.get("both") else f"the {NUM_WORDS.get(n, '')} {plural}".replace("  ", " ")
    if t == "function":
        return FUNCTION_NOUNS[e["category"]][e.get("variant", 0)]
    if t == "only":
        return f"the only {color_word(e['color'])} object"
    if t == "on":
        return f"the {np_} on {realize(e['region'], scene)}"
    if t == "ordinal":
        _, ranked, first = AXES[e["axis"]]
        nn = f"{color_word(e['color'])} {noun}" if e.get("color") else noun
        return first.format(n=nn) if e["k"] == 0 else ranked.format(o=ORD_WORDS[e["k"]], n=nn)
    if t == "relation":
        return f"the {np_} {SIDE_TEXT[e['side']]} {realize(e['anchor'], scene)}"
    if t == "nearest":
        return f"the {np_} nearest to {realize(e['anchor'], scene)}"
    raise ValueError(t)


def _attr_exprs(name: str, scene: Scene) -> list[dict]:
    s = scene.specs[name]
    out = [{"type": "attr", "noun": s.noun}, {"type": "attr", "noun": s.noun, "color": s.color}]
    if size_word(s):
        out += [{"type": "attr", "noun": s.noun, "size": size_word(s)},
                {"type": "attr", "noun": s.noun, "color": s.color, "size": size_word(s)}]
    if s.category == "object":
        out.append({"type": "attr", "noun": "object", "color": s.color})
    return out


def candidates(name: str, scene: Scene) -> dict[str, list[dict]]:
    """Valid (uniquely denoting) expressions for one object, grouped by kind; attr is shortest-first."""
    s = scene.specs[name]
    target = {name}
    out: dict[str, list[dict]] = {k: [] for k in ("attr", "function", "only", "on", "ordinal", "relation")}
    attrs = [e for e in _attr_exprs(name, scene) if denote(e, scene) == target]
    out["attr"] = attrs[:2]
    if s.category in FUNCTION_NOUNS:
        for v in range(len(FUNCTION_NOUNS[s.category])):
            e = {"type": "function", "category": s.category, "variant": v}
            if denote(e, scene) == target:
                out["function"].append(e)
    if s.category == "object":
        e = {"type": "only", "color": s.color}
        if denote(e, scene) == target and not any(x.get("color") and x["noun"] == "object" for x in attrs):
            out["only"].append(e)
        sup = scene.support.get(name)
        if sup is not None:
            for re_ in _anchor_exprs(sup, scene):
                e = {"type": "on", "noun": s.noun, "region": re_}
                if denote(e, scene) == target:
                    out["on"].append(e)
                    break
        for color in (None, s.color):
            for axis in AXES:
                for k in range(3):
                    e = {"type": "ordinal", "noun": s.noun, "axis": axis, "k": k, **({"color": color} if color else {})}
                    if denote(e, scene) == target and len(_match(scene, s.noun, color)) >= 2:
                        out["ordinal"].append(e)
        for other in scene.specs:
            if other == name:
                continue
            for anchor in _anchor_exprs(other, scene):
                for side in SIDE:
                    e = {"type": "relation", "noun": s.noun, "side": side, "anchor": anchor}
                    if denote(e, scene) == target and len(_match(scene, s.noun)) >= 2:
                        out["relation"].append(e)
                e = {"type": "nearest", "noun": s.noun, "anchor": anchor}
                if denote(e, scene) == target and len(_match(scene, s.noun)) >= 2:
                    out["relation"].append(e)
                break
    return out


def _anchor_exprs(name: str, scene: Scene) -> list[dict]:
    """Shortest unique attribute/function expression of an anchor object (relations nest one level only)."""
    out = [e for e in _attr_exprs(name, scene) if denote(e, scene) == {name}]
    if scene.specs[name].category in FUNCTION_NOUNS:
        e = {"type": "function", "category": scene.specs[name].category, "variant": 0}
        if denote(e, scene) == {name}:
            out.insert(0, e)
    return out[:1]


def group_candidates(names: list[str], scene: Scene) -> list[dict]:
    """Plural expressions that denote exactly `names`."""
    target = set(names)
    specs = [scene.specs[n] for n in names]
    nouns = {s.noun for s in specs}
    colors = {s.color for s in specs}
    out = []
    forms = []
    if len(nouns) == 1:
        (noun,) = nouns
        forms.append({"noun": noun})
        if len(colors) == 1:
            forms.append({"noun": noun, "color": next(iter(colors))})
    if len(colors) == 1:
        forms.append({"noun": "object", "color": next(iter(colors))})
    for f in forms:
        e = {"type": "group", **f, "n": len(names)}
        if denote(e, scene) == target:
            out.append(e)
            out.append({**e, "all": True})
            if len(names) == 2:
                out.append({**e, "both": True})
    return out


def describe_list(names: list[str], scene: Scene, rng: np.random.Generator) -> str:
    """Fallback group text: enumerate unique member expressions ("the red cube, the blue cube and the green cube")."""
    parts = [realize(pick(n, scene, rng, kinds=("attr",)), scene) for n in names]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def pick(name: str, scene: Scene, rng: np.random.Generator, kinds=None, weights=None) -> dict:
    """Choose one valid expression: kind drawn by weight among kinds that have a candidate; attribute is the fallback."""
    cand = candidates(name, scene)
    kinds = [k for k in (kinds or cand) if cand.get(k)]
    if not kinds:
        raise LookupError(f"no unique expression for {name}")
    w = np.array([(weights or {}).get(k, 1.0) for k in kinds], dtype=float)
    k = kinds[int(rng.choice(len(kinds), p=w / w.sum()))]
    opts = cand[k]
    return opts[int(rng.integers(len(opts)))] if k != "attr" else opts[0]
