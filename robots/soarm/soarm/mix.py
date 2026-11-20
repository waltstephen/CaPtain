"""Training data mixes (soarm/specs/mixes/<id>.json): one scene distribution + robot intro + task weights, run by
`tools/batch.py --mix <id> --target-total N` (written by tools/make_mix.py from scripted-solver results).

    {"id": "div2", "description": ..., "intro": "v4", "config": {SceneConfig},
     "retire": {"min_attempts": 10, "min_yield": 0.15},
     "groups": {"cube_base": {"share": 0.25, ...}, ...},
     "tasks": [{"id": "cube_pick_lift", "group": "cube_base", "weight": 0.021, "scripted": [9, 10], "difficulty": 1}, ...]}

Weights are shares of STRICT successes (normalised on load). A mix may only hold train specs whose goal structure is
not a held-out family's (train_safe) and a train-only scene (train cubes, train camera views).
"""
from __future__ import annotations

import json
from pathlib import Path

from .taskspec import SPEC_DIR, get
from .tasks import SceneConfig

MIX_DIR = SPEC_DIR / "mixes"
# held-out cube families (CONTRACT 7): gather (grouped), swap (at_initial_of), count_place (count), unstack (stacked start
# + resting_on_table goal) -- training specs must not carry their goal structure
HELDOUT_PREDICATES = ("grouped", "at_initial_of")
HELDOUT_VIEWS = ("heldout", "rear", "overhead")


def train_safe(spec: dict) -> str | None:
    """None if `spec` may be trained on, else the reason."""
    if spec["split"] != "train":
        return "split is not train"
    for g in spec["goal"]:
        if "count" in g:
            return "count goal (held-out count_place)"
        if g["pred"] in HELDOUT_PREDICATES:
            return f"{g['pred']} goal (held-out family)"
    stacked = any(c.get("type") == "stack" for c in spec.get("layout") or [])
    if stacked and any(g.get("pred") == "resting_on_table" for g in spec["goal"]):
        return "stacked start + resting_on_table goal (held-out unstack)"
    return None


def load(ref: str) -> dict:
    """{"id", "config": SceneConfig, "intro", "weights": {spec id: share (sum 1)}, "retire", "raw"}; raises on a
    held-out spec, an unknown spec or a non-train scene."""
    path = Path(ref) if str(ref).endswith(".json") else MIX_DIR / f"{ref}.json"
    d = json.loads(path.read_text())
    config = SceneConfig.from_json(d["config"])
    if config.cube_split != "train" or config.camera in HELDOUT_VIEWS:
        raise ValueError(f"{d['id']}: training mixes use train cubes and train camera views")
    weights = {}
    for t in d["tasks"]:
        if t["weight"] <= 0:
            continue
        why = train_safe(get(t["id"]))
        if why:
            raise ValueError(f"{d['id']}: {t['id']} is not trainable ({why})")
        weights[t["id"]] = float(t["weight"])
    total = sum(weights.values())
    return {"id": d["id"], "config": config, "intro": d.get("intro"), "raw": d,
            "weights": {k: v / total for k, v in weights.items()},
            "retire": {"min_attempts": 10, "min_yield": 0.15, **(d.get("retire") or {})}}


def targets(weights: dict[str, float], total: int) -> dict[str, int]:
    """Strict-success target per task: round(total x weight), at least 1."""
    return {k: max(1, round(total * w)) for k, w in weights.items()}
