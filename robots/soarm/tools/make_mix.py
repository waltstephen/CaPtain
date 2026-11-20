"""Write a training mix (soarm/specs/mixes/<id>.json, read by soarm.mix / tools/batch.py --mix) from scripted-solver
results (python -m taskgen.verify ... --scene-config <the mix's scene>), i.e. the "solver table".

    python tools/make_mix.py --id div2 --results r1.jsonl,r2.jsonl --config div2_config.json --intro v4
    python tools/make_mix.py --id div3 --shares div3 --precise-factor 0.35 --results ... --config configs/div3_scene.json

Groups (shares of strict successes): cube_base = hand cube specs (tid 100-199), cube_var = div2 variants (200-999),
cube_div3 = div3 cube variants (notes "div3 variant"; only with --shares div3, else part of cube_var), cube_llm =
taskgen specs (domain cube, provenance llm), noncube = broad-domain primitive specs, mesh = GSO/YCB registry-mesh specs
(only with --shares div3, else part of noncube). A cube group without accepted tasks hands its share to the other cube
groups, a non-cube group to the other non-cube group (to the cube groups if both are empty). Inside a group, weight ~
scripted success x difficulty factor (x --precise-factor for depth-precise goals: stacking, lines, slots, relations to a
cube, the single-step tasks that collapsed under MoGe-2 depth -- they stay in the mix, down-weighted); a task needs
scripted success >= --min-solved-frac, is train_safe (no held-out family structure), and gets at most --max-share of
the whole mix (excess goes to the rest of its group).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from soarm import taskspec
from soarm.mix import MIX_DIR, train_safe

SHARE_SETS = {
    "div2": {"cube_base": 0.25, "cube_var": 0.40, "cube_llm": 0.15, "noncube": 0.20},
    # div3: the MoGe-2-tolerant cube variants get the largest cube share; primitives + meshes 28 %
    "div3": {"cube_base": 0.14, "cube_var": 0.20, "cube_div3": 0.24, "cube_llm": 0.14, "noncube": 0.16, "mesh": 0.12},
}
CUBE_GROUPS = ("cube_base", "cube_var", "cube_div3", "cube_llm")
OTHER_GROUPS = ("noncube", "mesh")
DIFFICULTY_FACTOR = {1: 1.0, 2: 1.0, 3: 0.9, 4: 0.7, 5: 0.5}
# Depth-precise goal structure (MoGe-2 smoke + production 11-11: stacking, lines, slots and relations to a small cube
# fell from ~60 % to ~10 % strict) plus the single-step tasks that collapsed in the paired MoGe-2 smoke.
PRECISE_PREDS = ("on_top", "stacked", "in_line")
PRECISE_IDS = ("cube_pick_lift", "cube_put_all_in_container", "cube_relational_place", "cube_v_lift_small",
               "cube_v_lift_smallest", "cube_v_empty_container")


def group_of(spec: dict, shares: dict | None = None) -> str | None:
    shares = shares or SHARE_SETS["div2"]
    if spec["domain"] == "broad":
        return "mesh" if "mesh" in shares and spec["id"].startswith("mesh_") else "noncube"
    if spec["domain"] != "cube":
        return None
    if spec["provenance"] == "llm":
        return "cube_llm"
    if spec["tid"] < 200:
        return "cube_base"
    return "cube_div3" if "cube_div3" in shares and "div3 variant" in spec.get("notes", "") else "cube_var"


def precise(spec: dict) -> str | None:
    """Why a spec needs centimetre-precise depth; None if its goals tolerate ~1-2 cm of position error."""
    if spec["id"] in PRECISE_IDS:
        return "collapsed under MoGe-2 (11-11 paired smoke)"
    kinds = {o["role"]: (o["kind"] if isinstance(o["kind"], list) else [o["kind"]]) for o in spec["objects"]}
    if any("slot" in k for k in kinds.values()):
        return "tight slot"
    for g in spec["goal"]:
        g = g.get("count", g)
        if g["pred"] in PRECISE_PREDS:
            return g["pred"]
        if g["pred"] == "relation" and len(g["args"]) > 1 and isinstance(g["args"][1], str) \
                and g["args"][1].startswith("@") and kinds.get(taskspec._role_of(g["args"][1])) == ["cube"]:
            return "relation to a cube"
    return None


def cap_weights(w: dict[str, float], total: float, cap: float) -> dict[str, float]:
    """Scale w to sum `total` with no entry above `cap` (water-filling)."""
    out, free = {}, dict(w)
    remaining = total
    while free:
        s = sum(free.values())
        over = {k: v for k, v in free.items() if v / s * remaining > cap}
        if not over:
            out.update({k: v / s * remaining for k, v in free.items()})
            break
        for k in over:
            out[k] = cap
            remaining -= cap
            del free[k]
    return out


def group_shares(base: dict[str, float], live: set[str]) -> dict[str, float]:
    """Shares after empty groups hand theirs on: cube -> live cube groups, non-cube -> the other non-cube group, else
    -> live cube groups (proportionally)."""
    shares = {g: (s if g in live else 0.0) for g, s in base.items()}
    cubes = [g for g in CUBE_GROUPS if g in base and g in live]
    others = [g for g in OTHER_GROUPS if g in base and g in live]
    for g, s in base.items():
        if g in live:
            continue
        to = cubes if g in CUBE_GROUPS or not others else others
        for t in to:
            shares[t] += s * base[t] / sum(base[x] for x in to)
    return shares


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--results", required=True, help="comma-separated taskgen.verify result jsonl files (later wins)")
    ap.add_argument("--config", required=True, help="SceneConfig JSON file of the mix (the one the solver ran with)")
    ap.add_argument("--intro", default="v4")
    ap.add_argument("--description", default="")
    ap.add_argument("--shares", default="div2", choices=sorted(SHARE_SETS))
    ap.add_argument("--precise-factor", type=float, default=1.0,
                    help="weight factor for depth-precise specs (precise()); 1 = no down-weighting (div2)")
    ap.add_argument("--min-solved-frac", type=float, default=0.5)
    ap.add_argument("--max-share", type=float, default=0.025)
    a = ap.parse_args()
    base = SHARE_SETS[a.shares]
    rows = {}
    for path in a.results.split(","):
        for line in Path(path).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r
    reg = taskspec.registry()
    cand: dict[str, dict[str, float]] = {g: {} for g in base}
    info, excluded = {}, []
    for sid, r in sorted(rows.items()):
        spec = reg.get(sid)
        if spec is None:
            excluded.append({"id": sid, "reason": "not in the spec registry"})
            continue
        group, why = group_of(spec, base), train_safe(spec)
        ph = r.get("physics_summary") or {}
        solved, n = ph.get("solved", 0), ph.get("layouts", 0)
        prec = precise(spec) if a.precise_factor != 1.0 else None
        info[sid] = {"group": group, "scripted": [solved, n], "difficulty": spec["difficulty"],
                     "layouts": (r.get("layouts") or {}).get("accepted"), **({"precise": prec} if prec else {})}
        frac = solved / n if n else 0.0
        if group is None or why:
            excluded.append({"id": sid, "reason": why or "domain not in a mix group"})
        elif r["stage"] in ("schema", "layouts"):
            excluded.append({"id": sid, "reason": f"verifier stage {r['stage']}: {'; '.join(r.get('errors', []))[:160]}"})
        elif frac < a.min_solved_frac:
            excluded.append({"id": sid, "reason": f"scripted {solved}/{n} < {a.min_solved_frac:.0%}"})
        else:
            cand[group][sid] = frac * DIFFICULTY_FACTOR[spec["difficulty"]] * (a.precise_factor if prec else 1.0)
    shares = group_shares(base, {g for g, c in cand.items() if c})
    tasks = []
    for g, w in cand.items():
        for sid, weight in sorted(cap_weights(w, shares[g], a.max_share).items()):
            tasks.append({"id": sid, "group": g, "weight": round(weight, 5), **{k: v for k, v in info[sid].items()
                                                                               if k != "group"}})
    config = json.loads(Path(a.config).read_text())
    mix = {"id": a.id, "description": a.description, "intro": a.intro, "config": config,
           "retire": {"min_attempts": 10, "min_yield": 0.15}, "max_share": a.max_share,
           "precise_factor": a.precise_factor,
           "groups": {g: {"share": round(shares[g], 4), "tasks": len(cand[g])} for g in base},
           "tasks": tasks, "excluded": excluded}
    MIX_DIR.mkdir(parents=True, exist_ok=True)
    (MIX_DIR / f"{a.id}.json").write_text(json.dumps(mix, indent=1) + "\n")
    print(json.dumps(mix["groups"]), f"{len(tasks)} tasks, {len(excluded)} excluded")


if __name__ == "__main__":
    main()
