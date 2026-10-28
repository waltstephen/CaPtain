"""Official SimplerEnv episode list -> one task per episode, written as <OUT>/<setting>.jsonl (bridge_vm, google_vm, google_va).

The official RT-1 scripts of SimplerEnv (scripts/rt1x_bridge.sh, scripts/rt1_*_{visual_matching,variant_agg}.sh) were run
with a `python` shim that only records each main_inference.py argv (official/*.argv). Here every distinct invocation
(policy/ckpt args dropped) is parsed with SimplerEnv's own get_args() and expanded exactly like maniskill2_evaluator():
robot x * y * quat * (object x * y | episode ids). Run it with SimplerEnv's interpreter from the SimplerEnv checkout:

    cd $SIMPLERENV_ROOT && OUT=/tmp/simplerenv_tasks $SIMPLERENV_PYTHON <harness>/eval/transfer/simplerenv/gen_tasks.py

bridge_vm.jsonl is identical to the shipped bridge_vm_96.jsonl (the 4 WidowX tasks x 24 episodes of the paper).
"""
import json, sys, os, hashlib
from pathlib import Path
import numpy as np
sys.argv = ["x"]
from simpler_env.evaluation.argparse import get_args  # noqa: E402

OFF = Path(__file__).resolve().parent.parent / "official"
OUT = Path(os.environ.get("OUT", "simplerenv_tasks"))
SETTINGS = {"rt1x_bridge": ("bridge_vm", None), "rt1_pick_coke_can_visual_matching": ("google_vm", "pick_coke_can"),
            "rt1_move_near_visual_matching": ("google_vm", "move_near"), "rt1_drawer_visual_matching": ("google_vm", "drawer"),
            "rt1_put_in_drawer_visual_matching": ("google_vm", "apple_in_drawer"),
            "rt1_pick_coke_can_variant_agg": ("google_va", "pick_coke_can"), "rt1_move_near_variant_agg": ("google_va", "move_near"),
            "rt1_drawer_variant_agg": ("google_va", "drawer"), "rt1_put_in_drawer_variant_agg": ("google_va", "apple_in_drawer")}
BRIDGE = {"PutSpoonOnTableClothInScene-v0": "spoon_on_towel", "PutCarrotOnPlateInScene-v0": "carrot_on_plate",
          "StackGreenCubeOnYellowCubeBakedTexInScene-v0": "stack_cube", "PutEggplantInBasketScene-v0": "eggplant_in_basket"}


def parse_kv(items):
    out = {}
    for kv in items or []:
        k, v = kv.split("=", 1)
        try:
            v = eval(v)  # SimplerEnv's own parser does the same (simpler_env/evaluation/argparse.py)
        except Exception:
            pass
        out[k] = v
    return out


rows, seen = [], set()
for stem, (setting, task_group) in SETTINGS.items():
    raw = (OFF / f"{stem}.argv").read_bytes().split(b"\0\n")
    for rec in raw:
        if not rec.strip():
            continue
        argv = [a.decode() for a in rec.split(b"\0")]
        argv = argv[argv.index("simpler_env/main_inference.py") + 1:]
        keep, i = [], 0
        while i < len(argv):
            if argv[i] in ("--ckpt-path", "--policy-model"):
                i += 2
                continue
            keep.append(argv[i]); i += 1
        key = (stem, tuple(keep))
        if key in seen:
            continue
        seen.add(key)
        sys.argv = ["main_inference.py", "--policy-model", "rt1", "--ckpt-path", "none"] + keep
        a = get_args()
        add = dict(a.additional_env_build_kwargs or {})
        if a.enable_raytracing:
            add = {"shader_dir": "rt", **add}
        group = task_group or BRIDGE[a.env_name]
        variant = ("camera:" if "Camera" in a.env_name else "") + "|".join(
            [a.scene_name] + [f"{k}={v}" for k, v in sorted(add.items()) if k not in ("shader_dir",)]
            + ([a.env_name] if task_group else []))
        objs = ([dict(obj_init_x=float(x), obj_init_y=float(y)) for x in a.obj_init_xs for y in a.obj_init_ys]
                if a.obj_variation_mode == "xy" else
                [dict(obj_episode_id=int(e)) for e in range(a.obj_episode_range[0], a.obj_episode_range[1])])
        for rx in a.robot_init_xs:
            for ry in a.robot_init_ys:
                for q in a.robot_init_quats:
                    for o in objs:
                        spec = dict(env_name=a.env_name, robot=a.robot, scene_name=a.scene_name,
                                    control_freq=int(a.control_freq), sim_freq=int(a.sim_freq),
                                    max_episode_steps=int(a.max_episode_steps), rgb_overlay_path=a.rgb_overlay_path,
                                    robot_init_x=float(rx), robot_init_y=float(ry),
                                    robot_init_quat=[float(v) for v in q], additional_env_build_kwargs=add, **o)
                        h = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]
                        rows.append(dict(setting=setting, group=group, variant=variant, env_name=a.env_name,
                                         official_aggregate=not variant.startswith("camera:"),
                                         episode_key=f"{setting}-{group}-{h}", simplerenv=spec,
                                         advance_subtask=a.env_name.startswith("PlaceIntoClosed")))
OUT.mkdir(parents=True, exist_ok=True)
from collections import Counter
by_setting = {}
for r in rows:
    by_setting.setdefault(r["setting"], []).append(r)
for setting, rs in by_setting.items():
    with (OUT / f"{setting}.jsonl").open("w") as stream:
        for r in sorted(rs, key=lambda r: (r["group"], r["episode_key"])):
            stream.write(json.dumps(r) + "\n")
c = Counter((r["setting"], r["group"], r["official_aggregate"]) for r in rows)
for k in sorted(c):
    print(k, c[k])
print("variants per setting/group:", Counter((r["setting"], r["group"]) for r in {(r["setting"], r["group"], r["variant"]): r for r in rows}.values()))
print("distinct keys", len({r["episode_key"] for r in rows}), "rows", len(rows))
