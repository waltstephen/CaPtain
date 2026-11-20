"""Fixed successful workload (soarm-speed primary metric).

    python tools/success_workload.py build <out dir> <run dir>... --records rec_A.jsonl rec_B.jsonl ...
    python tools/success_workload.py cost <workload dir> --profile release|B|C|fast [--calls-ref <run dir>]
    python tools/success_workload.py append-only <workload dir> <out dir> --tokenizer <model dir> [--limit 110000]

build: every STRICT-success episode (success + finish()) of the given realistic-path runs whose request bodies were
recorded (tools/policy_proxy.py --record) -> <out>/<run>__<episode>.jsonl (the episode's request bodies in order) and
manifest.jsonl (run, episode, requests, recorded real-arm wall, tool calls). The bodies are replayed through every
serving config with tools/policy_replay.py --override '<config sampling>' (model seconds per successful episode).

cost: the same episodes' recorded tool calls priced under a harness profile (real-arm seconds, no model time):
  - pose move_to: max(translation / v, rotation / w) + tracking/settle overhead, with v = min(k * requested, 0.1) and
    w = min(k * requested, 1.0) (requested = the call's argument or the profile default; k = motion_speed_scale), and
    above-cap step profiles dividing the travel time by step_speed_scale (upper bound: ramp/deceleration ignored);
  - gripper open/close: 2.0 s (40 ticks) or, with gripper_settle_min_steps, the median measured with early stop;
  - recover / hold / release: recorded simulated time;
  - perception calls and observe: each call's own recorded wall in its episode (identical for every profile, so the
    SAM3 cache shows up only in live runs), observe without the sim-only rendering;
  - episode start: measured start-up (release 5.75 s, fast-start 4.6 s - twin measurement).
Output per episode + total; the cost model is an approximation, labelled as such in the report.
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import math
import re
import statistics as st
from pathlib import Path

CONTROL_HZ = 20.0
CWD_RE = re.compile(r"<cwd>(.*?)</cwd>", re.S)
PROFILES = {   # requested-speed handling of each stack
    "release": {"k": 1.0, "v0": 0.05, "w0": 0.5, "step": 1.0, "gripper_s": 2.0, "start_s": 5.75},
    "B": {"k": 3.0, "v0": 0.1, "w0": 1.0, "step": 1.0, "gripper_s": 2.0, "start_s": 5.75},
    "C": {"k": 3.0, "v0": 0.1, "w0": 1.0, "step": 1.0, "gripper_s": None, "start_s": 4.6},
    "fast": {"k": 3.0, "v0": 0.1, "w0": 1.0, "step": 3.0, "gripper_s": None, "start_s": 4.6},
}
PERCEPTION = {"segment", "mark_point", "measure_object_geometry", "select_detection", "track_target", "point",
              "measure_image_points", "inspect_contact_geometry", "compare_object_observations",
              "inspect_release_geometry", "observe"}


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def episode_key(body: dict) -> str | None:
    for m in body.get("messages", []):
        content = m.get("content")
        text = content if isinstance(content, str) else json.dumps(content)
        hit = CWD_RE.search(text or "")
        if hit:
            return str(Path(hit.group(1)).parent)
    return None


def strict_successes(run: Path) -> set[str]:
    eps = set()
    for f in (run / "speed.jsonl", Path(str(run) + "_ext") / "batch.jsonl"):
        for r in jsonl(f):
            if r.get("success") and r.get("finished") and r.get("episode"):
                eps.add(str(f.parent / r["episode"]))
    return eps


def build(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wanted = set().union(*(strict_successes(Path(r)) for r in a.runs))
    bodies: dict[str, list[dict]] = collections.defaultdict(list)
    for rec in a.records:
        for line in Path(rec).read_text().splitlines():
            if not line.strip():
                continue
            b = json.loads(line)
            key = episode_key(b)
            if key in wanted:
                bodies[key].append(b)
    manifest = []
    for ep in sorted(bodies):
        epp = Path(ep)
        name = f"{epp.parent.name}__{epp.name}"
        (out / f"{name}.jsonl").write_text("".join(json.dumps(b) + "\n" for b in bodies[ep]))
        speed = {r.get("episode"): r for r in jsonl(epp.parent / "speed.jsonl")}
        manifest.append({"name": name, "episode_dir": ep, "requests": len(bodies[ep]),
                         "real_wall_s": (speed.get(epp.name) or {}).get("real_wall_s")})
    (out / "manifest.jsonl").write_text("".join(json.dumps(m) + "\n" for m in manifest))
    print(json.dumps({"episodes": len(manifest), "missing_bodies": sorted(wanted - set(bodies))}))


def append_only(a) -> None:
    """Lever K1 counterfactual: the same episodes with every history rewrite (context compaction, image-cap
    placeholders) undone - request i = request i-1's FULL history + the messages that are new in request i. Same
    turns, tool outputs and images; only the context policy differs (what K1's large window does live). A rebuilt
    body that would reach --limit tokens (K1's context_limit) is replaced by the recorded body: the driver compacts
    there too, and appending restarts from that compacted history."""
    from captain_harness.react.chat_llm import cap_images
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    sizes: dict[str, int] = {}

    def tokens(msgs: list[dict]) -> int:   # per-message text tokens + 258 per 512x512 image (vLLM /tokenize)
        n = 0
        for x in msgs:
            key = json.dumps(x, sort_keys=True)
            if key not in sizes:
                c = x.get("content")
                parts = [{"type": "text", "text": c}] if isinstance(c, str) else (c or [])
                text = [p.get("text") or "" for p in parts if p.get("type") == "text"]
                text += [x.get("reasoning_content") or ""] + [json.dumps(t) for t in x.get("tool_calls") or []]
                sizes[key] = len(tok.encode("\n".join(text))) + 258 * sum(p.get("type") == "image_url" for p in parts)
            n += sizes[key]
        return n

    src, out = Path(a.workload), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = []
    for m in jsonl(src / "manifest.jsonl"):
        bodies = jsonl(src / f"{m['name']}.jsonl")
        full: list[dict] = []
        prev: list[dict] = []
        rebuilt, undone, kept = [], 0, 0
        for b in bodies:
            msgs = b["messages"]
            k = 0
            while k < min(len(prev), len(msgs)) and msgs[k] == prev[k]:
                k += 1
            if not prev or k == len(prev):
                full = full + msgs[k:]
            else:   # rewrite: the new turns follow the last kept copy of prev's final message
                last = max((j for j, x in enumerate(msgs) if x == prev[-1]), default=None)
                if last is None:
                    raise SystemExit(f"{m['name']}: cannot align a rewrite (prev final message not kept)")
                candidate = full + msgs[last + 1:]
                if tokens(candidate) < a.limit:
                    full, undone = candidate, undone + 1
                else:
                    full, kept = msgs, kept + 1
            prev = msgs
            capped = copy.deepcopy(full)   # the driver caps images on every request (K1: 32 images, chunk 6)
            cap_images(capped, a.max_images, a.image_drop_chunk)
            rebuilt.append({**b, "messages": capped})
        images = sum(p.get("type") == "image_url" for x in rebuilt[-1]["messages"]
                     if isinstance(x.get("content"), list) for p in x["content"])
        (out / f"{m['name']}.jsonl").write_text("".join(json.dumps(b) + "\n" for b in rebuilt))
        stats.append({**m, "rewrites_undone": undone, "rewrites_kept": kept, "final_images": images,
                      "final_tokens": tokens(rebuilt[-1]["messages"])})
    (out / "manifest.jsonl").write_text("".join(json.dumps(s) + "\n" for s in stats))
    print(json.dumps({"episodes": len(stats), "rewrites_undone": sum(s["rewrites_undone"] for s in stats),
                      "rewrites_kept": sum(s["rewrites_kept"] for s in stats),
                      "max_final_images": max(s["final_images"] for s in stats),
                      "max_final_tokens": max(s["final_tokens"] for s in stats)}))


def tool_medians(ref: Path) -> dict[str, float]:
    per = collections.defaultdict(list)
    for f in ref.glob("soarm_sim_*/calls.jsonl"):
        for c in jsonl(f):
            if c.get("name") in PERCEPTION:
                per[c["name"]].append(float(c.get("elapsed_s") or 0.0))
    med = {k: st.median(v) for k, v in per.items()}
    # observe on the real arm: no rendering; keep the depth + encoding part (render_s of the same run's timing.json)
    renders = [json.loads(p.read_text()) for p in ref.glob("soarm_sim_*/soarm/timing.json")]
    if renders and "observe" in med:
        per_render = sum(t.get("render_wall_s", 0) for t in renders) / max(sum(t.get("render_calls", 0) for t in renders), 1)
        med["observe"] = max(0.0, med["observe"] - per_render)
    return med


def gripper_median(ref: Path) -> float:
    vals = [float((c.get("value") or {}).get("control_steps") or 0) / CONTROL_HZ
            for f in ref.glob("soarm_sim_*/calls.jsonl") for c in jsonl(f)
            if c.get("name") == "move_to" and "gripper" in (c.get("arguments") or {}) and isinstance(c.get("value"), dict)]
    return st.median(vals) if vals else 2.0


def cost(a) -> None:
    prof = PROFILES[a.profile]
    ref = Path(a.calls_ref)
    med = tool_medians(ref)
    grip_s = prof["gripper_s"] if prof["gripper_s"] is not None else gripper_median(ref)
    rows = []
    for m in jsonl(Path(a.workload) / "manifest.jsonl"):
        calls = jsonl(Path(m["episode_dir"]) / "calls.jsonl")
        tj = Path(m["episode_dir"]) / "soarm" / "timing.json"
        t = json.loads(tj.read_text()) if tj.exists() else {}
        render_per_observe = t.get("render_wall_s", 0.0) / max(t.get("render_calls", 0), 1)
        motion = perception = 0.0
        for c in calls:
            name, args = c.get("name"), c.get("arguments") or {}
            v = c.get("value") if isinstance(c.get("value"), dict) else {}
            if name == "move_to" and "gripper" in args:
                motion += grip_s
            elif name == "move_to":
                d = float(v.get("executed_distance_m") or 0.0)
                rot = v.get("actual_rotation_vector_rad") or [0.0]
                theta = math.sqrt(sum(x * x for x in rot)) if isinstance(rot, list) else 0.0
                if d <= 0 and theta <= 0:
                    continue
                vel = min(prof["k"] * float(args["max_speed_m_s"]), 0.1) if args.get("max_speed_m_s") else prof["v0"]
                ang = (min(prof["k"] * float(args["max_angular_speed_rad_s"]), 1.0)
                       if args.get("max_angular_speed_rad_s") else prof["w0"])
                travel = max(d / vel, theta / ang) / prof["step"]
                motion += travel + 0.15   # tracking + endpoint settle (2-3 ticks measured)
            elif name in ("recover", "hold", "release"):
                motion += float(v.get("control_steps") or 0) / CONTROL_HZ
            elif name in PERCEPTION:
                # the call's own recorded wall (same for every profile: the workload fixes what is perceived);
                # observe without the sim-only rendering
                el = float(c.get("elapsed_s") or 0.0)
                perception += max(0.0, el - render_per_observe) if name == "observe" else el
        rows.append({"name": m["name"], "motion_s": round(motion, 1), "perception_s": round(perception, 1),
                     "start_s": prof["start_s"], "tool_s": round(motion + perception + prof["start_s"], 1)})
    out = {"profile": a.profile, "calls_ref": str(ref), "episodes": len(rows),
           "tool_s_total": round(sum(r["tool_s"] for r in rows), 1),
           "tool_s_median": round(st.median(r["tool_s"] for r in rows), 1) if rows else None,
           "motion_s_total": round(sum(r["motion_s"] for r in rows), 1),
           "perception_s_total": round(sum(r["perception_s"] for r in rows), 1), "per_episode": rows}
    print(json.dumps(out))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("out")
    b.add_argument("runs", nargs="+")
    b.add_argument("--records", nargs="+", required=True)
    c = sub.add_parser("cost")
    c.add_argument("workload")
    c.add_argument("--profile", choices=sorted(PROFILES), required=True)
    c.add_argument("--calls-ref", required=True, help="a realistic-path run of this profile (per-call medians)")
    c.add_argument("--out", default=None)
    k = sub.add_parser("append-only")
    k.add_argument("workload")
    k.add_argument("out")
    k.add_argument("--tokenizer", required=True, help="the policy's model dir")
    k.add_argument("--limit", type=int, default=110000, help="K1 context_limit: the driver compacts at this size")
    k.add_argument("--max-images", type=int, default=32)
    k.add_argument("--image-drop-chunk", type=int, default=6)
    a = ap.parse_args()
    {"build": build, "cost": cost, "append-only": append_only}[a.cmd](a)


if __name__ == "__main__":
    main()
