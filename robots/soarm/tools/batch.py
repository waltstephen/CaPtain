"""Resumable batch of teacher episodes: (task spec id, seed) jobs, low concurrency, skip finished, per-episode summary line.

    python tools/batch.py --out runs/pilot_terra --model gpt-6.1-terra --families legacy --seeds 100-104 --lanes 3
    python tools/batch.py --out runs/eval_<model>/cube_seen --model <model> --suite cube_seen --machine armhost --lanes 2
    python tools/batch.py --out runs/scale --model gpt-6.1-terra --plan plan.jsonl --lanes 3 --stop-file STOP
    python tools/batch.py --out runs/scale_terra --model gpt-6.1-terra --target-per-family 80 --seed-start 200 --lanes 4
    python tools/batch.py --out runs/scale_cube --model deepseek-v4.1-flash-1017 --families cube --target-per-family 60 \
        --machine armhost --lanes 6
    python tools/batch.py --out runs/div2_sol --model gpt-6.1-sol --mix div2 --target-total 2000 \
        --machine datahost --seed-start 1100000 --lanes 12

Episodes go to <out>/soarm_sim_<tid>_seed-<seed>/; <out>/batch.jsonl gets one row per finished episode (success, finish,
requests, tokens, wall). A `--stop-file` that appears stops launching new episodes. `--max-consecutive-errors` stops the
batch when the gateway keeps failing (episodes that end with model errors) or episodes keep ending in
infrastructure_error (log line "batch_stopped").
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parse_range(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def summarize(ep: Path) -> dict:
    res = json.loads((ep / "episode_result.json").read_text())
    reqs = [json.loads(line) for line in (ep / "driver" / "requests.jsonl").read_text().splitlines() if line.strip()] \
        if (ep / "driver" / "requests.jsonl").exists() else []
    tin = sum((r.get("usage") or {}).get("input_tokens", 0) or 0 for r in reqs)
    tout = sum((r.get("usage") or {}).get("output_tokens", 0) or 0 for r in reqs)
    cached = sum(((r.get("usage") or {}).get("input_tokens_details") or {}).get("cached_tokens", 0) or 0 for r in reqs)
    return {"episode": ep.name, "success": res.get("host_final_success"), "finished": res.get("agent_finished"),
            "false_finish": res.get("false_finish"), "termination": res.get("termination_reason"),
            "requests": len(reqs), "input_tokens": tin, "cached_tokens": cached, "output_tokens": tout,
            "failed_actions": res.get("failed_actions"), "model_latency_s": round(sum(r.get("latency_s") or 0 for r in reqs), 1),
            **prompt_summary(ep), **perception_summary(ep)}


def prompt_summary(ep: Path) -> dict:
    """What the agent's prompt was (soarm/prompt.json: intro version, harness prompt profile, TASK.md version; absent
    before div3c = the full TASK.md) plus the sha256 of the TASK.md the agent got."""
    import hashlib
    path = ep / "soarm" / "prompt.json"
    rec = json.loads(path.read_text()) if path.exists() else {"prompt_profile": "full"}
    task_md = ep / "workspace" / "TASK.md"
    return {"intro": rec.get("intro"), "prompt_profile": rec.get("prompt_profile"),
            "task_md_version": rec.get("task_md_version"),
            "task_md_sha256": hashlib.sha256(task_md.read_bytes()).hexdigest()[:16] if task_md.exists() else None}


def perception_summary(ep: Path) -> dict:
    """Perception mode of the episode (soarm/perception.json: mode + SAM3/Molmo model revisions, depth source + MoGe-2
    revision) and its service calls (soarm/perception.jsonl for real services, depth estimates as kind=depth rows;
    soarm/perception_sim.jsonl only ever exists for the privileged stand-in)."""
    cfg_path = ep / "soarm" / "perception.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {"mode": "privileged"}   # pre-flag episodes
    rows = []
    for name in ("perception.jsonl", "perception_sim.jsonl"):
        path = ep / "soarm" / name
        if path.exists():
            rows += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    lat = [r["latency_s"] for r in rows if r.get("latency_s") is not None]
    depth_rows = [r for r in rows if r.get("kind") == "depth"]
    scales = sorted(r["align"]["scale"] for r in depth_rows if isinstance(r.get("align"), dict) and "scale" in r["align"])
    return {"perception": cfg.get("mode"), "perception_mode": cfg.get("mode"),
            "perception_tool_revisions": cfg.get("tool_revisions"),
            "perception_calls": len(rows),
            "perception_real_requests": sum(1 for r in rows if r.get("privileged") is False and r.get("method") == "POST"
                                            and r.get("ok") and r.get("kind") != "depth"),
            "perception_privileged_calls": sum(1 for r in rows if r.get("privileged") is not False),
            "perception_sim_log": (ep / "soarm" / "perception_sim.jsonl").exists(),
            "perception_errors": sum(1 for r in rows if r.get("ok") is False),
            "perception_latency_s": round(sum(lat), 2),
            "depth_source": cfg.get("depth", "sim_sensor"),   # episodes before SOARM_DEPTH: rendered sensor depth
            "depth_revision": cfg.get("depth_revision"),
            "depth_url": cfg.get("depth_url"),
            "depth_requests": sum(1 for r in depth_rows if r.get("ok") and not r.get("cache_hit")),
            "depth_errors": sum(1 for r in depth_rows if r.get("ok") is False),
            "depth_latency_s": round(sum(r.get("latency_s") or 0 for r in depth_rows), 2),
            "depth_plane_scale_median": scales[len(scales) // 2] if scales else None}


def perception_violation(row: dict) -> str | None:
    """Why an episode is not real-perception data (None = clean)."""
    if row.get("perception_mode") != "real":
        return f"perception_mode={row.get('perception_mode')}"
    if row.get("perception_privileged_calls") or row.get("perception_sim_log"):
        return "privileged perception rows"
    if row.get("depth_source") != "moge2" or not row.get("depth_revision"):
        return f"depth_source={row.get('depth_source')}"
    return None


def check_services() -> None:
    """Every SAM3 / Molmo / MoGe-2 replica of SOARM_SAM3_URLS / SOARM_MOLMO_URLS / SOARM_DEPTH_URLS must answer before a
    batch starts."""
    import urllib.request
    from soarm import perception
    sam3 = (os.environ.get("SOARM_SAM3_URLS") or os.environ.get("SAM3_URL") or "http://127.0.0.1:8114").split(",")
    molmo = (os.environ.get("SOARM_MOLMO_URLS") or os.environ.get("MOLMO_URL") or "http://127.0.0.1:8122/v1").split(",")
    for url in [u.strip().rstrip("/") + "/openapi.json" for u in sam3 if u.strip()] + \
               [u.strip().rstrip("/") + "/models" for u in molmo if u.strip()]:
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                resp.read()
        except OSError as exc:
            raise SystemExit(f"real perception service not answering: {url}: {exc}") from exc
    depth = (os.environ.get("SOARM_DEPTH_URLS") or os.environ.get("CAPTAIN_DEPTH_URL") or "http://127.0.0.1:8130").split(",")
    for url in (u.strip() for u in depth if u.strip()):
        try:
            perception.require_depth_healthy(url)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc


PRUNE = ("public_sensor_artifacts", "observations", "public_measurements")


def prune(ep: Path) -> None:
    """Drop the bulky host-side sensor stores (up to 0.5 GB per episode: depth/point sqlite, raw observations). SFT export
    needs driver/, workspace/images, soarm/task.json, episode_result.json; RL critics need hidden_*.jsonl and calls.jsonl."""
    import shutil
    for name in PRUNE:
        shutil.rmtree(ep / name, ignore_errors=True)


def family_list(text: str) -> list[str]:
    """Spec ids: legacy (the 16 pre-pivot families) | cube (train cube specs, the teacher default) | cube_heldout |
    comma-separated spec ids (legacy specs are named after their family)."""
    from soarm import taskspec
    groups = {"legacy": taskspec.select("legacy"), "all": taskspec.select("legacy"),
              "cube": taskspec.select("cube", "train"), "cube_heldout": taskspec.select("cube", "heldout")}
    ids = [s["id"] for s in groups[text]] if text in groups else text.split(",")
    unknown = [f for f in ids if f not in taskspec.registry()]
    if unknown:
        raise SystemExit(f"unknown task specs: {unknown}")
    return ids


def seed_range(machine: str | None, kind: str) -> tuple[int, int] | None:
    from soarm.taskspec import SEED_RANGES
    if not machine:
        return None
    if machine not in SEED_RANGES:
        raise SystemExit(f"unknown machine {machine!r} (soarm.taskspec.SEED_RANGES)")
    return SEED_RANGES[machine][kind]


def main() -> None:
    from soarm import taskspec
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--policy-url", default="http://127.0.0.1:8650/v1",
                    help="comma-separated replicas: each episode goes to the one with the fewest running episodes")
    ap.add_argument("--stagger-s", type=float, default=0.0,
                    help="minimum gap between episode starts (spreads env-server / render context creation)")
    ap.add_argument("--families", default="cube", help="legacy | cube | cube_heldout | comma-separated spec ids")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--plan", default=None, help="jsonl of {family: <spec id>, seed}")
    ap.add_argument("--suite", default=None, help="fixed eval suite (soarm/specs/suites/<name>.json): its jobs, eval seeds "
                                                  "and scene config; summarise with tools/eval_summary.py")
    ap.add_argument("--machine", default=os.environ.get("SOARM_MACHINE"),
                    help="seed range owner (CONTRACT 1, taskspec.SEED_RANGES); seeds outside its range are refused")
    ap.add_argument("--seed-kind", choices=("data", "eval"), default="data")
    ap.add_argument("--lanes", type=int, default=3)
    ap.add_argument("--wall-time", type=int, default=3600,
                    help="per-episode wall budget (s); 100 requests at ~12 s plus 429 backoff do not fit in 2400")
    ap.add_argument("--stop-file", default=None)
    ap.add_argument("--max-consecutive-errors", type=int, default=6)
    ap.add_argument("--target-per-family", type=int, default=0,
                    help="adaptive mode: launch families in proportion to their strict yield (see pick()) until every "
                         "family has this many strict successes; seeds count up from --seed-start per family")
    ap.add_argument("--seed-start", type=int, default=200)
    ap.add_argument("--max-attempts-per-family", type=int, default=0, help="adaptive mode cap (0 = 4 x target)")
    ap.add_argument("--mix", default=None,
                    help="training mix (soarm/specs/mixes/<name>.json or a path): its tasks, per-task strict targets "
                         "(--target-total x weight), scene config and robot intro; adaptive like --target-per-family, "
                         "plus the mix's difficulty floor (tasks retired below min_yield after min_attempts)")
    ap.add_argument("--target-total", type=int, default=0, help="--mix: strict successes over all tasks")
    ap.add_argument("--intro", default=None, help="robot intro version (episode.INTROS) instead of the mix's; v5 = the "
                                                   "compact_v1 prompt profile")
    ap.add_argument("--quota-flag", default=os.path.join(os.environ.get("LOG_DIR", "."), "QUOTA_EXCEEDED"),
                    help="written by gateway_proxy on usage_limit_exceeded; no new episodes while it exists")
    a = ap.parse_args()
    if a.intro:
        from soarm.episode import INTROS
        if a.intro not in INTROS:
            raise SystemExit(f"--intro {a.intro!r} not in {sorted(INTROS)}")
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    from soarm import perception
    if perception.mode() != "real":
        raise SystemExit("batches run real SAM3/Molmo perception only (SOARM_PERCEPTION=real)")
    if perception.depth_mode() != "moge2":
        raise SystemExit("batches estimate depth with MoGe-2 only (SOARM_DEPTH=moge2); sensor depth is for tests/ablations")
    check_services()
    print(json.dumps({"perception": perception.mode(), "sam3_urls": os.environ.get("SOARM_SAM3_URLS") or os.environ.get("SAM3_URL"),
                      "molmo_urls": os.environ.get("SOARM_MOLMO_URLS") or os.environ.get("MOLMO_URL"),
                      "depth": perception.depth_mode(), "depth_revision": os.environ.get("SOARM_DEPTH_REVISION")
                      or perception.MOGE2_REVISION,
                      "depth_urls": os.environ.get("SOARM_DEPTH_URLS") or os.environ.get("CAPTAIN_DEPTH_URL")}), flush=True)
    scene_config, mix, jobs = None, None, []
    adaptive = bool(a.target_per_family or a.mix)
    if a.mix:
        from soarm.mix import load as load_mix
        if not a.target_total:
            raise SystemExit("--mix needs --target-total")
        mix = load_mix(a.mix)
        scene_config = json.dumps(mix["config"].to_json())
    elif a.suite:
        s = taskspec.suite(a.suite)
        jobs, a.seed_kind, scene_config = s["jobs"], "eval", json.dumps(s["config"].to_json())
    elif a.plan:
        jobs = [(r["family"], int(r["seed"])) for r in map(json.loads, Path(a.plan).read_text().splitlines()) if r]
    elif not adaptive:
        fams = family_list(a.families)
        jobs = [(f, s) for s in parse_range(a.seeds) for f in fams]
    # suites are fixed eval sets with the same seeds on every machine; ranges guard data and ad-hoc runs (adaptive
    # seeds count up from --seed-start; run() refuses any that leaves the range)
    allowed = None if a.suite else seed_range(a.machine, a.seed_kind)
    if allowed and any(not allowed[0] <= s <= allowed[1] for _, s in jobs):
        raise SystemExit(f"seeds outside {a.machine} {a.seed_kind} range {allowed}")
    if allowed and adaptive and not allowed[0] <= a.seed_start <= allowed[1]:
        raise SystemExit(f"--seed-start {a.seed_start} outside {a.machine} {a.seed_kind} range {allowed}")
    log = out / "batch.jsonl"
    lock = threading.Lock()
    state = {"consecutive_errors": 0, "stopped": False, "next_start": 0.0}
    policy_load = {u.strip(): 0 for u in a.policy_url.split(",") if u.strip()}

    def run(job):
        family, seed = job
        tid = taskspec.get(family)["tid"]
        ep = out / f"soarm_sim_{tid}_seed-{seed}"
        result = ep / "episode_result.json"
        if result.exists() and json.loads(result.read_text()).get("termination_reason") != "infrastructure_error":
            return None
        if ep.exists():   # killed by a restart or a server crash (no result / infrastructure_error): not an attempt;
            import shutil   # rerun the seed in a clean dir (a stale server.json hands the new driver a dead server's URL)
            shutil.rmtree(ep, ignore_errors=True)
        if allowed and not allowed[0] <= seed <= allowed[1]:
            with lock:
                state["stopped"] = True
            print(json.dumps({"family": family, "seed": seed, "batch_stopped": f"seed outside {allowed}"}), flush=True)
            return None
        if state["stopped"] or (a.stop_file and Path(a.stop_file).exists()) or Path(a.quota_flag).exists():
            return None
        with lock:
            start = max(time.time(), state["next_start"])
            state["next_start"] = start + a.stagger_s
            url = min(policy_load, key=policy_load.get)
            policy_load[url] += 1
        time.sleep(max(0.0, start - time.time()))
        t0 = time.time()
        env = dict(os.environ, POLICY_URL=url, POLICY_MODEL=a.model, WALL_TIME=str(a.wall_time))
        if scene_config:
            env["SOARM_SCENE_CONFIG"] = scene_config
        if a.intro or (mix and mix["intro"]):
            env["SOARM_INTRO"] = a.intro or mix["intro"]
        with (out / f"driver_{tid}_{seed}.log").open("w") as fh:
            try:
                subprocess.run(["bash", str(HERE / "run_episode.sh"), str(tid), str(seed), str(out)], env=env,
                               stdout=fh, stderr=subprocess.STDOUT)
            finally:
                with lock:
                    policy_load[url] -= 1
        row = {"family": family, "seed": seed, "model": a.model, "policy_url": url, "wall_s": round(time.time() - t0, 1)}
        try:
            row.update(summarize(ep))
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
        prune(ep)
        violation = (perception_violation(row) if ep.exists() and "error" not in row
                     and row.get("termination") != "infrastructure_error" else None)
        if violation:   # never data: moved out of the run dir, row kept apart
            import shutil
            qdir = out.parent / f"quarantine_{out.name}"
            qdir.mkdir(exist_ok=True)
            shutil.move(str(ep), str(qdir / ep.name))
            row["perception_violation"] = violation
            with lock, (qdir / "quarantine.jsonl").open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            return None
        if (ep / "episode_result.json").exists():   # the episode's own record states its perception
            res = json.loads((ep / "episode_result.json").read_text())
            res["soarm_perception"] = {k: row.get(k) for k in ("perception_mode", "perception_tool_revisions",
                                                                "perception_real_requests", "perception_privileged_calls",
                                                                "depth_source", "depth_revision", "depth_requests")}
            res["soarm_prompt"] = {k: row.get(k) for k in ("intro", "prompt_profile", "task_md_version", "task_md_sha256")}
            (ep / "episode_result.json").write_text(json.dumps(res, indent=1))
        flag = Path(a.quota_flag)
        if flag.exists() and flag.stat().st_mtime >= t0:
            # the gateway's daily cost quota cut this episode short: discard it so a resumed batch reruns the seed
            import shutil
            shutil.rmtree(ep, ignore_errors=True)
            row["quota_affected"] = True
            with lock, (out / "quota_discarded.jsonl").open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            return None
        if row.get("termination") == "infrastructure_error":
            # env server crash / batch restart: not an attempt; keep the seed rerunnable, stop if it keeps happening
            import shutil
            shutil.rmtree(ep, ignore_errors=True)
            with lock:
                state["consecutive_errors"] += 1
                if state["consecutive_errors"] >= a.max_consecutive_errors and not state["stopped"]:
                    state["stopped"] = True
                    print(json.dumps({"family": family, "seed": seed, "batch_stopped":
                                      f"{state['consecutive_errors']} consecutive infrastructure errors"}), flush=True)
                with (out / "infra_discarded.jsonl").open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
            return None
        model_error = "error" in row or str(row.get("termination", "")).startswith(("model_error", "request_failed"))
        with lock:
            state["consecutive_errors"] = state["consecutive_errors"] + 1 if model_error else 0
            if state["consecutive_errors"] >= a.max_consecutive_errors:
                state["stopped"] = True
                row["batch_stopped"] = "too many consecutive model/infra errors"
            with log.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        return row

    if not adaptive:
        with cf.ThreadPoolExecutor(a.lanes) as pool:
            list(pool.map(run, jobs))
        return

    # adaptive mode: per-family strict targets (--target-per-family) or the mix's per-task targets (--mix)
    if mix:
        from soarm.mix import targets as mix_targets
        target = mix_targets(mix["weights"], a.target_total)
    else:
        target = {f: a.target_per_family for f in family_list(a.families)}
    fams = list(target)
    strict = {f: 0 for f in fams}
    attempts = {f: 0 for f in fams}
    next_seed = {f: a.seed_start for f in fams}
    if log.exists():   # resume: count what is already done
        for r in map(json.loads, log.read_text().splitlines()):
            if r.get("family") in strict:
                next_seed[r["family"]] = max(next_seed[r["family"]], int(r["seed"]) + 1)
                if r.get("termination") == "infrastructure_error":
                    continue   # rows written before infra episodes were discarded: not attempts
                attempts[r["family"]] += 1
                strict[r["family"]] += bool(r.get("success") and r.get("finished") and not r.get("false_finish"))
    cap = {f: a.max_attempts_per_family or 4 * target[f] for f in fams}
    running = {f: 0 for f in fams}
    retired: set[str] = set()

    def pick():
        open_ = [f for f in fams if strict[f] + running[f] < target[f] and attempts[f] + running[f] < cap[f]
                 and f not in retired]
        if not open_:
            return None
        p = {f: (strict[f] + 1) / (attempts[f] + 2) for f in open_}
        if not mix:
            # Fill-to-target by strict count alone sends every lane to the 0%-yield families until their attempt cap.
            # Weight by the Laplace success estimate instead: strict successes accrue roughly in proportion to a
            # family's yield, and the 0.05 floor keeps hard families sampled (one start per 5-10 on an easy family).
            return min(open_, key=lambda f: ((strict[f] + running[f] + 1) / max(p[f], 0.05), attempts[f]))
        # Mix: fill every task toward its share (expected strict count incl. running episodes / target), favouring
        # high-yield tasks (x0.5-2 of the mean Laplace yield) so the teacher budget goes where successes are cheap.
        mean = sum(p.values()) / len(p)
        return min(open_, key=lambda f: ((strict[f] + running[f] * p[f] + 1) / target[f]
                                         / min(2.0, max(0.5, p[f] / mean)), attempts[f]))

    def retire(f):
        # difficulty floor: a task whose teacher yield stays below min_yield after min_attempts gets no more starts
        r = mix["retire"]
        if attempts[f] >= r["min_attempts"] and (strict[f] + 1) / (attempts[f] + 2) < r["min_yield"] \
                and f not in retired:
            retired.add(f)
            print(json.dumps({"family": f, "retired": f"{strict[f]}/{attempts[f]} strict < {r['min_yield']}"}),
                  flush=True)

    if mix:
        for f in fams:
            retire(f)

    with cf.ThreadPoolExecutor(a.lanes) as pool:
        futures = {}
        while True:
            stop = state["stopped"] or (a.stop_file and Path(a.stop_file).exists()) or Path(a.quota_flag).exists()
            while not stop and len(futures) < a.lanes:
                with lock:
                    f = pick()
                    if f is None:
                        break
                    seed = next_seed[f]
                    next_seed[f] += 1
                    running[f] += 1
                futures[pool.submit(run, (f, seed))] = f
            if not futures:
                break
            done, _ = cf.wait(futures, return_when=cf.FIRST_COMPLETED)
            for fut in done:
                f = futures.pop(fut)
                row = fut.result()
                with lock:
                    running[f] -= 1
                    if row is not None:
                        attempts[f] += 1
                        strict[f] += bool(row.get("success") and row.get("finished") and not row.get("false_finish"))
                        if mix:
                            retire(f)


if __name__ == "__main__":
    main()
