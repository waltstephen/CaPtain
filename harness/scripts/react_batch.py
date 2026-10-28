#!/usr/bin/env python3
"""Batch runner for the ReAct driver (one seed, a list of LIBERO tasks).

    .venv-libero/bin/python scripts/react_batch.py --model captain-4b --api chat --endpoint http://127.0.0.1:8600/v1 \
        --auth none --profile captain --inline-task --seed 11 --max-concurrency 8 --episode-arg=...   (see README.md)

Default task set: community-40 (libero_spatial/object/goal/10 x tasks 0-9), seed 11, one episode each,
resumable (episodes with episode_result.json are skipped). Concurrency: <= 6 while the 1-minute load
average is > 40, else <= 12, capped by --max-concurrency. Infrastructure failures
are retried once. Output: <root>/outputs/<experiment_id>/episodes/<suite>_<task>_seed-<seed>/ plus
results.jsonl / summary.json / summary.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
from summarize_rows import write_summary  # noqa: E402

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
REPO_ROOT = HERE.parent


def community40() -> list[tuple[str, int]]:
    return [(suite, task) for suite in SUITES for task in range(10)]


def global_running_episodes() -> int:
    try:
        out = subprocess.run(["pgrep", "-fc", "react_episode.py --suite"], capture_output=True, text=True)
        return int(out.stdout.strip() or 0)
    except Exception:
        return 0


def allowed_concurrency(cap: int, own_running: int, global_cap: int | None = None) -> int:
    """Slots this batch may fill now: machine-wide rule (6 while load > 40, else 12, or --global-cap) minus everyone's
    running episodes, and this batch's own cap minus its running episodes."""
    load = os.getloadavg()[0]
    rule = global_cap if global_cap is not None else (6 if load > 40 else 12)
    return max(0, min(rule - global_running_episodes(), cap - own_running))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--driver", choices=("react",), default="react")
    p.add_argument("--model", required=True)
    p.add_argument("--reasoning-effort", default=None)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--experiment-id", default=None)
    p.add_argument("--root", type=Path, default=REPO_ROOT)
    p.add_argument("--max-concurrency", type=int, default=None, help="default 8 (load rule applies on top)")
    p.add_argument("--global-cap", type=int, default=None, help="machine-wide running-episode ceiling counted across all users/batches (default: 6 if load>40 else 12)")
    p.add_argument("--launch-interval", type=float, default=10.0, help="seconds between launches (EGL context bursts fail)")
    p.add_argument("--tasks", default=None, help="comma list like libero_spatial:0,libero_10:3 (default community-40)")
    p.add_argument("--retry-infra", type=int, default=1)
    p.add_argument("--wall-time", type=float, default=2400.0)
    p.add_argument("--min-free-gib", type=float, default=20.0)
    p.add_argument("--mujoco-gl", choices=("egl", "osmesa"), default=os.environ.get("MUJOCO_GL", "egl"))
    p.add_argument("--egl-device", type=int, default=None)
    p.add_argument("--endpoint", default=None, help="chat: OpenAI-compatible base URL (http://host:port/v1); responses: Responses URL")
    p.add_argument("--auth", choices=("none", "bearer"), default=None)
    p.add_argument("--api", choices=("responses", "chat"), default=None, help="react driver: Responses API (default) or Chat Completions (vLLM students)")
    p.add_argument("--profile", default=None, help="react driver: borrow this catalog profile (template + tool mode) for models outside the catalog")
    p.add_argument("--inline-task", action="store_true", help="react driver: inline TASK.md and the initial frames into the first request")
    p.add_argument("--episode-arg", action="append", default=[], help="extra argument forwarded to the episode script (repeatable)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cap = args.max_concurrency if args.max_concurrency is not None else 8
    model_tag = args.model.replace("/", "_")
    experiment_id = args.experiment_id or f"eval-{args.driver}-{model_tag}-community40-s{args.seed}-{dt.date.today():%Y%m%d}"
    root = args.root.resolve() / "outputs" / experiment_id
    episodes_root = root / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=True)
    rows = [(s, int(t)) for s, t in (item.split(":") for item in args.tasks.split(","))] if args.tasks else community40()
    (root / "batch_config.json").write_text(json.dumps({
        "experiment_id": experiment_id, "driver": args.driver, "model": args.model, "reasoning_effort": args.reasoning_effort,
        "seed": args.seed, "rows": rows, "max_concurrency_cap": cap, "launch_interval_s": args.launch_interval,
        "wall_time_s": args.wall_time, "root": str(args.root.resolve()), "mujoco_gl": args.mujoco_gl, "endpoint": args.endpoint,
        "auth": args.auth, "api": args.api, "profile": args.profile, "inline_task": args.inline_task,
        "episode_args": args.episode_arg, "started": dt.datetime.now().isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")
    log_path = root / "batch.log"

    def log(message: str) -> None:
        line = f"{dt.datetime.now():%Y-%m-%dT%H:%M:%S} {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def episode_dir(suite: str, task: int) -> Path:
        return episodes_root / f"{suite}_{task}_seed-{args.seed}"

    pending: list[tuple[str, int]] = []
    retries: dict[tuple[str, int], int] = {}
    for suite, task in rows:
        result_path = episode_dir(suite, task) / "episode_result.json"
        if episode_dir(suite, task).exists() and not result_path.is_file():
            stamp = dt.datetime.now().strftime("%H%M%S")
            shutil.rmtree(episode_dir(suite, task) / "public_sensor_artifacts", ignore_errors=True)
            shutil.move(str(episode_dir(suite, task)), str(episode_dir(suite, task)) + f".partial-{stamp}")
        if result_path.is_file():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                result = {}
            if result.get("termination_reason") == "infrastructure_error" and args.retry_infra > 0:
                stamp = dt.datetime.now().strftime("%H%M%S")
                shutil.move(str(episode_dir(suite, task)), str(episode_dir(suite, task)) + f".infra-{stamp}")
                retries[(suite, task)] = 1
                pending.append((suite, task))
            continue
        pending.append((suite, task))
    log(f"{experiment_id}: {len(rows)} rows, {len(pending)} pending, cap {cap}, load {os.getloadavg()[0]:.1f}")
    if args.dry_run:
        for suite, task in pending:
            print(suite, task)
        return 0

    python = os.environ.get("CAPTAIN_PYTHON") or str(args.root.resolve() / ".venv-libero" / "bin" / "python")
    script = HERE / "react_episode.py"
    running: dict[tuple[str, int], subprocess.Popen] = {}
    last_launch = 0.0
    while pending or running:
        for key, proc in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            del running[key]
            suite, task = key
            result_path = episode_dir(suite, task) / "episode_result.json"
            result = {}
            if result_path.is_file():
                try:
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    pass
            log(f"done {suite}:{task} rc={rc} term={result.get('termination_reason')} success={result.get('host_final_success')} "
                f"gate_rejections={result.get('gate_rejections')} actions={result.get('action_requests')} wall={result.get('wall_time_s')}")
            if result.get("termination_reason", "infrastructure_error") == "infrastructure_error" and retries.get(key, 0) < args.retry_infra:
                retries[key] = retries.get(key, 0) + 1
                stamp = dt.datetime.now().strftime("%H%M%S")
                if episode_dir(suite, task).exists():
                    shutil.move(str(episode_dir(suite, task)), str(episode_dir(suite, task)) + f".infra-{stamp}")
                pending.append(key)
                log(f"retry {suite}:{task} (attempt {retries[key] + 1})")
            write_summary(root)
        slots = allowed_concurrency(cap, len(running), args.global_cap)
        free_gib = shutil.disk_usage(str(root)).free / 2**30
        if slots > 0 and free_gib < args.min_free_gib:
            slots = 0
            if time.monotonic() - last_launch >= 120:
                log(f"waiting: only {free_gib:.1f} GiB free on disk (< {args.min_free_gib})")
                last_launch = time.monotonic()
        if pending and slots > 0 and time.monotonic() - last_launch >= args.launch_interval:
            suite, task = pending.pop(0)
            out = episode_dir(suite, task)
            out.mkdir(parents=True, exist_ok=True)
            cmd = [python, str(script), "--suite", suite, "--task-id", str(task), "--seed", str(args.seed), "--model", args.model,
                   "--output-dir", str(out), "--root", str(args.root.resolve()), "--wall-time", str(args.wall_time), "--mujoco-gl", args.mujoco_gl]
            if args.reasoning_effort:
                cmd += ["--reasoning-effort", args.reasoning_effort]
            if args.egl_device is not None:
                cmd += ["--egl-device", str(args.egl_device)]
            if args.endpoint:
                cmd += ["--endpoint", args.endpoint]
            if args.auth:
                cmd += ["--auth", args.auth]
            if args.driver == "react" and args.api:
                cmd += ["--api", args.api]
            if args.driver == "react" and args.profile:
                cmd += ["--profile", args.profile]
            if args.driver == "react" and args.inline_task:
                cmd += ["--inline-task"]
            cmd += list(args.episode_arg)
            env = dict(os.environ)
            env["PYTHONPATH"] = str(args.root.resolve() / "src")
            driver_log = (out / "driver.log").open("ab")
            proc = subprocess.Popen(cmd, stdout=driver_log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True, env=env, cwd=str(args.root.resolve()))
            running[(suite, task)] = proc
            last_launch = time.monotonic()
            log(f"launched {suite}:{task} pid={proc.pid} running={len(running)}/{cap} global={global_running_episodes()} load={os.getloadavg()[0]:.1f}")
        time.sleep(2.0)
    summary = write_summary(root)
    log(f"batch complete: success {summary['success']}/{summary['scored']} false_finish {summary['false_finish']} infra {summary['infrastructure_errors']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
