#!/usr/bin/env python3
"""Smoke the V1.1 environment server: build one episode, render the workspace, finish without acting.

    .venv-libero/bin/python scripts/env_server_dryrun.py --suite libero_spatial --task-id 0 --seed 11 \
        --output-dir outputs/dryrun-spatial0 --mujoco-gl osmesa
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from captain_harness.server.episode_launch import (  # noqa: E402
    REPO_ROOT, build_workspace, http_json, start_server, stop_server, wait_for_server,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--root", type=Path, default=REPO_ROOT)
    p.add_argument("--mujoco-gl", choices=("egl", "osmesa"), default="egl")
    p.add_argument("--egl-device", type=int, default=None)
    p.add_argument("--observe", action="store_true", help="also call observe() once through the API")
    args = p.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "workspace" / "images").mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    server = start_server(args.root, out, suite=args.suite, task_id=args.task_id, seed=args.seed,
                          images_dir=out / "workspace" / "images", max_actions=150, max_tool_calls=600,
                          wall_time_s=2400.0, mujoco_gl=args.mujoco_gl, egl_device=args.egl_device)
    url = None
    try:
        info = wait_for_server(out, 600.0, server)
        url = info["url"]
        manifest = http_json(url + "/manifest")
        workspace = build_workspace(out, info, manifest, wall_time_s=2400.0, max_actions=150, max_tool_calls=600)
        task_md = (workspace / "TASK.md").read_text(encoding="utf-8")
        print(f"server up in {time.monotonic() - t0:.0f}s at {url}; task={info['task']!r} init_state={info.get('init_state_index')}")
        print(f"TASK.md {len(task_md)} chars, {len(manifest['manifest']['capabilities'])} capabilities: "
              f"{[c['name'] for c in manifest['manifest']['capabilities']]}")
        if args.observe:
            result = http_json(url + "/call", "POST", {"name": "observe", "arguments": {}})
            print("observe:", result["success"], [i["path"] for i in (result.get("value") or {}).get("images", [])])
        gate = http_json(url + "/finish", "POST", {"note": "dry run", "source": "agent"})
        print("agent finish ->", json.dumps(gate)[:300])
        if not gate.get("finished"):
            fin = http_json(url + "/finish", "POST", {"note": "dry run", "source": "driver"})
            print("driver finish ->", json.dumps(fin)[:200])
        result = json.loads((out / "result.json").read_text(encoding="utf-8"))
        print("result.json:", {k: result.get(k) for k in ("host_final_success", "gate_rejections", "action_requests", "tool_requests", "wall_time_s")})
    finally:
        stop_server(server, url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
