"""Smoke test of the soarm_sim env server through the agent's own client (robot.py): observe, segment, measure,
move_to, jaw, finish. Starts the server like the ReAct driver does (episode_launch.start_server).

    python tools/server_smoke.py --family put_in_container --seed 0 --out /tmp/soarm_smoke
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

HARNESS = Path(os.environ["HARNESS_DIR"])          # set by env.sh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="cube_put_in_container", help="task spec id or tid")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/soarm_smoke")
    a = ap.parse_args()
    from captain_harness.server.episode_launch import start_server, wait_for_server
    from soarm import taskspec
    out = Path(a.out)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    proc = start_server(HARNESS, out, suite="soarm_sim", task_id=taskspec.get(a.family)["tid"], seed=a.seed,
                        images_dir=out / "workspace" / "images", max_actions=150, max_tool_calls=600, wall_time_s=1080,
                        mujoco_gl=os.environ.get("SOARM_GL", "osmesa"), extra_args=["--receipts"])
    ok = False
    try:
        info = wait_for_server(out, 120, proc)
        os.environ["ROBOT_SERVER_URL"] = info["url"]
        sys.path.insert(0, str(HARNESS / "src" / "captain_harness" / "server"))
        import robot
        print("task:", info.get("task"))
        print("caps:", robot.capabilities()[:40] if hasattr(robot, "capabilities") else "?")
        obs = robot.call("observe")
        print("observe:", obs["success"], obs.get("message", "")[:120], [i.get("path") for i in obs["value"]["images"]])
        handle = obs["value"]["observation_handle"]
        detected = 0
        for prompt in ("bowl", "box", "cube", "ball", "cylinder", "block"):
            r = robot.call("segment", prompt=prompt, observation_handle=handle)
            v = r.get("value") or {}
            detected += bool(r["success"])
            print(f"segment({prompt!r}):", r["success"], r.get("failure_category"), str(r.get("message"))[:100],
                  json.dumps({k: v.get(k) for k in ("prompt", "reason")})[:200], len(v.get("detections") or []))
        state = robot.call("get_robot_state")
        print("robot state:", json.dumps(state.get("value"))[:300])
        fin = robot.finish("smoke")
        print("finish:", fin)
        ok = bool(obs["success"]) and detected > 0 and bool(state.get("success")) and bool(fin.get("finished"))
    finally:
        proc.terminate()
    print("server log tail:")
    print("\n".join((out / "server.log").read_text().splitlines()[-15:]) if (out / "server.log").exists() else "(none)")
    print("SMOKE OK" if ok else "SMOKE FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
