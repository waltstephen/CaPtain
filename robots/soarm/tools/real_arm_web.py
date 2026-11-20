"""Web control of real-arm episodes on armhost: http://<host>:8095/

  Start  runs tools/run_real_episode.sh with the instruction in the text box (one episode at a time); the episode
         moves the arm home, runs the policy, then parks at the standard rest pose with torque off.
  Stop   kills the running episode and parks the arm (tools/real_arm_stop.sh).
  Reset  parks the arm at the standard rest pose (configs/hardware/so101_armhost_real.json standard_rest_raw); refused
         while an episode runs.
Camera snapshots come from tools/camstream.py (:8090). Run: source env.sh && python tools/real_arm_web.py [port]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

CODE = Path(os.environ["SOARM_CODE"])
RUNS = Path(os.environ.get("SOARM_REAL_RUNS", os.path.expandvars("$SOARM_ROOT/speed/real_runs")))
CAMSTREAM = "http://127.0.0.1:8090"
DEFAULT_TASK = "Put the blue cube on the yellow sponge."

_lock = threading.Lock()
_job: dict | None = None      # {"kind", "proc", "dir", "log", "t0", "task"}
_last: dict | None = None     # the last finished job, same keys + "rc"


def _running() -> dict | None:
    global _job, _last
    with _lock:
        if _job is not None and _job["proc"].poll() is not None:
            _last = {**_job, "rc": _job["proc"].returncode, "t1": time.time()}
            _job = None
        return _job


def _launch(kind: str, cmd: list[str], log: Path, **extra) -> dict:
    global _job
    log.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(cmd, cwd=CODE, stdout=log.open("w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=True)
    with _lock:
        _job = {"kind": kind, "proc": proc, "log": log, "t0": time.time(), **extra}
    return _job


def start(task: str) -> tuple[int, dict]:
    if _running():
        return 409, {"error": f"{_job['kind']} is running"}
    task = " ".join(task.split()) or DEFAULT_TASK
    out = RUNS / time.strftime("web_%m%d_%H%M%S")
    _launch("episode", ["bash", "tools/run_real_episode.sh", task, str(out)], Path(f"{out}.log"), dir=out, task=task)
    return 200, {"started": str(out), "task": task}


def reset() -> tuple[int, dict]:
    if _running():
        return 409, {"error": f"{_job['kind']} is running: stop it first"}
    _launch("reset", [sys.executable, "tools/real_arm_park.py"], RUNS / "web_reset.log")
    return 200, {"reset": "parking at the standard rest pose"}


def stop() -> tuple[int, dict]:
    global _job, _last
    job = _running()
    if job is not None and job["kind"] != "episode":
        return 409, {"error": f"{job['kind']} is running"}
    with _lock:
        if _job is not None:   # the stop script kills the episode, then parks the arm
            _last = {**_job, "rc": "stopped", "t1": time.time()}
            _job = None
    _launch("stop", ["bash", "tools/real_arm_stop.sh"], RUNS / "web_stop.log")
    return 200, {"stop": "episode killed, parking"}


def _episode_summary(out: Path) -> dict:
    calls = []
    path = out / "calls.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("name") in ("get_robot_state", "observe", "robot_geometry"):
                continue
            calls.append(f"{row.get('t_s', 0):6.0f}s  {(row.get('receipt') or row.get('name') or '')[:170]}")
    result = None
    if (out / "episode_result.json").exists():
        r = json.loads((out / "episode_result.json").read_text())
        result = {k: r.get(k) for k in ("termination_reason", "agent_finished", "wall_time_s", "action_requests",
                                        "tool_requests")}
    return {"calls": calls[-25:], "n_calls": len(calls), "result": result}


def status() -> dict:
    job = _running()
    show = job or _last
    out = {"running": None if job is None else job["kind"], "last": None}
    if show is not None:
        info = {"kind": show["kind"], "elapsed_s": round((show.get("t1") or time.time()) - show["t0"]),
                "rc": show.get("rc"), "task": show.get("task"), "dir": str(show.get("dir") or "")}
        if show.get("dir"):
            info.update(_episode_summary(Path(show["dir"])))
        try:
            info["log_tail"] = show["log"].read_text(errors="replace").splitlines()[-4:]
        except OSError:
            info["log_tail"] = []
        out["last"] = info
    return out


# training placement annulus of the twin (soarm/tasks.py R_MIN, R_MAX, ANG_MAX): objects outside it are out of the
# policy's distribution, and beyond ~0.29 m top-down grasps are out of reach
REACH_R = (0.13, 0.29)
REACH_ANG_DEG = 70.0


def _fixed_camera():
    cfg = json.loads((CODE / "configs/hardware/so101_armhost_real.json").read_text())
    cams = json.loads(Path(os.path.expandvars(cfg["calibration"]["cameras"])).read_text())["fixed"]
    T = np.linalg.inv(np.array(cams["T_base_cam"], float))
    return np.array(cams["K"], float), np.array(cams["dist"], float), T


def draw_reach(img: np.ndarray) -> None:
    """Draw the reach annulus (table plane z = 0) on a full-resolution head-camera frame."""
    K, dist, T = _fixed_camera()
    a = np.deg2rad(np.linspace(-REACH_ANG_DEG, REACH_ANG_DEG, 60))
    arcs = [np.c_[r * np.cos(a), r * np.sin(a)] for r in REACH_R]
    ring = np.r_[arcs[0], arcs[1][::-1], arcs[0][:1]]
    cam = (T[:3, :3] @ np.c_[ring, np.zeros(len(ring))].T + T[:3, 3:4]).T
    px, _ = cv2.projectPoints(cam, np.zeros(3), np.zeros(3), K, dist)
    poly = px.reshape(-1, 2).astype(np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [poly], (0, 200, 0))
    cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
    cv2.polylines(img, [poly], True, (0, 220, 0), 3)


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>SO-101 real arm</title><style>
body{font-family:sans-serif;margin:16px;background:#111;color:#ddd} button{font-size:18px;padding:8px 22px;margin-right:8px}
#task{width:520px;font-size:16px;padding:6px} pre{background:#1b1b1b;padding:8px;white-space:pre-wrap;font-size:12px;max-height:420px;overflow:auto}
img{width:49%;margin-right:1%} .run{color:#7f7} .err{color:#f77}</style></head><body>
<h3>SO-101 real arm (armhost) &middot; policy r1113c fast</h3>
<div><input id="task" value="__TASK__"> <button onclick="post('start',{task:task.value})">Start</button>
<button onclick="post('stop',{})">Stop</button><button onclick="post('reset',{})">Reset</button></div>
<p id="msg"></p><p style="font-size:13px">green band on the head camera = training placement range (13-29 cm from the base, +-70 deg): keep objects inside it</p><div><img id="c2"><img id="c0"></div><pre id="st"></pre>
<script>
async function post(a,b){const r=await fetch('/api/'+a,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});
 const j=await r.json(); msg.textContent=JSON.stringify(j); msg.className=r.ok?'run':'err'; poll();}
async function poll(){try{const j=await (await fetch('/api/status')).json(); let s='running: '+(j.running||'-')+'\\n';
 const l=j.last; if(l){s+=`${l.kind}  ${l.elapsed_s}s  rc=${l.rc??'-'}  ${l.task||''}\\n${l.dir||''}\\n`;
 if(l.result) s+='result: '+JSON.stringify(l.result)+'\\n'; if(l.calls) s+=`\\n${l.n_calls} tool calls, last:\\n`+l.calls.join('\\n')+'\\n';
 s+='\\nlog: '+(l.log_tail||[]).join('\\n     ');} st.textContent=s;}catch(e){}}
function cams(){const t=Date.now(); c2.src='/snap/2?'+t; c0.src='/snap/0?'+t;}
setInterval(poll,1000); setInterval(cams,500); poll(); cams();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json")

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            self._send(200, PAGE.replace("__TASK__", DEFAULT_TASK).encode(), "text/html; charset=utf-8")
        elif path == "/api/status":
            self._json(200, status())
        elif path.startswith("/snap/"):
            try:
                png = urllib.request.urlopen(CAMSTREAM + path, timeout=5).read()
                img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)   # small JPEG for the relays
                if path == "/snap/2":
                    draw_reach(img)
                ok, jpg = cv2.imencode(".jpg", cv2.resize(img, (640, 360)), [cv2.IMWRITE_JPEG_QUALITY, 75])
                self._send(200, jpg.tobytes(), "image/jpeg")
            except OSError as exc:
                self._json(502, {"error": str(exc)})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        action = self.path.split("?")[0]
        if action == "/api/start":
            self._json(*start(str(body.get("task") or DEFAULT_TASK)))
        elif action == "/api/stop":
            self._json(*stop())
        elif action == "/api/reset":
            self._json(*reset())
        else:
            self._json(404, {"error": "not found"})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8095
    print(f"real-arm control on http://0.0.0.0:{port}/", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
