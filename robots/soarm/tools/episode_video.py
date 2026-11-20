"""1x real-arm-timeline video of a recorded SO-101 twin episode (soarm-speed; needs SOARM_RECORD_FRAMES=1).

    python tools/episode_video.py <episode dir> <out.mp4> [--fps 30] [--crf 28]

Timeline (real-arm equivalent, the same accounting as tools/speed_eval.py real_wall_s):
  - start-up (episode start -> first model request, without the recorder's rendering): held first frame, or 0 s with
    --prewarmed (J1: robot server, sandbox, services and calibration are up before the user's "go");
  - every model request: held frame for its measured latency, overlay "request k/N · thinking+answer X s";
  - between requests (tool execution): every recorded control tick is one frame of 1/control_rate s (motion at the
    real servo schedule, whatever the sim wall was); wall time between ticks that is not the recorder's own rendering
    (perception, observe, IK, receipts) is a held frame with its measured duration, overlay "perception / tools";
  - finish: last frame for 1 s.
Frames: agentview | wrist side by side (soarm/frames/f*.jpg + frames.jsonl from SoarmDriver.start_recording).
Encoded with ffmpeg H.264 (yuv420p), 1x real time.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

CONTROL_HZ = 20.0
BAR = 64


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def timeline(ep: Path, prewarmed: bool = False) -> tuple[list[dict], dict]:
    frames = jsonl(ep / "soarm" / "frames" / "frames.jsonl")
    events = [e for e in jsonl(ep / "driver" / "provider_events.jsonl") if e.get("event") == "response"]
    result = json.loads((ep / "episode_result.json").read_text())
    start = float(result["started_wall_time"])
    end = start + float(result["wall_time_s"])
    reqs = [(float(e["t"]) - float(e["latency_s"]), float(e["t"])) for e in events]
    segs: list[dict] = []
    first_req = reqs[0][0] if reqs else end
    # start-up without the recorder's own rendering (it renders every reset/calibration tick)
    startup = max(0.0, first_req - start - sum(f["render_s"] for f in frames if f["t_wall"] < first_req))
    pre = [f for f in frames if f["t_wall"] < first_req]
    if prewarmed:   # J1: robot server, sandbox, services and calibration are up before the user's "go"
        segs.append({"kind": "start", "dur": 0.0, "frame": pre[-1]["i"] if pre else 0,
                     "label": f"pre-warmed (start-up {startup:.1f} s done before go)"})
    else:
        segs.append({"kind": "start", "dur": startup, "frame": 0, "label": "episode start-up"})
    last_frame = pre[-1]["i"] if pre else 0
    for k, (r0, r1) in enumerate(reqs):
        segs.append({"kind": "model", "dur": r1 - r0, "frame": last_frame,
                     "label": f"request {k + 1}/{len(reqs)} · thinking+answer {r1 - r0:.1f} s"})
        tool_end = reqs[k + 1][0] if k + 1 < len(reqs) else end
        cursor = r1
        for f in (f for f in frames if r1 <= f["t_wall"] < tool_end and f["kind"] != "start"):
            gap = f["t_wall"] - cursor
            if gap > 0.03:
                segs.append({"kind": "tools", "dur": gap, "frame": last_frame,
                             "label": f"perception / tools {gap:.1f} s"})
            segs.append({"kind": "motion", "dur": f["ticks"] / CONTROL_HZ, "frame": f["i"], "label": "moving"})
            last_frame = f["i"]
            cursor = f["t_wall"] + f["render_s"]
        gap = tool_end - cursor
        if gap > 0.03 and k + 1 < len(reqs):
            segs.append({"kind": "tools", "dur": gap, "frame": last_frame, "label": f"perception / tools {gap:.1f} s"})
    segs.append({"kind": "end", "dur": 1.0, "frame": last_frame,
                 "label": "finish() called" if result.get("agent_finished") else "episode end"})
    totals: dict = {}
    for s in segs:
        totals[s["kind"]] = totals.get(s["kind"], 0.0) + s["dur"]
    totals = {k: round(v, 1) for k, v in totals.items()}
    totals["total_s"] = round(sum(s["dur"] for s in segs[:-1]), 1)
    acc = 0.0
    for seg in segs:
        if seg["kind"] == "motion":
            totals["first_motion_s"] = round(acc, 1)
            break
        acc += seg["dur"]
    totals["requests"] = len(reqs)
    totals["success"] = bool(result.get("host_final_success"))
    totals["finished"] = bool(result.get("agent_finished"))
    return segs, totals


def render(ep: Path, out: Path, fps: int, crf: int, prewarmed: bool = False) -> dict:
    from PIL import Image, ImageDraw, ImageFont
    segs, totals = timeline(ep, prewarmed)
    fdir = ep / "soarm" / "frames"
    first = Image.open(fdir / "f000000.jpg")
    w, h = first.size
    try:
        font = ImageFont.load_default(size=22)
        small = ImageFont.load_default(size=16)
    except TypeError:
        font = small = ImageFont.load_default()
    task = ""
    tj = ep / "soarm" / "task.json"
    if tj.exists():
        task = json.loads(tj.read_text()).get("language", "")
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                             f"{w}x{h + BAR}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium",
                             "-crf", str(crf), "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    colours = {"model": (70, 130, 220), "tools": (220, 160, 40), "motion": (60, 170, 90), "start": (120, 120, 120),
               "end": (60, 170, 90)}
    clock = 0.0
    emitted = 0
    cache: dict[int, Image.Image] = {}
    for s in segs:
        clock_end = clock + s["dur"]
        n = int(round(clock_end * fps)) - emitted
        if n > 0:
            if s["frame"] not in cache:
                cache.clear()
                cache[s["frame"]] = Image.open(fdir / f"f{s['frame']:06d}.jpg").convert("RGB")
            base = cache[s["frame"]]
            for j in range(n):
                t = (emitted + j) / fps
                canvas = Image.new("RGB", (w, h + BAR), (20, 20, 20))
                canvas.paste(base, (0, BAR))
                d = ImageDraw.Draw(canvas)
                d.rectangle([0, 0, 10, BAR], fill=colours.get(s["kind"], (200, 200, 200)))
                d.text((18, 6), f"{t:6.1f} s   {s['label']}", font=font, fill=(255, 255, 255))
                d.text((18, 38), f"SO-101 twin · real-arm timeline 1x · {task[:90]}", font=small, fill=(190, 190, 190))
                proc.stdin.write(canvas.tobytes())
            emitted += n
        clock = clock_end
    proc.stdin.close()
    proc.wait()
    totals["video_s"] = round(emitted / fps, 1)
    totals["mb"] = round(out.stat().st_size / 2**20, 1)
    return totals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("episode")
    ap.add_argument("out")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--crf", type=int, default=28)
    ap.add_argument("--prewarmed", action="store_true", help="J1: start the clock at the first model request")
    a = ap.parse_args()
    print(json.dumps(render(Path(a.episode), Path(a.out), a.fps, a.crf, a.prewarmed)))


if __name__ == "__main__":
    main()
