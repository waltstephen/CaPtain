"""Safe probing of the real SO-101 (Feetech STS3215, IDs 1-6, 1 Mbps) on /dev/ttyACM0.

Needs feetech-servo-sdk + pyserial (+ opencv-python-headless for snaps); on armhost: $SOARM_ROOT/venv-arm.
usage: arm_probe.py '<json plan>'      (from a file: arm_probe.py "$(cat plan.json)"; run from a dir with snaps/)
plan = {"torque_limit": 400, "speed": 200, "steps": [{"id": 6, "goal": 2200}, {"id": 6, "sweep": +1}, ...],
        "return": true}
  goal  : absolute raw target; stops early on stall
  sweep : +1/-1, drive toward the raw end of travel (or "limit") until stall (mechanical stop / obstacle), back off
  per-step: torque_limit, speed, snap (both cameras after the move, via tools/camstream.py), return_after (default
  true), squeeze (goal steps: on a stall keep the goal so the joint keeps pushing at torque_limit, e.g. a grasp)
  {"snap": name} alone grabs both cameras at the current pose.
Every step logs one JSON line. Torque is on only while the script runs; at the end the joints left standing
retrace their moves in reverse, the rest go back to the start pose, and torque is released there.
Raw <-> MJCF angles and measured stops: configs/hardware/so101_armhost_joint_map.json.
"""
import json
import sys
import time

import scservo_sdk as s

IDS = [1, 2, 3, 4, 5, 6]
A_TORQUE, A_ACC, A_GOAL, A_SPEED, A_TLIM = 40, 41, 42, 46, 48
A_POS, A_LOAD, A_VOLT, A_TEMP, A_MOVING, A_CUR = 56, 60, 62, 63, 66, 69

port = s.PortHandler("/dev/ttyACM0")
pk = s.PacketHandler(0)


def _read(fn, i, a):
    for _ in range(3):  # the bus drops an occasional packet; three misses in a row is a real fault
        v, c, e = fn(port, i, a)
        if c == s.COMM_SUCCESS:
            return v
    raise IOError(f"read id{i}@{a}: {pk.getTxRxResult(c)}")


def r1(i, a):
    return _read(pk.read1ByteTxRx, i, a)


def r2(i, a):
    return _read(pk.read2ByteTxRx, i, a)


def _write(fn, i, a, v):
    for _ in range(3):  # same as reads: an occasional dropped/garbled status packet is retried
        c, e = fn(port, i, a, v)
        if c == s.COMM_SUCCESS:
            return
    raise IOError(f"write id{i}@{a}: {pk.getTxRxResult(c)}")


def w1(i, a, v):
    _write(pk.write1ByteTxRx, i, a, v)


def w2(i, a, v):
    _write(pk.write2ByteTxRx, i, a, v)


def signed(v, bit):
    return -(v & ((1 << bit) - 1)) if v & (1 << bit) else v


def state(i):
    return {"pos": r2(i, A_POS), "load": signed(r2(i, A_LOAD), 10), "cur": signed(r2(i, A_CUR), 15),
            "temp": r1(i, A_TEMP), "volt": r1(i, A_VOLT) / 10}


def log(**kw):
    print(json.dumps(kw), flush=True)


_bad = {}


def guard(i, st):
    """Abort on overheating or supply sag seen on 3 consecutive reads (single bus glitches read e.g. 101 C)."""
    bad = st["temp"] > 55 or st["volt"] < 9.0
    _bad[i] = _bad.get(i, 0) + 1 if bad else 0
    if _bad[i] >= 3:
        raise RuntimeError(f"id{i} temp {st['temp']} C volt {st['volt']} V")


def drive(i, goal, max_s=12.0, window_s=0.5, min_progress=12, stall_err=30, load_max=None):
    """Command goal; return (final_pos, reason, peak_load).
    Stall = less than min_progress steps of travel over the last window_s while still stall_err away from goal."""
    goal = max(0, min(4095, int(goal)))
    w2(i, A_GOAL, goal)
    t0 = time.time()
    hist, peak = [], 0
    while True:
        time.sleep(0.03)
        st = state(i)
        guard(i, st)
        now = time.time()
        peak = max(peak, abs(st["load"]))
        hist.append((now, st["pos"]))
        hist = [h for h in hist if now - h[0] <= window_s]
        if abs(goal - st["pos"]) <= 8:
            return st["pos"], "reached", peak
        if load_max is not None and abs(st["load"]) >= load_max:
            w2(i, A_GOAL, st["pos"])
            return st["pos"], f"load>={load_max}", peak
        span = max(p for _, p in hist) - min(p for _, p in hist)
        if now - t0 > window_s + 0.2 and span < min_progress:
            if abs(goal - st["pos"]) <= stall_err:  # gravity/compliance steady-state error: close enough
                return st["pos"], "settled", peak
            w2(i, A_GOAL, st["pos"])
            return st["pos"], "stall", peak
        if now - t0 > max_s:
            w2(i, A_GOAL, st["pos"])
            return st["pos"], "timeout", peak


CAMSTREAM = "http://127.0.0.1:8090/snap/{}"  # tools/camstream.py owns the cameras: 0 = wrist, 2 = fixed


def snap(name):
    """Save full-res PNGs of both cameras (snaps/<name>_wrist.png, _fixed.png) + a small wrist JPEG for montages."""
    import urllib.request

    import cv2
    import numpy as np
    time.sleep(0.6)  # let the arm settle and the stream move past frames taken while moving
    ok = True
    for key, tag in (("0", "wrist"), ("2", "fixed")):
        try:
            buf = urllib.request.urlopen(CAMSTREAM.format(key), timeout=5).read()
        except OSError as exc:
            log(event="snap_error", name=name, cam=tag, error=repr(exc))
            ok = False
            continue
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite(f"snaps/{name}_{tag}.png", img)
        if tag == "wrist":
            cv2.imwrite(f"snaps/{name}.jpg", cv2.resize(img, (480, 270)))
    log(event="snap", name=name, ok=ok, raw={i: r2(i, A_POS) for i in IDS})


def main():
    plan = json.loads(sys.argv[1])
    tl, speed = int(plan.get("torque_limit", 400)), int(plan.get("speed", 200))
    assert port.openPort() and port.setBaudRate(1000000)
    start = {i: r2(i, A_POS) for i in IDS}
    log(event="start", pose=start, state={i: state(i) for i in IDS})
    held, jtl = [], {}
    try:
        for i in IDS:  # hold exactly where we are before torque on: no jump
            w1(i, A_ACC, 10)
            w2(i, A_SPEED, speed)
            w2(i, A_TLIM, tl)
            w2(i, A_GOAL, start[i])
        for i in IDS:
            w1(i, A_TORQUE, 1)
        for step in plan["steps"]:
            if "id" not in step:  # {"snap": name}: wrist-camera frame of the current pose
                snap(step["snap"])
                continue
            i = step["id"]
            jtl[i] = max(jtl.get(i, tl), int(step.get("torque_limit", tl)))
            w2(i, A_TLIM, int(step.get("torque_limit", tl)))
            w2(i, A_SPEED, int(step.get("speed", speed)))
            p0 = r2(i, A_POS)
            if not step.get("return_after", True) and i not in held:
                held.append(i)
            if "goal" in step:
                # long moves at low speed need more than the default 12 s (a timeout leaves the joint short of the
                # goal while the plan carries on with the next joint)
                sp = int(step.get("speed", speed))
                max_s = step.get("max_s", max(12.0, 4.0 + 1.5 * abs(int(step["goal"]) - p0) / max(sp, 1)))
                pos, why, peak = drive(i, step["goal"], max_s=max_s, load_max=step.get("load_max"))
                log(event="move", id=i, from_=p0, goal=step["goal"], pos=pos, why=why, peak_load=peak)
                if step.get("squeeze") and why == "stall":  # grasp: keep pushing at the torque limit, not goal=pos
                    w2(i, A_GOAL, int(step["goal"]))
            else:
                d = int(step["sweep"])
                end = 4095 if d > 0 else 0
                if "limit" in step:
                    end = int(step["limit"])
                pos, why, peak = drive(i, end, max_s=step.get("max_s", 30.0), load_max=step.get("load_max"))
                log(event="sweep", id=i, dir=d, from_=p0, pos=pos, why=why, peak_load=peak)
                back = pos - d * int(step.get("backoff", 40))
                bpos, bwhy, _ = drive(i, back)
                log(event="backoff", id=i, pos=bpos, why=bwhy)
            if "snap" in step:
                snap(step["snap"])
            if step.get("return_after", True):
                rpos, rwhy, _ = drive(i, p0)
                log(event="return", id=i, pos=rpos, why=rwhy)
    except Exception as exc:  # hold pose, then fall through to the slow return
        log(event="abort", error=repr(exc))
        for i in IDS:
            try:
                w2(i, A_GOAL, r2(i, A_POS))
            except Exception:
                pass
    finally:
        if plan.get("return", True):
            # undo the moves that were left standing in reverse order (retrace the path), then everything else
            for i in list(reversed(held)) + [j for j in [6, 5, 4, 3, 1, 2] if j not in held]:
                try:
                    w2(i, A_SPEED, min(speed, 200))
                    w2(i, A_TLIM, jtl.get(i, tl))
                    pos, why, _ = drive(i, start[i])
                    log(event="home", id=i, pos=pos, target=start[i], why=why)
                except Exception as exc:
                    log(event="home_error", id=i, error=repr(exc))
        if plan.get("release", True):
            time.sleep(0.3)
            for i in IDS:
                try:
                    w1(i, A_TORQUE, 0)
                except Exception:
                    pass
        log(event="end", pose={i: r2(i, A_POS) for i in IDS}, temps={i: r1(i, A_TEMP) for i in IDS})
        port.closePort()


if __name__ == "__main__":
    main()
