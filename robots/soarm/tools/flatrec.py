"""Record switch of camstream.py: fixed-camera calibration with the board lying flat on the table (flatcal.py).

Each press of the record button calibrates one board placement: read the arm (that pose is "rest" for this run),
park the arm out of the fixed view, fixed snapshot, wrist views of the board, back to rest (torque off), add the
placement, refit (written to the cameras json from 3 placements on, if the hold-out check passes). Planning runs in
venv-srv (mujoco/scipy), arm motion in venv-arm (arm_probe.py); any refusal or abort while the arm holds a pose
plans a sim-checked way back to rest. Log: flat/run.log.
While recording is on, the fixed stream shows the live board outline (green = whole board seen) and the outlines of
the placements done so far (cyan).
"""
import json
import os
import subprocess
import threading
import time

import cv2
import numpy as np

HERE = os.path.expandvars("$SOARM_ROOT/armcal")
FLAT = os.path.join(HERE, "flat")
PY_SRV = os.path.expandvars("$SOARM_ROOT/venv-srv/bin/python")
PY_ARM = os.path.expandvars("$SOARM_ROOT/venv-arm/bin/python")
CAMS = "so101_armhost_cameras.json"
READ_RAW = """import json, scservo_sdk as s
p = s.PortHandler("/dev/ttyACM0"); k = s.PacketHandler(0); p.openPort(); p.setBaudRate(1000000)
print(json.dumps({str(i): k.read2ByteTxRx(p, i, 56)[0] for i in range(1, 7)}))"""


class Refused(Exception):
    pass


_DP = cv2.aruco.DetectorParameters()  # same boards and tuning as flatcal.DETECTORS
_DP.adaptiveThreshWinSizeMax, _DP.minMarkerPerimeterRate, _DP.errorCorrectionRate = 53, 0.01, 0.8
_CP = cv2.aruco.CharucoParameters()
_CP.minMarkers = 1
_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
DETECTORS = [cv2.aruco.CharucoDetector(cv2.aruco.CharucoBoard((7, 5), 0.035, 0.026, _DICT,
                                                               np.arange(17 * b, 17 * b + 17)), _CP, _DP)
              for b in range(2)]


def outline(pts):
    """Convex hull of the detected ChArUco corners."""
    return cv2.convexHull(np.asarray(pts, np.float32)).astype(np.int32)


class FlatRec:
    def __init__(self, latest):
        self.latest, self.on, self.busy = latest, False, False
        self.phase, self.msg, self.live, self.live_t = "", "", [], 0.0
        self.lock = threading.Lock()
        os.makedirs(FLAT, exist_ok=True)
        threading.Thread(target=self._track, daemon=True).start()

    # ---- state shown on the page
    def placements(self):
        p = os.path.join(FLAT, "placements.json")
        return json.load(open(p)) if os.path.exists(p) else []

    def status(self):
        r = os.path.join(FLAT, "result.json")
        res = json.load(open(r)) if os.path.exists(r) and self.placements() else None
        return {"on": self.on, "busy": self.busy, "n": len(self.placements()), "phase": self.phase, "msg": self.msg,
                "board": bool(self.live), "result": res}

    def command(self, cmd):
        with self.lock:
            if cmd == "on":
                self.on = True
                if not self.busy:
                    self.busy = True
                    threading.Thread(target=self._run_safe, daemon=True).start()
            elif cmd == "off":
                self.on = False
            elif cmd == "clear" and not self.busy:
                for f in ("placements.json", "result.json"):
                    p = os.path.join(FLAT, f)
                    if os.path.exists(p):
                        os.replace(p, p + f".cleared-{time.strftime('%m%d-%H%M%S')}")
                self.msg = "cleared (previous data renamed as a backup in flat/)"
        return self.status()

    # ---- live overlay
    def _track(self):
        while True:
            img = self.latest.get("2")
            if self.on and img is not None:
                g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                live = []
                for det in DETECTORS:
                    cc, ci, _, _ = det.detectBoard(g)
                    if ci is not None and len(ci) >= 10:
                        live.append(cc.reshape(-1, 2))
                self.live, self.live_t = live, time.time()
            time.sleep(0.5)

    def overlay(self, img):
        if not self.on:
            return img
        img = img.copy()
        for p in self.placements():
            for bd in p["boards"].values():
                cv2.polylines(img, [outline(bd["fixed_px"])], True, (255, 255, 0), 2)
        if time.time() - self.live_t < 2:
            for pts in self.live:
                cv2.polylines(img, [outline(pts)], True, (0, 255, 0), 4)
        top = "ARM MOVING - hands off the table" if self.busy else (
            f"{len(self.live)} board(s) seen" if self.live else "no board found in the fixed view")
        cv2.putText(img, f"placements {len(self.placements())} | {top}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 255) if self.busy else (0, 255, 255), 2)
        return img

    # ---- one placement
    def _sh(self, args, log, what):
        with open(os.path.join(FLAT, "run.log"), "a") as lf:
            lf.write(f"\n=== {time.strftime('%H:%M:%S')} {what}: {' '.join(args)[:300]}\n")
            p = subprocess.run(args, cwd=HERE, capture_output=True, text=True)
            lf.write(p.stdout + p.stderr)
        if log:
            open(os.path.join(HERE, log), "w").write(p.stdout)
        if p.returncode != 0:
            last = (p.stdout + p.stderr).strip().splitlines()
            reason = next((l[7:] for l in reversed(last) if l.startswith("REFUSE ")), last[-1] if last else "?")
            raise Refused(f"{what}: {reason}")
        return p.stdout

    def _arm(self, plan_path, log, what):
        self._sh([PY_ARM, "arm_probe.py", open(os.path.join(HERE, plan_path)).read()], log, what)
        events = [json.loads(l) for l in open(os.path.join(HERE, log)) if l.startswith("{")]
        # a stall stops that joint where it is (arm_probe sets goal = pos) and the plan goes on; the snapshots record
        # the real joint positions, so the data stay valid. Only an abort (exception) fails the stage.
        stalls = [e for e in events if e.get("why") in ("stall", "timeout")]
        if stalls:
            self.notes.append(f"{what}: {len(stalls)} joint stalls (joints {sorted({e['id'] for e in stalls})}), the rest as planned")
        bad = [e for e in events if e.get("event") == "abort"]
        if bad:
            raise Refused(f"{what}: arm_probe {bad[0]}")

    def _raw(self):
        return self._sh([PY_ARM, "-c", READ_RAW], None, "read arm").strip()

    def _home(self, rest):
        """Sim-checked way from the current pose back to rest, torque released there."""
        self.phase = "returning to rest"
        self._sh([PY_SRV, "flatcal.py", "home", CAMS, self._raw(), rest, "flat/plan_home.json"], None, "plan home")
        self._arm("flat/plan_home.json", "flat/home.log", "home")

    def _run_safe(self):
        rest, holding, self.notes = None, False, []
        try:
            k = len(self.placements())
            while os.path.exists(os.path.join(HERE, "snaps", f"flat{k:02d}_v00_fixed.png")):
                k += 1
            pre, fl = f"flat{k:02d}", "flat/"
            self.phase, self.msg = "reading arm position", ""
            rest = self._raw()
            self.phase = "planning"
            self._sh([PY_SRV, "flatcal.py", "plan", CAMS, rest, fl + "plan.json", pre], None, "plan")
            self.phase, holding = "arm photographing the board (do not touch the table)", True
            self._arm(fl + "plan.json", fl + f"{pre}.log", "survey")
            self._home(rest)  # the survey ends at rest with torque on; home checks that and releases
            holding = False
            self.phase = "computing board position"
            self._sh([PY_SRV, "flatcal.py", "add", CAMS, fl + f"{pre}.log", "flat", pre], None, "add")
            self.phase = "fitting the fixed camera"
            out = self._sh([PY_SRV, "flatcal.py", "solve", CAMS, "flat", "--write"], None, "solve")
            n = len(self.placements())
            self.msg = (f"position {n} done, arm back at rest. " + ("Move the board elsewhere on the table (ideally rotated), then click 'Start recording' again."
                        if n < 3 else "Results below; record more positions for a better fit."))
            self.msg += " | " + out.strip().splitlines()[-1] + ("; " + "; ".join(self.notes) if self.notes else "")
        except Exception as exc:
            self.msg = f"refused: {exc}" if isinstance(exc, Refused) else f"error: {exc!r}"
            if holding:
                try:
                    self._home(rest)
                    self.msg += " (arm back at rest, torque off)"
                except Exception as exc2:
                    self.msg += f"; returning to rest also failed: {exc2}. The arm may be holding torque in mid-air; get help before touching it."
        finally:
            self.phase, self.busy = "", False
