"""Live MJPEG view of the armhost cameras on 127.0.0.1:8090 (relay it; never bind publicly) + the record switch for
the fixed camera's calibration with the board lying flat on the table (flatrec.py / flatcal.py).

/            page: streams, record buttons, progress and the current fit
/cam/<n>     MJPEG (n: 2 = fixed camera, with the board outlines while recording; 0 = wrist), /snap/<n> raw PNG
/rec/on|off|clear|status   on = calibrate the current board placement (the arm moves); off = hide the overlay
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

from flatrec import FlatRec

BY_PATH = "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:{}:1.0-video-index0"
DEVS = {"2": BY_PATH.format(12), "0": BY_PATH.format(8)}  # 2 = fixed camera (usb port 12), 0 = wrist (port 8)
latest = {}
rec = FlatRec(latest)


def grab(key, dev):
    """Read frames forever; reopen when the camera is unplugged/re-enumerated (the by-path name survives that)."""
    cap = None
    while True:
        if cap is None:
            cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(3, 1280)
            cap.set(4, 720)
        ok, f = cap.read()
        if ok:
            latest[key] = f
        else:
            cap.release()
            cap = None
            time.sleep(1.0)


PAGE = """<html><head><meta charset="utf-8"><title>armhost cameras</title></head>
<body style="background:#111;color:#ddd;font-family:sans-serif">
<div style="margin:6px 0">
 <button onclick="rec('on')">Start recording (mark this position)</button> <button onclick="rec('off')">Stop</button>
 <button onclick="if(confirm('Clear the marked positions and start over?'))rec('clear')">Clear</button>
 <span id="st" style="margin-left:10px;font-weight:bold">...</span></div>
<div id="hint" style="font-size:22px;color:#ff0;margin:6px 0">Lay the ChArUco board flat on the table (25-50 cm in front of the arm, inside the fixed camera view) and click "Start recording".
The arm takes a few shots (~2 min); do not touch the table. Then move and rotate the board and click again, 5-6 times in total, covering the whole image.</div>
<div id="res" style="font-size:16px;margin:6px 0"></div>
<img id="c2" src="/cam/2" style="max-width:98vw"><br>
<img id="c0" src="/cam/0" style="max-width:45vw">
<script>
for (const id of ["c2","c0"]) { const im = document.getElementById(id);
  im.onerror = () => setTimeout(() => { im.src = im.src.split("?")[0] + "?" + Date.now(); }, 1000); }
function show(s) {
  const st = document.getElementById("st");
  st.textContent = (s.busy ? "● arm moving: " + s.phase : (s.on ? "● recording" : "■ stopped")) + " | marked " + s.n + " positions"
    + (s.on && !s.busy ? (s.board ? " | fixed camera sees the whole board" : " | fixed camera does not see the whole board") : "");
  st.style.color = s.busy ? "#f55" : (s.on ? "#5f5" : "#ddd");
  if (s.msg) document.getElementById("hint").textContent = s.msg;
  const r = s.result;
  document.getElementById("res").textContent = !r ? "" : "fit (" + r.n_placements + " positions): focal length "
    + r.f_px.toFixed(1) + " px, reprojection " + r.rms_px.toFixed(2) + " px"
    + (r.holdout_median_mm !== undefined ? ", leave-one-out median table error " + r.holdout_median_mm.toFixed(1)
       + " mm (old calibration " + r.old_median_mm.toFixed(1) + " mm)" + (r.written ? ", written to the calibration file" : ", not written") : ", validated and written once there are 3 positions"); }
async function rec(cmd) { show(await (await fetch("/rec/" + cmd + "?" + Date.now())).json()); }
async function poll() { try { show(await (await fetch("/rec/status?" + Date.now())).json()); } catch (e) {}
  setTimeout(poll, 3000); }
poll();
</script></body></html>""".encode()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            return self.send(PAGE, "text/html; charset=utf-8")
        parts = path.strip("/").split("/")
        if parts[0] == "rec" and len(parts) == 2:
            s = rec.status() if parts[1] == "status" else rec.command(parts[1])
            return self.send(json.dumps(s, ensure_ascii=False).encode(), "application/json; charset=utf-8")
        if len(parts) != 2 or parts[1] not in DEVS or parts[1] not in latest:
            self.send_error(404)
            return
        kind, key = parts
        if kind == "snap":
            ok, buf = cv2.imencode(".png", latest[key])
            return self.send(buf.tobytes(), "image/png")
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
        self.end_headers()
        size = (960, 540) if key == "2" else (640, 360)
        try:
            while True:
                img = rec.overlay(latest[key]) if key == "2" else latest[key]
                ok, buf = cv2.imencode(".jpg", cv2.resize(img, size), [cv2.IMWRITE_JPEG_QUALITY, 70])
                self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
                time.sleep(0.1)
        except (BrokenPipeError, ConnectionResetError):
            pass


for k, d in DEVS.items():
    threading.Thread(target=grab, args=(k, d), daemon=True).start()
print("camstream ready on 127.0.0.1:8090", flush=True)
ThreadingHTTPServer(("127.0.0.1", 8090), H).serve_forever()
