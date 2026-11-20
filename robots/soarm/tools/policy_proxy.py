"""Recording proxy between the CaPtain driver and the vLLM policy server (soarm-speed; eval/deploy only).

    python tools/policy_proxy.py --port 8710 --upstream http://127.0.0.1:8700 --record requests.jsonl

POST /v1/chat/completions: the JSON body is appended to --record (full body, for tools/policy_replay.py and drafter
data) and forwarded unchanged. Everything else (GET /v1/models, /metrics, /health) is passed through. Request knobs
(thinking budget, output cap, temperature, chunked image cap) are driver flags of the speed profile
(soarm/speed_profile.py), not proxy edits.
"""
from __future__ import annotations

import argparse
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOCK = threading.Lock()


def make_handler(upstream: str, record: str):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # quiet
            pass

        def _forward(self, method: str, body: bytes | None) -> None:
            req = urllib.request.Request(upstream + self.path, data=body, method=method,
                                         headers={"Content-Type": self.headers.get("Content-Type", "application/json")})
            try:
                with urllib.request.urlopen(req, timeout=3600) as r:
                    data, status, ctype = r.read(), r.status, r.headers.get("Content-Type", "application/json")
            except urllib.error.HTTPError as e:
                data, status, ctype = e.read(), e.code, e.headers.get("Content-Type", "application/json")
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._forward("GET", None)

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.path.rstrip("/").endswith("/chat/completions"):
                line = json.dumps(json.loads(body))
                with LOCK, open(record, "a") as fh:
                    fh.write(line + "\n")
            self._forward("POST", body)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--record", required=True)
    a = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(a.upstream.rstrip("/"), a.record))
    print(f"policy_proxy :{a.port} -> {a.upstream} record={a.record}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
