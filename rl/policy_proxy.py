#!/usr/bin/env python3
"""Recording proxy between the harness driver (Chat Completions client) and the policy vLLM server.

    driver  -> POST http://127.0.0.1:8510/s/<sid>/v1/chat/completions  -> vLLM :8500 /v1/chat/completions

Per session id it keeps the token ids of every request (`prompt_token_ids`, per-choice `token_ids`, from vLLM's
`return_token_ids`), the request/response usage, the ordered image hashes of every request plus one session-level
image store (hash -> data URL, deduplicated), and the body of the LAST request (messages incl. base64 images, tools).
The rollout service selects original requests, preserving each prompt, completion and image payload. Invocation indices are assigned
before forwarding so retries and out-of-order responses cannot be mistaken for assistant-history positions. Nothing is written to disk by the
proxy; the rollout service fetches `GET /s/<sid>/record` when the episode is over and `DELETE /s/<sid>` afterwards.
--upstream may list several policy servers (comma separated, same weights): each session id is pinned to one of them
(hash of the sid) so an episode keeps its prefix cache while episodes spread over the servers.
With --sampler-topk K the proxy also asks vLLM for the sampler's top-K logprobs of every generated token (token ids) and
records them per request as `sampler_topk` (score centering in the trainer); the driver's response is returned unchanged.

Run inside envs/libero (fastapi + uvicorn + httpx or requests).
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import re
import threading
import time
from typing import Any

import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
import uvicorn

from chat_tokens import NEUTRAL_SAMPLING, RECORD_FORMAT

app = FastAPI()
DEFAULT_SAMPLING = dict(NEUTRAL_SAMPLING)
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty", "frequency_penalty",
                 "seed", "max_tokens", "max_completion_tokens", "min_tokens", "stop", "stop_token_ids", "ignore_eos",
                 "logit_bias", "allowed_token_ids", "n", "best_of", "bad_words", "chat_template_kwargs",
                 "logits_processors", "structured_outputs", "guided_json", "guided_regex", "guided_choice",
                 "guided_grammar", "guided_json_object", "response_format", "tool_choice")
STATE: dict[str, Any] = {"upstreams": ["http://127.0.0.1:8500"], "timeout": 1800.0, "sampling": dict(DEFAULT_SAMPLING), "sampler_topk": 0}


_CANDIDATE = re.compile(r"^s(\d+)g(\d+)k(\d+)_")


ROUTES: dict[str, str] = {}          # --route least-loaded: sid -> upstream, fixed at the session's first request
ROUTE_LOCK = threading.Lock()


def _active_sessions(url: str) -> int:
    return sum(1 for u in ROUTES.values() if u == url)


def upstream_for(sid: str | None) -> str:
    """The policy server of a session, stable for the sid (the session's requests share their prefix: one server keeps it
    in its prefix cache).

    --route static (default): training candidates (sid s<round>g<group>k<candidate>_...) are spread evenly: worker =
    (round + group * G + candidate) % workers, so every group size uses every worker; other sessions hash.
    --route least-loaded (Molt/vllm-router-style admission, still sticky): a new session goes to the server with the fewest
    live sessions (ties: fewest in-flight requests, then the static choice); the pin is dropped at DELETE /s/<sid>
    (the env server deletes every session after fetching its record) or when the session is idle for --route-idle-s."""
    ups = STATE["upstreams"]
    if not sid or len(ups) == 1:
        return ups[0]
    if STATE.get("route") == "request-rr":      # benchmark baseline only: no session affinity (prefix cache split)
        with ROUTE_LOCK:
            STATE["rr"] = STATE.get("rr", -1) + 1
            return ups[STATE["rr"] % len(ups)]
    if STATE.get("route") == "least-loaded":
        with ROUTE_LOCK:
            url = ROUTES.get(sid)
            if url is None:
                _expire_routes()
                static = _static_upstream(sid, ups)
                inflight = STATE.setdefault("inflight", {})
                url = min(ups, key=lambda u: (_active_sessions(u), inflight.get(u, 0), u != static))
                ROUTES[sid] = url
            STATE.setdefault("route_seen", {})[sid] = time.time()
            return url
    return _static_upstream(sid, ups)


def _expire_routes() -> None:
    """Caller holds ROUTE_LOCK."""
    idle = float(STATE.get("route_idle_s") or 0)
    if idle <= 0:
        return
    now, seen = time.time(), STATE.setdefault("route_seen", {})
    for sid in [s for s, t in seen.items() if now - t > idle]:
        seen.pop(sid, None)
        ROUTES.pop(sid, None)


def _static_upstream(sid: str, ups: list[str]) -> str:
    m = _CANDIDATE.match(sid)
    if m:
        step, group, cand = (int(x) for x in m.groups())
        return ups[(step + group * int(STATE.get("group_size") or 1) + cand) % len(ups)]
    return ups[int.from_bytes(hashlib.sha256(sid.encode("utf-8")).digest()[:8], "big") % len(ups)]


# Streaming SAO: the trainer replaces the weights while episodes keep running. The gate makes every
# request's policy version exact: /gate/close stops forwarding new requests and returns once none is in flight (a held
# request waits in the proxy), the trainer swaps the weights on every policy server, /gate/open?version=v resumes and
# tags every later request with v. A request never spans two versions; an episode may (partial rollout, per-request tags).
GATE = threading.Condition()
GATE_STATE: dict[str, Any] = {"open": True, "version": None, "forwarding": 0, "closed_at": None, "closures": 0}


def _gate_enter() -> tuple[Any, float]:
    t0 = time.time()
    with GATE:
        while not GATE_STATE["open"]:
            GATE.wait(timeout=5.0)
        GATE_STATE["forwarding"] += 1
        return GATE_STATE["version"], time.time() - t0


def _gate_exit() -> None:
    with GATE:
        GATE_STATE["forwarding"] -= 1
        GATE.notify_all()


RECORDS: dict[str, dict] = {}
LOCK = threading.Lock()


def _record(sid: str) -> dict:
    with LOCK:
        rec = RECORDS.get(sid)
        if rec is None:
            rec = RECORDS[sid] = {"sid": sid, "record_format": RECORD_FORMAT, "created": time.time(), "requests": [],
                                  "last_request": None, "last_response": None, "errors": 0, "images": {},
                                  "next_request_index": 1, "in_flight": 0, "payload_attempts": {}}
        return rec


@app.get("/s/{sid}/v1/models")
@app.get("/v1/models")
async def models(sid: str | None = None):
    try:
        r = requests.get(f"{upstream_for(sid)}/v1/models", timeout=30)
        return JSONResponse(r.json(), status_code=r.status_code)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": repr(exc)}, status_code=502)


def _image_urls(body: dict) -> list[str]:
    """Image data URLs of a chat request in message order (what the template turns into image pads)."""
    urls: list[str] = []
    for m in body.get("messages") or []:
        content = m.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = part.get("image_url")
                    url = url.get("url") if isinstance(url, dict) else url
                    if isinstance(url, str):
                        urls.append(url)
    return urls


def sampler_topk(choice: dict, token_ids: list[int] | None, k: int) -> dict:
    """The sampler's per-token top-k (ids, logprobs) and the sampled token's logprob, aligned with ``token_ids``
    (vLLM chat logprobs with return_tokens_as_token_ids: tokens are "token_id:<n>")."""
    content = ((choice.get("logprobs") or {}).get("content")) or []
    if not isinstance(token_ids, list) or len(content) != len(token_ids):
        raise ValueError(f"{len(content)} logprob rows for {len(token_ids or [])} tokens")
    def tid(entry):
        token = str(entry.get("token") or "")
        if not token.startswith("token_id:"):
            raise ValueError("logprob tokens are not token ids")
        return int(token.split(":", 1)[1])
    ids, lps, sampled = [], [], []
    for position, row in enumerate(content):
        if tid(row) != token_ids[position]:
            raise ValueError(f"logprob token {position} differs from the sampled token")
        top = row.get("top_logprobs") or []
        if len(top) != k:
            raise ValueError(f"{len(top)} top logprobs at token {position}, expected {k}")
        ids.append([tid(t) for t in top])
        lps.append([float(t["logprob"]) for t in top])
        sampled.append(float(row["logprob"]))
    return {"k": k, "ids": ids, "logprobs": lps, "sampled_logprob": sampled}


def _forward(sid: str, body: dict, t_arrive: float | None = None) -> tuple[int, dict | str, float]:
    body = dict(body)
    body["return_token_ids"] = True
    topk = int(STATE.get("sampler_topk") or 0)
    driver_logprobs = bool(body.get("logprobs"))
    if topk:
        body.update(logprobs=True, top_logprobs=topk, return_tokens_as_token_ids=True)
    body.setdefault("stream", False)
    body["stream"] = False
    for k, v in STATE["sampling"].items():
        body.setdefault(k, v)
    t0 = time.time()
    urls = _image_urls(body)
    hashes = [hashlib.sha256(u.encode("utf-8")).hexdigest() for u in urls]
    fingerprint = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    rec = _record(sid)
    with LOCK:
        index = rec["next_request_index"]
        rec["next_request_index"] += 1
        attempt = rec["payload_attempts"].get(fingerprint, 0) + 1
        rec["payload_attempts"][fingerprint] = attempt
        entry = {"request_index": index, "t": t0, "completed": False, "status": None,
                 "request_fingerprint": fingerprint, "payload_attempt": attempt,
                 "image_hashes": hashes, "n_messages": len(body.get("messages") or []),
                 "sampling": {k: copy.deepcopy(body[k]) for k in SAMPLING_KEYS if k in body},
                 "sampling_source": "request_after_proxy_defaults",
                 # ASGI arrival -> _forward (body parsing, event-loop and executor admission): proxy ingress delay
                 **({"t_arrive": t_arrive, "ingress_s": round(t0 - t_arrive, 3)} if t_arrive is not None else {})}
        rec["requests"].append(entry)
        rec["in_flight"] += 1
        for h, u in zip(hashes, urls):
            rec["images"].setdefault(h, u)
    version, gate_wait = _gate_enter()
    url = upstream_for(sid)
    inflight = STATE.setdefault("inflight", {})
    with ROUTE_LOCK:
        inflight[url] = inflight.get(url, 0) + 1
    with LOCK:
        entry["upstream"] = url
        if version is not None:
            entry["policy_version"] = version
        if gate_wait > 0.01:
            entry["gate_wait_s"] = round(gate_wait, 3)
    try:
        r = requests.post(f"{url}/v1/chat/completions", json=body, timeout=STATE["timeout"])
    except Exception as exc:  # noqa: BLE001
        _gate_exit()
        with ROUTE_LOCK:
            inflight[url] -= 1
        elapsed = time.time() - t0
        with LOCK:
            rec["errors"] += 1
            rec["in_flight"] -= 1
            entry.update(completed=True, status=502, latency_s=round(elapsed, 2), error=repr(exc)[:500])
        return 502, {"error": {"message": f"upstream error: {exc!r}", "type": "server_error"}}, elapsed
    _gate_exit()
    with ROUTE_LOCK:
        inflight[url] -= 1
    try:
        payload: dict | str = r.json()
    except Exception:  # noqa: BLE001
        payload = r.text
    elapsed = time.time() - t0
    with LOCK:
        rec["in_flight"] -= 1
        entry.update(completed=True, status=r.status_code, latency_s=round(elapsed, 2))
        if r.status_code == 200 and isinstance(payload, dict):
            choices = payload.get("choices")
            choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
            msg = choice.get("message") if isinstance(choice.get("message"), dict) else {}
            entry.update({
                "prompt_token_ids": payload.get("prompt_token_ids"),
                "token_ids": choice.get("token_ids"), "finish_reason": choice.get("finish_reason"),
                "usage": payload.get("usage"), "response_id": payload.get("id"),   # joins the driver's requests.jsonl
                "n_tool_calls": len(msg.get("tool_calls") or []), "reasoning_chars": len(str(msg.get("reasoning_content") or "")),
                "content_chars": len(str(msg.get("content") or "")),
            })
            if topk:
                try:
                    entry["sampler_topk"] = sampler_topk(choice, choice.get("token_ids"), topk)
                except (ValueError, TypeError, KeyError) as exc:
                    entry["sampler_topk_error"] = str(exc)[:300]
            if index > (rec.get("last_response") or {}).get("request_index", 0):
                rec["last_request"] = body
                rec["last_response"] = {"request_index": index, "message": msg, "finish_reason": choice.get("finish_reason"),
                                        "prompt_token_ids": payload.get("prompt_token_ids"), "token_ids": choice.get("token_ids"), "usage": payload.get("usage")}
        else:
            rec["errors"] += 1
            entry["error"] = str(payload)[:500]
    if isinstance(payload, dict):
        payload.pop("prompt_token_ids", None)
        for ch in payload.get("choices") or []:
            if isinstance(ch, dict):
                ch.pop("token_ids", None)
                if topk and not driver_logprobs:
                    ch.pop("logprobs", None)
    return r.status_code, payload, elapsed


EARLY_STOP: dict[str, str] = {}     # co-critic early stop: sid -> reason; the session's next request is refused (HTTP 403)


@app.post("/s/{sid}/early_stop")
async def early_stop(sid: str, request: Request):
    """The env server's critic early stop: refuse this session's next chat request with a non-retryable 403 whose message names the
    request cap, so the driver ends the loop before generating it (the env server then records the episode as budget-exhausted)."""
    try:
        reason = str((await request.json()).get("reason") or "early stop")
    except Exception:  # noqa: BLE001
        reason = "early stop"
    with LOCK:
        EARLY_STOP[sid] = reason[:500]
    return {"ok": True}


@app.post("/s/{sid}/v1/chat/completions")
async def chat(sid: str, request: Request):
    t_arrive = time.time()
    body = await request.json()
    if EARLY_STOP:
        with LOCK:
            reason = EARLY_STOP.get(sid)
        if reason is not None:
            return JSONResponse({"error": {"message": f"request cap reached early: {reason}", "type": "early_stop",
                                           "code": "early_stop"}}, status_code=403)
    loop = asyncio.get_event_loop()
    status, payload, _ = await loop.run_in_executor(None, _forward, sid, body, t_arrive)
    if isinstance(payload, str):
        return JSONResponse({"error": {"message": payload[:2000]}}, status_code=status)
    return JSONResponse(payload, status_code=status)


def record_snapshot(rec: dict, full: bool) -> dict:
    """Caller holds LOCK. A shallow snapshot: completed request entries are never mutated in place (updates replace the entry's
    keys, which the per-entry dict copy isolates; their token/probability lists are assigned once), so the large arrays are
    shared instead of deep-copied under the lock shared by every live episode."""
    out = {k: v for k, v in rec.items() if k not in ("payload_attempts", "requests", "images")}
    out["requests"] = [dict(q) for q in rec["requests"]]
    if full:
        out["images"] = dict(rec.get("images") or {})
    else:
        out.pop("last_request", None)
    return out


def encode_record(out: dict) -> bytes:
    # the same rendering as Starlette's JSONResponse, without FastAPI's per-element jsonable_encoder pass
    return json.dumps(out, ensure_ascii=False, allow_nan=False, indent=None, separators=(",", ":")).encode("utf-8")


@app.get("/s/{sid}/status")
async def status(sid: str):
    """Lifecycle counters only (the env server polls this until in-flight requests finish)."""
    with LOCK:
        rec = RECORDS.get(sid)
        if rec is None:
            return JSONResponse({"error": f"unknown session {sid}"}, status_code=404)
        return {"sid": sid, "in_flight": rec["in_flight"], "requests": len(rec["requests"]),
                "completed": sum(1 for q in rec["requests"] if q.get("completed")), "errors": rec["errors"]}


@app.get("/s/{sid}/record")
async def record(sid: str, full: int = 1):
    with LOCK:
        rec = RECORDS.get(sid)
        if rec is None:
            return JSONResponse({"error": f"unknown session {sid}"}, status_code=404)
        out = record_snapshot(rec, bool(full))
    body = await asyncio.to_thread(encode_record, out)       # off the event loop and outside LOCK
    return Response(content=body, media_type="application/json")


@app.delete("/s/{sid}")
async def delete(sid: str):
    with LOCK:
        RECORDS.pop(sid, None)
        EARLY_STOP.pop(sid, None)
    with ROUTE_LOCK:
        ROUTES.pop(sid, None)
        STATE.get("route_seen", {}).pop(sid, None)
    return {"ok": True}


@app.post("/gate/close")
async def gate_close(timeout: float = 900.0):
    """Stop forwarding new requests; return when the in-flight ones have finished (weights may then be replaced)."""
    t0 = time.time()

    def drain():
        with GATE:
            GATE_STATE["open"] = False
            GATE_STATE["closed_at"] = t0
            GATE_STATE["closures"] += 1
            while GATE_STATE["forwarding"] > 0 and time.time() - t0 < timeout:
                GATE.wait(timeout=1.0)
            return GATE_STATE["forwarding"]

    left = await asyncio.to_thread(drain)
    if left:
        return JSONResponse({"closed": False, "forwarding": left, "drain_s": round(time.time() - t0, 2)}, status_code=504)
    return {"closed": True, "drain_s": round(time.time() - t0, 2), "version": GATE_STATE["version"]}


@app.post("/gate/open")
async def gate_open(version: int | None = None):
    with GATE:
        if version is not None:
            GATE_STATE["version"] = int(version)
        GATE_STATE["open"] = True
        held = round(time.time() - (GATE_STATE["closed_at"] or time.time()), 2)
        GATE.notify_all()
    return {"open": True, "version": GATE_STATE["version"], "held_s": held}


@app.get("/gate")
async def gate_status():
    with GATE:
        return dict(GATE_STATE)


@app.get("/health")
async def health():
    with LOCK:
        n = len(RECORDS)
        n_req = sum(len(r["requests"]) for r in RECORDS.values())
        n_img = sum(len(r.get("images") or {}) for r in RECORDS.values())
    try:
        codes = [requests.get(f"{u}/health", timeout=5).status_code for u in STATE["upstreams"]]
        up = 200 if all(c == 200 for c in codes) else next(c for c in codes if c != 200)
    except Exception:  # noqa: BLE001
        up = None
    with ROUTE_LOCK:
        route = {"mode": STATE.get("route", "static"), "inflight": dict(STATE.get("inflight", {})),
                 "pinned": {u: _active_sessions(u) for u in STATE["upstreams"]} if STATE.get("route") == "least-loaded" else None}
    return {"ok": True, "sessions": n, "requests": n_req, "images": n_img, "upstream_health": up, "sampling": STATE["sampling"],
            "route": route}


@app.post("/sampling")
async def sampling(request: Request):
    STATE["sampling"] = {**DEFAULT_SAMPLING, **(await request.json())}
    return {"ok": True, "sampling": STATE["sampling"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8510)
    ap.add_argument("--host", default="127.0.0.1", help="bind address (0.0.0.0 when another node must reach it)")
    ap.add_argument("--upstream", default="http://127.0.0.1:8500", help="policy server URL, or a comma list of servers with the same weights")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--group-size", type=int, default=1, help="candidates per group (balanced worker assignment of training sessions)")
    ap.add_argument("--sampler-topk", type=int, default=0, help="record the sampler's top-k logprobs per generated token (score centering; <= vLLM --max-logprobs, default 20)")
    ap.add_argument("--route", default=os.environ.get("RL_PROXY_ROUTE", "static"), choices=["static", "least-loaded", "request-rr"],
                    help="session -> policy server: static candidate spread (default), least-loaded sticky admission, or "
                         "request-rr (benchmark baseline without affinity)")
    ap.add_argument("--route-idle-s", type=float, default=1800.0, help="least-loaded: forget a session's pin after this idle time")
    ap.add_argument("--policy-version", type=int, default=None if os.environ.get("RL_POLICY_VERSION") is None else int(os.environ["RL_POLICY_VERSION"]),
                    help="streaming: tag requests with this policy version until the trainer's first /gate/open")
    args = ap.parse_args()
    GATE_STATE["version"] = args.policy_version
    STATE.update({"route": args.route, "route_idle_s": args.route_idle_s, "upstreams": [u.strip() for u in args.upstream.split(",") if u.strip()], "timeout": args.timeout, "sampler_topk": args.sampler_topk,
                  "group_size": args.group_size,
                  "sampling": {**DEFAULT_SAMPLING, "temperature": args.temperature, "top_p": args.top_p}})
    print(f"[policy_proxy] :{args.port} -> {args.upstream} route={args.route} sampling={STATE['sampling']}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", workers=1, timeout_keep_alive=120)


if __name__ == "__main__":
    main()
