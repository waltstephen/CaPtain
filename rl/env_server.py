#!/usr/bin/env python3
"""Rollout (environment) service of CaPtain RL: runs CaPtain-Harness episodes against the policy proxy.

    POST /rollout  {suite, task_id, seed, sid?, step?, tag?}
                   -> runs one `scripts/react_episode.py --api chat` episode against the policy proxy,
                      scores it with rl/reward.py, fetches the proxy record (token ids + image hashes of every
                      request, image store) and writes <episode_dir>/rollout.json (the training sample).
                   <- {reward, verdict, components, diagnostics, episode_dir, sample_path, env_wall_s, sid}
    GET  /health

New rollouts use per_request_v1: selected original requests retain their exact
prompt, sampled completion and images. Only the sampled suffix is a loss target.
The full episode completion-token total is recorded before uniform subsampling.
The student serving profile (MUST equal the one used for the teacher data / SFT windowing / eval) is
`--profile captain --context-strategy window --context-limit 60000 --keep-recent-tokens 16000 --tool-output-tokens 4000
--spill-tool-output` with a 65536-token driver window on an 81920 vLLM model window.

Run inside envs/libero. Concurrency = one episode per request, bounded by --max-concurrency; process
launches are staggered (EGL context bursts) and EGL devices are round-robined.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, Request
import uvicorn
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reward import GOAL_AUDIT_SCHEMA, REWARD_VERSION, score_episode  # noqa: E402
from rollout_storage import write_rollout_json_atomic  # noqa: E402
from chat_tokens import RECORD_FORMAT, SAMPLE_FORMAT, align_history, validate_sampling_settings, validate_token_ids, with_im_end  # noqa: E402

app = FastAPI()
STATE: dict[str, Any] = {}
LAUNCH_LOCK = threading.Lock()
LAUNCH_STATE = {"last": 0.0, "count": 0}
ACTIVE: dict[str, float] = {}
ACTIVE_LOCK = threading.Lock()


def code_fields(label: dict | None) -> dict:
    """python_exec_cost_v1 fields of one eligible request (rl/code_reward.py): cost in [0, 1], its category, whether it
    is a pure error (no host effect: positive task credit withheld by the trainer) and whether it was scored at all."""
    label = label or {"category": "unmapped", "cost": 0.0, "scored": False, "no_effect": False}
    return {"code_cost": float(label.get("cost") or 0.0) if label.get("scored") else 0.0,
            "code_category": str(label.get("category")), "code_no_effect": bool(label.get("no_effect")) and bool(label.get("scored")),
            "code_scored": bool(label.get("scored")),
            **({"code_delta_cost": float(label["delta_cost"]), "code_interface_rejections": int(label.get("interface_rejections") or 0)}
               if "delta_cost" in label else {})}


def build_training_sample(rec: dict, *, sid: str, max_training_requests: int = 4, ended_at: float | None = None,
                          solved_suffix: set[int] | None = None, code: dict[int, dict] | None = None,
                          max_request_tokens: int | None = None) -> dict:
    """Validate the complete original record before uniformly selecting requests. With ``ended_at`` (the driver's exit
    time) a request whose response arrived after the episode ended (the driver was stopped at the wall-time deadline
    while waiting for it) is recorded but not eligible: the policy's output was never acted on.
    Over-length guard (None = off, the output is unchanged bit for bit): with ``max_request_tokens``
    (the trainer's hard cap min(max_seq_tokens, hard_max_tokens)) a request with prompt + completion tokens above it is recorded
    (over_cap) but not eligible, BEFORE the uniform selection: the inclusion probabilities, loss weights and the completion-token
    denominator are those of the remaining eligible requests (the trainer could never train it; its length check would refuse the
    whole batch). An episode with no eligible request left has full_completion_tokens 0 (no sample_path: not trained)."""
    if not isinstance(rec, dict):
        raise ValueError("proxy record must be an object")
    if rec.get("record_format") != RECORD_FORMAT:
        raise ValueError(f"expected proxy record_format={RECORD_FORMAT!r}")
    if rec.get("sid") != sid:
        raise ValueError("proxy session id does not match the episode")
    if rec.get("in_flight", 0):
        raise ValueError("proxy still has in-flight requests")
    if type(max_training_requests) is not int or max_training_requests == 0 or max_training_requests < -1:
        raise ValueError("max_training_requests must be positive or -1 for all requests")
    if max_request_tokens is not None and (type(max_request_tokens) is not int or max_request_tokens < 1):
        raise ValueError("max_request_tokens must be a positive integer or None")
    requests_raw = rec.get("requests")
    if not isinstance(requests_raw, list) or any(not isinstance(q, dict) for q in requests_raw):
        raise ValueError("proxy requests must be a list of request records")
    indices = [q.get("request_index") for q in requests_raw]
    if any(type(i) is not int or i < 1 for i in indices) or sorted(indices) != list(range(1, len(indices) + 1)):
        raise ValueError("proxy request indices must be unique and contiguous from one")
    reqs = sorted(requests_raw, key=lambda q: q["request_index"])
    store = rec.get("images")
    if not isinstance(store, dict):
        raise ValueError("proxy image store is missing")
    checked_images: set[str] = set()
    eligible = []
    metadata = []
    empty_completions = failed_requests = over_cap = 0
    for q in reqs:
        index = q["request_index"]
        if q.get("completed") is not True:
            raise ValueError(f"request {index}: response is incomplete")
        if type(q.get("status")) is not int:
            raise ValueError(f"request {index}: HTTP status is missing or invalid")
        summary = {k: v for k, v in q.items() if k not in ("prompt_token_ids", "token_ids", "sampler_topk")}
        summary["eligible"] = False
        metadata.append(summary)
        if ended_at is not None and isinstance(q.get("t"), (int, float)) and q["t"] + float(q.get("latency_s") or 0.0) > ended_at:
            summary["after_episode_end"] = True
            continue
        if q.get("status") != 200:
            failed_requests += 1
            continue
        prompt = validate_token_ids(q.get("prompt_token_ids"), name=f"request {index} prompt_token_ids", allow_empty=False)
        completion = validate_token_ids(q.get("token_ids"), name=f"request {index} token_ids")
        hashes = q.get("image_hashes")
        if not isinstance(hashes, list) or any(not isinstance(h, str) for h in hashes):
            raise ValueError(f"request {index}: image_hashes must be a list of strings")
        images = []
        for h in hashes:
            url = store.get(h)
            if not isinstance(url, str) or not url.startswith("data:image/") or ";base64," not in url:
                raise ValueError(f"request {index}: original image payload {h!r} is missing or not embedded")
            if h not in checked_images:
                actual_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()
                if actual_hash != h:
                    raise ValueError(f"request {index}: original image payload hash differs")
                try:
                    blob = base64.b64decode(url.split(",", 1)[1], validate=True)
                    with Image.open(io.BytesIO(blob)) as im:
                        im.verify()
                except Exception as exc:
                    raise ValueError(f"request {index}: original image payload is invalid") from exc
                checked_images.add(h)
            images.append(url)
        sampling = q.get("sampling")
        if not isinstance(sampling, dict):
            raise ValueError(f"request {index}: sampling settings are missing")
        try:
            validate_sampling_settings(sampling)
        except ValueError as exc:
            raise ValueError(f"request {index}: {exc}") from exc
        summary.update(prompt_len=len(prompt), completion_len=len(completion))
        if not completion:
            empty_completions += 1
            continue
        if max_request_tokens is not None and len(prompt) + len(completion) > max_request_tokens:
            summary["over_cap"] = True          # B1: never trainable (trainer hard cap); not a selection candidate
            over_cap += 1
            continue
        summary["eligible"] = True
        eligible.append({
            **({"task_credit": 0 if index in solved_suffix else 1} if solved_suffix is not None else {}),
            **(code_fields(code.get(index)) if code is not None else {}),
            "request_index": index, "prompt_token_ids": prompt, "token_ids": completion,
            "prompt_length": len(prompt), "completion_length": len(completion),
            "finish_reason": q.get("finish_reason"), "image_hashes": list(hashes), "images": images,
            **({"policy_version": q["policy_version"]} if type(q.get("policy_version")) is int else {}),
            "sampling": dict(sampling),
            **({"sampler_topk": q["sampler_topk"]} if isinstance(q.get("sampler_topk"), dict) else {}),
        })
    full_tokens = sum(q["completion_length"] for q in eligible)
    seed = int.from_bytes(hashlib.sha256(f"{SAMPLE_FORMAT}:{sid}".encode("utf-8")).digest()[:8], "big")
    selected_count = len(eligible) if max_training_requests < 0 else min(max_training_requests, len(eligible))
    chosen = sorted(random.Random(seed).sample(range(len(eligible)), selected_count))
    probability = selected_count / len(eligible) if eligible else 1.0
    selected = [dict(eligible[i], inclusion_probability=probability, loss_weight=1.0 / probability) for i in chosen]
    return {
        "sample_format": SAMPLE_FORMAT, "format_version": 1, "training_requests": selected,
        "full_completion_tokens": full_tokens, "eligible_request_count": len(eligible),
        "eligible_requests": [{k: q[k] for k in ("request_index", "prompt_length", "completion_length")} for q in eligible],
        "selected_request_count": selected_count, "selection_seed": seed,
        "selection_method": "uniform_without_replacement", "max_training_requests": max_training_requests,
        "failed_request_count": failed_requests, "empty_completion_request_count": empty_completions,
        **({"max_request_tokens": max_request_tokens, "over_cap_request_count": over_cap} if max_request_tokens is not None else {}),
        "requests": metadata,
    }


def _launch_gate() -> int:
    with LAUNCH_LOCK:
        now = time.time()
        wait = max(0.0, LAUNCH_STATE["last"] + STATE["launch_interval"] - now)
        LAUNCH_STATE["last"] = now + wait
        LAUNCH_STATE["count"] += 1
        devices = STATE["egl_devices"]
        dev = devices[LAUNCH_STATE["count"] % len(devices)]
    if wait > 0:
        time.sleep(wait)
    return dev


ARTIFACT_LINK = "public_sensor_artifacts"


def artifact_store_open(out_dir: Path, sid: str) -> Path | None:
    """Speed-up (config artifact_root, default None = off): the harness keeps its host-private
    observation store (capx HostObjectStore: SQLite with ~11 MB of pickled RGB-D per observe) in <episode>/public_sensor_artifacts,
    which on a sync-mounted NFS run directory costs ~2.0 s per observe vs 15 ms on tmpfs. With artifact_root set, that
    name becomes a symlink into a per-episode directory under artifact_root (/dev/shm: RAM, outside the model's bwrap sandbox,
    whose --dev /dev hides the host /dev/shm). The harness itself is unchanged; its rmtree of the symlink at finish is a
    no-op (ignore_errors), so artifact_store_close() removes the directory and the link after the episode process has exited."""
    root = STATE.get("artifact_root")
    if root is None:
        return None
    target = root / f"{os.getpid()}-{sid}"
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    link = out_dir / ARTIFACT_LINK
    if link.is_symlink() or link.exists():
        if link.is_symlink() or link.is_file():
            link.unlink()
        else:
            shutil.rmtree(link, ignore_errors=True)
    link.symlink_to(target, target_is_directory=True)
    return target


def artifact_store_close(out_dir: Path, target: Path | None) -> None:
    if target is None:
        return
    shutil.rmtree(target, ignore_errors=True)
    link = out_dir / ARTIFACT_LINK
    try:
        if link.is_symlink():
            link.unlink()
    except OSError:
        pass


def artifact_store_sweep(root: Path) -> int:
    """At env-server start: remove per-episode store directories of env-server processes that no longer exist (a killed server
    cannot clean up; tmpfs is RAM)."""
    removed = 0
    if not root.is_dir():
        return 0
    for d in root.iterdir():
        pid = d.name.split("-", 1)[0]
        if not pid.isdigit():
            continue
        alive = int(pid) == os.getpid()
        if not alive:
            try:
                os.kill(int(pid), 0)
                alive = True
            except ProcessLookupError:
                alive = False
            except PermissionError:
                alive = True
        if not alive:
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
    return removed


# ---------------------------------------------------------------- speed-up: co-critic early stop
# config early_stop = {"tau": 0.02, "k": 5, "min_request": 5, "control": 0.05, "critic_dir": ..., "critic_live": ...}; None = off.
# optional "exempt_tasks": [suite:task_id, ...] = episodes of these tasks are never stopped by the CRITIC rule
# (its verdict is logged as critic_would / would_cut), the success rule (success_k) still applies; absent = the behaviour below.
# From the min_request-th live request on, when the co-critic's V(s_t) has been < tau for k consecutive requests, the policy proxy
# refuses the session's next model request (HTTP 403); the driver ends its loop and this server records the episode as
# request_cap_exhausted (return = the run's exhaust_reward, -0.1), like a used-up request budget. A deterministic `control`
# fraction of sessions is never stopped (the rule's verdict is only logged) to measure wrongly cut successes online.
# V(s_t) comes from the harness's live request marks (flag file .live_request_marks -> the host appends every mark to
# hidden_request_marks.jsonl as it is taken; marks are host-only and invisible to the model) + calls.jsonl, featurised exactly
# like the trainer (critic_features.episode_states on complete lines only), with the co-critic weights the trainer exports
# after every critic update (critic_live).
LIVE_MARKS_FLAG = ".live_request_marks"
EARLY: dict[str, Any] = {"spec": None, "critic": None, "version": None, "lock": threading.Lock(), "log_lock": threading.Lock()}


def early_stop_control(sid: str, frac: float) -> bool:
    return int(hashlib.sha256(f"early-stop-control:{sid}".encode()).hexdigest()[:8], 16) / 2 ** 32 < frac


def early_stop_cut(values: list[float], tau: float, k: int, min_request: int) -> int | None:
    """First row position p (0-based over the live requests) where rows p-k+1..p all have V < tau, p >= max(k, min_request - 1);
    (cut position 'low')."""
    for p in range(max(k, min_request - 1), len(values)):
        if all(v < tau for v in values[p - k + 1:p + 1]):
            return p
    return None


def success_stop_cut(all_sat: list[bool], k: int) -> int | None:
    """Missed-finish fix (early_stop["success_k"]): first row position p where the marks of rows p-k+1..p
    all show every goal satisfied (the host already judges success and k requests passed without finish(), which would have ended
    the episode). The proxy then refuses the next request; the episode settles like a used-up budget: a missed finish while the host
    success still holds, else budget exhausted."""
    for p in range(k - 1, len(all_sat)):
        if all(all_sat[p - k + 1:p + 1]):
            return p
    return None


def _early_critic():
    """The co-critic with the newest exported weights (reloaded when critic_live changes); None until the trainer exported one."""
    spec = EARLY["spec"]
    live = Path(spec["critic_live"])
    try:
        st_ = live.stat()
    except OSError:
        return None, None
    version = (st_.st_mtime_ns, st_.st_size)
    if EARLY["critic"] is None or EARLY["version"] != version:
        import torch
        import critic_cotrain as CC
        torch.set_num_threads(2)
        blob = torch.load(live, map_location="cpu", weights_only=False)
        crit = EARLY["critic"] if EARLY["critic"] is not None else CC.CoCritic(spec["critic_dir"], 1, use_hidden=False)
        crit.load_state_dict(blob["state_dict"])
        crit.eval()
        EARLY.update(critic=crit, version=version, critic_step=blob.get("step"))
    return EARLY["critic"], EARLY.get("critic_step")


def _early_rows(out_dir: Path, snap: Path) -> list[dict]:
    """The episode's live critic rows from a snapshot of complete jsonl lines (a line being appended is left out)."""
    import critic_features as cf
    for rel in ("hidden_request_marks.jsonl", "calls.jsonl", "driver/requests.jsonl"):
        try:
            data = (out_dir / rel).read_bytes()
        except OSError:
            data = b""
        cut = data.rfind(b"\n")
        (snap / rel).parent.mkdir(parents=True, exist_ok=True)
        (snap / rel).write_bytes(data[:cut + 1] if cut >= 0 else b"")
    rows, _ = cf.episode_states(str(snap))
    return sorted((r for r in rows if not r["replayed"]), key=lambda r: r["request_index"])


def early_stop_begin(out_dir: Path, sid: str, suite: str, task_id: int, segment: dict | None) -> dict | None:
    spec = EARLY["spec"]
    if spec is None or (segment is not None and int(segment.get("K", 0)) != 0) or (segment or {}).get("monitor"):
        return None                      # off, a replayed-prefix segment (not a fresh start), or a held-out monitor episode (no early stop)
    (out_dir / LIVE_MARKS_FLAG).write_text("1\n")
    snap = Path(spec.get("snapshot_root") or "/dev/shm/captain-earlystop") / f"{os.getpid()}-{sid}"
    snap.mkdir(parents=True, exist_ok=True)
    es = {"sid": sid, "task": f"{suite}:{int(task_id)}", "control": early_stop_control(sid, float(spec.get("control", 0.0))),
          "snap": snap, "marks_size": -1, "n_rows": 0, "decision": None, "stopped": False, "t0": time.time(), "evals": 0,
          "errors": 0}
    if spec.get("exempt_tasks"):         # tasks exempt from the CRITIC rule (the success rule d still applies); absent = as before
        es["exempt"] = es["task"] in spec["exempt_tasks"]
        es["critic_would"] = None
    return es


def _early_evaluate(es: dict, out_dir: Path) -> None:
    spec = EARLY["spec"]
    try:
        size = (out_dir / "hidden_request_marks.jsonl").stat().st_size
    except OSError:
        return
    if size == es["marks_size"]:
        return
    es["marks_size"] = size
    try:
        with EARLY["lock"]:
            crit, cstep = _early_critic()
            if crit is None:
                return
            rows = _early_rows(out_dir, es["snap"])
            if not rows:
                return
            values = crit.values(crit.featurizer.featurize(rows, es["task"])).tolist()
        es["evals"] += 1
        es["n_rows"] = len(rows)
        sk = spec.get("success_k")
        if sk:                                   # part B first: a solved episode is never a critic cut
            ps = success_stop_cut([bool((r.get("f") or {}).get("all_sat")) for r in rows], int(sk))
            if ps is not None:
                es["decision"] = {"kind": "success", "row": ps, "request_index": rows[ps]["request_index"],
                                  "t_s": round(time.time() - es["t0"], 1), "success_k": int(sk), "n_rows": len(rows)}
                # applied to control episodes too: the control slice measures the CRITIC rule, and an episode that already holds
                # success for k requests is not a critic-rule case
                reason = f"missed-finish stop: all goals satisfied for {int(sk)} requests up to request {rows[ps]['request_index']} without finish"
                r = requests.post(f"{STATE['proxy']}/s/{es['sid']}/early_stop", json={"reason": reason}, timeout=30)
                es["stopped"] = r.status_code == 200
                return
        if es.get("exempt") and es["critic_would"] is not None:
            return                               # exempt: the critic verdict is already logged; keep watching for the success rule
        p = early_stop_cut(values, float(spec["tau"]), int(spec["k"]), int(spec.get("min_request", 5)))
        if p is None:
            return
        k = int(spec["k"])
        if es.get("exempt"):                     # exemption: log what the critic rule would have done, never stop, keep evaluating
            es["critic_would"] = {"row": p, "request_index": rows[p]["request_index"], "t_s": round(time.time() - es["t0"], 1),
                                  "v_window": [round(v, 4) for v in values[p - k + 1:p + 1]], "critic_step": cstep, "n_rows": len(rows),
                                  "kind": "critic"}
            return
        es["decision"] = {"row": p, "request_index": rows[p]["request_index"], "t_s": round(time.time() - es["t0"], 1),
                          "v_window": [round(v, 4) for v in values[p - k + 1:p + 1]], "critic_step": cstep, "n_rows": len(rows),
                          **({"kind": "critic"} if sk else {})}
        if not es["control"]:
            reason = (f"critic early stop: V < {spec['tau']} for {k} requests up to request {rows[p]['request_index']}")
            r = requests.post(f"{STATE['proxy']}/s/{es['sid']}/early_stop", json={"reason": reason}, timeout=30)
            es["stopped"] = r.status_code == 200
    except Exception as exc:  # noqa: BLE001 - the monitor must never break an episode
        es["errors"] += 1
        es["last_error"] = repr(exc)[:300]


def early_stop_wait(proc, timeout: float, es: dict, out_dir: Path) -> int:
    """proc.wait(timeout) that evaluates the early-stop rule whenever a new mark appears (until the first verdict)."""
    deadline = time.time() + timeout
    while True:
        try:
            return proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            if time.time() >= deadline:
                raise subprocess.TimeoutExpired(proc.args, timeout)
        if es["decision"] is None:
            _early_evaluate(es, out_dir)


def early_stop_settle(es: dict | None, out_dir: Path) -> None:
    """After the episode process ended: an episode the proxy refused at its next request (driver loop model_error ->
    infrastructure_error) is recorded as request_cap_exhausted, with the early-stop record kept in episode_result.json."""
    if es is None:
        return
    shutil.rmtree(es["snap"], ignore_errors=True)
    if not es["stopped"]:
        return
    path = out_dir / "episode_result.json"
    try:
        er = json.loads(path.read_text())
    except (OSError, ValueError):
        return
    driver = er.get("driver") or {}
    refused = er.get("termination_reason") == "infrastructure_error" and driver.get("loop_termination") == "model_error" \
        and "request cap reached early" in str(driver.get("loop_reason") or er.get("error") or "")
    es["refused"] = refused
    if not refused:
        return                         # the episode ended on its own (finish, wall time, ...) before the next request
    er["early_stop"] = {**(es["decision"] or {}), "termination_raw": er.get("termination_reason"), "error_raw": er.pop("error", None)}
    er["termination_reason"] = "request_cap_exhausted"
    tmp = path.with_name(".episode_result.json.tmp")
    tmp.write_text(json.dumps(er, indent=2, sort_keys=True))
    os.replace(tmp, path)


def early_stop_record(es: dict | None, result: dict, step) -> None:
    if es is None:
        return
    diag = result.setdefault("diagnostics", {})
    row = {"step": step, "sid": es["sid"], "task": es["task"], "control": es["control"], "would_cut": es["decision"] is not None and (es["decision"] or {}).get("kind", "critic") == "critic",
           "cut": bool(es.get("refused")), "proxy_flagged": es["stopped"], "decision": es["decision"], "evals": es["evals"],
           "errors": es["errors"], "last_error": es.get("last_error"), "verdict": result.get("verdict"),
           "termination": diag.get("termination_reason"), "final_success": diag.get("final_success"),
           "host_final_success": diag.get("host_final_success"), "env_wall_s": result.get("env_wall_s")}
    if "exempt" in es:                   # exemption on (exempt_tasks configured): the flag + the critic rule's logged verdict
        row["exempt"] = es["exempt"]
        if es["exempt"]:
            row["would_cut"] = es["critic_would"] is not None
            row["critic_would"] = es["critic_would"]
    diag["early_stop"] = {k: row[k] for k in ("control", "would_cut", "cut", "decision", "errors") + (("exempt",) if "exempt" in row else ())}
    try:
        with EARLY["log_lock"], open(STATE["root"].parent / "early_stop.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")
    except OSError:
        pass


def episode_harness(segment: dict | None) -> Path:
    """held-out monitor: an entry with harness_alt runs on that registered harness copy (same code, other start
    data, e.g. the ev12 held-out layouts); everything else on the run's harness"""
    alt = (segment or {}).get("harness_alt")
    if not alt:
        return STATE["harness"]
    if alt not in STATE.get("alt_harness", {}):
        raise ValueError(f"unknown harness_alt {alt!r}")
    return STATE["alt_harness"][alt]


def episode_command(*, suite: str, task_id: int, seed: int, sid: str, out_dir: Path, egl_device: int,
                    max_requests: int | None = None, harness: Path | None = None) -> list[str]:
    h = harness or STATE["harness"]
    cfg = STATE["episode"]
    cmd = [str(h / ".venv-libero" / "bin" / "python"), str(h / "scripts" / "react_episode.py"),
           "--suite", suite, "--task-id", str(task_id), "--seed", str(seed),
           "--model", STATE["model_name"], "--api", "chat", "--auth", "none", "--endpoint", f"{STATE['proxy']}/s/{sid}/v1",
           "--profile", cfg["profile"], "--output-dir", str(out_dir), "--root", str(h),
           "--mujoco-gl", "egl", "--egl-device", str(egl_device),
           # student profile: --small-context-window N = catalog.small_context_profile(profile, context_window=N); else --context-window N
           ("--small-context-window" if cfg.get("student_profile") else "--context-window"), str(cfg["context_window"]),
           "--max-output-tokens", str(cfg["max_output_tokens"]),
           "--max-images", str(cfg["max_images"]), "--wall-time", str(cfg["wall_time"]), "--grace-s", str(cfg["grace_s"]),
           "--max-actions", str(cfg["max_actions"]), "--max-tool-calls", str(cfg["max_tool_calls"]),
           "--sam3-url", STATE["sam3_url"], "--molmo-url", STATE["molmo_url"]]
    if max_requests is not None or cfg.get("max_requests"):
        cmd += ["--max-requests", str(max_requests if max_requests is not None else cfg["max_requests"])]
    if cfg.get("runtime_tool_contract"):
        cmd += ["--runtime-tool-contract", cfg["runtime_tool_contract"]]
    if cfg.get("inline_task"):
        cmd += ["--inline-task"]
    if cfg.get("enable_thinking") in ("on", "off"):
        cmd += ["--enable-thinking", cfg["enable_thinking"]]
    if cfg.get("no_history_reasoning"):
        cmd += ["--no-history-reasoning"]
    if cfg.get("student_profile"):
        sp = cfg["student_profile"]
        cmd += ["--context-strategy", sp["strategy"], "--context-limit", str(sp["context_limit"]), "--keep-recent-tokens", str(sp["keep_recent_tokens"]),
                "--tool-output-tokens", str(sp["tool_output_tokens"])]
        if sp.get("spill"):
            cmd += ["--spill-tool-output"]
        if sp.get("hard_limit"):
            cmd += ["--context-hard-limit", str(sp["hard_limit"])]   # trigger = hard_limit - one turn
    if cfg.get("student_python"):
        cmd += ["--student-python", cfg["student_python"]]
    for hidden in [h for h in (STATE.get("root"), *(cfg.get("sandbox_hide") or [])) if h]:
        cmd += ["--sandbox-hide", str(hidden)]     # other rollouts and the teacher episodes are not the model's to read
    cmd += list(cfg.get("extra_args") or [])
    return cmd


PREFIX_SLACK = 4   # tokens at the end of a prompt that may re-tokenize once the assistant turn is appended (generation prompt boundary)


def segment_requests(requests: list[dict], compactions_per_request: list[int] | None = None) -> list[int]:
    """Indices (into the proxy record's request list) of the requests that end a segment: the last successful request and every
    request after which the driver compacted its history. Exact source: driver/views.jsonl (`compactions` counter per request, same
    length as the proxy record); fallback: request i+1's prompt does not extend prompt i (misses a compaction that only dropped the
    newest turn, because the change then sits inside the generation-prompt slack)."""
    ok = [i for i, q in enumerate(requests) if q.get("prompt_token_ids") and q.get("token_ids") is not None]
    ends: list[int] = []
    views = compactions_per_request if compactions_per_request and len(compactions_per_request) == len(requests) else None
    for a, b in zip(ok, ok[1:]):
        if views is not None:
            if views[b] > views[a]:
                ends.append(a)
            continue
        pa, pb = requests[a]["prompt_token_ids"], requests[b]["prompt_token_ids"]
        n = max(0, len(pa) - PREFIX_SLACK)
        if len(pb) < len(pa) or pb[:n] != pa[:n]:
            ends.append(a)
    if ok:
        ends.append(ok[-1])
    return ends


def view_compactions(episode_dir: Path) -> list[int] | None:
    """`compactions` counter of every model request from driver/views.jsonl, ordered by request index; None if absent."""
    rows = []
    try:
        for line in (Path(episode_dir) / "driver" / "views.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    rows.sort(key=lambda r: int(r.get("request") or 0))
    return [int(r.get("compactions") or 0) for r in rows] or None


def _request_sample(reqs: list[dict], i: int, store: dict) -> dict:
    q = reqs[i]
    hashes = q.get("image_hashes") or []
    images = [store.get(h) for h in hashes]
    return {"request": i + 1, "prompt_token_ids": q["prompt_token_ids"], "token_ids": q["token_ids"], "finish_reason": q.get("finish_reason"),
            "images": images, "images_missing": sum(1 for u in images if u is None), "n_images": len(hashes)}


def build_segments(rec: dict, max_segments: int, compactions_per_request: list[int] | None = None, max_repairs: int = 6,
                   rng: random.Random | None = None, tokenizer=None) -> tuple[list[dict], list[dict], dict]:
    """Legacy archive reconstruction only; the live service never calls this.

    Every segment carries `history`: for each assistant block inside its prompt, the proxy request that produced it and whether the
    block is token-for-token the completion the policy sampled (`exact`). The driver re-renders every earlier turn from its parsed form,
    so an unparseable or non-canonical output shows up in later prompts as a different (canonical) block - training that block would
    give the gradient to text the policy never produced. Requests whose sampled completion is not present exactly in
    any kept segment become `repairs`: their own prompt + sampled completion, trained on the final block only. At most `max_repairs`
    are stored (uniform sample, loss weight candidates/stored keeps the expectation unbiased); the rest are reported in diagnostics."""
    reqs = rec.get("requests") or []
    store = rec.get("images") or {}
    ok = [i for i, q in enumerate(reqs) if q.get("prompt_token_ids") and q.get("token_ids") is not None]
    ends = segment_requests(reqs, compactions_per_request)
    segs = []
    for i in ends:
        sg = _request_sample(reqs, i, store)
        ids = with_im_end(list(sg["prompt_token_ids"]) + list(sg["token_ids"]))
        history, aligned = align_history(ids, i, ok, reqs, tokenizer=tokenizer)
        sg["history"] = history
        sg["alignment_ok"] = aligned
        segs.append(sg)
    dropped_ends = []
    if max_segments and len(segs) > max_segments:
        dropped_ends = ends[:-max_segments]
        segs = segs[-max_segments:]   # keep the most recent segments (the earliest ones lose their gradient)
    covered = {sg["request"] for sg in segs}
    first_request = (dropped_ends[-1] + 2) if dropped_ends else 1     # requests up to the last dropped segment end are out of scope
    n_hist = n_exact = n_text = 0
    for sg in segs:
        for h in sg["history"]:
            n_hist += 1
            n_text += 1 if h.get("text_equal") else 0
            if h.get("exact") and h.get("request") is not None:
                n_exact += 1
                covered.add(int(h["request"]))
    candidates = [i for i in ok if i + 1 >= first_request and (i + 1) not in covered]
    kept = candidates
    if max_repairs >= 0 and len(candidates) > max_repairs:
        kept = sorted((rng or random).sample(candidates, max_repairs))
    weight = (len(candidates) / len(kept)) if kept else 1.0
    repairs = []
    for i in kept:
        rp = _request_sample(reqs, i, store)
        rp["weight"] = round(weight, 4)
        repairs.append(rp)
    diag = {"history_blocks": n_hist, "history_exact": n_exact, "history_text_equal": n_text, "history_rerendered": n_hist - n_exact,
            "alignment_ok": all(sg["alignment_ok"] for sg in segs), "repair_candidates": len(candidates), "repairs_stored": len(repairs),
            "repair_weight": round(weight, 4), "segments_dropped": len(dropped_ends), "tokenizer": tokenizer is not None}
    return segs, repairs, diag


def store_raw_record(rec: dict, root: Path, step, sid: str) -> Path:
    """The complete proxy record of an episode (every request's prompt/completion token ids, sampling settings, sampler
    top-k, the deduplicated image store) as gzipped JSON, written atomically before the session is deleted: requests
    that were not selected for backprop stay recoverable for audits and later re-selection."""
    import gzip

    path = Path(root) / f"step-{step}" / f"{sid}.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=3) as fh:
        json.dump(rec, fh, separators=(",", ":"))
    os.replace(tmp, path)
    return path


def write_replay_prefix(entry: dict, path: Path) -> int:
    """The first K recorded model responses of the segment's source episode, verbatim (react.replay input)."""
    k = int(entry["K"])
    if k == 0:
        return 0
    rows = [json.loads(line) for line in Path(entry["source_responses"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = sorted(rows, key=lambda r: int(r.get("request") or 0))[:k]
    if [r.get("request") for r in rows] != list(range(1, k + 1)):
        raise ValueError(f"segment source has no contiguous prefix of {k} responses")
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return k


def run_episode(body: dict, segment: dict | None = None) -> dict:
    t_slot = time.time()                 # this worker thread (one of --max-concurrency slots) is now held by the episode
    t_arrival = float(body.get("_t_arrival_wall") or t_slot)
    suite, task_id, seed = body["suite"], int(body["task_id"]), int(body["seed"])
    sid = body.get("sid") or uuid.uuid4().hex[:16]
    step = body.get("step", "na")
    root = STATE["root"] / f"step-{step}"
    out_dir = root / (f"{suite}_{task_id}_seed-{seed}_{sid}" if segment is None else f"{suite}_{task_id}_seed-{seed}_K{segment['K']}_{sid}")
    if out_dir.is_dir() and any(out_dir.iterdir()):
        # a relaunched trainer step reusing a sid: never let the driver see a finished episode's files (it exits at once and the stale
        # episode_result.json gets scored as infrastructure_error)
        out_dir.rename(out_dir.with_name(out_dir.name + f".stale-{int(time.time())}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = _launch_gate()
    if segment is None:
        command = episode_command(suite=suite, task_id=task_id, seed=seed, sid=sid, out_dir=out_dir, egl_device=dev)
        hroot = STATE["harness"]
    else:
        hroot = episode_harness(segment)
        prefix = out_dir / "replay_prefix.jsonl"
        k = write_replay_prefix(segment, prefix)
        command = episode_command(suite=suite, task_id=task_id, seed=seed, sid=sid, out_dir=out_dir, egl_device=dev,
                                  max_requests=k + int(segment["horizon"]), harness=hroot)
        command += ["--mark-requests"] + (["--replay-prefix", str(prefix)] if k else [])
        if k and segment.get("source_dir"):
            command += ["--sandbox-hide", str(segment["source_dir"])]
        if k and segment.get("handoff") == "clean":
            command += ["--handoff", "clean"]
    env = dict(os.environ)
    env.update({"PYTHONPATH": str(hroot / "src"), "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
                "MUJOCO_EGL_DEVICE_ID": str(dev), "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                "__EGL_VENDOR_LIBRARY_FILENAMES": env.get("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")})
    # no provider credentials in the episode process (the model's sandbox inherits a filtered copy of this environment)
    for k in [k for k in env if k.startswith("OPENAI_") or any(w in k for w in ("TOKEN", "KEY", "SECRET", "PASSW"))]:
        env.pop(k, None)
    store = artifact_store_open(out_dir, sid)
    es = early_stop_begin(out_dir, sid, suite, task_id, segment)
    started = time.time()
    with ACTIVE_LOCK:
        ACTIVE[sid] = started
    timeout = STATE["episode"]["wall_time"] + STATE["episode"]["grace_s"] + 900  # server build + finalize + stragglers
    log = out_dir / "driver.log"
    rc: int
    with log.open("w") as fh:
        fh.write(" ".join(command) + "\n")
        fh.flush()
        try:
            proc = subprocess.Popen(command, cwd=str(hroot), env=env, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    start_new_session=True)
            rc = proc.wait(timeout=timeout) if es is None else early_stop_wait(proc, timeout, es, out_dir)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, 9)
            except Exception:  # noqa: BLE001
                pass
            rc = -9
        finally:
            artifact_store_close(out_dir, store)
    ended_at = time.time()
    elapsed = ended_at - started
    with ACTIVE_LOCK:
        ACTIVE.pop(sid, None)
    # a driver stopped at the wall-time deadline may leave a model request in flight: let the proxy finish recording it
    for _ in range(150):
        try:
            # counters only: the record route copies and encodes the whole episode (seconds of proxy time per call)
            if not requests.get(f"{STATE['proxy']}/s/{sid}/status", timeout=30).json().get("in_flight"):
                break
        except Exception:  # noqa: BLE001 - the full fetch below reports a broken proxy
            break
        time.sleep(2)
    early_stop_settle(es, out_dir)
    if es is not None and es.get("refused"):
        es["rc_raw"], rc = rc, 0          # the driver exits 2 on the refused request; the episode is a policy outcome
    # proxy record first: the reward (v11b verbosity term) reads the per-request token counts
    rec: dict | None = None
    proxy_error = None
    try:
        r = requests.get(f"{STATE['proxy']}/s/{sid}/record", params={"full": 1}, timeout=120)
        if r.status_code == 200:
            rec = r.json()
    except Exception as exc:  # noqa: BLE001
        proxy_error = repr(exc)
    if segment is None:
        result = score_episode(out_dir, proxy_requests=(rec or {}).get("requests"))
    else:
        from reward_segment import score_segment
        result = score_segment(out_dir, segment)
    result["rc"] = rc
    result["env_wall_s"] = round(elapsed, 1)
    early_stop_record(es, result, step)
    result["sid"] = sid
    result["episode_dir"] = str(out_dir)
    if proxy_error:
        result["diagnostics"]["proxy_error"] = proxy_error
    # proxy record -> training sample
    sample_path = out_dir / "rollout.json"
    if rec is not None and STATE.get("raw_record_dir"):
        try:
            result["diagnostics"]["raw_record"] = str(store_raw_record(rec, STATE["raw_record_dir"], step, sid))
        except OSError as exc:            # a record that cannot be stored makes the rollout unusable for training
            result["diagnostics"]["raw_record_error"] = repr(exc)
            rec = None
    try:
        requests.delete(f"{STATE['proxy']}/s/{sid}", timeout=30)
    except Exception:  # noqa: BLE001
        pass
    n_req = len((rec or {}).get("requests") or [])
    last = (rec or {}).get("last_response") or {}
    sample = None
    sample_error = None
    try:
        if rec is None:
            raise ValueError("the proxy returned no episode record")
        cap = STATE["max_training_requests"]
        if segment is not None and segment.get("train_all_on_success") and (result.get("diagnostics") or {}).get("final_success"):
            cap = -1                     # a successful trajectory contributes every request (p = 1)
        solved_suffix = None
        if segment is not None and STATE.get("solved_suffix_credit"):
            # a successful trajectory's requests made when every goal already held and kept holding: zero task credit
            from reward_segment import solved_suffix_requests
            credit = solved_suffix_requests(out_dir, rec.get("requests") or [], bool((result.get("diagnostics") or {}).get("final_success")))
            solved_suffix = credit["proxy_request_indices"]
            result["diagnostics"].update(solved_suffix_requests=len(solved_suffix), credit_unmapped=credit["unmapped"])
        code = None
        if STATE.get("code_reward"):
            # per-request code-execution cost from the host audit, joined to the proxy records by response id
            from code_reward import proxy_request_costs
            costs = proxy_request_costs(out_dir, rec.get("requests") or [], STATE["code_reward"])
            code = costs["labels"]
            result["diagnostics"].update(code_reward_version=costs["version"], code_categories=costs["category_counts"],
                                         code_unmapped=costs["unmapped"], code_episode_cost=costs["episode_cost_sum"],
                                         **({"interface": costs["interface"]} if "interface" in costs else {}))
        sample = build_training_sample(rec, sid=sid, max_training_requests=cap, ended_at=ended_at, solved_suffix=solved_suffix,
                                       code=code, **({"max_request_tokens": STATE["max_request_tokens"]}
                                                     if STATE.get("max_request_tokens") is not None else {}))
        if code is not None:
            sample["code_reward_version"] = costs["version"]
            result["diagnostics"]["code_selected_cost"] = sum(q.get("code_cost", 0.0) for q in sample["training_requests"])
            result["diagnostics"]["code_selected_errors"] = sum(1 for q in sample["training_requests"] if q.get("code_cost", 0.0) > 0)
            result["diagnostics"]["code_selected_delta"] = sum(1 for q in sample["training_requests"] if q.get("code_delta_cost", 0.0) > 0)
    except ValueError as exc:
        sample_error = str(exc)
    if sample_error is not None:
        result["diagnostics"]["no_sample"] = True
        result["diagnostics"]["sample_error"] = sample_error
        result["sample_path"] = None
        result["verdict"] = "infrastructure_error"
        result["reward"] = 0.0
        result["components"] = {}
        result["diagnostics"]["infrastructure"] = True
        (out_dir / "sample_error.json").write_text(json.dumps({"sid": sid, "error": sample_error}), encoding="utf-8")
    else:
        sample.update(sid=sid, suite=suite, task_id=task_id, seed=seed, step=step,
                      reward=result["reward"], verdict=result["verdict"])
        if segment is not None:
            sample.update(start_id=segment.get("start_id"), K=int(segment["K"]), horizon=int(segment["horizon"]))
        write_rollout_json_atomic(sample_path, sample)
        result["sample_path"] = str(sample_path) if sample["full_completion_tokens"] else None
        result["diagnostics"]["n_requests_proxy"] = n_req
        result["diagnostics"]["sample_format"] = SAMPLE_FORMAT
        result["diagnostics"]["n_training_requests"] = sample["selected_request_count"]
        result["diagnostics"]["eligible_request_count"] = sample["eligible_request_count"]
        result["diagnostics"]["full_completion_tokens"] = sample["full_completion_tokens"]
        result["diagnostics"]["empty_completion_requests"] = sample["empty_completion_request_count"]
        if "over_cap_request_count" in sample:      # over-length guard: requests above the trainer's hard cap, excluded before selection
            result["diagnostics"]["over_cap_request_count"] = sample["over_cap_request_count"]
            if sample["over_cap_request_count"]:
                with ACTIVE_LOCK:
                    STATE["over_cap_requests"] = STATE.get("over_cap_requests", 0) + sample["over_cap_request_count"]
                    STATE["over_cap_episodes"] = STATE.get("over_cap_episodes", 0) + 1
                    if not sample["full_completion_tokens"]:
                        STATE["over_cap_untrainable_episodes"] = STATE.get("over_cap_untrainable_episodes", 0) + 1
                if not sample["full_completion_tokens"]:
                    result["diagnostics"]["no_trainable_requests"] = "over_cap"
        result["diagnostics"]["images_missing"] = 0
        result["diagnostics"]["final_prompt_tokens"] = len(last.get("prompt_token_ids") or [])
        result["diagnostics"]["final_completion_tokens"] = len(last.get("token_ids") or [])
        result["diagnostics"]["sum_completion_tokens"] = sample["full_completion_tokens"]
        try:                                   # learner scheduling estimate after exact-prefix packing (trainer --pack-prefixes)
            from prefix_chains import packed_cost
            result["diagnostics"].update(packed_cost(sample["training_requests"]))
        except Exception as exc:  # noqa: BLE001 - a missing estimate only degrades rank balancing
            result["diagnostics"]["packed_cost_error"] = repr(exc)
    if rc != 0 and result["verdict"] == "infrastructure_error":
        try:
            result["diagnostics"]["log_tail"] = log.read_text(errors="replace")[-1500:]
        except Exception:  # noqa: BLE001
            pass
    # disk hygiene: the observation store is large; keep PNGs/JSON only
    for sub in ("public_sensor_artifacts", "observations"):
        shutil.rmtree(out_dir / sub, ignore_errors=True)
    # wall clocks for the trainer's slot-occupancy metrics (infra_pipeline.slot_stats): POST received, slot taken, driver
    # launched (after the EGL launch gate), driver exited, slot released
    result.update(t_arrival_wall=round(t_arrival, 3), t_start_wall=round(t_slot, 3), t_driver_start_wall=round(started, 3),
                  t_driver_end_wall=round(ended_at, 3), t_end_wall=round(time.time(), 3))
    return result


@app.post("/rollout")
async def rollout(request: Request):
    body = await request.json()
    body["_t_arrival_wall"] = time.time()
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(STATE["pool"], run_episode, body)


def run_segment(body: dict) -> dict:
    """POST /segment {entry: <segment-bank start>, horizon, sid, step, group}: replay the start's prefix, run H live requests."""
    entry = dict(body["entry"])
    if STATE["episode"].get("runtime_tool_contract") != "python-runtime/v1":
        raise ValueError("segment rollouts need --runtime-tool-contract python-runtime/v1")
    entry["horizon"] = max(1, int(body.get("horizon") or entry.get("horizon") or 1))
    result = run_episode({"suite": entry["suite"], "task_id": entry["task_id"], "seed": entry["seed"], "sid": body.get("sid"),
                          "step": body.get("step", "na"), "_t_arrival_wall": body.get("_t_arrival_wall")}, segment=entry)
    result.update(start_id=entry.get("start_id"), trajectory_id=entry.get("trajectory_id"), K=int(entry["K"]), horizon=entry["horizon"])
    return result


@app.post("/segment")
async def segment(request: Request):
    body = await request.json()
    body["_t_arrival_wall"] = time.time()
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(STATE["pool"], run_segment, body)


@app.get("/health")
async def health():
    with ACTIVE_LOCK:
        active = len(ACTIVE)
        oldest = round(time.time() - min(ACTIVE.values()), 0) if ACTIVE else 0
    return {"ok": True, "active": active, "oldest_active_s": oldest, "max_concurrency": STATE["max_concurrency"],
            "sample_format": SAMPLE_FORMAT, "max_training_requests": STATE["max_training_requests"],
            "reward_version": REWARD_VERSION, "goal_audit_schema": GOAL_AUDIT_SCHEMA if REWARD_VERSION in ("v12a", "v12b") else None,
            "tool_audit_schema": "tool-execution-audit/v1" if STATE["episode"].get("runtime_tool_contract") else None,
            "segment_reward_version": "v13s" if STATE["episode"].get("runtime_tool_contract") == "python-runtime/v1" else None,
            "environment_preflight": STATE.get("environment_preflight"),
            **({"max_request_tokens": STATE["max_request_tokens"],
                "over_cap": {k: STATE.get(f"over_cap_{k}", 0) for k in ("requests", "episodes", "untrainable_episodes")}}
               if STATE.get("max_request_tokens") is not None else {}),
            "episode": STATE["episode"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", type=Path, default=Path(__file__).resolve().parents[1] / "harness")
    ap.add_argument("--root", type=Path, default=Path("/tmp/captain-rl-env"))
    ap.add_argument("--port", type=int, default=8400)
    ap.add_argument("--host", default="127.0.0.1", help="bind address (0.0.0.0 when another node must reach it)")
    ap.add_argument("--proxy", default="http://127.0.0.1:8510")
    ap.add_argument("--model-name", default="rl-policy")
    ap.add_argument("--max-concurrency", type=int, default=48)
    ap.add_argument("--egl-devices", default="0,1")
    ap.add_argument("--launch-interval", type=float, default=1.0)
    ap.add_argument("--sam3-url", default="http://127.0.0.1:8114")
    ap.add_argument("--molmo-url", default="http://127.0.0.1:8122/v1")
    ap.add_argument("--profile", default="captain", help="driver request profile (captain = the template CaPtain-4B is trained with)")
    ap.add_argument("--runtime-tool-contract", choices=("code-mode-runtime/v2", "python-runtime/v1"), default=None)
    ap.add_argument("--context-window", type=int, default=65536)
    ap.add_argument("--max-output-tokens", type=int, default=4096)
    ap.add_argument("--max-images", type=int, default=12)
    ap.add_argument("--wall-time", type=float, default=900.0)
    ap.add_argument("--grace-s", type=float, default=60.0)
    ap.add_argument("--max-requests", type=int, default=40)
    ap.add_argument("--max-actions", type=int, default=60)
    ap.add_argument("--max-tool-calls", type=int, default=200)
    ap.add_argument("--enable-thinking", default="server")
    ap.add_argument("--no-history-reasoning", action="store_true")
    ap.add_argument("--inline-task", action="store_true", help="inline TASK.md + initial frames into the first request (must match how the SFT data was exported; CaPtain-4B-SFT: on)")
    ap.add_argument("--extra-arg", action="append", default=[])
    ap.add_argument("--student-python", default=None,
                    help="the pinned interpreter of the model's own Python session (react_episode --student-python; checked in the sandbox per episode)")
    ap.add_argument("--raw-record-dir", type=Path, default=None,
                    help="store every episode's complete proxy record (all requests, images, sampler top-k) here before training selection")
    ap.add_argument("--sandbox-hide", action="append", default=[], help="directories no rollout's sandbox may see (teacher episodes, other runs)")
    ap.add_argument("--student-profile", action="store_true", help="pass the student context policy (window strategy, --context-limit/--keep-recent-tokens/--tool-output-tokens/--spill-tool-output)")
    ap.add_argument("--context-limit", type=int, default=60000)
    ap.add_argument("--keep-recent-tokens", type=int, default=16000)
    ap.add_argument("--tool-output-tokens", type=int, default=4000)
    ap.add_argument("--context-strategy", default="window", choices=("window", "summary", "notes"))
    ap.add_argument("--no-spill-tool-output", action="store_true")
    ap.add_argument("--context-hard-limit", type=int, default=65536, help="no request may exceed this many real tokens (training sample cap); 0 disables")
    ap.add_argument("--code-reward", choices=("python_exec_cost_v1", "python_exec_contract_v2"), default=None,
                    help="attach per-request code-execution costs (rl/code_reward.py) to the training records")
    ap.add_argument("--solved-suffix-credit", action="store_true",
                    help="segment rollouts: a successful trajectory's requests issued once every goal held for good get task_credit 0")
    ap.add_argument("--alt-harness", action="append", default=[],
                    help="NAME=PATH: another harness copy an entry may select with harness_alt (held-out monitor starts)")
    ap.add_argument("--early-stop", default=None,
                    help="JSON {tau, k, min_request, control, critic_dir, critic_live} of the co-critic early stop; default off")
    ap.add_argument("--artifact-root", type=Path, default=None,
                    help="per-episode harness host object store under this local directory (e.g. /dev/shm/captain-artifacts) "
                         "instead of <episode>/public_sensor_artifacts on NFS; default off")
    ap.add_argument("--max-training-requests", type=int, default=4,
                    help="uniformly select at most N original requests per episode; -1 keeps all")
    ap.add_argument("--max-request-tokens", type=int, default=None,
                    help="over-length guard: a request with prompt + completion tokens above this (the trainer's hard cap) is not eligible "
                         "for training selection (recorded as over_cap); default off")
    ap.add_argument("--max-segments", type=int, default=None, help="deprecated compatibility argument; ignored")
    ap.add_argument("--max-repairs", type=int, default=None, help="deprecated compatibility argument; ignored")
    ap.add_argument("--tokenizer", default=None, help="deprecated compatibility argument; raw requests need no tokenizer")
    args = ap.parse_args()
    if args.max_training_requests == 0 or args.max_training_requests < -1:
        ap.error("--max-training-requests must be positive or -1")
    if args.max_request_tokens is not None and args.max_request_tokens < 1:
        ap.error("--max-request-tokens must be positive")
    if REWARD_VERSION == "v12b" and args.runtime_tool_contract not in ("code-mode-runtime/v2", "python-runtime/v1"):
        ap.error("v12b requires --runtime-tool-contract code-mode-runtime/v2 or python-runtime/v1")
    preflight = None
    if args.runtime_tool_contract:
        from toolchain_preflight import load_preflight_receipt, validate_receipt
        import socket
        try:
            preflight = load_preflight_receipt(os.environ.get("FRESH_ENV_PREFLIGHT_PATH"), contract=args.runtime_tool_contract)
            validate_receipt(preflight, hostname=socket.gethostname().split(".")[0])
        except (OSError, ValueError) as exc:
            ap.error(f"episode toolchain preflight failed: {exc}")
    if args.max_segments is not None or args.max_repairs is not None or args.tokenizer is not None:
        print("[env_server] legacy segment/repair/tokenizer options ignored; emitting per_request_v1", flush=True)
    STATE.update({
        "harness": args.harness.resolve(), "root": args.root.resolve(), "proxy": args.proxy, "model_name": args.model_name,
        "max_concurrency": args.max_concurrency, "pool": ThreadPoolExecutor(max_workers=args.max_concurrency),
        "egl_devices": [int(x) for x in args.egl_devices.split(",")], "launch_interval": args.launch_interval,
        "sam3_url": args.sam3_url, "molmo_url": args.molmo_url,
        "environment_preflight": preflight,
        "episode": {"profile": args.profile, "context_window": args.context_window, "max_output_tokens": args.max_output_tokens,
                    "runtime_tool_contract": args.runtime_tool_contract,
                    "max_images": args.max_images, "wall_time": args.wall_time, "grace_s": args.grace_s, "max_requests": args.max_requests,
                    "max_actions": args.max_actions, "max_tool_calls": args.max_tool_calls, "enable_thinking": args.enable_thinking,
                    "no_history_reasoning": args.no_history_reasoning, "inline_task": bool(args.inline_task), "extra_args": args.extra_arg,
                    "student_python": args.student_python, "sandbox_hide": list(args.sandbox_hide),
                    "student_profile": ({"strategy": args.context_strategy, "context_limit": args.context_limit, "keep_recent_tokens": args.keep_recent_tokens,
                                         "tool_output_tokens": args.tool_output_tokens, "spill": not args.no_spill_tool_output,
                                         "hard_limit": args.context_hard_limit or None} if args.student_profile else None)},
        "max_training_requests": args.max_training_requests, "solved_suffix_credit": bool(args.solved_suffix_credit),
        "code_reward": args.code_reward,
        "raw_record_dir": args.raw_record_dir,
    })
    if args.max_request_tokens is not None:
        STATE["max_request_tokens"] = args.max_request_tokens
        print(f"[env_server] max_request_tokens={args.max_request_tokens} (over-length guard: longer requests are not eligible)", flush=True)
    STATE["alt_harness"] = {}
    for item in args.alt_harness:
        name, _, path = item.partition("=")
        if not name or not path or not Path(path, "src").is_dir():
            ap.error(f"--alt-harness {item!r}: NAME=<harness root>")
        STATE["alt_harness"][name] = Path(path).resolve()
    if args.early_stop:
        spec = json.loads(args.early_stop)
        missing = {"tau", "k", "critic_dir", "critic_live"} - set(spec)
        if missing:
            ap.error(f"--early-stop lacks {sorted(missing)}")
        EARLY["spec"] = spec
        print(f"[env_server] early_stop={spec}", flush=True)
    if args.artifact_root is not None:
        STATE["artifact_root"] = args.artifact_root.resolve()
        STATE["artifact_root"].mkdir(parents=True, exist_ok=True)
        print(f"[env_server] artifact_root={STATE['artifact_root']} (swept {artifact_store_sweep(STATE['artifact_root'])} stale stores)", flush=True)
    STATE["root"].mkdir(parents=True, exist_ok=True)
    print(f"[env_server] harness={STATE['harness']} proxy={args.proxy} port={args.port} concurrency={args.max_concurrency} "
          f"egl={STATE['egl_devices']} sample_format={SAMPLE_FORMAT} max_training_requests={args.max_training_requests} episode={STATE['episode']}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", workers=1)


if __name__ == "__main__":
    main()
