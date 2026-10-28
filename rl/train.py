#!/usr/bin/env python3
"""Group-normalized policy gradients on original per-request policy samples.

Live training requires per_request_v1: original prompts, sampled completion IDs,
original images, and explicit inclusion probabilities. No re-rendered history is
used as a loss target. Request subsampling weights the entire policy/KL objective
against a fixed, pre-sampling completion-token denominator. Legacy segment helpers
remain solely for reading historical artifacts, never for the live training loop.

    torchrun --nproc_per_node=4 rl/train.py --model <init> --output-dir runs/<run> ...   (rl/v11_launch.sh train)
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import torch
import torch.distributed as dist
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reward import COMPONENT_KEYS, VERDICTS  # noqa: E402
from raw_request_training import LM_HEAD_CHUNK, InvalidRollout, build_request_samples, request_token_values, sampled_objective  # noqa: E402
from raw_request_training import OverLengthRollout, overlength_drop_set  # noqa: E402
from physics_critic import batch_advantages, proxy_values  # noqa: E402
from rollout_storage import read_rollout_json  # noqa: E402
from training_resume import load_resume, prepare_resume_logs  # noqa: E402
from infra_pipeline import lookahead_draw, pack_plan, slot_stats  # noqa: E402
from chat_tokens import IM_END, IM_START, NEWLINE, NLNL, ROLE_ASSISTANT, THINK_CLOSE, THINK_OPEN, assistant_blocks, assistant_mask, with_im_end  # noqa: E402,F401

ENV_SERVERS = [u.strip() for u in os.environ.get("RL_ENV_SERVER", "http://127.0.0.1:8400").split(",") if u.strip()]   # 2-node RL: comma list
ENV_SERVER = ENV_SERVERS[0]
POLICY_URLS = [u.strip() for u in os.environ.get("RL_POLICY_URL", "http://127.0.0.1:8500").split(",") if u.strip()]
POLICY_URL = POLICY_URLS[0]
PROXY_URL = os.environ.get("RL_PROXY_URL", "http://127.0.0.1:8510")


# ----------------------------------------------------------------------------- prompts / rollouts

def broadcast_payload(payload: list, cpu_group, device) -> None:
    """Rank 0's per-step batch (results + async_info, ~20 MB pickled) to every rank. Default: one gloo broadcast_object_list on
    the CPU group. RL_PAYLOAD_NCCL=1 (large multi-node jobs): only the pickled LENGTH goes over gloo (the other ranks keep
    waiting on the CPU group while rank 0 works alone, so the NCCL watchdog is not involved); the bytes then go as one uint8
    NCCL broadcast over IB. Gloo runs over TCP; on a congested cross-node link one flow took ~10 min for the 21 MB payload
    while 31 ranks waited in the next barrier."""
    if os.environ.get("RL_PAYLOAD_NCCL") != "1":
        dist.broadcast_object_list(payload, src=0, group=cpu_group)
        return
    import pickle
    if dist.get_rank() == 0:
        blob = pickle.dumps(payload[0], protocol=pickle.HIGHEST_PROTOCOL)
        size = torch.tensor([len(blob)], dtype=torch.int64)
    else:
        size = torch.zeros(1, dtype=torch.int64)
    dist.broadcast(size, src=0, group=cpu_group)
    n = int(size.item())
    buf = (torch.frombuffer(bytearray(blob), dtype=torch.uint8).to(device) if dist.get_rank() == 0
           else torch.empty(n, dtype=torch.uint8, device=device))
    dist.broadcast(buf, src=0)
    if dist.get_rank() != 0:
        payload[0] = pickle.loads(buf.cpu().numpy().tobytes())
    del buf

def community_prompts(suites: list[str], tasks: list[int], seeds: list[int]) -> list[tuple[str, int, int]]:
    return [(s, t, sd) for s in suites for t in tasks for sd in seeds]


PREFETCH_CRITIC_ROWS = {"on": False}   # streaming + physcritic: featurize episodes as they land (physics_critic.prefetch_rows)
FRESH_ONLY = {"on": False}     # --fresh-seeds-only: refuse to post any official-state (seed < 1000) episode


def _assert_fresh(payload: dict) -> None:
    from segment_curriculum import EVAL_SEEDS as _EVAL, is_fresh_seed
    seed = int((payload.get("entry") or payload).get("seed"))
    if not is_fresh_seed(seed) or seed in _EVAL:
        raise RuntimeError(f"fresh-seeds-only: refusing an official/evaluation initial state (seed {seed})")


def _post_rollout(payload: dict, attempts: int = 2) -> dict:
    if FRESH_ONLY["on"]:
        _assert_fresh(payload)      # outside the retry loop: a violation is a bug, never an infrastructure error
    last = None
    server = payload.get("_server") or ENV_SERVER
    route = payload.get("_route") or "/rollout"
    for i in range(attempts):
        try:
            r = requests.post(f"{server}{route}", json={k: v for k, v in payload.items() if not k.startswith("_")}, timeout=4000)
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(10 * (i + 1))
    return {"reward": 0.0, "verdict": "infrastructure_error", "components": {}, "diagnostics": {"error": repr(last), "infrastructure": True},
            "sample_path": None, "env_wall_s": None}


_SID_RNG = random.Random()  # OS-entropy seeded: sids must differ between relaunches of the same step, or the env server reuses a finished
                            # episode dir and the step comes back as 48 instant infrastructure errors (seen 11:13, 16:59, 17:11, 17:13)


def collect_rollouts(prompts: list[tuple[str, int, int]], step: int, num_generations: int) -> list[dict]:
    payloads = []
    for gi, (suite, task_id, seed) in enumerate(prompts):
        for k in range(num_generations):
            payloads.append({"suite": suite, "task_id": task_id, "seed": seed, "step": step, "sid": f"s{step}g{gi}k{k}_{_SID_RNG.randrange(16**6):06x}",
                             "group": gi, "_server": ENV_SERVERS[gi % len(ENV_SERVERS)]})   # whole groups per env server (2-node RL)
    with ThreadPoolExecutor(max_workers=len(payloads)) as pool:
        results = list(pool.map(_post_rollout, payloads))
    for p, r in zip(payloads, results):
        r["group"] = p["group"]
        r["suite"], r["task_id"], r["seed"], r["sid"] = p["suite"], p["task_id"], p["seed"], p["sid"]
    return results


def _rank_max(x: float, device) -> float:
    t = torch.tensor([x], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def _locked_state(curriculum):
    with CURRICULUM_LOCK:
        return curriculum.state_dict()


DEAD_ENV: dict[str, float] = {}      # streaming: env server -> time it was found unreachable (preemptible rollout nodes)


def _env_alive(url: str) -> bool:
    t = DEAD_ENV.get(url)
    if t is None:
        return True
    if time.time() - t < 600:
        return False
    try:                                  # re-probe every 10 min
        requests.get(f"{url}/health", timeout=10).raise_for_status()
        DEAD_ENV.pop(url, None)
        print(f"[stream] env server {url} is back", flush=True)
        return True
    except Exception:  # noqa: BLE001
        DEAD_ENV[url] = time.time()
        return False


ENV_INFLIGHT: dict[str, int] = {}    # streaming: episodes this trainer has in flight per env server
ENV_CAP: dict[str, int] = {}         # its episode slots (/health max_concurrency, read once)
ENV_PICK_LOCK = __import__("threading").Lock()


def _env_order(gi: int) -> list[str]:
    """Env servers by load: the least occupied (in flight / slots) first. A round-robin split (gi % 2) once sent half of
    the 60 in-flight episodes to a node with 18 slots while another ran 16 of its 42; the fill is now proportional."""
    if len(ENV_SERVERS) == 1:
        return list(ENV_SERVERS)
    for u in ENV_SERVERS:
        if u not in ENV_CAP:
            try:
                ENV_CAP[u] = int(requests.get(f"{u}/health", timeout=10).json().get("max_concurrency") or 1)
            except Exception:  # noqa: BLE001
                pass
    rr = [ENV_SERVERS[(gi + i) % len(ENV_SERVERS)] for i in range(len(ENV_SERVERS))]
    return sorted(rr, key=lambda u: ENV_INFLIGHT.get(u, 0) / max(1, ENV_CAP.get(u, 1)))


def run_segment_one(entry: dict, step: int, gi: int) -> dict:
    """One streaming episode (same payload and post-processing as collect_segment_rollouts, G=1). An env server that cannot
    be reached (e.g. a preempted node) is skipped for 10 minutes and the episode goes to another one."""
    payload = {"entry": {key: value for key, value in entry.items() if key != "curriculum"}, "horizon": entry["horizon"],
               "step": step, "group": gi, "sid": f"s{step}g{gi}k0_{_SID_RNG.randrange(16**6):06x}", "_route": "/segment"}
    r = None
    for _ in range(len(ENV_SERVERS)):
        with ENV_PICK_LOCK:
            order = _env_order(gi)
            alive = [u for u in order if _env_alive(u) and u not in payload.get("_tried", ())]
            if not alive:
                break
            url = alive[0]
            ENV_INFLIGHT[url] = ENV_INFLIGHT.get(url, 0) + 1
        payload["_server"] = url
        payload.setdefault("_tried", []).append(url)
        try:
            try:
                requests.get(f"{url}/health", timeout=10).raise_for_status()
            except Exception:  # noqa: BLE001
                DEAD_ENV[url] = time.time()
                print(f"[stream] env server {url} unreachable: skipped for 10 min", flush=True)
                continue
            r = _post_rollout({k: v for k, v in payload.items() if k != "_tried"})
            break
        finally:
            with ENV_PICK_LOCK:
                ENV_INFLIGHT[url] -= 1
    if r is None:
        r = {"reward": 0.0, "verdict": "infrastructure_error", "components": {}, "sample_path": None, "env_wall_s": None,
             "diagnostics": {"error": "no env server reachable", "infrastructure": True}}
    if PREFETCH_CRITIC_ROWS["on"] and r.get("episode_dir") and r.get("verdict") not in ("infrastructure_error", "replay_infidelity"):
        from physics_critic import prefetch_rows
        prefetch_rows(r["episode_dir"])          # off the learner's critical path (this is an episode worker thread)
    r["group"] = gi
    r["suite"], r["task_id"], r["seed"], r["sid"] = entry["suite"], entry["task_id"], entry["seed"], payload["sid"]
    r["start_id"], r["K"] = entry["start_id"], entry["K"]
    if r.get("verdict") == "replay_infidelity":
        r["sample_path"] = None
    return r


import threading as _threading
CURRICULUM_LOCK = _threading.Lock()   # streaming: the pool thread draws while the learner commits and checkpoints
GATE_URLS = [u.strip() for u in os.environ.get("RL_GATE_URLS", "").split(",") if u.strip()]   # streaming: the policy proxies


def _gate_post(u: str, route: str, params: dict, timeout: float):
    try:
        return requests.post(f"{u}{route}", params=params, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        return exc


def gate_close(log, urls: list[str] | None = None) -> dict:
    """Streaming: every proxy stops forwarding and drains its in-flight requests (episodes keep running, new requests wait).
    An unreachable proxy (preempted rollout node) is logged and skipped; a reachable proxy that fails to drain is fatal."""
    urls = GATE_URLS if urls is None else urls
    t0 = time.time()
    if not urls:
        return {}
    with ThreadPoolExecutor(max_workers=len(urls)) as pool:
        rs = list(pool.map(lambda u: _gate_post(u, "/gate/close", {"timeout": 900}, 960), urls))
    for u, r in zip(urls, rs):
        if isinstance(r, Exception):
            log(f"[stream] WARNING gate close: proxy {u} unreachable ({r!r}); skipped")
        elif r.status_code != 200:
            raise RuntimeError(f"gate close failed at {u}: {r.status_code} {r.text[:300]}")
    return {"gate/drain_s": round(time.time() - t0, 2)}


def gate_open(version: int | None, log, urls: list[str] | None = None) -> dict:
    """version None reopens without changing the tag (the servers behind kept their weights)."""
    out = {}
    for u in (GATE_URLS if urls is None else urls):
        r = _gate_post(u, "/gate/open", {} if version is None else {"version": int(version)}, 60)
        if isinstance(r, Exception) or r.status_code != 200:
            log(f"[stream] WARNING gate open: proxy {u} failed ({r!r})")
            continue
        out = r.json()
    return {"gate/held_s": out.get("held_s")}


# Hybrid refit (r5): policy servers on a preemptible node take the weights by file (export -> push -> reload_weights, all
# with timeouts, in a background thread on rank 0) instead of joining the NCCL group, where a vanished peer would hang the
# broadcast. RL_REFIT_FILE_URLS = those policy servers, RL_GATE_FILE_URLS = the proxy in front of them,
# RL_LIVE_HOSTS_FILE = their host(s) for push_live.sh. A failure marks the group dead (warning) and the run continues.
FILE_URLS = [u.strip() for u in os.environ.get("RL_REFIT_FILE_URLS", "").split(",") if u.strip()]
FILE_GATES = [u.strip() for u in os.environ.get("RL_GATE_FILE_URLS", "").split(",") if u.strip()]
FILE_GROUP = {"dead": False, "thread": None, "last": {}}


def file_group_refit(live_dir: Path, version: int, log) -> None:
    """Rank 0, background: push the live dir to the file group's host(s), then gate + reload those servers."""
    import shlex
    import subprocess
    t0 = time.time()
    try:
        env = dict(os.environ, RL_LIVE_HOSTS=os.environ.get("RL_LIVE_HOSTS_FILE", ""))
        subprocess.run([*shlex.split(LIVE_PUSH), str(live_dir)], check=True, env=env, timeout=1200)
        t_push = time.time() - t0
        gate_close(log, FILE_GATES)
        with ThreadPoolExecutor(max_workers=len(FILE_URLS)) as pool:
            list(pool.map(lambda u: _sync_one(u, live_dir, log), FILE_URLS))
        gate_open(version, log, FILE_GATES)
        FILE_GROUP["last"] = {"filegroup/push_s": round(t_push, 1), "filegroup/total_s": round(time.time() - t0, 1),
                              "filegroup/version": version}
        log(f"[sync] file group ({len(FILE_URLS)} vLLM) at version {version} in {time.time() - t0:.0f}s (push {t_push:.0f}s)")
    except Exception as exc:  # noqa: BLE001
        FILE_GROUP["dead"] = True
        gate_open(None, log, FILE_GATES)        # if reachable at all: release its sessions, keep the old version tag
        log(f"[sync] WARNING file group refit failed ({exc!r}); its servers are dropped from refits (stale weights "
            "would be caught by the staleness bound only if their proxy still tags; check the node)")


def focus_list(probe: dict, name: str | None) -> list[str]:
    """The focus starts: a probe set (probed and practised) or a practice list of the probe file (practised only, weighted by repeats)."""
    if not name:
        return []
    if name in probe["sets"]:
        return list(probe["sets"][name])
    if name in (probe.get("practice") or {}):
        return list(probe["practice"][name])
    raise ValueError(f"focus set {name!r} is neither a probe set nor a practice list")


def collect_segment_rollouts(entries: list[dict], step: int, num_generations: int) -> list[dict]:
    """Per-call RL: every group replays one segment start, then samples the next H requests G times."""
    payloads = []
    for gi, entry in enumerate(entries):
        for k in range(num_generations):
            payloads.append({"entry": {key: value for key, value in entry.items() if key != "curriculum"}, "horizon": entry["horizon"],
                             "step": step, "group": gi, "sid": f"s{step}g{gi}k{k}_{_SID_RNG.randrange(16**6):06x}",
                             "_server": ENV_SERVERS[gi % len(ENV_SERVERS)], "_route": "/segment"})
    with ThreadPoolExecutor(max_workers=len(payloads)) as pool:
        results = list(pool.map(_post_rollout, payloads))
    for p, r in zip(payloads, results):
        r["group"] = p["group"]
        r["suite"], r["task_id"], r["seed"], r["sid"] = p["entry"]["suite"], p["entry"]["task_id"], p["entry"]["seed"], p["sid"]
        r["start_id"], r["K"] = p["entry"]["start_id"], p["entry"]["K"]
        if r.get("verdict") == "replay_infidelity":
            r["sample_path"] = None   # a diverged replay is not a sample of this start
    return results


def filler_sample(s: dict, image_pad_id: int, n: int = 32) -> dict | None:
    """A cheap zero-weight FSDP pass: the first n text tokens of a real sample (every FSDP unit runs its collectives, the loss
    is multiplied by 0). None when those tokens contain an image placeholder (the caller then repeats a whole sample)."""
    ids = s["input_ids"][:n]
    if bool((ids == image_pad_id).any()) or len(ids) < 2:
        return None
    mask = torch.zeros(len(ids), dtype=torch.bool)
    mask[-1] = True
    return {"input_ids": ids, "mask": mask, "length": len(ids), "n_tokens": 1, "weight": 1.0,
            "advantage": 0.0, "index": s.get("index"), "trajectory_tokens": s.get("trajectory_tokens", 1)}


def balanced_assignment(indices: list[int], costs: list[int], world: int) -> list[list[int]]:
    """Longest-processing-time assignment of trajectories to ranks by their trained-request count: FSDP runs the same number of
    forward/backward passes on every rank (fillers pad the rest), so the busiest rank sets the step time. Deterministic; the
    objective is unchanged (every trajectory is trained once, on exactly one rank)."""
    loads = [[0, 0, r] for r in range(world)]          # (requests, trajectories, rank)
    out: list[list[int]] = [[] for _ in range(world)]
    for i, c in sorted(zip(indices, costs), key=lambda ic: (-ic[1], ic[0])):
        slot = min(loads)
        out[slot[2]].append(i)
        slot[0] += c
        slot[1] += 1
    return [sorted(x) for x in out]


def group_advantages(results: list[dict], eps: float = 1e-4, std_floor: float = 0.0) -> list[float]:
    """GRPO group-normalised advantages. ``std_floor`` (stabiliser): once the reward saturates
    (group std 0.05-0.15 at reward ~+0.36) dividing by the tiny std turns sampling noise into +-1..2 advantages; with no
    trust region in this trainer the policy drifted its tool-call syntax within ~8 steps, twice. A floor on the divisor
    keeps the sign of the advantage but scales near-uniform groups down (Dr.GRPO-style)."""
    by = defaultdict(list)
    for i, r in enumerate(results):
        if r.get("verdict") != "infrastructure_error" and r.get("sample_path"):
            by[r["group"]].append(i)
    adv = [0.0] * len(results)
    for idxs in by.values():
        if len(idxs) < 2:
            continue
        rewards = [float(results[i]["reward"]) for i in idxs]
        mean = sum(rewards) / len(rewards)
        std = math.sqrt(sum((x - mean) ** 2 for x in rewards) / len(rewards))
        if std < 1e-6:
            continue
        for i, x in zip(idxs, rewards):
            adv[i] = (x - mean) / (max(std, std_floor) + eps)
    return adv


def shuffle_within_groups(results: list[dict], adv: list[float], *, seed: int) -> list[float]:
    """Diagnostic null control: permute the advantages among the members of each group. Every group keeps the same
    multiset of advantages (same scale, zero mean) but they no longer follow the reward, so the expected policy
    gradient of the task term is zero and only the update noise and the fixed pipeline remain."""
    import random as _random
    rng = _random.Random(seed)
    by = defaultdict(list)
    for i, r in enumerate(results):
        by[r["group"]].append(i)
    out = list(adv)
    for idxs in by.values():
        vals = [adv[i] for i in idxs]
        rng.shuffle(vals)
        for i, v in zip(idxs, vals):
            out[i] = v
    return out


# ----------------------------------------------------------------------------- samples
# assistant_mask / assistant_blocks live in rl/chat_tokens.py (shared with the rollout service's history alignment)


def _decode_image(url: str) -> Image.Image:
    if not url.startswith("data:"):
        raise ValueError("only data URLs are recorded")
    b64 = url.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def images_of_request(body: dict) -> list[Image.Image]:
    images = []
    for m in body.get("messages") or []:
        content = m.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = part.get("image_url")
                    url = url.get("url") if isinstance(url, dict) else url
                    if isinstance(url, str):
                        images.append(_decode_image(url))
    return images


def build_sample(sample_path: str, processor, *, max_seq_tokens: int, image_pad_id: int, hard_max_tokens: int | None = None) -> dict | None:
    """Single-segment sample (rollout.json: last_request + last_response)."""
    s = json.loads(Path(sample_path).read_text(encoding="utf-8"))
    if s.get("segments"):
        out = build_segment_samples(s, processor, max_seq_tokens=max_seq_tokens, image_pad_id=image_pad_id, hard_max_tokens=hard_max_tokens, path=sample_path)
        return out[-1] if out else None
    last = s["last_response"]
    return _build_one(list(last["prompt_token_ids"]), list(last["token_ids"]), last.get("finish_reason"), images_of_request(s["last_request"]), processor,
                      max_seq_tokens=max_seq_tokens, image_pad_id=image_pad_id, hard_max_tokens=hard_max_tokens, path=sample_path)


def _segment_images(sg: dict, path: str) -> list | dict:
    images = []
    for url in sg.get("images") or []:
        if url is None:
            return {"error": f"segment {sg.get('request')}: image missing from the proxy store", "path": path}
        images.append(_decode_image(url))
    return images


def _same_text(tokenizer, a: list[int], b: list[int]) -> bool:
    """Historical helper: only the same sampled token action is identical."""
    return list(a) == list(b)


def build_segment_samples(s: dict, processor, *, max_seq_tokens: int, image_pad_id: int, hard_max_tokens: int | None, path: str,
                          train_repairs: bool = True, stats: dict | None = None) -> list[dict]:
    """One sample per compaction segment (chronological) plus one 'repair' sample per stored non-exact request.

    Identity-based masking: each assistant block of a segment is mapped to the proxy request that produced it
    (segment `history`, recorded by rl/env_server.py). A history block is trained iff (a) it is exactly the completion the policy
    sampled for that request (or decodes to the same text) and (b) that request has not been trained in an earlier segment. Blocks that
    are the driver's re-rendering of a parsed/unparseable output are masked; the sampled tokens of such a request are trained instead
    from the request's own prompt (`repairs`: prompt_token_ids + token_ids, final block only, loss weight = candidates/stored so a cap
    on stored repairs stays unbiased). The old content-based dedup only applies to rollout.json files without `history`."""
    tokenizer = getattr(processor, "tokenizer", None)
    seen_blocks: set[tuple[int, ...]] = set()
    seen_requests: set[int] = set()
    repairs = {int(rp["request"]): rp for rp in (s.get("repairs") or []) if rp.get("request") is not None} if train_repairs else {}
    resolved_by_text: set[int] = set()   # non-exact history blocks that turned out to be the same text: trained in place, repair dropped
    st = stats if stats is not None else {}
    out: list[dict] = []
    for sg in s["segments"]:
        images = _segment_images(sg, path)
        if isinstance(images, dict):
            out.append(images)
            continue
        history = sg.get("history")
        end_request = sg.get("request")
        keep = None
        if history is not None and end_request is not None:
            def keep(blocks: list[tuple[int, int]], ids: list[int], history=history, end_request=int(end_request)) -> list[bool] | None:
                if len(blocks) != len(history) + 1:
                    st["history_mismatch"] = st.get("history_mismatch", 0) + 1
                    return [False] * max(0, len(blocks) - 1) + [True]
                flags = []
                for (a, b), h in zip(blocks[:-1], history):
                    r = h.get("request")
                    exact = bool(h.get("exact"))
                    if r is None:
                        flags.append(False); st["history_unmapped"] = st.get("history_unmapped", 0) + 1; continue
                    if not exact and r in repairs and _same_text(tokenizer, ids[a:b], with_im_end(list(repairs[r]["token_ids"]))):
                        exact = True
                        resolved_by_text.add(r)
                    if not exact:
                        flags.append(False); st["history_rerendered"] = st.get("history_rerendered", 0) + 1; continue
                    if r in seen_requests:
                        flags.append(False); st["dedup_identity"] = st.get("dedup_identity", 0) + 1; continue
                    seen_requests.add(r)
                    flags.append(True)
                if int(end_request) in seen_requests:
                    flags.append(False)
                else:
                    seen_requests.add(int(end_request)); flags.append(True)
                return flags
        r = _build_one(list(sg["prompt_token_ids"]), list(sg["token_ids"]), sg.get("finish_reason"), images, processor, max_seq_tokens=max_seq_tokens,
                       image_pad_id=image_pad_id, hard_max_tokens=hard_max_tokens, path=path, seen_blocks=None if keep is not None else seen_blocks,
                       keep_blocks=keep)
        if r is None:
            continue
        r["segment"] = end_request
        out.append(r)
    for req, rp in sorted(repairs.items()):
        if req in resolved_by_text or req in seen_requests:
            continue
        images = _segment_images(rp, path)
        if isinstance(images, dict):
            out.append(images)
            continue
        r = _build_one(list(rp["prompt_token_ids"]), list(rp["token_ids"]), rp.get("finish_reason"), images, processor, max_seq_tokens=max_seq_tokens,
                       image_pad_id=image_pad_id, hard_max_tokens=hard_max_tokens, path=path, keep_blocks=lambda blocks, ids: [False] * (len(blocks) - 1) + [True])
        if r is None:
            continue
        r["segment"] = f"repair:{req}"
        r["weight"] = float(rp.get("weight") or 1.0)
        seen_requests.add(req)
        st["repairs"] = st.get("repairs", 0) + 1
        out.append(r)
    return out


def _build_one(prompt_ids: list[int], completion_ids: list[int], finish_reason, images: list, processor, *, max_seq_tokens: int, image_pad_id: int,
               hard_max_tokens: int | None, path: str, seen_blocks: set | None = None, keep_blocks=None) -> dict | None:
    ids = prompt_ids + completion_ids
    if hard_max_tokens and len(ids) > hard_max_tokens:
        return {"error": f"sequence {len(ids)} tokens > hard cap {hard_max_tokens} (skipped to avoid OOM)", "path": path}
    # Do not invent a stop token that the policy did not sample.
    mask = assistant_mask(ids)
    if keep_blocks is not None:
        blocks = assistant_blocks(ids, mask)
        flags = keep_blocks(blocks, ids)
        if flags is not None:
            for (a, b), f in zip(blocks, flags):
                if not f:
                    for k in range(a, b):
                        mask[k] = 0
    elif seen_blocks is not None:
        # legacy (rollout.json without per-request history): a block kept verbatim across a compaction was already trained in its
        # original context: no second gradient. Same-segment repeats of one text are different sampled decisions and keep theirs.
        blocks = assistant_blocks(ids, mask)
        for a, b in blocks:
            if tuple(ids[a:b]) in seen_blocks:
                for k in range(a, b):
                    mask[k] = 0
        for a, b in blocks:
            seen_blocks.add(tuple(ids[a:b]))
    if sum(mask) == 0:
        return None
    n_pad = sum(1 for t in ids if t == image_pad_id)
    mm = {}
    if images:
        enc = processor.image_processor(images=images, return_tensors="pt")
        grid = enc["image_grid_thw"]
        merge = getattr(processor.image_processor, "merge_size", 2)
        expected = int((grid.prod(-1) // (merge * merge)).sum())
        if expected != n_pad:
            return {"error": f"image pad mismatch: ids {n_pad} vs processor {expected} for {len(images)} images", "path": path}
        mm = {"pixel_values": enc["pixel_values"], "image_grid_thw": grid}
    elif n_pad:
        return {"error": f"{n_pad} image pads but no images in the request", "path": path}
    if len(ids) > max_seq_tokens:
        # keep the whole sequence (the model needs the images/prefix) but drop gradient from the earliest assistant blocks
        over = len(ids) - max_seq_tokens
        cut = 0
        for i, m in enumerate(mask):
            if cut >= over:
                break
            if m:
                mask[i] = 0
                cut += 1
        if sum(mask) == 0:
            # every assistant token fell inside the truncated region -> a constant loss with no grad_fn
            return {"error": f"sequence {len(ids)} tokens > max-seq-tokens {max_seq_tokens}: no trainable tokens left after truncation (skipped)", "path": path}
    return {"input_ids": torch.tensor(ids, dtype=torch.long), "mask": torch.tensor(mask, dtype=torch.bool), "n_tokens": int(sum(mask)),
            "length": len(ids), **mm}


# ----------------------------------------------------------------------------- model
def lora_targets(model) -> list[str]:
    names = set()
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if "visual" in name or "vision" in name:
            continue
        if "lm_head" in name:
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf.endswith("_proj") or leaf.startswith("in_proj") or leaf in ("out_proj", "gate_proj", "up_proj", "down_proj"):
            names.add(leaf if "language_model" not in name else name)
    return sorted(names)


def masked_logprobs(model_core, lm_head, inputs: dict, mask: torch.Tensor, chunk: int = LM_HEAD_CHUNK, per_token: bool = False) -> torch.Tensor:
    """Sum of log p(token) over masked positions (or the per-token vector when ``per_token``), lm_head applied in chunks
    (no full-vocab logits tensor)."""
    out = model_core(**inputs)
    hidden = out.last_hidden_state[0]  # [T, H]
    ids = inputs["input_ids"][0]
    # position t predicts token t+1: logits from hidden[t], label ids[t+1], mask[t+1]
    positions = torch.nonzero(mask[0, 1:], as_tuple=False).squeeze(-1)  # indices into ids[1:] -> hidden index = same
    total = hidden.new_zeros((), dtype=torch.float32)
    pieces = []
    for start in range(0, positions.numel(), chunk):
        pos = positions[start:start + chunk]
        h = hidden.index_select(0, pos)
        logits = lm_head(h).float()
        labels = ids.index_select(0, pos + 1)
        lp = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        if per_token:
            pieces.append(lp)
        else:
            total = total + lp.sum()
    if per_token:
        return torch.cat(pieces) if pieces else hidden.new_zeros((0,), dtype=torch.float32)
    return total


def masked_logprobs_centered(model_core, lm_head, inputs: dict, mask: torch.Tensor, topk_ids: torch.Tensor, topk_logprobs: torch.Tensor,
                             chunk: int = LM_HEAD_CHUNK, is_cap: float | None = None,
                             band: tuple[float, float] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token log p(token) over masked positions and the score-centering correction (Marek & Ryabinin 2026, "Score
    Centering Stabilizes Off-policy RL", eq. 6): sum_{v in top-k} sg(q_v - rho p_v) log p_v, q = the sampler's top-k
    probabilities, the tail modelled by the trainer rescaled to the sampler's tail mass (rho). Subtracting it from log p
    makes the expected score under the sampler zero at every prefix, removing the drift a trainer/sampler mismatch adds.
    With truncated importance weights w(r) = min(r, is_cap) the correction is the WEIGHTED one of the authors' composition
    (martin-marek/score-centering @7c56e9e, README): sum_head sg(q_v w(p_v/q_v) - alpha p_v) log p_v with alpha = rho w(1/rho)
    = min(1, cap rho); the caller then uses w(p_y/q_y) log p_y - correction. (w = 1 recovers the plain correction above.)
    ``band`` = (eps_low, eps_high): the same composition for SAO's masked weight w(x) = x if 1 - eps_low < x < 1 + eps_high
    else 0, for which alpha = rho w(1/rho) = [1/rho inside the band]."""
    out = model_core(**inputs)
    hidden = out.last_hidden_state[0]
    ids = inputs["input_ids"][0]
    positions = torch.nonzero(mask[0, 1:], as_tuple=False).squeeze(-1)
    if positions.numel() != topk_ids.shape[0]:
        raise InvalidRollout(f"{positions.numel()} trained tokens but {topk_ids.shape[0]} sampler top-k rows")
    lps, corrections = [], []
    for start in range(0, positions.numel(), chunk):
        pos = positions[start:start + chunk]
        logp_all = torch.log_softmax(lm_head(hidden.index_select(0, pos)).float(), dim=-1)
        lps.append(logp_all.gather(-1, ids.index_select(0, pos + 1).unsqueeze(-1)).squeeze(-1))
        head = logp_all.gather(-1, topk_ids[start:start + chunk])
        q = torch.exp(topk_logprobs[start:start + chunk])
        with torch.no_grad():
            p_head = torch.exp(head)
            rho = (1.0 - q.sum(-1)).clamp_min(0.0) / (1.0 - p_head.sum(-1)).clamp_min(1e-6)
            if band is not None:
                lo, hi = 1.0 - band[0], 1.0 + band[1]
                x = p_head / q.clamp_min(1e-30)
                w_head = x * ((x > lo) & (x < hi)).float()                   # w(p_v / q_v)
                inv = 1.0 / rho.clamp_min(1e-30)
                alpha = ((inv > lo) & (inv < hi)).float()                     # rho * w(1/rho)
                residual = q * w_head - alpha.unsqueeze(-1) * p_head
            elif is_cap is None:
                residual = q - rho.unsqueeze(-1) * p_head
            else:
                alpha = torch.clamp(is_cap * rho, max=1.0)                      # rho * min(1/rho, cap)
                w_head = torch.clamp(p_head / q.clamp_min(1e-30), max=is_cap)   # w(p_v / q_v)
                residual = q * w_head - alpha.unsqueeze(-1) * p_head
        corrections.append((residual * head).sum(-1))
        del logp_all
    empty = hidden.new_zeros((0,), dtype=torch.float32)
    return (torch.cat(lps) if lps else empty), (torch.cat(corrections) if corrections else empty)


def sample_loss(model, core, lm_head, inputs: dict, mask: torch.Tensor, s: dict, *, score_centering: bool, kl_beta: float,
                full_completion_tokens: int, world_size: int, reference=None, is_cap: float | None = None,
                is_stats: list | None = None, code_weight: float = 0.0, code_norm: float = 1.0,
                code_stats: list | None = None, ratio_band: tuple[float, float] | None = None,
                mask_stats: list | None = None, kl_stats: list | None = None,
                ref_weight: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """One trained request: (loss to backpropagate, weighted k3 KL (detached), score-centering stats row or None).
    The KL-to-SFT anchor (earlier runs collapsed without it) uses the same weights with the LoRA adapters
    disabled as reference (``reference(inputs, mask)`` -> per-token log-probs overrides it, e.g. a frozen full copy). With score_centering and the sampler's top-k in the sample, the advantage term uses the
    centered log-probability; a sample without top-k is trained plainly (the caller counts it).
    Simulator critic (rl/physics_critic.py): a sample with ``token_advantage`` (per-request critic advantages, advantage 1.0) scales
    each target token by its request's advantage; ``ratio_band`` = (eps_low, eps_high) is SAO's double-sided masking
    (arXiv 2607.07508): the policy-gradient score of a token is weighted by r = exp(log p_train - log p_rollout) when
    1 - eps_low < r < 1 + eps_high and dropped otherwise (code term included); the KL term keeps every token unless
    RL_KL_ON_KEPT=1 (avoids KL spikes from masked tokens): then the KL is computed on the SAME kept tokens only, each
    sample rescaled by sum(w_all)/sum(w_kept) (per-token average over the kept tokens; exactly the old value when every
    token is kept), and ``kl_stats`` gets [weight * KL before the mask, weight * KL after] for the metrics."""
    device = mask.device
    centered = score_centering and "topk_ids" in s
    keep = None
    ref_lp = None
    # speed-up (--ref-every N): ref_weight 0 = this pass skips the reference forward (no KL term or
    # KL metric row); N = the 1-in-N pass that runs it (KL term and metrics x N: unbiased with inclusion probability 1/N);
    # 1 = the old path, no extra multiplication anywhere
    if kl_beta > 0 and ref_weight != 0:
        with torch.no_grad(), _PTime("t_ref_fwd"):
            if reference is not None:
                ref_lp = reference(inputs, mask)
            else:
                with model.disable_adapter():
                    ref_lp = masked_logprobs(core, lm_head, inputs, mask, per_token=True)
    sc_row = None
    import contextlib
    saved = offload_ctx()
    with saved, _PTime("t_policy_fwd"):
        if centered:
            lp_tok, correction = masked_logprobs_centered(core, lm_head, inputs, mask, s["topk_ids"].to(device), s["topk_logprobs"].to(device),
                                                          is_cap=is_cap, band=ratio_band)
        else:
            lp_tok, correction = masked_logprobs(core, lm_head, inputs, mask, per_token=True), None
    score = lp_tok
    if is_cap is not None:
        # bounded off-policy surrogate (async lookahead arm): per-token truncated importance weight pi_theta / pi_behavior,
        # detached, on log p; with score centering the weighted correction is subtracted after (never w * (log p - C))
        if "sampled_logprobs" not in s:
            raise InvalidRollout("truncated importance weighting needs the sampler's per-token log-probabilities")
        log_ratio = lp_tok.detach() - s["sampled_logprobs"].to(device)
        rho = torch.exp(log_ratio.clamp(max=math.log(is_cap)))
        if is_stats is not None:
            is_stats.append(torch.tensor([float(rho.sum()), float((log_ratio > math.log(is_cap)).sum()), float((rho * rho).sum()),
                                          float(rho.numel())], device=device, dtype=torch.float64))
        score = rho * score
    if ratio_band is not None:
        # SAO double-sided masking (arXiv 2607.07508 eq. 1-3): f(r) = r inside (1 - eps_low, 1 + eps_high), 0 outside, with
        # r = pi_theta / pi_rollout per token (detached); with score centering the correction is the band-weighted one
        if "sampled_logprobs" not in s:
            raise InvalidRollout("ratio masking needs the sampler's per-token log-probabilities")
        ratio = torch.exp(lp_tok.detach() - s["sampled_logprobs"].to(device))
        low, high = ratio <= 1.0 - ratio_band[0], ratio >= 1.0 + ratio_band[1]
        keep = (~(low | high)).float()
        score = ratio * keep * score
        if mask_stats is not None:
            mask_stats.append(torch.tensor([float((~(low | high)).sum()), float(low.sum()), float(high.sum()), float(ratio.numel())],
                                           device=device, dtype=torch.float64))
    if centered:
        score = score - correction          # with IS / masking the correction is already the weighted one (masked_logprobs_centered)
    # a prefix-packed sample (several original requests of one trajectory in one causal forward) carries each target token's
    # own request inclusion weight; the sum over tokens then equals the sum of the separate per-request objectives
    token_w = s["token_weights"].to(device) if "token_weights" in s else None
    weight = 1.0 if token_w is not None else s["weight"]
    # per-token task credit (0 on a successful trajectory's already-solved suffix): scales the policy-gradient term only
    credit = s["token_credit"].to(device) if "token_credit" in s else None
    pure = s["token_pure_error"].to(device) if (code_weight > 0 and "token_pure_error" in s) else None
    token_adv = s["token_advantage"].to(device) if "token_advantage" in s else None
    if pure is not None and token_adv is not None:
        # per-request advantages: the pure-error rule applies request by request (no positive credit for a pure error)
        withheld = pure * (token_adv > 0).float()
        credit = (1.0 - withheld) if credit is None else credit * (1.0 - withheld)
    elif pure is not None and s["advantage"] > 0:
        # python_exec_cost_v1: a request whose code failed with no host effect gets no POSITIVE task credit (a negative
        # trajectory advantage still applies to it)
        credit = (1.0 - pure) if credit is None else credit * (1.0 - pure)
    pg_w = token_w if credit is None else (credit if token_w is None else token_w * credit)
    if token_adv is not None:
        pg_w = token_adv if pg_w is None else pg_w * token_adv
    objective_logp = (score * pg_w).sum() if pg_w is not None else score.sum()
    if centered:
        sc_row = torch.tensor([float((lp_tok.detach() - s["sampled_logprobs"].to(device)).abs().sum()),
                               float(correction.detach().abs().sum()), float(lp_tok.numel())], device=device, dtype=torch.float64)
    kl = kl_k3(ref_lp, lp_tok, token_w) if ref_lp is not None else lp_tok.new_zeros(())
    if ref_lp is not None and keep is not None and KL_ON_KEPT:
        kl_all = kl.detach()
        w_all = token_w if token_w is not None else torch.ones_like(keep)
        w_kept = w_all * keep
        kept_mass = w_kept.sum()
        scale_kl = (w_all.sum() / kept_mass) if float(kept_mass) > 0 else kept_mass.new_zeros(())
        kl = kl_k3(ref_lp, lp_tok, w_kept) * scale_kl.detach()
        if ref_weight != 1.0:
            # speed-up 2: 1-in-N subsampled reference forward, Horvitz-Thompson weight N on the KL term and its metrics
            kl, kl_all = kl * ref_weight, kl_all * ref_weight
        if kl_stats is not None:
            kl_stats.append(torch.tensor([float(weight * kl_all), float(weight * kl.detach())], device=device, dtype=torch.float64))
    elif ref_lp is not None:
        if ref_weight != 1.0:
            kl = kl * ref_weight
        if kl_stats is not None:
            kl_stats.append(torch.tensor([float(weight * kl.detach())] * 2, device=device, dtype=torch.float64))
    loss = sampled_objective(objective_logp, kl, advantage=s["advantage"], weight=weight, kl_beta=kl_beta,
                             full_completion_tokens=full_completion_tokens, world_size=world_size)
    if code_weight > 0 and "token_code" in s:
        # local code term (not group-centered): + lambda * sum_r (1/p_r) * c_r * mean_t score / (H * N trajectories);
        # minimising it lowers the probability of the erroring request's tokens (token_code = c_r / m_r)
        code_w = s["token_code"].to(device) * (token_w if token_w is not None else weight)
        code_term = (score * code_w).sum()
        loss = loss + code_weight * code_term * world_size / code_norm
        if code_stats is not None:
            code_stats.append(torch.tensor([float(code_w.sum()), float((s["token_code"] > 0).any()),
                                            float(0 if pure is None or s["advantage"] <= 0 else (pure > 0).any())],
                                           device=device, dtype=torch.float64))
    return loss, (weight * kl.detach()).to(torch.float32), sc_row


def export_cocritic(cocritic, path, step: int) -> None:
    """Speed-up 3 (--cocritic-live): the co-critic's current weights for the env servers' early-stop monitor, written atomically
    (rank 0, after every critic update and once at start). Never read by training."""
    if not path or cocritic is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save({"state_dict": {k: v.detach().float().cpu() for k, v in cocritic.state_dict().items()}, "step": int(step),
                "time": time.time()}, tmp)
    os.replace(tmp, path)


def monitor_starts(spec: dict, suites: list, tasks: list) -> list[dict]:
    """held-out monitor starts: every (suite, task) x spec["ev12_seeds"] on the held-out wide-layout ("ev12") harness copy (harness_alt
    "ev12") and x spec["official_seeds"] on the run's harness (official LIBERO init states); K = 0 fresh episodes, never trained"""
    out = []
    for suite in suites:
        for task in tasks:
            for alt, seeds in (("ev12", spec.get("ev12_seeds") or []), (None, spec.get("official_seeds") or [])):
                for seed in seeds:
                    sid = f"monitor-{suite}-{task}-{seed}-{alt or 'official'}"
                    out.append({"start_id": sid, "trajectory_id": sid, "suite": suite, "task_id": int(task), "seed": int(seed), "K": 0,
                                "remaining": None, "source_dir": None, "source_responses": None, "expected": None, "teacher_delta": None,
                                **({"harness_alt": alt} if alt else {})})
    return out


def monitor_record(entry: dict, res: dict, step: int) -> dict:
    """one held-out monitor episode -> a monitor.jsonl row (episode server result.json for the physical action count)"""
    diag = res.get("diagnostics") or {}
    acts = None
    try:
        acts = json.loads((Path(res["episode_dir"]) / "result.json").read_text()).get("action_requests")
    except (KeyError, OSError, ValueError, TypeError):
        pass
    return {"step": step, "set": "ev12" if entry.get("harness_alt") else "official", "suite": entry["suite"], "task_id": entry["task_id"],
            "seed": entry["seed"], "success": bool(diag.get("final_success", diag.get("host_final_success"))), "verdict": res.get("verdict"),
            "termination": diag.get("termination_reason"), "requests": res.get("eligible_request_count"), "physical_actions": acts,
            "early_stop_cut": bool((diag.get("early_stop") or {}).get("cut")), "infrastructure": bool(diag.get("infrastructure")),
            "env_wall_s": res.get("env_wall_s"), "episode_dir": res.get("episode_dir")}


def monitor_metrics(rows: list) -> dict:
    if not rows:
        return {}
    out = {"monitor/n": len(rows)}
    for name, sel in (("", rows), ("ev12_", [r for r in rows if r["set"] == "ev12"]), ("official_", [r for r in rows if r["set"] == "official"])):
        ok = [r for r in sel if not r["infrastructure"]]
        if ok:
            out[f"monitor/{name}success"] = sum(r["success"] for r in ok) / len(ok)
            out[f"monitor/{name}n"] = len(ok)
    ok = [r for r in rows if not r["infrastructure"]]
    acts = [r["physical_actions"] for r in ok if isinstance(r["physical_actions"], (int, float))]
    reqs = [r["requests"] for r in ok if isinstance(r["requests"], (int, float))]
    out.update({"monitor/cut": sum(r["early_stop_cut"] for r in ok), "monitor/infra": len(rows) - len(ok),
                "monitor/actions_per_ep": sum(acts) / len(acts) if acts else None, "monitor/requests_per_ep": sum(reqs) / len(reqs) if reqs else None})
    return out


def early_stop_metrics(results: list) -> dict:
    """Speed-up 3: per-round counts of the env servers' critic early stop (diagnostics.early_stop of each episode); {} when off."""
    rows = [(r.get("diagnostics") or {}).get("early_stop") for r in results if isinstance(r, dict)]
    rows_r = [(e, r) for e, r in zip(rows, results) if isinstance(e, dict)]
    if not rows_r:
        return {}
    ex = [(e, r) for e, r in rows_r if e.get("exempt")]
    if ex:              # tasks exempt from the CRITIC rule (env server exempt_tasks) are reported separately; the rest as before
        rows_r = [(e, r) for e, r in rows_r if not e.get("exempt")]
        okx = lambda r: bool((r.get("diagnostics") or {}).get("final_success", (r.get("diagnostics") or {}).get("host_final_success")))  # noqa: E731
        exm = {"es/exempt": len(ex), "es/exempt_would_cut": sum(1 for e, _ in ex if e.get("would_cut")),
               "es/exempt_would_cut_success": sum(1 for e, r in ex if e.get("would_cut") and okx(r)),
               "es/exempt_success": sum(1 for _, r in ex if okx(r)),
               "es/exempt_success_cut": sum(1 for e, _ in ex if e.get("cut") and (e.get("decision") or {}).get("kind") == "success"),
               "es/exempt_critic_cut": sum(1 for e, _ in ex if e.get("cut") and (e.get("decision") or {}).get("kind", "critic") == "critic")}
        return {**(early_stop_metrics([r for _, r in rows_r]) if rows_r else {}), **exm}
    ok = lambda r: bool((r.get("diagnostics") or {}).get("final_success", (r.get("diagnostics") or {}).get("host_final_success")))  # noqa: E731
    ctrl = [(e, r) for e, r in rows_r if e.get("control")]
    # es/cut and es/cut_frac count the co-critic rule's cuts only; the missed-finish rule's cuts (decision kind "success") are separate
    crit_cut = lambda e: e.get("cut") and (e.get("decision") or {}).get("kind", "critic") == "critic"  # noqa: E731
    succ_cut = lambda e: e.get("cut") and (e.get("decision") or {}).get("kind") == "success"          # noqa: E731
    return {"es/monitored": len(rows_r), "es/cut": sum(1 for e, _ in rows_r if crit_cut(e)),
            "es/cut_frac": sum(1 for e, _ in rows_r if crit_cut(e)) / len(rows_r),
            **({"es/success_cut": sum(1 for e, _ in rows_r if succ_cut(e)),
                "es/success_cut_still_success": sum(1 for e, r in rows_r if succ_cut(e) and ok(r))}
               if any((e.get("decision") or {}).get("kind") == "success" for e, _ in rows_r) else {}),
            "es/control": len(ctrl), "es/control_would_cut": sum(1 for e, _ in ctrl if e.get("would_cut")),
            "es/control_would_cut_success": sum(1 for e, r in ctrl if e.get("would_cut") and ok(r)),
            "es/control_success": sum(1 for _, r in ctrl if ok(r)),
            "es/monitor_errors": sum(int(e.get("errors") or 0) for e, _ in rows_r)}


GROUP_SEEN: dict = {}       # group sampling smoke check: group keys already trained in this process
KL_ON_KEPT = os.environ.get("RL_KL_ON_KEPT") == "1"     # KL only on the SAO ratio-mask kept tokens (default off)


def kl_k3(ref_logp: torch.Tensor, logp: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """Sum k3 estimates of KL(pi || pi_ref) over sampled tokens.

    Direct fixed-sample backpropagation is a GRPO regularization surrogate, not
    an unbiased gradient of that full policy-dependent expectation.
    """
    r = ref_logp - logp
    k3 = torch.exp(r) - r - 1.0
    return (k3 * weights).sum() if weights is not None else k3.sum()


def merged_state_dict(model) -> dict:
    """bf16 state dict of base + scaling * B @ A (computed in fp32) with plain HF key names; the model is not modified,
    so repeated syncs do not accumulate bf16 merge/unmerge rounding in the base weights."""
    from peft.tuners.lora import LoraLayer
    base = model.get_base_model()
    sd: dict = {}
    for name, module in base.named_modules():
        if isinstance(module, LoraLayer):
            w = module.base_layer.weight.detach().float()
            for adapter in module.active_adapters:
                a = module.lora_A[adapter].weight.detach().float()
                b = module.lora_B[adapter].weight.detach().float()
                w = w + module.scaling[adapter] * (b @ a)
            sd[name + ".weight"] = w.to(torch.bfloat16).cpu()
            if getattr(module.base_layer, "bias", None) is not None:
                sd[name + ".bias"] = module.base_layer.bias.detach().cpu()
    for name, p in base.named_parameters():
        if "lora_" in name:
            continue
        key = name.replace(".base_layer.weight", ".weight").replace(".base_layer.bias", ".bias")
        if key not in sd:
            sd[key] = p.detach().cpu()
    return sd


def save_merged(model, processor, out_dir: Path, is_main: bool) -> None:
    """Write the merged bf16 model (HF layout) from rank 0 without touching the trainable model."""
    if not is_main:
        return
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    sd = merged_state_dict(model)
    base = model.get_base_model()
    base.save_pretrained(str(tmp), state_dict=sd, safe_serialization=True, max_shard_size="5GB")
    processor.save_pretrained(str(tmp))
    shutil.rmtree(out_dir, ignore_errors=True)
    os.replace(tmp, out_dir)


def chunk_mlp(model, chunk: int) -> None:
    """A100-40GB port (RL_MLP_CHUNK=<tokens>): the decoder MLPs run over the sequence in chunks of `chunk` tokens, each chunk
    under its own (non-reentrant) activation checkpoint. Row-wise identical math; the layer backward then holds one chunk's
    12288-wide intermediates and their gradients instead of the whole sequence's (~10 GiB at 70k tokens)."""
    from torch.utils.checkpoint import checkpoint

    mlps = {type(layer.mlp) for layer in model.model.language_model.layers}
    for cls in mlps:
        if getattr(cls, "_rl_chunked", None) == chunk:
            continue
        base = getattr(cls, "_rl_base_forward", cls.forward)

        def forward(self, x, _base=base):
            n = x.shape[-2]
            if n <= chunk:
                return _base(self, x)
            if torch.is_grad_enabled():
                parts = [checkpoint(_base, self, x[..., i:i + chunk, :], use_reentrant=False) for i in range(0, n, chunk)]
            else:
                parts = [_base(self, x[..., i:i + chunk, :]) for i in range(0, n, chunk)]
            return torch.cat(parts, dim=-2)

        cls._rl_base_forward, cls.forward, cls._rl_chunked = base, forward, chunk


# GiB of saved activations per 1k tokens of one un-checkpointed full-attention decoder layer (bf16, A100 measurement:
# 3.1 GiB at 16k, 6.3 GiB at 32k tokens), with a 25 % margin
KEEP_ACT_GIB_PER_KTOK = 0.195 * 1.25


def selective_recompute(model, length: int, budget_gib: float) -> int:
    """Speed-up (--keep-attn-act-gib): leave the first k full-attention decoder layers un-checkpointed for
    this pass (their activations are kept instead of recomputed in backward), k = floor(budget / (KEEP_ACT_GIB_PER_KTOK x length/1k)),
    all other layers keep activation checkpointing. Full-attention layers are the ones whose recompute is expensive (quadratic
    attention); GatedDeltaNet layers save ~1.3x the memory for no measurable time. Same math either way (checkpointing only
    replays the forward). budget <= 0: every layer checkpointed (the old behaviour). Returns k."""
    lm = model.model.language_model
    types = getattr(getattr(lm, "config", None), "layer_types", None) or []
    full = [i for i, t in enumerate(types) if t == "full_attention"]
    k = 0 if budget_gib <= 0 else min(len(full), int(budget_gib // (KEEP_ACT_GIB_PER_KTOK * max(1, length) / 1000.0)))
    keep = set(full[:k])
    for i, layer in enumerate(lm.layers):
        if hasattr(layer, "gradient_checkpointing"):
            layer.gradient_checkpointing = i not in keep
    return k


def compile_regional(model) -> int:
    """Speed-up (--compile-regional): torch.compile (dynamic shapes) of every language-model decoder layer's MLP
    and its two RMSNorms, in place (module.compile keeps the module objects, so FSDP units, activation checkpointing and state-dict
    keys are unchanged). Attention / GatedDeltaNet / vision stay eager."""
    n = 0
    for layer in model.model.language_model.layers:
        for name in os.environ.get("RL_COMPILE_PARTS", "mlp,input_layernorm,post_attention_layernorm").split(","):
            sub = getattr(layer, name, None)
            if sub is not None:
                sub.compile(dynamic=True)
                n += 1
    return n


def shard_full_model(model, device, *, trainable: bool):
    """Full fine-tuning (FSDP2): decoder layers, the token embeddings, the language model (final norm) and lm_head are sharded over
    the ranks; the vision tower stays replicated, frozen and in bf16 (ranks whose samples carry different images, or
    none, never wait on each other's vision all-gathers). Trainable models keep fp32 sharded master weights and compute
    in bf16 with fp32 gradient reduction."""
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, OffloadPolicy, fully_shard

    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    # A100-40GB port (both off by default = the H100 behaviour):
    #   RL_FSDP_CPU_OFFLOAD=1 - the trainable model's sharded fp32 params, grads and AdamW state live in pinned host memory
    #     (FSDP2 CPUOffloadPolicy; the optimizer steps on the CPU once per RL round); the frozen KL reference stays on the GPU
    #   RL_LMHEAD_RESHARD=1 - the LM heads (2 x 1.9 GiB bf16 unsharded) are resharded after each call instead of kept
    #   RL_MLP_CHUNK=<tokens> - chunk_mlp above
    if int(os.environ.get("RL_MLP_CHUNK", "0") or 0) > 0:
        chunk_mlp(model, int(os.environ["RL_MLP_CHUNK"]))
    offload = CPUOffloadPolicy(pin_memory=True) if (trainable and os.environ.get("RL_FSDP_CPU_OFFLOAD") == "1") else OffloadPolicy()
    keep_head = os.environ.get("RL_LMHEAD_RESHARD") != "1"
    model.model.visual.to(device=device, dtype=torch.bfloat16)
    for p in model.model.visual.parameters():
        p.requires_grad_(False)
    if not trainable:
        for p in model.parameters():
            p.requires_grad_(False)
    # A100-40GB port: no model.to(device) of the whole (fp32: ~38 GB) model before sharding; fully_shard moves each unit's
    # parameters to the GPU and shards them one unit at a time (same result, peak = one unit instead of the full model)
    lm = model.model.language_model
    for layer in lm.layers:
        fully_shard(layer, mp_policy=mp, offload_policy=offload)
    # the VLM embeds (and splices image features into) the tokens before it calls the language model: the embedding
    # table is its own unit, and the outer VLM module is the FSDP root (called first; the vision tower is ignored) so
    # every unit below it is initialized under one root
    fully_shard(lm.embed_tokens, mp_policy=mp, offload_policy=offload)
    fully_shard(lm, mp_policy=mp, offload_policy=offload)
    fully_shard(model.model, mp_policy=mp, ignored_params=set(model.model.visual.parameters()), offload_policy=offload)
    fully_shard(model.lm_head, mp_policy=mp, reshard_after_forward=not keep_head,     # called on its own: a separate root
                offload_policy=offload)
    on_host = isinstance(offload, CPUOffloadPolicy)
    stray = [n for n, t in [*([] if on_host else model.named_parameters()), *model.named_buffers(),
                            *model.model.visual.named_parameters()] if t.device.type != "cuda"]
    if stray:
        raise RuntimeError(f"tensors left off the GPU after sharding: {stray[:8]}")
    torch.cuda.empty_cache()
    return model


def full_state_dict_bf16(model) -> dict:
    """Collective: the full bf16 state dict on rank 0 (CPU), plain HF key names (empty on other ranks)."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    sd = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    return {k: v.to(torch.bfloat16) for k, v in sd.items()} if dist.get_rank() == 0 else {}


def save_full(model, base_model: str, out_dir: Path, is_main: bool) -> None:
    """Write the full model's current weights (bf16 safetensors shards + index) plus the base model's config/processor
    files, atomically replacing ``out_dir`` (what the policy vLLMs reload)."""
    from safetensors.torch import save_file

    sd = full_state_dict_bf16(model)
    if not is_main:
        return
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for f in Path(base_model).iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json":
            shutil.copy2(f, tmp / f.name)
    shards, current, size, limit = [], {}, 0, 5 * 2**30
    for key in sorted(sd):
        t = sd[key].contiguous()
        if current and size + t.numel() * t.element_size() > limit:
            shards.append(current); current, size = {}, 0
        current[key] = t
        size += t.numel() * t.element_size()
    if current:
        shards.append(current)
    weight_map = {}
    for i, shard in enumerate(shards, start=1):
        name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, str(tmp / name), metadata={"format": "pt"})
        weight_map.update({k: name for k in shard})
    total = sum(t.numel() * t.element_size() for t in sd.values())
    (tmp / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}))
    shutil.rmtree(out_dir, ignore_errors=True)
    os.replace(tmp, out_dir)


def _sync_one(url: str, live_dir: Path, log) -> None:
    requests.post(f"{url}/pause", params={"mode": "wait"}, timeout=1800).raise_for_status()
    r = requests.post(f"{url}/collective_rpc", timeout=1800, json={"method": "reload_weights", "kwargs": {"weights_path": str(live_dir)}, "timeout": 1700})
    r.raise_for_status()
    for route in ("/reset_prefix_cache", "/reset_mm_cache", "/reset_encoder_cache"):
        rr = requests.post(f"{url}{route}", timeout=300)
        rr.raise_for_status()
        if route == "/reset_prefix_cache" and not rr.content:
            raise RuntimeError(f"prefix cache reset returned no success receipt: {url}{route}")
        if rr.content:
            try:
                reset = rr.json()
            except ValueError as exc:
                raise RuntimeError(f"cache reset returned an unverifiable result: {url}{route}") from exc
            if reset is not True and not (isinstance(reset, dict) and reset.get("success") is True):
                raise RuntimeError(f"cache reset did not confirm success: {url}{route}: {reset!r}")
    requests.post(f"{url}/resume", timeout=300).raise_for_status()


# One-card RL (smoke test on a single GPU): vLLM servers (policy, perception) listed here are put to sleep
# (--enable-sleep-mode, VLLM_SERVER_DEV_MODE=1) while the trainer holds activations and woken before rollouts.
SLEEP_URLS = [u.strip() for u in os.environ.get("RL_SLEEP_URLS", "").split(",") if u.strip()]
OFFLOAD_MODEL = os.environ.get("RL_OFFLOAD_MODEL") == "1"
RETURN_COSTS: dict | None = None    # reward costs (set from the CLI in main; None = off)


def PHYS_COTRAIN_ARGS(args) -> bool:
    """the physics-only co-trained critic path (critic_cotrain, critic_hidden none): the only path that reads RETURN_COSTS' critic_target"""
    return bool(args.critic_cotrain and args.critic_hidden == "none")
EVAL_SEEDS = (11, 12, 13, 14, 15, 16)   # official LIBERO evaluation states (fresh-only training states must avoid all of them)
# tensors saved for backward go to pinned host memory (full fine-tuning: fp32 master weights, gradients and Adam states
# fill ~36 GiB per GPU, and a 70k-token request's layer activations would not fit next to them)
OFFLOAD_ACTIVATIONS = {"on": False}   # keep the trainer's weights in host memory while vLLM serves rollouts
class AsyncOffload(torch.autograd.graph.saved_tensors_hooks):
    """Activation offload that does not stall the compute stream (streaming, RL_ASYNC_OFFLOAD=1).

    torch.autograd.graph.save_on_cpu copies every saved tensor to pinned host memory on the compute stream and blocks; on
    A100 nodes without GPUDirect RDMA ( NCCL's IB traffic is staged through host memory) these copies run at ~2 GB/s and
    cost ~5 s of a ~45 s pass. Here a large saved tensor is copied on a side stream (after an event on the producing
    stream) while compute continues; the GPU tensor is kept alive until its copy has finished (then released, so the
    memory saving is the same). Small tensors stay on the GPU. Unpack copies back on the consuming stream after the
    D2H event (the backward H2D was ~1.3 s/pass and stays synchronous)."""

    _stream = None
    _pending: list = []          # (gpu tensor, copy-done event): released only after the copy (module-wide: outlives a context)

    def __init__(self, min_bytes: int = int(os.environ.get("RL_ASYNC_OFFLOAD_MIN_MB", "8")) << 20):
        dev = torch.cuda.current_device()
        if AsyncOffload._stream is None:
            AsyncOffload._stream = torch.cuda.Stream(dev)
        side, pending = AsyncOffload._stream, AsyncOffload._pending

        def reap(block: bool = False):
            while pending and (block or pending[0][1].query()):
                t_gpu, ev = pending.pop(0)
                if block:
                    ev.synchronize()
                del t_gpu

        order: list = []          # offloaded handles in pack order (backward unpacks them roughly in reverse)
        prefetch = os.environ.get("RL_ASYNC_OFFLOAD_PREFETCH", "0") == "1"   # measured slower (backward 24.7 -> 32.4 s on 45k packs)

        def pack(t):
            if not t.is_cuda or t.numel() * t.element_size() < min_bytes:
                return ("gpu", t)
            reap()
            cpu = torch.empty(t.size(), dtype=t.dtype, layout=t.layout, pin_memory=True)
            produced = torch.cuda.Event()
            produced.record()
            with torch.cuda.stream(side):
                side.wait_event(produced)
                cpu.copy_(t, non_blocking=True)
                done = torch.cuda.Event()
                done.record(side)
            pending.append((t, done))
            h = {"cpu": cpu, "done": done, "device": t.device, "i": len(order), "gpu": None, "ready": None}
            order.append(h)
            return ("cpu", h)

        def fetch(h):
            """H2D of an offloaded tensor on the side stream (after its D2H finished); returns nothing, fills h."""
            if h["gpu"] is not None:
                return
            with torch.cuda.stream(side):
                side.wait_event(h["done"])
                h["gpu"] = h["cpu"].to(h["device"], non_blocking=True)
                h["ready"] = torch.cuda.Event()
                h["ready"].record(side)

        def unpack(handle):
            if handle[0] == "gpu":
                return handle[1]
            h = handle[1]
            fetch(h)
            cur = torch.cuda.current_stream()
            cur.wait_event(h["ready"])
            out = h["gpu"]
            out.record_stream(cur)
            h["gpu"] = None
            if prefetch and h["i"] > 0:          # the layer before this one is unpacked next in the backward
                fetch(order[h["i"] - 1])
            reap()
            return out

        self._reap = reap
        super().__init__(pack, unpack)

    def __exit__(self, *exc):
        out = super().__exit__(*exc)
        self._reap(block=False)
        return out


def offload_ctx():
    """The activation-offload context of a training forward (A100 port): synchronous save_on_cpu, or AsyncOffload."""
    import contextlib
    if not OFFLOAD_ACTIVATIONS["on"]:
        return contextlib.nullcontext()
    if os.environ.get("RL_ASYNC_OFFLOAD") == "1":
        return AsyncOffload()
    return torch.autograd.graph.save_on_cpu(pin_memory=True)


# trainer profiling: RL_PROFILE=1 times every pass phase with CUDA syncs (adds small sync overhead) and
# writes <output>/profile/rank<r>.jsonl; RL_PROFILE_TRACE="<step>:<pass>,..." also records a torch.profiler kernel table
PROFILE = {"on": os.environ.get("RL_PROFILE") == "1", "cur": None}


class _PTime:
    def __init__(self, key):
        self.key = key

    def __enter__(self):
        if PROFILE["on"] and PROFILE["cur"] is not None:
            torch.cuda.synchronize()
            self.t = time.time()
        return self

    def __exit__(self, *exc):
        if PROFILE["on"] and PROFILE["cur"] is not None:
            torch.cuda.synchronize()
            PROFILE["cur"][self.key] = PROFILE["cur"].get(self.key, 0.0) + time.time() - self.t
        return False


def vllm_sleep(log, level: int = 1) -> None:
    for url in SLEEP_URLS:
        requests.post(f"{url}/sleep", params={"level": level}, timeout=600).raise_for_status()
    if SLEEP_URLS:
        log(f"[sleep] {len(SLEEP_URLS)} vLLM asleep (level {level})")


def vllm_wake(log) -> None:
    for url in SLEEP_URLS:
        requests.post(f"{url}/wake_up", timeout=600).raise_for_status()
        if requests.get(f"{url}/is_sleeping", timeout=60).json().get("is_sleeping"):
            raise RuntimeError(f"vLLM did not wake up: {url}")
    if SLEEP_URLS:
        log(f"[wake] {len(SLEEP_URLS)} vLLM awake")


LIVE_PUSH = os.environ.get("RL_LIVE_PUSH")   # `<command> <live_dir>` run by rank 0 before the reloads
# direct refit: RL_REFIT=nccl broadcasts the bf16 weights from the trainer straight into the
# policy vLLMs (servers need --weight-transfer-config '{"backend":"nccl"}'; see nccl_refit.py). The live dir is then
# written only on checkpoint rounds (the checkpoint keeps a loadable bf16 copy). Default "file" = export + reload_weights.
REFIT = os.environ.get("RL_REFIT", "file")


def sync_vllm(live_dir: Path, log, push: bool = True) -> None:
    """Reload the merged live weights into every policy vLLM (multi-node RL: the live dir must be on a shared filesystem).
    With RL_LIVE_PUSH set, the live dir is node-local and the command first copies it to the same path on
    every policy node (no shared filesystem fast enough for 19 GB per round)."""
    t0 = time.time()
    file_hosts = set(os.environ.get("RL_LIVE_HOSTS_FILE", "").split())
    urls = [u for u in POLICY_URLS if u not in FILE_URLS]          # the preemptible file group is synced tolerantly below
    if LIVE_PUSH and push:
        import shlex
        import subprocess
        hosts = " ".join(h for h in os.environ.get("RL_LIVE_HOSTS", "").split() if h not in file_hosts)
        subprocess.run([*shlex.split(LIVE_PUSH), str(live_dir)], check=True, env=dict(os.environ, RL_LIVE_HOSTS=hosts))
        log(f"[sync] live dir pushed in {time.time() - t0:.0f}s")
    with ThreadPoolExecutor(max_workers=len(urls)) as pool:
        list(pool.map(lambda u: _sync_one(u, live_dir, log), urls))
    log(f"[sync] {len(urls)} vLLM reloaded in {time.time() - t0:.0f}s")
    try:     # the layerwise reload leaves 4.6-6 GiB cached per server (r5 13:55): give it back (vllm_worker_ext hook)
        from nccl_refit import release_server_cache
        log(f"[sync] released server caches: {release_server_cache(urls, log)}")
    except Exception as exc:  # noqa: BLE001
        log(f"[sync] cache release skipped: {exc!r}")
    if FILE_URLS and not FILE_GROUP["dead"]:
        file_group_refit(live_dir, None, log)


def push_live(live_dir: Path, log) -> None:
    if LIVE_PUSH:
        import shlex
        import subprocess
        t0 = time.time()
        subprocess.run([*shlex.split(LIVE_PUSH), str(live_dir)], check=True)
        log(f"[sync] live dir pushed in {time.time() - t0:.0f}s (episodes keep running)")


def refit_vllm_nccl(model, sender, meta, device, is_main: bool, log) -> dict:
    """Collective (all trainer ranks: the FSDP gathers): pause every policy server (in-flight requests finish), stream the
    bf16 weights over NCCL, finish the layerwise reload, reset the prefix/mm/encoder caches and resume. A failure leaves
    the servers mid-update: it is fatal (restart the policy servers and resume from the last checkpoint)."""
    from nccl_refit import iter_full_bf16, reset_caches
    t0 = time.time()
    urls = sender.urls if is_main else []
    if is_main:
        with ThreadPoolExecutor(max_workers=len(urls)) as pool:
            list(pool.map(lambda u: requests.post(f"{u}/pause", params={"mode": "wait"}, timeout=1800).raise_for_status(), urls))
        # r5 13:47: a server on an EGL-shared GPU (38.2 of 39.5 GB used, 1.77 GiB of it its own cache) OOMed receiving the
        # 1.89 GiB embedding: with the servers paused, return their caches before the transfer allocates
        from nccl_refit import release_server_cache
        info_pre = release_server_cache(urls, log)
        if info_pre:
            log(f"[sync] pre-refit cache release: {info_pre}")
    it = iter_full_bf16(model, meta, device)
    info: dict = {}
    if is_main:
        info = sender.send(meta, it)
        with ThreadPoolExecutor(max_workers=len(urls)) as pool:
            list(pool.map(reset_caches, urls))
            list(pool.map(lambda u: requests.post(f"{u}/resume", timeout=300).raise_for_status(), urls))
        info["t_refit_s"] = round(time.time() - t0, 2)
        log(f"[sync] NCCL refit of {len(urls)} vLLM: {info['bytes'] / 2**30:.1f} GiB in {info['t_refit_s']:.1f}s "
            f"(broadcast {info['t_broadcast_s']:.1f}s)")
    else:
        for _ in it:
            pass
    torch.cuda.synchronize()
    return info


def save_full_training_state(model, opt, staging: Path) -> None:
    """Collective (full fine-tuning): the sharded fp32 master weights and AdamW state with torch.distributed.checkpoint
    (every rank writes its shards) - what an exact resume needs; the bf16 export is only for vLLM and evals."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    msd, osd = get_state_dict(model, opt)
    dcp.save({"model": msd, "optim": osd}, checkpoint_id=str(staging))


def load_full_training_state(model, opt, source: Path) -> None:
    """Collective inverse of save_full_training_state (DCP reshards, so the world size may differ)."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    msd, osd = get_state_dict(model, opt)       # initializes the optimizer state layout (a zero-gradient step)
    state = {"model": msd, "optim": osd}
    dcp.load(state, checkpoint_id=str(source))
    set_state_dict(model, opt, model_state_dict=state["model"], optim_state_dict=state["optim"])


def prune_full_states(output_dir: Path, keep: Path, log) -> None:
    """Keep the ~100 GiB sharded training state only in the newest checkpoint (the resume point); older checkpoints keep
    their bf16 weights, metadata and evals."""
    olds = sorted((p for p in output_dir.glob("checkpoint-*/fsdp") if p.parent.name.split("-")[-1].isdigit()),
                  key=lambda p: int(p.parent.name.split("-")[-1]))
    # RL_KEEP_PREV_FULL_STATE=1 (protects against losing a checkpoint before it is copied elsewhere): also keep the previous
    # checkpoint's state; it is pruned one checkpoint later, when the new one has long been distributed and verified
    keep_prev = None
    if os.environ.get("RL_KEEP_PREV_FULL_STATE") == "1":
        older = [p for p in olds if int(p.parent.name.split("-")[-1]) < int(keep.name.split("-")[-1])]
        keep_prev = older[-1].parent if older else None
    for old in olds:
        if old.parent == keep or old.parent == keep_prev:
            continue
        if old.is_symlink():                      # the state lives on a node-local disk (--full-state-dir)
            shutil.rmtree(old.resolve(), ignore_errors=True)
            old.unlink()
        elif old.is_dir():
            shutil.rmtree(old)
        else:
            continue
        log(f"[checkpoint] pruned the training state of {old.parent.name} (weights kept)")


def _link_or_copy(src, dst):
    """Hard link (instant; save_full never rewrites a file in place - it builds a new directory and swaps it in - so a linked
    checkpoint keeps the exported weights even after the live dir moves on), else copy."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _same_fs(a: Path, b: Path) -> bool:
    return os.stat(a).st_dev == os.stat(b).st_dev


def stage_checkpoint_weights(live_dir: Path, output_dir: Path, step: int) -> Path:
    """Rank 0, synchronous and fast: the exported weights of this step, hard-linked into a private staging directory.
    When the run dir is on another filesystem (e.g. an sshfs mount of the rollout node) the links go next to the live dir
    instead and the copy into the run dir is left to the background publisher (finish_staged_weights): copying 17 GB over
    sshfs on the critical path cost ~85 s per checkpoint."""
    tmp = Path(tempfile.mkdtemp(prefix=f".checkpoint-{step}-", dir=output_dir if _same_fs(live_dir, output_dir) else live_dir.parent))
    shutil.copytree(live_dir, tmp, dirs_exist_ok=True, copy_function=_link_or_copy)
    return tmp


def finish_staged_weights(staged: Path, output_dir: Path, step: int) -> Path:
    """Background: a staging directory outside the run dir (see stage_checkpoint_weights) is copied into it."""
    if staged.parent.resolve() == output_dir.resolve():
        return staged
    tmp = Path(tempfile.mkdtemp(prefix=f".checkpoint-{step}-", dir=output_dir))
    shutil.copytree(staged, tmp, dirs_exist_ok=True)
    shutil.rmtree(staged, ignore_errors=True)
    return tmp


def save_training_checkpoint(model, live_dir: Path, output_dir: Path, step: int, state: dict, log, fsdp_staging: Path | None = None,
                             staged: Path | None = None) -> None:
    """Publish complete model/adapter/optimizer state with one atomic rename (staged: weights already linked by
    stage_checkpoint_weights)."""
    checkpoint = output_dir / f"checkpoint-{step}"
    tmp = staged or Path(tempfile.mkdtemp(prefix=f".checkpoint-{step}-", dir=output_dir))
    try:
        if staged is None:
            shutil.copytree(live_dir, tmp, dirs_exist_ok=True, copy_function=_link_or_copy)
        if fsdp_staging is not None:
            if fsdp_staging.parent.resolve() != output_dir.resolve():
                os.symlink(fsdp_staging, tmp / "fsdp")    # node-local training state (see --full-state-dir)
            else:
                os.replace(fsdp_staging, tmp / "fsdp")
        if hasattr(model, "peft_config"):
            model.save_pretrained(str(tmp / "adapter"))    # full fine-tuning: the weights are the copied live dir
        torch.save(state, tmp / "training_state.pt")
        (tmp / "trainer_state.json").write_text(json.dumps({"global_step": step, "lr": state["args"]["lr"],
                                                           "full_finetune": fsdp_staging is not None,
                                                           "sample_format": "per_request_v1", "args": state["args"]}, default=str))
        if checkpoint.exists():
            raise FileExistsError(f"checkpoint already exists: {checkpoint}; refusing to replace a lineage")
        os.replace(tmp, checkpoint)
    except Exception:
        log(f"[checkpoint] incomplete staging directory retained for diagnosis: {tmp}")
        raise
    log(f"[step {step}] checkpoint saved: {checkpoint}")


# ----------------------------------------------------------------------------- main
def advantage_arg_error(args) -> str | None:
    """Consistency of the advantage-mode and ratio-mask options (None when valid)."""
    band = (args.ratio_mask_low, args.ratio_mask_high)
    if (band[0] is None) != (band[1] is None):
        return "--ratio-mask-low and --ratio-mask-high go together"
    if band[0] is not None:
        if not (0.0 < band[0] < 1.0 and band[1] > 0.0):
            return "ratio mask: 0 < --ratio-mask-low < 1 and --ratio-mask-high > 0"
        if not args.score_centering:
            return "ratio masking needs --score-centering (the sampler log-probabilities come with the top-k record)"
        if args.is_cap is not None:
            return "ratio masking replaces --is-cap; use one off-policy correction"
    if args.advantage_mode == "grpo":
        if args.critic_dir is not None or args.gae_lambda != 0.0 or getattr(args, "critic_cotrain", False) \
                or float(getattr(args, "false_finish_reward", 0.0)):
            return "--critic-dir / --gae-lambda / --critic-cotrain / --false-finish-reward need --advantage-mode physcritic"
        return None
    if getattr(args, "critic_cotrain", False):
        if not getattr(args, "full_finetune", False):
            return "--critic-cotrain reads hidden states from the policy forward; it needs --full-finetune (separate KL reference)"
        if getattr(args, "pack_assignment", "trajectory") != "trajectory" and getattr(args, "critic_hidden", "current") != "none":
            return "--critic-cotrain processes a trajectory's chains newest first on one rank: --pack-assignment trajectory"
        if not (args.critic_lr > 0 and args.critic_updates >= 1):
            return "--critic-lr must be > 0 and --critic-updates >= 1"
    if args.critic_dir is None:
        return "--advantage-mode physcritic needs --critic-dir"
    from physics_critic import validate_critic_dir
    try:
        validate_critic_dir(args.critic_dir)
    except ValueError as exc:
        return str(exc)
    if not 0.0 <= args.gae_lambda <= 1.0:
        return "--gae-lambda must be in [0, 1]"
    ffr = float(getattr(args, "false_finish_reward", 0.0))
    if not -1.0 <= ffr <= 0.0 or (ffr and args.advantage_mode != "physcritic"):
        return "--false-finish-reward must be in [-1, 0] and needs --advantage-mode physcritic"
    if args.advantage_shuffle:
        return "--advantage-shuffle is a GRPO diagnostic"
    if args.reward_mode != "final_success_binary_v1":
        return "physcritic trains on the final 0/1 reward: --reward-mode final_success_binary_v1"
    return None


def tree_cost(lengths: list, prefix: int = 17344, quadratic_tokens: float = 140_000.0) -> int:
    """--prefix-tree rank balancing: estimated learner work of one trajectory from the env server's per-request (chain) lengths:
    c(P) + sum_i (c(L_i) - c(P)), c(n) = n + n^2 / quadratic_tokens (the packed_cost proxy), P ~ the shared prefix."""
    c = lambda n: n + n * n / quadratic_tokens  # noqa: E731
    lens = [int(x) for x in lengths]
    p = max(0, min(prefix, min(lens) - 64))
    return max(1, int(c(p) + sum(c(n) - c(p) for n in lens)))


def run_tree_round(samples, *, args, core, lm_head, model, tree_ref, device, image_pad_id, n_trajectories, full_completion_tokens,
                   world, ratio_band, code_on, acc, rows, step, rank, act_offload, ref_every: int = 1, ref_offset: int = 0) -> dict:
    """--prefix-tree: this rank's samples grouped by trajectory, one rl/prefix_tree.run_episode per trajectory, every rank running
    the same number of episodes x branches (zero-weight dummies) so the FSDP collectives stay in lockstep. The per-branch loss is
    the per-request pass body (sample_loss with the same arguments); accumulators in `acc` are updated in place."""
    import prefix_tree as PT
    groups: dict = {}
    for s in samples:
        groups.setdefault(s["index"], []).append(s)
    plans, fallbacks = [], 0
    for idx, group in groups.items():
        try:
            plans.append(PT.plan_episode(group, core, image_pad_id, device=device))
        except ValueError:
            fallbacks += 1                       # e.g. a prefix image differs: one tree per request (still exact)
            plans += [PT.plan_episode([s], core, image_pad_id, device=device) for s in group]
    # FSDP lockstep: slot (episode e, branch b) costs the slowest rank's; every rank goes longest first (objective is a sum)
    for p in plans:
        p.branches.sort(key=lambda br: -len(br.ids))
    plans.sort(key=lambda p: -(p.P + sum(len(br.ids) for br in p.branches)))
    real_eps = len(plans)
    real_branches = sum(len(p.branches) for p in plans)
    shape = torch.tensor([len(plans), max([len(p.branches) for p in plans], default=1)], device=device)
    dist.all_reduce(shape, op=dist.ReduceOp.MAX)
    n_eps, n_br = int(shape[0]), int(shape[1])
    scales = []
    for p in plans:
        scales.append([1.0] * len(p.branches) + [0.0] * (n_br - len(p.branches)))
        p.branches += [PT.dummy_branch_like(p) for _ in range(n_br - len(p.branches))]
    while len(plans) < n_eps:
        plans.append(PT.dummy_plan(n_br))
        scales.append([0.0] * n_br)
    info = {"episodes": real_eps, "padded_episodes": n_eps - real_eps, "branches": real_branches, "branch_slots": n_eps * n_br,
            "fallback_trajectories": fallbacks, "prefix_tokens": sum(p.P for p in plans[:real_eps]),
            "branch_tokens": sum(len(b.ids) for p in plans[:real_eps] for b in p.branches), "sc_missing": 0, "ref_episodes": 0}
    ref_core, ref_head = tree_ref
    cur_ref_w = [1.0]

    def loss_fn(core_b, inputs, mask, s, ref_lp, scale):
        denominator = (s.get("trajectory_tokens", 1) * n_trajectories) if args.loss_normalization == "trajectory_mean" else full_completion_tokens
        if act_offload[1] is not None:
            OFFLOAD_ACTIVATIONS["on"] = act_offload[0] and int(inputs["input_ids"].shape[1]) >= act_offload[1]
        loss, kl_part, sc_row = sample_loss(model, core_b, lm_head, inputs, mask, s, score_centering=args.score_centering,
                                            kl_beta=args.kl_beta, full_completion_tokens=denominator, world_size=world,
                                            reference=(lambda i, m, r=ref_lp: r) if ref_lp is not None else None,
                                            is_cap=args.is_cap if scale else None, is_stats=rows["is"] if scale else None,
                                            code_weight=args.code_reward_weight if code_on else 0.0,
                                            code_norm=float(args.code_reward_budget * max(1, n_trajectories)),
                                            code_stats=rows["code"] if scale else None, ratio_band=ratio_band if scale else None,
                                            mask_stats=rows["mask"] if scale else None, kl_stats=rows["kl"] if scale else None,
                                            # no reference tree for this episode -> 0 (skip; never the LoRA disable_adapter path)
                                            ref_weight=cur_ref_w[0] if ref_lp is not None else (0.0 if args.kl_beta > 0 else 1.0))
        if scale:
            acc["kl_sum"] += kl_part
            if sc_row is not None:
                acc["sc_stats"] += sc_row
            elif args.score_centering:
                info["sc_missing"] += 1
        else:
            loss = loss * 0.0
        with _PTime("t_backward"):
            loss.backward()
        acc["loss_sum"] += loss.detach()
        ec = os.environ.get("RL_EMPTY_CACHE", "1")
        if ec == "1" or (ec == "auto" and torch.cuda.mem_get_info()[0] < float(os.environ.get("RL_MIN_FREE_GIB", "3")) * 2**30):
            torch.cuda.empty_cache()
        return None

    for ei, (plan, sc) in enumerate(zip(plans, scales)):
        # --ref-every N at episode level: episode slot ei (the same slot sequence on every rank: lockstep) runs the reference
        # tree iff (ei - offset) % N == 0, its KL term and metrics weighted N (inclusion probability 1/N over the offset)
        cur_ref_w[0] = 1.0 if ref_every == 1 else (float(ref_every) if (ei - ref_offset) % ref_every == 0 else 0.0)
        use_ref = ref_core is not None and cur_ref_w[0] != 0.0
        info["ref_episodes"] += int(use_ref and ei < real_eps)
        if PROFILE["on"]:
            torch.cuda.synchronize()
            PROFILE["cur"] = {"step": step, "tree_episode": ei, "P": plan.P, "branches": len(plan.branches),
                              "branch_tokens": sum(len(b.ids) for b in plan.branches), "real": ei < real_eps, "t0": time.time()}
        PT.run_episode(plan, core=core, lm_head=lm_head, device=device, loss_fn=loss_fn, masked_logprobs=masked_logprobs,
                       ref_core=ref_core if use_ref else None, ref_head=ref_head, scales=sc, timer=_PTime)
        if PROFILE["on"]:
            torch.cuda.synchronize()
            row = PROFILE["cur"]
            row["t_episode"] = time.time() - row.pop("t0")
            row["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            torch.cuda.reset_peak_memory_stats()
            pdir = args.output_dir / "profile"
            pdir.mkdir(parents=True, exist_ok=True)
            with open(pdir / f"rank{rank}.jsonl", "a") as fh:
                fh.write(json.dumps({"rank": rank, **row}) + "\n")
            PROFILE["cur"] = None
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--live-dir", type=Path, default=Path("/tmp/captain-policy-live"))
    ap.add_argument("--suites", nargs="*", default=["libero_spatial", "libero_object", "libero_goal"])
    ap.add_argument("--tasks", nargs="*", type=int, default=list(range(10)))
    ap.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3])
    ap.add_argument("--prompts-per-step", type=int, default=8)
    ap.add_argument("--num-generations", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--kl-beta", type=float, default=0.0, help="fixed-sample k3 KL regularizer weight per token (reference = adapters disabled); 0 = off")
    ap.add_argument("--adv-std-floor", type=float, default=0.0, help="floor on the per-group reward std used to normalise advantages (0 = plain GRPO)")
    ap.add_argument("--warmup-steps", type=int, default=2)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--max-seq-tokens", type=int, default=76000, help="maximum original request length; overlong selected requests fail the batch")
    ap.add_argument("--hard-max-tokens", type=int, default=76000, help="hard memory bound; samples are never silently truncated or skipped")
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--save-steps", type=int, default=10)
    ap.add_argument("--max-infrastructure-rate", type=float, default=0.05,
                    help="abort before updating if too many rollouts fail infrastructure or data validation")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--resume-adapter", default=None, help="adapter dir to continue from (weights only)")
    ap.add_argument("--resume-state", type=Path, default=None, help="training_state.pt from the same checkpoint: optimizer, task order, RNG states")
    ap.add_argument("--start-step", type=int, default=0)
    ap.add_argument("--no-sync-at-start", action="store_true", help="skip the initial merged save + reload (vLLM already serves --model)")
    ap.add_argument("--smoke-sample", default=None, help="rollout.json to run one forward/backward + merged save on, then exit (no env server, no vLLM)")
    ap.add_argument("--no-repairs", action="store_true", help="do not train the stored per-request repair samples (re-rendered history blocks stay masked)")
    ap.add_argument("--segment-bank", type=Path, default=None, help="per-call RL: segment-bank/v1 JSON (rl/segment_bank.py); prompts become its trajectories")
    ap.add_argument("--segment-horizon", type=int, default=1, help="live requests per segment rollout (1 = one python call)")
    ap.add_argument("--curriculum-episode-fraction", type=float, default=0.25, help="share of groups that start a fresh episode (K = 0)")
    ap.add_argument("--curriculum-frontier-step", type=int, default=1)
    ap.add_argument("--curriculum-advance-fraction", type=float, default=0.5)
    ap.add_argument("--curriculum-mode", choices=("reverse", "uniform"), default="reverse",
                    help="reverse: per-trajectory frontier from the teacher's last call; uniform: K drawn over every teacher call")
    ap.add_argument("--curriculum-sourced-only", action="store_true", help="drop prompt-only trajectories (no teacher source)")
    ap.add_argument("--curriculum-active-fraction", type=float, default=0.0,
                    help="uniform mode: share of starts drawn among teacher calls that changed the goal potential")
    ap.add_argument("--curriculum-active-min-delta", type=float, default=0.0,
                    help="> 0: an active start's teacher call made at least this much progress (default: any nonzero change)")
    ap.add_argument("--curriculum-fresh-horizon", type=int, default=None,
                    help="scored window of the K=0 starts (default: --segment-horizon); large = complete episodes")
    ap.add_argument("--curriculum-suites", nargs="*", default=None, help="restrict the bank population to these suites")
    ap.add_argument("--probe-set", type=Path, default=None,
                    help="segment-probe-set/v1: fixed starts rolled out without gradient; its held-out trajectories are not trained")
    ap.add_argument("--curriculum-focus-set", default=None,
                    help="name of a probe set whose starts every non-fresh slot cycles through (repeated practice on fixed states)")
    ap.add_argument("--segment-budget", type=int, default=None,
                    help="remaining-task protocol: total request budget B; a teacher start at K gets B-K live requests, and "
                         "starts already solved or with K > B-min_live are not used (the reverse frontier starts at the last unsolved one)")
    ap.add_argument("--fresh-unique-states", action="store_true", help="one training prompt per official initial state (suite, task, seed)")
    ap.add_argument("--fresh-seeds-only", action="store_true",
                    help="the segment bank must hold only prompt-only K = 0 starts on fresh seeds >= 1000 (harness "
                         "fresh-init-states); official and evaluation states are refused at init and at every rollout post")
    ap.add_argument("--fresh-min-tasks", type=int, default=40, help="--fresh-seeds-only: tasks the start population must cover")
    ap.add_argument("--eval-seed", type=int, default=11, help="seed of the fresh evaluation states; fresh-only training refuses to use it")
    ap.add_argument("--segment-min-live", type=int, default=4, help="with --segment-budget: fewest live requests a start may leave")
    ap.add_argument("--segment-handoff", choices=("continuation", "clean"), default="continuation",
                    help="clean: the live policy gets only the physical state at K (fresh Python session, workspace and conversation)")
    ap.add_argument("--offload-activations", choices=("auto", "on", "off"), default="auto",
                    help="keep tensors saved for backward in pinned host memory (auto: on with --full-finetune)")
    ap.add_argument("--curriculum-manifest", type=Path, default=None,
                    help="curriculum-boundaries/v1 manifest (calibrated tracks): remaining-task reverse curriculum with promotion/rollback")
    ap.add_argument("--boundary-fresh-only", action="store_true", help="with --curriculum-manifest: every group starts at K = 0 (F-bin)")
    ap.add_argument("--reward-mode", default=None,
                    choices=("final_success_binary_v1", "staged_progress_v1", "staged_progress_v2"),
                    help="final_success_binary_v1: R = final task success (0/1); staged_progress_v1/v2: host-mark stages "
                         "(v2 also stages the state that closes the window); default: the v13s shaped reward")
    ap.add_argument("--code-reward", choices=("python_exec_cost_v1", "python_exec_contract_v2"), default=None,
                    help="per-request code-execution cost from the env server (rl/code_reward.py): a local loss term on the erroring "
                         "request and no positive task credit for pure errors")
    ap.add_argument("--code-reward-weight", type=float, default=0.0, help="lambda of the code term")
    ap.add_argument("--advantage-shuffle", action="store_true",
                    help="DIAGNOSTIC null control: permute task advantages within each group (zero expected task gradient)")
    ap.add_argument("--code-reward-budget", type=int, default=40, help="fixed request budget H normalizing the code term")
    ap.add_argument("--loss-normalization", choices=("token_mean", "trajectory_mean"), default="token_mean",
                    help="token_mean: divide by all eligible completion tokens of the batch; trajectory_mean: each trajectory by its "
                         "own eligible completion tokens, then the mean over the batch's trajectories")
    ap.add_argument("--skip-zero-signal", action="store_true",
                    help="no optimizer step (and no weight sync) when every advantage of the round is zero")
    ap.add_argument("--warmup-attempts", type=int, default=0,
                    help="> 0: reach --lr after this many valid rollouts (exposure-based warmup) instead of --warmup-steps updates")
    ap.add_argument("--full-state-dir", type=Path, default=None,
                    help="with --full-finetune: write the ~100 GiB sharded training state here (e.g. the trainer node's local disk); "
                         "the checkpoint links to it")
    ap.add_argument("--full-finetune", action="store_true",
                    help="train every language-model weight (FSDP2, fp32 master weights, bf16 compute) instead of LoRA; the vision "
                         "tower stays frozen; the KL reference is a frozen sharded copy of --model. No resume (optimizer state is not saved)")
    ap.add_argument("--score-centering", action="store_true",
                    help="subtract the sampler-expected score (needs the proxy's --sampler-topk); a no-op when trainer and sampler agree")
    ap.add_argument("--probe-every", type=int, default=10)
    ap.add_argument("--probe-generations", type=int, default=2)
    ap.add_argument("--max-infidelity-rate", type=float, default=0.25, help="abort before updating if too many segment replays diverged")
    ap.add_argument("--streaming", action="store_true",
                    help="streaming: a rollout pool keeps env_concurrency episodes running at all times; the learner takes "
                         "the oldest prompts_per_step finished ones; weights are replaced behind the proxy gate while episodes "
                         "run (per-request policy versions, partial rollouts). Needs RL_GATE_URLS and num_generations 1")
    ap.add_argument("--max-staleness", type=int, default=2, help="streaming: requests older than this many optimizer versions "
                    "get no policy-gradient credit (KL term kept)")
    ap.add_argument("--stream-inflight", type=int, default=0, help="streaming: episodes kept in flight (0 = the env servers' max_concurrency)")
    ap.add_argument("--async-lookahead", type=int, choices=(0, 1), default=0,
                    help="1: while rank 0's learner trains batch k, the next curriculum batch is collected with the current weights "
                         "(one-batch lookahead, staleness <= 1 optimizer version; no trajectory spans a weight reload; checkpoint "
                         "rounds drain).")
    ap.add_argument("--pack-prefixes", action="store_true",
                    help="one causal forward per exact-prefix chain of a trajectory's selected requests (raw_request_training."
                         "prefix_chains); per-token inclusion weights keep the objective the sum of the per-request objectives")
    ap.add_argument("--pack-assignment", choices=("trajectory", "pack"), default="trajectory",
                    help="pack (with --pack-prefixes): balance individual packed sequences over the ranks instead of whole trajectories "
                         "(FSDP passes run in lockstep; same samples, same objective)")
    ap.add_argument("--async-checkpoint", action="store_true",
                    help="full fine-tuning: stage the sharded state to host memory and write it in the background "
                         "(torch.distributed.checkpoint.async_save); the checkpoint directory appears once the write completes")
    ap.add_argument("--lookahead-through-checkpoints", action="store_true",
                    help="with --async-lookahead: keep collecting the next batch on checkpoint rounds; the checkpoint stores the "
                         "curriculum state from before that draw, so a resume redraws and recollects it")
    ap.add_argument("--is-cap", type=float, default=None,
                    help="truncated per-token importance weight min(pi/pi_behavior, cap) on the policy-gradient score (the off-policy "
                         "surrogate of the lookahead arm); needs the sampler's log-probabilities")
    ap.add_argument("--advantage-mode", choices=("grpo", "physcritic"), default="grpo",
                    help="grpo: group-normalized trajectory advantages; physcritic: per-request advantages from a frozen physics "
                         "critic (rl/physics_critic.py), globally normalized")
    ap.add_argument("--critic-dir", type=Path, default=None, help="physcritic: frozen critic (feature_spec.json + critic_mlp.pt)")
    ap.add_argument("--gae-lambda", type=float, default=0.0, help="physcritic: GAE lambda over requests (0 = V(s_t+1) - V(s_t))")
    ap.add_argument("--false-finish-reward", type=float, default=0.0,
                    help="physcritic: return R of a false finish (finish() called, host predicate false) in the "
                         "last-step advantage A_T = R - V(s_T); success 1, otherwise 0; the critic target stays the 0/1 success")
    ap.add_argument("--critic-cotrain", action="store_true",
                    help="physcritic: co-train the critic SAO-style (critic_cotrain.py): physics + the current policy's detached "
                         "boundary hidden state, advantages computed inside the training forward, K updates per policy step")
    ap.add_argument("--critic-hidden", choices=("current", "none"), default="current",
                    help="co-trained critic input besides physics: the current policy's boundary hidden state, or none")
    ap.add_argument("--critic-lr", type=float, default=1e-4, help="co-trained critic learning rate (AdamW)")
    ap.add_argument("--critic-updates", type=int, default=2, help="co-trained critic updates per policy step (SAO K)")
    ap.add_argument("--start-exclude", type=Path, default=None,
                    help="segment curriculum: start-exclude/v1 file; these training starts are never drawn (thin-basket fix)")
    ap.add_argument("--start-weights", type=Path, default=None,
                    help="segment curriculum: start-weights/v1 file (rl/make_start_weights.py); a drawn start is kept with its weight")
    # reward costs, all off by default (off = the plain objective bit for bit)
    ap.add_argument("--act-cost", type=float, default=0.0,
                    help="immediate reward -c per physical robot action of a request beyond the task reference length (--act-ref)")
    ap.add_argument("--fail-cost", type=float, default=0.0,
                    help="immediate reward -c per failed / rejected / stopped-short physical action of a request")
    ap.add_argument("--act-ref", type=Path, default=None,
                    help="JSON {'ref': {'suite:task': actions, 'suite:*': fallback}}: --act-cost charges only actions beyond it")
    ap.add_argument("--missed-finish-return", type=float, default=None,
                    help="return of a host success without finish() (e.g. 0.5); None = 1 like a finished success")
    ap.add_argument("--exhaust-reward", type=float, default=None,
                    help="return of a failure that exhausted a budget (request cap, action budget, wall) (e.g. -0.1); None = 0")
    ap.add_argument("--keep-attn-act-gib", type=float, default=0.0,
                    help="speed-up: GiB of extra activation memory per pass for keeping (not recomputing) full-attention layers' "
                         "activations (rl/train.selective_recompute); 0 = every layer checkpointed (old behaviour)")
    ap.add_argument("--compile-regional", action="store_true",
                    help="speed-up: torch.compile the decoder layers' MLP + RMSNorms (policy and KL reference); default eager")
    ap.add_argument("--reward-patch", default=None,
                    help="missed-finish fix: JSON {missed_finish_return, post_success_act} applied on top of "
                         "the reward costs; post_success_act charges --act-cost on every physical action from the first request whose mark "
                         "shows all goals satisfied (perception free); post_success_req = flat cost on EVERY "
                         "request from that request on (b), critic_target success_finish = co-critic target host success AND finish (c); "
                         "default off")
    ap.add_argument("--staged-reward", default=None,
                    help="staged one-shot subgoal reward: JSON {credit, ladders: {suite:task_id: "
                         "[rung, ...]}, grasp?}; a laddered task's episode that is neither a host success nor a false finish and reaches "
                         "d >= 1 of its n rungs (host marks only, each rung once, in order) gets the return credit x d / (n + 1) instead of "
                         "0 / exhaust; needs the reward costs and the physics-only co-trained critic; from the config on every start "
                         "(fresh_run STAGE_SWITCH_ALLOWANCES); default off")
    ap.add_argument("--max-overlength-drops", type=int, default=0,
                    help="over-length guard (trainer side): up to N trajectories per round whose ONLY build errors (on all ranks, "
                         "after the all-gather) are OverLengthRollout are dropped on every rank (objective tokens, trajectory count and "
                         "segment count adjusted) instead of refusing the batch; any other build error or more than N stays fail-closed; "
                         "from the config (fresh_run overlength_guard) on every start; default 0 = off")
    ap.add_argument("--heldout-monitor", default=None,
                    help='held-out monitor JSON {"every": 10, "ev12_seeds": [...], "official_seeds": [...]} (no gradient; streaming only)')
    ap.add_argument("--attn-backend", choices=("sdpa", "fa3", "sdpa_cudnn"), default=None,
                    help="speed-up: attention kernel of the language model's full-attention layers (rl/attn_backends.py); default = SDPA")
    ap.add_argument("--attn-overlay", default=None, help="directory holding the prebuilt flash_attn_interface (FA3) package")
    ap.add_argument("--cocritic-live", type=Path, default=None,
                    help="speed-up 3: rank 0 writes the co-critic weights here after every critic update (env-server early stop)")
    ap.add_argument("--ref-every", type=int, default=1,
                    help="speed-up 2: run the frozen KL reference forward on 1 in N passes (a random offset per round, the same on "
                         "every rank) and weight that pass's KL term and KL metric by N (unbiased); 1 = every pass (old behaviour)")
    # group sampling for the streaming pool, off by default (1 = the plain pool bit for bit). Not
    # --num-generations: that also sets the policy proxy's worker routing (group * G) and the boundary curriculum's group size
    ap.add_argument("--stream-group-size", type=int, default=1,
                    help="streaming: run every drawn start this many times (2 or 4) and take complete groups; prompts_per_step "
                         "stays the batch in episodes (prompts_per_step / G distinct starts)")
    ap.add_argument("--group-advantage", choices=("group", "critic"), default="group",
                    help="with --stream-group-size > 1: 'group' = outcome part group-relative (G_i - group mean of the total return) "
                         "+ the critic's per-request shaping; 'critic' = plain critic GAE advantages (only the sampling changes)")
    ap.add_argument("--grad-accum-rounds", type=int, default=1,
                    help="one optimizer step per this many collection rounds (gradients of the rounds summed with the normalisation of "
                         "one big batch, rl/grad_accum.py); checkpoints only at cycle boundaries (save_steps, start_step, max_steps % K == 0)")
    ap.add_argument("--ratio-mask-low", type=float, default=None,
                    help="SAO double-sided masking: drop policy-gradient tokens with exp(log p - log p_rollout) < 1 - this")
    ap.add_argument("--ratio-mask-high", type=float, default=None,
                    help="SAO double-sided masking: drop policy-gradient tokens with exp(log p - log p_rollout) > 1 + this")
    # shared-prefix training (opt-in; off = the per-request passes bit for bit): rl/prefix_tree.py
    ap.add_argument("--prefix-tree", action="store_true",
                    help="shared-prefix training: per trajectory, the requests' common token prefix (system prompt, tools, task, first "
                         "frame) is run once (ref forward, policy forward, backward) and every trained request is a branch on top of "
                         "its states; same objective and gradient as the per-request passes (rl/prefix_tree.py)")
    args = ap.parse_args()
    error = advantage_arg_error(args)
    if error:
        ap.error(error)
    if args.pack_assignment == "pack" and not args.pack_prefixes:
        ap.error("--pack-assignment pack needs --pack-prefixes")
    if args.prefix_tree:
        if not args.full_finetune:
            ap.error("--prefix-tree is implemented for --full-finetune (frozen reference copy)")
        if args.pack_assignment != "trajectory" or args.pack_prefixes:
            ap.error("--prefix-tree needs every request of a trajectory on one rank, unpacked: --pack-assignment trajectory, no --pack-prefixes")
        if args.critic_cotrain and args.critic_hidden != "none":
            ap.error("--prefix-tree: the hidden-state co-trained critic reads the per-request forward; use --critic-hidden none")
        if os.environ.get("RL_FA2"):
            ap.error("--prefix-tree wraps the sdpa attention interface; unset RL_FA2")
    if args.async_checkpoint and not args.full_finetune:
        ap.error("--async-checkpoint is implemented for --full-finetune (sharded DCP state)")
    if args.lookahead_through_checkpoints and not args.async_lookahead:
        ap.error("--lookahead-through-checkpoints needs --async-lookahead")
    if args.async_lookahead and (args.segment_bank is None or OFFLOAD_MODEL or SLEEP_URLS):
        ap.error("--async-lookahead needs a segment curriculum and dedicated rollout GPUs (no model offload / vLLM sleep)")
    if (args.code_reward is None) != (args.code_reward_weight == 0.0) or args.code_reward_weight < 0 or args.code_reward_budget < 1:
        ap.error("--code-reward needs a positive --code-reward-weight (and a weight needs --code-reward); budget >= 1")
    if args.is_cap is not None and not (args.is_cap > 1.0 and args.score_centering):
        ap.error("--is-cap must be > 1 and needs --score-centering (the sampler log-probabilities come with the top-k record)")
    if args.no_repairs:
        ap.error("--no-repairs is obsolete; live training uses raw per-request samples")
    if args.ref_every < 1:
        raise SystemExit("--ref-every must be >= 1")
    if args.grad_accum_rounds < 1:
        ap.error("--grad-accum-rounds must be >= 1")
    if args.stream_group_size < 1 or (args.stream_group_size > 1 and (not args.streaming or args.prompts_per_step % args.stream_group_size)):
        ap.error("--stream-group-size G > 1 needs --streaming and prompts_per_step divisible by G")
    if args.stream_group_size > 1 and args.group_advantage == "group" and not (args.advantage_mode == "physcritic" and args.critic_cotrain
                                                                             and args.critic_hidden == "none"):
        ap.error("--group-advantage group needs the physics-only co-trained critic (advantage_mode physcritic, critic_hidden none)")
    if args.act_cost < 0 or args.fail_cost < 0:
        ap.error("--act-cost / --fail-cost must be >= 0")
    global RETURN_COSTS
    if args.act_cost or args.fail_cost or args.missed_finish_return is not None or args.exhaust_reward is not None:
        if args.advantage_mode != "physcritic":
            ap.error("the reward costs apply to advantage_mode physcritic")
        RETURN_COSTS = {"c_act": args.act_cost, "c_fail": args.fail_cost, "missed_finish_return": args.missed_finish_return,
                        "exhaust_reward": args.exhaust_reward,
                        "ref": json.loads(args.act_ref.read_text())["ref"] if args.act_ref else {}}
    if args.reward_patch:
        if RETURN_COSTS is None:
            ap.error("--reward-patch needs the reward costs on")
        patch = json.loads(args.reward_patch)
        if set(patch) - {"missed_finish_return", "post_success_act", "post_success_req", "post_success_cap", "critic_target"}:
            ap.error("--reward-patch keys: missed_finish_return, post_success_act, post_success_req, post_success_cap, critic_target")
        if "post_success_cap" in patch and (not patch.get("post_success_req") or type(patch["post_success_cap"]) not in (int, float)
                                            or not 0 < patch["post_success_cap"] <= 1):
            ap.error("--reward-patch post_success_cap needs post_success_req and must be in (0, 1]")
        if "post_success_req" in patch and (type(patch["post_success_req"]) not in (int, float) or not 0 <= patch["post_success_req"] <= 0.05):
            ap.error("--reward-patch post_success_req must be in [0, 0.05]")
        if patch.get("critic_target", "success") not in ("success", "success_finish"):
            ap.error("--reward-patch critic_target must be success or success_finish")
        if patch.get("critic_target") == "success_finish" and not PHYS_COTRAIN_ARGS(args):
            ap.error("--reward-patch critic_target needs the physics-only co-trained critic (critic_cotrain, critic_hidden none)")
        RETURN_COSTS = dict(RETURN_COSTS, **patch)
    if args.staged_reward:
        if RETURN_COSTS is None or not PHYS_COTRAIN_ARGS(args):
            ap.error("--staged-reward needs the reward costs on and the physics-only co-trained critic (critic_cotrain, critic_hidden none)")
        from physics_critic import staged_spec_error
        staged = json.loads(args.staged_reward)
        err = staged_spec_error(staged, RETURN_COSTS.get("missed_finish_return"))
        if err:
            ap.error("--staged-reward: " + err)
        RETURN_COSTS = dict(RETURN_COSTS, staged=staged)
        print(f"[staged] staged one-shot subgoal reward ON: credit {staged['credit']}, ladders {sorted(staged['ladders'])}", flush=True)
    if args.grad_accum_rounds > 1:
        k = args.grad_accum_rounds
        if args.save_steps % k or args.start_step % k or args.max_steps % k:
            ap.error("--grad-accum-rounds K needs save_steps, start_step and max_steps divisible by K (no partial gradient at a checkpoint)")
        if OFFLOAD_MODEL:
            ap.error("--grad-accum-rounds > 1 is not implemented with RL_OFFLOAD_MODEL (the model would leave the GPU between rounds)")
        if args.code_reward is not None and args.loss_normalization != "trajectory_mean":
            ap.error("--grad-accum-rounds > 1 with the code term needs loss_normalization trajectory_mean (both terms scale with 1/N)")
    if args.start_step and args.resume_state is None:
        ap.error("nonzero --start-step requires --resume-state; adapter-only resumes change the experiment")
    if args.resume_adapter and args.resume_state is None:
        ap.error("adapter-only initialization is forbidden; fresh RL starts from --model without an RL adapter")
    if (args.output_dir / "metrics.jsonl").exists() and args.resume_state is None:
        ap.error("output already contains metrics; use a new run directory or a full-state resume")

    # the env phase (rank 0 collects rollouts, ranks 1-3 wait in broadcast_object_list) can exceed NCCL's default 10 min once episodes
    # run to the 900 s wall / request cap instead of dying on context (v11c+window, step 51 crashed on the watchdog timeout)
    from datetime import timedelta
    if "LOCAL_RANK" in os.environ:      # bind the GPU before NCCL init (else every rank leaves a ~0.4 GiB context on GPU 0)
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    # RL_FSDP_CPU_OFFLOAD: the gradient-norm all-reduce runs on CPU DTensors -> the default group also needs a CPU backend
    dist.init_process_group("cpu:gloo,cuda:nccl" if os.environ.get("RL_FSDP_CPU_OFFLOAD") == "1" else "nccl", timeout=timedelta(hours=3))
    # Where rank 0 works alone for minutes (rollouts, probes, checkpoint saves) the other ranks wait on this CPU group:
    # a NCCL collective left pending that long trips the NCCL watchdog (heartbeat 480 s) and aborts the job.
    cpu_group = dist.new_group(backend="gloo", timeout=timedelta(hours=3))
    # background checkpoint writes (async_save) run their planning/finish collectives in a worker thread: a group of their own
    # keeps them from interleaving with the main thread's cpu_group barriers and broadcasts
    ckpt_group = dist.new_group(backend="gloo", timeout=timedelta(hours=3))
    rank, world = dist.get_rank(), dist.get_world_size()
    if os.environ.get("RL_CPU_THREADS"):        # host-side AdamW (RL_FSDP_CPU_OFFLOAD) runs on torch's CPU threads
        torch.set_num_threads(int(os.environ["RL_CPU_THREADS"]))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    # CUDA's current device is per thread and new threads start on cuda:0.
    # The worker threads of dcp.async_save (and the checkpoint publish thread) each opened a ~0.4 GiB context on GPU 0 of
    # their node, 7 x 416 MiB on local rank 0, which then OOMed in r4 round 6. Bind every later thread to this rank's
    # device first (binding the existing primary context allocates nothing).
    import threading as _threading
    _thread_run = _threading.Thread.run

    def _run_on_rank_device(self, *a, **k):
        torch.cuda.set_device(local_rank)
        return _thread_run(self, *a, **k)

    _threading.Thread.run = _run_on_rank_device
    if os.environ.get("RL_MEM_FRACTION"):
        # streaming: cap the caching allocator (reserved) so nvidia-smi keeps headroom on every pack; the allocator
        # frees its own cached blocks before exceeding the cap (instead of an empty_cache after every pass)
        torch.cuda.set_per_process_memory_fraction(float(os.environ["RL_MEM_FRACTION"]), local_rank)
    device = torch.device("cuda", local_rank)
    is_main = rank == 0
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train.log"

    def log(msg: str) -> None:
        if is_main:
            line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
            print(line, flush=True)
            with log_path.open("a") as fh:
                fh.write(line + "\n")

    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import LoraConfig, get_peft_model, PeftModel

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if is_main and SLEEP_URLS:
        vllm_sleep(log)
    dist.barrier()
    attn = "flash_attention_2" if os.environ.get("RL_FA2") else None
    if args.attn_backend not in (None, "sdpa"):
        # FlashAttention-3 / cuDNN SDPA for the language model's full-attention layers
        import attn_backends
        if attn is not None:
            raise SystemExit("--attn-backend and RL_FA2 are exclusive")
        attn = attn_backends.install(args.attn_backend, args.attn_overlay)
        log(f"[init] attention backend {args.attn_backend} (attn_implementation {attn})")
    OFFLOAD_ACTIVATIONS["on"] = args.offload_activations == "on" or (args.offload_activations == "auto" and args.full_finetune)
    OFFLOAD_BASE = OFFLOAD_ACTIVATIONS["on"]
    # RL_ACT_OFFLOAD_MIN_TOKENS=<n> offloads saved activations only for sequences of >= n tokens
    ACT_OFFLOAD_MIN = int(os.environ["RL_ACT_OFFLOAD_MIN_TOKENS"]) if os.environ.get("RL_ACT_OFFLOAD_MIN_TOKENS") else None
    # physics-only co-trained critic: advantages precomputed on all request states (allows request subsampling)
    PHYS_COTRAIN = bool(args.critic_cotrain and args.critic_hidden == "none")
    reference = None
    if args.full_finetune:
        if args.resume_adapter:
            raise SystemExit("--full-finetune resumes from the checkpoint's sharded training state (--resume-state), not an adapter")
        model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.float32, attn_implementation=attn)
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if args.compile_regional:
            log(f"[init] regional torch.compile: {compile_regional(model)} modules (policy)")
        model = shard_full_model(model, device, trainable=True)
        if args.kl_beta > 0:
            ref_model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation=attn)
            ref_model.config.use_cache = False
            if args.compile_regional:
                compile_regional(ref_model)
            ref_model = shard_full_model(ref_model, device, trainable=False).eval()
            reference = lambda inputs, mask: masked_logprobs(ref_model.model, ref_model.lm_head, inputs, mask, per_token=True)  # noqa: E731
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)       # DTensor numel is the global size
        log(f"[init] full fine-tuning: {n_train / 1e9:.2f}B trainable (sharded over {world}), vision tower frozen, "
            f"KL reference {'frozen sharded copy' if reference else 'off'}")
    else:
        model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation=attn)
        model.to(device)
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        targets = lora_targets(model)
        log(f"[init] lora targets ({len(targets)}): {targets[:12]}{' ...' if len(targets) > 12 else ''}")
    if args.full_finetune:
        pass
    elif args.resume_adapter:
        model = PeftModel.from_pretrained(model, args.resume_adapter, is_trainable=True)
    else:
        lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, target_modules=targets, task_type="CAUSAL_LM")
        model = get_peft_model(model, lcfg)
    if not args.full_finetune:
        model.print_trainable_parameters() if is_main else None
    params = [p for p in model.parameters() if p.requires_grad]
    from grad_accum import RoundAccumulator
    accum = RoundAccumulator(args.grad_accum_rounds)
    if not args.full_finetune:
        # Manual gradient synchronization needs the same initial adapters on every rank (FSDP shards one copy instead).
        for parameter in params:
            dist.broadcast(parameter.data, src=0)
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    resume_state = None
    if args.resume_state:
        resume_state = load_resume(args, world)
        for switch in resume_state.get("resume_switches") or []:      # pinned switches only (training_resume.RESUME_SWITCHES)
            log(f"[resume] pinned argument switch accepted: {json.dumps(switch, sort_keys=True)}")
        if args.full_finetune:
            load_full_training_state(model, opt, Path(args.resume_state).parent / "fsdp")
            log(f"[resume] full fine-tuning state (fp32 weights, AdamW) loaded from {Path(args.resume_state).parent.name}")
        else:
            opt.load_state_dict(resume_state["optimizer"])
        if is_main:
            log(f"[resume] {json.dumps(prepare_resume_logs(args.output_dir, args.start_step))}")
        dist.barrier()
    base = model if args.full_finetune else model.get_base_model()
    core = base.model  # Qwen3_5Model (vision + language), returns hidden states
    lm_head = base.lm_head
    if args.prefix_tree:
        import prefix_tree as PT
        PT.install_attention()                     # sdpa wrapper: only PrefixBranchMask calls change
        PT.allow_cache_under_checkpointing(core)
        PT.install_capture_hooks(core)             # inactive outside a prefix capture forward
        tree_ref = (ref_model.model, ref_model.lm_head) if (args.kl_beta > 0 and reference is not None) else (None, None)
        log(f"[init] prefix tree on (rl/prefix_tree.py): KL reference {'tree' if tree_ref[0] is not None else 'off'}")

    def publish_weights() -> None:
        """Write the current policy to the live dir (all ranks participate for the full model)."""
        if args.full_finetune:
            save_full(model, args.model, args.live_dir, is_main)
        else:
            save_merged(model, processor, args.live_dir, is_main)

    if args.smoke_sample:
        t0 = time.time()
        raw = read_rollout_json(args.smoke_sample)
        built, _ = build_request_samples(raw, processor, image_pad_id=image_pad_id,
                                        hard_max_tokens=min(args.max_seq_tokens, args.hard_max_tokens))
        s = max(built, key=lambda b: b["length"])
        log(f"[smoke] length {s['length']} trainable {s['n_tokens']} images {int(s['image_grid_thw'].shape[0]) if 'image_grid_thw' in s else 0} build {time.time() - t0:.1f}s")
        inputs = {"input_ids": s["input_ids"].unsqueeze(0).to(device), "attention_mask": torch.ones(1, s["length"], dtype=torch.long, device=device)}
        if "pixel_values" in s:
            inputs["pixel_values"] = s["pixel_values"].to(device, dtype=torch.bfloat16)
            inputs["image_grid_thw"] = s["image_grid_thw"].to(device)
            inputs["mm_token_type_ids"] = (inputs["input_ids"] == image_pad_id).long()  # transformers 5: M-RoPE needs 0 text / 1 image
        mask = s["mask"].unsqueeze(0).to(device)
        model.train()
        s = {**s, "advantage": 1.0, "weight": 1.0}
        if args.keep_attn_act_gib > 0 and args.full_finetune:
            log(f"[smoke] selective recompute: {selective_recompute(model, int(s['length']), args.keep_attn_act_gib)} full-attention layers kept")
        for it in range(1, int(os.environ.get("RL_SMOKE_PASSES", "2")) + 1):   # pass 2+ = steady state (optimizer state allocated)
            torch.cuda.reset_peak_memory_stats(device)
            t1 = time.time()
            opt.zero_grad(set_to_none=True)
            loss, kl, _ = sample_loss(model, core, lm_head, inputs, mask, s, score_centering=False, kl_beta=args.kl_beta,
                                      full_completion_tokens=max(1, s["n_tokens"]), world_size=world, reference=reference)
            loss.backward()
            torch.cuda.synchronize(device)
            gn = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm, error_if_nonfinite=True)
            gn = gn.full_tensor() if hasattr(gn, "full_tensor") else gn
            opt.step()
            log(f"[smoke] rank {rank} pass {it} loss {float(loss):.4f} kl {float(kl):.5f} fwd+bwd+step {time.time() - t1:.1f}s "
                f"peak {torch.cuda.max_memory_allocated(device) / 2**30:.1f} GiB grad_norm {float(gn):.3f}")
        dist.barrier()
        if os.environ.get("RL_SMOKE_TIMING_ONLY") == "1":     # timing check: passes only, nothing written
            dist.destroy_process_group()
            return
        if args.full_finetune:          # exact resume: the sharded fp32 weights and AdamW state round-trip
            probe_param = next(p for p in params if p.requires_grad)
            before = (float(probe_param.to_local().double().sum()), float(sum(float(v["exp_avg"].to_local().double().abs().sum())
                                                                              for v in opt.state.values())))
            t3 = time.time()
            save_full_training_state(model, opt, args.output_dir / "fsdp-smoke")
            with torch.no_grad():
                for p in params:
                    p.to_local().mul_(0.0)
                for v in opt.state.values():
                    v["exp_avg"].to_local().zero_()
            load_full_training_state(model, opt, args.output_dir / "fsdp-smoke")
            after = (float(probe_param.to_local().double().sum()), float(sum(float(v["exp_avg"].to_local().double().abs().sum())
                                                                             for v in opt.state.values())))
            log(f"[smoke] rank {rank} fsdp state save+load {time.time() - t3:.0f}s: before {before} after {after} match {before == after}")
        t2 = time.time()
        publish_weights()
        dist.barrier()
        log(f"[smoke] merged save {time.time() - t2:.0f}s -> {args.live_dir}")
        dist.destroy_process_group()
        return

    critic = None
    cocritic = cocritic_opt = None
    cocritic_sigma = None
    if args.critic_cotrain:
        import critic_cotrain as CC
        hidden_dim = int(getattr(base.config, "text_config", base.config).hidden_size)
        cocritic = CC.CoCritic(args.critic_dir, hidden_dim, use_hidden=args.critic_hidden == "current").to(device)
        cocritic.eval()                     # no dropout; trained by critic_cotrain.update
        cocritic_opt = torch.optim.AdamW([q for q in cocritic.parameters() if q.requires_grad], lr=args.critic_lr, weight_decay=1e-4)
        if resume_state is not None and resume_state.get("cocritic"):
            cocritic.load_state_dict(resume_state["cocritic"]["state_dict"])
            cocritic_opt.load_state_dict(resume_state["cocritic"]["optimizer"])
            cocritic_sigma = resume_state["cocritic"].get("sigma")
            opt_steps = max((int(st.get("step", 0)) for st in (cocritic_opt.state_dict().get("state") or {}).values()
                             if isinstance(st, dict) and "step" in st), default=0)
            log(f"[critic] co-critic restored from {args.resume_state} (opt steps {opt_steps}, sigma {cocritic_sigma})")
        elif resume_state is not None:
            log(f"[critic] WARNING: {args.resume_state} holds no co-critic state: the co-critic starts from {args.critic_dir}")
        log(f"[critic] co-trained: hidden {args.critic_hidden} dim {hidden_dim} lr {args.critic_lr} K {args.critic_updates}")
        if is_main and args.cocritic_live:
            export_cocritic(cocritic, args.cocritic_live, args.start_step)
    if args.advantage_mode == "physcritic" and is_main:
        from physics_critic import FrozenCritic
        critic = FrozenCritic(args.critic_dir)
        PREFETCH_CRITIC_ROWS["on"] = bool(args.streaming) and os.environ.get("RL_PREFETCH_CRITIC_ROWS", "1") == "1"
        log(f"[critic] {args.critic_dir} features {len(critic.features)} time features {critic.uses_time or 'none'} "
            f"lambda {args.gae_lambda}")
    ratio_band = (args.ratio_mask_low, args.ratio_mask_high) if args.ratio_mask_low is not None else None
    curriculum = None
    probe = {"sets": {}, "heldout_trajectories": []}
    probe_entries: list[dict] = []
    if args.segment_bank is not None:
        from segment_bank import load_bank
        from segment_curriculum import BoundaryCurriculum, SegmentCurriculum, load_boundary_manifest, load_probe_set, load_start_weights, load_start_exclude
        bank = load_bank(args.segment_bank)
        probe = load_probe_set(args.probe_set, bank)
        if args.curriculum_manifest:
            # remaining-task reverse curriculum over calibrated tracks (or, with --boundary-fresh-only, their K = 0 starts)
            curriculum = BoundaryCurriculum(bank, load_boundary_manifest(args.curriculum_manifest, bank), budget=args.segment_budget or 40,
                                            handoff=args.segment_handoff, reward_mode=args.reward_mode, group_size=args.num_generations,
                                            fresh_only=args.boundary_fresh_only, seed=args.seed)
            log(f"[init] boundary curriculum: {len(curriculum.prompts)} tracks, fresh_only={args.boundary_fresh_only}, "
                f"reward={args.reward_mode or 'v13s'}, frontiers {dict(curriculum.frontier)}")
        else:
            curriculum = None
    if args.segment_bank is not None and curriculum is None:
        curriculum = SegmentCurriculum(bank, horizon=args.segment_horizon,
                                       episode_fraction=args.curriculum_episode_fraction, frontier_step=args.curriculum_frontier_step,
                                       advance_fraction=args.curriculum_advance_fraction, seed=args.seed,
                                       mode=args.curriculum_mode, sourced_only=args.curriculum_sourced_only,
                                       active_fraction=args.curriculum_active_fraction, fresh_horizon=args.curriculum_fresh_horizon,
                                       active_min_delta=args.curriculum_active_min_delta, suites=args.curriculum_suites,
                                       exclude=probe["heldout_trajectories"],
                                       focus_starts=focus_list(probe, args.curriculum_focus_set),
                                       budget=args.segment_budget, min_live=args.segment_min_live, handoff=args.segment_handoff,
                                       unique_states=args.fresh_unique_states, reward_mode=args.reward_mode,
                                       train_all_on_success=bool(args.reward_mode),
                                       start_weights=load_start_weights(args.start_weights),
                                       start_exclude=load_start_exclude(args.start_exclude))
        if args.start_exclude is not None:
            log(f"[init] start exclusion {args.start_exclude}: {len(curriculum.excluded)} of {len(curriculum.trajectories)} "
                f"training starts never drawn (sha {curriculum.start_exclude_sha[:12]})")
        if args.curriculum_episode_fraction >= 1.0 and isinstance(curriculum, SegmentCurriculum):
            # fresh-only protocol: every rollout starts from an official initial state (K = 0: no teacher replay, a new
            # session and workspace); the training states must not be the evaluation's (seed --eval-seed)
            states = {(t["suite"], int(t["task_id"]), int(t["seed"])) for t in curriculum.trajectories.values()}
            clash = sorted(st for st in states if st[2] in {args.eval_seed, *EVAL_SEEDS})
            if clash:
                raise SystemExit(f"fresh training states include the evaluation seed {args.eval_seed}: {clash[:3]}")
            if args.fresh_seeds_only:
                from segment_curriculum import check_fresh_population
                summary = check_fresh_population(curriculum.trajectories, min_tasks=args.fresh_min_tasks)
                FRESH_ONLY["on"] = True
                log(f"[init] fresh-seeds-only: {summary['states']} FRESH initial states (seeds >= 1000) over {summary['tasks']} tasks "
                    f"({summary['per_task_min']}-{summary['per_task_max']} per task), drawn without replacement per epoch; "
                    f"official and evaluation states refused")
            else:
                log(f"[init] fresh-only: {len(states)} official initial states over {sorted({st[0] for st in states})}, "
                    f"seeds {sorted({st[2] for st in states})}")
    if args.fresh_seeds_only and not FRESH_ONLY["on"]:
        raise SystemExit("--fresh-seeds-only needs a segment bank (fresh starts) and --curriculum-episode-fraction 1.0")
    if args.fresh_seeds_only and probe["sets"]:
        raise SystemExit("--fresh-seeds-only: probe sets hold official-state starts; run without --probe-set")
    if args.segment_bank is not None:
        # probes run under the training protocol (window, budget, handoff) so their numbers describe what is trained
        probe_entries = [curriculum.dress(bank["starts"][start]) for ids in probe["sets"].values() for start in ids]
        if resume_state is not None:
            curriculum.load_state_dict(resume_state["segment_curriculum"])
            if getattr(curriculum, "start_exclude_previous", None):
                log(f"[init] start exclusion changed on resume: {curriculum.start_exclude_previous[:12]} -> "
                    f"{(curriculum.start_exclude_sha or 'none')[:12]} (population, order and rng state kept)")
            if getattr(curriculum, "start_weights_previous", None):
                log(f"[init] start weights switched on resume (pinned): {curriculum.start_weights_previous[:12]} -> "
                    f"{curriculum.start_weights_sha[:12]} (population, order and rng state kept)")
    prompts = curriculum.prompts if curriculum is not None else community_prompts(args.suites, args.tasks, args.seeds)
    rng = random.Random(args.seed)
    order = list(range(len(prompts)))
    rng.shuffle(order)
    cursor = 0
    if resume_state is not None:
        if resume_state["prompts"] != prompts:
            raise ValueError("resume prompt population changed")
        order = resume_state["prompt_order"]
        cursor = resume_state["prompt_cursor"]
        rng.setstate(resume_state["prompt_rng"])
        if len(resume_state["rank_rng"]) == world:
            state = resume_state["rank_rng"][rank]
            random.setstate(state["python"])
            torch.set_rng_state(state["torch"])
            torch.cuda.set_rng_state(state["cuda"], device)
        else:                                   # full FT resharded onto another world size: fresh per-rank streams
            random.seed(args.seed * 100003 + args.start_step * 1009 + rank)
            torch.manual_seed(args.seed * 100003 + args.start_step * 1009 + rank)
            log(f"[resume] world {resume_state['world_size']} -> {world}: fp32 weights/AdamW resharded, per-rank RNG reseeded")
    rollouts_path = args.output_dir / "rollouts.jsonl"
    metrics_path = args.output_dir / "metrics.jsonl"
    log(f"[init] {len(prompts)} prompts, {args.prompts_per_step} x {args.num_generations} episodes/step, world={world}")

    refit_sender, refit_meta = None, None
    if REFIT == "nccl":
        if OFFLOAD_MODEL or SLEEP_URLS or not args.full_finetune:
            raise SystemExit("RL_REFIT=nccl needs full fine-tuning with the model resident on the trainer GPUs "
                             "(no RL_OFFLOAD_MODEL / RL_SLEEP_URLS)")
        from nccl_refit import VllmRefitSender, refit_names
        refit_meta = refit_names(model, include_frozen_visual=os.environ.get("RL_REFIT_VISUAL") == "1")
        if is_main:
            refit_sender = VllmRefitSender([u for u in POLICY_URLS if u not in FILE_URLS], log=log)
            refit_sender.connect()
        dist.barrier()
    if not args.no_sync_at_start:
        publish_weights()
        dist.barrier()
        if OFFLOAD_MODEL:
            model.to("cpu")
            torch.cuda.empty_cache()
        if is_main:
            vllm_wake(log)
            sync_vllm(args.live_dir, log)
        if OFFLOAD_MODEL:
            if is_main:
                vllm_sleep(log)
            dist.barrier()
            model.to(device)
    dist.barrier()

    valid_attempts = int((resume_state or {}).get("valid_attempts", 0))
    optimizer_updates = int((resume_state or {}).get("optimizer_updates", args.start_step))
    if args.streaming and is_main:     # every request from here on carries the version the policies now serve
        log(f"[stream] proxies {GATE_URLS} tag requests with version {optimizer_updates}: {gate_open(optimizer_updates, log)}")
    def checkpoint_state(step, rank_rng, curriculum_state=None):
        """curriculum_state: the state from before a lookahead draw made on this checkpoint round (else the current one)."""
        return {"sample_format": "per_request_v1", "global_step": step, "model": args.model, "world_size": world,
                "valid_attempts": valid_attempts, "optimizer_updates": optimizer_updates,
                "optimizer": {"sharded": "fsdp"} if args.full_finetune else opt.state_dict(), "full_finetune": bool(args.full_finetune), "prompts": prompts, "prompt_order": order,
                "prompt_cursor": cursor, "prompt_rng": rng.getstate(), "rank_rng": rank_rng, "args": vars(args),
                **({"segment_curriculum": curriculum_state if curriculum_state is not None else _locked_state(curriculum)}
                   if curriculum is not None else {}),
                **({"cocritic": {"state_dict": {k: v.detach().cpu() for k, v in cocritic.state_dict().items()},
                                 "optimizer": cocritic_opt.state_dict(), "sigma": cocritic_sigma}} if cocritic is not None else {})}

    def cotrain_step(critic_m, critic_opt, xs, hs, ys, rows, advs, results) -> dict:
        """All ranks: K critic updates on this step's boundary states, the raw-advantage std, and (rank 0) the co-trained
        critic's calibration on these fresh rollouts (values from before the update)."""
        def reduce_sum(t):
            dist.all_reduce(t)
        x = torch.cat(xs) if xs else torch.zeros(0, 1, device=device)
        h = torch.cat(hs) if hs else None
        y = torch.tensor(ys, dtype=torch.float32, device=device)
        losses = CC.update(critic_m, critic_opt, x, h, y, steps=args.critic_updates, reduce_sum=reduce_sum)
        st = torch.tensor([float(len(advs)), sum(advs), sum(a * a for a in advs)], dtype=torch.float64, device=device)
        dist.all_reduce(st)
        n, sm, sq = (float(v) for v in st)
        out = {"cocritic/loss_before": losses[0] if losses else None, "cocritic/loss_after_k": losses[-1] if losses else None,
               "cocritic/states": int(n), "cocritic/adv_std": math.sqrt(max(0.0, sq / n - (sm / n) ** 2)) if n else None}
        gathered = [None] * world
        dist.all_gather_object(gathered, rows)
        if is_main:
            from physics_critic import calibration
            by = defaultdict(list)
            for i, chain, k, v in sorted(r for g in gathered for r in g):
                by[i].append((chain, k, v))
            traj = sorted(by)
            vals = [[(0, v) for _, _, v in by[i]] for i in traj]
            cal = calibration(vals, [results[i]["critic_R"] for i in traj])
            out.update({k.replace("critic/", "cocritic/"): v for k, v in cal.items()})
            if any("host_R" in results[i] for i in traj):     # round 2 part c: the same values against plain host success
                calh = calibration(vals, [results[i].get("host_R", results[i]["critic_R"]) for i in traj])
                out.update({k.replace("critic/", "cocritic_host/"): v for k, v in calh.items()})
        return out

    def run_probe(tag: int) -> dict:
        """Rank 0: the fixed probe starts under the current policy (no gradient); per-rollout rows go to probe.jsonl."""
        from reward_segment import probe_stats
        t0 = time.time()
        results = collect_segment_rollouts(probe_entries, f"probe{tag}", args.probe_generations)
        stats = probe_stats(results, probe["sets"])
        with (args.output_dir / "probe.jsonl").open("a") as fh:
            for r in results:
                fh.write(json.dumps({"probe_step": tag, "start_id": r.get("start_id"), "K": r.get("K"), "group": r.get("group"),
                                     "reward": r.get("reward"), "verdict": r.get("verdict"), "components": r.get("components"),
                                     "episode_dir": r.get("episode_dir")}) + "\n")
        log(f"[probe {tag}] {json.dumps({k: round(v, 3) for k, v in stats.items()})} {time.time() - t0:.0f}s")
        return stats

    ckpt_job = {"future": None, "thread": None, "error": None, "step": None, "blocking_s": None, "background_s": None}

    def wait_checkpoint_job() -> None:
        """All ranks: the previous background checkpoint (this rank's DCP upload; on rank 0 also the publish) is complete."""
        if ckpt_job["future"] is not None:
            ckpt_job["future"].result()
            ckpt_job["future"] = None
        if ckpt_job["thread"] is not None:
            ckpt_job["thread"].join()
            ckpt_job["thread"] = None
        if ckpt_job["error"] is not None:
            raise RuntimeError(f"background checkpoint {ckpt_job['step']} failed: {ckpt_job['error']}")

    def write_checkpoint(step, rank_rng, curriculum_state=None) -> None:
        """All ranks: the full fine-tuning state is written collectively before rank 0 publishes the checkpoint. With
        --async-checkpoint the sharded state is staged to host memory here and written (then published by rank 0) in the
        background; the next checkpoint and the end of training wait for it."""
        t0 = time.time()
        wait_checkpoint_job()
        staging = None
        if args.full_finetune:
            staging = (args.full_state_dir / f"checkpoint-{step}") if args.full_state_dir else (args.output_dir / f".fsdp-staging-{step}")
            if is_main:
                shutil.rmtree(staging, ignore_errors=True)
                if args.async_checkpoint:     # the background writer would hit an unwritable disk only at the next checkpoint
                    staging.mkdir(parents=True)
            dist.barrier(group=cpu_group)
            if args.async_checkpoint:
                import torch.distributed.checkpoint as dcp
                from torch.distributed.checkpoint.state_dict import get_state_dict
                msd, osd = get_state_dict(model, opt)
                if os.environ.get("RL_ASYNC_CKPT_PROCESS") == "1":
                    # the thread-based writer (2.45 GB/rank) took 10-15 min under the training loop's GIL and
                    # held the next step's first barrier that long; the process-based checkpointer stages to shared memory
                    # here and writes from its own process (its own process group)
                    from torch.distributed.checkpoint.state_dict_saver import AsyncCheckpointerType
                    # the save's process group must have a CPU backend (async_save asserts on the default group, which is
                    # nccl-only without RL_FSDP_CPU_OFFLOAD; it crashed a checkpoint save): use the gloo ckpt_group as the
                    # thread path does
                    res = dcp.async_save({"model": msd, "optim": osd}, checkpoint_id=str(staging), process_group=ckpt_group,
                                         async_checkpointer_type=AsyncCheckpointerType.PROCESS)
                else:
                    res = dcp.async_save({"model": msd, "optim": osd}, checkpoint_id=str(staging), process_group=ckpt_group)
                ckpt_job["future"] = getattr(res, "upload_completion", res)     # staging (the host copy) is done on return
            else:
                save_full_training_state(model, opt, staging)
            dist.barrier(group=cpu_group)
        if is_main:
            state = checkpoint_state(step, rank_rng, curriculum_state)
            if not args.async_checkpoint:
                save_training_checkpoint(model, args.live_dir, args.output_dir, step, state, log, fsdp_staging=staging)
                if args.full_finetune:
                    prune_full_states(args.output_dir, args.output_dir / f"checkpoint-{step}", log)
            else:
                staged = stage_checkpoint_weights(args.live_dir, args.output_dir, step)    # before the live dir moves on
                upload, ckpt_job["future"] = ckpt_job["future"], None

                def publish(t_start=time.time()):
                    try:
                        upload.result()        # rank 0's upload ends with the coordinator's metadata write after every rank wrote
                        save_training_checkpoint(model, args.live_dir, args.output_dir, step, state, log, fsdp_staging=staging,
                                                 staged=finish_staged_weights(staged, args.output_dir, step))
                        prune_full_states(args.output_dir, args.output_dir / f"checkpoint-{step}", log)
                        ckpt_job["background_s"] = round(time.time() - t_start)
                        log(f"[checkpoint] step {step} background write+publish {ckpt_job['background_s']}s")
                    except Exception as exc:  # noqa: BLE001 - surfaced by wait_checkpoint_job on the next save / at the end
                        ckpt_job["error"] = repr(exc)
                        log(f"[checkpoint] step {step} background write FAILED: {exc!r}")

                import threading
                ckpt_job["thread"] = threading.Thread(target=publish, name=f"checkpoint-{step}", daemon=False)
                ckpt_job["thread"].start()
        ckpt_job["step"] = step
        ckpt_job["blocking_s"] = round(time.time() - t0)

    if resume_state is None:
        initial_rng = [None] * world
        dist.all_gather_object(initial_rng, {"python": random.getstate(), "torch": torch.get_rng_state(),
                                            "cuda": torch.cuda.get_rng_state(device)})
        write_checkpoint(0, initial_rng)
        dist.barrier(group=cpu_group)
        if is_main and probe_entries and os.environ.get("RL_SKIP_PROBE0") != "1":     # smoke runs may skip the step-0 probe
            with metrics_path.open("a") as fh:       # the policy before any update, on the same fixed starts
                fh.write(json.dumps({"step": 0, "probe_only": True, "time": time.time(), **run_probe(0)}) + "\n")

    lookahead_pool = ThreadPoolExecutor(max_workers=1) if (args.async_lookahead and is_main and not args.streaming) else None
    pending = None           # rank 0: (entries, future -> (results, collect seconds), behavior optimizer version)
    stream_pool = None       # rank 0, --streaming: StreamPool (stream_pool.py)
    if args.streaming and (curriculum is None or args.num_generations != 1 or not GATE_URLS):
        raise SystemExit("--streaming needs the segment curriculum, --num-generations 1 and RL_GATE_URLS (the policy proxies)")
    co_pre = None            # rank 0, physics-only co-training: this step's critic states (see critic_cotrain.physics_only_advantages)

    def timed_collect(entries, at_step):
        t0 = time.time()
        return collect_segment_rollouts(entries, at_step, args.num_generations), time.time() - t0

    env_concurrency = 0
    if is_main:
        for url in ENV_SERVERS:            # slot-occupancy metrics: the env servers' episode slots
            try:
                env_concurrency += int(requests.get(f"{url}/health", timeout=30).json().get("max_concurrency") or 0)
            except Exception:  # noqa: BLE001 - metrics only
                pass
    for step in range(args.start_step + 1, args.max_steps + 1):
        t_step = time.time()
        ckpt_curriculum_state = None
        critic_metrics: dict = {}
        async_info = {"behavior_version": optimizer_updates, "t_collect": 0.0, "t_wait": 0.0, "lookahead_launched": False}
        # ---- rollouts (rank 0 drives the env server; results broadcast)
        payload: list = [None]
        segment_stats: dict = {}
        if OFFLOAD_MODEL:
            model.to("cpu")          # one-card RL: the vLLM servers need the memory during rollouts
            torch.cuda.empty_cache()
        if is_main:
            vllm_wake(log)
            if curriculum is not None and args.streaming:
                if stream_pool is None:
                    from stream_pool import StreamPool
                    inflight = args.stream_inflight or env_concurrency or 45

                    def _draw(n):
                        with CURRICULUM_LOCK:
                            return curriculum.next_batch(n)
                    mon_kw = {}
                    if args.heldout_monitor:
                        mspec = json.loads(args.heldout_monitor)
                        # the monitor covers the CURRICULUM suites (incl. libero_10), not the legacy args.suites 3-suite list
                        mstarts = [curriculum.dress(x) for x in monitor_starts(mspec, args.curriculum_suites or args.suites, args.tasks)]
                        mrng = random.Random(f"monitor:{args.seed}:{args.start_step}")
                        mstate = {"order": [], "i": 0}

                        def _mdraw():
                            if mstate["i"] >= len(mstate["order"]):
                                mstate["order"] = list(range(len(mstarts)))
                                mrng.shuffle(mstate["order"])
                                mstate["i"] = 0
                            mstate["i"] += 1
                            return dict(mstarts[mstate["order"][mstate["i"] - 1]])
                        mon_kw = {"monitor_draw": _mdraw, "monitor_every": int(mspec.get("every", 10))}
                        log(f"[monitor] held-out monitor: {len(mstarts)} starts, every {mon_kw['monitor_every']}th admission, no gradient")
                    stream_pool = StreamPool(draw=_draw, run_one=run_segment_one, batch=args.prompts_per_step, inflight=inflight,
                                             draw_chunk=args.prompts_per_step, version=lambda: optimizer_updates,
                                             step=lambda: step, log=log,
                                             **({"group_size": args.stream_group_size} if args.stream_group_size > 1 else {}), **mon_kw)
                    log(f"[stream] pool started: {inflight} episodes in flight, batch {args.prompts_per_step}, max staleness {args.max_staleness}")
                entries, results, sinfo = stream_pool.take(args.prompts_per_step)
                if args.heldout_monitor:
                    mrows = [monitor_record(e, r, step) for e, r in stream_pool.pop_monitor()]
                    if mrows:
                        with open(args.output_dir / "monitor.jsonl", "a") as fh:
                            fh.writelines(json.dumps(x) + "\n" for x in mrows)
                    sinfo = dict(sinfo, **monitor_metrics(mrows))
                if args.stream_group_size > 1:
                    # group-sampling checks: complete groups only, no group split across rounds
                    gk = [r.get("start_group") for r in results]
                    cnt = Counter(gk)
                    seen = GROUP_SEEN.setdefault("keys", set())
                    tdone: dict = {}
                    for r in results:
                        tdone.setdefault(r.get("start_group"), []).append(float((r.get("stream") or {}).get("t_done") or 0.0))
                    waits = [max(v) - min(v) for v in tdone.values() if v]
                    sinfo = dict(sinfo, **{"group/completion_wait_mean_s": round(sum(waits) / max(1, len(waits)), 1),
                                           "group/completion_wait_max_s": round(max(waits, default=0.0), 1)})
                    sinfo = dict(sinfo, **{"group/batch_groups": len(cnt), "group/incomplete": sum(1 for k, c in cnt.items() if k is None or c != args.stream_group_size),
                                           "group/split_across_rounds": sum(1 for k in cnt if k in seen)})
                    seen.update(k for k in cnt if k is not None)
                async_info.update(stream=sinfo, t_wait=sinfo["stream/t_wait_s"], t_collect=0.0)
                with CURRICULUM_LOCK:
                    segment_stats = curriculum.update(entries, results)
                segment_stats["segment/infidelity_rate"] = sum(r.get("verdict") == "replay_infidelity" for r in results) / max(1, len(results))
                from reward_segment import branch_stats, group_signal_stats
                segment_stats.update(group_signal_stats(results))
                segment_stats.update(branch_stats(results))
            elif curriculum is not None:
                if pending is not None:          # collected during the previous learner step, with the weights served then
                    entries, future, async_info["behavior_version"] = pending
                    pending = None
                    t_wait = time.time()
                    results, async_info["t_collect"] = future.result()
                    async_info["t_wait"] = time.time() - t_wait
                else:
                    entries = curriculum.next_batch(args.prompts_per_step)
                    results, async_info["t_collect"] = timed_collect(entries, step)
                segment_stats = curriculum.update(entries, results)
                segment_stats["segment/infidelity_rate"] = sum(r.get("verdict") == "replay_infidelity" for r in results) / max(1, len(results))
                from reward_segment import branch_stats, group_signal_stats
                segment_stats.update(group_signal_stats(results))
                segment_stats.update(branch_stats(results))
            else:
                batch = []
                for _ in range(args.prompts_per_step):
                    batch.append(prompts[order[cursor % len(order)]])
                    cursor += 1
                    if cursor % len(order) == 0:
                        rng.shuffle(order)
                results = collect_rollouts(batch, step, args.num_generations)
            if critic is not None:
                # per-request advantages; the trajectory's "advantage" (records, zero-signal check) is their sum. Co-training:
                # these frozen offline-critic advantages are only logged (critic/* = the offline critic on fresh rollouts) and
                # their raw std seeds the first step's scale; the trained advantages come from the co-trained critic
                req_adv, critic_metrics = batch_advantages(results, critic, lam=args.gae_lambda, normalize=not args.critic_cotrain,
                                                                 false_finish_reward=args.false_finish_reward)
                for r, a in zip(results, req_adv):
                    r["request_advantages"] = a
                    r["advantage"] = float(sum(a.values())) if a else 0.0
                if args.critic_cotrain:
                    flat = [x for a in req_adv if a for x in a.values()]
                    m = sum(flat) / max(1, len(flat))
                    async_info["critic_sigma0"] = math.sqrt(sum((x - m) ** 2 for x in flat) / max(1, len(flat))) or 1.0
                if PHYS_COTRAIN:
                    # a physics-only co-trained critic (critic_hidden none) needs no training
                    # forward, so every live request state of every trajectory is scored here with the co-trained critic as it
                    # stands before this step's update (same weights the in-forward path would use), the GAE runs over ALL
                    # requests, and only the subsampled requests are trained (inclusion-weighted, max_training_requests).
                    sigma = cocritic_sigma if cocritic_sigma is not None else async_info["critic_sigma0"]
                    group_on = args.stream_group_size > 1 and args.group_advantage == "group"
                    keyed, co_pre = CC.physics_only_advantages(cocritic, critic, results, lam=args.gae_lambda, sigma=sigma, device=device,
                                                              false_finish_reward=args.false_finish_reward, costs=RETURN_COSTS,
                                                              **({"group_keys": [r.get("start_group") for r in results]} if group_on else {}))
                    gs = co_pre.get("group_stats")
                    if gs:      # group sampling: groups used, flat groups (no outcome signal), fallbacks
                        critic_metrics.update({"group/groups": gs["groups"], "group/flat_fraction": gs["flat_groups"] / max(1, gs["groups"]),
                                               "group/fallback_episodes": gs["fallback_episodes"],
                                               "group/mean_abs_outcome": gs["outcome_abs"] / max(1, gs["episodes"]),
                                               "group/outcome_mean_absmax": gs.get("outcome_mean_absmax"),
                                               "group/scored_group_sizes": gs.get("group_sizes")})
                    cs = co_pre.get("cost_stats") or []
                    if cs:      # reward-cost monitoring: actions, bad actions, perception calls, costs, missed finish, exhaustion
                        n = len(cs)
                        critic_metrics.update({"cost/phys_per_ep": sum(c["phys"] for c in cs) / n, "cost/bad_per_ep": sum(c["bad"] for c in cs) / n,
                                               "cost/perc_per_ep": sum(c["perc"] for c in cs) / n,
                                               "cost/phys_per_request": sum(c["phys"] for c in cs) / max(1, sum(c["requests"] for c in cs)),
                                               "cost/mean_cost": sum(c["cost"] for c in cs) / n, "cost/mean_return": sum(c["R"] for c in cs) / n,
                                               "cost/missed_finish_rate": sum(c["missed_finish"] for c in cs) / n,
                                               "cost/exhausted_rate": sum(c["exhausted"] for c in cs) / n})
                        if "post_req" in cs[0]:     # missed-finish round 2 monitoring: requests after the host's first success
                            hs = [c for c in cs if c["host_success"]]
                            ms = [c for c in cs if c["missed_finish"]]
                            critic_metrics.update({"cost/post_success_req_per_ep": sum(c["post_req"] for c in cs) / n,
                                                   "cost/post_success_req_per_success": sum(c["post_req"] for c in hs) / max(1, len(hs)),
                                                   "cost/post_success_req_per_missed": sum(c["post_req"] for c in ms) / max(1, len(ms)),
                                                   "cost/host_success_rate": len(hs) / n})
                    ss = co_pre.get("staged_stats") or []
                    if ss:      # staged subgoal reward monitoring (laddered tasks only)
                        fails = [c for c in ss if not c["host_success"]]
                        critic_metrics.update({"staged/episodes": len(ss), "staged/host_success": sum(c["host_success"] for c in ss),
                                               "staged/false_finish": sum(c["false_finish"] for c in ss),
                                               "staged/credited": sum(c["credited"] for c in ss),
                                               "staged/credited_frac_of_fail": sum(c["credited"] for c in ss) / max(1, len(fails)),
                                               "staged/mean_depth_fail": sum(c["depth"] for c in fails) / max(1, len(fails)),
                                               "staged/mean_return": sum(c["R"] for c in ss) / len(ss),
                                               # per task: [episodes, host successes, credited, episodes reaching depth >= 1, 2, ...]
                                               "staged/by_task": {t: [sum(1 for c in ss if c["task"] == t),
                                                                      sum(c["host_success"] for c in ss if c["task"] == t),
                                                                      sum(c["credited"] for c in ss if c["task"] == t)]
                                                                     + [sum(1 for c in ss if c["task"] == t and c["depth"] >= k)
                                                                        for k in range(1, 1 + max(c["n"] or 0 for c in ss if c["task"] == t))]
                                                                  for t in sorted({c["task"] for c in ss})}})
                    for r, a in zip(results, keyed):
                        r["request_advantages"] = a
                        r["advantage"] = float(sum(a.values())) if a else 0.0
                elif args.critic_cotrain:
                    for r in results:
                        ok = r.get("sample_path") and r.get("verdict") not in ("infrastructure_error", "replay_infidelity") and r.get("episode_dir")
                        r["critic_phys"] = CC.physics_inputs(critic, r["episode_dir"], f'{r["suite"]}:{int(r["task_id"])}') if ok else None
                        r["critic_R"] = float(bool((r.get("diagnostics") or {}).get("host_final_success")))
            else:
                adv = group_advantages(results, std_floor=args.adv_std_floor)
                if args.advantage_shuffle:
                    adv = shuffle_within_groups(results, adv, seed=args.seed * 100003 + step)
                for r, a in zip(results, adv):
                    r["advantage"] = a
            # one-batch lookahead: the next batch is chosen after this batch's curriculum commit (same order as synchronous),
            # then collected with the currently served weights while this batch trains; checkpoint rounds drain the pipeline
            # unless --lookahead-through-checkpoints (the checkpoint then keeps the curriculum state from before this draw)
            ckpt_round = step % args.save_steps == 0
            if lookahead_pool is not None and step < args.max_steps and (args.lookahead_through_checkpoints or not ckpt_round):
                upcoming, ckpt_curriculum_state = lookahead_draw(curriculum, args.prompts_per_step, snapshot=ckpt_round)
                pending = (upcoming, lookahead_pool.submit(timed_collect, upcoming, step + 1), optimizer_updates)
                async_info["lookahead_launched"] = True
            payload = [(results, async_info)]
        broadcast_payload(payload, cpu_group, device)
        results, async_info = payload[0]
        staleness = optimizer_updates - int(async_info["behavior_version"])
        if not args.streaming and (staleness not in (0, 1) or (staleness and not args.async_lookahead)):
            raise InvalidRollout(f"batch behavior version {async_info['behavior_version']} vs learner {optimizer_updates}: "
                                 "unsupported staleness")
        t_env = time.time() - t_step
        infrastructure_rate = sum(r.get("verdict") == "infrastructure_error" for r in results) / max(1, len(results))
        infidelity_rate = sum(r.get("verdict") == "replay_infidelity" for r in results) / max(1, len(results))
        # r5 (RL_INFRA_WARN_ONLY=1): a batch over the infrastructure / infidelity limits is trained on its valid episodes only
        # (infra-error and diverged-replay episodes carry no sample and are excluded below anyway) and logged loudly
        warn_only = os.environ.get("RL_INFRA_WARN_ONLY") == "1"
        if infidelity_rate > args.max_infidelity_rate:
            log(f"[invalid_batch] replay infidelity rate {infidelity_rate:.3f} exceeds {args.max_infidelity_rate:.3f}"
                + (" -- WARNING ONLY: training on the valid episodes" if warn_only else ""))
            if not warn_only:
                raise InvalidRollout("segment replays diverge from their sources; optimizer update refused")
        if infrastructure_rate > args.max_infrastructure_rate:
            log(f"[invalid_batch] infrastructure rate {infrastructure_rate:.3f} exceeds {args.max_infrastructure_rate:.3f}"
                + (" -- WARNING ONLY: training on the valid episodes" if warn_only else ""))
            if not warn_only:
                raise InvalidRollout("rollout infrastructure is unhealthy; optimizer update refused")
        if any(not math.isfinite(float(r.get("advantage", 0))) for r in results):
            raise InvalidRollout("non-finite group advantage")
        valid_now = sum(1 for r in results if r.get("verdict") not in ("infrastructure_error", "replay_infidelity"))
        valid_attempts += valid_now
        code_on = args.code_reward is not None
        if code_on and any((r.get("diagnostics") or {}).get("code_reward_version") != args.code_reward for r in results
                           if r.get("sample_path") and r.get("verdict") not in ("infrastructure_error", "replay_infidelity")):
            raise InvalidRollout("code reward is on but a trained rollout lacks the matching code_reward_version")
        code_signal = code_on and any(float((r.get("diagnostics") or {}).get("code_selected_cost") or 0.0) > 0 for r in results
                                      if r.get("sample_path") and r.get("verdict") not in ("infrastructure_error", "replay_infidelity"))
        zero_signal = (bool(args.skip_zero_signal) and all(float(r.get("advantage", 0.0)) == 0.0 for r in results)
                       and all(not any((r.get("request_advantages") or {}).values()) for r in results) and not code_signal)
        if is_main:
            vllm_sleep(log)
        dist.barrier()
        if OFFLOAD_MODEL:
            model.to(device)
        # ---- samples for this rank
        active = [i for i, r in enumerate(results) if r.get("sample_path") and r.get("verdict") not in ("infrastructure_error", "replay_infidelity")]
        def learner_cost(r):                 # packed: the env server's post-packing estimate; else the trained-request count
            d = r.get("diagnostics") or {}
            if args.pack_prefixes and isinstance(d.get("packed_cost"), (int, float)):
                return max(1, int(d["packed_cost"]))
            if args.prefix_tree and isinstance(d.get("packed_lengths"), list) and d["packed_lengths"]:
                return tree_cost(d["packed_lengths"])
            return max(1, int(d.get("n_training_requests") or 1))
        costs = [learner_cost(results[i]) for i in active]
        t_build = time.time()
        # pack-level assignment needs the env server's per-chain lengths (prefix_chains.packed_cost "packed_lengths")
        pack_level = (args.pack_assignment == "pack" and bool(active)
                      and all(isinstance((results[i].get("diagnostics") or {}).get("packed_lengths"), list)
                              and (results[i].get("diagnostics") or {}).get("packed_lengths") for i in active))
        my_chains: dict[int, set[int]] | None = None
        owner: dict[int, int] = {}
        if zero_signal:
            mine = []
        elif pack_level:
            plans, owner = pack_plan(active, {i: results[i]["diagnostics"]["packed_lengths"] for i in active}, world)
            my_chains = plans[rank]
            mine = sorted(my_chains)
        else:
            mine = balanced_assignment(active, costs, world)[rank]
        samples = []
        n_err = 0
        n_segments = 0
        build_stats: dict = {}
        objective_tokens_local = 0
        build_errors = []
        max_drops = int(getattr(args, "max_overlength_drops", 0) or 0)   # over-length guard (absent = 0 = off, e.g. test harness args)
        guard = max_drops > 0       # over-length guard: per-trajectory contributions, undone if the trajectory is dropped
        traj_tokens: dict[int, int] = {}
        traj_segments: dict[int, int] = {}
        traj_stats: dict[int, dict] = {}
        for i in mine:
            stats_before = dict(build_stats) if guard else None
            try:
                raw = read_rollout_json(results[i]["sample_path"])
                built, full_tokens = build_request_samples(raw, processor, image_pad_id=image_pad_id,
                    hard_max_tokens=min(args.max_seq_tokens, args.hard_max_tokens), pack_prefixes=args.pack_prefixes,
                    chains=None if my_chains is None else my_chains[i])
                if my_chains is not None:
                    got = {b["chain"] for b in built}
                    if got != my_chains[i] or len(results[i]["diagnostics"]["packed_lengths"]) <= max(got):
                        raise InvalidRollout(f"trainer chains {sorted(got)} differ from the env server's packing "
                                             f"({len(results[i]['diagnostics']['packed_lengths'])} chains)")
                if args.advantage_mode == "physcritic" and args.critic_cotrain and not PHYS_COTRAIN \
                        and int(raw["selected_request_count"]) != int(raw["eligible_request_count"]):
                    raise InvalidRollout("critic co-training needs every eligible request trained (max_training_requests >= max_requests)")
                if my_chains is None or owner[i] == rank:      # once per trajectory (its chains may sit on several ranks)
                    objective_tokens_local += full_tokens
                    if guard:
                        traj_tokens[i] = full_tokens
            except Exception as exc:
                build_errors.append({"rank": rank, "path": results[i]["sample_path"], "error": repr(exc),
                                     **({"index": i, "overlength": isinstance(exc, OverLengthRollout)} if guard else {})})
                continue
            if args.advantage_mode == "physcritic":     # critic advantages (keyed by response id) on this rollout's requests
                values, unmapped = proxy_values(results[i].get("request_advantages"), raw)
                build_stats["critic_unmapped"] = build_stats.get("critic_unmapped", 0) + unmapped
                if getattr(args, "streaming", False):
                    # per-request behaviour version (proxy gate tags): SAO's ratio mask + score centering use each request's
                    # own sampler log-probs; requests older than --max-staleness versions lose their policy-gradient credit
                    counted = my_chains is None or owner[i] == rank
                    for q in raw["training_requests"]:
                        v = q.get("policy_version")
                        lag = optimizer_updates - int(v) if type(v) is int else None
                        key = "stale_unknown" if lag is None else ("stale_drop" if lag > args.max_staleness else f"stale_{min(lag, 2)}")
                        if counted:
                            build_stats[key] = build_stats.get(key, 0) + 1
                        if (lag is None or lag > args.max_staleness) and values is not None:
                            values[int(q["request_index"])] = 0.0
            for s in built:
                if args.advantage_mode == "physcritic":
                    s["advantage"] = 1.0
                    if args.critic_cotrain and not PHYS_COTRAIN:     # advantages are computed in the training forward
                        s["critic_members"] = CC.chain_members(s, raw, results[i].get("critic_phys"))
                    else:
                        s["token_advantage"] = request_token_values(s, raw, values)
                else:
                    s["advantage"] = float(results[i]["advantage"])
                s["index"] = i
                s["trajectory_tokens"] = int(full_tokens)
                samples.append(s)
                n_segments += 1
            if guard:
                traj_segments[i] = len(built)
                traj_stats[i] = {k: v - stats_before.get(k, 0) for k, v in build_stats.items() if v != stats_before.get(k, 0)}
        all_errors = [None] * world
        dist.all_gather_object(all_errors, build_errors)
        errors = [error for group in all_errors for error in group]
        overlength_dropped: set[int] = set()
        if errors:
            # over-length guard: every rank sees the same gathered errors, so every rank computes the same drop set (or every rank raises)
            drop = overlength_drop_set(errors, max_drops) if guard else None
            if drop is None:
                log(f"[invalid_batch] {json.dumps(errors)}")
                raise InvalidRollout("selected raw sample is invalid; refusing a biased partial update")
            overlength_dropped = drop
            if is_main:
                log(f"[overlength_drop] round {step}: dropping {len(drop)} trajectories (max {max_drops}) "
                    f"{json.dumps([{'index': i, 'path': results[i]['sample_path'], 'sid': results[i].get('sid')} for i in sorted(drop)])} "
                    f"errors {json.dumps(errors)}")
            samples = [s for s in samples if s["index"] not in drop]
            for i in drop:
                objective_tokens_local -= traj_tokens.pop(i, 0)
                n_segments -= traj_segments.pop(i, 0)
                for k, v in traj_stats.pop(i, {}).items():
                    build_stats[k] -= v
            mine = [i for i in mine if i not in drop]
            active = [i for i in active if i not in drop]
        t_build = time.time() - t_build
        try:                                   # the slowest rank's sample build (block-level tests run without the helper)
            t_build_max = _rank_max(t_build, device)
        except NameError:
            t_build_max = t_build
        n_tokens_local = torch.tensor([sum(s["n_tokens"] for s in samples)], dtype=torch.float64, device=device)
        dist.all_reduce(n_tokens_local)
        n_tokens_global = max(1.0, float(n_tokens_local.item()))
        objective_tokens = torch.tensor([objective_tokens_local], dtype=torch.float64, device=device)
        dist.all_reduce(objective_tokens)
        full_completion_tokens = int(objective_tokens.item())
        if full_completion_tokens <= 0 and not zero_signal:
            raise InvalidRollout("no eligible original completions; no optimizer step performed")
        n_mine = len(mine) if my_chains is None else sum(1 for i in mine if owner[i] == rank)
        n_traj_t = torch.tensor([0 if zero_signal else n_mine], dtype=torch.float64, device=device)
        dist.all_reduce(n_traj_t)
        n_trajectories = int(n_traj_t.item())
        stat_keys = ("repairs", "history_rerendered", "dedup_identity", "history_mismatch", "history_unmapped", "critic_unmapped",
                     "stale_0", "stale_1", "stale_2", "stale_drop", "stale_unknown")
        stat_t = torch.tensor([n_segments] + [build_stats.get(k, 0) for k in stat_keys], dtype=torch.float64, device=device)
        dist.all_reduce(stat_t)
        n_segments_global = int(stat_t[0].item())
        build_global = {k: int(v) for k, v in zip(stat_keys, stat_t[1:].tolist())}
        # ---- optimisation
        if args.warmup_attempts > 0:
            lr = args.lr * min(1.0, valid_attempts / args.warmup_attempts)
        else:
            lr = args.lr * min(1.0, step / max(1, args.warmup_steps))
        for g in opt.param_groups:
            g["lr"] = lr
        # gradient accumulation (--grad-accum-rounds K, rl/grad_accum.py): gradients are zeroed at the start of a K-round cycle
        # only; this round's normaliser (the same quantity its losses are divided by) rescales the gradients held so far; a
        # round without signal is undone afterwards (stash/restore); K = 1 is exactly the old per-round step
        accum_first, accum_last = accum.is_first(step), accum.is_last(step)
        if accum_first:
            opt.zero_grad(set_to_none=True)
            accum.reset()
        accum_norm = 0 if zero_signal else (n_trajectories if args.loss_normalization == "trajectory_mean" else full_completion_tokens)
        accum_stash = accum.stash(params) if (zero_signal and not accum_first) else None
        accum.before_round(params, accum_norm)
        model.train()
        loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        kl_sum = torch.zeros((), device=device, dtype=torch.float32)
        sc_stats = torch.zeros(3, device=device, dtype=torch.float64)   # |log p_train - log p_sampler|, |correction|, tokens
        sc_missing = 0
        is_rows: list = []
        code_rows: list = []
        mask_rows: list = []
        kl_rows: list = []
        t_train = time.time()
        max_len = max([s["length"] for s in samples], default=0)
        if args.pack_prefixes:
            # FSDP passes run in lockstep (per-layer collectives): pass i costs the slowest rank's i-th sequence, so every rank
            # goes longest first; the objective is a sum over samples and does not depend on the order
            samples.sort(key=lambda x: (-x["length"], x.get("index", 0)))
        if args.critic_cotrain and not PHYS_COTRAIN:
            samples.sort(key=CC.sample_order_key)      # each trajectory's chains newest first (the GAE needs later values)
            credit = CC.CreditState({i: results[i]["critic_R"] for i in {s["index"] for s in samples}})
            if cocritic_sigma is None:
                cocritic_sigma = float(async_info.get("critic_sigma0") or 1.0)
            co_buf = CC.new_buffers()
        schedule = [(s, 1.0) for s in samples]
        if args.full_finetune and not zero_signal:
            # FSDP all-gathers per layer: every rank must run the same number of forward/backward passes
            counts = torch.tensor([len(samples)], device=device)
            dist.all_reduce(counts, op=dist.ReduceOp.MIN)
            if int(counts.item()) == 0:
                raise InvalidRollout("a rank has no trainable request samples; refusing an FSDP step with unequal passes")
            most = torch.tensor([len(samples)], device=device)
            dist.all_reduce(most, op=dist.ReduceOp.MAX)
            filler = filler_sample(min(samples, key=lambda x: x["length"]), image_pad_id) if args.pack_prefixes else None
            schedule += [(filler or samples[i % len(samples)], 0.0) for i in range(int(most.item()) - len(samples))]   # zero-weight fillers
            # the LM head is its own FSDP unit and is applied once per LM_HEAD_CHUNK targets: a longer sample would make this
            # rank's collectives diverge from the others' (a silent hang); refuse instead
            longest = torch.tensor([max(int(x["mask"][1:].sum()) for x, _ in schedule)], device=device)
            dist.all_reduce(longest, op=dist.ReduceOp.MAX)          # every rank refuses together (no rank left in a collective)
            if int(longest.item()) > LM_HEAD_CHUNK:
                raise InvalidRollout(f"a sample has {int(longest.item())} target positions > {LM_HEAD_CHUNK}: unequal FSDP LM-head calls")
        prof_rows: list = []
        sac_len = None
        if args.keep_attn_act_gib > 0 and args.full_finetune:
            selective_recompute(model, 1, 0.0)                     # default: every layer checkpointed (zero-signal rounds stay so)
        if args.keep_attn_act_gib > 0 and args.full_finetune and not zero_signal:
            sac_t = torch.tensor([int(x["length"]) for x, _ in schedule], device=device, dtype=torch.long)
            dist.all_reduce(sac_t, op=dist.ReduceOp.MAX)          # every rank runs len(schedule) passes (fillers included)
            sac_len = sac_t.tolist()
        # speed-up 2: the passes that run the reference forward (a collective under FSDP: the same pass indices on every rank)
        ref_every = max(1, int(getattr(args, "ref_every", 1) or 1))
        ref_offset = random.Random(f"ref-every:{args.seed}:{step}").randrange(ref_every) if ref_every > 1 else 0
        n_ref_passes = 0
        trace_set = {tuple(int(v) for v in x.split(":")) for x in os.environ.get("RL_PROFILE_TRACE", "").split(",") if ":" in x}
        tree_on = bool(args.prefix_tree and not zero_signal)
        tree_info = {}
        if tree_on:
            tree_info = run_tree_round(samples, args=args, core=core, lm_head=lm_head, model=model, tree_ref=tree_ref, device=device,
                                       image_pad_id=image_pad_id, n_trajectories=n_trajectories,
                                       full_completion_tokens=full_completion_tokens, world=world, ratio_band=ratio_band,
                                       code_on=code_on, acc={"loss_sum": loss_sum, "kl_sum": kl_sum, "sc_stats": sc_stats},
                                       rows={"is": is_rows, "code": code_rows, "mask": mask_rows, "kl": kl_rows}, step=step, rank=rank,
                                       act_offload=(OFFLOAD_BASE, ACT_OFFLOAD_MIN), ref_every=ref_every, ref_offset=ref_offset)
            n_ref_passes += tree_info["ref_episodes"]
            sc_missing += tree_info["sc_missing"]
            log(f"[step {step}] prefix tree: {json.dumps({k: v for k, v in tree_info.items() if k != 'sc_missing'})}")
        for si, (s, scale) in enumerate([] if tree_on else schedule):
            if PROFILE["on"]:
                torch.cuda.synchronize()
                PROFILE["cur"] = {"step": step, "pass": si, "len": int(s["length"]), "targets": int(s["mask"].sum()), "scale": scale,
                                  "t0": time.time()}
            tracer = None
            if PROFILE["on"] and (step, si) in trace_set and str(rank) in os.environ.get("RL_PROFILE_TRACE_RANKS", "0").split(","):
                from torch.profiler import ProfilerActivity, profile as _tprofile
                tracer = _tprofile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
                tracer.__enter__()
            inputs = {"input_ids": s["input_ids"].unsqueeze(0).to(device), "attention_mask": torch.ones(1, s["length"], dtype=torch.long, device=device)}
            if "pixel_values" in s:
                inputs["pixel_values"] = s["pixel_values"].to(device, dtype=torch.bfloat16)
                inputs["image_grid_thw"] = s["image_grid_thw"].to(device)
                inputs["mm_token_type_ids"] = (inputs["input_ids"] == image_pad_id).long()  # transformers 5: M-RoPE needs 0 text / 1 image
            mask = s["mask"].unsqueeze(0).to(device)
            # DDP-free manual gradient sync: accumulate locally, all-reduce grads once below
            denominator = (s["trajectory_tokens"] * n_trajectories) if args.loss_normalization == "trajectory_mean" else full_completion_tokens
            core_s = core
            if ACT_OFFLOAD_MIN is not None:
                # save_on_cpu only for long sequences (plain recompute checkpointing below the threshold)
                OFFLOAD_ACTIVATIONS["on"] = OFFLOAD_BASE and s["length"] >= ACT_OFFLOAD_MIN
            if args.critic_cotrain and not PHYS_COTRAIN:
                # the boundary hidden states of this chain's requests from the training forward itself (detached), the
                # co-trained critic's values, and the chain's GAE advantages given the later chains' values
                core_s = CC.CachedCore(core)
                # the cached training forward runs here, outside sample_loss: keep its saved tensors in the same
                # activation-offload context (otherwise --offload-activations has no effect under critic co-training)
                import contextlib
                with offload_ctx():
                    out = core_s(**inputs)
                # FIX: zero-weight filler passes take the same path. Before, a filler ran the KL
                # reference forward BEFORE the policy forward while a real sample ran it after, so in a pass where one rank
                # trained a real chain and another a filler, the FSDP all-gathers of the policy and of the reference model
                # were paired across ranks (same shapes, no error): the gathered weights mixed policy and reference shards
                # (identical only before the first update)
                if scale:
                    s = dict(s, token_advantage=CC.score_chain(cocritic, out.last_hidden_state[0], s["critic_members"], s["index"],
                                                               s.get("chain", 0), results[s["index"]]["critic_R"], credit,
                                                               args.gae_lambda, cocritic_sigma, co_buf))
            if sac_len is not None:
                # the same layers on every rank (FSDP lockstep): k from the longest sequence of this pass index over the ranks
                n_keep = selective_recompute(model, int(sac_len[si]), args.keep_attn_act_gib)
                if PROFILE["on"] and PROFILE.get("cur") is not None:
                    PROFILE["cur"]["kept_attn_layers"] = n_keep
            ref_w = 1.0 if ref_every == 1 else (float(ref_every) if (si - ref_offset) % ref_every == 0 else 0.0)
            n_ref_passes += int(ref_w != 0.0)
            loss, kl_part, sc_row = sample_loss(model, core_s, lm_head, inputs, mask, s, score_centering=args.score_centering,
                                                kl_beta=args.kl_beta, full_completion_tokens=denominator, world_size=world,
                                                reference=reference, is_cap=args.is_cap if scale else None,
                                                is_stats=is_rows if scale else None,
                                                code_weight=args.code_reward_weight if code_on else 0.0,
                                                code_norm=float(args.code_reward_budget * max(1, n_trajectories)),
                                                code_stats=code_rows if scale else None,
                                                # FIX: a zero-weight filler carries no sampler
                                                # log-probabilities; with SAO ratio masking on, sample_loss raised
                                                # InvalidRollout on that rank only and the other ranks hung in FSDP
                                                ratio_band=ratio_band if scale else None,
                                                mask_stats=mask_rows if scale else None,
                                                kl_stats=kl_rows if scale else None, ref_weight=ref_w)
            if scale:
                kl_sum += kl_part
                if sc_row is not None:
                    sc_stats += sc_row
                elif args.score_centering:
                    sc_missing += 1
            else:
                loss = loss * 0.0
            with _PTime("t_backward"):
                loss.backward()
            loss_sum += loss.detach()
            del inputs, loss
            with _PTime("t_empty_cache"):
                # RL_EMPTY_CACHE=1 (default, = r4): release the allocator cache after every pass (~3-8 s/pass measured);
                # 0: never; auto: only when the device has less than RL_MIN_FREE_GIB (3) free, so nvidia-smi headroom stays
                ec = os.environ.get("RL_EMPTY_CACHE", "1")
                if ec == "1" or (ec == "auto" and torch.cuda.mem_get_info()[0] < float(os.environ.get("RL_MIN_FREE_GIB", "3")) * 2**30):
                    torch.cuda.empty_cache()
            if PROFILE["on"]:
                torch.cuda.synchronize()
                row = PROFILE["cur"]
                row["offload"] = bool(OFFLOAD_ACTIVATIONS["on"])
                row["t_pass"] = time.time() - row.pop("t0")
                row["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                row["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
                row["free_gib"] = round(torch.cuda.mem_get_info()[0] / 2**30, 2)
                # allocator churn per pass: retries = cache flushed + cudaMalloc retried after hitting the cap/OOM;
                # device_alloc/free = cudaMalloc/cudaFree calls (thrash under RL_MEM_FRACTION shows up here)
                ms = torch.cuda.memory_stats()
                cnt = {"alloc_retries": ms.get("num_alloc_retries", 0), "device_alloc": ms.get("num_device_alloc", 0),
                       "device_free": ms.get("num_device_free", 0), "sync_all_streams": ms.get("num_sync_all_streams", 0)}
                prev = PROFILE.get("ms") or {}
                row.update({k: v - prev.get(k, 0) for k, v in cnt.items()})
                PROFILE["ms"] = cnt
                torch.cuda.reset_peak_memory_stats()
                pdir = args.output_dir / "profile"
                pdir.mkdir(parents=True, exist_ok=True)
                with open(pdir / f"rank{rank}.jsonl", "a") as fh:      # per pass: survives a crash mid-step
                    fh.write(json.dumps({"rank": rank, **row}) + "\n")
                PROFILE["cur"] = None
            if tracer is not None:
                tracer.__exit__(None, None, None)
                pdir = args.output_dir / "profile"
                pdir.mkdir(parents=True, exist_ok=True)
                (pdir / f"trace_s{step}_p{si}_rank{rank}.txt").write_text(
                    tracer.key_averages().table(sort_by="cuda_time_total", row_limit=60, max_name_column_width=90))
                tracer.export_chrome_trace(str(pdir / f"trace_s{step}_p{si}_rank{rank}.json"))
        # all-reduce gradients (average) and step
        if args.score_centering:
            missing = torch.tensor([sc_missing], device=device, dtype=torch.float64)
            dist.all_reduce(missing)
            if int(missing.item()):
                raise InvalidRollout(f"score centering is on but {int(missing.item())} trained samples lack the sampler's top-k; "
                                     "refusing a partly uncentered update")
        if not args.full_finetune:            # FSDP already reduced (averaged) the gradients in every backward
            for p in params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        if zero_signal:
            if accum_stash is not None:
                accum.restore(params, accum_stash)       # earlier rounds of this cycle keep their gradients
            else:
                opt.zero_grad(set_to_none=True)
            accum_stash = None
        stepped = False
        grad_norm = torch.zeros((), device=device)
        if accum_last and accum.finish(params):
            t_opt = time.time()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm, error_if_nonfinite=True)
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor()
            opt.step()
            opt.zero_grad(set_to_none=True)
            optimizer_updates += 1
            stepped = True
            if PROFILE["on"]:
                torch.cuda.synchronize()
                prof_rows.append({"step": step, "pass": -1, "t_optimizer": time.time() - t_opt})
        if PROFILE["on"]:
            pdir = args.output_dir / "profile"
            pdir.mkdir(parents=True, exist_ok=True)
            with open(pdir / f"rank{rank}.jsonl", "a") as fh:
                for row in prof_rows:
                    fh.write(json.dumps({"rank": rank, **row}) + "\n")
        if PHYS_COTRAIN:
            # critic update on every live state of the batch (rank 0 holds them; the other ranks join the reductions empty)
            mine = co_pre if is_main else CC.new_buffers()
            critic_metrics.update(cotrain_step(cocritic, cocritic_opt, mine["x"], mine["h"], mine["y"], mine["rows"], mine["adv"], results))
            critic_metrics["cocritic/precomputed"] = True
            cocritic_sigma = CC.ema_std(cocritic_sigma, critic_metrics["cocritic/adv_std"]) if critic_metrics.get("cocritic/adv_std") else cocritic_sigma
            critic_metrics["cocritic/sigma"] = cocritic_sigma
            if is_main and args.cocritic_live:
                export_cocritic(cocritic, args.cocritic_live, step)
        elif args.critic_cotrain:
            critic_metrics.update(cotrain_step(cocritic, cocritic_opt, co_buf["x"], co_buf["h"], co_buf["y"], co_buf["rows"],
                                               co_buf["adv"], results))
            cocritic_sigma = CC.ema_std(cocritic_sigma, critic_metrics["cocritic/adv_std"]) if critic_metrics.get("cocritic/adv_std") else cocritic_sigma
            critic_metrics["cocritic/sigma"] = cocritic_sigma
            if is_main and args.cocritic_live:
                export_cocritic(cocritic, args.cocritic_live, step)
        dist.all_reduce(loss_sum, op=dist.ReduceOp.AVG)
        dist.all_reduce(kl_sum, op=dist.ReduceOp.SUM)
        kl_per_token = float(kl_sum.item()) / max(1, full_completion_tokens)
        dist.all_reduce(sc_stats)
        sc_missing_t = torch.tensor([sc_missing], device=device, dtype=torch.float64)
        dist.all_reduce(sc_missing_t)
        sc_metrics = ({"tim/abs_logp_gap": float(sc_stats[0] / max(1.0, float(sc_stats[2]))),
                       "sc/abs_correction": float(sc_stats[1] / max(1.0, float(sc_stats[2]))),
                       "sc/tokens": int(sc_stats[2]), "sc/missing_samples": int(sc_missing_t.item())} if args.score_centering else {})
        is_t = torch.stack(is_rows).sum(0) if is_rows else torch.zeros(4, device=device, dtype=torch.float64)
        dist.all_reduce(is_t)
        is_metrics = ({"is/mean_ratio": float(is_t[0] / max(1.0, float(is_t[3]))), "is/clip_frac": float(is_t[1] / max(1.0, float(is_t[3]))),
                       "is/ess_frac": float(is_t[0] ** 2 / max(1e-12, float(is_t[2]) * float(is_t[3]))), "is/tokens": int(is_t[3])}
                      if args.is_cap is not None else {})
        mask_t = torch.stack(mask_rows).sum(0) if mask_rows else torch.zeros(4, device=device, dtype=torch.float64)
        dist.all_reduce(mask_t)
        kl_t = torch.stack(kl_rows).sum(0) if kl_rows else torch.zeros(2, device=device, dtype=torch.float64)
        dist.all_reduce(kl_t)
        kl_metrics = {"kl/pre_mask": float(kl_t[0]) / max(1, full_completion_tokens), "kl/post_mask": float(kl_t[1]) / max(1, full_completion_tokens),
                      "kl/on_kept_only": int(KL_ON_KEPT), **({"kl/ref_every": ref_every, "kl/ref_passes": n_ref_passes,
                                                               "kl/passes": (tree_info["episodes"] + tree_info["padded_episodes"]) if tree_on else len(schedule)}
                                                            if ref_every > 1 else {}),
                      **({f"tree/{k}": v for k, v in tree_info.items() if k != "sc_missing"} if tree_on else {})}
        mask_metrics = ({"mask/kept_frac": float(mask_t[0] / max(1.0, float(mask_t[3]))),
                         "mask/masked_out_frac": float((mask_t[1] + mask_t[2]) / max(1.0, float(mask_t[3]))),
                         "mask/low_frac": float(mask_t[1] / max(1.0, float(mask_t[3]))),
                         "mask/high_frac": float(mask_t[2] / max(1.0, float(mask_t[3]))), "mask/tokens": int(mask_t[3])}
                        if ratio_band is not None else {})
        code_t = torch.stack(code_rows).sum(0) if code_rows else torch.zeros(3, device=device, dtype=torch.float64)
        dist.all_reduce(code_t)
        code_metrics = ({"code/weighted_cost_mass": float(code_t[0]), "code/trained_error_samples": int(code_t[1]),
                         "code/pos_credit_withheld_samples": int(code_t[2]),
                         "code/selected_cost_sum": sum(float((r.get("diagnostics") or {}).get("code_selected_cost") or 0.0) for r in results),
                         "code/selected_errors": sum(int((r.get("diagnostics") or {}).get("code_selected_errors") or 0) for r in results),
                         "code/selected_delta_requests": sum(int((r.get("diagnostics") or {}).get("code_selected_delta") or 0) for r in results),
                         **{f"iface/{k}": sum(int(((r.get("diagnostics") or {}).get("interface") or {}).get(k) or 0) for r in results)
                            for k in ("attempts", "contract_rejections", "requests_with_rejection", "recovered_within_response",
                                      "unattributed", "delta_cost_requests")},
                         "code/signal_only_update": bool(code_signal and all(float(r.get("advantage", 0.0)) == 0.0 for r in results)),
                         **{f"code/cat_{k}": sum(int(((r.get("diagnostics") or {}).get("code_categories") or {}).get(k, 0)) for r in results)
                            for k in ("ok", "syntax_invalid", "runtime_invalid", "mixed_effect_error", "unknown", "environment", "no_cell")}}
                        if code_on else {})
        t_train = time.time() - t_train
        # ---- export the new version while the lookahead batch still runs on the served one (the endpoints hold their
        # weights in GPU memory; the live dir is only read by the reload below)
        t_export = time.time()
        exported = False
        file_alive = [bool(FILE_URLS) and not FILE_GROUP["dead"]]
        if FILE_URLS:
            dist.broadcast_object_list(file_alive, src=0, group=cpu_group)
        need_live_dir = REFIT != "nccl" or step % args.save_steps == 0 or file_alive[0]   # NCCL: disk copy for ckpt / file group
        if is_main and FILE_GROUP["thread"] is not None:      # the previous step's push must not read a live dir being replaced
            FILE_GROUP["thread"].join()
            FILE_GROUP["thread"] = None
        if stepped and not OFFLOAD_MODEL and need_live_dir:
            publish_weights()
            dist.barrier()
            exported = True
        t_export = time.time() - t_export
        # ---- wait for the lookahead collection: no trajectory may span the weight replacement below
        t_drain = time.time()
        if is_main and pending is not None:
            pending[1].result()
        dist.barrier(group=cpu_group)
        t_drain = time.time() - t_drain
        # ---- sync weights to vLLM
        t_sync = time.time()
        refit_info: dict = {}
        gate_info: dict = {}
        nccl_gates = [u for u in GATE_URLS if u not in FILE_GATES]
        if args.streaming and stepped and is_main:
            # episodes keep running: push the live dir first (file path), then hold new requests at the proxies, drain the
            # in-flight ones, replace the weights, and reopen with the new version
            if REFIT != "nccl":
                push_live(args.live_dir, log)
            gate_info.update(gate_close(log, nccl_gates if REFIT == "nccl" else None))
        if stepped and REFIT == "nccl":
            refit_info = refit_vllm_nccl(model, refit_sender, refit_meta, device, is_main, log)
        elif stepped:                      # unchanged weights need no new version
            if not exported:
                publish_weights()
                dist.barrier()
            if OFFLOAD_MODEL:
                opt.zero_grad(set_to_none=True)
                model.to("cpu")
                torch.cuda.empty_cache()
            if is_main:
                vllm_wake(log)
                sync_vllm(args.live_dir, log, push=not args.streaming)
        elif is_main:
            vllm_wake(log)
            log(f"[step {step}] zero task signal: every advantage is 0; no optimizer update, no weight sync" if zero_signal and accum.k == 1 else
                f"[step {step}] gradient accumulation round {accum.rounds}/{accum.k}{' (no signal)' if zero_signal else ''}: "
                f"{'no optimizer update, no weight sync' if not stepped else ''}")
        if args.streaming and stepped and is_main:
            gate_info.update(gate_open(optimizer_updates, log, nccl_gates if REFIT == "nccl" else None))
            if REFIT == "nccl" and FILE_URLS and not FILE_GROUP["dead"]:
                import threading
                gate_info.update(FILE_GROUP["last"])
                FILE_GROUP["thread"] = threading.Thread(target=file_group_refit, args=(args.live_dir, optimizer_updates, log),
                                                        name="file-group-refit", daemon=True)
                FILE_GROUP["thread"].start()
        dist.barrier(group=cpu_group)
        t_sync = time.time() - t_sync
        rank_rng = None
        if step % args.save_steps == 0:
            rank_rng = [None] * world
            dist.all_gather_object(rank_rng, {"python": random.getstate(), "torch": torch.get_rng_state(),
                                             "cuda": torch.cuda.get_rng_state(device)})
        # ---- logging / checkpoints
        if is_main:
            rewards = [float(r["reward"]) for r in results if r.get("verdict") != "infrastructure_error"]
            verdicts = Counter(r.get("verdict") for r in results)
            comps = [r.get("components") or {} for r in results if r.get("verdict") != "infrastructure_error"]
            diags = [r.get("diagnostics") or {} for r in results]
            rec = {"step": step, "time": time.time(), "reward": sum(rewards) / max(1, len(rewards)),
                   "run_name": args.output_dir.name, "reward_version": os.environ.get("RL_REWARD_VERSION"),
                   "reward_std": float(torch.tensor(rewards).std()) if len(rewards) > 1 else 0.0, "n": len(results), "n_scored": len(rewards),
                   "n_trained": len(active), "n_segments": n_segments_global, "n_sample_errors": n_err, "n_tokens_trained": n_tokens_global, "loss": float(loss_sum.item()), "kl": kl_per_token, "kl_beta": args.kl_beta,
                   "sample_format": "per_request_v1", "objective_completion_tokens": full_completion_tokens,
                   **({"overlength/dropped": len(overlength_dropped),
                       "overlength/over_cap_requests": sum(int((r.get("diagnostics") or {}).get("over_cap_request_count") or 0) for r in results),
                       "overlength/untrainable_episodes": sum((r.get("diagnostics") or {}).get("no_trainable_requests") == "over_cap" for r in results)}
                      if max_drops > 0 else {}),
                   **{f"build/{k}": v for k, v in build_global.items()},
                   **sc_metrics, **mask_metrics, **kl_metrics, **critic_metrics, **early_stop_metrics(results),
                   "grad_norm": float(grad_norm), "lr": lr, "collection_round": step, "optimizer_updates": optimizer_updates,
                   "zero_signal": zero_signal, "accum/k": accum.k, "accum/round": accum.rounds, "accum/normaliser": accum_norm,
                   "accum/stepped": stepped, "valid_attempts": valid_attempts, "loss_normalization": args.loss_normalization, "t_env_s": round(t_env), "t_train_s": round(t_train), "t_sync_s": round(t_sync),
                   "t_step_s": round(time.time() - t_step), "max_seq_len": max_len,
                   "async/lookahead": int(args.async_lookahead), "async/staleness": staleness, "async/behavior_version": int(async_info["behavior_version"]),
                   "async/lookahead_launched": bool(async_info["lookahead_launched"]), "t_collect_s": round(async_info["t_collect"]),
                   "async/t_wait_s": round(async_info["t_wait"]), "async/t_drain_s": round(t_drain), **is_metrics, **code_metrics,
                   "t_export_s": round(t_export), "t_reload_s": round(t_sync), "refit_mode": REFIT,
                   **(dict(async_info.get("stream") or {}, **gate_info, streaming=True, **{f"stream/{k}": v for k, v in build_global.items() if k.startswith("stale")})
                      if args.streaming else {}),
                   **({"refit/t_s": refit_info.get("t_refit_s"), "refit/broadcast_s": refit_info.get("t_broadcast_s"),
                       "refit/gib": round(refit_info.get("bytes", 0) / 2**30, 2)} if refit_info else {}), "n_packed_sequences": n_segments_global,
                   "t_build_s": round(t_build, 1), "t_build_max_s": round(t_build_max, 1), "pack_assignment": "pack" if pack_level else "trajectory",
                   "ckpt/async": bool(args.async_checkpoint), "ckpt/last_step": ckpt_job["step"],
                   "ckpt/last_blocking_s": ckpt_job["blocking_s"], "ckpt/last_background_s": ckpt_job["background_s"],
                   "async/lookahead_through_checkpoints": bool(args.lookahead_through_checkpoints),
                   **slot_stats(results, env_concurrency),
                   "success_rate": sum(1 for r in results if (r.get("diagnostics") or {}).get("host_final_success")) / max(1, len(results)),
                   "false_finish_rate": verdicts.get("false_finish", 0) / max(1, len(results)),
                   "mean_requests": sum((d.get("requests") or 0) for d in diags) / max(1, len(diags)),
                   "mean_output_tokens": sum((d.get("output_tokens") or 0) for d in diags) / max(1, len(diags)),
                   "mean_final_prompt_tokens": sum((d.get("final_prompt_tokens") or 0) for d in diags) / max(1, len(diags)),
                   "mean_executed": sum(((d.get("physical") or {}).get("executed") or 0) for d in diags) / max(1, len(diags)),
                   "mean_clean": sum(((d.get("physical") or {}).get("clean") or 0) for d in diags) / max(1, len(diags)),
                   "retain_rate": sum(1 for d in diags if (d.get("physical") or {}).get("retain")) / max(1, len(diags)),
                   "mean_env_wall_s": sum((r.get("env_wall_s") or 0) for r in results) / max(1, len(results)),
                   **segment_stats}
            if probe_entries and step % args.probe_every == 0:
                rec.update(run_probe(step))
            for v in VERDICTS:
                rec[f"verdict/{v}"] = verdicts.get(v, 0) / max(1, len(results))
            for k in COMPONENT_KEYS:
                rec[f"component/{k}"] = sum(c.get(k, 0.0) for c in comps) / max(1, len(comps))
            with metrics_path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            with rollouts_path.open("a") as fh:
                for r in results:
                    d = r.get("diagnostics") or {}
                    fh.write(json.dumps({"step": step, "suite": r["suite"], "task_id": r["task_id"], "seed": r["seed"], "sid": r["sid"], "group": r["group"],
                                         "start_id": r.get("start_id"), "K": r.get("K"),
                                         "reward": r["reward"], "advantage": r.get("advantage"), "verdict": r["verdict"], "components": r.get("components"),
                                         "termination": d.get("termination_reason"), "requests": d.get("requests"), "output_tokens": d.get("output_tokens"),
                                         "final_prompt_tokens": d.get("final_prompt_tokens"), "physical": d.get("physical"), "tool_errors": d.get("tool_errors"),
                                         "gate_rejections": d.get("gate_rejections"), "success": d.get("host_final_success"), "env_wall_s": r.get("env_wall_s"),
                                         "reward_version": d.get("reward_version"), "reward_valid": d.get("reward_valid"),
                                         "sample_format": d.get("sample_format"), "eligible_request_count": d.get("eligible_request_count"),
                                         "n_training_requests": d.get("n_training_requests"), "full_completion_tokens": d.get("full_completion_tokens"),
                                         "goal_progress": d.get("goal_progress"), "goal_count": d.get("goal_count"),
                                         "initial_goals_satisfied": d.get("initial_goals_satisfied"), "final_goals_satisfied": d.get("final_goals_satisfied"),
                                         "episode_dir": r.get("episode_dir"), "finish_note": d.get("finish_note"), "error": d.get("error"),
                                         **({"start_group": r["start_group"]} if "start_group" in r else {})}) + "\n")
            log(f"[step {step}] reward {rec['reward']:.3f} (std {rec['reward_std']:.2f}) success {rec['success_rate']:.2f} ff {rec['false_finish_rate']:.2f} "
                f"verdicts {dict(verdicts)} trained {len(active)}/{len(results)} segments {n_segments_global} tokens {n_tokens_global:.0f} loss {rec['loss']:.4f} kl {rec['kl']:.4f} gn {rec['grad_norm']:.3f} "
                f"env {t_env:.0f}s train {t_train:.0f}s export {t_export:.0f}s sync {t_sync:.0f}s build {build_global}"
                + (f" async stale {staleness} collect {async_info['t_collect']:.0f}s wait {async_info['t_wait']:.0f}s drain {t_drain:.0f}s" if args.async_lookahead else "")
                + (f" is mean {is_metrics['is/mean_ratio']:.3f} clip {is_metrics['is/clip_frac']:.4f} ess {is_metrics['is/ess_frac']:.3f}" if is_metrics else "")
                + (f" code errors sel {code_metrics['code/selected_errors']} withheld {code_metrics['code/pos_credit_withheld_samples']}" if code_metrics else ""))
        if step % args.save_steps == 0:
            t_ckpt = time.time()
            write_checkpoint(step, rank_rng, ckpt_curriculum_state)
            if is_main:
                log(f"[checkpoint] step {step} {'staged (background write)' if args.async_checkpoint else 'written'} in "
                    f"{time.time() - t_ckpt:.0f}s (not included in t_step_s)")
        dist.barrier(group=cpu_group)
    if stream_pool is not None:
        stream_pool.close()
    wait_checkpoint_job()
    dist.barrier()
    if is_main and not args.full_finetune:
        model.save_pretrained(str(args.output_dir / "final-adapter"))
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        if exc.code not in (None, 0):
            sys.stderr.flush()
            os._exit(exc.code if isinstance(exc.code, int) else 1)
        raise
    except BaseException:  # noqa: BLE001
        # r5 13:47: rank 0 raised in the refit, but the non-daemon rollout-pool / checkpoint threads kept the interpreter
        # alive, every other rank waited in NCCL and the supervisor (process liveness) saw no crash: exit hard instead
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
