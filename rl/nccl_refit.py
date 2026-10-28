"""NCCL weight refit straight into the policy vLLM servers (trainer side, no vLLM import).

Borrowed design (Molt PolicyTrainer.broadcast_to_vllm, veRL, vLLM's own RLHF weight-transfer engine): trainer rank 0
opens a NCCL group with every policy vLLM worker and broadcasts the bf16 weights in packed buffers (256 MiB); the servers
load them in place with the layerwise-reload lifecycle. Replaces "export 19 GB safetensors -> push -> reload_weights".

Server side is UPSTREAM vLLM (>= 0.28): `vllm serve ... --weight-transfer-config '{"backend":"nccl"}'` with
VLLM_SERVER_DEV_MODE=1 exposes /init_weight_transfer_engine, /start_weight_update, /update_weights and
/finish_weight_update (vllm/entrypoints/serve/dev/rlhf/api_router.py). This module speaks the trainer half of that
protocol (vllm/distributed/weight_transfer/{nccl_engine,nccl_common,packed_tensor}.py, Apache-2.0, vLLM contributors):

  * rendezvous: vLLM's StatelessProcessGroup = a torch TCPStore served by the trainer (rank 0); the NCCL unique id is
    published under "broadcast_from/0/0" as a pickled vllm...pynccl_wrapper.ncclUniqueId; then ncclCommInitRank and one
    float32 all-reduce warmup on every rank;
  * per update: names/dtypes/shapes go over HTTP, the bytes go through ncclBroadcast in packed uint8 buffers. Packing
    rule (must match packed_nccl_broadcast_consumer): append tensors until the running size EXCEEDS the buffer size.

The trainer process loads the SAME libnccl as the servers through ctypes (RL_REFIT_NCCL_SO): NCCL refuses peers with a
different version, and the trainer's torch (2.10/cu128, NCCL 2.27) differs from the vLLM env (torch 2.13/cu130,
NCCL 2.29). NCCL statically links cudart, so the cu13 build runs in the cu12.8 trainer process on a 580 driver.
"""
from __future__ import annotations

import ctypes
import json
import os
import pickle
import socket
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Iterable, Iterator

import requests
import torch

# vLLM's default is 1 GiB x 2; the servers run at --gpu-memory-utilization 0.85 on 40 GB cards and the receive buffers sit
# next to the layerwise-reload staging (the 1.9 GiB embedding is the peak): 1 GiB buffers OOMed a policy server sharing its GPU
# (Molt cut its flush to 512 MiB for the same reason). The size is a wire parameter sent to the servers at connect().
DEFAULT_BUFFER_BYTES = int(os.environ.get("RL_REFIT_BUFFER_MB", "256")) * 1024 * 1024
DEFAULT_NUM_BUFFERS = 2                       # vLLM DEFAULT_PACKED_NUM_BUFFERS
NCCL_UINT8, NCCL_FLOAT32, NCCL_SUM = 1, 7, 0
_VLLM_UID_MODULE = "vllm.distributed.device_communicators.pynccl_wrapper"


class NcclError(RuntimeError):
    pass


class _UniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


class Nccl:
    """Minimal ctypes binding of the NCCL calls the sender needs (after vLLM's pynccl_wrapper)."""

    def __init__(self, path: str):
        self.path = path
        self.lib = ctypes.CDLL(path)          # RTLD_LOCAL: coexists with the NCCL torch already loaded
        c = ctypes
        self.lib.ncclGetErrorString.restype = c.c_char_p
        self.lib.ncclGetErrorString.argtypes = [c.c_int]
        for name, args in {"ncclGetVersion": [c.POINTER(c.c_int)],
                           "ncclGetUniqueId": [c.POINTER(_UniqueId)],
                           "ncclCommInitRank": [c.POINTER(c.c_void_p), c.c_int, _UniqueId, c.c_int],
                           "ncclAllReduce": [c.c_void_p, c.c_void_p, c.c_size_t, c.c_int, c.c_int, c.c_void_p, c.c_void_p],
                           "ncclBroadcast": [c.c_void_p, c.c_void_p, c.c_size_t, c.c_int, c.c_int, c.c_void_p, c.c_void_p],
                           "ncclCommDestroy": [c.c_void_p]}.items():
            f = getattr(self.lib, name)
            f.restype, f.argtypes = c.c_int, args

    def check(self, rc: int) -> None:
        if rc != 0:
            raise NcclError(f"NCCL error {rc}: {self.lib.ncclGetErrorString(rc).decode()} ({self.path})")

    def version(self) -> int:
        v = ctypes.c_int()
        self.check(self.lib.ncclGetVersion(ctypes.byref(v)))
        return v.value

    def unique_id(self) -> _UniqueId:
        u = _UniqueId()
        self.check(self.lib.ncclGetUniqueId(ctypes.byref(u)))
        return u

    def init_rank(self, world: int, uid: _UniqueId, rank: int) -> ctypes.c_void_p:
        comm = ctypes.c_void_p()
        self.check(self.lib.ncclCommInitRank(ctypes.byref(comm), world, uid, rank))
        return comm


def pickle_unique_id(uid: _UniqueId) -> bytes:
    """Pickle the id exactly as vLLM's StatelessProcessGroup.broadcast_obj would (the class is resolved by its vLLM
    module path on the server). The trainer env has no vLLM: register a stand-in module only for the dump."""
    cls = type("ncclUniqueId", (ctypes.Structure,), {"_fields_": [("internal", ctypes.c_byte * 128)], "__module__": _VLLM_UID_MODULE})
    obj = cls()
    ctypes.memmove(ctypes.byref(obj), ctypes.byref(uid), 128)
    parts = _VLLM_UID_MODULE.split(".")
    added = []
    try:
        for i in range(1, len(parts) + 1):
            name = ".".join(parts[:i])
            if name not in sys.modules:
                sys.modules[name] = types.ModuleType(name)
                added.append(name)
        if not hasattr(sys.modules[_VLLM_UID_MODULE], "ncclUniqueId") or _VLLM_UID_MODULE in added:
            sys.modules[_VLLM_UID_MODULE].ncclUniqueId = cls
        cls.__qualname__ = "ncclUniqueId"
        return pickle.dumps(obj)
    finally:
        for name in added:
            sys.modules.pop(name, None)


def default_nccl_so() -> str:
    path = os.environ.get("RL_REFIT_NCCL_SO")
    if path:
        return path
    import glob
    hits = sorted(glob.glob(os.path.join(os.environ.get("CAPTAIN_WORKSPACE", os.path.expanduser("~/captain")),
                                         "envs/vllm/lib/python3*/site-packages/nvidia/nccl/lib/libnccl.so.2")))
    if not hits:
        raise RuntimeError("set RL_REFIT_NCCL_SO to the libnccl.so.2 of the policy vLLM env (versions must match)")
    return hits[0]


def _post(url: str, route: str, body: dict | None = None, timeout: float = 1800.0) -> dict:
    r = requests.post(f"{url}{route}", json=body, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"{url}{route} -> HTTP {r.status_code}: {r.text[:500]}")
    try:
        return r.json()
    except ValueError:
        return {}


def pack_chunks(items: Iterator[tuple[str, torch.Tensor]], buffer_bytes: int) -> Iterator[list[tuple[str, torch.Tensor]]]:
    """Group (name, tensor) pairs exactly like vLLM's pack_tensors / consumer: append, then flush once the running size
    exceeds buffer_bytes."""
    chunk, size = [], 0
    for name, t in items:
        chunk.append((name, t))
        size += t.numel() * t.element_size()
        if size > buffer_bytes:
            yield chunk
            chunk, size = [], 0
    if chunk:
        yield chunk


class VllmRefitSender:
    """Trainer rank 0's endpoint. connect() once (all servers up), then send() per policy version."""

    def __init__(self, urls: list[str], *, master_addr: str | None = None, master_port: int = 0, nccl_so: str | None = None,
                 buffer_bytes: int = DEFAULT_BUFFER_BYTES, num_buffers: int = DEFAULT_NUM_BUFFERS, log=print):
        self.urls = list(urls)
        self.master_addr = master_addr or os.environ.get("RL_REFIT_ADDR") or _routable_ip(self.urls[0])
        self.master_port = int(master_port or os.environ.get("RL_REFIT_PORT") or _free_port())
        self.nccl = Nccl(nccl_so or default_nccl_so())
        self.buffer_bytes, self.num_buffers = int(buffer_bytes), int(num_buffers)
        self.log = log
        self.comm = None
        self.store = None
        self.device = torch.device("cuda", torch.cuda.current_device())

    # ---- rendezvous
    def connect(self, timeout_s: int = 600) -> float:
        t0 = time.time()
        world = 1 + len(self.urls)
        from torch.distributed import TCPStore
        self.store = TCPStore(self.master_addr, self.master_port, world_size=world, is_master=True,
                              timeout=timedelta(seconds=timeout_s), use_libuv=False, wait_for_workers=False)
        uid = self.nccl.unique_id()
        self.store.set("broadcast_from/0/0", pickle_unique_id(uid))
        infos = [{"master_address": self.master_addr, "master_port": self.master_port, "rank_offset": 1 + i, "world_size": world,
                  "packed": True, "packed_buffer_size_bytes": self.buffer_bytes, "packed_num_buffers": self.num_buffers}
                 for i in range(len(self.urls))]
        with ThreadPoolExecutor(max_workers=len(self.urls)) as pool:
            futs = [pool.submit(_post, u, "/init_weight_transfer_engine", {"init_info": info}, timeout_s) for u, info in zip(self.urls, infos)]
            with torch.cuda.device(self.device):
                self.comm = self.nccl.init_rank(world, uid, 0)
                warm = torch.zeros(1, device=self.device)          # PyNcclCommunicator's warmup all-reduce
                s = torch.cuda.current_stream()
                self.nccl.check(self.nccl.lib.ncclAllReduce(warm.data_ptr(), warm.data_ptr(), 1, NCCL_FLOAT32, NCCL_SUM, self.comm,
                                                            ctypes.c_void_p(s.cuda_stream)))
                s.synchronize()
            for f in futs:
                f.result()
        dt = time.time() - t0
        self.log(f"[refit] NCCL group up: {len(self.urls)} vLLM servers, nccl {self.nccl.version()} ({self.nccl.path}), "
                 f"{self.master_addr}:{self.master_port}, {dt:.1f}s")
        return dt

    def _broadcast(self, t: torch.Tensor, stream: torch.cuda.Stream) -> None:
        self.nccl.check(self.nccl.lib.ncclBroadcast(t.data_ptr(), t.data_ptr(), t.numel(), NCCL_UINT8, 0, self.comm,
                                                    ctypes.c_void_p(stream.cuda_stream)))

    # ---- one policy version
    def send(self, metas: list[tuple[str, torch.dtype, tuple[int, ...]]], tensors: Iterable[tuple[str, torch.Tensor]],
             version: str | None = None) -> dict:
        """metas: (name, dtype, shape) in iteration order (told to the servers first); tensors: the same sequence of
        full CUDA tensors. Caller pauses generation before and resets caches / resumes after."""
        assert self.comm is not None, "connect() first"
        t0 = time.time()
        info = {"names": [m[0] for m in metas], "dtype_names": [str(m[1]).split(".")[-1] for m in metas],
                "shapes": [list(m[2]) for m in metas]}
        with ThreadPoolExecutor(max_workers=len(self.urls)) as pool:
            list(pool.map(lambda u: _post(u, "/start_weight_update"), self.urls))
        pool = ThreadPoolExecutor(max_workers=len(self.urls))
        futs = [pool.submit(_post, u, "/update_weights", {"update_info": info}) for u in self.urls]
        n_bytes, n_chunks, checked = 0, 0, 0
        streams = [torch.cuda.Stream(self.device) for _ in range(self.num_buffers)]
        in_flight: list[torch.Tensor | None] = [None] * self.num_buffers
        k = 0
        try:
            def checked_iter():
                nonlocal checked
                for name, t in tensors:
                    m = metas[checked]
                    if name != m[0] or t.dtype != m[1] or tuple(t.shape) != tuple(m[2]):
                        raise ValueError(f"refit tensor {checked} {name} {t.dtype} {tuple(t.shape)} != declared {m}")
                    checked += 1
                    yield name, t
            for chunk in pack_chunks(checked_iter(), self.buffer_bytes):
                for f in futs:
                    if f.done():
                        f.result()           # a server that already failed would leave the broadcast hanging
                streams[k].synchronize()
                in_flight[k] = None
                streams[k].wait_stream(torch.cuda.current_stream())     # the gathered tensors were produced there
                with torch.cuda.stream(streams[k]):
                    flat = torch.cat([t.contiguous().view(torch.uint8).view(-1) for _, t in chunk])
                    self._broadcast(flat, streams[k])
                    in_flight[k] = flat
                n_bytes += flat.numel()
                n_chunks += 1
                k = (k + 1) % self.num_buffers
            if checked != len(metas):
                raise ValueError(f"refit sent {checked} tensors, declared {len(metas)}")
            for s in streams:
                s.synchronize()
            t_bcast = time.time() - t0
            for f in futs:
                f.result()
        finally:
            pool.shutdown(wait=False)
        t1 = time.time()
        with ThreadPoolExecutor(max_workers=len(self.urls)) as p2:
            list(p2.map(lambda u: _post(u, "/finish_weight_update", {"weight_version": version} if version else None), self.urls))
        t_finish = time.time() - t1
        released = release_server_cache(self.urls, self.log)
        return {"bytes": n_bytes, "chunks": n_chunks, "tensors": checked, "t_broadcast_s": round(t_bcast, 2),
                "t_finish_s": round(t_finish, 2), "t_total_s": round(time.time() - t0, 2), **released}


_EXT_WARNED = {"done": False}


def release_server_cache(urls: list[str], log=print) -> dict:
    """Streaming: after a weight update, return the servers' cached transfer blocks to the driver (vllm_worker_ext;
    RL_REFIT_EMPTY_CACHE=0 disables). Servers without the extension answer 500 once and are skipped with a warning."""
    if os.environ.get("RL_REFIT_EMPTY_CACHE", "1") != "1":
        return {}
    def one(u):
        try:
            r = requests.post(f"{u}/collective_rpc", json={"method": "refit_empty_cache", "timeout": 120}, timeout=180)
            return r.json()["results"][0] if r.status_code == 200 else None
        except Exception:  # noqa: BLE001
            return None
    with ThreadPoolExecutor(max_workers=len(urls)) as pool:
        res = list(pool.map(one, urls))
    ok = [r for r in res if isinstance(r, dict)]
    if len(ok) < len(urls) and not _EXT_WARNED["done"]:
        _EXT_WARNED["done"] = True
        log(f"[refit] {len(urls) - len(ok)} policy servers lack the vllm_worker_ext empty-cache hook (their footprint can grow)")
    if not ok:
        return {}
    return {"server_reserved_max_gib": max(r["reserved_gib"] for r in ok),
            "server_released_gib": round(sum(r["reserved_before_gib"] - r["reserved_gib"] for r in ok), 2)}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _routable_ip(url: str) -> str:
    """The local address used to reach the first policy server (its node must reach our TCPStore on it)."""
    from urllib.parse import urlparse
    host = urlparse(url).hostname or "127.0.0.1"
    port = urlparse(url).port or 80
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect((host, port))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


# ---------------------------------------------------------------- trainer-side tensors (FSDP2 full fine-tuning)

def refit_names(model, *, include_frozen_visual: bool = False) -> list[tuple[str, torch.dtype, tuple[int, ...]]]:
    """(name, bf16, full shape) of every floating state-dict entry the servers must receive, in a fixed order. The frozen
    vision tower never changes during RL and is skipped by default (the servers keep the init weights)."""
    out = []
    for name, t in model.state_dict().items():
        if not torch.is_tensor(t) or not t.is_floating_point() or name.endswith("_extra_state"):
            continue
        if not include_frozen_visual and ".visual." in f".{name}":
            continue
        out.append((name, torch.bfloat16, tuple(t.shape)))
    return out


def iter_full_bf16(model, names: list[tuple[str, torch.dtype, tuple[int, ...]]], device) -> Iterator[tuple[str, torch.Tensor | None]]:
    """Collective over all trainer ranks: yields (name, full bf16 tensor on `device`) on every rank (callers on ranks > 0
    just drop them). Sharded DTensors are all-gathered over NCCL even when FSDP keeps the shards in host memory
    (RL_FSDP_CPU_OFFLOAD): the local shard is copied to the GPU first, so the gather never goes through gloo."""
    from torch.distributed.tensor import DTensor

    sd = model.state_dict()
    for name, _, shape in names:
        t = sd[name]
        if isinstance(t, DTensor):
            local = t.to_local()
            if local.device.type != "cuda":
                local = local.to(device, non_blocking=True)
            local = local.to(torch.bfloat16)
            full = DTensor.from_local(local, t.device_mesh, t.placements, shape=t.shape, stride=t.stride(),
                                      run_check=False).full_tensor()
        else:
            full = t.to(device=device, dtype=torch.bfloat16)
        if tuple(full.shape) != tuple(shape):
            raise RuntimeError(f"refit gather {name}: {tuple(full.shape)} != {shape}")
        yield name, full


def reset_caches(url: str) -> None:
    for route in ("/reset_prefix_cache", "/reset_mm_cache", "/reset_encoder_cache"):
        r = requests.post(f"{url}{route}", timeout=300)
        r.raise_for_status()


if __name__ == "__main__":         # tiny self-check of the packing rule
    xs = [(str(i), torch.empty(n, dtype=torch.uint8)) for i, n in enumerate([300, 300, 500, 100, 900, 50])]
    print(json.dumps([[n for n, _ in c] for c in pack_chunks(iter(xs), 512)]))
