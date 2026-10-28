"""Streaming rollout pool for SAO (train --streaming).

Rank 0 keeps `inflight` episodes running on the env server(s) at all times; finished episodes go into a ready queue and
the learner takes the oldest `batch` of them as soon as that many are ready (no round barrier, no batch tail). The pool
stops admitting new episodes while `ready + inflight` would exceed `batch + inflight` (one batch buffered beyond the one
being trained), which bounds the behaviour-version lag: a request is at most ~2 optimizer versions old when trained.

Exact per-request versions come from the policy proxy's gate (policy_proxy /gate/close, /gate/open?version=v): the
learner swaps weights while episodes keep running (an episode may span versions: partial rollout); each request records
the version that produced it, and the learner's SAO double-sided ratio mask + score centering use that request's own
sampler log-probabilities. Requests older than --max-staleness versions are dropped from the policy gradient.

Curriculum: entries are drawn in chunks of `draw` (curriculum.next_batch under a lock) and committed by the learner in the
order they are consumed (completion order, not draw order); entries in flight at a checkpoint are not persisted.

Group sampling (group_size G > 1, default 1 = the behaviour above bit for bit): each drawn
start is run G times (consecutive admissions, one group key); take(n) returns the n / G oldest COMPLETE groups (a group's
position = its earliest-finished member), members of a group adjacent, so the learner can use a group baseline. The batch
stays n episodes (n / G distinct starts per batch). Each result gets result["stream"]["group_key"] and result["start_group"].
"""
from __future__ import annotations

import collections
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable


class StreamPool:
    def __init__(self, *, draw: Callable[[int], list], run_one: Callable[[dict, int, int], dict], batch: int, inflight: int,
                 draw_chunk: int, version: Callable[[], int], step: Callable[[], int], log=print, group_size: int = 1,
                 monitor_draw: Callable[[], dict] | None = None, monitor_every: int = 0):
        if group_size < 1 or batch % group_size or draw_chunk % group_size or (group_size > 1 and inflight < group_size):
            # inflight >= G keeps take() live: when admission blocks, at most G - 1 ready episodes belong to incomplete groups
            raise ValueError(f"group_size {group_size} must divide batch {batch} and draw_chunk {draw_chunk} and be <= inflight {inflight}")
        self.draw, self.run_one, self.batch, self.inflight_max, self.draw_chunk = draw, run_one, batch, inflight, draw_chunk
        self.group_size = group_size
        self.gkey = 0
        # held-out monitor (None = the pool above bit for bit): every monitor_every-th admission runs a held-out
        # start (monitor_draw) instead; its result never enters the ready queue (no training, no critic, no curriculum update) and is
        # collected with pop_monitor()
        self.monitor_draw, self.monitor_every = monitor_draw, int(monitor_every or 0)
        self.monitor_done: list = []
        self.admitted = 0
        self.version, self.step, self.log = version, step, log
        self.cond = threading.Condition()
        self.ready: collections.deque = collections.deque()
        self.backlog: collections.deque = collections.deque()
        self.inflight = 0
        self.gi = 0
        self.stop = False
        self.started = time.time()
        self.done = 0
        self.busy_s = 0.0
        self.exec = ThreadPoolExecutor(max_workers=inflight, thread_name_prefix="stream-episode")
        self.thread = threading.Thread(target=self._admit, name="stream-admit", daemon=True)
        self.thread.start()

    def _admit(self) -> None:
        while True:
            with self.cond:
                while not self.stop and (self.inflight >= self.inflight_max
                                         or len(self.ready) + self.inflight >= self.batch + self.inflight_max):
                    self.cond.wait(timeout=5.0)
                if self.stop:
                    return
                self.inflight += 1
                gi, self.gi = self.gi, self.gi + 1
            self.admitted += 1
            if self.monitor_draw is not None and self.monitor_every > 0 and self.admitted % self.monitor_every == 0:
                self.exec.submit(self._one, dict(self.monitor_draw(), monitor=True), gi, self.version(), self.step())
                continue
            if not self.backlog:
                if self.group_size == 1:
                    self.backlog.extend(self.draw(self.draw_chunk))
                else:
                    for e in self.draw(self.draw_chunk // self.group_size):
                        self.backlog.extend((e, self.gkey) for _ in range(self.group_size))
                        self.gkey += 1
            entry = self.backlog.popleft()
            if self.group_size == 1:
                self.exec.submit(self._one, entry, gi, self.version(), self.step())
            else:
                self.exec.submit(self._one, entry[0], gi, self.version(), self.step(), entry[1])

    def _one(self, entry: dict, gi: int, version: int, step: int, group_key: int | None = None) -> None:
        t0 = time.time()
        try:
            res = self.run_one(entry, step, gi)
        except Exception as exc:  # noqa: BLE001 - recorded as an infrastructure error like _post_rollout
            res = {"reward": 0.0, "verdict": "infrastructure_error", "components": {}, "sample_path": None,
                   "diagnostics": {"error": repr(exc), "infrastructure": True}}
        res.setdefault("diagnostics", {})
        res["stream"] = {"submit_version": version, "submit_step": step, "t_submit": t0, "t_done": time.time(), "gi": gi}
        if group_key is not None:
            res["stream"]["group_key"] = res["start_group"] = group_key
        with self.cond:
            self.inflight -= 1
            self.busy_s += time.time() - t0
            if entry.get("monitor"):
                self.monitor_done.append((entry, res))
            else:
                self.done += 1
                self.ready.append((entry, res))
            self.cond.notify_all()

    def take(self, n: int) -> tuple[list, list, dict]:
        t0 = time.time()
        with self.cond:
            ready_at_call = len(self.ready)
            if self.group_size == 1:
                while len(self.ready) < n:
                    self.cond.wait(timeout=10.0)
                items = [self.ready.popleft() for _ in range(n)]
            else:
                if n % self.group_size:
                    raise ValueError(f"take({n}) with group_size {self.group_size}")
                while True:
                    keys = self._complete_groups()
                    if len(keys) >= n // self.group_size:
                        break
                    self.cond.wait(timeout=10.0)
                want = keys[:n // self.group_size]
                items = [it for k in want for it in self.ready if it[1]["stream"]["group_key"] == k]
                keep = set(want)
                self.ready = collections.deque(it for it in self.ready if it[1]["stream"]["group_key"] not in keep)
            info = {"stream/ready_at_take": ready_at_call, "stream/inflight": self.inflight, "stream/ready_left": len(self.ready),
                    "stream/episodes_done": self.done,
                    "stream/throughput_eps_per_h": round(3600 * self.done / max(1.0, time.time() - self.started), 1)}
            self.cond.notify_all()
        info["stream/t_wait_s"] = round(time.time() - t0, 1)
        return [e for e, _ in items], [r for _, r in items], info

    def pop_monitor(self) -> list:
        """finished held-out monitor episodes since the last call: [(entry, result)]"""
        with self.cond:
            out, self.monitor_done = self.monitor_done, []
        return out

    def _complete_groups(self) -> list:
        """group keys with all group_size members ready, ordered by their earliest-finished member (caller holds cond)"""
        count: dict = {}
        first: list = []
        for _, res in self.ready:
            k = res["stream"]["group_key"]
            if k not in count:
                count[k] = 0
                first.append(k)
            count[k] += 1
        return [k for k in first if count[k] == self.group_size]

    def close(self) -> None:
        with self.cond:
            self.stop = True
            self.cond.notify_all()
        self.exec.shutdown(wait=False, cancel_futures=True)
