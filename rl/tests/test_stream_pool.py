"""StreamPool: continuous admission, oldest-first take, buffer bound, version tags; proxy gate."""
import asyncio
import random
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stream_pool import StreamPool  # noqa: E402


def test_pool_keeps_inflight_full_and_bounds_buffer():
    state = {"version": 0, "max_inflight": 0, "inflight": 0}
    lock = threading.Lock()
    drawn = []

    def draw(n):
        out = [{"id": len(drawn) + i} for i in range(n)]
        drawn.extend(out)
        return out

    def run_one(entry, step, gi):
        with lock:
            state["inflight"] += 1
            state["max_inflight"] = max(state["max_inflight"], state["inflight"])
        time.sleep(random.uniform(0.005, 0.03))
        with lock:
            state["inflight"] -= 1
        return {"verdict": "ok", "entry_id": entry["id"]}

    pool = StreamPool(draw=draw, run_one=run_one, batch=8, inflight=4, draw_chunk=8, version=lambda: state["version"], step=lambda: 1)
    try:
        entries, results, info = pool.take(8)
        assert len(entries) == len(results) == 8
        assert all(r["entry_id"] == e["id"] for e, r in zip(entries, results))
        assert all("stream" in r and r["stream"]["submit_version"] == 0 for r in results)
        assert state["max_inflight"] <= 4
        time.sleep(0.5)                                  # learner busy: the pool fills one batch and stops admitting
        assert len(pool.ready) <= 8 + 4 and pool.done <= 8 + 8 + 4
        state["version"] = 3
        _, results2, _ = pool.take(8)
        assert {r["stream"]["submit_version"] for r in results2} <= {0, 3}
        assert info["stream/t_wait_s"] >= 0
    finally:
        pool.close()


def test_proxy_gate_drains_and_tags_versions():
    import policy_proxy as P
    P.GATE_STATE.update(open=True, version=5, forwarding=0, closed_at=None, closures=0)
    v, _ = P._gate_enter()
    assert v == 5 and P.GATE_STATE["forwarding"] == 1
    out = {}

    def closer():
        out["close"] = asyncio.run(P.gate_close(timeout=10))

    t = threading.Thread(target=closer)
    t.start()
    time.sleep(0.3)
    assert t.is_alive() and not P.GATE_STATE["open"]        # waits for the in-flight request
    got = {}

    def late():                                             # a request arriving while closed waits for the reopen
        got["v"] = P._gate_enter()[0]

    t2 = threading.Thread(target=late)
    t2.start()
    P._gate_exit()
    t.join(5)
    assert out["close"]["closed"] is True
    time.sleep(0.2)
    assert "v" not in got
    asyncio.run(P.gate_open(version=6))
    t2.join(5)
    assert got["v"] == 6
    P._gate_exit()
    assert P.GATE_STATE["forwarding"] == 0
