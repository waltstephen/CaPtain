"""The proxy's status route and cheap record snapshots."""
import asyncio
import copy
import json

import policy_proxy as proxy


def fake_session(sid="s0"):
    rec = proxy._record(sid)
    with proxy.LOCK:
        rec["requests"] = [{"request_index": i, "completed": i < 3, "status": 200, "prompt_token_ids": list(range(2000)),
                            "token_ids": [i] * 50, "sampler_topk": {"k": 2, "ids": [[1, 2]] * 50, "logprobs": [[-0.1, -2.0]] * 50,
                                                                    "sampled_logprob": [-0.1] * 50}} for i in range(4)]
        rec["in_flight"], rec["errors"] = 1, 0
        rec["images"] = {"h": "data:image/png;base64,AAAA"}
        rec["last_request"] = {"messages": [{"role": "user", "content": "x"}]}
        rec["payload_attempts"] = {"fp": 1}
    return rec


def test_status_returns_counters_only():
    fake_session()
    got = asyncio.run(proxy.status("s0"))
    assert got == {"sid": "s0", "in_flight": 1, "requests": 4, "completed": 3, "errors": 0}
    assert asyncio.run(proxy.status("missing")).status_code == 404


def test_record_bytes_equal_the_previous_deep_copied_response():
    rec = fake_session("s1")
    for full in (1, 0):
        expected = {k: copy.deepcopy(v) for k, v in rec.items() if k != "payload_attempts"}
        if not full:
            expected.pop("last_request", None)
            expected.pop("images", None)
        got = json.loads(asyncio.run(proxy.record("s1", full=full)).body)
        assert got == json.loads(json.dumps(expected))


def test_snapshot_is_isolated_from_later_entry_updates():
    rec = fake_session("s2")
    with proxy.LOCK:
        snap = proxy.record_snapshot(rec, True)
    rec["requests"][3].update(completed=True, status=502)
    rec["requests"].append({"request_index": 9})
    rec["images"]["h2"] = "later"
    assert snap["requests"][3]["completed"] is False and len(snap["requests"]) == 4 and "h2" not in snap["images"]
