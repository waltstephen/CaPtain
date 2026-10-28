"""Original-request sampling and proxy provenance; no models or live services."""
from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import io
import json
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import env_server as env
import policy_proxy as proxy
from chat_tokens import IM_END, IM_START, RECORD_FORMAT, ROLE_ASSISTANT, completion_mask, same_text


def request(index, prompt, completion, **extra):
    return {"request_index": index, "completed": True, "status": 200,
            "prompt_token_ids": list(prompt), "token_ids": list(completion),
            "image_hashes": [], "finish_reason": "stop", "sampling": dict(proxy.DEFAULT_SAMPLING), **extra}


def record(requests, images=None):
    return {"record_format": RECORD_FORMAT, "sid": "fixed-session", "in_flight": 0,
            "requests": requests, "images": images or {}}


def record_json(sid, full=1):
    return json.loads(asyncio.run(proxy.record(sid, full=full)).body)


def png_url(color):
    buf = io.BytesIO()
    Image.new("RGB", (2, 2), color).save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    return hashlib.sha256(url.encode("utf-8")).hexdigest(), url


class RawRequestSamplesTest(unittest.TestCase):
    def test_response_arriving_after_the_episode_ended_is_recorded_but_not_trained(self):
        original = [request(1, [10, 11], [21, IM_END], t=100.0, latency_s=5.0),
                    request(2, [10, 99, 11], [22, IM_END], t=110.0, latency_s=20.0)]
        sample = env.build_training_sample(record(original), sid="fixed-session", max_training_requests=-1, ended_at=120.0)
        self.assertEqual([q["request_index"] for q in sample["training_requests"]], [1])
        self.assertEqual(sample["eligible_request_count"], 1)
        self.assertTrue(sample["requests"][1]["after_episode_end"])
        both = env.build_training_sample(record(original), sid="fixed-session", max_training_requests=-1)
        self.assertEqual(both["eligible_request_count"], 2)

    def test_repeated_and_unparsed_responses_keep_their_original_contexts(self):
        original = [
            request(1, [10, 11], [21, IM_END]),
            request(2, [10, 99, 11], [21, IM_END]),
            request(3, [10, 77, 11], [31, 32], finish_reason="length", n_tool_calls=0, content_chars=0),
        ]
        sample = env.build_training_sample(record(original), sid="fixed-session", max_training_requests=-1)
        self.assertEqual(sample["sample_format"], "per_request_v1")
        self.assertEqual(sample["full_completion_tokens"], 6)
        self.assertEqual(len(sample["training_requests"]), 3)
        for raw, selected in zip(original, sample["training_requests"]):
            self.assertEqual(selected["prompt_token_ids"], raw["prompt_token_ids"])
            self.assertEqual(selected["token_ids"], raw["token_ids"])
            self.assertEqual(selected["loss_weight"], 1.0)
        self.assertNotIn("segments", sample)
        self.assertNotIn("repairs", sample)

    def test_mask_covers_only_sampled_suffix_without_role_parsing_or_eos(self):
        prompt = [IM_START, ROLE_ASSISTANT, 198, 19, IM_END]
        generated = [IM_START, ROLE_ASSISTANT, 198, 7]
        mask = completion_mask(len(prompt), len(generated))
        self.assertEqual(mask, [0] * len(prompt) + [1] * len(generated))
        self.assertEqual([t for t, m in zip(prompt + generated, mask) if m], generated)
        self.assertEqual(completion_mask(1, 0), [0])
        for lengths in [(0, 1), (-1, 1), (2, -1), (True, 1)]:
            with self.assertRaises(ValueError):
                completion_mask(*lengths)

    def test_no_decoded_text_or_whitespace_equivalence(self):
        class Tok:
            def decode(self, ids, **kwargs):
                return {1: 'text("red mug")', 2: 'text("redmug")', 3: 'text("red mug")'}[ids[0]]
        self.assertFalse(same_text(Tok(), [1], [2]))
        self.assertFalse(same_text(Tok(), [1], [3]))
        self.assertTrue(same_text(Tok(), [1], [1]))

    def test_observation_replacement_does_not_change_saved_original_request(self):
        old_hash, old_url = png_url("red")
        new_hash, new_url = png_url("blue")
        original = [request(1, [10, 1, 11], [21], image_hashes=[old_hash]),
                    request(2, [10, 2, 11], [21], image_hashes=[new_hash])]
        sample = env.build_training_sample(record(original, {old_hash: old_url, new_hash: new_url}),
                                           sid="fixed-session", max_training_requests=-1)
        first, second = sample["training_requests"]
        self.assertEqual(first["images"], [old_url])
        self.assertEqual(second["images"], [new_url])
        self.assertEqual(first["prompt_token_ids"], [10, 1, 11])
        self.assertEqual(second["prompt_token_ids"], [10, 2, 11])

    def test_uniform_sampling_keeps_fixed_full_normalizer_and_exact_weights(self):
        original = [request(i, [10, i], [20] * i) for i in range(1, 8)]
        full = record(original)
        sample = env.build_training_sample(full, sid="fixed-session", max_training_requests=2)
        repeated = env.build_training_sample(record(list(reversed(original))), sid="fixed-session", max_training_requests=2)
        self.assertEqual(sample["training_requests"], repeated["training_requests"])
        self.assertEqual(sample["selection_seed"], repeated["selection_seed"])
        self.assertEqual(sample["full_completion_tokens"], 28)
        self.assertEqual(sample["eligible_request_count"], 7)
        self.assertEqual(sample["selected_request_count"], 2)
        self.assertEqual(sum(q["completion_length"] for q in sample["eligible_requests"]), 28)
        for q in sample["training_requests"]:
            self.assertEqual(q["inclusion_probability"], 2 / 7)
            self.assertEqual(q["loss_weight"], 3.5)
        for cap in [1, 4, -1]:
            self.assertEqual(env.build_training_sample(full, sid="fixed-session", max_training_requests=cap)["full_completion_tokens"], 28)

    def test_errors_and_empty_completions_have_explicit_eligibility(self):
        original = [request(1, [10], []),
                    {"request_index": 2, "completed": True, "status": 502, "error": "timeout"},
                    request(3, [10], [21])]
        sample = env.build_training_sample(record(original), sid="fixed-session")
        self.assertEqual(sample["eligible_requests"], [{"request_index": 3, "prompt_length": 1, "completion_length": 1}])
        self.assertEqual(sample["empty_completion_request_count"], 1)
        self.assertEqual(sample["failed_request_count"], 1)
        self.assertEqual([q["eligible"] for q in sample["requests"]], [False, False, True])

    def test_transformed_sampling_is_rejected_before_request_subsampling(self):
        original = [request(i, [10], [21]) for i in range(1, 10)]
        for key, value in [("min_tokens", 1), ("ignore_eos", True), ("tool_choice", "required")]:
            rows = copy.deepcopy(original)
            rows[-1]["sampling"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "request 9"):
                env.build_training_sample(record(rows), sid="fixed-session", max_training_requests=1)

    def test_invalid_records_fail_before_selection_instead_of_being_dropped(self):
        base = record([request(i, [10], [21]) for i in range(1, 10)])
        variants = []
        missing_image = copy.deepcopy(base)
        missing_image["requests"][-1]["image_hashes"] = ["missing"]
        variants.append(missing_image)
        for key, value in [("prompt_token_ids", []), ("token_ids", [True]), ("token_ids", [-1]), ("sampling", None)]:
            bad = copy.deepcopy(base)
            bad["requests"][-1][key] = value
            variants.append(bad)
        for key, value in [("record_format", "old"), ("sid", "different"), ("in_flight", 1)]:
            bad = copy.deepcopy(base)
            bad[key] = value
            variants.append(bad)
        bad = copy.deepcopy(base)
        bad["requests"][-1]["request_index"] = 1
        variants.append(bad)
        for bad in variants:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                env.build_training_sample(bad, sid="fixed-session", max_training_requests=1)

    def test_invalid_original_image_is_rejected_even_when_hash_matches(self):
        url = "data:image/png;base64," + base64.b64encode(b"not an image").decode("ascii")
        h = hashlib.sha256(url.encode("utf-8")).hexdigest()
        with self.assertRaisesRegex(ValueError, "image payload is invalid"):
            env.build_training_sample(record([request(1, [10], [21], image_hashes=[h])], {h: url}), sid="fixed-session")


class ProxyRecordingTest(unittest.TestCase):
    def setUp(self):
        proxy.RECORDS.clear()
        self.state_patch = mock.patch.dict(proxy.STATE, {"sampling": dict(proxy.DEFAULT_SAMPLING)})
        self.state_patch.start()
        self.addCleanup(self.state_patch.stop)

    @staticmethod
    def response(token):
        return type("Response", (), {"status_code": 200, "json": lambda self: {
            "prompt_token_ids": [10, 11], "choices": [{"token_ids": [token], "finish_reason": "length",
                                                       "message": {"role": "assistant", "content": None}}],
        }})()

    def test_retry_indices_settings_and_empty_parsed_responses_are_recorded(self):
        body = {"messages": [{"role": "user", "content": "task"}], "temperature": 0.5, "max_tokens": 128}
        with mock.patch.object(proxy.requests, "post", side_effect=[TimeoutError("retry"), self.response(21)]):
            self.assertEqual(proxy._forward("s", body)[0], 502)
            status, response, _ = proxy._forward("s", body)
        rec = record_json("s")
        self.assertEqual(status, 200)
        self.assertNotIn("prompt_token_ids", response)
        self.assertEqual([q["request_index"] for q in rec["requests"]], [1, 2])
        self.assertEqual([q["payload_attempt"] for q in rec["requests"]], [1, 2])
        self.assertEqual([q["status"] for q in rec["requests"]], [502, 200])
        q = rec["requests"][1]
        self.assertEqual(q["sampling"]["temperature"], 0.5)
        self.assertEqual(q["sampling"]["top_k"], -1)
        self.assertEqual(q["sampling"]["max_tokens"], 128)
        self.assertEqual(q["token_ids"], [21])
        self.assertEqual(q["n_tool_calls"], 0)
        self.assertEqual(rec["in_flight"], 0)
        self.assertEqual(rec["record_format"], RECORD_FORMAT)

    def test_out_of_order_responses_keep_invocation_order_and_latest_identity(self):
        started, release = threading.Event(), threading.Event()
        def post(url, json, timeout):
            if json["messages"][0]["content"] == "slow":
                started.set()
                if not release.wait(3):
                    raise RuntimeError("test release timeout")
                return self.response(21)
            return self.response(22)
        with mock.patch.object(proxy.requests, "post", side_effect=post), ThreadPoolExecutor(max_workers=2) as pool:
            slow = pool.submit(proxy._forward, "s", {"messages": [{"role": "user", "content": "slow"}]})
            self.assertTrue(started.wait(3))
            snapshot = record_json("s")
            self.assertEqual(snapshot["in_flight"], 1)
            self.assertFalse(snapshot["requests"][0]["completed"])
            proxy._forward("s", {"messages": [{"role": "user", "content": "fast"}]})
            release.set()
            slow.result(timeout=3)
        rec = record_json("s")
        self.assertEqual([q["token_ids"] for q in rec["requests"]], [[21], [22]])
        self.assertEqual(rec["last_response"]["request_index"], 2)
        self.assertEqual(rec["last_response"]["token_ids"], [22])
        self.assertFalse(snapshot["requests"][0]["completed"])

    def test_proxy_image_payload_round_trips_through_the_new_sample_builder(self):
        h, url = png_url("red")
        body = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}]}
        with mock.patch.object(proxy.requests, "post", return_value=self.response(21)):
            proxy._forward("fixed-session", body)
        rec = record_json("fixed-session")
        sample = env.build_training_sample(rec, sid="fixed-session")
        self.assertEqual(sample["training_requests"][0]["image_hashes"], [h])
        self.assertEqual(sample["training_requests"][0]["images"], [url])


if __name__ == "__main__":
    unittest.main()


class UpstreamRoutingTest(unittest.TestCase):
    def test_sessions_are_pinned_to_one_policy_server_and_spread_over_all(self):
        saved = proxy.STATE["upstreams"]
        try:
            proxy.STATE["upstreams"] = ["http://a", "http://b", "http://c"]
            picks = {sid: proxy.upstream_for(sid) for sid in (f"s{i}" for i in range(300))}
            self.assertEqual({sid: proxy.upstream_for(sid) for sid in picks}, picks)
            counts = [list(picks.values()).count(u) for u in proxy.STATE["upstreams"]]
            self.assertTrue(min(counts) > 60, counts)
            proxy.STATE["upstreams"] = ["http://only"]
            self.assertEqual(proxy.upstream_for("s1"), "http://only")
        finally:
            proxy.STATE["upstreams"] = saved


class CandidateRoutingTest(unittest.TestCase):
    def test_training_candidates_use_every_worker_for_any_group_size(self):
        saved = dict(proxy.STATE)
        try:
            proxy.STATE.update(upstreams=[f"http://w{i}" for i in range(8)], group_size=16)
            g16 = [proxy.upstream_for(f"s3g{g}k{k}_abc123") for g in range(2) for k in range(16)]
            self.assertEqual({u: g16.count(u) for u in set(g16)}, {f"http://w{i}": 4 for i in range(8)})
            proxy.STATE.update(group_size=4)
            g4 = [proxy.upstream_for(f"s3g{g}k{k}_abc123") for g in range(8) for k in range(4)]
            self.assertEqual(len(set(g4)), 8)
            self.assertEqual(proxy.upstream_for("s3g1k2_x"), proxy.upstream_for("s3g1k2_x"))
        finally:
            proxy.STATE.clear(); proxy.STATE.update(saved)
