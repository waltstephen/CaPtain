#!/usr/bin/env python3
"""Per-request history alignment (rl/chat_tokens.py) + identity-based masking / repair samples (rl/train.py, rollout service
rl/env_server.py). Pure token-id tests: no model, no images. Run: python -m pytest -q rl/test_history_alignment.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chat_tokens import IM_END, IM_START, NEWLINE, NLNL, ROLE_ASSISTANT, THINK_CLOSE, THINK_OPEN, align_history, assistant_blocks, assistant_mask, with_im_end  # noqa: E402

USER = 872
GEN = [IM_START, ROLE_ASSISTANT, NEWLINE, THINK_OPEN, NLNL, THINK_CLOSE, NLNL]   # generation prompt, thinking off


def user_turn(k: int) -> list[int]:
    return [IM_START, USER, NEWLINE, 5000 + k, 5001 + k, IM_END, NEWLINE]


def render(completion: list[int]) -> list[int]:
    """How the driver re-renders an earlier turn into the next prompt (the same content unless it was re-parsed)."""
    return GEN + [t for t in completion if t != IM_END] + [IM_END, NEWLINE]


def make_requests(completions: list[list[int]], rerender: dict[int, list[int]] | None = None) -> list[dict]:
    """Proxy record: request k's prompt = system + all earlier turns (re-rendered) + user k + generation prompt."""
    rerender = rerender or {}
    reqs = []
    for k, comp in enumerate(completions):
        prompt = [IM_START, 999, NEWLINE, 7, 7, IM_END, NEWLINE]
        for j in range(k):
            prompt += user_turn(j) + render(rerender.get(j, completions[j]))
        prompt += user_turn(k) + GEN
        reqs.append({"prompt_token_ids": prompt, "token_ids": list(comp), "finish_reason": "stop", "image_hashes": []})
    return reqs


def seq(req: dict) -> list[int]:
    return with_im_end(list(req["prompt_token_ids"]) + list(req["token_ids"]))


def test_mask_excludes_think_prefix_and_history_maps_one_block_per_request():
    comps = [[11, 12, IM_END], [21, 22, 23, IM_END], [31, IM_END]]
    reqs = make_requests(comps)
    ids = seq(reqs[2])
    blocks = assistant_blocks(ids, assistant_mask(ids))
    assert len(blocks) == 3
    assert [ids[a:b] for a, b in blocks] == comps
    history, ok = align_history(ids, 2, [0, 1, 2], reqs)
    assert ok and history == [{"request": 1, "exact": True, "block_len": 3, "completion_len": 3}, {"request": 2, "exact": True, "block_len": 4, "completion_len": 4}]


def test_rerendered_turn_is_flagged_not_exact():
    comps = [[11, 12, IM_END], [21, 22, 23, 24, 25, IM_END], [31, IM_END]]
    canonical = [40, 41, IM_END]          # the driver could not parse request 2's output and rendered the canonical empty call
    reqs = make_requests(comps, rerender={1: canonical})
    history, ok = align_history(seq(reqs[2]), 2, [0, 1, 2], reqs)
    assert ok
    assert history[0]["exact"] is True and history[1] == {"request": 2, "exact": False, "block_len": 3, "completion_len": 6}


def test_text_equal_retokenisation_is_not_a_sampled_token_match():
    class Tok:
        def decode(self, ids, skip_special_tokens=False):
            return "".join({1112: "ab", 11: "a", 12: "b", IM_END: "<e>", 21: "c"}.get(t, "?") for t in ids)

    comps = [[11, 12, IM_END], [21, IM_END]]
    reqs = make_requests(comps, rerender={0: [1112, IM_END]})
    history, _ = align_history(seq(reqs[1]), 1, [0, 1], reqs)
    assert history[0]["exact"] is False
    history, _ = align_history(seq(reqs[1]), 1, [0, 1], reqs, tokenizer=Tok())
    assert history[0]["exact"] is False and not history[0].get("text_equal")


def test_failed_requests_are_skipped_in_the_backward_mapping():
    comps = [[11, IM_END], [21, IM_END], [31, IM_END]]
    reqs = make_requests(comps)
    reqs.insert(1, {"status": 500, "error": "boom"})     # a failed proxy request produced no assistant turn
    history, ok = align_history(seq(reqs[3]), 3, [0, 2, 3], reqs)
    assert ok and [h["request"] for h in history] == [1, 3] and all(h["exact"] for h in history)


def test_identical_repeats_in_one_segment_keep_both_gradients_and_repairs_are_built():
    import types
    import train as T

    comps = [[11, 12, IM_END], [11, 12, IM_END], [55, 56, 57, IM_END], [31, IM_END]]   # requests 1 and 2 sampled the same text
    canonical = [40, 41, IM_END]
    reqs = make_requests(comps, rerender={2: canonical})                                # request 3 was re-rendered
    last = 3
    ids = seq(reqs[last])
    history, ok = align_history(ids, last, [0, 1, 2, 3], reqs)
    assert [h["exact"] for h in history] == [True, True, False]
    sample = {"segments": [{"request": last + 1, "prompt_token_ids": reqs[last]["prompt_token_ids"], "token_ids": reqs[last]["token_ids"], "finish_reason": "stop",
                            "images": [], "history": history}],
              "repairs": [{"request": 3, "prompt_token_ids": reqs[2]["prompt_token_ids"], "token_ids": reqs[2]["token_ids"], "finish_reason": "stop", "images": [], "weight": 2.0}]}

    class Tok:
        def decode(self, ids, skip_special_tokens=False):
            return " ".join(map(str, ids))

    processor = types.SimpleNamespace(tokenizer=Tok(), image_processor=None)
    stats: dict = {}
    out = T.build_segment_samples(sample, processor, max_seq_tokens=10_000, image_pad_id=-1, hard_max_tokens=None, path="x", stats=stats)
    assert [o["segment"] for o in out] == [4, "repair:3"]
    seg, rep = out
    trained = [ids[a:b] for a, b in assistant_blocks(ids, seg["mask"].tolist())]
    assert trained == [[11, 12, IM_END], [11, 12, IM_END], [31, IM_END]], trained      # both repeats trained, re-rendered block masked
    rep_ids = seq(reqs[2])
    assert [rep_ids[a:b] for a, b in assistant_blocks(rep_ids, rep["mask"].tolist())] == [[55, 56, 57, IM_END]]   # repair: final block only
    assert rep["weight"] == 2.0 and stats == {"history_rerendered": 1, "repairs": 1}

    # legacy file (no history): cross-segment dedup by content still applies, same-segment repeats are kept
    legacy = {"segments": [dict(sample["segments"][0], history=None)]}
    out2 = T.build_segment_samples(legacy, processor, max_seq_tokens=10_000, image_pad_id=-1, hard_max_tokens=None, path="x")
    trained2 = [ids[a:b] for a, b in assistant_blocks(ids, out2[0]["mask"].tolist())]
    assert trained2 == [[11, 12, IM_END], [11, 12, IM_END], [40, 41, IM_END], [31, IM_END]]


def test_text_equal_retokenisation_uses_the_original_completion_repair():
    import types
    import train as T

    comps = [[11, 12, IM_END], [31, IM_END]]
    reqs = make_requests(comps, rerender={0: [1112, IM_END]})   # same text, merged into one token by the re-tokenisation
    ids = seq(reqs[1])
    history, _ = align_history(ids, 1, [0, 1], reqs)
    assert history[0]["exact"] is False

    class Tok:
        def decode(self, ids, skip_special_tokens=False):
            return "".join("ab" if t == 1112 else {11: "a", 12: "b", IM_END: "<e>"}.get(t, "?") for t in ids)

    sample = {"segments": [{"request": 2, "prompt_token_ids": reqs[1]["prompt_token_ids"], "token_ids": reqs[1]["token_ids"], "finish_reason": "stop", "images": [], "history": history}],
              "repairs": [{"request": 1, "prompt_token_ids": reqs[0]["prompt_token_ids"], "token_ids": reqs[0]["token_ids"], "finish_reason": "stop", "images": [], "weight": 1.0}]}
    out = T.build_segment_samples(sample, types.SimpleNamespace(tokenizer=Tok()), max_seq_tokens=10_000, image_pad_id=-1, hard_max_tokens=None, path="x")
    assert [o["segment"] for o in out] == [2, "repair:1"]
    assert [ids[a:b] for a, b in assistant_blocks(ids, out[0]["mask"].tolist())] == [[31, IM_END]]
    repair_ids = out[1]["input_ids"].tolist()
    assert [repair_ids[a:b] for a, b in assistant_blocks(repair_ids, out[1]["mask"].tolist())] == [[11, 12, IM_END]]


def test_env_server_build_segments_history_and_repairs():
    import types
    for mod in ("fastapi", "uvicorn"):      # the rollout service's web deps are not needed for the pure segment builder
        try:
            __import__(mod)
        except ImportError:
            stub = types.ModuleType(mod)
            stub.FastAPI = lambda *a, **k: types.SimpleNamespace(post=lambda *a, **k: (lambda f: f), get=lambda *a, **k: (lambda f: f))
            stub.Request = object
            sys.modules[mod] = stub
    import env_server as E

    comps = [[11, IM_END], [21, 22, IM_END], [31, IM_END], [41, IM_END], [51, IM_END]]
    reqs = make_requests(comps, rerender={1: [90, IM_END], 3: [91, IM_END]})   # requests 2 and 4 re-rendered
    rec = {"requests": reqs, "images": {}}
    segs, repairs, diag = E.build_segments(rec, max_segments=6, compactions_per_request=None, max_repairs=1)
    assert len(segs) == 1 and segs[0]["request"] == 5
    assert [(h["request"], h["exact"]) for h in segs[0]["history"]] == [(1, True), (2, False), (3, True), (4, False)]
    assert diag["repair_candidates"] == 2 and diag["repairs_stored"] == 1 and diag["repair_weight"] == 2.0 and diag["alignment_ok"]
    assert repairs[0]["request"] in (2, 4) and repairs[0]["weight"] == 2.0 and repairs[0]["token_ids"] == comps[repairs[0]["request"] - 1]
    segs, repairs, diag = E.build_segments(rec, max_segments=6, compactions_per_request=None, max_repairs=-1)
    assert sorted(r["request"] for r in repairs) == [2, 4] and diag["repair_weight"] == 1.0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
