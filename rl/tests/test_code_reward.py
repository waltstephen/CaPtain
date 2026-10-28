"""python_exec_cost_v1: classification from host audit records, sample tensors, trainer loss terms and wiring."""
from __future__ import annotations

import json

import pytest
import torch

import code_reward as CR
import raw_request_training as RRT
import train as T
import env_server as env
from test_sc_is_credit import Core


def cell(req, status="ok", exc=None, tid=None, env_error=None, error_class=None):
    return {"type": "tool_end", "tool_name": "python", "request_index": req, "tool_call_id": tid or f"t{req}",
            "result": {"status": status, "exception": exc}, "environment_error": env_error, "error_class": error_class}


def episode(tmp_path, cells, calls_by_request, n_requests, response_ids=None):
    """Requests 1..n, one pre-tool mark each; calls_by_request: request -> number of host calls it ran."""
    (tmp_path / "driver").mkdir(parents=True, exist_ok=True)
    (tmp_path / "driver" / "tool_audit.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cells))
    rid = response_ids or {i: f"r{i}" for i in range(1, n_requests + 1)}
    (tmp_path / "driver" / "requests.jsonl").write_text("".join(json.dumps({"index": i, "replayed": False, "response_id": rid.get(i)}) + "\n"
                                                               for i in range(1, n_requests + 1)))
    marks, calls, ci = [], [], 0
    for req in range(1, n_requests + 1):
        marks.append({"request_index": req, "call_index": ci})
        for _ in range(calls_by_request.get(req, 0)):
            ci += 1
            calls.append({"call_index": ci, "name": "observe", "executed": True, "success": True})
    (tmp_path / "hidden_request_marks.jsonl").write_text("".join(json.dumps(m) + "\n" for m in marks))
    (tmp_path / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
    return tmp_path


TB = 'Traceback (most recent call last):\n  File "<cell 2>", line 3, in <module>\n    x = q[0]\n'


def test_categories_follow_phase_effect_and_attribution(tmp_path):
    cells = [cell(1), cell(2, "error", TB + "NameError: name 'q' is not defined"),
             cell(3, "error", '  File "<cell 3>", line 2\n    a=1 b=2\n      ^\nSyntaxError: invalid syntax'),
             cell(4, "error", TB + "KeyError: 'value'"),
             cell(5, "error", "IndentationError: unexpected indent"),
             cell(6, "error", None, env_error={"kind": "worker"}, error_class="environment"),
             cell(7, "timeout", None), cell(8, "error", TB + "CellTimeout: cell exceeded 600 s"),
             cell(10, "error", TB + "RuntimeError: requested_cartesian_path_self_collision")]
    labels = CR.episode_request_costs(episode(tmp_path, cells, {1: 2, 4: 1, 5: 1, 10: 1}, 10))
    got = {r: (l["category"], l["cost"], l["scored"], l["no_effect"]) for r, l in labels.items()}
    assert got[1] == ("ok", 0.0, True, False)
    assert got[2] == ("runtime_invalid", 0.5, True, True)                  # raised, no host call during the request
    assert got[3] == ("syntax_invalid", 1.0, True, True)
    assert got[4] == ("mixed_effect_error", 0.25, True, False)             # raised after a host call ran: effect kept
    assert got[5] == ("syntax_invalid", 1.0, True, True)                   # compile errors never started, even with calls
    assert got[6][0] == "environment" and not got[6][2]
    assert got[7][0] == "unknown" and not got[7][2]
    assert got[8][0] == "unknown" and not got[8][2]                       # interrupts / timeouts are not code errors
    assert got[9] == ("no_cell", 0.0, True, False)
    assert got[10] == ("physical_failure", 0.0, True, False)               # a planning failure is not a code error


def test_multi_cell_takes_the_worst_and_duplicates_count_once(tmp_path):
    cells = [cell(1, tid="a"), cell(1, "error", TB + "TypeError: bad", tid="b"), cell(1, "error", TB + "TypeError: bad", tid="b")]
    labels = CR.episode_request_costs(episode(tmp_path, cells, {}, 1))
    assert labels[1]["category"] == "runtime_invalid" and labels[1]["cells"] == 2


def test_proxy_join_by_response_id_not_index(tmp_path):
    cells = [cell(1), cell(2, "error", TB + "NameError: x")]
    ep = episode(tmp_path, cells, {}, 2, response_ids={1: "A", 2: "B"})
    out = CR.proxy_request_costs(ep, [{"request_index": 1, "response_id": "B"}, {"request_index": 2, "response_id": "A"},
                                      {"request_index": 3, "response_id": "C"}])
    assert out["labels"][1]["category"] == "runtime_invalid" and out["labels"][2]["category"] == "ok"
    assert out["labels"][3] == {"category": "unmapped", "cost": 0.0, "scored": False, "no_effect": False} and out["unmapped"] == 1


def test_code_fields_reach_samples_and_packed_chains():
    from test_prefix_packing import raw
    r = raw(select=(1, 2), prompts={1: ([1, 2, 3], [4, 5]), 2: ([1, 2, 3, 4, 5, 6], [7])})
    r["training_requests"][0].update(code_cost=1.0, code_no_effect=True)
    r["training_requests"][1].update(code_cost=0.0, code_no_effect=False)
    plain, _ = RRT.build_request_samples(r, None, image_pad_id=-1, hard_max_tokens=100)
    assert plain[0]["token_code"].tolist() == [0.5, 0.5] and plain[0]["token_pure_error"].tolist() == [1.0, 1.0]
    packed, _ = RRT.build_request_samples(r, None, image_pad_id=-1, hard_max_tokens=100, pack_prefixes=True)
    assert packed[0]["token_code"].tolist() == [0.5, 0.5, 0.0] and packed[0]["token_pure_error"].tolist() == [1.0, 1.0, 0.0]
    r["training_requests"][1].pop("code_cost")
    with pytest.raises(RRT.InvalidRollout):
        RRT.build_request_samples(r, None, image_pad_id=-1, hard_max_tokens=100, pack_prefixes=True)
    r["training_requests"][1].update(code_cost=1.5)
    with pytest.raises(RRT.InvalidRollout):
        RRT.build_request_samples(r, None, image_pad_id=-1, hard_max_tokens=100)


def _loss(s, **kw):
    torch.manual_seed(0)
    logits = torch.randn(5, requires_grad=True)
    inputs, mask = {"input_ids": torch.tensor([[0, 2, 3]])}, torch.tensor([[False, True, True]])
    loss, _, _ = T.sample_loss(None, Core(logits), torch.nn.Identity(), inputs, mask, s, score_centering=False, kl_beta=0.0,
                               full_completion_tokens=2, world_size=1, **kw)
    grad = torch.autograd.grad(loss, logits)[0]
    return loss, grad, logits


def test_off_or_zero_weight_is_the_old_loss():
    base = {"advantage": 0.7, "weight": 1.0}
    with_fields = dict(base, token_code=torch.tensor([0.5, 0.5]), token_pure_error=torch.ones(2))
    old, g_old, _ = _loss(base)
    off, g_off, _ = _loss(with_fields)                                       # code_weight defaults to 0
    assert torch.allclose(old, off) and torch.allclose(g_old, g_off)


def test_pure_error_loses_only_positive_task_credit():
    pure = {"token_code": torch.zeros(2), "token_pure_error": torch.ones(2), "weight": 1.0}
    pos, _, _ = _loss(dict(pure, advantage=0.7), code_weight=0.25, code_norm=40.0)
    assert float(pos) == 0.0                                                 # no task term, zero code cost
    neg, _, _ = _loss(dict(pure, advantage=-0.7), code_weight=0.25, code_norm=40.0)
    ref, _, _ = _loss({"advantage": -0.7, "weight": 1.0})
    assert torch.allclose(neg, ref)                                          # negative advantage still applies


def test_code_term_lowers_the_erroring_tokens_probability():
    s = {"advantage": 0.0, "weight": 2.0, "token_code": torch.tensor([0.5, 0.5]), "token_pure_error": torch.zeros(2)}
    loss, grad, logits = _loss(s, code_weight=0.25, code_norm=40.0)
    lp = torch.log_softmax(logits.detach(), -1)
    assert torch.isclose(loss, 0.25 * 2.0 * 0.5 * (lp[2] + lp[3]) / 40.0)   # + lambda * (1/p) * c * mean_t logp / H
    probs = torch.softmax(logits.detach(), -1)
    step = -grad                                                             # gradient descent direction
    assert float((step * (torch.eye(5)[2] - probs)).sum()) < 0                # moves logit mass away from the error token


def test_build_training_sample_attaches_code_fields():
    import test_raw_request_samples as TR
    reqs = [TR.request(1, [1, 2], [3]), TR.request(2, [1, 2, 3, 4], [5])]
    code = {1: {"category": "syntax_invalid", "cost": 1.0, "scored": True, "no_effect": True},
            2: {"category": "unknown", "cost": None, "scored": False, "no_effect": False}}
    sample = env.build_training_sample(TR.record(reqs), sid="fixed-session", max_training_requests=-1, code=code)
    got = [(q["code_cost"], q["code_category"], q["code_no_effect"], q["code_scored"]) for q in sample["training_requests"]]
    assert got == [(1.0, "syntax_invalid", True, True), (0.0, "unknown", False, False)]
    sample = env.build_training_sample(TR.record(reqs), sid="fixed-session", max_training_requests=-1)
    assert all("code_cost" not in q for q in sample["training_requests"])


def test_fresh_run_validates_and_passes_the_code_reward():
    import fresh_run as F
    assert F.DEFAULTS["code_reward_mode"] is None and F.DEFAULTS["code_reward_weight"] == 0.0


def attempt(seq, req, tool="preview_motion", events=(), early=None, outcome="failed"):
    return {"schema_version": "interface-audit/v1", "attempt_seq": seq, "driver_request_index": req, "tool": tool,
            "events": list(events), "early_return": early, "outcome": outcome}


REJ = {"validation_source": "motion_argument_validator", "validation_phase": "input", "outcome": "rejected",
       "rule_id": "jsonschema.maxItems", "argument_path": ["arguments", "position"]}
OUT = {"validation_source": "broker.result_schema", "validation_phase": "output", "outcome": "rejected", "rule_id": "jsonschema.type"}


def interface_episode(tmp_path, attempts, cells=None, calls=None, n=4):
    ep = episode(tmp_path, cells if cells is not None else [cell(i) for i in range(1, n + 1)], calls or {}, n)
    (ep / "interface_audit.jsonl").write_text("".join(json.dumps(a) + "\n" for a in attempts))
    return ep


def test_v2_scores_confirmed_contract_rejections_only(tmp_path):
    attempts = [attempt(1, 1, events=[REJ]), attempt(2, 1, events=[REJ]),                     # same request: bounded, once
                attempt(3, 2, tool="move_to", events=[OUT]),                                   # backend output error: not the caller's
                attempt(4, 3, tool="nope", early="unknown_capability"),
                attempt(5, 4, tool="move_to", early="action_budget_exhausted"),                # budget refusal: never scored
                attempt(6, 4, events=[REJ]), attempt(6, 4, events=[REJ]),                      # duplicate attempt line
                attempt(7, None, events=[REJ])]                                                # unattributed: not guessed
    ep = interface_episode(tmp_path, attempts)
    out = CR.proxy_request_costs(ep, [{"request_index": i, "response_id": f"r{i}"} for i in range(1, 5)], CR.VERSION_V2)
    lab = out["labels"]
    assert (lab[1]["cost"], lab[1]["interface_rejections"], lab[1]["delta_cost"]) == (0.5, 2, 0.5)
    assert lab[2]["cost"] == 0.0 and lab[2]["interface_rejections"] == 0
    assert lab[3]["cost"] == 0.5 and lab[3]["interface_rules"] == ["unknown_capability"]
    assert lab[4]["interface_rejections"] == 1 and lab[4]["cost"] == 0.5
    assert out["interface"]["unattributed"] == 1 and out["interface"]["contract_rejections"] == 4
    assert all(not l["no_effect"] for l in lab.values())                                         # v1 mask unchanged (all ok cells)


def test_v2_is_the_max_of_python_and_interface_costs_and_keeps_the_v1_mask(tmp_path):
    cells = [cell(1, "error", TB + "NameError: q"), cell(2, "error", TB + "KeyError: 'x'"), cell(3, "error", 'SyntaxError: bad')]
    ep = interface_episode(tmp_path, [attempt(1, 1, events=[REJ]), attempt(2, 2, events=[REJ]), attempt(3, 2, tool="observe", outcome="ok")],
                           cells=cells, calls={2: 1}, n=3)
    v1 = CR.proxy_request_costs(ep, [{"request_index": i, "response_id": f"r{i}"} for i in (1, 2, 3)])["labels"]
    v2 = CR.proxy_request_costs(ep, [{"request_index": i, "response_id": f"r{i}"} for i in (1, 2, 3)], CR.VERSION_V2)["labels"]
    assert (v1[1]["cost"], v2[1]["cost"], v2[1]["delta_cost"]) == (0.5, 0.5, 0.0)          # already 0.5: no double penalty
    assert (v1[2]["cost"], v2[2]["cost"], v2[2]["delta_cost"]) == (0.25, 0.5, 0.25)         # mixed 0.25 -> 0.5
    assert v2[2]["recovered_within_response"] is True
    assert (v1[3]["cost"], v2[3]["cost"]) == (1.0, 1.0)
    assert [v2[i]["no_effect"] for i in (1, 2, 3)] == [v1[i]["no_effect"] for i in (1, 2, 3)]


def test_v2_refuses_an_episode_without_the_interface_audit(tmp_path):
    ep = episode(tmp_path, [cell(1)], {}, 1)
    assert CR.proxy_request_costs(ep, [{"request_index": 1, "response_id": "r1"}])["labels"][1]["cost"] == 0.0
    with pytest.raises(ValueError, match="interface_audit_missing"):
        CR.proxy_request_costs(ep, [{"request_index": 1, "response_id": "r1"}], CR.VERSION_V2)


def test_advantage_shuffle_permutes_within_groups_only():
    results = [{"group": g} for g in (0, 0, 0, 1, 1, 1)]
    adv = [1.0, -0.5, -0.5, 2.0, -1.0, -1.0]
    out = T.shuffle_within_groups(results, adv, seed=3)
    assert sorted(out[:3]) == sorted(adv[:3]) and sorted(out[3:]) == sorted(adv[3:])
    assert T.shuffle_within_groups(results, adv, seed=3) == out                  # deterministic for a round
