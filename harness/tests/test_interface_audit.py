"""Host interface audit: validator conclusions and every /call attempt, invisible to the student."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from test_server_env_server import FORBIDDEN_KEYS, FakeAdapter, _args, _json, _serve, _spec, _walk_keys

from captain_harness.ports import interface_audit
from captain_harness.ports.broker import ValidatedCapabilityBroker
from captain_harness.ports.capability import CapabilityManifest, CapabilityResult, SideEffect
from captain_harness.server import env_server


def _manifest():
    move = _spec("move_to", SideEffect.ROBOT_MOTION, {"position": {"type": "array", "minItems": 3, "maxItems": 3}}, ("position",))
    return CapabilityManifest(schema_version="t", provider="t", capabilities=(move,), usage_instructions="u")


def test_broker_reports_rule_path_and_is_silent_without_an_observer():
    broker = ValidatedCapabilityBroker(_manifest(), lambda name, args: CapabilityResult.ok({"ok": True}))
    bad = {"position": [1, 2, 3, 4, 5, 6]}
    plain = broker.invoke("move_to", bad)                          # no observer: same result, nothing recorded
    with interface_audit.capture() as events:
        observed = broker.invoke("move_to", bad)
        broker.invoke("move_to", {"position": [1, 2, 3]})
        broker.invoke("teleport", {})
    assert plain == observed and not plain.success
    rejected, accepted, unknown = events[0], events[1], events[-1]
    assert (rejected["validation_source"], rejected["rule_id"], rejected["argument_path"], rejected["outcome"]) == \
        ("broker.arguments_schema", "jsonschema.maxItems", ["position"], "rejected")
    assert accepted["outcome"] == "accepted" and rejected["schema_sha256"] == accepted["schema_sha256"]
    assert unknown["rule_id"] == "unknown_capability" and unknown["validation_phase"] == "input"


def test_every_call_attempt_is_audited_and_bound_to_the_marked_request(tmp_path):
    from test_hidden_object_audit import audit_env

    adapter = FakeAdapter(tmp_path, reject_first_finish=False)
    private_env = audit_env([["on", "object_1", "plate"]], objects=("object_1", "plate"))
    state = env_server.EpisodeServerState(_args(tmp_path), builder=lambda args, output_dir, log:
        env_server.BuiltEpisode(env=private_env, adapter=adapter, task="t", initial=adapter.reset(args.seed)))
    state.build()
    server, url = _serve(state)
    try:
        before = _json(url + "/call", "POST", {"name": "ping", "arguments": {}})          # before any mark: no request
        _json(url + "/mark", "POST", {"request_index": 1, "token": state.driver_token})
        found = set()
        _walk_keys(_json(url + "/call", "POST", {"name": "move_to", "arguments": {"position": [0, 0, 0]}}), found)
        _walk_keys(_json(url + "/call", "POST", {"name": "teleport", "arguments": {}}), found)
        _walk_keys(_json(url + "/call", "POST", {"name": "ping", "arguments": [1]}), found)
        _json(url + "/call", "POST", {"name": "move_to", "arguments": {"position": [0, 0, 0]}})
        _json(url + "/call", "POST", {"name": "move_to", "arguments": {"position": [0, 0, 0]}})   # action budget (2) exhausted
        request = urllib.request.Request(url + "/call", data=json.dumps({"name": 7}).encode(), method="POST",
                                         headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(request, timeout=10)
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
        assert "attempt_seq" not in found and "interface" not in json.dumps(before)          # never in a student response
        rows = [json.loads(l) for l in state.interface_audit_path.read_text().splitlines()]
        got = [(r["attempt_seq"], r["driver_request_index"], r["tool"], r["early_return"], r["public_call_index"]) for r in rows]
        assert got == [(1, None, "ping", None, 1), (2, 1, "move_to", None, 2), (3, 1, "teleport", "unknown_capability", None),
                       (4, 1, "ping", "arguments_not_object", None), (5, 1, "move_to", None, 3),
                       (6, 1, "move_to", "action_budget_exhausted", None), (7, 1, None, "transport_invalid", None)]
        assert state.call_index == 3                                                      # public numbering unchanged
        assert [json.loads(l)["call_index"] for l in state.calls_path.read_text().splitlines()] == [1, 2, 3]
        assert all(r["schema_version"] == "interface-audit/v1" for r in rows)
    finally:
        server.shutdown()
        state.worker.shutdown(wait=True)
