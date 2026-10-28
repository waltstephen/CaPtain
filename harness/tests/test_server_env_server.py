"""Environment server: routes, budgets, gated finish, hidden-state boundary (fake adapter, real HTTP)."""
from __future__ import annotations

import argparse
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from captain_harness.ports.capability import CapabilityManifest, CapabilityResult, CapabilitySpec, SideEffect
from captain_harness.ports.environment import EnvironmentObservation, HostCompletion, ImageObservation
from captain_harness.server import env_server
from captain_harness.server.episode_launch import build_workspace

FORBIDDEN_KEYS = {"check_success", "object_poses", "contacts", "success_latch", "reward", "host_final_success",
                  "goal_states", "hidden_object_rows", "distance_to_goal", "region_geometry"}
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def _spec(name, side_effect, properties=None, required=()):
    return CapabilitySpec(
        name=name, description=f"{name} capability", timeout_s=5.0, side_effect=side_effect,
        arguments_schema={"type": "object", "properties": properties or {}, "required": list(required), "additionalProperties": False},
        result_schema={"type": "object"}, units={}, coordinate_frames={}, argument_order=tuple(properties or {}),
    )


class FakeAdapter:
    def __init__(self, tmp_path: Path, *, reject_first_finish: bool):
        self.tmp_path = tmp_path
        self.manifest = CapabilityManifest(
            schema_version="test", provider="fake",
            capabilities=(
                _spec("observe", SideEffect.SENSOR_CAPTURE),
                _spec("move_to", SideEffect.ROBOT_MOTION, {"position": {"type": "array"}}, ("position",)),
                _spec("ping", SideEffect.READ_ONLY),
            ),
            usage_instructions="# Fake usage. The host suspends nothing.",
        )
        self.finalize_calls = 0
        self.audit_calls = 0
        self.gate_calls = 0
        self.reject_first_finish = reject_first_finish
        self.observations: dict[str, EnvironmentObservation] = {}

    def _image(self, tag):
        path = self.tmp_path / f"{tag}.png"
        path.write_bytes(PNG)
        return ImageObservation(camera="agentview", rgb_path=str(path), role="head")

    def reset(self, seed):
        return EnvironmentObservation(step=0, images=(self._image("initial"),), public_state={"init_state_index": seed % 50})

    def capabilities(self):
        return self.manifest

    def observe(self):
        return EnvironmentObservation(step=1, images=(self._image("current"),), public_state={})

    def execute(self, name, arguments):
        if name == "observe":
            handle = f"artifact://obs/{len(self.observations)}"
            self.observations[handle] = EnvironmentObservation(step=1, images=(self._image(handle[-1]),), public_state={})
            return CapabilityResult.ok({"observation_handle": handle})
        if name == "move_to":
            return CapabilityResult.ok({"moved": True})
        return CapabilityResult.ok({"pong": True})

    def public_evidence_observations(self, handles):
        return {h: self.observations[h] for h in handles if h in self.observations}

    def audit_hidden_success(self):
        self.audit_calls += 1
        return False

    def completion_gate(self):
        self.gate_calls += 1
        if self.reject_first_finish and self.gate_calls == 1:
            return {"failure": "completion_geometry_unmeasured", "message": "completion gate: measure first", "rejections": 1, "max_rejections": 2}
        return None

    def finalize(self):
        self.finalize_calls += 1
        return HostCompletion(success_ever=False, final_success=True, first_success_step=None)

    def _require_robot(self):
        raise RuntimeError("no robot")


def _args(tmp_path: Path, **overrides) -> argparse.Namespace:
    values = dict(suite="libero_spatial", task_id=0, seed=11, output_dir=tmp_path / "out", images_dir=None,
                  max_actions=2, max_tool_calls=3, wall_time=2400.0, completion_gate=True,
                  audit_hidden_success=True)
    values.update(overrides)
    return argparse.Namespace(**values)


def _serve(state: env_server.EpisodeServerState):
    handler = type("H", (env_server.EpisodeHandler,), {"state": state})
    server = env_server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _json(url, method="GET", payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode())


def _walk_keys(value, found):
    if isinstance(value, dict):
        for k, v in value.items():
            found.add(k)
            _walk_keys(v, found)
    elif isinstance(value, list):
        for v in value:
            _walk_keys(v, found)


@pytest.fixture
def served(tmp_path):
    adapter = FakeAdapter(tmp_path, reject_first_finish=True)

    def builder(args, output_dir, log):
        return env_server.BuiltEpisode(env=None, adapter=adapter, task="put the bowl on the plate", initial=adapter.reset(args.seed))

    state = env_server.EpisodeServerState(_args(tmp_path), builder=builder)
    state.build()
    server, url = _serve(state)
    yield state, adapter, url
    server.shutdown()


def test_manifest_and_workspace(served, tmp_path):
    state, adapter, url = served
    manifest = _json(url + "/manifest")
    assert manifest["task"] == "put the bowl on the plate"
    assert [c["name"] for c in manifest["manifest"]["capabilities"]] == ["observe", "move_to", "ping"]
    assert "def move_to(" in manifest["python_interface"]
    assert manifest["initial_images"][0]["path"].endswith("obs_000_head.png")
    info = {"url": url, "task": manifest["task"], "initial_images": manifest["initial_images"]}
    workspace = build_workspace(state.output_dir, info, manifest, wall_time_s=2400.0, max_actions=150, max_tool_calls=600)
    task_md = (workspace / "TASK.md").read_text()
    assert "**Task: put the bowl on the plate**" in task_md
    assert "images/obs_000_head.png" in task_md
    assert "Standards the final scene is judged by" in task_md
    assert "suspends" not in task_md  # protocol sentence stripped from usage instructions
    assert (workspace / "robot.py").is_file() and (workspace / ".robot_server").read_text().strip() == url


def test_calls_budgets_and_observation_export(served):
    state, adapter, url = served
    observe = _json(url + "/call", "POST", {"name": "observe", "arguments": {}})
    assert observe["success"] and observe["value"]["images"][0]["path"].endswith("obs_001_head.png")
    assert observe["budget"]["tool_calls_used"] == 1
    for _ in range(2):
        _json(url + "/call", "POST", {"name": "ping", "arguments": {}})
    exhausted = _json(url + "/call", "POST", {"name": "ping", "arguments": {}})
    assert not exhausted["success"] and "tool budget exhausted" in exhausted["message"]
    for _ in range(2):
        moved = _json(url + "/call", "POST", {"name": "move_to", "arguments": {"position": [0, 0, 0]}})
        assert moved["success"]
    blocked = _json(url + "/call", "POST", {"name": "move_to", "arguments": {"position": [0, 0, 0]}})
    assert "action budget exhausted" in blocked["message"]
    assert adapter.audit_calls == 2 and not state.audit_path.exists()  # in memory until finish
    unknown = _json(url + "/call", "POST", {"name": "teleport", "arguments": {}})
    assert unknown["failure_category"] == "schema_error"
    health = _json(url + "/health")
    # requests refused because the budget is exhausted are not counted past the cap
    assert health["physical_actions_used"] == 2 and health["tool_calls_used"] == 3


def test_agent_finish_is_gated_then_scored_once(served):
    state, adapter, url = served
    first = _json(url + "/finish", "POST", {"note": "done", "source": "agent"})
    assert first == {**first, "finished": False, "rejected": True, "rejections": 1}
    assert "measure first" in first["message"] and adapter.finalize_calls == 0
    assert not state.finished
    still_ok = _json(url + "/call", "POST", {"name": "ping", "arguments": {}})
    assert still_ok["success"]
    second = _json(url + "/finish", "POST", {"note": "measured", "source": "agent"})
    assert second["finished"] is True and adapter.finalize_calls == 1
    assert set(second) == {"finished", "message", "physical_actions_used", "tool_calls_used", "wall_time_s"}
    again = _json(url + "/finish", "POST", {"note": "again", "source": "agent"})
    assert again["already_finished"] is True and "host_final_success" not in again
    assert adapter.finalize_calls == 1
    result = json.loads((state.output_dir / "result.json").read_text())
    assert result["host_final_success"] is True and result["gate_rejections"] == 1 and result["finish_source"] == "agent"
    attempts = [json.loads(l) for l in state.finish_attempts_path.read_text().splitlines()]
    assert [a["verdict"] for a in attempts] == ["reject", "accepted"]
    assert state.audit_path.exists()
    rejected_call = _json(url + "/call", "POST", {"name": "ping", "arguments": {}})
    assert "finish() was already called" in rejected_call["message"]


def test_driver_finish_bypasses_gate(tmp_path):
    adapter = FakeAdapter(tmp_path, reject_first_finish=True)

    def builder(args, output_dir, log):
        return env_server.BuiltEpisode(env=None, adapter=adapter, task="t", initial=adapter.reset(args.seed))

    state = env_server.EpisodeServerState(_args(tmp_path), builder=builder)
    state.build()
    response = state.finish("wall time", source="driver_timeout")
    assert response["finished"] is True and adapter.gate_calls == 0 and adapter.finalize_calls == 1


def test_served_json_never_carries_hidden_state(served):
    state, adapter, url = served
    found: set[str] = set()
    _walk_keys(_json(url + "/health"), found)
    _walk_keys(_json(url + "/manifest"), found)
    _walk_keys(_json(url + "/call", "POST", {"name": "move_to", "arguments": {"position": [0, 0, 0]}}), found)
    _walk_keys(_json(url + "/finish", "POST", {"note": "", "source": "agent"}), found)
    _walk_keys(_json(url + "/finish", "POST", {"note": "", "source": "agent"}), found)
    assert not (found & FORBIDDEN_KEYS), found & FORBIDDEN_KEYS


def test_history_route_returns_public_records(served):
    state, adapter, url = served
    _json(url + "/call", "POST", {"name": "ping", "arguments": {}})
    _json(url + "/call", "POST", {"name": "move_to", "arguments": {"position": [0, 0, 0]}})
    _json(url + "/call", "POST", {"name": "observe", "arguments": {}})
    h = _json(url + "/history")
    assert h["total_calls"] == 3 and [r["name"] for r in h["records"]] == ["ping", "move_to", "observe"]
    assert h["records"][1]["consumes_action"] is True and h["records"][2]["images"][0].endswith("obs_001_head.png")
    found: set[str] = set()
    _walk_keys(h, found)
    assert not (found & FORBIDDEN_KEYS)
    h2 = _json(url + "/history?start=2&limit=1")
    assert [r["call_index"] for r in h2["records"]] == [3]


def test_private_goal_audit_lifecycle_and_http_boundary(tmp_path):
    from test_hidden_object_audit import audit_env

    adapter = FakeAdapter(tmp_path, reject_first_finish=True)
    private_env = audit_env([["on", "object_1", "plate"]], objects=("object_1", "plate"))
    builder = lambda args, output_dir, log: env_server.BuiltEpisode(
        env=private_env, adapter=adapter, task="t", initial=adapter.reset(args.seed))
    state = env_server.EpisodeServerState(_args(tmp_path), builder=builder)
    state.build()
    server, url = _serve(state)
    try:
        assert state.hidden_object_rows[0]["capability"] == "reset"
        assert state.hidden_object_rows[0]["goals"][0]["destination"]["name"] == "plate"
        assert not state.object_audit_path.exists()
        found = set()
        for route in ("/health", "/manifest", "/history"):
            _walk_keys(_json(url + route), found)
        _walk_keys(_json(url + "/call", "POST", {"name": "move_to", "arguments": {"position": [0, 0, 0]}}), found)
        assert [row["capability"] for row in state.hidden_object_rows] == ["reset", "move_to"]
        rejected = _json(url + "/finish", "POST", {"source": "agent"})
        assert rejected["rejected"] and not state.object_audit_path.exists()
        assert len(state.hidden_object_rows) == 2
        private_env._eval_predicate = lambda goal: True
        _walk_keys(_json(url + "/finish", "POST", {"source": "agent"}), found)
        _walk_keys(_json(url + "/history"), found)
        assert not found & FORBIDDEN_KEYS
        rows = [json.loads(line) for line in state.object_audit_path.read_text().splitlines()]
        assert [row["capability"] for row in rows] == ["reset", "move_to", "finish"]
        assert rows[-1]["call_index"] == rows[-2]["call_index"] == 1
        assert rows[-1]["goal_states"]["goal_0"]["satisfied"] is True
        assert rows[0]["goal_states"]["goal_0"]["satisfied"] is False
    finally:
        server.shutdown()
        state.worker.shutdown(wait=True)


def test_no_action_episode_has_initial_and_final_private_goal_snapshots(tmp_path):
    from test_hidden_object_audit import audit_env

    adapter = FakeAdapter(tmp_path, reject_first_finish=False)
    private_env = audit_env([["on", "object_1", "plate"]], objects=("object_1", "plate"))
    state = env_server.EpisodeServerState(_args(tmp_path), builder=lambda args, output_dir, log:
        env_server.BuiltEpisode(env=private_env, adapter=adapter, task="t", initial=adapter.reset(args.seed)))
    try:
        state.build()
        state.finish("done", source="driver")
        rows = [json.loads(line) for line in state.object_audit_path.read_text().splitlines()]
        assert [row["capability"] for row in rows] == ["reset", "finish"]
        assert [row["call_index"] for row in rows] == [0, 0]
    finally:
        state.worker.shutdown(wait=True)


def test_request_marks_are_token_gated_private_and_written_at_finish(tmp_path):
    from test_hidden_object_audit import audit_env

    adapter = FakeAdapter(tmp_path, reject_first_finish=False)
    private_env = audit_env([["on", "object_1", "plate"]], objects=("object_1", "plate"))
    state = env_server.EpisodeServerState(_args(tmp_path), builder=lambda args, output_dir, log:
        env_server.BuiltEpisode(env=private_env, adapter=adapter, task="t", initial=adapter.reset(args.seed)))
    state.build()
    server, url = _serve(state)
    try:
        request = urllib.request.Request(url + "/mark", data=json.dumps({"request_index": 1, "token": "wrong"}).encode(), method="POST",
                                         headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as denied:
            urllib.request.urlopen(request, timeout=10)
        assert denied.value.code == 403
        found = set()
        _walk_keys(_json(url + "/mark", "POST", {"request_index": 1, "token": state.driver_token}), found)
        _json(url + "/call", "POST", {"name": "move_to", "arguments": {"position": [0, 0, 0]}})
        _walk_keys(_json(url + "/mark", "POST", {"request_index": 2, "token": state.driver_token}), found)
        _walk_keys(_json(url + "/health"), found)
        assert not found & FORBIDDEN_KEYS and "driver_token" not in json.dumps(_json(url + "/health"))
        assert state.action_requests == 1 and state.tool_requests == 0  # marks are not calls
        assert not state.marks_path.exists()
        _json(url + "/finish", "POST", {"source": "driver"})
        marks = [json.loads(line) for line in state.marks_path.read_text().splitlines()]
        assert [(m["request_index"], m["call_index"], m["action_index"]) for m in marks] == [(1, 0, 0), (2, 1, 1)]
        assert marks[0]["schema_version"] == "hidden-request-mark/v1" and "goal_0" in marks[0]["goal_states"]
        assert _json(url + "/mark", "POST", {"request_index": 3, "token": state.driver_token}) == {"ok": False, "reason": "finished"}
    finally:
        server.shutdown()
        state.worker.shutdown(wait=True)


def test_receipts_summarize_calls_only_when_enabled(tmp_path):
    adapter = FakeAdapter(tmp_path, reject_first_finish=False)
    builder = lambda args, output_dir, log: env_server.BuiltEpisode(env=None, adapter=adapter, task="t", initial=adapter.reset(args.seed))
    plain = env_server.EpisodeServerState(_args(tmp_path / "a"), builder=builder)
    plain.build()
    assert "receipt" not in plain.call("move_to", {"position": [0, 0, 0]})
    state = env_server.EpisodeServerState(_args(tmp_path / "b", receipts=True), builder=builder)
    state.build()
    moved = state.call("move_to", {"position": [0, 0, 0]})
    assert moved["receipt"].startswith("move_to: ok")
    observed = state.call("observe", {})
    assert observed["receipt"].startswith("observe: ok") and "images=obs_001_head.png" in observed["receipt"]
    unknown = state.call("teleport", {})
    assert "receipt" in unknown or unknown["failure_category"] == "schema_error"
    physical = {"success": False, "executed": True, "failure_category": "contact_detected", "message": "stopped   on contact",
                "value": {"stopped_reason": "contact", "executed_distance_m": 0.0412, "robot_before": {"gripper_open_fraction": 1.0},
                          "robot_after": {"gripper_open_fraction": 0.12}, "trajectory": list(range(500))}}
    receipt = env_server.call_receipt("move_to", physical, True)
    assert receipt.startswith("move_to: FAILED contact_detected (executed); stopped on contact;")
    assert "executed_distance_m=0.0412" in receipt and "gripper_open_fraction=1->0.12" in receipt and "trajectory" not in receipt
    assert len(env_server.call_receipt("mark_point", {"success": True, "message": "x" * 1000, "value": {}}, False)) <= env_server.RECEIPT_LIMIT


def test_clean_handoff_renders_frames_hides_earlier_calls_and_spends_nothing(tmp_path):
    from test_hidden_object_audit import audit_env

    adapter = FakeAdapter(tmp_path, reject_first_finish=False)
    private_env = audit_env([["on", "object_1", "plate"]], objects=("object_1", "plate"))
    state = env_server.EpisodeServerState(_args(tmp_path, max_tool_calls=5), builder=lambda args, output_dir, log:
        env_server.BuiltEpisode(env=private_env, adapter=adapter, task="t", initial=adapter.reset(args.seed)))
    state.build()
    server, url = _serve(state)
    try:
        _json(url + "/call", "POST", {"name": "ping", "arguments": {}})
        _json(url + "/call", "POST", {"name": "move_to", "arguments": {"position": [0, 0, 0]}})
        request = urllib.request.Request(url + "/handoff", data=json.dumps({"token": "wrong"}).encode(), method="POST",
                                         headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as denied:
            urllib.request.urlopen(request, timeout=10)
        assert denied.value.code == 403
        spent = ("physical_actions_used", "physical_actions_remaining", "tool_calls_used", "tool_calls_remaining")
        before = {k: state.counters()[k] for k in spent}
        reply = _json(url + "/handoff", "POST", {"token": state.driver_token})
        assert reply["ok"] and reply["call_index"] == 2 and len(reply["images"]) == 1
        assert Path(reply["images"][0]["path"]).is_file()
        # a render, not a call: the budget the next actor sees is exactly what the earlier actor left
        assert {k: reply["budget"][k] for k in spent} == before and (state.call_index, state.action_requests, state.tool_requests) == (2, 1, 1)
        assert reply["budget"]["physical_actions_remaining"] == 1
        assert _json(url + "/history")["total_calls"] == 0
        _json(url + "/call", "POST", {"name": "ping", "arguments": {}})
        h = _json(url + "/history")
        assert h["total_calls"] == 1 and [r["call_index"] for r in h["records"]] == [3]
        _json(url + "/finish", "POST", {"source": "driver"})
        result = json.loads((state.output_dir / "result.json").read_text())
        assert [(r["call_index"], r["action_index"]) for r in result["handoffs"]] == [(2, 1)]
        assert _json(url + "/handoff", "POST", {"token": state.driver_token}) == {"ok": False, "reason": "finished"}
    finally:
        server.shutdown()
        state.worker.shutdown(wait=True)
