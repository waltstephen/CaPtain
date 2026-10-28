"""Explicit runtime contracts cannot disappear between launch and an episode."""
import asyncio
import sys

import pytest

import env_server as env


def state(tmp_path, contract=None):
    return {"harness": tmp_path, "proxy": "http://proxy", "model_name": "policy",
            "sam3_url": "http://perception", "molmo_url": "http://grounding",
            "max_concurrency": 4, "max_training_requests": 4,
            "episode": {"profile": "captain", "context_window": 65536,
                        "max_output_tokens": 4096, "max_images": 12,
                        "max_requests": 40, "wall_time": 720, "grace_s": 60,
                        "max_actions": 150, "max_tool_calls": 600,
                        "runtime_tool_contract": contract}}


@pytest.mark.parametrize("contract", [None, "code-mode-runtime/v2"])
def test_episode_command_preserves_explicit_runtime_selection(tmp_path, monkeypatch, contract):
    monkeypatch.setattr(env, "STATE", state(tmp_path, contract))
    command = env.episode_command(suite="libero_object", task_id=0, seed=1,
                                  sid="test", out_dir=tmp_path / "episode", egl_device=1)
    if contract:
        assert command[command.index("--runtime-tool-contract") + 1] == contract
    else:
        assert "--runtime-tool-contract" not in command


def test_health_reports_private_preflight_and_required_audit_schema(tmp_path, monkeypatch):
    cfg = state(tmp_path, "code-mode-runtime/v2")
    cfg["environment_preflight"] = {"ok": True, "fingerprint": "test"}
    monkeypatch.setattr(env, "STATE", cfg)
    monkeypatch.setattr(env, "REWARD_VERSION", "v12b")
    result = asyncio.run(env.health())
    assert result["goal_audit_schema"] == "hidden-object-audit/v2"
    assert result["tool_audit_schema"] == "tool-execution-audit/v1"
    assert result["environment_preflight"] == cfg["environment_preflight"]


def test_new_reward_cannot_start_without_its_runtime_contract(monkeypatch):
    monkeypatch.setattr(env, "REWARD_VERSION", "v12b")
    monkeypatch.setattr(sys, "argv", ["env_server.py"])
    with pytest.raises(SystemExit) as exc:
        env.main()
    assert exc.value.code == 2


def test_new_runtime_cannot_start_without_preflight(monkeypatch):
    monkeypatch.delenv("FRESH_ENV_PREFLIGHT_PATH", raising=False)
    monkeypatch.setattr(sys, "argv", ["env_server.py", "--runtime-tool-contract", "code-mode-runtime/v2"])
    with pytest.raises(SystemExit) as exc:
        env.main()
    assert exc.value.code == 2


def test_segment_commands_pin_the_student_python_hide_teacher_data_and_request_a_clean_handoff(tmp_path, monkeypatch):
    cfg = state(tmp_path, "python-runtime/v1")
    cfg["root"] = tmp_path / "rollouts"
    cfg["episode"].update(student_python="/envs/student/bin/python", sandbox_hide=["/data/teacher"])
    monkeypatch.setattr(env, "STATE", cfg)
    command = env.episode_command(suite="libero_object", task_id=0, seed=1, sid="t", out_dir=tmp_path / "e", egl_device=1)
    assert command[command.index("--student-python") + 1] == "/envs/student/bin/python"
    assert [command[i + 1] for i, a in enumerate(command) if a == "--sandbox-hide"] == [str(tmp_path / "rollouts"), "/data/teacher"]
    captured = {}

    class Stop(Exception):
        pass

    def fake_popen(command, **kw):
        captured["command"] = command
        raise Stop

    source = tmp_path / "teacher_episode"
    (source / "driver").mkdir(parents=True)
    (source / "driver" / "responses.jsonl").write_text("".join(f'{{"request": {i}, "items": []}}\n' for i in (1, 2, 3)))
    monkeypatch.setattr(env, "_launch_gate", lambda: 0)
    monkeypatch.setattr(env.subprocess, "Popen", fake_popen)
    entry = {"K": 2, "horizon": 38, "handoff": "clean", "source_dir": str(source), "source_responses": str(source / "driver" / "responses.jsonl")}
    with pytest.raises(Stop):
        env.run_episode({"suite": "libero_object", "task_id": 0, "seed": 1, "sid": "s", "step": 1}, segment=entry)
    command = captured["command"]
    assert command[command.index("--max-requests") + 1] == "40" and command[command.index("--handoff") + 1] == "clean"
    assert str(source) in [command[i + 1] for i, a in enumerate(command) if a == "--sandbox-hide"]


def test_complete_proxy_record_is_stored_gzipped_and_atomically(tmp_path):
    import gzip
    import json

    rec = {"sid": "s1", "requests": [{"request_index": 1, "prompt_token_ids": [1, 2], "token_ids": [3]}], "images": {"h": "data:image/png;base64,AA"}}
    path = env.store_raw_record(rec, tmp_path, 7, "s1")
    assert path == tmp_path / "step-7" / "s1.json.gz" and json.loads(gzip.open(path, "rt").read()) == rec
    assert not list(path.parent.glob(".*.tmp"))
