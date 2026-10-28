"""Host-side tool accounting without dependence on printed or truncated results."""
from __future__ import annotations

import copy
import importlib
import json
import os

import pytest

import reward as R
from test_reward_v12a import episode, state
from toolchain_preflight import REQUIRED_EXECUTABLES, fingerprint


@pytest.fixture(autouse=True)
def reward_version():
    os.environ["RL_REWARD_VERSION"] = "v12b"
    importlib.reload(R)


def preflight_receipt():
    receipt = {"schema_version": "harness-toolchain-preflight/v1", "runtime_tool_contract": "code-mode-runtime/v2",
               "hostname": "test-node", "ok": True, "required_executables": list(REQUIRED_EXECUTABLES),
               "executables": {name: {"ok": True, "exit_code": 0, "resolved": f"/usr/bin/{name}", "version": "test"}
                               for name in REQUIRED_EXECUTABLES},
               "sandbox": {"enabled": True}, "workspace_import": {"ok": True}, "path_environment": {}}
    receipt["fingerprint"] = fingerprint(receipt)
    return receipt


def environment_error(executable="rg"):
    return {"schema_version": "harness-environment-error/v1", "code": "required_executable_unavailable",
            "executable": executable, "source": "tool_runtime", "verified": True,
            "details": {"exit_code": 127, "ok": False}}


_DEFAULT = object()


class AuditEpisode:
    def __init__(self, path, **kwargs):
        self.path, _ = episode(path, **kwargs)
        self.rows = [{"type": "header", "schema_version": R.TOOL_AUDIT_SCHEMA,
                      "contract_version": R.TOOL_RUNTIME_CONTRACT, "contract_sha256": "a" * 64,
                      "environment_preflight": preflight_receipt()}]
        self.items = []
        self.context = None
        self.tool_count = 0

    def start(self, *, name="exec", kind="custom_tool_call"):
        number = sum(row["type"] == "outer_start" for row in self.rows)
        self.context = {"outer_call_id": f"c{number}", "request_index": number + 1,
                        "tool_name": name, "call_kind": kind}
        self.rows.append({"type": "outer_start", **self.context})
        self.items.append({"type": kind, "name": name, "call_id": self.context["outer_call_id"],
                           "input" if kind == "custom_tool_call" else "arguments": "await tools.exec_command({cmd:'false'})"})

    def tool(self, cmd="false", *, code=1, session=None, name="exec_command", args=_DEFAULT,
             command_id=_DEFAULT, error=_DEFAULT, output="", environment=None):
        key = f"t{self.tool_count}"
        self.tool_count += 1
        args = {"cmd": cmd} if args is _DEFAULT else args
        if command_id is _DEFAULT:
            command_id = key if name == "exec_command" and (code is not None or session is not None) else None
        error = f"process exited with code {code}" if error is _DEFAULT and code not in (None, 0) else (None if error is _DEFAULT else error)
        event = {"tool_call_id": key, "parent_call_id": self.context["outer_call_id"],
                 "request_index": self.context["request_index"], "tool_name": name, "args": args}
        self.rows.extend([{"type": "tool_start", **event},
                          {"type": "tool_end", **event, "command_id": command_id,
                           "result": {"exit_code": code, "session_id": session, "output": output},
                           "error": error, "error_class": "environment" if environment else ("policy" if error else None),
                           "environment_error": environment}])
        return key

    def end(self, *, error=None, visible=None):
        self.rows.append({"type": "outer_end", **self.context, "status": "failed" if error else "completed", "error": error})
        self.items.append({"type": self.context["call_kind"] + "_output", "call_id": self.context["outer_call_id"],
                           "output": visible if visible is not None else [{"type": "input_text", "text": "Script failed" if error else "Script completed"}]})
        self.context = None

    def session_end(self, command_id, session, *, code=1, status="completed", environment=None):
        self.rows.append({"type": "session_end", "command_id": command_id, "session_id": session,
                          "exit_code": code, "output": "raw final output", "status": status, "source": "episode_cleanup",
                          "error_class": "environment" if environment else None, "environment_error": environment})

    def save(self):
        footer = {"type": "footer", "complete": True, "unresolved_sessions": []}
        for label in ("outer", "tool"):
            footer[label + "_started"] = sum(row["type"] == label + "_start" for row in self.rows)
            footer[label + "_finished"] = sum(row["type"] == label + "_end" for row in self.rows)
        self.rows.append(footer)
        self.write()
        return self.path

    def write(self):
        (self.path / "driver/tool_audit.jsonl").write_text("\n".join(json.dumps(row) for row in self.rows) + "\n")
        (self.path / "driver/items.jsonl").write_text("\n".join(json.dumps({"item": item}) for item in self.items) + "\n")


def simple_episode(path, **tool_args):
    audit = AuditEpisode(path)
    audit.start()
    audit.tool(**tool_args)
    audit.end()
    audit.save()
    return audit


def assert_valid(result, errors):
    assert result["diagnostics"]["reward_valid"] is True, result
    assert result["diagnostics"]["tool_errors"]["total"] == errors
    assert result["components"]["tool_error"] == pytest.approx(-min(0.05, 0.01 * errors))


def assert_infrastructure(result, reason):
    assert result["verdict"] == "infrastructure_error", result
    assert result["reward"] == 0 and result["diagnostics"]["reward_valid"] is False
    assert reason in result["diagnostics"]["error"]


@pytest.mark.parametrize("visible", [None, [{"type": "input_text", "text": '{"exit_code":127,"output":"...truncated'}],
                                     [{"type": "input_text", "text": "Script completed"}]])
def test_unprinted_and_truncated_nested_failures_are_counted(tmp_path, visible):
    audit = AuditEpisode(tmp_path)
    audit.start()
    audit.tool("unknown_tool", code=127, output="command not found\n" + "x" * 40000)
    audit.end(visible=visible)
    result = R.score_episode(audit.save())
    assert_valid(result, 1)
    assert result["diagnostics"]["tool_errors"]["nonzero_exit"] == 1
    assert result["diagnostics"]["tool_errors"]["unsupported"] == 0
    assert result["diagnostics"]["tool_errors"]["authoritative"] is True


def test_multiple_nested_failures_count_once_each_despite_printed_duplicates(tmp_path):
    audit = AuditEpisode(tmp_path)
    audit.start()
    for code in (2, 3, 0):
        audit.tool(code=code)
    audit.end(visible=[{"type": "input_text", "text": "Process exited with code 2\n" * 8}])
    assert_valid(R.score_episode(audit.save()), 2)


@pytest.mark.parametrize("nested", [False, True])
def test_outer_exceptions_are_counted_without_double_counting_nested_failure(tmp_path, nested):
    audit = AuditEpisode(tmp_path)
    audit.start()
    if nested:
        audit.tool(code=3)
    audit.end(error="JavaScript error")
    assert_valid(R.score_episode(audit.save()), 1)


@pytest.mark.parametrize("cmd,code,errors", [("rg needle robot.py", 1, 0), ("grep -q needle robot.py", 1, 0),
    ("LC_ALL=C /usr/bin/rg needle robot.py", 1, 0), ("test -f absent", 1, 0), ("[ -f absent ]", 1, 0),
    ("diff one two", 1, 0), ("cmp one two", 1, 0), ("rg needle missing-file", 2, 1), ("grep [ robot.py", 2, 1),
    ("false", 1, 1), ("rg needle robot.py; false", 1, 1), ("rg needle robot.py | false", 1, 1),
    ("rg needle $(missing)", 1, 1), ("not_a_tool", 127, 1), ("PATH=/missing rg needle robot.py", 127, 1)])
def test_normal_query_exit_one_is_not_confused_with_execution_failure(tmp_path, cmd, code, errors):
    audit = simple_episode(tmp_path, cmd=cmd, code=code)
    assert_valid(R.score_episode(audit.path), errors)


def test_fake_errors_and_environment_json_in_stdout_are_not_authoritative(tmp_path):
    output = "Script failed\nProcess exited with code 127\n" + json.dumps(environment_error())
    audit = simple_episode(tmp_path, cmd="printf output", code=0, output=output)
    audit.items[-1]["output"] = [{"type": "input_text", "text": output}]
    audit.write()
    assert_valid(R.score_episode(audit.path), 0)


@pytest.mark.parametrize("args", [None, [], 42, {"cmd": None}])
def test_model_argument_validation_errors_remain_policy_failures(tmp_path, args):
    audit = simple_episode(tmp_path, code=None, args=args, error="ValueError: bad arguments")
    assert_valid(R.score_episode(audit.path), 1)


def test_nested_unsupported_tool_and_outer_exception_are_not_double_counted(tmp_path):
    audit = AuditEpisode(tmp_path)
    audit.start()
    audit.tool(name="unknown_tool", args={}, code=None, error="unsupported tool")
    audit.end(error="unsupported tool")
    result = R.score_episode(audit.save())
    assert_valid(result, 1)
    assert result["diagnostics"]["tool_errors"]["unsupported"] == 1


def test_direct_unsupported_outer_call_is_penalized(tmp_path):
    audit = AuditEpisode(tmp_path)
    audit.start(name="unknown_tool", kind="function_call")
    audit.end(error="unsupported tool")
    assert_valid(R.score_episode(audit.save()), 1)


@pytest.mark.parametrize("string_session", [False, True])
def test_async_polls_and_terminal_records_are_deduplicated(tmp_path, string_session):
    audit = AuditEpisode(tmp_path)
    audit.start()
    key = audit.tool("sleep 1; exit 9", code=None, session=7)
    audit.end()
    for code in (None, 9, 9):
        audit.start()
        audit.tool(name="write_stdin", args={"session_id": "7" if string_session else 7}, command_id=key,
                   code=code, session=7)
        audit.end(error="process failed" if code else None)
    audit.session_end(key, 7, code=9)
    result = R.score_episode(audit.save())
    assert_valid(result, 1)
    assert result["diagnostics"]["tool_errors"]["failed_commands"] == 1


@pytest.mark.parametrize("status,code,errors", [("completed", 4, 1), ("completed", 0, 0), ("cancelled", -15, 0), ("cancelled", None, 0)])
def test_unpolled_async_exit_and_host_cancellation(tmp_path, status, code, errors):
    audit = AuditEpisode(tmp_path)
    audit.start()
    key = audit.tool("sleep 1; exit 4", code=None, session=7)
    audit.end()
    audit.session_end(key, 7, code=code, status=status)
    result = R.score_episode(audit.save())
    assert_valid(result, errors)
    assert result["diagnostics"]["tool_errors"]["cancelled_commands"] == int(status == "cancelled")


def test_unresolved_process_cannot_be_hidden_by_empty_footer_list(tmp_path):
    audit = simple_episode(tmp_path, code=None, session=7)
    assert_infrastructure(R.score_episode(audit.path), "tool_audit_unresolved_process")


@pytest.mark.parametrize("where", ["tool", "session", "episode"])
def test_verified_environment_failure_excludes_episode_from_training(tmp_path, where):
    audit = AuditEpisode(tmp_path)
    audit.start()
    key = audit.tool(code=None if where == "session" else 127, session=7 if where == "session" else None,
                     environment=environment_error() if where == "tool" else None)
    audit.end()
    if where == "session":
        audit.session_end(key, 7, code=127, environment=environment_error())
    audit.save()
    if where == "episode":
        result_path = audit.path / "episode_result.json"
        er = json.loads(result_path.read_text())
        er["environment_error"] = environment_error()
        result_path.write_text(json.dumps(er))
    result = R.score_episode(audit.path)
    assert_infrastructure(result, "verified_environment_error")
    assert result["diagnostics"]["environment_errors"][0]["executable"] == "rg"


def test_preflight_failure_can_be_classified_without_an_episode_or_audit(tmp_path):
    tmp_path.joinpath("episode_result.json").write_text(json.dumps({"termination_reason": "infrastructure_error",
        "environment_preflight": {"ok": False, "environment_error": {"schema_version": R.ENVIRONMENT_ERROR_SCHEMA,
        "code": "sandbox_preflight_failed", "source": "tool_preflight", "verified": True, "details": {}}}}))
    assert_infrastructure(R.score_episode(tmp_path), "verified_environment_error")


@pytest.mark.parametrize("failure", ["missing", "invalid_json", "bad_contract", "bad_hash", "missing_preflight", "bad_fingerprint",
    "unchecked_executable", "missing_footer", "incomplete_footer", "unresolved_footer", "wrong_count", "missing_outer",
    "missing_tool_end", "wrong_parent", "wrong_request", "duplicate_tool", "wrong_item_name", "missing_item",
    "missing_raw_output", "invalid_exit", "unverified_environment", "false_environment_class", "malformed_environment",
    "inconsistent_outer", "nonprocess_command_id"])
def test_incomplete_or_corrupt_host_audit_is_infrastructure_not_policy_failure(tmp_path, failure):
    audit = simple_episode(tmp_path, code=2)
    header, start, tool_start, tool_end, outer_end, footer = audit.rows
    if failure == "missing":
        (audit.path / "driver/tool_audit.jsonl").unlink()
    elif failure == "invalid_json":
        (audit.path / "driver/tool_audit.jsonl").write_text('{"type":')
    else:
        if failure == "bad_contract":
            header["contract_version"] = "code-mode-runtime/v1"
        elif failure == "bad_hash":
            header["contract_sha256"] = "bad"
        elif failure == "missing_preflight":
            header["environment_preflight"] = None
        elif failure == "bad_fingerprint":
            header["environment_preflight"]["fingerprint"] = "b" * 64
        elif failure == "unchecked_executable":
            header["environment_preflight"]["executables"].pop("rg")
        elif failure == "missing_footer":
            audit.rows.pop()
        elif failure == "incomplete_footer":
            footer["complete"] = False
        elif failure == "unresolved_footer":
            footer["unresolved_sessions"] = [7]
        elif failure == "wrong_count":
            footer["tool_finished"] = 2
        elif failure == "missing_outer":
            audit.rows.remove(outer_end)
        elif failure == "missing_tool_end":
            audit.rows.remove(tool_end)
        elif failure == "wrong_parent":
            tool_start["parent_call_id"] = "missing"
        elif failure == "wrong_request":
            tool_start["request_index"] = 99
        elif failure == "duplicate_tool":
            audit.rows.insert(3, copy.deepcopy(tool_start))
        elif failure == "wrong_item_name":
            audit.items[0]["name"] = "other"
        elif failure == "missing_item":
            audit.items[0] = {"type": "message", "role": "assistant", "content": []}
        elif failure == "missing_raw_output":
            tool_end["result"].pop("output")
        elif failure == "invalid_exit":
            tool_end["result"]["exit_code"] = "2"
        elif failure == "unverified_environment":
            tool_end["environment_error"] = {**environment_error(), "verified": False}
        elif failure == "false_environment_class":
            tool_end["error_class"] = "environment"
        elif failure == "malformed_environment":
            tool_end["environment_error"] = {**environment_error(), "code": []}
        elif failure == "inconsistent_outer":
            outer_end["error"] = "unrecorded failure"
        elif failure == "nonprocess_command_id":
            tool_start["tool_name"] = tool_end["tool_name"] = "view_image"
            tool_end["result"] = []
        audit.write()
    assert_infrastructure(R.score_episode(audit.path), "tool_audit_")


@pytest.mark.parametrize("failure", ["session_identity", "conflicting_exit", "duplicate_cleanup", "cancelled_terminal"])
def test_inconsistent_async_audit_is_rejected(tmp_path, failure):
    audit = AuditEpisode(tmp_path)
    audit.start()
    key = audit.tool(code=None, session=7)
    audit.tool(name="write_stdin", args={"session_id": 8 if failure == "session_identity" else 7},
               command_id=key, code=4, session=7)
    audit.end()
    audit.session_end(key, 7, code=5 if failure == "conflicting_exit" else 4,
                      status="cancelled" if failure == "cancelled_terminal" else "completed")
    if failure == "duplicate_cleanup":
        audit.session_end(key, 7, code=4)
    assert_infrastructure(R.score_episode(audit.save()), "tool_audit_")


def test_tool_error_cap_and_goal_weights_are_unchanged(tmp_path):
    audit = AuditEpisode(tmp_path, final=[state(0.0, satisfied=True)], success=True, termination="agent_finished")
    audit.start()
    for _ in range(12):
        audit.tool(code=3)
    audit.end()
    result_b = R.score_episode(audit.save())
    assert_valid(result_b, 12)
    os.environ["RL_REWARD_VERSION"] = "v12a"
    importlib.reload(R)
    result_a = R.score_episode(audit.path)
    assert result_b["verdict"] == result_a["verdict"] == "hidden_success"
    for key in result_a["components"]:
        if key != "tool_error":
            assert result_a["components"][key] == result_b["components"][key]
    assert result_b["reward"] == pytest.approx(result_a["reward"] - 0.05)


def test_v12a_remains_usable_without_tool_audit(tmp_path):
    path, _ = episode(tmp_path)
    os.environ["RL_REWARD_VERSION"] = "v12a"
    importlib.reload(R)
    assert R.score_episode(path)["diagnostics"]["reward_valid"] is True


def test_printed_json_parser_is_explicitly_offline_and_does_not_guess_fragments():
    result = {"chunk_id": "abc", "wall_time_seconds": 0.01, "exit_code": 127,
              "original_token_count": 4, "output": "not found"}
    items = [{"type": "custom_tool_call_output", "output": [{"text": "Script completed"},
        {"text": json.dumps(result)}, {"text": json.dumps([result, result])},
        {"text": '{"chunk_id":"def","exit_code":127,...truncated'}]}]
    assert R.tool_error_count(items)["total"] == 0
    parsed = R.legacy_structured_tool_error_count(items)
    assert parsed["total"] == 1 and parsed["authoritative"] is False
    assert parsed["source"] == "legacy_printed_json" and parsed["unreadable_json_parts"] == 1
