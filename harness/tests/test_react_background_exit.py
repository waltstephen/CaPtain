"""Main-process outcomes survive cleanup of descendants holding output pipes."""
from __future__ import annotations

import os

import pytest

from captain_harness.react.sandbox import Sandbox
from captain_harness.react.unified_exec import SessionManager, child_environment


@pytest.fixture
def manager(tmp_path):
    sessions = SessionManager(sandbox=Sandbox(workspace=tmp_path),
                              env=child_environment(os.environ), default_cwd=tmp_path)
    yield sessions
    sessions.close_all()


@pytest.mark.parametrize("exit_code", [0, 7])
def test_finished_main_command_is_not_cancelled_when_a_child_holds_stdout(manager, exit_code):
    result = manager.exec_command(cmd=f"sleep 30 & exit {exit_code}", login=False, yield_time_ms=250)
    assert result.session_id is not None
    session = manager.sessions[result.session_id]
    assert session.process.wait(timeout=5) == exit_code
    assert not session.exited.is_set()

    [final] = manager.close_all()
    assert final["exit_code"] == exit_code
    assert final["completed"] is True and final["cancelled"] is False
    assert not session.reader.is_alive()
    assert manager.sessions == {}


@pytest.mark.parametrize("command", ["sleep 30", "sleep 30 & wait", "exec >/dev/null 2>&1; sleep 30"])
def test_main_command_still_running_at_cleanup_is_cancelled(manager, command):
    result = manager.exec_command(cmd=command, login=False, yield_time_ms=250)
    session = manager.sessions[result.session_id]
    assert session.process.poll() is None

    [final] = manager.close_all()
    assert final["completed"] is True and final["cancelled"] is True
    assert final["exit_code"] < 0
    assert not session.reader.is_alive()


def test_unpolled_exit_with_closed_output_keeps_its_terminal_failure(manager):
    result = manager.exec_command(cmd="sleep 0.4; exit 9", login=False, yield_time_ms=250)
    session = manager.sessions[result.session_id]
    assert session.exited.wait(timeout=5)

    [final] = manager.close_all()
    assert final["exit_code"] == 9
    assert final["completed"] is True and final["cancelled"] is False
