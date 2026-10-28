"""Byte-exact truncation and output envelopes against real Codex 0.154 rollouts."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from captain_harness.react.tools import ExecResult, approx_tokens, truncate_output

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "codex_capture" / "truncation_fixtures.json").read_text(encoding="utf-8"))


def _body_after_output(text: str) -> str:
    return text.split("Output:\n", 1)[1]


def _header_tokens(text: str) -> int:
    return int(re.search(r"Original token count: (\d+)", text).group(1))


@pytest.mark.parametrize("fixture", [f for f in FIXTURES if f["mode"] == "function" and "aggregated_output" in f], ids=lambda f: f["tag"])
def test_function_mode_output_matches_codex(fixture):
    output, original = truncate_output(fixture["aggregated_output"], fixture["arguments"].get("max_output_tokens"))
    assert original == _header_tokens(fixture["model_output"])
    assert output == _body_after_output(fixture["model_output"])
    envelope = ExecResult("abcdef", 0.0, output, original, exit_code=fixture["exit_code"]).as_text()
    assert envelope.split("\n", 2)[2] == fixture["model_output"].split("\n", 2)[2]  # everything after chunk id + wall time


@pytest.mark.parametrize("fixture", [f for f in FIXTURES if f["mode"] == "code" and "model_object" in f], ids=lambda f: f["tag"])
def test_code_mode_output_matches_codex(fixture):
    budgets = [int(x) for x in re.findall(r"max_output_tokens\"?\s*:\s*(\d+)", fixture["script"])] or [None]
    expected = fixture["model_object"]
    matches = [truncate_output(fixture["aggregated_output"], b) for b in budgets]
    assert any(out == expected["output"] and original == expected["original_token_count"] for out, original in matches)


def test_token_count_is_bytes_over_four_rounded_up():
    assert approx_tokens(b"") == 0 and approx_tokens(b"abcd") == 1 and approx_tokens(b"abcde") == 2
    assert approx_tokens("é" * 2) == 1  # 4 bytes


def test_truncation_keeps_head_and_tail_and_counts_lines():
    text = "".join(f"line {i}\n" for i in range(3000))
    out, original = truncate_output(text, 100)
    assert out.startswith(f"Warning: truncated output (original token count: {original})\nTotal output lines: 3000\n\n")
    body = out.split("\n\n", 1)[1]
    head, tail = body.split("…", 1)[0], body.rsplit("…", 1)[1]
    assert len(head.encode()) == 200 and len(tail.encode()) == 200
    assert re.search(r"…\d+ tokens truncated…", out)


def test_no_truncation_when_within_budget():
    assert truncate_output("short\n", 10) == ("short\n", 2)


def test_running_process_envelope():
    text = ExecResult("67e9a8", 30.0015, "", 0, session_id=88202).as_text()
    assert text == "Chunk ID: 67e9a8\nWall time: 30.0015 seconds\nProcess running with session ID 88202\nOriginal token count: 0\nOutput:\n"


def test_code_mode_result_object_key_order():
    obj = ExecResult("c8652c", 0.00000535, "x", 1, exit_code=0).as_object()
    assert list(obj) == ["chunk_id", "wall_time_seconds", "exit_code", "original_token_count", "output"]
    running = ExecResult("bba61e", 10.0, "", 0, session_id=86662).as_object()
    assert list(running) == ["chunk_id", "wall_time_seconds", "session_id", "original_token_count", "output"]
