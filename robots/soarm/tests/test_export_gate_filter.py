"""Export filter: requests after a gate-rejected first finish are not supervised (LIBERO r7 rule)."""
from soarm.export_sft import _first_finish_request


def test_first_finish_request():
    responses = {
        1: {"items": [{"type": "custom_tool_call", "input": "print(robot.get_robot_state())"}]},
        2: {"items": [{"type": "custom_tool_call", "input": "robot.finish('done')"}]},
        3: {"items": [{"type": "custom_tool_call", "input": "robot.finish('again')"}]},
    }
    assert _first_finish_request(responses) == 2


def test_no_finish():
    assert _first_finish_request({}) is None
    assert _first_finish_request({1: {"items": [{"type": "custom_tool_call", "input": "x = 1"}]}}) is None
