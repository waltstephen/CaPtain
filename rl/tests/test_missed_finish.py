"""Missed-finish fix (default off): request_costs post_success_act and the reward_patch missed-finish return.
Runs under pytest or directly (python test_missed_finish.py)."""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import physics_critic as PC  # noqa: E402


def _rows(phys, sat):
    return [{"request_index": i + 1, "req_phys": n, "req_phys_bad": 0, "f": {"all_sat": float(s)}} for i, (n, s) in enumerate(zip(phys, sat))]


REF = {"libero_spatial:0": 10}


def test_post_success_act_charges_every_action_from_first_success():
    c = 0.008
    rows = _rows([4, 4, 4, 4, 4], [0, 0, 1, 0, 1])          # success first seen at request 3 (then undone, then again)
    off = PC.request_costs(rows, {"c_act": c, "c_fail": 0.0, "ref": REF}, "libero_spatial:0")
    on = PC.request_costs(rows, {"c_act": c, "c_fail": 0.0, "ref": REF, "post_success_act": True}, "libero_spatial:0")
    # ref 10: requests 1-2 free (8 actions), request 3 pays 2 beyond the ref when off, all 4 when on; 4 and 5 pay 4 either way
    assert off[1] == on[1] == 0.0 and off[2] == on[2] == 0.0
    assert abs(off[3] - (-c * 2)) < 1e-12 and abs(on[3] - (-c * 4)) < 1e-12
    assert all(abs(on[k] - (-c * 4)) < 1e-12 for k in (4, 5))
    # before any success the ref still applies: success only at the last request -> earlier requests identical to off
    rows2 = _rows([1, 1, 1, 6], [0, 0, 0, 1])
    off2 = PC.request_costs(rows2, {"c_act": c, "c_fail": 0.0, "ref": REF}, "libero_spatial:0")
    on2 = PC.request_costs(rows2, {"c_act": c, "c_fail": 0.0, "ref": REF, "post_success_act": True}, "libero_spatial:0")
    assert [off2[k] for k in (1, 2, 3)] == [on2[k] for k in (1, 2, 3)] and abs(on2[4] - (-c * 6)) < 1e-12 and off2[4] == 0.0   # 9 actions in total: still under the ref of 10 when off
    # perception-only requests (req_phys 0) stay free after success
    rows3 = _rows([0, 0], [1, 1])
    assert PC.request_costs(rows3, {"c_act": c, "c_fail": 0.0, "ref": REF, "post_success_act": True}, "libero_spatial:0") == {1: 0.0, 2: 0.0}


def test_reward_patch_missed_finish_return():
    costs = {"c_act": 0.008, "c_fail": 0.02, "missed_finish_return": 0.5, "exhaust_reward": -0.1, "ref": REF}
    patched = dict(costs, missed_finish_return=0.25, post_success_act=True)
    missed = {"diagnostics": {"host_final_success": True, "termination_reason": "request_cap_exhausted"}}
    fin = {"diagnostics": {"host_final_success": True, "termination_reason": "agent_finished"}}
    fail = {"diagnostics": {"host_final_success": False, "termination_reason": "request_cap_exhausted"}}
    assert PC.advantage_return(missed, -0.3, costs) == 0.5 and PC.advantage_return(missed, -0.3, patched) == 0.25
    assert PC.advantage_return(fin, -0.3, patched) == 1.0 and PC.advantage_return(fail, -0.3, patched) == -0.1


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"MISSED_FINISH_TESTS OK ({len(tests)})")
