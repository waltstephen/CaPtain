import numpy as np
import pytest
from captain_harness.adapters.robotics.swept_collision import SweptSceneValidator
from captain_harness.ports.capability import FailureCategory
from test_swept_scene_edges import validator
from test_whole_planning_deadline import make_tool, clock_at


def test_rejection_keeps_nominal_witness_between_clear_samples():
    edge, collision = validator()
    assert not edge([0., 0.], [.02, 0.])
    witness = edge.receipt()["first_rejection"]
    assert witness["kind"] == "nominal_collision"
    assert collision(witness["joints"])["collision"]
    assert witness["collision_receipt"]["collision"] is True
    np.testing.assert_array_equal(witness["edge_start_joints"], [0., 0.])
    np.testing.assert_array_equal(witness["edge_end_joints"], [.02, 0.])
    # Bounded immutable first witness survives later rejected and clear edges.
    saved = edge.receipt()
    assert not edge([0., 0.], [.02, 0.])
    assert edge([0., .03], [.02, .03])
    assert edge.receipt()["first_rejection"] == saved["first_rejection"]
    witness["collision_receipt"]["collision"] = False
    assert edge.receipt()["first_rejection"]["collision_receipt"]["collision"] is True


def test_inflated_hit_is_not_reported_as_a_nominal_collision():
    edge = SweptSceneValidator(lambda q, extra_margin=0.: {"collision": extra_margin > 0},
        lambda delta: float(np.abs(delta).sum()))
    assert not edge([0.], [1e-8])
    witness = edge.receipt()["first_rejection"]
    assert witness["kind"] == "interval_unresolved"
    assert witness["collision_receipt"]["collision"] is False
    assert witness["remaining_motion_bound_m"] > 0


def test_successful_clearance_certificate_does_not_invent_a_failure():
    edge, _ = validator()
    assert edge([0., .03], [.02, .03])
    assert edge.receipt()["first_rejection"] is None


def test_public_planning_timeout_preserves_witness_without_post_deadline_geometry(monkeypatch):
    now = clock_at(monkeypatch); tool, model = make_tool(); queries = []
    model.solve_ik = lambda *a, **kw: np.array([.2])
    def collision(q, extra_margin=0., first_hit_only=False):
        assert now[0] < .1, 'diagnostic reconstructed after deadline'
        queries.append(float(q[0]))
        hit = abs(float(q[0]) - .1) < .01 + extra_margin
        return {"collision": hit, "observed_collisions": {"test_hand": 1} if hit else {}, "self_collisions": []}
    tool._scene_collision_checker = lambda *a, **kw: (collision, {"kind": "unchanged_test_policy"})
    def path(*a, **kw):
        assert not kw["edge_valid"]([0.], [.2])
        now[0] = .11
        return None, {"reason": "deadline_expired", "checks": 1}
    monkeypatch.setattr('captain_harness.adapters.capx.motion_tools.plan_joint_path', path)
    result = tool.plan_motion({"position_m": [.2,0,0], "quaternion_wxyz": [1,0,0,0]},
        'end_effector', 'scene', timeout_s=.1)
    assert not result.success and not result.executed
    assert result.failure_category is FailureCategory.TIMEOUT
    assert "plan_handle" not in result.value
    witness = result.value["path_search"]["swept_scene_validation"]["first_rejection"]
    assert witness["kind"] == "nominal_collision"
    assert witness["collision_receipt"]["observed_collisions"] == {"test_hand": 1}
    assert result.value["collision_policy"] == {"kind": "unchanged_test_policy"}
    assert len(queries) == 2
