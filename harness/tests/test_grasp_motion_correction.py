"""Only measured corresponding surface motion may alter an issued grasp."""
import numpy as np
import pytest

from captain_harness.adapters.capx.grasp_motion_correction import assess_grasp_correction
from captain_harness.adapters.capx.visual_tracking import track_rgbd
from test_sensor_motion_tools import camera


def textured_pair(shift=0):
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(12)
    image = np.zeros((128, 128, 3), np.uint8)
    image[30:90, 30:90] = rng.integers(0, 255, (60, 60, 3), dtype=np.uint8)
    mask = np.zeros((128, 128), bool)
    mask[30:90, 30:90] = True
    moved = cv2.warpAffine(image, np.float32([[1, 0, shift], [0, 1, 0]]), (128, 128))
    return camera(image), camera(moved), mask


def test_stationary_surface_and_calibrated_camera_motion_do_not_shift_grasp():
    first, latest, mask = textured_pair()
    result = assess_grasp_correction(track_rgbd(first, latest, mask))
    assert result["decision"] == "retain"
    assert result["applied_m"] is None
    first, latest, mask = textured_pair(4)
    latest.extrinsics[0, 3] = -.02
    result = assess_grasp_correction(track_rgbd(first, latest, mask))
    assert result["decision"] == "retain"


def test_corresponding_translation_preserves_off_center_grasp_offset():
    first, latest, mask = textured_pair(4)
    result = assess_grasp_correction(track_rgbd(first, latest, mask))
    assert result["decision"] == "translate"
    # The measured 2 cm displacement is independent of the selected grasp's
    # original intentional offset from the object's segmented centroid.
    np.testing.assert_allclose(result["applied_m"], [.02, 0, 0], atol=.001)


def test_excessive_measured_motion_requests_fresh_grasp_instead_of_clamping():
    first, latest, mask = textured_pair(9)
    result = assess_grasp_correction(track_rgbd(first, latest, mask))
    assert result["decision"] == "reacquire"
    assert result["applied_m"] is None


@pytest.mark.parametrize("evidence", [
    {"status": "insufficient", "translation_m": None},
    {"status": "image_only", "translation_m": None},
    {"status": "supported", "translation_m": [float("nan"), 0, 0], "motion_spread_m": 0},
    {"status": "supported", "translation_m": [.02, 0, 0], "motion_spread_m": .01},
    {"status": "supported", "translation_m": [.02, 0, 0]},
])
def test_uncertain_evidence_never_creates_an_automatic_correction(evidence):
    result = assess_grasp_correction(evidence)
    assert result["decision"] == "retain"
    assert result["applied_m"] is None


def test_supported_vertical_drift_invalidates_horizontal_only_correction():
    result = assess_grasp_correction({"status": "supported", "translation_m": [.01, 0, .03], "motion_spread_m": .001})
    assert result["decision"] == "reacquire"
    assert result["reason"] == "vertical_target_motion"


@pytest.mark.parametrize('translation,expected', [([0,0,0], 'retain'), ([.02,0,0], 'translate'), ([.05,0,0], 'reacquire')])
def test_measured_correction_is_used_by_actual_descent(tmp_path, translation, expected):
    from test_v18_grasp_references import issued
    from captain_harness.ports.capability import CapabilityResult
    f, broker, candidate, _ = issued(tmp_path)
    assert broker.invoke('align_pregrasp', {'candidate':candidate}).success
    aligned = f.backend._aligned_pregrasp
    calls = []
    f.backend.track_target = lambda *args: CapabilityResult.ok({
        'status':'supported', 'translation_m':translation, 'motion_spread_m':.001})
    def descent(position, orientation, **kwargs):
        calls.append(position)
        return CapabilityResult.ok({'closure_executed':False})
    f.robot.descend_and_close = descent
    result = f.backend.invoke('descend_and_close', {})
    diagnostic = result.value['pre_descent_centering']
    assert diagnostic['decision'] == expected
    if expected == 'reacquire':
        assert not result.executed and not result.success
        assert not calls
        assert f.backend._aligned_pregrasp is None
    else:
        assert result.success
        np.testing.assert_allclose(calls[0], np.asarray(aligned.eef_position) + translation)
