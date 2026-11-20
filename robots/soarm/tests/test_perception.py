"""Perception mode (soarm/perception.py): real services by default, privileged only on request; MoGe-2 depth by default."""
from __future__ import annotations

import json
import types

import numpy as np
import pytest

from soarm import perception


def _args():
    return types.SimpleNamespace(sam3_url="http://127.0.0.1:8114", molmo_url="http://127.0.0.1:8122/v1")


def test_default_mode_is_real_with_paper_revisions_and_moge2_depth(monkeypatch):
    for k in ("SOARM_PERCEPTION", "SOARM_SAM3_URLS", "SOARM_MOLMO_URLS", "SOARM_DEPTH_NOISE", "SOARM_DEPTH",
              "SOARM_DEPTH_URLS", "SOARM_DEPTH_REVISION", "CAPTAIN_DEPTH_URL"):
        monkeypatch.delenv(k, raising=False)
    cfg = perception.config(_args(), 7)
    assert cfg["mode"] == "real" and cfg["depth"] == "moge2" and cfg["depth_noise_sigma_1m"] == 0.0
    assert cfg["sam3_url"] == "http://127.0.0.1:8114" and cfg["molmo_url"] == "http://127.0.0.1:8122/v1"
    assert "privileged" not in json.dumps(cfg["tool_revisions"])
    assert cfg["tool_revisions"]["molmo"].endswith("e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b")
    assert cfg["depth_url"] == "http://127.0.0.1:8130" and cfg["depth_align"] == "plane"
    assert cfg["depth_support_plane_z_m"] == 0.0 and cfg["depth_table_footprint_m"] == perception.TABLE_FOOTPRINT_M
    assert cfg["tool_revisions"]["depth"] == cfg["depth_revision"] == perception.MOGE2_REVISION
    assert "cb0e8bbd6b1e243589717c78e750b1ba4c093acf" in cfg["depth_revision"]


def test_sensor_depth_is_tests_or_explicit_ablation_only(monkeypatch):
    monkeypatch.delenv("SOARM_PERCEPTION", raising=False)
    monkeypatch.delenv("SOARM_DEPTH_ABLATION", raising=False)
    monkeypatch.setenv("SOARM_DEPTH", "sensor")
    cfg = perception.config(_args(), 7)
    assert cfg["depth"] == "sim_sensor" and "depth" not in cfg["tool_revisions"]
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    with pytest.raises(RuntimeError):
        perception.depth_mode()
    monkeypatch.setenv("SOARM_DEPTH_ABLATION", "1")
    assert perception.depth_mode() == "sensor"
    monkeypatch.setenv("SOARM_DEPTH", "unidepth")
    with pytest.raises(ValueError):
        perception.depth_mode()
    monkeypatch.setenv("SOARM_DEPTH", "moge2")
    monkeypatch.setenv("SOARM_DEPTH_NOISE", "0.002")
    with pytest.raises(ValueError):   # a sensor-noise model on estimated depth would be meaningless
        perception.config(_args(), 7)


def test_depth_replicas_are_stable_per_seed(monkeypatch):
    monkeypatch.delenv("SOARM_PERCEPTION", raising=False)
    monkeypatch.delenv("SOARM_DEPTH", raising=False)
    monkeypatch.delenv("SOARM_DEPTH_NOISE", raising=False)
    monkeypatch.setenv("SOARM_DEPTH_URLS", "http://h:8130,http://h:8131")
    picks = [perception.config(_args(), s)["depth_url"] for s in range(1_020_000, 1_020_040)]
    assert set(picks) == {"http://h:8130", "http://h:8131"}


def test_privileged_needs_the_flag_and_bad_modes_fail(monkeypatch):
    monkeypatch.setenv("SOARM_PERCEPTION", "privileged")
    assert perception.config(_args(), 0)["tool_revisions"] == {"sam3": "privileged-sim", "molmo": "privileged-sim"}
    monkeypatch.setenv("SOARM_PERCEPTION", "sim")
    with pytest.raises(ValueError):
        perception.mode()


def test_replicas_are_stable_per_seed_and_spread(monkeypatch):
    monkeypatch.delenv("SOARM_PERCEPTION", raising=False)
    monkeypatch.setenv("SOARM_MOLMO_URLS", "http://h:8122/v1, http://h:8123/v1")
    picks = [perception.config(_args(), s)["molmo_url"] for s in range(1_060_000, 1_060_040)]
    assert set(picks) == {"http://h:8122/v1", "http://h:8123/v1"}
    assert perception.config(_args(), 1_060_003)["molmo_url"] == picks[3]
    monkeypatch.setenv("SOARM_SAM3_URLS", "sim://sam3")
    with pytest.raises(ValueError):
        perception.config(_args(), 0)


def test_depth_noise_keeps_holes_and_scales_with_depth():
    rng = np.random.default_rng(0)
    depth = np.full((64, 64), 0.5, np.float32)
    depth[:4] = 0.0
    far = np.full((64, 64), 1.5, np.float32)
    noisy, noisy_far = perception.depth_noise(depth, 0.002, rng), perception.depth_noise(far, 0.002, rng)
    assert noisy.dtype == np.float32 and (noisy[:4] == 0).all()
    valid = noisy[4:][noisy[4:] > 0]
    assert abs(float(valid.mean()) - 0.5) < 1e-3
    assert np.std(noisy_far[noisy_far > 0]) > 3 * np.std(valid)


def test_logged_transport_records_real_calls(tmp_path):
    t = perception.LoggedHttpTransport(tmp_path / "perception.jsonl")

    class Inner:
        def request(self, method, url, *, payload=None, timeout_s):
            if url.endswith("/segment"):
                return {"results": [{"score": 0.9, "box": [1, 2, 3, 4], "label": "red cube", "mask_base64": "x"}]}
            raise OSError("down")

    t.inner = Inner()
    t.request("POST", "http://s/segment", payload={"image_base64": "png", "text_prompt": "red cube"}, timeout_s=5)
    with pytest.raises(OSError):
        t.request("POST", "http://m/v1/chat/completions", payload={"model": "allenai/Molmo2-8B", "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Point at red cube"}]}]}, timeout_s=5)
    rows = [json.loads(line) for line in (tmp_path / "perception.jsonl").read_text().splitlines()]
    assert [r["ok"] for r in rows] == [True, False] and all(r["privileged"] is False for r in rows)
    assert rows[0]["results"][0]["label"] == "red cube" and "mask_base64" not in json.dumps(rows)
    assert rows[1]["prompt"] == "Point at red cube" and rows[1]["error"].startswith("OSError")


def test_real_mode_driver_renders_no_segmentation():
    from soarm.driver import SoarmDriver
    from soarm.scene import CAMERAS
    from soarm.sim import SoarmSim
    from soarm.twin import build_twin_xml
    sim = SoarmSim("cube_pick_lift", 1_060_000)
    drv = SoarmDriver(sim, build_twin_xml(), force_seed=1_060_000, privileged_frames=False)
    frames = drv.frames(list(CAMERAS))
    assert drv.cameras._seg is None and not drv.frame_registry
    assert all(f.depth_m.dtype == np.float32 and f.rgb.shape[2] == 3 for f in frames.values())
    drv.close()


def test_privileged_is_refused_outside_pytest(monkeypatch):
    from soarm.perception_sim import SimPerceptionTransport
    monkeypatch.setenv("SOARM_PERCEPTION", "privileged")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    with pytest.raises(RuntimeError):
        perception.mode()
    with pytest.raises(RuntimeError):
        SimPerceptionTransport(types.SimpleNamespace(sim=types.SimpleNamespace(specs={})))


def test_unhealthy_services_refuse_the_episode():
    class Down:
        def health(self, names=None):
            return {"sam3": {"healthy": True}, "molmo": {"healthy": False, "error": "OSError"}}
    with pytest.raises(RuntimeError):
        perception.require_healthy(Down())


def test_moge2_driver_renders_rgb_only_and_privileged_needs_sensor_depth():
    from soarm.driver import SoarmDriver
    from soarm.scene import CAMERAS
    from soarm.sim import SoarmSim
    from soarm.twin import build_twin_xml
    sim = SoarmSim("cube_pick_lift", 1_060_000)
    drv = SoarmDriver(sim, build_twin_xml(), force_seed=1_060_000, privileged_frames=False, sensor_depth=False)
    frames = drv.frames(list(CAMERAS))
    assert drv.cameras._depth is None and all(f.depth_m is None for f in frames.values())
    drv.close()
    with pytest.raises(ValueError):
        SoarmDriver(sim, build_twin_xml(), force_seed=1_060_000, privileged_frames=True, sensor_depth=False)


def test_moge2_estimators_are_the_hardware_mono_depth_path_with_plane_scale(monkeypatch, tmp_path):
    """mono_depth_estimators = harness MonoDepthSource (align=plane, table at z=0) for both cameras, logged per call. A fake
    service answering the rendered depth x1.37 (a metric-scale error) comes back at the rendered scale."""
    from captain_harness.adapters.hardware.mono_depth import MonoDepthSource
    from soarm.driver import SoarmDriver
    from soarm.scene import CAMERAS
    from soarm.sim import SoarmSim
    from soarm.twin import build_twin_xml
    for k in ("SOARM_PERCEPTION", "SOARM_DEPTH", "SOARM_DEPTH_NOISE", "SOARM_DEPTH_URLS"):
        monkeypatch.delenv(k, raising=False)
    perc = perception.config(_args(), 1_020_000)
    transport = perception.LoggedHttpTransport(tmp_path / "perception.jsonl")
    estimators = perception.mono_depth_estimators(perc, CAMERAS, transport)
    assert set(estimators) == set(CAMERAS)
    source = estimators["agentview"].source
    assert isinstance(source, MonoDepthSource) and source.align == "plane" and source.support_plane_z_m == 0.0
    assert source.model == "moge2" and source.url == "http://127.0.0.1:8130"
    assert source.table_footprint_m == perception.TABLE_FOOTPRINT_M
    sim = SoarmSim("cube_pick_lift", 1_060_000)
    drv = SoarmDriver(sim, build_twin_xml(), force_seed=1_060_000, privileged_frames=False)
    f = drv.frames(["agentview"])["agentview"]
    monkeypatch.setattr(MonoDepthSource, "_request", lambda self, rgb, intr: (f.depth_m.astype(np.float64) * 1.37,
                                                                             {"model": "moge2", "elapsed_s": 0.1}))
    depth = estimators["agentview"](camera="agentview", rgb=f.rgb, raw_sim_depth=None, intrinsics_3x3=f.intrinsics,
                                    camera_to_base_4x4=f.optical_in_robot_base)
    assert np.nanmedian(np.abs(depth - f.depth_m)) < 1e-3
    estimators["agentview"](camera="agentview", rgb=f.rgb, raw_sim_depth=None, intrinsics_3x3=f.intrinsics,
                            camera_to_base_4x4=f.optical_in_robot_base)
    rows = [json.loads(line) for line in (tmp_path / "perception.jsonl").read_text().splitlines()]
    assert [r["kind"] for r in rows] == ["depth", "depth"] and [r["cache_hit"] for r in rows] == [False, True]
    assert abs(rows[0]["align"]["scale"] - 1 / 1.37) < 1e-3 and rows[0]["privileged"] is False
    assert rows[0]["align"]["footprint"] is True
    drv.close()
    assert perception.mono_depth_estimators({"depth": "sim_sensor"}, CAMERAS, transport) == {}


def test_batch_quarantines_sensor_depth_episodes():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("soarm_batch", Path(__file__).resolve().parents[1] / "tools" / "batch.py")
    batch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(batch)
    clean = {"perception_mode": "real", "perception_privileged_calls": 0, "perception_sim_log": False,
             "depth_source": "moge2", "depth_revision": perception.MOGE2_REVISION}
    assert batch.perception_violation(clean) is None
    assert batch.perception_violation({**clean, "depth_source": "sim_sensor"}) == "depth_source=sim_sensor"
    assert batch.perception_violation({**clean, "depth_revision": None})


def test_table_footprint_lies_on_every_table():
    """The plane fit's calibrated table rectangle is covered by every SO-ARM table (center, edge and edge3 mounts)."""
    from soarm.predicates import TABLE_CENTER_X, TABLE_HALF
    from soarm.scene_random import sample_table
    fx, fy = perception.TABLE_FOOTPRINT_M["x"], perception.TABLE_FOOTPRINT_M["y"]
    assert TABLE_CENTER_X - TABLE_HALF[0] <= fx[0] and fx[1] <= TABLE_CENTER_X + TABLE_HALF[0] and fy[1] <= TABLE_HALF[1]
    rng = np.random.default_rng(0)
    for mount in ("edge", "edge3"):
        for _ in range(2000):
            t = sample_table(rng, mount)
            assert t["back"] <= fx[0] and t["back"] + t["depth"] >= fx[1]
            assert t["y"] - t["width"] / 2 <= fy[0] and t["y"] + t["width"] / 2 >= fy[1]
