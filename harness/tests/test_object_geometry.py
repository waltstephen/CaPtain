"""Symmetry completion on synthetic half-visible objects; no environment."""
import numpy as np
import pytest
from captain_harness.adapters.capx.object_geometry import fit_circle, min_area_rectangle, measure_cloud, destination_details


def half_cylinder(center=(0.5, 0.1), radius=0.05, z0=0.0, height=0.08, arc=(20, 170), n=600, seed=0):
    rng = np.random.default_rng(seed)
    ang = np.radians(rng.uniform(arc[0], arc[1], n))
    z = rng.uniform(z0, z0 + height, n)
    side = np.column_stack([center[0] + radius * np.cos(ang), center[1] + radius * np.sin(ang), z])
    # visible half of the top disc
    r = radius * np.sqrt(rng.uniform(0, 1, n // 2)); a = np.radians(rng.uniform(arc[0], arc[1], n // 2))
    top = np.column_stack([center[0] + r * np.cos(a), center[1] + r * np.sin(a), np.full(n // 2, z0 + height)])
    pts = np.concatenate([side, top]) + rng.normal(0, 0.0007, (len(side) + len(top), 3))
    return pts


def test_circle_fit_recovers_center_from_a_150_degree_arc():
    pts = half_cylinder()
    from captain_harness.adapters.capx.object_geometry import fit_circle_robust
    (center, radius, rms, coverage), source = fit_circle_robust(pts, top_z=0.08, bottom_z=0.0)
    assert np.linalg.norm(center - [0.5, 0.1]) < 0.002
    assert abs(radius - 0.05) < 0.001
    assert 120 <= coverage <= 180


def test_measure_cloud_cylinder_center_top_bottom_and_uncertainty():
    pts = half_cylinder()
    g = measure_cloud(pts, camera_position=(0.0, 0.1, 0.8), support_z=0.0, plane_rms=0.002)
    assert g["footprint"]["model"] == "cylinder"
    assert np.linalg.norm(np.array(g["center_xy_m"]) - [0.5, 0.1]) < 0.002
    assert abs(g["footprint"]["radius_m"] - 0.05) < 0.001
    assert abs(g["top_z_m"] - 0.08) < 0.003 and g["bottom_z_m"] == 0.0 and abs(g["height_m"] - 0.08) < 0.003
    assert g["quality"] == "measured" and g["center_uncertainty_m"][0] < 0.003
    # the naive visible-surface median is biased toward the camera side; the fit is not
    median_xy = np.median(pts[:, :2], axis=0)
    assert np.linalg.norm(median_xy - [0.5, 0.1]) > 0.01


def test_box_fit_recovers_extents_and_yaw():
    rng = np.random.default_rng(1)
    yaw = np.radians(25); u = np.array([np.cos(yaw), np.sin(yaw)]); v = np.array([-u[1], u[0]])
    a = rng.uniform(-0.06, 0.06, 800); b = rng.uniform(-0.03, 0.03, 800)
    top = np.column_stack([0.4 + a * u[0] + b * v[0], -0.2 + a * u[1] + b * v[1], np.full(800, 0.04)])
    top += rng.normal(0, 0.0005, top.shape)
    g = measure_cloud(top, camera_position=(0.0, 0.0, 0.8), support_z=0.0)
    assert g["footprint"]["model"] == "box"
    w, h = sorted(g["footprint"]["extent_xy_m"], reverse=True)
    assert abs(w - 0.12) < 0.004 and abs(h - 0.06) < 0.004
    assert np.linalg.norm(np.array(g["center_xy_m"]) - [0.4, -0.2]) < 0.002
    assert abs(((g["footprint"]["yaw_rad"] - yaw + np.pi / 2) % np.pi) - np.pi / 2) < np.radians(3)


def test_low_confidence_flags_and_disc_destination():
    pts = half_cylinder(n=30, arc=(60, 100))
    g = measure_cloud(pts, camera_position=(0.0, 0.1, 0.8), support_z=None)
    assert g["quality"] == "low_confidence" and {"few_points", "no_support_plane"} <= set(g["quality_reasons"])
    rng = np.random.default_rng(2)
    r = 0.09 * np.sqrt(rng.uniform(0, 1, 1500)); a = rng.uniform(0, 2 * np.pi, 1500)
    plate = np.column_stack([0.6 + r * np.cos(a), 0.0 + r * np.sin(a), 0.01 + 0.004 * (r > 0.075)])
    d = measure_cloud(plate, camera_position=(0.0, 0.0, 0.8), support_z=0.0)
    det = destination_details(plate, d)
    assert d["footprint"]["model"] == "disc" and abs(d["footprint"]["radius_m"] - 0.09) < 0.003
    assert det["support_surface_kind"] == "floor" and abs(det["support_surface_z_m"] - 0.01) < 0.003


def test_offset_sign_convention():
    obj = half_cylinder(center=(0.62, 0.03), radius=0.04, z0=0.014, height=0.05)
    g = measure_cloud(obj, camera_position=(0.0, 0.0, 0.8), support_z=0.014)
    plate_center = np.array([0.6, 0.0])
    offset = np.array(g["center_xy_m"]) - plate_center
    assert np.allclose(offset, [0.02, 0.03], atol=0.003)
