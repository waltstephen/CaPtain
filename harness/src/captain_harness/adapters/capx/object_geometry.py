"""Symmetry-completed object geometry from public RGB-D only.

Turns a selected detection's visible surface cloud into control-grade numbers: fitted center (circle axis /
rectangle center rather than the visible-surface median), radius or XY extents, top and bottom heights from a
support plane, destination center / rim / support surface, and the object-to-destination XY offset with
uncertainties. Never reads simulator object state; never selects actions.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .backend import _depth_to_base_points, _filter_point_cloud_noise, _fuse_scored_point_clouds
from .effect_geometry import cloud, provenance
from .visible_crossview import visible_seed, overlap_receipt
from captain_harness.adapters.robotics.surfaces import support_plane

SHAPES = ("auto", "cylinder", "box", "disc")
MIN_POINTS = 30
LOW_CONFIDENCE_POINTS = 60
LOW_CONFIDENCE_ARC_DEG = 60.0
TOP_SLAB_M = 0.015
RIM_SLAB_M = 0.010
from captain_harness.core.placement_standard import LOW_CONFIDENCE_SCORE
LIMITATION = ("Symmetry-completed fit of visible RGB-D surfaces in robot_base; identity is the selected hypothesis; "
              "bottom height assumes the object rests on the fitted support plane; occluded parts are inferred, not seen.")


# ----------------------------------------------------------------------------- pure fits

def fit_circle(xy):
    """Taubin algebraic circle fit. Returns center (2,), radius, rms radial residual, arc coverage in degrees."""
    xy = np.asarray(xy, float)
    if len(xy) < 5:
        raise ValueError("circle fit needs at least 5 points")
    mean = xy.mean(axis=0)
    x, y = xy[:, 0] - mean[0], xy[:, 1] - mean[1]
    z = x * x + y * y
    zm = z.mean()
    if zm <= 1e-12:
        raise ValueError("degenerate circle fit")
    z0 = (z - zm) / (2.0 * np.sqrt(zm))
    _, _, vt = np.linalg.svd(np.column_stack([z0, x, y]), full_matrices=False)
    a = vt[-1]
    a0 = a[0] / (2.0 * np.sqrt(zm))
    if abs(a0) < 1e-12:
        raise ValueError("degenerate circle fit")
    a3 = -zm * a0
    cx, cy = -a[1] / (2 * a0), -a[2] / (2 * a0)
    radius = float(np.sqrt(max(a[1] ** 2 + a[2] ** 2 - 4 * a0 * a3, 0.0)) / (2 * abs(a0)))
    center = np.array([cx + mean[0], cy + mean[1]])
    dist = np.linalg.norm(xy - center, axis=1)
    rms = float(np.sqrt(np.mean((dist - radius) ** 2)))
    ang = np.degrees(np.arctan2(xy[:, 1] - center[1], xy[:, 0] - center[0]))
    bins = np.unique(np.floor((ang + 180.0) / 10.0).astype(int) % 36)
    return center, radius, rms, float(10.0 * len(bins))


def min_area_rectangle(xy):
    """Minimum-area bounding rectangle via convex-hull edge directions. Returns center, (w, h), yaw, rms."""
    xy = np.asarray(xy, float)
    if len(xy) < 3:
        raise ValueError("rectangle fit needs at least 3 points")
    try:
        hull = ConvexHull(xy)
    except QhullError as exc:
        raise ValueError("points do not span an area") from exc
    hp = xy[hull.vertices]
    best = None
    for i in range(len(hp)):
        e = hp[(i + 1) % len(hp)] - hp[i]
        n = np.linalg.norm(e)
        if n < 1e-9:
            continue
        u = e / n
        v = np.array([-u[1], u[0]])
        pu, pv = xy @ u, xy @ v
        w, h = pu.max() - pu.min(), pv.max() - pv.min()
        area = w * h
        if best is None or area < best[0]:
            cu, cv = (pu.max() + pu.min()) / 2, (pv.max() + pv.min()) / 2
            best = (area, cu * u + cv * v, (w, h), float(np.arctan2(u[1], u[0])), u, v)
    _, center, (w, h), yaw, u, v = best
    # residual: distance of hull points to the nearest rectangle side
    pu, pv = hp @ u - (center @ u), hp @ v - (center @ v)
    res = np.minimum(np.abs(np.abs(pu) - w / 2), np.abs(np.abs(pv) - h / 2))
    return center, (float(w), float(h)), yaw, float(np.sqrt(np.mean(res ** 2)))


def choose_slab(points, top_z, bottom_z):
    """Top-face slab when it is well observed, else the body above the support plane."""
    top = points[points[:, 2] > top_z - TOP_SLAB_M]
    if len(top) >= 40:
        return top, "top_slab"
    if bottom_z is not None:
        body = points[points[:, 2] > bottom_z + 0.005]
        if len(body) >= MIN_POINTS:
            return body, "body_above_support"
    return points, "all_points"


def _hull_vertices(xy):
    try:
        hull = ConvexHull(xy)
    except QhullError as exc:
        raise ValueError("points do not span an area") from exc
    return xy[hull.vertices]


def fit_circle_robust(points, top_z, bottom_z):
    """Circle through the object's outline, not its filled top face: side-wall points (cylinders) or the
    convex-hull vertices of the top slab (bowls, plates, cans seen from above). Returns (fit, source)."""
    lower = bottom_z + 0.005 if bottom_z is not None else -np.inf
    side = points[(points[:, 2] < top_z - TOP_SLAB_M) & (points[:, 2] > lower)]
    top = points[points[:, 2] > top_z - TOP_SLAB_M]
    candidates = []
    if len(side) >= 30:
        try:
            candidates.append(("side_walls", fit_circle(side[:, :2]), side[:, :2]))
        except ValueError:
            pass
    for name, sub in (("top_hull", top), ("all_hull", points)):
        if len(sub) >= 8:
            try:
                hv = _hull_vertices(sub[:, :2])
                if len(hv) >= 5:
                    candidates.append((name, fit_circle(hv), hv))
            except ValueError:
                pass
    if not candidates:
        raise ValueError("circle fit unavailable")
    good = [c for c in candidates if c[1][3] >= LOW_CONFIDENCE_ARC_DEG]
    name, fit, xy = min(good or candidates, key=lambda c: c[1][2])
    center, radius, rms, _ = fit
    keep = np.abs(np.linalg.norm(xy - center, axis=1) - radius) <= max(2.5 * rms, 0.002)
    if keep.sum() >= 5 and keep.sum() < len(xy):
        try:
            fit = fit_circle(xy[keep])
        except ValueError:
            pass
    return fit, name


def footprint_fit(points, shape, top_z, bottom_z, camera_position):
    slab, slab_kind = choose_slab(points, top_z, bottom_z)
    xy = slab[:, :2]
    circle = rect = None
    circle_source = None
    try:
        circle, circle_source = fit_circle_robust(points, top_z, bottom_z)
    except ValueError:
        pass
    try:
        rect = min_area_rectangle(xy)
    except ValueError:
        pass
    if circle is None and rect is None:
        raise ValueError("footprint fit unavailable")
    height = float(top_z - bottom_z) if bottom_z is not None else None
    model = shape
    if shape == "auto":
        # A partially visible round object has a non-square hull, so compare fit residuals, not aspect.
        round_ok = circle is not None and circle[2] < 0.003 and (rect is None or circle[2] <= rect[3] + 0.001)
        if round_ok:
            model = "disc" if (height is not None and height < 0.025) else "cylinder"
        else:
            model = "box" if rect is not None else "cylinder"
    if model in ("cylinder", "disc") and circle is None:
        model = "box"
    if model == "box" and rect is None:
        model = "cylinder"
    cam_xy = np.asarray(camera_position, float)[:2]
    if model in ("cylinder", "disc"):
        center, radius, rms, coverage = circle
        ray = center - cam_xy
        far = bool(np.any(((xy - center) @ ray) > 0.5 * radius * np.linalg.norm(ray)))
        return {"model": model, "center_xy_m": center.tolist(), "radius_m": radius, "fit_rms_m": rms,
                "arc_coverage_deg": coverage, "far_side_observed": far, "slab": slab_kind, "slab_points": int(len(slab)),
                "fit_source": circle_source}
    center, (w, h), yaw, rms = rect
    ray = center - cam_xy
    far = bool(np.any(((xy - center) @ ray) > 0.35 * max(w, h) * np.linalg.norm(ray)))
    return {"model": "box", "center_xy_m": center.tolist(), "extent_xy_m": [w, h], "yaw_rad": yaw, "fit_rms_m": rms,
            "arc_coverage_deg": None, "far_side_observed": far, "slab": slab_kind, "slab_points": int(len(slab))}


def measure_cloud(points, camera_position, *, shape="auto", support_z=None, plane_rms=None, views_used=1):
    """Pure geometry of one filtered object cloud (robot_base frame)."""
    points = np.asarray(points, float)
    if len(points) < MIN_POINTS:
        raise ValueError("insufficient measured object surface")
    top_z = float(np.percentile(points[:, 2], 98))
    bottom_z = float(support_z) if support_z is not None else float(points[:, 2].min())
    fp = footprint_fit(points, shape, top_z, bottom_z, camera_position)
    height = top_z - bottom_z
    center_xy = np.asarray(fp["center_xy_m"])
    center = [float(center_xy[0]), float(center_xy[1]), float((top_z + bottom_z) / 2)]
    coverage = fp.get("arc_coverage_deg")
    cov_frac = (coverage / 360.0) if coverage else 0.5
    rms = fp["fit_rms_m"]
    sxy = rms / np.sqrt(max(cov_frac, 0.1))
    if fp["model"] == "box" and not fp["far_side_observed"]:
        sxy += 0.25 * max(fp["extent_xy_m"])
    quality = "measured"
    reasons = []
    if len(points) < LOW_CONFIDENCE_POINTS:
        reasons.append("few_points")
    if coverage is not None and coverage < LOW_CONFIDENCE_ARC_DEG:
        reasons.append("small_arc_coverage")
    if support_z is None:
        reasons.append("no_support_plane")
    if reasons:
        quality = "low_confidence"
    return {"center_m": center, "center_xy_m": center_xy.tolist(), "top_z_m": top_z, "bottom_z_m": bottom_z, "height_m": float(height),
            "footprint": fp, "support_plane_z_m": support_z, "support_plane_source": "scene_ransac" if support_z is not None else "unavailable",
            "center_uncertainty_m": [float(sxy), float(sxy), float(plane_rms if plane_rms is not None else 0.003)],
            "views_used": int(views_used), "point_count": int(len(points)), "quality": quality, "quality_reasons": reasons}


def destination_details(points, geometry):
    """Rim radius and support surface height of a destination (plate/tray/bowl/basket)."""
    points = np.asarray(points, float)
    fp = geometry["footprint"]
    top_z = geometry["top_z_m"]
    out = {}
    rim = points[points[:, 2] > top_z - RIM_SLAB_M]
    if fp["model"] in ("cylinder", "disc"):
        radius = fp["radius_m"]
        try:
            if len(rim) >= 8:
                rc, rr, _, _ = fit_circle(rim[:, :2])
                out["rim_radius_m"] = float(rr)
                out["rim_center_xy_m"] = rc.tolist()
        except ValueError:
            pass
        bowl_like = geometry["height_m"] > 0.3 * max(radius, 1e-3)
        inside = points[np.linalg.norm(points[:, :2] - np.asarray(fp["center_xy_m"]), axis=1) < 0.6 * radius]
        if bowl_like or len(inside) < 10:
            out["support_surface_z_m"] = float(top_z); out["support_surface_kind"] = "rim"
        else:
            out["support_surface_z_m"] = float(np.median(inside[:, 2])); out["support_surface_kind"] = "floor"
    else:
        w, h = fp["extent_xy_m"]
        yaw = fp["yaw_rad"]
        u = np.array([np.cos(yaw), np.sin(yaw)]); v = np.array([-u[1], u[0]])
        rel = points[:, :2] - np.asarray(fp["center_xy_m"])
        inner = points[(np.abs(rel @ u) < 0.3 * w) & (np.abs(rel @ v) < 0.3 * h)]
        if len(inner) >= 10:
            out["support_surface_z_m"] = float(np.median(inner[:, 2])); out["support_surface_kind"] = "floor"
        else:
            out["support_surface_z_m"] = float(top_z); out["support_surface_kind"] = "rim"
        wall = 0.01
        out["inner_extent_xy_m"] = [max(w - 2 * wall, 0.0), max(h - 2 * wall, 0.0)]
        out["inner_extent_assumption"] = "rectangle shrunk by an assumed 1 cm wall"
    return out


# ----------------------------------------------------------------------------- honesty / relation helpers

def footprint_area_m2(fp):
    if fp["model"] in ("cylinder", "disc"):
        return float(np.pi * fp["radius_m"] ** 2)
    w, h = fp["extent_xy_m"]
    return float(w * h)


def point_in_footprint(xy, fp, margin_m=0.0):
    rel = np.asarray(xy, float) - np.asarray(fp["center_xy_m"], float)
    if fp["model"] in ("cylinder", "disc"):
        return bool(np.linalg.norm(rel) <= fp["radius_m"] + margin_m)
    w, h = fp["extent_xy_m"]
    yaw = fp["yaw_rad"]
    u = np.array([np.cos(yaw), np.sin(yaw)]); v = np.array([-u[1], u[0]])
    return bool(abs(rel @ u) <= w / 2 + margin_m and abs(rel @ v) <= h / 2 + margin_m)


def mask_iou(a, b):
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    if a.shape != b.shape:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def same_surface(target, destination):
    """True when two detections describe the same visible surface: same mask, or the same size at the same place."""
    if target is destination:
        return True
    if (getattr(target, "camera_name", None) == getattr(destination, "camera_name", None)
            and getattr(target, "observation_handle", None) == getattr(destination, "observation_handle", None)
            and mask_iou(target.mask, destination.mask) > 0.5):
        return True
    ext_t = np.asarray(target.bounds_max, float) - np.asarray(target.bounds_min, float)
    ext_d = np.asarray(destination.bounds_max, float) - np.asarray(destination.bounds_min, float)
    same_size = bool(np.all(np.abs(ext_t - ext_d) <= 0.1 * np.maximum(ext_t, ext_d) + 0.005))
    return same_size and float(np.linalg.norm(np.asarray(target.centroid, float) - np.asarray(destination.centroid, float))) < 0.01


def placement_relation(obj, dst):
    """Containment / resting fields between a measured object and destination (pure)."""
    inside = point_in_footprint(obj["center_xy_m"], dst["footprint"])
    support = dst.get("support_surface_z_m", dst["top_z_m"])
    bottom_gap = float(obj["bottom_z_m"] - support)
    return {"object_center_inside_destination_footprint": bool(inside),
            "object_bottom_minus_destination_support_m": bottom_gap,
            "resting_height_consistent": bool(abs(bottom_gap) <= 0.03),
            "object_inside_destination_volume": bool(inside and obj["bottom_z_m"] < dst["top_z_m"] + 0.005)}


# ----------------------------------------------------------------------------- tool-bound helpers

def fuse_views(tools, obs, primary_camera, primary_score, points):
    receipts = []
    fused = [(points, float(primary_score or 0.5))]
    for cam in obs.cameras:
        if cam.camera_name == primary_camera.camera_name or cam.depth is None:
            continue
        seed, receipt = visible_seed(points, cam)
        receipt = {"camera": cam.camera_name, **receipt}
        if seed is None:
            receipts.append(receipt); continue
        try:
            candidates = tools.services.segment_point(cam.image, seed)
        except Exception as exc:  # service failure must not break the measurement
            receipts.append({**receipt, "status": "service_error", "error": str(exc)[:120]}); continue
        best = None
        for cand in candidates:
            pts = _filter_point_cloud_noise(_depth_to_base_points(cam.depth, cam.intrinsics, cam.extrinsics, mask=cand.mask))
            if len(pts) < 6:
                continue
            ov = overlap_receipt(points, pts)
            if ov["supported"] and (best is None or cand.score > best[1]):
                best = (pts, float(cand.score), ov)
        if best is None:
            receipts.append({**receipt, "status": "no_overlapping_mask"}); continue
        fused.append((best[0], best[1]))
        receipts.append({**receipt, "status": "fused", "points": int(len(best[0])), "overlap": best[2]})
    return _fuse_scored_point_clouds(fused) if len(fused) > 1 else points, len(fused), receipts


def strip_robot_points(tools, obs, points):
    try:
        model = tools._motion_model()
        state = obs.robot_state
        model.set_state(state.joint_positions_rad, state.gripper_open_fraction)
        own, _ = model.points_in_robot(points, margin=0.01, include_visual=True)
        return points[~own], int(own.sum())
    except Exception:
        return points, 0


def scene_support_z(camera, points):
    if len(points) < 3:
        return None, None
    scene = _depth_to_base_points(camera.depth, camera.intrinsics, camera.extrinsics)
    if len(scene) > 20000:
        scene = scene[np.linspace(0, len(scene) - 1, 20000, dtype=int)]
    plane = support_plane(scene, points, neighborhood_m=0.08)
    if plane is None:
        return None, None
    normal, offset, count = plane
    if abs(normal[2]) < 0.7:
        return None, None
    center = np.median(points, axis=0)
    z = -(offset + normal[0] * center[0] + normal[1] * center[1]) / normal[2]
    return float(z), 0.003


def _one(tools, handle, shape, cameras, fresh=True):
    detection, obs, camera, points = cloud(tools, handle, fresh=fresh)
    views = 1; fusion = []
    if cameras == "all":
        points, views, fusion = fuse_views(tools, obs, camera, detection.score, points)
    stripped, removed = strip_robot_points(tools, obs, points)
    if len(stripped) >= MIN_POINTS:
        points = stripped
    else:
        removed = 0  # the object sits inside the gripper envelope (small target near the pads): keep the full cloud
    support_z, plane_rms = scene_support_z(camera, points)
    geometry = measure_cloud(points, camera.extrinsics[:3, 3], shape=shape, support_z=support_z, plane_rms=plane_rms, views_used=views)
    geometry["gripper_points_removed"] = removed
    geometry["fusion"] = fusion
    score = getattr(detection, "score", None)
    geometry["detection_score"] = None if score is None else float(score)
    rgb = getattr(detection, "appearance_rgb_mean", None)
    geometry["appearance_rgb_mean"] = None if rgb is None else [round(float(v), 1) for v in np.asarray(rgb).ravel()[:3]]
    if score is not None and float(score) < LOW_CONFIDENCE_SCORE:
        geometry["quality"] = "low_confidence"
        geometry["quality_reasons"] = list(geometry.get("quality_reasons") or []) + ["low_confidence_detection"]
    return detection, obs, camera, points, geometry


def part_motion(before, after):
    """Displacement and in-plane rotation of the same part measured twice (pure). Rotation uses the rectangle yaw,
    with width/height axis swaps removed; rectangles remain ambiguous modulo 180 degrees, squares modulo 90."""
    d = np.asarray(after["center_m"], float) - np.asarray(before["center_m"], float)
    dist_xy = float(np.hypot(d[0], d[1]))
    rot = None
    period = None
    basis = "unavailable"
    if before["footprint"].get("model") == "box" and after["footprint"].get("model") == "box":
        axes = []
        for geometry in (before, after):
            fp = geometry["footprint"]
            size = np.asarray(fp.get("extent_xy_m", []), float)
            if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0):
                break
            yaw = float(fp["yaw_rad"])
            if not np.isfinite(yaw):
                break
            square = abs(size[0] - size[1]) <= max(.003, 3. * float(fp.get("fit_rms_m") or 0.))
            # The rectangle fitter may exchange its two axes without physical motion.
            axes.append((yaw if square or size[0] >= size[1] else yaw + np.pi / 2, square))
        if len(axes) == 2 and axes[0][1] == axes[1][1]:
            period = np.pi / 2 if axes[0][1] else np.pi
            basis = "square_footprint" if axes[0][1] else "rectangle_long_axis"
            delta = axes[1][0] - axes[0][0]
            rot = float((delta + period / 2) % period - period / 2)
    # Repeatability, not absolute accuracy, bounds a before/after difference: the same part seen from the same camera
    # carries the same occlusion bias in both frames, so only the fit residuals enter (floor 3 mm).
    unc = float(max(0.003, np.hypot(before["footprint"].get("fit_rms_m") or 0.0, after["footprint"].get("fit_rms_m") or 0.0)))
    significant = dist_xy > max(0.01, 3.0 * unc) or (rot is not None and abs(rot) > np.radians(10))
    return {"part_displacement_m": d.tolist(), "part_distance_m": dist_xy, "part_rotation_rad": rot,
            "part_rotation_basis": basis, "part_rotation_symmetry_period_rad": period,
            "part_motion_uncertainty_m": unc, "part_motion_significant": bool(significant),
            "part_motion_convention": "after minus before in robot_base; rotation is the smallest equivalent in-plane shape rotation, "
                "modulo 180 deg for elongated rectangles or 90 deg for square footprints; symmetric turns and full revolutions are not observable from these fits"}


def measure(tools, target_detection_id, destination_detection_id=None, shape="auto", cameras="all", before_detection_id=None):
    if shape not in SHAPES:
        raise ValueError("shape must be auto, cylinder, box or disc")
    if cameras not in ("all", "primary"):
        raise ValueError("cameras must be all or primary")
    target, obs, cam, tpoints, tgeo = _one(tools, target_detection_id, shape, cameras)
    result = {"object": tgeo, "target_source": provenance(target_detection_id, target, obs, cam),
              "physical_action_executed": False, "limitation": LIMITATION}
    if destination_detection_id is not None:
        if destination_detection_id == target_detection_id:
            raise ValueError("destination_is_target: the destination must be a different detection (the support surface or "
                             "container), not the object itself")
        dest, dobs, dcam, dpoints, dgeo = _one(tools, destination_detection_id, "auto", cameras)
        if abs(obs.time_s - dobs.time_s) > 1e-9:
            raise ValueError("target and destination must share a current observation time")
        if same_surface(target, dest):
            raise ValueError("destination_is_target: both detections cover the same visible surface (same mask or identical "
                             "size and place); select the container or support surface, not the object again")
        dgeo.update(destination_details(dpoints, dgeo))
        # V1.2: a reference smaller than the object (a bowl on a ramekin, a mug on a coaster) is a legitimate thing to
        # measure against when verifying which instance the task refers to. Report the geometry and flag it instead of
        # refusing; the completion gate still refuses to accept such a record as the destination of a placement goal.
        smaller = footprint_area_m2(dgeo["footprint"]) < footprint_area_m2(tgeo["footprint"])
        offset = np.asarray(tgeo["center_xy_m"]) - np.asarray(dgeo["center_xy_m"])
        su = np.hypot(tgeo["center_uncertainty_m"][0], dgeo["center_uncertainty_m"][0])
        result.update({"destination": dgeo, "destination_source": provenance(destination_detection_id, dest, dobs, dcam),
                       "object_to_destination_xy_offset_m": offset.tolist(), "xy_distance_m": float(np.linalg.norm(offset)),
                       "xy_offset_uncertainty_m": float(su), **placement_relation(tgeo, dgeo),
                       "destination_smaller_than_object": bool(smaller),
                       "offset_convention": "object center minus destination center in robot_base XY; move the object by the negative offset to center it"})
        if smaller:
            result["destination_note"] = ("destination_smaller_than_object: the destination footprint (%.4f m2) is smaller than the object "
                                          "footprint (%.4f m2). Fine for checking which instance rests on a small reference; as the final "
                                          "destination of a placement it cannot support or contain the object (fragment or wrong detection)."
                                          % (footprint_area_m2(dgeo["footprint"]), footprint_area_m2(tgeo["footprint"])))
    if before_detection_id is not None:
        if before_detection_id == target_detection_id:
            raise ValueError("before_detection_id must be a detection of the same part from an EARLIER observation")
        before, bobs, bcam, _bpoints, bgeo = _one(tools, before_detection_id, shape, cameras, fresh=False)
        if bcam.camera_name != cam.camera_name:
            raise ValueError("before_detection_id must come from the same camera as the target detection")
        if bobs.time_s >= obs.time_s:
            raise ValueError("before_detection_id must come from an observation taken before the target's observation")
        result.update(part_motion(bgeo, tgeo))
        result.update({"before": bgeo, "before_source": provenance(before_detection_id, before, bobs, bcam),
                       "before_time_s": float(bobs.time_s), "after_time_s": float(obs.time_s)})
    return result
