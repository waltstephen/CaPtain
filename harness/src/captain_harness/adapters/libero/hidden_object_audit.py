"""Host-only goal measurements. Never attach these records to capability results."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = "hidden-object-audit/v2"


def _vector(value) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("expected a finite position vector")
    return value


def _rounded(value):
    return np.asarray(value, dtype=float).round(6).tolist()


class HiddenObjectAudit:
    def __init__(self, env):
        self.env = getattr(env, "env", env)
        self.problem = getattr(self.env, "parsed_problem", None) or {}
        self.regions = self.problem.get("regions") or {}
        self.body_ids = getattr(self.env, "obj_body_id", None) or {}
        self.objects = getattr(self.env, "objects_dict", None) or {}
        self.fixtures = getattr(self.env, "fixtures_dict", None) or {}
        self.sites = getattr(self.env, "object_sites_dict", None) or {}
        self.goals = [list(map(str, goal)) for goal in self.problem.get("goal_state", [])]

    def _entity(self, name: str) -> dict[str, Any]:
        region = self.regions.get(name) or {}
        parent = region.get("target") if isinstance(region, Mapping) else None
        if name in self.sites or name in self.regions:
            kind = "site"
            parent = parent or getattr(self.sites.get(name), "parent_name", None)
        elif name in self.body_ids:
            kind = "object" if name in self.objects else "fixture"
            parent = name
        else:
            kind = "unavailable"
        return {"name": name, "kind": kind, "parent": parent,
                "movable": kind == "object", "approach_supported": kind == "object"}

    def header(self) -> dict[str, Any]:
        # Keep legacy body lists for archived reward versions; v2 consumers use goals.
        entities, targets, destinations = [], [], []
        goals = []
        for index, goal in enumerate(self.goals):
            arguments = [self._entity(name) for name in goal[1:]]
            goals.append({"goal_id": f"goal_{index}", "predicate": goal[0].lower(),
                          "arguments": goal[1:], "target": arguments[0] if arguments else None,
                          "destination": arguments[1] if len(arguments) > 1 else None})
            for offset, entity in enumerate(arguments):
                name = entity["parent"] or entity["name"]
                if name not in self.body_ids:
                    continue
                if name not in entities:
                    entities.append(name)
                bucket = targets if offset == 0 else destinations
                if name not in bucket:
                    bucket.append(name)
        return {"schema_version": SCHEMA_VERSION, "coordinate_frame": "robot_base",
                "goal_state": self.goals, "goals": goals, "entities": entities,
                "targets": targets, "destinations": destinations,
                "movable": [name for name in entities if name in self.objects],
                "fixtures": [name for name in entities if name not in self.objects]}

    def _pose_world(self, name):
        data = self.env.sim.data
        entity = self._entity(name)
        if entity["kind"] == "site":
            return (_vector(data.get_site_xpos(name)),
                    np.asarray(data.get_site_xmat(name), dtype=float).reshape(3, 3))
        if name in self.body_ids:
            index = self.body_ids[name]
            return (_vector(data.body_xpos[index]),
                    np.asarray(data.xmat[index], dtype=float).reshape(3, 3))
        raise ValueError(f"goal entity geometry unavailable: {name}")

    def _region_distance(self, predicate, target, destination, base_rotation, base_origin):
        site = self.sites[destination]
        if getattr(site, "site_type", "box") != "box":
            raise ValueError("unsupported goal region shape")
        center, rotation = self._pose_world(destination)
        size = _vector(site.size)
        if np.any(size < 0):
            raise ValueError("invalid goal region size")
        if predicate == "in":
            # Match LIBERO SiteObject.in_box, including its world-axis bounds.
            half = np.abs(rotation @ size)
            lower, upper = -half, half.copy()
            lower[2] -= .01
            axes = np.eye(3)
        else:
            # Match SiteObject.under's actual matrix convention and height interval.
            lower = np.array([-size[0], -size[1], size[2] - .005])
            upper = np.array([size[0], size[1], size[2] + .10])
            axes = rotation.T
        local = axes.T @ (target - center)
        residual = np.maximum(np.maximum(lower - local, local - upper), 0.)
        box_center = center + axes @ ((lower + upper) / 2.)
        geometry = {"kind": "predicate_box", "site": destination,
                    "center_m": _rounded(base_rotation.T @ (box_center - base_origin)),
                    "rotation_robot_base": _rounded(base_rotation.T @ axes),
                    "half_size_m": _rounded((upper - lower) / 2.),
                    "source": "LIBERO SiteObject.in_box" if predicate == "in" else "LIBERO SiteObject.under"}
        parent = getattr(site, "parent_name", None)
        requires_contact = predicate == "on" and parent in {**self.objects, **self.fixtures}
        return float(np.linalg.norm(residual)), geometry, requires_contact

    def _joint_distance(self, predicate, name):
        entity = self._entity(name)
        owner_name = entity["parent"] or name
        owner = self.objects.get(owner_name) or self.fixtures.get(owner_name)
        holder = self.sites.get(name) if entity["kind"] == "site" else owner
        articulation = (getattr(owner, "object_properties", None) or {}).get("articulation") or {}
        opposite = {"open": "close", "close": "open", "turnon": "turnoff", "turnoff": "turnon"}[predicate]
        desired = np.asarray(articulation.get(f"default_{predicate}_ranges", []), dtype=float)
        other = np.asarray(articulation.get(f"default_{opposite}_ranges", []), dtype=float)
        if desired.shape != (2,) or other.shape != (2,) or not np.isfinite([desired, other]).all():
            raise ValueError("articulation goal ranges unavailable")
        if min(desired) > max(other):
            threshold, span, direction = float(min(desired)), float(min(desired) - max(other)), "above"
        elif max(desired) < min(other):
            threshold, span, direction = float(max(desired)), float(min(other) - max(desired)), "below"
        else:
            raise ValueError("articulation goal ranges overlap")
        rows = []
        model, data = self.env.sim.model, self.env.sim.data
        for joint in getattr(holder, "joints", None) or []:
            address = model.get_joint_qpos_addr(joint)
            if not isinstance(address, (int, np.integer)):
                raise ValueError("goal joint is not scalar")
            qpos = float(data.qpos[address])
            if not np.isfinite(qpos):
                raise ValueError("nonfinite goal joint position")
            distance = max(0., threshold - qpos if direction == "above" else qpos - threshold)
            rows.append({"name": joint, "position": qpos, "goal_range": desired.tolist(),
                         "opposite_range": other.tolist(), "goal_threshold": threshold,
                         "goal_direction": direction, "distance_to_goal": distance,
                         "distance_scale": span, "normalized_distance": distance / span})
        if not rows:
            raise ValueError("articulated goal has no measured joints")
        # LIBERO open/turnon accept any joint; close/turnoff require all joints.
        aggregate = min if predicate in {"open", "turnon"} else max
        return aggregate(row["normalized_distance"] for row in rows), rows

    def sample(self) -> dict[str, Any]:
        model, data = self.env.sim.model, self.env.sim.data
        base = model.body_name2id("robot0_base")
        rotation = np.asarray(data.xmat[base], dtype=float).reshape(3, 3)
        origin = _vector(data.xpos[base])
        to_base = lambda point: _rounded(rotation.T @ (_vector(point) - origin))
        header = self.header()
        states = {}
        for descriptor, goal in zip(header["goals"], self.goals):
            row = {"satisfied": None, "distance_to_goal": None, "distance_unit": None,
                   "distance_scale": None, "progress_kind": "predicate_only", "requires_contact": False,
                   "contact_gap_m": None, "target_position_m": None, "destination_position_m": None,
                   "target_position_kind": descriptor["target"]["kind"] if descriptor["target"] else "unavailable",
                   "target_approach_supported": bool(descriptor["target"] and descriptor["target"]["approach_supported"]),
                   "joints": [], "region_geometry": None}
            try:
                satisfied = self.env._eval_predicate(goal)
                if not isinstance(satisfied, (bool, np.bool_)):
                    raise ValueError("predicate did not return a boolean")
                row["satisfied"] = bool(satisfied)
            except Exception as exc:
                row["error"] = f"predicate_unavailable: {type(exc).__name__}: {exc}"
            predicate = descriptor["predicate"]
            try:
                target, _ = self._pose_world(goal[1])
                row["target_position_m"] = to_base(target)
                if len(goal) == 3:
                    destination, _ = self._pose_world(goal[2])
                    row["destination_position_m"] = to_base(destination)
                if predicate in {"open", "close", "turnon", "turnoff"} and len(goal) == 2:
                    distance, row["joints"] = self._joint_distance(predicate, goal[1])
                    row.update(distance_to_goal=distance, distance_unit="normalized", distance_scale=1., progress_kind="joint_range")
                elif predicate in {"on", "in"} and len(goal) == 3 and goal[2] in self.sites:
                    distance, geometry, contact = self._region_distance(predicate, target, goal[2], rotation, origin)
                    row.update(distance_to_goal=distance, distance_unit="m", distance_scale=.10,
                               progress_kind="region_distance", region_geometry=geometry, requires_contact=contact)
                elif predicate in {"on", "in"} and len(goal) == 3:
                    row.update(requires_contact=True, unsupported_reason="exact_object_contact_distance_unavailable")
                else:
                    row["error"] = f"unsupported_goal_predicate: {predicate}"
            except Exception as exc:
                row["error"] = f"goal_geometry_unavailable: {type(exc).__name__}: {exc}"
                row["target_approach_supported"] = False
            states[descriptor["goal_id"]] = row
        objects = {name: to_base(data.body_xpos[self.body_ids[name]]) for name in header["entities"]}
        return {"schema_version": SCHEMA_VERSION, "objects": objects,
                "eef": to_base(data.xpos[model.body_name2id("gripper0_eef")]), "goal_states": states}
