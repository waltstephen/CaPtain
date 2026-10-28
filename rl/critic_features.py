"""Privileged-state features for the simulator critic V(s_t).

One row per student request t, describing the simulator state BEFORE request t
(hidden_request_marks.jsonl, mark t) plus policy-information features computed from
calls made in requests < t.  Pure functions: the RL trainer can import
`episode_states(episode_dir)` and `FEATURES` to score new rollouts.

Calls belonging to request t are those with mark[t].call_index < call_index <= mark[t+1].call_index
(call_index starts at 1; the mark stores the number of calls completed before the request).

The
trainer scores rollouts with rl/physics_critic.py, which selects a subset of these features per critic (feature_spec.json).
"""
import json, math, re

FEATURE_VERSION = "physics-critic-features/v1"
PERCEPTION = {"segment", "mark_point", "point", "measure_object_geometry", "select_detection", "track_target",
              "compare_object_observations", "measure_image_points", "inspect_contact_geometry",
              "inspect_release_geometry"}
GRIP_RE = re.compile(r"gripper_open_fraction=([0-9.eE+-]+)->([0-9.eE+-]+)")

# phase thresholds (metres)
NEAR_EEF = 0.10      # eef within 10 cm of the active target object -> pre-grasp
HOLD_EEF = 0.08      # object within 8 cm of eef while gripper closed and lifted -> held
LIFT_Z = 0.02        # object z above its initial z
NEAR_GOAL_M = 0.05   # distance_to_goal below 5 cm (or normalized < 0.25) -> near-goal
NEAR_GOAL_NORM = 0.25
MOVED_XY = 0.05      # held object moved >5 cm horizontally from its start -> transport


def jl(p):
    try:
        with open(p) as f:
            return [json.loads(l) for l in f if l.strip()]
    except Exception:
        return []


def _d(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def classify_request(calls):
    """Request type from the calls made during one request (priority order)."""
    names = [c.get("name") for c in calls]
    grips = [str((c.get("arguments") or {}).get("gripper")) for c in calls if c.get("name") == "move_to"]
    if "close" in grips:
        return "grasp_close"
    if "open" in grips or "release" in names:
        return "release_open"
    if "move_to" in names:
        return "move"
    if "recover" in names:
        return "recover"
    if "hold" in names:
        return "hold"
    if any(n in PERCEPTION for n in names):
        return "perception"
    if names:
        return "observe_query"   # observe / preview_motion / robot_geometry / gripper_state ...
    return "no_call"             # python error before any tool call, or text only


def _grip_after(c, cur):
    m = GRIP_RE.search(str(c.get("receipt") or ""))
    if m:
        try:
            return float(m.group(2))
        except ValueError:
            pass
    v = c.get("value")
    if isinstance(v, dict) and isinstance(v.get("measured_open_fraction"), (int, float)):
        return float(v["measured_open_fraction"])
    return cur


def episode_states(d, task_key=None):
    """Return (rows, meta) for one episode dir. rows: list of dicts (features + bookkeeping)."""
    marks = sorted(jl(d + "/hidden_request_marks.jsonl"), key=lambda m: m["request_index"])
    if not marks:
        return [], {"error": "no_marks"}
    calls = sorted([c for c in jl(d + "/calls.jsonl") if isinstance(c.get("call_index"), int)],
                   key=lambda c: c["call_index"])
    reqs = {int(r.get("index", -1)): r for r in jl(d + "/driver/requests.jsonl")}
    gids = sorted(marks[0]["goal_states"])
    g0 = marks[0]["goal_states"]
    obj0 = {k: v for k, v in (marks[0].get("objects") or {}).items()}
    tp0 = {g: (g0[g].get("target_position_m") or None) for g in gids}
    dist0 = {g: g0[g].get("distance_to_goal") for g in gids}
    j0 = {g: list(g0[g].get("joints") or []) for g in gids}
    is_joint = any(j0[g] for g in gids) or any("joint" in str(g0[g].get("progress_kind")) for g in gids)

    # assign calls to requests
    bounds = [m["call_index"] for m in marks] + [10 ** 9]
    per_req = [[] for _ in marks]
    k = 0
    for c in calls:
        ci = c["call_index"]
        while k + 1 < len(bounds) - 1 and ci > bounds[k + 1]:
            k += 1
        if ci > bounds[k]:
            per_req[k].append(c)
        else:  # before first mark's bound (should not happen)
            per_req[0].append(c)

    rows = []
    grip = 1.0
    n_perc = n_phys = n_fail_phys = n_fail = n_close = n_recover = n_nocall = 0
    last_phys_req = 0
    ever_held = False
    for t, m in enumerate(marks):
        gs = m["goal_states"]
        eef = m.get("eef") or [0, 0, 0]
        sat = [bool((gs.get(g) or {}).get("satisfied")) for g in gids]
        nsat = sum(sat)
        # staged subgoal reward (bookkeeping only, not a critic feature): per goal (gids order) the share of the goal's initial
        # distance closed at this mark, 1 - d_t / d_0 (None when either distance is missing or d_0 <= 0)
        closure = []
        for g in gids:
            x = (gs.get(g) or {}).get("distance_to_goal")
            closure.append(1.0 - float(x) / float(dist0[g]) if isinstance(x, (int, float)) and isinstance(dist0.get(g), (int, float))
                           and dist0[g] > 0 else None)
        act = next((g for g, s in zip(gids, sat) if not s), gids[-1] if gids else None)
        ga = gs.get(act) or {}
        tp = ga.get("target_position_m") or None
        dest = ga.get("destination_position_m") or None
        dtg = ga.get("distance_to_goal")
        d0 = dist0.get(act)
        f = {}
        f["eef_x"], f["eef_y"], f["eef_z"] = eef[:3]
        f["grip_open"] = grip
        f["grip_closed"] = float(grip < 0.5)
        f["n_goals"] = len(gids)
        f["n_sat"] = nsat
        f["frac_sat"] = nsat / max(1, len(gids))
        f["all_sat"] = float(nsat == len(gids) and len(gids) > 0)
        f["active_goal_idx"] = gids.index(act) if act in gids else 0
        f["is_joint_task"] = float(is_joint)
        f["active_is_joint"] = float(bool(ga.get("joints")) or "joint" in str(ga.get("progress_kind")))
        f["requires_contact"] = float(bool(ga.get("requires_contact")))
        cg = ga.get("contact_gap_m")
        f["contact_gap"] = float(cg) if isinstance(cg, (int, float)) else -1.0
        f["has_contact_gap"] = float(isinstance(cg, (int, float)))
        f["dist_to_goal"] = float(dtg) if isinstance(dtg, (int, float)) else -1.0
        f["norm_dist"] = (float(dtg) / max(1e-3, float(d0))) if isinstance(dtg, (int, float)) and isinstance(d0, (int, float)) and d0 > 0 else -1.0
        nd = []
        for g in gids:
            x = (gs.get(g) or {}).get("distance_to_goal")
            if isinstance(x, (int, float)) and isinstance(dist0.get(g), (int, float)) and dist0[g] > 0:
                nd.append(min(3.0, x / max(1e-3, dist0[g])))
        f["norm_dist_mean"] = sum(nd) / len(nd) if nd else -1.0
        f["norm_dist_min"] = min(nd) if nd else -1.0
        if tp:
            rel = [tp[i] - eef[i] for i in range(3)]
            f["tgt_rel_x"], f["tgt_rel_y"], f["tgt_rel_z"] = rel
            f["eef_tgt_dist"] = _d(tp, eef)
            t0 = tp0.get(act) or tp
            f["tgt_lift"] = tp[2] - t0[2]
            f["tgt_moved_xy"] = math.hypot(tp[0] - t0[0], tp[1] - t0[1])
        else:
            f["tgt_rel_x"] = f["tgt_rel_y"] = f["tgt_rel_z"] = 0.0
            f["eef_tgt_dist"] = -1.0
            f["tgt_lift"] = f["tgt_moved_xy"] = 0.0
        f["has_target_pos"] = float(bool(tp))
        if tp and dest:
            r2 = [dest[i] - tp[i] for i in range(3)]
            f["dst_rel_x"], f["dst_rel_y"], f["dst_rel_z"] = r2
            f["tgt_dst_dist"] = _d(dest, tp)
        else:
            f["dst_rel_x"] = f["dst_rel_y"] = f["dst_rel_z"] = 0.0
            f["tgt_dst_dist"] = -1.0
        # nearest other object to the target (clutter / blocking)
        objs = m.get("objects") or {}
        if tp and objs:
            ds = [_d(tp, p) for p in objs.values() if p and _d(tp, p) > 1e-4]
            f["tgt_nearest_other"] = min(ds) if ds else 1.0
        else:
            f["tgt_nearest_other"] = 1.0
        # max displacement of any object from its initial pose (knocked over / disturbed scene)
        mv = [(_d(p, obj0[k2]) if k2 in obj0 and p else 0.0) for k2, p in objs.items()]
        f["max_obj_disp"] = max(mv) if mv else 0.0
        js = ga.get("joints") or []
        jj0 = j0.get(act) or []
        f["joint_pos"] = float(js[0]) if js and isinstance(js[0], (int, float)) else 0.0
        f["joint_delta"] = (float(js[0]) - float(jj0[0])) if js and jj0 and isinstance(js[0], (int, float)) and isinstance(jj0[0], (int, float)) else 0.0
        held = bool(tp) and grip < 0.5 and f["eef_tgt_dist"] < HOLD_EEF and f["tgt_lift"] > LIFT_Z
        ever_held = ever_held or held
        f["held"] = float(held)
        f["ever_held"] = float(ever_held)
        # policy-information features (from requests < t)
        f["req_idx"] = m["request_index"]
        f["env_step"] = m.get("environment_step") or 0
        f["perc_called"] = float(n_perc > 0)
        f["n_perc"] = n_perc
        f["n_phys"] = n_phys
        f["n_fail_phys"] = n_fail_phys
        f["n_fail_calls"] = n_fail
        f["n_close"] = n_close
        f["n_recover"] = n_recover
        f["n_nocall_req"] = n_nocall
        f["req_since_phys"] = m["request_index"] - last_phys_req if last_phys_req else m["request_index"]
        # phase
        if f["all_sat"]:
            ph = "solved"
        elif (not tp) or ga.get("joints") or "joint" in str(ga.get("progress_kind")) or ga.get("target_position_kind") not in (None, "object"):
            ph = "articulated"
        elif isinstance(dtg, (int, float)) and (dtg < NEAR_GOAL_M or (f["norm_dist"] >= 0 and f["norm_dist"] < NEAR_GOAL_NORM)):
            ph = "near_goal"
        elif held:
            ph = "transport" if f["tgt_moved_xy"] > MOVED_XY else "held_lifted"
        elif f["eef_tgt_dist"] > NEAR_EEF:
            ph = "pre_approach"
        else:
            ph = "pre_grasp"
        rq = reqs.get(m["request_index"]) or {}
        rows.append({"f": f, "phase": ph, "request_index": m["request_index"],
                     "replayed": str(rq.get("replayed")) == "True",
                     "req_type": classify_request(per_req[t]),
                     "req_failed_phys": sum(1 for c in per_req[t] if c.get("consumes_action") and not c.get("success")),
                     # reward costs: physical actions of this request, and the bad ones (failed, rejected or stopped
                     # short: endpoint not reached, e.g. a contact stop); perception calls (never charged)
                     "req_phys": sum(1 for c in per_req[t] if c.get("consumes_action")),
                     "req_phys_bad": sum(1 for c in per_req[t] if c.get("consumes_action") and (not c.get("success") or (
                         isinstance(c.get("value"), dict) and c["value"].get("reached") is False))),
                     "req_perc": sum(1 for c in per_req[t] if c.get("name") in PERCEPTION),
                     "active_goal": act, "_tp_lift": f["tgt_lift"], "_held": held,
                     "goal_sat": sat, "goal_closure": closure, "goal_ids": gids,
                     # staged-reward ladder bookkeeping (host marks only; not critic features): per goal the raw distance to the goal,
                     # the target object's position (object targets only), the destination site position; the gripper position
                     "goal_dist": [(gs.get(g) or {}).get("distance_to_goal") for g in gids],
                     "goal_tp": [((gs.get(g) or {}).get("target_position_m") or None) if (gs.get(g) or {}).get("target_position_kind") == "object"
                                 else None for g in gids],
                     "goal_dest": [(gs.get(g) or {}).get("destination_position_m") or None for g in gids],
                     "eef": list(m["eef"]) if m.get("eef") else None})
        # update running counters with request t's calls
        cs = per_req[t]
        if not cs:
            n_nocall += 1
        for c in cs:
            nm = c.get("name")
            if nm in PERCEPTION:
                n_perc += 1
            if c.get("consumes_action"):
                n_phys += 1
                last_phys_req = m["request_index"]
                if not c.get("success"):
                    n_fail_phys += 1
            if not c.get("success"):
                n_fail += 1
            if nm == "move_to" and str((c.get("arguments") or {}).get("gripper")) == "close":
                n_close += 1
            if nm == "recover":
                n_recover += 1
            if nm in ("move_to", "hold", "release", "recover", "gripper_state"):
                grip = _grip_after(c, grip)
    # auxiliary label: the goal active at t gets held (closed gripper, lifted, near eef) at some later state
    held_by_goal = {}
    for t in range(len(rows) - 1, -1, -1):
        g = rows[t]["active_goal"]
        rows[t]["aux_held_later"] = bool(held_by_goal.get(g, False))
        if rows[t]["_held"]:
            held_by_goal[g] = True
        # a goal that becomes satisfied later also counts (it must have been moved there)
    sat_later = {}
    for t in range(len(rows) - 1, -1, -1):
        g = rows[t]["active_goal"]
        if sat_later.get(g):
            rows[t]["aux_held_later"] = True
        gs = marks[t]["goal_states"]
        for gg in gids:
            if (gs.get(gg) or {}).get("satisfied"):
                sat_later[gg] = True
    meta = {"n_marks": len(marks), "n_calls": len(calls), "is_joint": is_joint, "goals": len(gids)}
    return rows, meta


FEATURES = None  # filled from the first row by build_dataset (stored in feature_spec.json)
