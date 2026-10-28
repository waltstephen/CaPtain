"""Compare a failed Cartesian goal with a later receipt's public robot measurements."""
from collections.abc import Mapping
import math


def _vector(value, size):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError("missing or malformed public pose")
    if not all(type(x) in (int, float) and math.isfinite(x) for x in value):
        raise ValueError("non-finite public pose")
    return tuple(float(x) for x in value)


def _unit_quaternion(value):
    q = _vector(value, 4)
    norm = math.sqrt(sum(x * x for x in q))
    if norm < 1e-12:
        raise ValueError("zero public quaternion")
    return tuple(x / norm for x in q)


def cartesian_recovery_evidence(failed_arguments, result, *, argument_schema):
    """No simulator access: successful new motion must also satisfy the old measured endpoint."""
    if "position" not in failed_arguments or "quaternion" not in failed_arguments:
        return None
    evidence = {"scope": "execution_goal", "resolved": False, "status": "unknown",
        "basis": "public_robot_after_pose_against_failed_requested_endpoint",
        "original_goal_recovery": "not_assessed"}
    value = result.value if isinstance(result.value, Mapping) else {}
    if (not result.success or not result.executed or value.get("endpoint_reached") is not True
            or value.get("source") != "public_robot_measurements"):
        return {**evidence, "reason": "successful settled public endpoint required"}
    try:
        after = value["robot_after"]
        if not isinstance(after, Mapping):
            raise ValueError("missing public robot_after measurement")
        requested = _vector(failed_arguments["position"], 3)
        measured = _vector(after["end_effector_position_m"], 3)
        target_q = _unit_quaternion(failed_arguments["quaternion"])
        measured_q = _unit_quaternion(after["end_effector_quaternion_wxyz"])
        defaults = argument_schema["properties"]
        position_tolerance = failed_arguments.get(
            "position_tolerance_m", defaults["position_tolerance_m"]["default"])
        orientation_tolerance = failed_arguments.get(
            "orientation_tolerance_rad", defaults["orientation_tolerance_rad"]["default"])
        if not all(type(t) in (int, float) and math.isfinite(t) and t > 0
                   for t in (position_tolerance, orientation_tolerance)):
            raise ValueError("invalid original request tolerance")
        position_error = math.dist(requested, measured)
        dot = min(1.0, abs(sum(a * b for a, b in zip(target_q, measured_q))))
        orientation_error = 2 * math.acos(dot)
        evidence.update(
            original_position_error_m=position_error,
            original_orientation_error_rad=orientation_error,
            original_position_tolerance_m=position_tolerance,
            original_orientation_tolerance_rad=orientation_tolerance,
            resolved=position_error <= position_tolerance and orientation_error <= orientation_tolerance)
        evidence["status"] = "supported" if evidence["resolved"] else "contradicted"
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        evidence["reason"] = str(exc)
    return evidence
