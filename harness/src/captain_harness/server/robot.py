"""robot.py - thin client for the robot capability server (stdlib only, Python >= 3.8).

    from robot import observe, segment, select_detection, point, move_to, hold, finish
    obs = observe()                       # -> dict; obs["value"]["images"] lists PNG paths
    det = segment("the red bowl", obs["value"]["observation_handle"])

Every capability returns a plain dict:
    {"success": bool, "executed": bool, "failure_category": str, "message": str,
     "value": <capability-specific JSON or None>, "diagnostics": {...} or None,
     "budget": {"physical_actions_remaining": int, "tool_calls_remaining": int,
                "wall_time_remaining_s": float, ...}}
Always check result["success"] before using result["value"].

Positional arguments follow the order shown in TASK.md / help(<capability>); keyword
arguments are recommended. Call finish() when you are done: it ends the episode (the
outcome is scored by the environment and is NOT reported back to you). finish() may come
back with {"finished": false, "rejected": true, "message": ...}: the episode then continues
and you should address the message before calling finish() again.

The server URL is read from the ROBOT_SERVER_URL environment variable or from the
`.robot_server` file next to this module.
"""
import json
import os
import sys
import urllib.error
import urllib.request

__all__ = ["call", "finish", "status", "history", "manifest", "capabilities"]

_HERE = os.path.dirname(os.path.abspath(__file__))


def _server_url():
    url = os.environ.get("ROBOT_SERVER_URL")
    if not url:
        path = os.path.join(_HERE, ".robot_server")
        with open(path, "r", encoding="utf-8") as stream:
            url = stream.read().strip()
    return url.rstrip("/")


def _plain(value):
    """Array-likes (NumPy arrays and scalars, anything with tolist()/item()) are sent as plain JSON lists and numbers."""
    for method in ("tolist", "item"):
        if callable(getattr(value, method, None)):
            return getattr(value, method)()
    raise TypeError("Object of type %s is not JSON serializable" % type(value).__name__)


def _post(route, payload, timeout=1200.0):
    data = json.dumps(payload, default=_plain).encode("utf-8")
    request = urllib.request.Request(_server_url() + route, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise RuntimeError("robot server HTTP %s: %s" % (exc.code, body[:500]))


def _get(route, timeout=60.0):
    with urllib.request.urlopen(_server_url() + route, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


_CELL = []
_CELL_LIMIT = 20


def _note(line):
    if len(_CELL) < 200:
        _CELL.append(str(line)[:300])


def _cell_receipts():
    """One line per robot call since the previous call of this function (the python tool prints it after each cell)."""
    if not _CELL:
        return None
    lines = _CELL[:_CELL_LIMIT]
    more = len(_CELL) - len(lines)
    del _CELL[:]
    return "Robot calls in this cell:\n" + "\n".join("- " + line for line in lines) + ("\n- ... %d more" % more if more > 0 else "")


def call(name, **arguments):
    """Invoke any capability by name with keyword arguments; returns the result dict."""
    result = _post("/call", {"name": name, "arguments": arguments})
    if isinstance(result, dict):
        _note(result.get("receipt") or "%s: %s" % (name, "ok" if result.get("success") else "FAILED %s" % result.get("failure_category")))
    return result


def status():
    """Remaining budgets (physical actions, tool calls, wall time) and whether the episode is finished."""
    return _get("/health")


def history(start=None, end=None, limit=64):
    """Objective records of your past calls (name, arguments, success, message, image paths), by call index.

    Use it to recover what you already did when your conversation was compacted. Does not change the scene."""
    query = "&".join("%s=%s" % (k, v) for k, v in (("start", start), ("end", end), ("limit", limit)) if v is not None)
    return _get("/history" + ("?" + query if query else ""))


def finish(note=""):
    """End the episode. The environment scores the final state once; you receive no verdict.

    May return {"finished": false, "rejected": true, "message": ...} when your own recorded
    measurements do not support completion; then fix the scene or the measurement and call again."""
    result = _post("/finish", {"note": str(note), "source": "agent"})
    if isinstance(result, dict):
        _note("finish: finished" if result.get("finished") else "finish: rejected: %s" % str(result.get("message") or "")[:200])
    return result


_MANIFEST = None


def manifest():
    """The full public capability manifest (schemas, units, frames, examples)."""
    global _MANIFEST
    if _MANIFEST is None:
        _MANIFEST = _get("/manifest")
    return _MANIFEST


def capabilities():
    """Names of all capabilities."""
    return [spec["name"] for spec in manifest()["manifest"]["capabilities"]]


def _make(spec):
    name = spec["name"]
    order = list(spec.get("argument_order") or [])
    required = set(spec.get("arguments_schema", {}).get("required", []))

    def function(*args, **kwargs):
        if len(args) > len(order):
            raise TypeError("%s() takes at most %d positional arguments (%s)" % (name, len(order), ", ".join(order)))
        arguments = dict(zip(order, args))
        duplicate = set(arguments).intersection(kwargs)
        if duplicate:
            raise TypeError("%s() got multiple values for %s" % (name, sorted(duplicate)))
        arguments.update(kwargs)
        missing = [key for key in order if key in required and key not in arguments]
        if missing:
            raise TypeError("%s() missing required arguments: %s" % (name, ", ".join(missing)))
        return call(name, **arguments)

    function.__name__ = name
    doc_lines = [spec.get("description", "")]
    if order:
        doc_lines.append("Arguments (in order): " + ", ".join(
            ("%s" if key in required else "[%s]") % key for key in order))
    if spec.get("units"):
        doc_lines.append("Units: %s" % json.dumps(spec["units"]))
    if spec.get("coordinate_frames"):
        doc_lines.append("Frames: %s" % json.dumps(spec["coordinate_frames"]))
    if spec.get("examples"):
        doc_lines.append("Examples: " + " | ".join(spec["examples"]))
    function.__doc__ = "\n".join(doc_lines)
    return function


def _install():
    module = sys.modules[__name__]
    for spec in manifest()["manifest"]["capabilities"]:
        setattr(module, spec["name"], _make(spec))
        __all__.append(spec["name"])


try:
    _install()
except Exception as exc:  # server not reachable at import time
    sys.stderr.write("robot.py: could not load the capability manifest from the server (%s)\n" % exc)
