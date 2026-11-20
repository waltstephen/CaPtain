"""Eval-only timing probe (soarm-speed hop-by-hop; not part of the harness): with SOARM_TIMING_LOG=<jsonl> and this
directory on PYTHONPATH, the env server appends {"fn", "s", "t", "pid"} for every call of the wrapped functions
(perception service calls, mono depth, observation fetch, measure_object_geometry stages and their helpers).
Summarise with tools/timing_probe/summarize.py. Wrapping costs one JSON line per call.
The harness starts the env server with PYTHONPATH=<harness>/src only (episode_launch.server_env), so this file must be
importable from there: symlink it as <harness>/src/sitecustomize.py (inactive unless SOARM_TIMING_LOG is set). Python
imports only the first sitecustomize on sys.path, and env.sh puts <harness>/src before e.g. the vLLM patch directory
(tools/vllm_patches, patch_latest): this file therefore always runs the next sitecustomize.py on sys.path too."""
import functools
import json
import os
import threading
import time

_PATH = os.environ.get("SOARM_TIMING_LOG")
_LOCK = threading.Lock()

# module -> [(class name or None, [attribute, ...])]; module-level names imported from elsewhere are wrapped where
# they are looked up (e.g. object_geometry's own _depth_to_base_points binding)
TARGETS = {
    "captain_harness.adapters.capx.services": [("CapXHttpToolServices", ["segment_text", "segment_point", "point"])],
    "captain_harness.adapters.hardware.mono_depth": [("MonoDepthSource", ["__call__", "_request"])],
    "captain_harness.adapters.capx.object_geometry": [(None, [
        "cloud", "fuse_views", "strip_robot_points", "scene_support_z", "measure_cloud", "footprint_fit", "_one",
        "measure", "visible_seed", "overlap_receipt", "support_plane", "_depth_to_base_points",
        "_filter_point_cloud_noise", "_fuse_scored_point_clouds", "fit_circle_robust", "min_area_rectangle"])],
    "captain_harness.adapters.capx.backend": [("ToolCapabilityBackend", ["get_observation", "observe", "mark_point"])],
    "captain_harness.adapters.hardware.facade": [("XR1FacadeEnv", ["_fetch_frames"])],
}


def _log(name, dt):
    with _LOCK, open(_PATH, "a") as fh:
        fh.write(json.dumps({"fn": name, "s": round(dt, 5), "t": time.time(), "pid": os.getpid()}) + "\n")


def _wrap(owner, attr, name):
    fn = getattr(owner, attr, None)
    if fn is None or getattr(fn, "_soarm_timed", False):
        return

    @functools.wraps(fn)
    def timed(*a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            _log(name, time.perf_counter() - t0)

    timed._soarm_timed = True
    setattr(owner, attr, timed)


def _install():
    import importlib.util
    import sys

    class Finder:
        def find_spec(self, name, path, target=None):
            if name not in TARGETS:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0, self)
            inner = spec.loader

            class Loader:
                def create_module(self, s):
                    return inner.create_module(s)

                def exec_module(self, module):
                    inner.exec_module(module)
                    for cls_name, attrs in TARGETS[name]:
                        owner = module if cls_name is None else getattr(module, cls_name, None)
                        if owner is None:
                            continue
                        short = name.rsplit(".", 1)[-1]
                        for attr in attrs:
                            _wrap(owner, attr, f"{short}.{cls_name + '.' if cls_name else ''}{attr}")
            spec.loader = Loader()
            return spec

    sys.meta_path.insert(0, Finder())


def _chain():
    """Run the next sitecustomize.py on sys.path (the one this file shadows)."""
    import importlib.util
    import sys
    here = os.path.realpath(__file__)
    for entry in sys.path:
        candidate = os.path.join(entry or ".", "sitecustomize.py")
        if os.path.isfile(candidate) and os.path.realpath(candidate) != here:
            spec = importlib.util.spec_from_file_location("_soarm_chained_sitecustomize", candidate)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return


if _PATH:
    _install()
_chain()
