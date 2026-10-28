"""Opt-in host-side speedups of the LIBERO control loop that leave every simulator / observation value bit-identical.

Gates (environment variables, read when a robot backend is created; default off = the original code path):
  CAPTAIN_FAST_LOOP=1    - read the wrist force sensor through cached indices instead of per-substep name lookups, and
                         cache jsonschema checks of constant capability / motion-argument schemas (ports/capability.py,
                         adapters/capx/direct_programming.py).
  CAPTAIN_LAZY_RENDER=1  - camera observables keep robosuite's exact sampling schedule, but instead of rendering at the sample
                         instant they snapshot the full mjData (copy.copy = mj_copyData); the image/depth are rendered from that snapshot
                         only when the harness reads them (observe / pixel_to_world). Unread samples are never rendered.
Identity was checked offline against the plain loop on recorded traces (physics fields, depth, responses bit-identical;
RGB within the renderer's own run-to-run noise).
"""
from __future__ import annotations

import os

import numpy as np


def fast_loop_enabled() -> bool:
    return os.environ.get("CAPTAIN_FAST_LOOP", "") == "1"


def lazy_render_enabled() -> bool:
    return os.environ.get("CAPTAIN_LAZY_RENDER", "") == "1"


def _inner_env(env):
    """The robosuite MujocoEnv (owner of sim, robots and the native _pre_action) under LIBERO's wrappers."""
    seen = set()
    e = env
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if hasattr(e, "sim") and hasattr(e, "robots") and callable(getattr(e, "_pre_action", None)):
            return e
        e = getattr(e, "env", None)
    return None


# NOTE: skipping robosuite's second mj_forward per substep was tried and rejected: it is NOT bit-identical (the constraint
# solver warm-starts from qacc_warmstart, which the first forward of the substep updates).


class ForceSensorCache:
    """Cached sensordata slice / site id of the wrist force sensor (same values as robot.ee_force, minus name lookups)."""

    def __init__(self, env):
        inner = _inner_env(env)
        robot = inner.robots[0]
        name = robot.gripper.important_sensors["force_ee"]
        model = inner.sim.model
        sid = model.sensor_name2id(name)
        self.idx = np.sum(model.sensor_dim[:sid])
        self.dim = model.sensor_dim[sid]
        model_raw = getattr(model, "_model", model)
        self.site = int(np.asarray(model_raw.sensor_objid)[sid])
        self.sim = inner.sim

    def ee_force(self):
        return np.array(self.sim.data.sensordata[self.idx: self.idx + self.dim])

    def site_xmat(self):
        data = getattr(self.sim.data, "_data", self.sim.data)
        return data.site_xmat[self.site]


# --------------------------------------------------------------------------------------------------------- lazy render
class LazyCameras:
    """Defers camera rendering from robosuite's per-control-step sample instant to the harness's read, rendering from an
    exact mjData copy of the sample instant (the live mjData is never written: the renderer reads the copy through a
    temporary swap of the MjData wrapper's handle)."""

    def __init__(self, env):
        import copy
        self._copy = copy.copy
        inner = _inner_env(env)
        self.inner = inner
        self.sim = inner.sim
        self.snap = None
        self.cams = {}
        for name, obs in inner._observables.items():
            if obs.modality == "image" and (name.endswith("_image") or name.endswith("_depth")):
                self.cams[name] = obs
        if not self.cams:
            raise RuntimeError("no camera observables")
        for obs in self.cams.values():
            self._wrap(obs)

    def _wrap(self, obs):
        lazy = self

        def update(timestep, obs_cache, force=False, _o=obs):
            # robosuite Observable.update with the sensor call replaced by a snapshot (identical timer / flag logic)
            if _o._enabled:
                _o._time_since_last_sample += timestep
                if (not _o._sampled and _o._sampling_timestep - _o._current_delay >= _o._time_since_last_sample) or force:
                    if force:
                        lazy._render_one(_o, obs_cache)     # forced update: render now from the live state, as before
                    else:
                        lazy._snapshot()
                    _o._sampled = True
                    _o._current_delay = _o._delayer()
                if _o._time_since_last_sample >= _o._sampling_timestep:
                    if not _o._sampled:
                        print(f"Warning: sampling rate for observable {_o.name} is either too low or delay is too high. "
                              f"Please adjust one (or both)")
                        lazy._snapshot()
                        _o._current_delay = _o._delayer()
                    _o._time_since_last_sample %= _o._sampling_timestep
                    _o._sampled = False

        obs.update = update

    def _snapshot(self):
        # every camera observable samples at the same substep; copy the state once per substep
        data = self.sim.data._data
        if self.snap is None or self._snap_src_time != data.time or self._snap_src_id != id(data):
            self.snap = self._copy(data)
            self._snap_src_time = data.time
            self._snap_src_id = id(data)

    @staticmethod
    def _render_one(o, obs_cache):
        v = np.array(o._filter(o._corrupter(o._sensor(obs_cache))))
        o._current_observed_value = v[0] if len(v.shape) == 1 and v.shape[0] == 1 else v
        obs_cache[o.name] = np.array(o._current_observed_value)

    def materialize(self, observation):
        """Render the pending snapshot (if any) and put the camera arrays into @observation (the latest step's dict)."""
        if self.snap is None:
            return observation
        wrapper = self.sim.data
        live = wrapper._data
        wrapper._data = self.snap
        try:
            cache = self.inner._obs_cache
            for o in self.cams.values():        # same order as robosuite's update loop (rgb renders, depth reads its cache)
                self._render_one(o, cache)
        finally:
            wrapper._data = live
        self.snap = None
        if isinstance(observation, dict):
            for name, o in self.cams.items():
                if name in observation:
                    observation[name] = o.obs
        return observation
