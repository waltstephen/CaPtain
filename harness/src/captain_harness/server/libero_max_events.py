"""LIBERO-MAX (arXiv 2609.36518) mid-task events for the env server, eval only.

One frozen LIBERO-MAX case (``--max-case case.json``) is attached to a LIBERO-plus episode. The benchmark's own code does the work:
``libero_max.runtime.InterventionRuntime`` (exactly-once scheduling) and ``libero_max.libero_backend.LiberoMujocoBackend`` (the
physical/visual change and the post-event observation transform), imported from the LIBERO-MAX checkout (``~/benchmarks/LIBERO-MAX``
or ``$LIBERO_MAX_ROOT``). Nothing is resampled: the stored change payload and trigger are applied as released.

Mapping of the reference runner (``scripts/run_xvla_persistent_shard.py`` + ``CosmosInterventionEnv``) onto the harness:
- scenario ``setup`` changes (e.g. distractors / obstacle moved off-world) are applied in BOTH arms right after ``set_init_state``,
  followed by a refreshed observation;
- trigger ``on_proximity``: after every env step past the harness's reset settling steps, the distance from ``robot0_eef_pos`` to the
  trigger entity is checked (backend.distance_to_entity); the first time it is <= ``distance_m`` the trigger is reached. In the
  ``intervention`` arm the change is applied at that step (mid ``move_to`` if that is where it happens), the observation is refreshed,
  and every later observation passes through ``backend.transform_observation`` (sensor-noise / illumination exposure). The
  ``control`` arm only records when the trigger would have fired;
- success stays the original LIBERO goal predicate (MAX ``physical_completion`` track).
Deviation (documented in the eval report): the reference protocol replays the Base arm's exact action prefix in the Dynamic arm; an
LLM agent sampled at T=1 cannot be replayed this way, so the two arms are independent rollouts from the same initial state / env seed.

The patch wraps the env instance's own ``reset`` / ``set_init_state`` / ``step`` (like libero_variants' plus patches) and must be
installed BEFORE those, so plus robot-init / sensor-noise wrap around it. A JSON trace is written to ``<output_dir>/libero_max.json``.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

SETTLING_STEPS = 5  # reset_to_seeded_init_state's settling steps; the trigger is only armed after them


def max_root() -> Path:
    return Path(os.environ.get("LIBERO_MAX_ROOT", str(Path.home() / "benchmarks" / "LIBERO-MAX")))


def _import_max():
    src = str(max_root() / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from libero_max.libero_backend import LiberoMujocoBackend  # noqa: PLC0415
    from libero_max.runtime import InterventionRuntime, TriggerContext  # noqa: PLC0415
    return LiberoMujocoBackend, InterventionRuntime, TriggerContext


def _jsonable(value: Any) -> Any:
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def install_max_event(env: Any, case: Mapping[str, Any], arm: str, instruction: str, output_dir: Path,
                      log: Callable[[str], None] | None = None) -> dict:
    if arm not in ("control", "intervention"):
        raise ValueError("--max-arm must be control or intervention")
    scenario = case["scenario"]
    trigger = scenario["trigger"]
    if trigger.get("type") != "on_proximity":
        raise ValueError(f"unsupported LIBERO-MAX trigger {trigger.get('type')!r} (only on_proximity is used by the release)")
    LiberoMujocoBackend, InterventionRuntime, TriggerContext = _import_max()
    backend = LiberoMujocoBackend(env)
    runtime = InterventionRuntime(dict(scenario), backend)
    trace_path = Path(output_dir) / "libero_max.json"
    state = {"case_id": case["case_id"], "arm": arm, "change_type": scenario["change_type"], "trigger": trigger,
             "change": scenario["change"], "setup": scenario.get("setup", []), "env_steps": 0, "setup_trace": [],
             "trigger_reached": False, "trigger_step": None, "trigger_distance_m": None, "event": None,
             "event_wall_time": None, "min_distance_m": None, "goal_done_after_event": False}

    def write():
        trace_path.write_text(json.dumps(_jsonable(state), indent=1, sort_keys=True), encoding="utf-8")

    inner_reset, inner_set, inner_step = env.reset, env.set_init_state, env.step

    def reset(*args, **kwargs):
        result = inner_reset(*args, **kwargs)
        backend.reset_episode_state()
        runtime.reset(instruction)
        state.update(env_steps=0, setup_trace=[], trigger_reached=False, trigger_step=None, trigger_distance_m=None,
                     event=None, event_wall_time=None, min_distance_m=None, goal_done_after_event=False)
        return result

    def set_init_state(init_state):
        observation = inner_set(init_state)
        if scenario.get("setup") and not runtime.setup_trace:
            state["setup_trace"] = runtime.apply_setup()
            observation = backend.refresh_observation()
            if log is not None:
                log(f"libero-max setup applied ({len(state['setup_trace'])} changes)")
        write()
        return observation

    def step(action):
        observation, step_value, done, info = inner_step(action)
        if runtime.applied:
            observation = backend.transform_observation(observation)
            if done:
                state["goal_done_after_event"] = True
        state["env_steps"] += 1
        if state["env_steps"] > SETTLING_STEPS and not state["trigger_reached"] and isinstance(observation, dict):
            distance = backend.distance_to_entity(observation, trigger["value"])
            if state["min_distance_m"] is None or distance < state["min_distance_m"]:
                state["min_distance_m"] = distance
            if distance <= float(trigger["distance_m"]):
                state.update(trigger_reached=True, trigger_step=state["env_steps"], trigger_distance_m=distance)
                if arm == "intervention" and not done:
                    event = runtime.maybe_apply(TriggerContext(step=state["env_steps"], max_steps=10 ** 9,
                                                               events=frozenset({f"proximity:{trigger['value']}"})))
                    if event is not None:
                        observation = backend.refresh_observation()
                        state["event"] = event
                        state["event_wall_time"] = time.time()
                        info = dict(info or {})
                        info["libero_max_event"] = event
                        if log is not None:
                            log(f"libero-max event {scenario['change_type']} applied at env step {state['env_steps']} "
                                f"(distance {distance:.3f} m to {trigger['value']})")
                write()
        return observation, step_value, done, info

    env.reset, env.set_init_state, env.step = reset, set_init_state, step
    if log is not None:
        log(f"libero-max case {case['case_id']} arm={arm} change={scenario['change_type']} trigger={trigger}")
    write()
    return {"case_id": case["case_id"], "arm": arm, "change_type": scenario["change_type"]}


__all__ = ["install_max_event", "max_root", "SETTLING_STEPS"]
