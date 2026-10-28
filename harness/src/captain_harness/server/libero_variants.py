"""LIBERO forks used as evaluation benchmarks: LIBERO-PRO and LIBERO-plus.

Both are drop-in replacements of the ``libero`` package (their own bddl/init files, benchmark registry and, for plus,
a modified ``ControlEnv``), so each runs in its own venv: ``<root>/.venv-libero-<variant>`` if present, else
``~/benchmarks/.venv-libero-<variant>``. The env server is the only process that imports ``libero``; the driver is
unchanged.

Differences from the original LIBERO path that matter for a fair run:
- instruction: the forks keep the original *file names* but change ``(:language ...)`` (PRO "task" perturbation,
  plus "language" perturbation) or append perturbation ids to the name (plus). ``task.language`` (derived from the
  file name) is therefore wrong for them; the instruction is read from the parsed BDDL (``env.language_instruction``),
  as CaP-X does (capx/integrations/libero ``_extract_language_from_bddl``).
- init states: plus maps a perturbed task name to the original init file inside ``Benchmark.get_task_init_states``,
  so init states are loaded through the suite object, not from ``task.init_states_file``.
- plus "Robot Initial States": the perturbation lives in the robot class's ``init_qpos`` (``MountedPanda<N>``), but
  ``set_init_state`` restores the full MuJoCo state saved with the default robot pose and silently undoes it. With
  ``robot_init_override`` the robot joints are put back to ``init_qpos`` after every ``set_init_state`` (objects keep
  the saved state). This is our reading of the benchmark; the upstream evaluation script is not published.
"""
from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, Callable, Mapping

VARIANTS = ("libero", "pro", "plus")
CHECKOUT_DIRS = {"pro": "LIBERO-PRO", "plus": "LIBERO-plus"}
DEFAULT_PANDA_INIT_QPOS = (0.0, -0.161037389, 0.0, -2.44459747, 0.0, 2.2267522, 0.785398163)


def benchmarks_home() -> Path:
    return Path(os.environ.get("CAPTAIN_BENCHMARKS_HOME", str(Path.home() / "benchmarks")))


def variant_checkout(variant: str, override: Path | None = None) -> Path:
    """Repository root of the fork (the directory that contains ``libero/libero``)."""
    if variant not in CHECKOUT_DIRS:
        raise ValueError(f"no checkout for LIBERO variant {variant!r}")
    root = Path(override) if override is not None else benchmarks_home() / CHECKOUT_DIRS[variant]
    if not (root / "libero" / "libero" / "bddl_files").is_dir():
        raise FileNotFoundError(f"{root} is not a {CHECKOUT_DIRS[variant]} checkout (libero/libero/bddl_files missing)")
    return root


def variant_python(root: Path, variant: str) -> Path:
    """Interpreter for the env server: the original ``.venv-libero`` or the fork's own venv."""
    if variant == "libero":
        return Path(root) / ".venv-libero" / "bin" / "python"
    for candidate in (Path(root) / f".venv-libero-{variant}", benchmarks_home() / f".venv-libero-{variant}"):
        if (candidate / "bin" / "python").exists():
            return candidate / "bin" / "python"
    raise FileNotFoundError(f"no venv for LIBERO variant {variant!r} (.venv-libero-{variant} in {root} or {benchmarks_home()})")


def variant_env(variant: str) -> dict[str, str]:
    """Extra process environment for the env server. LIBERO-plus imports ``wand`` (ImageMagick) for its sensor-noise
    perturbation; on hosts without a system ImageMagick the libraries are unpacked under ~/local/magick."""
    if variant != "plus" or os.environ.get("MAGICK_HOME"):
        return {}
    libdir = Path.home() / "local" / "magick" / "usr" / "lib" / "x86_64-linux-gnu"
    if not (libdir / "libMagickWand-6.Q16.so.6").exists():
        return {}
    modules = libdir / "ImageMagick-6.9.10" / "modules-Q16"
    ld = os.environ.get("LD_LIBRARY_PATH")
    return {
        "MAGICK_HOME": str(Path.home() / "local" / "magick" / "whome"),
        "LD_LIBRARY_PATH": str(libdir) + (":" + ld if ld else ""),
        "MAGICK_CODER_MODULE_PATH": str(modules / "coders"),
        "MAGICK_CODER_FILTER_PATH": str(modules / "filters"),
        "MAGICK_CONFIGURE_PATH": f"{Path.home() / 'local/magick/etc/ImageMagick-6'}:{libdir / 'ImageMagick-6.9.10/config-Q16'}",
    }


def load_init_states(suite: Any, task_id: int) -> Any:
    """``suite.get_task_init_states`` (fork-specific path mapping) with pickled init files allowed under torch>=2.6."""
    import torch

    original = torch.load
    torch.load = functools.partial(original, weights_only=False)
    try:
        return suite.get_task_init_states(task_id)
    finally:
        torch.load = original


def bddl_language(env: Any) -> str | None:
    for holder in (env, getattr(env, "env", None)):
        value = getattr(holder, "language_instruction", None)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    parsed = getattr(getattr(env, "env", env), "parsed_problem", None) or {}
    value = parsed.get("language_instruction") if isinstance(parsed, Mapping) else None
    if isinstance(value, (list, tuple)):
        value = " ".join(str(v) for v in value)
    return " ".join(str(value).split()) if value else None


def robot_init_qpos(env: Any) -> list[float] | None:
    """The robot class's ``init_qpos`` when it differs from the stock Panda pose (LIBERO-plus robot perturbation)."""
    import numpy as np

    robot = env.robots[0]
    qpos = np.asarray(getattr(robot.robot_model, "init_qpos", DEFAULT_PANDA_INIT_QPOS), dtype=np.float64).reshape(-1)
    if qpos.shape != (7,) or np.allclose(qpos, DEFAULT_PANDA_INIT_QPOS, atol=1e-6):
        return None
    return qpos.tolist()


def install_robot_init_override(env: Any, log: Callable[[str], None] | None = None) -> list[float] | None:
    """Wrap ``env.set_init_state`` so the robot joints go back to the perturbed ``init_qpos`` after the saved MuJoCo state
    is restored. Returns the qpos, or None when the robot is not perturbed (nothing installed)."""
    import numpy as np

    qpos = robot_init_qpos(env)
    if qpos is None:
        return None
    inner = env.set_init_state
    target = np.asarray(qpos, dtype=np.float64)

    def set_init_state(state):
        inner(state)
        sim = env.env.sim
        robot = env.robots[0]
        sim.data.qpos[robot._ref_joint_pos_indexes] = target
        sim.data.qvel[robot._ref_joint_vel_indexes] = 0.0
        sim.forward()  # the inner set_init_state already refreshed the task predicates; joints only change here
        env._post_process()
        env._update_observables(force=True)
        return env.env._get_observations()

    env.set_init_state = set_init_state
    if log is not None:
        log(f"robot init override installed: init_qpos={[round(v, 4) for v in qpos]}")
    return qpos


NOISE_RESOLUTION = 256  # LIBERO-plus corruptions are tuned for policy-resolution frames (fog needs <= 256 px)


class _LazyNoisyObservation(dict):
    """Observation dict whose ``agentview_image`` is corrupted on first access only (frames the agent never sees cost
    nothing; LIBERO-plus corrupts every simulator step, which at 512 px and glass blur is ~6 s per step)."""

    def __init__(self, base, corrupt):
        super().__init__(base)
        self._corrupt = corrupt
        self._corrupted = False

    def _ensure(self, key):
        if key == "agentview_image" and not self._corrupted and dict.__contains__(self, key):
            dict.__setitem__(self, key, self._corrupt(dict.__getitem__(self, key)))
            self._corrupted = True

    def __getitem__(self, key):
        self._ensure(key)
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        self._ensure(key)
        return dict.get(self, key, default)

    def items(self):
        self._ensure("agentview_image")
        return dict.items(self)

    def values(self):
        self._ensure("agentview_image")
        return dict.values(self)

    def copy(self):
        self._ensure("agentview_image")
        return dict(self)


def sensor_noise_corruption(level: int):
    """LIBERO-plus's own corruption for a noise level (1-10 motion blur, 11-20 gaussian blur, 21-30 zoom blur,
    31-40 fog, 41-50 glass blur), applied at NOISE_RESOLUTION and resized back to the camera resolution."""
    import numpy as np
    from PIL import Image
    from libero.libero.envs import env_wrapper as plus

    level = int(level)
    if not 1 <= level <= 50:
        raise ValueError(f"LIBERO-plus noise level {level} outside 1..50")
    family, severity = divmod(level - 1, 10)
    fn = (plus.motion_blur, plus.gaussian_blur, plus.zoom_blur, plus.fog, plus.glass_blur)[family]

    def corrupt(image):
        array = np.asarray(image)
        if array.dtype != np.uint8:
            array = (array * 255).astype(np.uint8)
        height, width = array.shape[:2]
        small = Image.fromarray(array).resize((NOISE_RESOLUTION, NOISE_RESOLUTION), Image.BICUBIC)
        out = np.clip(np.asarray(fn(small, severity=severity + 1), dtype=np.float64), 0, 255).astype(np.uint8)
        return np.asarray(Image.fromarray(out).resize((width, height), Image.BICUBIC))

    return corrupt


def install_lazy_sensor_noise(env, log: Callable[[str], None] | None = None) -> int | None:
    """Move LIBERO-plus's per-step agentview corruption to observation time. Returns the noise level or None."""
    level = int(getattr(env, "noise", 0) or 0)
    if level == 0:
        return None
    corrupt = sensor_noise_corruption(level)
    env.noise = 0  # ControlEnv.step/reset no longer corrupt every frame
    inner_step, inner_set = env.step, env.set_init_state

    def step(action):
        result = inner_step(action)
        if isinstance(result, tuple) and result and isinstance(result[0], dict):
            return (_LazyNoisyObservation(result[0], corrupt),) + tuple(result[1:])
        return result

    def set_init_state(state):
        obs = inner_set(state)
        return _LazyNoisyObservation(obs, corrupt) if isinstance(obs, dict) else obs

    env.step, env.set_init_state = step, set_init_state
    if log is not None:
        log(f"LIBERO-plus sensor noise level {level}: agentview corrupted at {NOISE_RESOLUTION}px on observation access")
    return level


__all__ = ["NOISE_RESOLUTION", "install_lazy_sensor_noise", "sensor_noise_corruption", "CHECKOUT_DIRS", "VARIANTS", "bddl_language", "benchmarks_home", "install_robot_init_override", "load_init_states",
           "robot_init_qpos", "variant_checkout", "variant_env", "variant_python"]
