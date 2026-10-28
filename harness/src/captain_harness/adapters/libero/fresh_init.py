"""Fresh (non-official) LIBERO initial states for training.

Seeds below ``FRESH_SEED_BASE`` keep the official mapping (``init_states[seed % len(init_states)]`` of the task's
``*.pruned_init``). Seeds ``FRESH_SEED_BASE + k`` select fresh state ``k`` of the task's ``<task>.fresh_init`` file,
which lives outside the official init directory (default ``<harness root>/data/libero_fresh_init/<problem_folder>/``).
A fresh seed never falls back to an official state: a missing file or an index past its end is an error.

File format: ``<task>.fresh_init`` is a ``torch.save``d float64 ``numpy`` array (N, 1 + nq + nv), the same flattened
MuJoCo state layout as the official ``*.pruned_init`` files. The sidecar ``<task>.fresh_init.json`` records, per
state, the reset seed that produced it: fixture poses are model parameters that are not part of that vector, so the
harness resets with the recorded seed before loading the state (``env.seed(s); env.reset(); set_init_state(x)``),
which reproduces the generated scene exactly.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

FRESH_SEED_BASE = 1000
FRESH_SUFFIX = ".fresh_init"
FRESH_SCHEMA = "libero-fresh-init/v1"
DEFAULT_FRESH_DIR = Path("data") / "libero_fresh_init"


@dataclass(frozen=True)
class FreshInitStates:
    states: np.ndarray
    reset_seeds: tuple[int, ...]
    path: str
    sha256: str
    visual: tuple[dict[str, Any], ...] | None = None  # optional per-state scene variation (sidecar "visual")

    def __len__(self) -> int:
        return int(self.states.shape[0])


def is_fresh_seed(seed: int) -> bool:
    return int(seed) >= FRESH_SEED_BASE


def fresh_index(seed: int) -> int:
    if not is_fresh_seed(seed):
        raise ValueError(f"seed {seed} is an official-state seed (< {FRESH_SEED_BASE})")
    return int(seed) - FRESH_SEED_BASE


def fresh_init_path(fresh_dir: Path | str, problem_folder: str, task_name: str) -> Path:
    return Path(fresh_dir) / problem_folder / f"{task_name}{FRESH_SUFFIX}"


def load_fresh_init_states(path: Path | str, *, expected_dim: int | None = None) -> FreshInitStates:
    """Load and check a fresh init file and its sidecar (shape, dimension, seed list, file hash)."""
    import torch

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"fresh init file not found: {path}")
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    try:
        states = torch.load(path, weights_only=False)
    except TypeError:  # older torch
        states = torch.load(path)
    states = np.asarray(states, dtype=np.float64)
    meta_path = path.with_name(path.name + ".json")
    meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("schema") != FRESH_SCHEMA:
        raise ValueError(f"{meta_path}: not a {FRESH_SCHEMA} sidecar")
    if meta.get("sha256") != sha:
        raise ValueError(f"{path}: content does not match the sidecar's sha256")
    seeds = tuple(int(s) for s in meta["reset_seeds"])
    if states.ndim != 2 or len(states) == 0 or len(states) != len(seeds):
        raise ValueError(f"{path}: expected (N, D) states with N = len(reset_seeds) > 0, got {states.shape} / {len(seeds)}")
    if expected_dim is not None and states.shape[1] != int(expected_dim):
        raise ValueError(f"{path}: state dimension {states.shape[1]} != official {expected_dim}")
    if any(s < FRESH_SEED_BASE for s in seeds):
        raise ValueError(f"{path}: reset seeds must be >= {FRESH_SEED_BASE}")
    visual = meta.get("visual")
    if visual is not None and len(visual) != len(seeds):
        raise ValueError(f"{meta_path}: 'visual' must have one entry per state")
    return FreshInitStates(states=states, reset_seeds=seeds, path=str(path), sha256=sha,
                           visual=tuple(visual) if visual is not None else None)


def apply_visual_variation(env: Any, v: dict[str, Any]) -> None:
    """Scene variation for one fresh state (data collection only; official seeds never get one): light intensity /
    position, headlight, small agentview camera pose jitter, material tints. Applied to the MuJoCo model after the
    seeded reset; renders and the published camera extrinsics (read live from sim.data) both follow it."""
    import mujoco

    m = env.sim.model._model
    for name, (scale, dx, dy) in (v.get("lights") or {}).items():
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_LIGHT, name)
        if i < 0:
            continue
        m.light_diffuse[i] = np.clip(m.light_diffuse[i] * scale, 0.0, 1.0)
        m.light_specular[i] = np.clip(m.light_specular[i] * scale, 0.0, 1.0)
        m.light_pos[i][:2] += (dx, dy)
    if "headlight_diffuse" in v:
        m.vis.headlight.diffuse[:] = v["headlight_diffuse"]
        m.vis.headlight.ambient[:] = v["headlight_ambient"]
    cam = v.get("camera")
    if cam:
        c = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam["name"])
        if c >= 0:
            m.cam_pos[c] += np.asarray(cam["dpos"], dtype=np.float64)
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, np.asarray(m.cam_quat[c], dtype=np.float64), np.asarray(cam["dquat"], dtype=np.float64))
            m.cam_quat[c] = q / np.linalg.norm(q)
    for name, rgb in (v.get("tints") or {}).items():
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, name)
        if i >= 0:
            m.mat_rgba[i][:3] = np.clip(m.mat_rgba[i][:3] * np.asarray(rgb), 0.0, 1.0)
    env.sim.forward()


def select_fresh_state(fresh: FreshInitStates | None, seed: int) -> tuple[int, int, np.ndarray]:
    """(fresh index k, reset seed, state) for a fresh seed; never an official state."""
    k = fresh_index(seed)
    if fresh is None:
        raise ValueError(f"seed {seed} selects a fresh initial state (>= {FRESH_SEED_BASE}) but no fresh init file is loaded")
    if k >= len(fresh):
        raise IndexError(f"seed {seed} selects fresh state {k}, but {fresh.path} has only {len(fresh)}")
    return k, fresh.reset_seeds[k], fresh.states[k]


def official_dimension(init_states: Sequence[Any]) -> int | None:
    try:
        return int(np.asarray(init_states[0]).reshape(-1).shape[0])
    except Exception:
        return None


__all__ = [
    "DEFAULT_FRESH_DIR", "FRESH_SCHEMA", "apply_visual_variation", "FRESH_SEED_BASE", "FRESH_SUFFIX", "FreshInitStates", "fresh_index",
    "fresh_init_path", "is_fresh_seed", "load_fresh_init_states", "official_dimension", "select_fresh_state",
]
