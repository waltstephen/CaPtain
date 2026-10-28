"""Shared helpers for generating fresh LIBERO initial states (outside the official init files).

A LIBERO initial state is the flattened MuJoCo state ``[time, qpos, qvel]`` that ``env.reset()`` leaves after the BDDL
placement samplers ran (the official ``*.init`` / ``*.pruned_init`` arrays are exactly this, with time = 0).
Fixture poses (cabinets, stoves, ...) are model parameters (``body_pos``) and therefore *not* in that vector: they come
from the reset that precedes ``set_init_state``. A fresh state is therefore stored together with the reset seed that
produced it, and the harness resets with that seed before loading the state (fixtures then match exactly).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

COMMUNITY_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SETTLE_STEPS = 5  # what the harness does after set_init_state (backend.reset_to_seeded_init_state)


def configure(root: Path, config_dir: Path) -> None:
    import sys

    src = Path(__file__).resolve().parents[2] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from captain_harness.server.libero_setup import configure_libero

    configure_libero(Path(root), config_dir)


def task_info(suite: str, task_id: int):
    from libero.libero import benchmark, get_libero_path

    task = benchmark.get_benchmark_dict()[suite]().get_task(task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    init_dir = Path(get_libero_path("init_states")) / task.problem_folder
    return task, bddl, init_dir


def make_env(bddl: Path, *, controller: str = "OSC_POSE"):
    """Same simulator/placement code as the harness env (OffScreenRenderEnv), without any renderer."""
    from libero.libero.envs.env_wrapper import ControlEnv

    return ControlEnv(bddl_file_name=str(bddl), controller=controller, has_renderer=False,
                      has_offscreen_renderer=False, use_camera_obs=False, camera_depths=False, control_freq=20,
                      horizon=45000)


def load_official(init_dir: Path, task_name: str) -> dict[str, np.ndarray]:
    import torch

    out = {}
    for suffix in ("pruned_init", "init"):
        path = init_dir / f"{task_name}.{suffix}"
        if path.exists():
            try:
                out[suffix] = np.asarray(torch.load(path, weights_only=False), dtype=np.float64)
            except TypeError:
                out[suffix] = np.asarray(torch.load(path), dtype=np.float64)
    return out


@dataclass(frozen=True)
class Layout:
    """qpos addresses of every non-robot joint: free joints = movable objects, the rest = articulations."""
    free_names: tuple[str, ...]
    free_adr: tuple[int, ...]
    art_names: tuple[str, ...]
    art_adr: tuple[int, ...]
    nq: int
    nv: int
    art_hinge: tuple[bool, ...] = ()


def layout(env) -> Layout:
    import mujoco

    m = env.sim.model._model
    free_n, free_a, art_n, art_a, art_h = [], [], [], [], []
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}"
        if name.startswith(("robot0_", "gripper0_", "mount0_")):
            continue
        adr = int(m.jnt_qposadr[j])
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            free_n.append(name); free_a.append(adr)
        elif m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            art_n.append(name); art_a.append(adr); art_h.append(bool(m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE))
    return Layout(tuple(free_n), tuple(free_a), tuple(art_n), tuple(art_a), int(m.nq), int(m.nv), tuple(art_h))


def poses(states: np.ndarray, lay: Layout) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(N, n_obj, 3) positions, (N, n_obj, 4) wxyz quaternions, (N, n_art) articulation values."""
    states = np.atleast_2d(states)
    q = states[:, 1:1 + lay.nq]
    pos = np.stack([q[:, a:a + 3] for a in lay.free_adr], axis=1) if lay.free_adr else np.zeros((len(q), 0, 3))
    quat = np.stack([q[:, a + 3:a + 7] for a in lay.free_adr], axis=1) if lay.free_adr else np.zeros((len(q), 0, 4))
    art = q[:, list(lay.art_adr)] if lay.art_adr else np.zeros((len(q), 0))
    return pos, quat, art


def yaw_of(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = (quat[..., i] for i in range(4))
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def rot_angle(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    """Geodesic angle between unit quaternions (broadcast over leading axes)."""
    na = qa / np.linalg.norm(qa, axis=-1, keepdims=True)
    nb = qb / np.linalg.norm(qb, axis=-1, keepdims=True)
    dot = np.abs(np.sum(na * nb, axis=-1)).clip(0, 1)
    return 2 * np.arccos(dot)


def pairwise(a_states: np.ndarray, b_states: np.ndarray, lay: Layout) -> dict[str, np.ndarray]:
    """Per pair (i in a, j in b): max over movable objects of xyz distance [m], of rotation angle [rad], and the max
    articulation difference of hinge joints [rad] ("art_hinge") and of slide joints [m] ("art_slide"); "art" = both."""
    pa, qa, aa = poses(a_states, lay)
    pb, qb, ab = poses(b_states, lay)
    dpos = np.linalg.norm(pa[:, None] - pb[None], axis=-1).max(-1) if pa.shape[1] else np.zeros((len(pa), len(pb)))
    drot = rot_angle(qa[:, None], qb[None]).max(-1) if qa.shape[1] else np.zeros((len(pa), len(pb)))
    diff = np.abs(aa[:, None] - ab[None])
    hinge = np.asarray(lay.art_hinge, dtype=bool) if lay.art_hinge else np.zeros(aa.shape[1], dtype=bool)
    zero = np.zeros((len(pa), len(pb)))
    dh = diff[..., hinge].max(-1) if hinge.any() else zero
    ds = diff[..., ~hinge].max(-1) if (~hinge).any() else zero
    dart = diff.max(-1) if aa.shape[1] else zero
    return {"pos": dpos, "rot": drot, "art": dart, "art_hinge": dh, "art_slide": ds}


def penetration(env) -> float:
    """Deepest contact penetration [m] (>0) among contacts that involve a non-robot movable object, at the current
    sim state (after forward)."""
    import mujoco

    m, d = env.sim.model._model, env.sim.data._data
    robot = ("robot0_", "gripper0_", "mount0_")
    worst = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        if -c.dist <= worst:
            continue
        b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[c.geom1])) or ""
        b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[c.geom2])) or ""
        if b1.startswith(robot) and b2.startswith(robot):
            continue
        worst = -float(c.dist)
    return worst


def hold_action(env, obs) -> np.ndarray:
    """The harness's settling action: zero arm motion, gripper command matching its current opening."""
    g = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64).reshape(-1)
    frac = float(np.clip(g[0] / 0.04, 0.0, 1.0))
    act = np.zeros(int(env.robots[0].action_dim), dtype=np.float64)
    act[-1] = 1.0 - 2.0 * frac
    return act


def evaluate_state(env, state: np.ndarray, lay: Layout, *, extra_steps: int = 50) -> dict[str, Any]:
    """Load a state exactly like the harness (set_init_state + SETTLE_STEPS hold steps), then keep holding for
    extra_steps; report penetration at load, success before/after, object motion while settling/holding, min z."""
    obs = env.set_init_state(state)
    pen = penetration(env)
    success_raw = bool(env.check_success())
    raw = env.get_sim_state().copy()
    act = hold_action(env, obs)
    for _ in range(SETTLE_STEPS):
        obs, _, _, _ = env.step(act)
    settled = env.get_sim_state().copy()
    success_settled = bool(env.check_success())
    success_any = success_settled
    for _ in range(extra_steps):
        obs, _, _, _ = env.step(act)
        success_any = success_any or bool(env.check_success())
    held = env.get_sim_state().copy()
    pen_held = penetration(env)
    p0, q0, a0 = poses(raw, lay)
    p1, q1, a1 = poses(settled, lay)
    p2, q2, a2 = poses(held, lay)
    nobj = p0.shape[1]
    def mx(x):
        return float(np.max(x)) if x.size else 0.0
    return {
        "penetration_m": pen,
        "penetration_held_m": pen_held,
        "success_raw": success_raw, "success_settled": success_settled, "success_any": success_any,
        "settle_move_m": mx(np.linalg.norm(p1 - p0, axis=-1)) if nobj else 0.0,
        "xy_move_m": mx(np.linalg.norm((p2 - p0)[..., :2], axis=-1)) if nobj else 0.0,
        "z_drop_m": mx(p0[..., 2] - p2[..., 2]) if nobj else 0.0,
        "hold_move_m": mx(np.linalg.norm(p2 - p1, axis=-1)) if nobj else 0.0,
        "hold_rot_rad": mx(rot_angle(q2, q1)) if nobj else 0.0,
        "total_rot_rad": mx(rot_angle(q2, q0)) if nobj else 0.0,
        "art_move": mx(np.abs(a2 - a0)) if a0.size else 0.0,
        "min_z_m": float(p2[..., 2].min()) if nobj else 0.0,
        "max_speed": float(np.abs(held[1 + lay.nq:]).max()),
    }


def env_seed_reset(env, seed: int) -> np.ndarray:
    env.seed(int(seed))
    env.reset()
    return env.get_sim_state().copy()


def setup_from_env() -> tuple[Path, Path]:
    root = Path(os.environ.get("FRESH_LIBERO_ROOT", str(Path(__file__).resolve().parents[2])))
    cfg = Path(os.environ.get("FRESH_LIBERO_CFG", f"/tmp/fresh-init-libcfg-{os.getpid()}"))
    configure(root, cfg)
    return root, cfg


__all__ = [n for n in dir() if not n.startswith("_")]
_ = math
