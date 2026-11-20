"""Eval-only disturbance: teleport the target object during the FIRST grasp.

Enabled by SOARM_DISTURB=<seed> (unset = off; never used for training or teacher data). State-triggered, not timed:
fires once, on the first control tick where the grip point is within 4 cm (xy) of the target object, descending
(z dropped > 1 mm since the previous tick) and below the object's top + 3.5 cm (final grasp descent), with the object not held. The object is
moved, keeping its orientation, to a pose drawn from rng(seed) relative to its INITIAL layout position (6-10 cm away,
0.13-0.28 m from the base, |y| <= 0.22 m, >= 3 cm clearance from every other object), so every system gets the same
new pose on a seed. After the move the hook records the first tick at which the gripper holds the object again.
Log: <record dir>/disturb.json (trigger / grasp frame index, sim time, wall time, old and new position).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import mujoco
import numpy as np

TRIGGER_XY_M = 0.04
TRIGGER_ABOVE_TOP_M = 0.035


def enabled() -> int | None:
    v = os.environ.get("SOARM_DISTURB")
    return int(v) if v not in (None, "") else None


class Disturbance:
    def __init__(self, sim, seed: int, log_dir: Path) -> None:
        self.sim, self.seed, self.log_path = sim, seed, Path(log_dir) / "disturb.json"
        self.target = sim.task.roles["target"]
        self.fired = False
        self.regrasp_done = False
        self.prev_z: float | None = None
        self.info: dict = {"seed": seed, "target": self.target}
        self.new_xy = self._new_xy()
        self.info["new_xy"] = [round(float(v), 4) for v in self.new_xy]

    def _radius(self, name: str) -> float:
        s = self.sim.specs[name].size
        return 0.5 * float(np.hypot(s[0], s[1]))

    def _new_xy(self) -> np.ndarray:
        sim = self.sim
        start = sim.initial[self.target].pos[:2]
        rng = np.random.default_rng([self.seed, 4242])
        others = {n: p.pos[:2] for n, p in sim.initial.items() if n != self.target}
        r_t = self._radius(self.target)
        for _ in range(2000):
            ang = rng.uniform(-np.pi, np.pi)
            d = rng.uniform(0.06, 0.10)
            xy = start + d * np.array([np.cos(ang), np.sin(ang)])
            if not (0.13 <= np.hypot(*xy) <= 0.28 and abs(xy[1]) <= 0.22):
                continue
            if all(np.linalg.norm(xy - o) >= r_t + self._radius(n) + 0.03 for n, o in others.items()):
                return xy
        raise RuntimeError("disturbance: no free reachable pose")

    def after_tick(self, frame_index: int | None) -> None:
        sim = self.sim
        grip = sim.grip_pose()[:3, 3]
        if not self.fired:
            obj = sim.poses()[self.target].pos
            top = sim.world().top(self.target)
            descending = self.prev_z is not None and grip[2] < self.prev_z - 0.001
            self.prev_z = float(grip[2])
            if (descending and np.linalg.norm(grip[:2] - obj[:2]) < TRIGGER_XY_M and grip[2] < top + TRIGGER_ABOVE_TOP_M
                    and self.target not in sim.held()):
                self._teleport()
                self.fired = True
                self.info.update(trigger_frame=frame_index, trigger_sim_s=round(sim.time_s, 3), trigger_wall=time.time(),
                                 old_pos=[round(float(v), 4) for v in obj], grip_at_trigger=[round(float(v), 4) for v in grip])
                self._write()
        elif not self.regrasp_done and self.target in sim.held():
            self.regrasp_done = True
            self.info.update(regrasp_frame=frame_index, regrasp_sim_s=round(sim.time_s, 3), regrasp_wall=time.time())
            self._write()

    def _teleport(self) -> None:
        sim = self.sim
        m, d = sim.model, sim.data
        body = sim.body_of[self.target]
        jnt = int(m.body_jntadr[body])
        adr, dof = int(m.jnt_qposadr[jnt]), int(m.jnt_dofadr[jnt])
        d.qpos[adr:adr + 2] = self.new_xy
        d.qpos[adr + 2] += 0.002                     # 2 mm above its support, settles at once
        d.qvel[dof:dof + 6] = 0.0
        mujoco.mj_forward(m, d)

    def _write(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text(json.dumps(self.info, indent=1))
