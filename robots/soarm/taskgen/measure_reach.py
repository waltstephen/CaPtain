"""Measure the reach band the driver's own IK achieves for top-down grasps: for grip heights z and radii r, solve
SoarmDriver._ik (the per-tick IK of every move) from the home pose and a few seeds; reachable = position error < 4 mm and
approach axis within max_tilt (default 15) deg of vertical. Prints a REACH table for taskgen.planner.

    python -m taskgen.measure_reach [max_tilt_deg]
"""
from __future__ import annotations

from OpenGL import GL  # noqa: F401  (load OSMesa before mujoco)
import mujoco
import numpy as np

from soarm.driver import SoarmDriver
from soarm.sim import HOME_Q, SoarmSim
from soarm.twin import build_twin_xml

DOWN = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])


def main(max_tilt: float = 15.0) -> None:
    drv = SoarmDriver(SoarmSim("pick_lift", 0), build_twin_xml())
    m, d = drv.model, drv._data
    seeds = [HOME_Q, HOME_Q + np.array([0, 0.4, -0.4, 0.2, 0]), HOME_Q + np.array([0, -0.3, 0.5, -0.3, 0])]
    rows = []
    for z in np.arange(0.01, 0.141, 0.01):
        ok = []
        for r in np.arange(0.06, 0.361, 0.005):
            best = False
            for ang in (0.0, 0.9):
                g = np.eye(4)
                c, s = np.cos(ang), np.sin(ang)
                g[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ DOWN
                g[:3, 3] = [r * np.cos(ang), r * np.sin(ang), z]
                good = False
                for q0 in seeds:
                    q0 = q0.copy()
                    q0[0] = ang
                    q = drv._ik(g, q0, iterations=300)
                    d.qpos[drv._q] = np.r_[q, 0.0, 0.0]
                    mujoco.mj_kinematics(m, d)
                    perr = float(np.linalg.norm(d.site_xpos[drv._site] - g[:3, 3]))
                    approach = d.site_xmat[drv._site].reshape(3, 3)[:, 2]
                    tilt = float(np.degrees(np.arccos(np.clip(-approach[2], -1, 1))))
                    if perr < 0.004 and tilt < max_tilt:
                        good = True
                        break
                best = good if ang == 0.0 else (best and good)
            ok.append((round(float(r), 3), best))
        rs = [r for r, b in ok if b]
        rows.append((round(float(z), 3), (min(rs), max(rs)) if rs else None))
        print(f"z {z:.3f}: {rows[-1][1]}  holes: {[r for r, b in ok if not b and rs and min(rs) < r < max(rs)]}",
              flush=True)


if __name__ == "__main__":
    import sys
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 15.0)
