"""RGB-D + segmentation frames of the SO-101 scene (CPU OSMesa or EGL), with intrinsics and optical extrinsics.

MoGe-2 depth episodes (SOARM_DEPTH=moge2) render RGB only (depth=False): the frame's depth_m is None and the harness's
monocular depth path estimates it from the RGB, as on the real RGB-only arm."""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

# MuJoCo camera frame: x right, y up, looks along -z; optical frame: x right, y down, z forward.
CAM_TO_OPTICAL = np.diag([1.0, -1.0, -1.0])


@dataclass
class Frame:
    rgb: np.ndarray            # HxWx3 uint8
    depth_m: np.ndarray | None  # HxW float32 (rendered depth); None when the scene renders RGB only
    seg_object: np.ndarray | None   # HxW int16: index into names (-1 = background / robot); None without segmentation
    seg_robot: np.ndarray | None    # HxW bool: robot geoms; None without segmentation
    intrinsics: np.ndarray     # 3x3
    optical_in_base: np.ndarray  # 4x4 (world = robot base)


class Cameras:
    def __init__(self, model: mujoco.MjModel, geom_object: dict[int, str], names: list[str], size=(512, 512),
                 segmentation: bool = True, depth: bool = True) -> None:
        self.model = model
        self.h, self.w = size
        self._rgb = mujoco.Renderer(model, self.h, self.w)
        self._depth = None
        if depth:   # simulator sensor depth: tests / ablation / host-side measurement only in MoGe-2 episodes
            self._depth = mujoco.Renderer(model, self.h, self.w)
            self._depth.enable_depth_rendering()
        self._seg = None
        if segmentation:   # privileged (simulator ground truth): never rendered in real-perception episodes
            self._seg = mujoco.Renderer(model, self.h, self.w)
            self._seg.enable_segmentation_rendering()
        self.names = list(names)
        lut = np.full(model.ngeom + 1, -1, np.int16)
        index = {n: i for i, n in enumerate(self.names)}
        for g, n in geom_object.items():
            lut[g] = index[n]
        self._lut = lut
        robot_bodies = set()
        root = model.body("base").id
        for b in range(model.nbody):
            p = b
            while p != 0 and p != root:
                p = int(model.body_parentid[p])
            if p == root:
                robot_bodies.add(b)
        self._robot_geom = np.zeros(model.ngeom + 1, bool)
        for g in range(model.ngeom):
            if int(model.geom_bodyid[g]) in robot_bodies:
                self._robot_geom[g] = True

    def render(self, data: mujoco.MjData, camera: str) -> Frame:
        m = self.model
        cid = m.camera(camera).id
        self._rgb.update_scene(data, camera=camera)
        rgb = self._rgb.render().copy()
        depth = None
        if self._depth is not None:
            self._depth.update_scene(data, camera=camera)
            depth = self._depth.render().astype(np.float32)
        f = 0.5 * self.h / np.tan(np.deg2rad(m.cam_fovy[cid]) / 2)
        K = np.array([[f, 0, (self.w - 1) / 2], [0, f, (self.h - 1) / 2], [0, 0, 1.0]])
        pose = np.eye(4)
        pose[:3, :3] = data.cam_xmat[cid].reshape(3, 3) @ CAM_TO_OPTICAL
        pose[:3, 3] = data.cam_xpos[cid]
        if self._seg is None:
            return Frame(rgb=rgb, depth_m=depth, seg_object=None, seg_robot=None, intrinsics=K, optical_in_base=pose)
        self._seg.update_scene(data, camera=camera)
        seg = self._seg.render()
        ids = seg[..., 0].astype(np.int64)
        is_geom = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
        ids = np.where(is_geom & (ids >= 0), ids, m.ngeom)
        return Frame(rgb=rgb, depth_m=depth, seg_object=self._lut[ids], seg_robot=self._robot_geom[ids],
                     intrinsics=K, optical_in_base=pose)

    def close(self) -> None:
        for r in (self._rgb, self._depth, self._seg):
            if r is not None:
                r.close()
