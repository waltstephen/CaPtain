"""mujoco_py/robosuite-style accessors over a plain mujoco model, as LiberoActionBackend expects."""
from __future__ import annotations

import mujoco
import numpy as np

# URDF/ROS camera links use x forward, y left, z up; images use the optical frame
# (x right, y down, z forward). Columns are the optical axes in link coordinates.
LINK_TO_OPTICAL = np.array([[0., 0., 1., 0.],
                            [-1., 0., 0., 0.],
                            [0., -1., 0., 0.],
                            [0., 0., 0., 1.]])


class ModelView:
    def __init__(self, model: mujoco.MjModel) -> None:
        self._model = model

    def __getattr__(self, name):
        return getattr(self._model, name)

    def _id(self, kind, name: str) -> int:
        index = mujoco.mj_name2id(self._model, kind, name)
        if index < 0:
            raise ValueError(f"no {kind.name} named {name!r}")
        return index

    def body_name2id(self, name):
        return self._id(mujoco.mjtObj.mjOBJ_BODY, name)

    def site_name2id(self, name):
        return self._id(mujoco.mjtObj.mjOBJ_SITE, name)

    def joint_name2id(self, name):
        return self._id(mujoco.mjtObj.mjOBJ_JOINT, name)

    def geom_name2id(self, name):
        return self._id(mujoco.mjtObj.mjOBJ_GEOM, name)

    def camera_name2id(self, name):
        return self._id(mujoco.mjtObj.mjOBJ_CAMERA, name)

    def sensor_name2id(self, name):
        return self._id(mujoco.mjtObj.mjOBJ_SENSOR, name)


class DataView:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._model = model
        self._data = data

    def __getattr__(self, name):
        return getattr(self._data, name)

    def get_camera_xmat(self, name):
        return self._data.cam_xmat[mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, name)].reshape(3, 3)

    def get_camera_xpos(self, name):
        return self._data.cam_xpos[mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, name)]


class SimView:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = ModelView(model)
        self.data = DataView(model, data)

    def forward(self):
        mujoco.mj_forward(self.model._model, self.data._data)


def site_pose(model, data, site_id: int) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = np.asarray(data.site_xmat[site_id]).reshape(3, 3)
    pose[:3, 3] = data.site_xpos[site_id]
    return pose


def body_pose(model, data, body_id: int) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = np.asarray(data.xmat[body_id]).reshape(3, 3)
    pose[:3, 3] = data.xpos[body_id]
    return pose
