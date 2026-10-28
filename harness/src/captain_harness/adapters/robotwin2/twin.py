"""Kinematic twin of one RoboTwin Franka arm with robosuite-style names.

Built from the embodiment's own panda.urdf (the URDF SAPIEN simulates), so twin FK of
the measured joints equals the simulated pose. world -> robot0_base (the published
robot_base = the primary arm's panda_link0) -> panda_link0 placed at this arm's base
pose in robot_base -> robot0_joint1..7 -> panda_hand -> gripper0_eef (Franka convention:
z approach, x closing; same hand->EEF transform as the LIBERO profile) with two sliding
fingers and gripper0_finger{1,2}_pad_collision boxes. Collision pairs disabled in the
embodiment SRDF are excluded.
"""
from __future__ import annotations

import re
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ARM_JOINTS = tuple(f"panda_joint{k}" for k in range(1, 8))
# LIBERO profile pyroki_hand_to_libero_eef: EEF = hand rotated -90 deg about z, 0.0965 m along the approach axis
HAND_TO_EEF_POS = (0.0, 0.0, 0.0965)
HAND_TO_EEF_ROT = Rotation.from_euler("z", -np.pi / 2)
FINGER_STROKE_M = 0.04          # per finger, URDF panda_finger_joint limit
FINGER_HALF_WIDTH_M = 0.0105    # finger.stl x extent
FINGER_THICKNESS_M = 0.0264     # finger.stl y extent (inner face at 0)
FINGER_Z_RANGE_EEF = (0.0584 - HAND_TO_EEF_POS[2], 0.0584 + 0.0538 - HAND_TO_EEF_POS[2])  # joint .. tip, in EEF z
PAD_HALF = (0.004, FINGER_HALF_WIDTH_M, 0.012)
FINGER_BODIES = {"panda_leftfinger": "gripper0_finger1", "panda_rightfinger": "gripper0_finger2"}


def _quat_wxyz(rotation: Rotation) -> list[float]:
    x, y, z, w = rotation.as_quat()
    return [float(w), float(x), float(y), float(z)]


def _urdf_spec(urdf: Path) -> mujoco.MjSpec:
    folder = urdf.resolve().parent
    src = urdf.read_text()
    src = re.sub(r"<visual>.*?</visual>", "", src, flags=re.S)
    src = re.sub(r'filename="([^"]*)"', lambda m: f'filename="{(folder / m.group(1)).as_posix()}"', src)
    src = re.sub(r"(<robot[^>]*>)", r'\1<mujoco><compiler fusestatic="false" discardvisual="true" '
                 r'strippath="false" balanceinertia="true"/></mujoco>', src, count=1)
    return mujoco.MjSpec.from_string(src)


def _srdf_pairs(srdf: Path | None) -> list[tuple[str, str]]:
    if srdf is None or not Path(srdf).exists():
        return []
    root = ET.parse(srdf).getroot()
    return [(e.get("link1"), e.get("link2")) for e in root.iter("disable_collisions")]


def build_arm_twin(urdf: Path, srdf: Path | None, base_in_robot_base: np.ndarray, *, name: str = "robotwin_franka") -> str:
    """MJCF string of one arm whose panda_link0 sits at `base_in_robot_base` (4x4)."""
    spec = _urdf_spec(Path(urdf))
    spec.modelname = name
    spec.worldbody.add_body(name="robot0_base")
    link0 = spec.body("panda_link0")
    link0.pos = np.asarray(base_in_robot_base[:3, 3], float).tolist()
    link0.quat = _quat_wxyz(Rotation.from_matrix(np.asarray(base_in_robot_base[:3, :3], float)))
    for body in list(FINGER_BODIES):
        spec.delete(spec.body(body))
    spec.compile()
    hand = spec.body("panda_hand")
    for g in hand.geoms:
        g.name = "gripper0_palm_collision"
    eef = hand.add_body(name="gripper0_eef", pos=list(HAND_TO_EEF_POS), quat=_quat_wxyz(HAND_TO_EEF_ROT))
    eef.add_site(name="gripper0_grip_site", size=[0.004, 0, 0])
    z0, z1 = FINGER_Z_RANGE_EEF
    for i, sign in ((1, -1.0), (2, 1.0)):
        f = eef.add_body(name=f"gripper0_finger{i}")
        f.add_joint(name=f"gripper0_finger_joint{i}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[sign, 0, 0],
                    range=[0, FINGER_STROKE_M], limited=True)
        f.add_geom(name=f"gripper0_finger{i}_collision", type=mujoco.mjtGeom.mjGEOM_BOX,
                   pos=[sign * FINGER_THICKNESS_M / 2, 0, (z0 + z1) / 2],
                   size=[FINGER_THICKNESS_M / 2, FINGER_HALF_WIDTH_M, (z1 - z0) / 2])
        f.add_geom(name=f"gripper0_finger{i}_pad_collision", type=mujoco.mjtGeom.mjGEOM_BOX,
                   pos=[sign * PAD_HALF[0], 0, z1 - PAD_HALF[2]], size=list(PAD_HALF))
    for b in spec.bodies:
        for i, g in enumerate(b.geoms):
            if not g.name:
                g.name = f"{b.name}_hull{i}"
    for k, joint in enumerate(ARM_JOINTS, 1):
        spec.joint(joint).name = f"robot0_joint{k}"
    spec.compile()
    names = {b.name for b in spec.bodies}
    srdf_pairs = [(FINGER_BODIES.get(a, a), FINGER_BODIES.get(b, b)) for a, b in _srdf_pairs(srdf)]
    gripper = ["panda_link7", "panda_link8", "panda_hand", "gripper0_eef", "gripper0_finger1", "gripper0_finger2"]
    pairs = srdf_pairs + [(a, b) for i, a in enumerate(gripper) for b in gripper[i + 1:]]
    seen = set()
    for a, b in pairs:
        key = tuple(sorted((a, b)))
        if a in names and b in names and a != b and key not in seen:
            seen.add(key)
            spec.add_exclude(bodyname1=a, bodyname2=b)
    spec.compile()
    return spec.to_xml()


def base_pose_in(reference_world: np.ndarray, base_world: np.ndarray) -> np.ndarray:
    """Pose of `base_world` expressed in the frame `reference_world` (both 4x4 world poses)."""
    return np.linalg.inv(reference_world) @ base_world
