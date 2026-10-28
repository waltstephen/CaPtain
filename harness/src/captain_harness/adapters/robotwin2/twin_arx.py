"""Kinematic twin of one arm of RoboTwin's aloha-agilex (ARX5, 6-DoF) with robosuite-style names (XR1 facade contract).

Adapted from the RoboDojo backend's ARX X5 twin (adapters/robodojo/twin.py). The arm chain (<p>_base_link, <p>_link1..8, <p>_joint1..8, wrist
camera) is cut out of RoboTwin's own arx5_description_isaac.urdf (the robot SAPIEN simulates) and renamed to the X5
names, so twin FK of the measured joints equals the simulated link6 pose. Names: robot0_base (the published robot_base = the primary arm's base_link) -> base_link
of this arm at its base pose in robot_base -> robot0_joint1..6 (= X5 joint1..6) -> link6 -> robot0_joint7, a
locked placeholder joint (range +-1e-3 rad) so the 7-joint facade contract holds for the 6-DoF arm -> gripper0_eef
(Franka/LIBERO convention: z = approach = link6 x, x = jaw closing axis = link6 y) placed TCP_X_M along the
approach axis (finger pad centre) with two sliding fingers and gripper0_finger{1,2}_pad_collision boxes.
"""
from __future__ import annotations

import re
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ARM_JOINTS_URDF = tuple(f"joint{k}" for k in range(1, 7))
# arx5_description_isaac.urdf: jaw joints sit 0.08457 m along link6 x; link7/8 meshes reach 0.071 m beyond (tips ~0.156).
# EEF/grip site at the pad centre ~1.6 cm behind the tips, like LIBERO's Panda EEF (0.0965 vs tips 0.112).
TCP_X_M = 0.140
JAW_JOINT_X_M = 0.08457
FINGER_STROKE_M = 0.04765        # per finger (joint7 / joint8 upper limit)
FINGER_HALF_WIDTH_M = 0.015
FINGER_THICKNESS_M = 0.012
FINGER_Z_RANGE_EEF = (JAW_JOINT_X_M - TCP_X_M, 0.1556 - TCP_X_M)
PAD_HALF = (0.004, FINGER_HALF_WIDTH_M, 0.012)
# eef axes expressed in link6: x_eef = link6 y, y_eef = link6 z, z_eef = link6 x
LINK6_TO_EEF_ROT = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
DROP_BODIES = ("link7", "link8")


def _quat_wxyz(matrix: np.ndarray) -> list[float]:
    x, y, z, w = Rotation.from_matrix(matrix).as_quat()
    return [float(w), float(x), float(y), float(z)]


def _urdf_spec(urdf: Path) -> mujoco.MjSpec:
    folder = urdf.resolve().parent
    src = urdf.read_text()
    src = re.sub(r"<visual>.*?</visual>", "", src, flags=re.S)
    # keep only STL collision meshes (glb camera meshes are not loadable by MuJoCo)
    src = re.sub(r"<collision>(?:(?!</collision>).)*?\.glb.*?</collision>", "", src, flags=re.S)
    src = re.sub(r'filename="([^"]*)"', lambda m: f'filename="{(folder / m.group(1)).as_posix()}"', src)
    src = re.sub(r"(<robot[^>]*>)", r'\1<mujoco><compiler fusestatic="false" discardvisual="true" '
                 r'strippath="false" balanceinertia="true"/></mujoco>', src, count=1)
    return mujoco.MjSpec.from_string(src)


def extract_arm_urdf(full_urdf: Path, prefix: str, camera_link: str, out: Path) -> Path:
    """One arm of the aloha URDF as a standalone URDF with X5 names (base_link, link1..8, joint1..8, camera)."""
    import xml.etree.ElementTree as ET
    root = ET.parse(full_urdf).getroot()
    rename = {f"{prefix}_base_link": "base_link", camera_link: "camera", f"{camera_link}_joint": "camera_joint"}
    rename.update({f"{prefix}_link{k}": f"link{k}" for k in range(1, 9)})
    rename.update({f"{prefix}_joint{k}": f"joint{k}" for k in range(1, 9)})
    keep_links = {k for k in rename if not k.endswith("joint") and "_joint" not in k}
    robot = ET.Element("robot", name=f"arx5_{prefix}")
    for el in root.findall("link"):
        if el.get("name") in keep_links:
            el.set("name", rename[el.get("name")]); robot.append(el)
    for el in root.findall("joint"):
        parent, child = el.find("parent").get("link"), el.find("child").get("link")
        if parent in keep_links and child in keep_links:
            el.set("name", rename.get(el.get("name"), el.get("name")))
            el.find("parent").set("link", rename[parent]); el.find("child").set("link", rename[child])
            robot.append(el)
    xml = ET.tostring(robot, encoding="unicode")
    xml = xml.replace('filename="', f'filename="{Path(full_urdf).resolve().parent.as_posix()}/')
    out.write_text(xml)
    return out


def build_arm_twin(urdf: Path, base_in_robot_base: np.ndarray, joint_limits: np.ndarray | None = None, *,
                   name: str = "robotwin_arx5") -> str:
    """MJCF string of one ARX5 arm whose base_link sits at `base_in_robot_base` (4x4)."""
    spec = _urdf_spec(Path(urdf))
    spec.modelname = name
    spec.worldbody.add_body(name="robot0_base")
    base = spec.body("base_link")
    if base is None:  # root link folded into the world: wrap the first child chain
        raise RuntimeError("ARX5 URDF import: base_link body missing")
    base.pos = np.asarray(base_in_robot_base[:3, 3], float).tolist()
    base.quat = _quat_wxyz(np.asarray(base_in_robot_base[:3, :3], float))
    for body in DROP_BODIES:
        b = spec.body(body)
        if b is not None:
            spec.delete(b)
    spec.compile()
    link6 = spec.body("link6")
    for i, g in enumerate(link6.geoms):
        g.name = "gripper0_palm_collision" if i == 0 else f"gripper0_palm_collision{i}"
    lock = link6.add_body(name="robot0_link7", pos=[0, 0, 0])
    lock.add_joint(name="robot0_joint7", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0], range=[-1e-3, 1e-3],
                   limited=True)
    lock.add_geom(name="robot0_link7_marker", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.001, 0, 0],
                  contype=0, conaffinity=0, mass=1e-4)
    eef = lock.add_body(name="gripper0_eef", pos=[TCP_X_M, 0, 0], quat=_quat_wxyz(LINK6_TO_EEF_ROT))
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
    for k, joint in enumerate(ARM_JOINTS_URDF, 1):
        j = spec.joint(joint)
        j.name = f"robot0_joint{k}"
        if joint_limits is not None:
            lo, hi = (float(v) for v in np.asarray(joint_limits, float)[k - 1])
            j.range = [lo, hi]
            j.limited = True
    spec.compile()
    names = {b.name for b in spec.bodies}
    chain = ["base_link", "link1", "link2", "link3", "link4", "link5", "link6", "camera_base", "camera",
             "robot0_link7", "gripper0_eef", "gripper0_finger1", "gripper0_finger2"]
    # ARX5 (aloha): convex hulls of the long forearm links overlap the wrist/hand in bent postures where SAPIEN has no
    # contact (link3_hull0 vs gripper0_palm_collision 6 cm "penetration" at a top-down ready pose) -> exclude
    # pairs up to three links apart and the forearm (link3/link4) against the hand
    near = [(a, b) for i, a in enumerate(chain) for b in chain[i + 1:i + 4]]
    hand = ["link3", "link4", "link5", "link6", "camera_base", "camera", "robot0_link7", "gripper0_eef", "gripper0_finger1",
            "gripper0_finger2"]
    pairs = near + [(a, b) for i, a in enumerate(hand) for b in hand[i + 1:]]
    seen = set()
    for a, b in pairs:
        key = tuple(sorted((a, b)))
        if a in names and b in names and a != b and key not in seen:
            seen.add(key)
            spec.add_exclude(bodyname1=a, bodyname2=b)
    spec.compile()
    return spec.to_xml()
