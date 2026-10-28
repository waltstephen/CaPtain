"""Kinematic twin of a SimplerEnv robot arm (Google Robot 7-DoF, WidowX 250s 6-DoF) with robosuite-style names.

The arm chain (root link -> ... -> TCP link) is cut out of the robot's own URDF (the one SAPIEN simulates), so twin FK
of the measured joints equals the simulated TCP pose. Names follow the XR1 facade contract: robot0_base (= the
published robot_base: the arm mount, i.e. the root link shifted by base_offset) -> robot0_joint1..N; a 6-DoF arm gets robot0_joint7, a locked
placeholder (+-1e-3 rad), so the 7-joint contract holds -> gripper0_eef (Franka/LIBERO convention: z = approach,
x = jaw closing axis) at the grip point, with two sliding fingers and gripper0_finger{1,2}_pad_collision boxes sized
like the real fingers. Fingers, wheels, head and sensor links are dropped (the fingers are the twin's sliders).
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

# per robot family. eef_rot: columns = EEF x (jaw closing), y, z (approach) in TCP-link coordinates; grip_offset: grip
# point in TCP-link coordinates (between the closed fingertips). finger_*: twin finger boxes (per finger stroke = half
# the opening range, measured in SAPIEN: Google tips 0.159 m apart open / 0.010 m closed; WidowX 0.074 / 0.030 m);
# tip: how far the finger collision shapes reach beyond the grip point along the approach axis (SAPIEN hulls).
# Google Robot: revolute, curved fingers whose nails sweep 3.7 cm along the approach axis while closing (open: 3.1 cm
# behind the TCP; closing on a 2-15 cm object: at the TCP, up to 0.6 cm beyond it; tools/finger_sweep.py), so the grip
# point is the official TCP itself (where the nails meet the object) and the twin fingers reach 1.2 cm beyond it.
FAMILIES = {
    "google_robot": dict(root="link_base", tcp="link_gripper_tcp", base_offset=[0.0, 0.0, 0.5945],
                         arm=["joint_torso", "joint_shoulder", "joint_bicep", "joint_elbow", "joint_forearm",
                              "joint_wrist", "joint_gripper"],
                         eef_rot=[[0, -1, 0], [1, 0, 0], [0, 0, 1]], grip_offset=[0.0, 0.0, 0.0],
                         stroke=0.08, half_width=0.028, thickness=0.02, finger_len=0.09, tip=0.012,
                         pad_half=(0.004, 0.02, 0.012)),
    "widowx": dict(root="base_link", tcp="ee_gripper_link",
                   arm=["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"],
                   eef_rot=[[0, 0, 1], [1, 0, 0], [0, 1, 0]], grip_offset=[-0.012, 0.0, 0.0],
                   stroke=0.022, half_width=0.0145, thickness=0.016, finger_len=0.055, tip=0.0276,
                   pad_half=(0.004, 0.0145, 0.012)),
}


def _quat_wxyz(matrix) -> list[float]:
    x, y, z, w = Rotation.from_matrix(np.asarray(matrix, float)).as_quat()
    return [float(w), float(x), float(y), float(z)]


def chain_urdf(full_urdf: Path, root: str, tip: str) -> str:
    """The root..tip kinematic chain of a URDF as a standalone URDF string (absolute mesh paths, OBJ/STL collision only,
    no visuals)."""
    tree = ET.parse(full_urdf).getroot()
    joints = {j.find("child").get("link"): j for j in tree.findall("joint")}
    chain, link = [tip], tip
    while link != root:
        j = joints[link]
        link = j.find("parent").get("link")
        chain.append(link)
    keep = set(chain)
    robot = ET.Element("robot", name=tree.get("name", "robot"))
    folder = Path(full_urdf).resolve().parent
    for el in tree.findall("link"):
        if el.get("name") not in keep:
            continue
        for vis in el.findall("visual"):
            el.remove(vis)
        for col in el.findall("collision"):
            mesh = col.find("geometry/mesh")
            if mesh is not None:
                fn = mesh.get("filename")
                if not fn.lower().endswith((".obj", ".stl")):
                    el.remove(col)
                    continue
                mesh.set("filename", (folder / fn).as_posix())
        robot.append(el)
    for el in tree.findall("joint"):
        if el.find("child").get("link") in keep and el.find("parent").get("link") in keep:
            for tag in ("mimic", "dynamics", "safety_controller", "calibration"):
                for x in el.findall(tag):
                    el.remove(x)
            lim = el.find("limit")
            if lim is not None:
                for k in ("acceleration", "jerk"):
                    lim.attrib.pop(k, None)
            robot.append(el)
    return ET.tostring(robot, encoding="unicode")


def build_arm_twin(full_urdf: Path, family: str, *, name: str = "simplerenv_arm") -> str:
    """MJCF string of the arm; robot0_base = the URDF root link (published robot_base)."""
    fam = FAMILIES[family]
    src = chain_urdf(Path(full_urdf), fam["root"], fam["tcp"])
    src = re.sub(r"(<robot[^>]*>)", r'\1<mujoco><compiler fusestatic="false" discardvisual="true" '
                 r'strippath="false" balanceinertia="true"/></mujoco>', src, count=1)
    spec = mujoco.MjSpec.from_string(src)
    spec.modelname = name
    # the URDF root becomes the world body: wrap the tree under robot0_base
    spec.compile()
    tcp = spec.body(fam["tcp"])
    for b in spec.bodies:
        for i, g in enumerate(b.geoms):
            if not g.name:
                g.name = f"{b.name}_hull{i}"
    parent = tcp
    if len(fam["arm"]) < 7:
        lock = tcp.add_body(name="robot0_link7", pos=[0, 0, 0])
        lock.add_joint(name="robot0_joint7", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0], range=[-1e-3, 1e-3],
                       limited=True)
        lock.add_geom(name="robot0_link7_marker", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.001, 0, 0],
                      contype=0, conaffinity=0, mass=1e-4)
        parent = lock
    rot = np.asarray(fam["eef_rot"], float)
    eef = parent.add_body(name="gripper0_eef", pos=list(fam["grip_offset"]), quat=_quat_wxyz(rot))
    eef.add_site(name="gripper0_grip_site", size=[0.004, 0, 0])
    z1 = fam["tip"]
    z0 = z1 - fam["finger_len"]
    for i, sign in ((1, -1.0), (2, 1.0)):
        f = eef.add_body(name=f"gripper0_finger{i}")
        f.add_joint(name=f"gripper0_finger_joint{i}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[sign, 0, 0],
                    range=[0, fam["stroke"]], limited=True)
        f.add_geom(name=f"gripper0_finger{i}_collision", type=mujoco.mjtGeom.mjGEOM_BOX,
                   pos=[sign * fam["thickness"] / 2, 0, (z0 + z1) / 2],
                   size=[fam["thickness"] / 2, fam["half_width"], (z1 - z0) / 2])
        ph = fam["pad_half"]
        f.add_geom(name=f"gripper0_finger{i}_pad_collision", type=mujoco.mjtGeom.mjGEOM_BOX,
                   pos=[sign * ph[0], 0, z1 - ph[2]], size=list(ph))
    for k, joint in enumerate(fam["arm"], 1):
        spec.joint(joint).name = f"robot0_joint{k}"
    spec.compile()
    # robot0_base = the published robot_base: the root link moved by -base_offset (Google Robot: the arm mount = the torso
    # joint frame at q=0, 0.5945 m above the mobile base; WidowX: its base_link)
    root = spec.body(fam["root"])
    root.pos = (-np.asarray(fam.get("base_offset", [0, 0, 0]), float)).tolist()
    spec.worldbody.add_body(name="robot0_base")
    spec.compile()
    names = [b.name for b in spec.bodies if b.name not in ("world", "robot0_base")]
    gripper = [n for n in names if n.startswith("gripper0") or n == "robot0_link7"]
    chain = [n for n in names if n not in gripper]
    hand = chain[-3:] + gripper
    pairs = [(a, b) for i, a in enumerate(chain) for b in chain[i + 1:i + 4]]
    pairs += [(a, b) for i, a in enumerate(hand) for b in hand[i + 1:]]
    seen = set()
    for a, b in pairs:
        key = tuple(sorted((a, b)))
        if a != b and key not in seen:
            seen.add(key)
            spec.add_exclude(bodyname1=a, bodyname2=b)
    spec.compile()
    return spec.to_xml()
