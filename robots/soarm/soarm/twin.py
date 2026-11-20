"""Kinematic twin of the SO-101 with the robosuite names the harness facade expects (adapters/hardware/facade.py).

Cut from the same Menagerie MJCF the physics uses, so twin FK of the measured joints equals the simulated pose:
robot0_joint1..5 = shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll; robot0_joint6/7 are locked
placeholders (+-1e-3 rad) so the 7-joint contract holds (the WidowX twin does the same for one joint).
gripper0_eef / gripper0_grip_site follow the Franka/LIBERO convention: z = approach, x = jaw closing axis, origin at the
grip point. Two symmetric slider fingers stand in for the SO-101's fixed + hinged jaw (opening 0..9 cm).
Only primitive collision geoms are kept (no meshes), so the twin XML is self-contained.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .scene import ARM_JOINTS, menagerie_dir

# Menagerie `gripperframe` site: x = approach (forward at q = 0), z = jaw closing axis. The fixed jaw is a thin finger
# whose inner face lies at z = -0.020 (x -0.06..+0.005); the moving jaw tip goes from z = -0.016 (closed) to +0.067
# (1.0 rad). An object of width w ends up centred at z = -0.020 + w/2 once the jaw closes.
# Grip point in site coordinates: 0.8 cm behind the tips (the jaw collision meshes reach ~0.6 cm past the tip spheres,
# so a grip point at the centre of a >= 3 cm object keeps them off the table), at z = -0.002 so objects up to 3.5 cm
# wide clear the fixed jaw on the way down; closing slides narrower objects <= 6 mm onto the fixed jaw.
GRIP_OFFSET_SITE = np.array([-0.008, 0.0, -0.002])
# EEF axes as columns in site coordinates: x_eef = site z (closing), y_eef = -site y, z_eef = site x (approach)
EEF_ROT_SITE = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
FINGER_STROKE = 0.036                  # per finger; opening = 2 x stroke x open_fraction (7.2 cm at JAW_OPEN_RAD)
FINGER_HALF_WIDTH = 0.010
FINGER_THICKNESS = 0.008
FINGER_LEN = 0.045
FINGER_TIP = 0.010                     # finger shapes reach this far beyond the grip point along the approach axis
JAW_CLOSED_RAD = -0.17
JAW_OPEN_RAD = 0.75                    # ~7 cm tip gap; wider swings sweep the hinged jaw into neighbouring objects


def grip_in_gripper_body(model: mujoco.MjModel) -> np.ndarray:
    """4x4 grip-site pose in the `gripper` body frame (from the Menagerie gripperframe site + the offsets above)."""
    sid = model.site("gripperframe").id
    site = np.eye(4)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.site_quat[sid])
    site[:3, :3] = rot.reshape(3, 3)
    site[:3, 3] = model.site_pos[sid]
    local = np.eye(4)
    local[:3, :3] = EEF_ROT_SITE
    local[:3, 3] = GRIP_OFFSET_SITE
    return site @ local


def build_twin_xml() -> str:
    spec = mujoco.MjSpec.from_file(str(menagerie_dir() / "so101.xml"))
    spec.modelname = "soarm_twin"
    physics = spec.compile()
    grip = grip_in_gripper_body(physics)
    for a in list(spec.actuators):
        spec.delete(a)
    for c in list(spec.cameras):
        spec.delete(c)
    for lgt in list(spec.lights):
        spec.delete(lgt)
    spec.delete(spec.body("moving_jaw_so101_v1"))
    spec.delete(spec.body("camera_mount"))
    for g in list(spec.geoms):
        if g.type == mujoco.mjtGeom.mjGEOM_MESH or g.group not in (3,):
            spec.delete(g)
    for m in list(spec.meshes):
        spec.delete(m)
    for b in spec.bodies:
        for i, g in enumerate(b.geoms):
            if not g.name:
                g.name = f"{b.name}_col{i}"
    for k, name in enumerate(ARM_JOINTS, 1):
        spec.joint(name).name = f"robot0_joint{k}"
    parent = spec.body("gripper")
    for k in (6, 7):
        lock = parent.add_body(name=f"robot0_link{k}")
        lock.add_joint(name=f"robot0_joint{k}", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0], range=[-1e-3, 1e-3],
                       limited=mujoco.mjtLimited.mjLIMITED_TRUE)
        lock.add_geom(name=f"robot0_link{k}_marker", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.001, 0, 0],
                      contype=0, conaffinity=0, mass=1e-4)
        parent = lock
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, grip[:3, :3].reshape(-1))
    eef = parent.add_body(name="gripper0_eef", pos=grip[:3, 3].tolist(), quat=q.tolist())
    eef.add_site(name="gripper0_grip_site", size=[0.004, 0, 0])
    z1, z0 = FINGER_TIP, FINGER_TIP - FINGER_LEN
    for i, sign in ((1, -1.0), (2, 1.0)):
        f = eef.add_body(name=f"gripper0_finger{i}")
        f.add_joint(name=f"gripper0_finger_joint{i}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[sign, 0, 0],
                    range=[0, FINGER_STROKE], limited=mujoco.mjtLimited.mjLIMITED_TRUE)
        f.add_geom(name=f"gripper0_finger{i}_collision", type=mujoco.mjtGeom.mjGEOM_BOX,
                   pos=[sign * FINGER_THICKNESS / 2, 0, (z0 + z1) / 2],
                   size=[FINGER_THICKNESS / 2, FINGER_HALF_WIDTH, (z1 - z0) / 2])
        f.add_geom(name=f"gripper0_finger{i}_pad_collision", type=mujoco.mjtGeom.mjGEOM_BOX,
                   pos=[sign * 0.002, 0, z1 - 0.008], size=[0.002, FINGER_HALF_WIDTH, 0.008])
    spec.body("base").name = "robot0_arm_root"
    spec.worldbody.add_body(name="robot0_base")
    spec.compile()
    names = [b.name for b in spec.bodies if b.name not in ("world", "robot0_base")]
    gripper = [n for n in names if n.startswith("gripper0") or n.startswith("robot0_link")]
    chain = [n for n in names if n not in gripper]
    hand = chain[-3:] + gripper
    seen = set()
    pairs = [(a, b) for i, a in enumerate(chain) for b in chain[i + 1:i + 4]]
    pairs += [(a, b) for i, a in enumerate(hand) for b in hand[i + 1:]]
    for a, b in pairs:
        key = tuple(sorted((a, b)))
        if a != b and key not in seen:
            seen.add(key)
            spec.add_exclude(bodyname1=a, bodyname2=b)
    spec.compile()
    return spec.to_xml()


def jaw_to_open_fraction(angle: float) -> float:
    return float(np.clip((angle - JAW_CLOSED_RAD) / (JAW_OPEN_RAD - JAW_CLOSED_RAD), 0.0, 1.0))


def gripper_action_to_jaw(action: float) -> float:
    """Harness gripper action in [-1, 1] (+1 closes) -> jaw angle target."""
    s = (float(np.clip(action, -1.0, 1.0)) + 1.0) / 2.0
    return JAW_OPEN_RAD + (JAW_CLOSED_RAD - JAW_OPEN_RAD) * s
