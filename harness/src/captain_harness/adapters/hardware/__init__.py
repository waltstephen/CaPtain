"""Hardware backend interface: a single robot arm behind the frozen LIBERO/CaP-X tool surface.

`facade.XR1Driver` is the protocol a robot (or simulator) driver implements: `reset`, `read`, `command` (absolute
grip-site goal in robot_base + gripper action), `hold` and `frames` (RGB, metric depth, intrinsics, optional extrinsics).
`facade.XR1FacadeEnv` turns such a driver into the robosuite-shaped environment that the LIBERO action backend drives,
with a MuJoCo kinematic twin for forward kinematics, so every tool, receipt and TASK.md section is the one used on
LIBERO. The RoboTwin 2.0, RoboDojo and SimplerEnv adapters are drivers of this interface; `episode.py` builds a
real-robot episode (suite `hardware_real`) from a user-supplied driver. The class names keep the `XR1` prefix of the
first robot this interface was written for.
"""
