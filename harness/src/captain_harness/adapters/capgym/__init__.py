"""CaP-Gym (CaP-X 2603.22435) single-arm Robosuite tasks behind the LIBERO adapter.

The tasks are rebuilt on the harness's robosuite 1.4 (CaP-Gym pins robosuite 1.5; scene, objects, placement samplers and
success predicates are identical between the two). The wrapper exposes the
LIBERO ControlEnv surface that the CaP-X LIBERO adapter drives, so the model-facing tools and TASK.md are unchanged.
"""

from .environment import CAPGYM_TASKS, CapGymRobosuiteEnv, capgym_init_state

__all__ = ["CAPGYM_TASKS", "CapGymRobosuiteEnv", "capgym_init_state"]
