"""Robot-derived contact limits shared by the LIBERO adapter and the direct-programming layer.

Only EMERGENCY_LIMIT_N is an absolute stop. Contact detection uses a per-episode calibrated residual
(``adapters/libero/contact_residual.py``); manipulation load budgets are chosen by the policy per request and
capped by the emergency limit. None of these numbers is tuned on task outcomes.
"""

# Franka Emika Panda: the collision-reflex Cartesian force thresholds are configurable per axis; the manufacturer
# lower/upper reflex defaults are 20 N / 100 N (libfranka setCollisionBehavior documentation). 60 N sits well inside
# the payload-safe envelope and far above the 5-15 N loads of LIBERO drawers and knobs. Safety net only.
EMERGENCY_LIMIT_N = 60.0
# Detection statistics: multiples of the calibrated noise scale, a floor that no calibration can undercut, and the
# number of consecutive 2 ms substeps a residual must persist before it counts as contact.
K_SIGMA = 6.0
MIN_THRESHOLD_N = 1.0
PERSIST_SUBSTEPS = 3
# A residual above IMMEDIATE_FACTOR x threshold on a single substep is an impact and stops at once (no persistence).
IMMEDIATE_FACTOR = 5.0
# Manipulation mode: stall = less than STALL_PROGRESS_M of progress along the commanded direction over STALL_WINDOW_S
# while loaded; a load above the policy's budget must persist SUSTAIN_S before stopping.
STALL_WINDOW_S = 0.5
STALL_PROGRESS_M = 0.002
SUSTAIN_S = 0.25
DEFAULT_LOAD_BUDGET_N = 25.0
LEGACY_FORCE_LIMIT_N = 18.0
