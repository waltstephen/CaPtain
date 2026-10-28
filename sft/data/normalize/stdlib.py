# CaPtain standard helper library (PREAMBLE): prepended to the first python cell of every agent episode in the SFT data.
# The model therefore writes it itself at request 1 -> works unchanged on the frozen harness (no harness change needed).
PREAMBLE = '''import math, json
import numpy as np
def val(r):
    if not r["success"]:
        print(r["failure_category"], r["message"])
        return None
    return r["value"]
def need(r):
    if not r["success"]:
        raise RuntimeError(r["message"])
    return r["value"]
def downq(yaw):
    return [0.0, math.cos(yaw / 2), math.sin(yaw / 2), 0.0]
def go(p, q, speed=0.05, mode="guarded", duration=20, angular=0.5, **kw):
    args = dict(position=np.asarray(p, dtype=float).tolist(), quaternion=np.asarray(q, dtype=float).tolist(),
                max_speed_m_s=speed, max_angular_speed_rad_s=angular, max_duration_s=duration, contact_mode=mode, **kw)
    pre = robot.preview_motion("move_to", args)
    if not pre["success"]:
        print("preview:", pre["failure_category"], pre["message"])
        return pre
    r = robot.move_to(**args)
    print("go:", r["success"], r["failure_category"], r["message"])
    return r
'''
# what the harness prints in the cell's "Changed:" line for these definitions
CHANGED = ("val=function(r); need=function(r); downq=function(yaw); "
           "go=function(p, q, speed=0.05, mode='guarded', duration=20, angular=0.5, **kw)")
NAMES = ("val", "need", "downq", "go")
GO_DEFAULTS = dict(speed=0.05, mode="guarded", duration=20, angular=0.5)
# move_to API defaults (system prompt): used to normalise recorded robot calls before comparing
API_DEFAULTS = dict(max_speed_m_s=0.05, max_angular_speed_rad_s=0.5, max_duration_s=20.0, position_tolerance_m=0.003,
                    orientation_tolerance_rad=0.03, contact_mode="guarded", load_budget_n=None, gripper=None)
