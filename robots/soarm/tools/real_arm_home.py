"""Move the real SO-101 from wherever it rests to the twin's home pose (the episode start pose) and keep torque on.
Used by armcal/flatrec.py before a calibration survey: survey plans start from home, not from the folded rest pose
(from rest the joint-by-joint survey moves swept the wrist camera into the base plate, 11-14). Park afterwards with
tools/real_arm_park.py."""
import json
import os

from soarm.real_driver import RealSoarmDriver

cfg = json.load(open(os.environ.get("SOARM_REAL_CONFIG") or os.environ["SOARM_CODE"] + "/configs/hardware/so101_armhost_real.json"))
d = RealSoarmDriver(cfg)
d.reset()
d.bus.close()   # no park: the arm holds the home pose with torque on
print("at home, torque on", flush=True)
