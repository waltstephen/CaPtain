"""Park the real SO-101: torque on where it stands, lift if low, home, the standard rest pose of
configs/hardware/so101_armhost_real.json, torque off. Used by tools/real_arm_web.py (stop / reset)."""
import json
import os

from soarm.real_driver import RealSoarmDriver

cfg = json.load(open(os.environ.get("SOARM_REAL_CONFIG") or os.environ["SOARM_CODE"] + "/configs/hardware/so101_armhost_real.json"))
cfg["log_path"] = os.environ.get("SOARM_PARK_LOG") or None
RealSoarmDriver(cfg).close()
print("parked at the standard rest pose, torque off", flush=True)
