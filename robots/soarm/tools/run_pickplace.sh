#!/usr/bin/env bash
# Pick & place on the real SO-101 (armhost): fixed camera -> hover -> one wrist-camera estimate -> grasp -> place -> rest.
# usage: run_pickplace.sh [red|yellow|blue|any]   (block color to pick; default any = the largest block)
# Run in the dir holding pickplace.py / arm_probe.py / so101_ik.py / so101_armhost_*.json and snaps/; needs
# tools/camstream.py serving 127.0.0.1:8090. Torque stays on between stages; the last stage releases at rest.
# The wrist estimate is taken once, from the hover pose: closer views are partly hidden by the jaws and their estimate
# drifts away from the gripper (11-12: 10 mm per step, a loop chased it).
set -euo pipefail
COLOR=${1:-any}
PY=${PY:-${SOARM_ROOT}/venv-srv/bin/python}      # mujoco + opencv + scipy
ARM=${ARM:-${SOARM_ROOT}/venv-arm/bin/python}    # scservo_sdk
CAL=${CAL:-so101_armhost_cameras.json}
mkdir -p pp snaps

last_raw() {  # joint positions recorded at the last snapshot of an arm_probe log
  "$PY" -c 'import json,sys; print(json.dumps([e for e in map(json.loads, open(sys.argv[1])) if e.get("event") == "snap"][-1]["raw"]))' "$1"
}
read_raw() {  # joint positions read from the servos now
  "$ARM" -c 'import json, scservo_sdk as s
p = s.PortHandler("/dev/ttyACM0"); k = s.PacketHandler(0); p.openPort(); p.setBaudRate(1000000)
print(json.dumps({str(i): k.read2ByteTxRx(p, i, 56)[0] for i in range(1, 7)}))'
}
HOLDING=0
go_home() {  # the arm holds a pose with torque on: plan from where it really is back to rest, release there
  [[ $HOLDING == 1 ]] || return 0
  HOLDING=0
  echo "stage refused/failed: returning to rest"
  "$PY" pickplace.py home "$(read_raw)" pp/plan_home.json && "$ARM" arm_probe.py "$(cat pp/plan_home.json)" > pp/log_home.log
  tail -1 pp/log_home.log | cut -c1-160
}
trap go_home ERR
run() {  # run an arm_probe plan; on abort go home (if holding) and stop the pipeline
  "$ARM" arm_probe.py "$(cat "$1")" > "$2"
  if grep -q '"abort"' "$2"; then grep '"abort"' "$2"; echo "arm_probe aborted ($2)"; go_home; exit 1; fi
}

read_raw > pp/start.json
curl -sf -m 20 -o pp/scene.png http://127.0.0.1:8090/snap/2

"$PY" pickplace.py hover pp/scene.png "$COLOR" "$CAL" "$(cat pp/start.json)" pp/plan_hover.json
HOLDING=1  # from the first motion on, any failure must bring the arm home (hover plans keep torque on)
run pp/plan_hover.json pp/log_hover.log
raw=$(last_raw pp/log_hover.log)

"$PY" pickplace.py approach "$CAL" "$raw" snaps/pp_hover_wrist.png pp/plan_near.json
run pp/plan_near.json pp/log_near.log
raw=$(last_raw pp/log_near.log)

"$PY" pickplace.py grasp "$CAL" "$raw" pp/plan_grasp.json
run pp/plan_grasp.json pp/log_grasp.log
HOLDING=0  # the grasp plan ends at rest with torque released
grep -E '"id": 6' pp/log_grasp.log | cut -c1-120
