#!/usr/bin/env bash
# Supervisor of a streaming CaPtain RL run. Start it on the trainer node:
#   POLICY_LAYOUT="<rollout host>:<gpu,gpu,...>:<mem>" PERC_LAYOUT="<rollout host>:<sam3 gpu>:<molmo gpu>:<molmo mem>" \
#   setsid nohup bash rl/scripts/run_stream.sh <config.json> [--resume] > $CAPTAIN_LOGS/<run>-supervisor.log 2>&1 < /dev/null &
# What it does:
#   1. perception (SAM3 :8114, Molmo :8122) on every rollout host (harness/services/start_perception.sh; healthy ones are reused);
#   2. the policy vLLMs (rl/scripts/policies.sh; ports from the config's policy_workers), loaded with base_model (the trainer
#      publishes its current weights to them at start, so a resumed run needs no other weights here);
#   3. fresh_run init (fresh runs only), env server + policy proxy on every rollout node, then the trainer;
#   4. watches the trainer: a dead trainer is resumed from the latest complete checkpoint (MAX_RESTARTS within 2 h, default 3);
#      exits when max_steps is reached. Stop the run with `kill <supervisor pid>` (the trap tears every role down);
#      KEEP_SERVICES=1 leaves the GPU services running.
# Hosts are reached with $CAPTAIN_SSH unless they are this machine. TRAIN_ENV overrides the trainer environment (env.sh).
set -uo pipefail
source "$(dirname "$0")/env.sh"
CFG=$(readlink -f "${1:?config.json}"); RESUME=${2:-}
S="$CAPTAIN_ROOT/rl/scripts"
RUN_DIR=$(cfgget "$CFG" "c['run_dir']"); RUN=$(basename "$RUN_DIR")
MAXSTEPS=$(cfgget "$CFG" "c['max_steps']")
TRAIN_NODES=$(cfgget "$CFG" "' '.join(c.get('train_nodes') or ['a'])")
ROLL_NODES=$(cfgget "$CFG" "' '.join(c.get('rollout_nodes') or ['a'])")
host() { cfgget "$CFG" "c['nodes']['$1']['host']"; }
FIRST=${TRAIN_NODES%% *}
[ "$(echo $TRAIN_NODES | wc -w)" = 1 ] || { echo "this supervisor drives one trainer node (node-local DCP shards)"; exit 2; }
ROLL_HOSTS=""; for n in $ROLL_NODES; do ROLL_HOSTS="$ROLL_HOSTS $(host $n)"; done
POL_HOSTS=$(cfgget "$CFG" "' '.join(sorted({u.split('//')[1].split(':')[0] for u in c['policy_workers']}))")
POL=$(cfgget "$CFG" "' '.join(u.split('//')[1] for u in c['policy_workers'])"); NPOL=$(echo $POL | wc -w)
TRAIN_ENV=${TRAIN_ENV:-$(default_train_env)}
ME_ADDRS=" 127.0.0.1 localhost $(hostname) $(hostname -s) $(hostname -I 2>/dev/null) "
log() { echo "$(date -u +%FT%TZ) $*"; }
on() { local h=$1; shift; case "$ME_ADDRS" in *" $h "*) bash -c "$*" ;; *) $CAPTAIN_SSH "$h" "$*" ;; esac; }
ENVSET="CAPTAIN_ROOT=$CAPTAIN_ROOT CAPTAIN_WORKSPACE=$CAPTAIN_WORKSPACE CAPTAIN_PY=$CAPTAIN_PY CAPTAIN_VLLM_PY=$CAPTAIN_VLLM_PY"
running() { on "$(host $1)" "$ENVSET bash $S/rl_run.sh status $CFG $1" 2>/dev/null | "$CAPTAIN_PY" -c "import json,sys; print(json.load(sys.stdin)['roles']['train']['running'])" 2>/dev/null; }
latest_ckpt() { ls -d "$RUN_DIR"/checkpoint-* 2>/dev/null | sed -n 's/.*checkpoint-\([0-9][0-9]*\)$/\1/p' | sort -n -r | while read k; do
                  [ -s "$RUN_DIR/checkpoint-$k/trainer_state.json" ] && { echo $k; break; }; done; }
mkdir -p "$CAPTAIN_WORKSPACE/state"; echo $$ > "$CAPTAIN_WORKSPACE/state/$RUN.supervisor.pid"
teardown() {
  local rc=$?; log "teardown (exit $rc)"
  on "$(host $FIRST)" "$ENVSET bash $S/rl_run.sh stop $CFG $FIRST" > /dev/null 2>&1
  for n in $ROLL_NODES; do on "$(host $n)" "$ENVSET bash $S/rl_run.sh stop $CFG $n; sleep 5; $ENVSET bash $S/rl_run.sh stop $CFG $n" > /dev/null 2>&1; done
  if [ "${KEEP_SERVICES:-0}" != 1 ]; then for h in $POL_HOSTS; do on "$h" "$ENVSET bash $S/policies.sh down"; done; fi
  rm -f "$CAPTAIN_WORKSPACE/state/$RUN.supervisor.pid"
}
trap teardown EXIT
trap 'log "signal received"; exit 143' INT TERM
log "run $RUN ${RESUME:-fresh}; trainer node $FIRST; rollout nodes $ROLL_NODES"

# ---- 1. perception
for h in $ROLL_HOSTS; do
  pl=$(echo "${PERC_LAYOUT:-}" | tr ';' '\n' | grep "^$h:" | head -1); IFS=: read -r _h sg mg mm <<< "${pl:-$h:0:0:0.45}"
  on "$h" "curl -sf -m 3 localhost:8114/docs > /dev/null || SAM3_GPU=$sg HOST=0.0.0.0 LOG_DIR=$CAPTAIN_LOGS bash $CAPTAIN_ROOT/harness/services/start_perception.sh sam3"
  on "$h" "curl -sf -m 3 localhost:8122/health > /dev/null || MOLMO_GPU=$mg MOLMO_GPU_UTIL=${mm:-0.45} HOST=0.0.0.0 LOG_DIR=$CAPTAIN_LOGS bash $CAPTAIN_ROOT/harness/services/start_perception.sh molmo"
done
# ---- 2. policies
INIT=$(cfgget "$CFG" "c['base_model']")
start_policies() {
  for h in $POL_HOSTS; do
    first=$(cfgget "$CFG" "min(int(u.rsplit(':',1)[1]) for u in c['policy_workers'] if u.split('//')[1].split(':')[0] == '$h')")
    on "$h" "curl -sf -m 3 localhost:$first/health > /dev/null" && continue
    lay=$(echo "${POLICY_LAYOUT:-}" | tr ';' '\n' | grep "^$h:" | head -1)
    [ -n "$lay" ] || { log "POLICY_LAYOUT has no entry for $h"; exit 1; }
    IFS=: read -r _h gpus mem <<< "$lay"
    on "$h" "$ENVSET INIT=$INIT SEQS=${SEQS:-8} bash $S/policies.sh up '$(echo $gpus | tr , ' ')' $mem $first"
  done
  for i in $(seq 90); do up=0; for hp in $POL; do curl -sf -m 3 --noproxy '*' "http://$hp/health" > /dev/null && up=$((up + 1)); done
    [ "$up" = "$NPOL" ] && break; sleep 10; done
  log "policies up: $up/$NPOL"
}
start_policies; [ "$up" = "$NPOL" ] || { log "policy servers did not come up"; exit 1; }
for h in $ROLL_HOSTS; do for p in 8114/docs 8122/health; do
  for i in $(seq 60); do curl -sf -m 5 --noproxy '*' "http://$h:$p" > /dev/null && break; sleep 10; done
  curl -sf -m 5 --noproxy '*' "http://$h:$p" > /dev/null || { log "perception $h:$p not healthy"; exit 1; }; done; done
# ---- 3. manifest, env servers, trainer
if [ -z "$RESUME" ]; then on "$(host $FIRST)" "$ENVSET bash $S/rl_run.sh init $CFG $FIRST" > "$CAPTAIN_LOGS/$RUN-init.json" 2>&1 \
  || { log "init failed: $(tail -3 "$CAPTAIN_LOGS/$RUN-init.json")"; exit 1; }; fi
for n in $ROLL_NODES; do
  on "$(host $n)" "cd /tmp && RL_PROXY_ROUTE=least-loaded $ENVSET bash $S/rl_run.sh services $CFG $n" > "$CAPTAIN_LOGS/$RUN-services-$n.json" 2>&1 \
    || { log "services $n failed: $(tail -5 "$CAPTAIN_LOGS/$RUN-services-$n.json")"; exit 1; }
done
log "env servers + proxies healthy on$ROLL_HOSTS"
on "$(host $FIRST)" "cd /tmp && $TRAIN_ENV $ENVSET bash $S/rl_run.sh train $CFG $FIRST $RESUME" > "$CAPTAIN_LOGS/$RUN-train-$FIRST.json" 2>&1 \
  || { log "train $FIRST failed: $(tail -3 "$CAPTAIN_LOGS/$RUN-train-$FIRST.json")"; exit 1; }
log "trainer started on $FIRST (log $RUN_DIR/logs/)"
# ---- 4. watch
MAX_RESTARTS=${MAX_RESTARTS:-3}; RESTART_TIMES=""
run_done() { "$CAPTAIN_PY" -c "import json
last = 0
for l in open('$RUN_DIR/metrics.jsonl'):
    try:
        s = json.loads(l).get('step')
    except ValueError:
        continue
    if type(s) is int:
        last = max(last, s)
print('yes' if last >= $MAXSTEPS else 'no')" 2>/dev/null | tail -1; }
resume_from_crash() {
  local now recent=0 t N ts; now=$(date +%s)
  for t in $RESTART_TIMES; do [ $((now - t)) -lt 7200 ] && recent=$((recent + 1)); done
  [ $recent -ge "$MAX_RESTARTS" ] && { log "CRASH LIMIT: $recent restarts within 2 h; stopping"; exit 2; }
  RESTART_TIMES="$RESTART_TIMES $now"
  on "$(host $FIRST)" "$ENVSET bash $S/rl_run.sh stop $CFG $FIRST" > /dev/null 2>&1
  N=$(latest_ckpt); log "CRASH RECOVERY $((recent + 1))/$MAX_RESTARTS: resuming from checkpoint-$N"
  # metrics/rollouts rows after the checkpoint are re-collected: trim them (a copy is kept)
  ts=$(date -u +%Y%m%dT%H%M%SZ)
  for f in metrics.jsonl rollouts.jsonl; do [ -f "$RUN_DIR/$f" ] || continue; cp "$RUN_DIR/$f" "$RUN_DIR/$f.before-resume-ckpt$N-$ts"
    "$CAPTAIN_PY" -c "import json,sys; rows=[l for l in open(sys.argv[1]) if l.strip() and int(json.loads(l).get('step', 0)) <= $N]; open(sys.argv[1], 'w').writelines(rows)" "$RUN_DIR/$f"; done
  # policies may have been left mid-NCCL-update: restart them (the resumed trainer republishes its weights); reopen the gates
  for h in $POL_HOSTS; do on "$h" "$ENVSET bash $S/policies.sh down" > /dev/null 2>&1; done
  sleep 10; start_policies
  for h in $ROLL_HOSTS; do curl -s -m 10 --noproxy '*' -X POST "http://$h:$(cfgget "$CFG" "c['proxy_port']")/gate/open" > /dev/null; done
  on "$(host $FIRST)" "cd /tmp && $TRAIN_ENV $ENVSET bash $S/rl_run.sh train $CFG $FIRST --resume" > "$CAPTAIN_LOGS/$RUN-train-$FIRST.json" 2>&1 \
    || log "WARNING resume launch: $(tail -3 "$CAPTAIN_LOGS/$RUN-train-$FIRST.json")"
  sleep 300
}
while sleep 60; do
  if [ "$(run_done)" = yes ]; then
    for i in $(seq 60); do [ "$(running $FIRST)" = True ] || break; sleep 30; done
    log "run complete: step $MAXSTEPS reached"; exit 0
  fi
  if [ "$(running $FIRST)" != True ]; then
    sleep 30; [ "$(run_done)" = yes ] && { log "run complete: step $MAXSTEPS reached"; exit 0; }
    log "!!! trainer not running on $FIRST (see $RUN_DIR/logs/)"; resume_from_crash
  fi
done
