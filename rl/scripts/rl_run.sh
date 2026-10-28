#!/usr/bin/env bash
# fresh_run lifecycle on ONE node (run it on the node it names). The run dir must be on a filesystem every node shares.
#   bash rl/scripts/rl_run.sh init     <config.json> [node]   # manifest + source hashes (default: the first training node)
#   bash rl/scripts/rl_run.sh services <config.json> [node]   # rollout node: toolchain preflight, policy proxy :8512, env server :8402
#   bash rl/scripts/rl_run.sh train    <config.json> [node] [--resume]
#   bash rl/scripts/rl_run.sh status   <config.json> [node]
#   bash rl/scripts/rl_run.sh stop     <config.json> [node]   # stops the train/env/proxy roles fresh_run recorded on this host
#   bash rl/scripts/rl_run.sh relocate <config.json> [node]   # change layout keys (hosts, GPUs, ports) of an existing run
# GPU services (SAM3, Molmo: harness/services/start_perception.sh; policy vLLMs: rl/scripts/policies.sh) must be up first.
set -uo pipefail
source "$(dirname "$0")/env.sh"
ACTION=${1:-}; CFG=${2:-}
case "$ACTION" in init|services|status|stop|train|relocate) ;; *) sed -n 2,10p "$0"; exit 2 ;; esac
[ -n "$CFG" ] || { echo "config.json required"; exit 2; }
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY MASTER_ADDR MASTER_PORT RANK WORLD_SIZE LOCAL_RANK NODE_RANK
M="$(cfgget "$CFG" "c['run_dir']")/manifest.json"
DEF=$(cfgget "$CFG" "(c.get('rollout_nodes') or ['a'])[0] if '$ACTION' == 'services' else (c.get('train_nodes') or ['a'])[0]")
NODE=${3:-$DEF}
mkdir -p "$(dirname "$M")"
export PYTHONPATH="$CAPTAIN_ROOT/rl"
cd "$CAPTAIN_ROOT"
case "$ACTION" in
  init|relocate) exec "$CAPTAIN_PY" rl/fresh_run.py --manifest "$M" --node "$NODE" "$ACTION" --config "$CFG" ;;
  train) exec "$CAPTAIN_PY" rl/fresh_run.py --manifest "$M" --node "$NODE" train ${4:-} ;;
  *) exec "$CAPTAIN_PY" rl/fresh_run.py --manifest "$M" --node "$NODE" "$ACTION" ;;
esac
