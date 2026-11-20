#!/bin/bash
# One-time setup: make the harness's env-server launcher run the SO-101 twin.
# The ReAct driver (harness/server/episode_launch.py) always starts `$HARNESS_DIR/.venv-libero/bin/python -m
# captain_harness.server.env_server` with PYTHONPATH=$HARNESS_DIR/src. This writes that file as a wrapper that restores
# this pipeline's environment (env.sh) and routes the env server to soarm.env_server_main (suite soarm_sim registered);
# every other invocation (e.g. scripts/react_episode.py) runs unchanged in $SOARM_VENV.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env.sh"
[ -x "$SOARM_VENV/bin/python" ] || { echo "SOARM_VENV=$SOARM_VENV has no bin/python (ROLLOUT.md section 2)" >&2; exit 1; }
mkdir -p "$HARNESS_DIR/.venv-libero/bin"
cat > "$HARNESS_DIR/.venv-libero/bin/python" <<WRAP
#!/bin/bash
source "$HERE/env.sh"
args=("\$@")
if [ "\${args[0]:-}" = "-m" ] && [ "\${args[1]:-}" = "captain_harness.server.env_server" ]; then args[1]=soarm.env_server_main; fi
exec "\$SOARM_VENV/bin/python" "\${args[@]}"
WRAP
chmod +x "$HARNESS_DIR/.venv-libero/bin/python"
echo "wrapper: $HARNESS_DIR/.venv-libero/bin/python -> $SOARM_VENV/bin/python (env from $HERE/env.sh)"
