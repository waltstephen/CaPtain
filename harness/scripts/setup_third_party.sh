#!/usr/bin/env bash
# Fetch the pinned upstream code into third_party/ (revisions from third_party/manifest.json).
#   scripts/setup_third_party.sh            # LIBERO only (required by the environment server)
#   scripts/setup_third_party.sh --capx     # also CaP-X + its sam3 submodule (only needed to run the SAM3 perception service)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
rev() { python3 -c "import json,sys; print(json.load(open('third_party/manifest.json'))['dependencies'][sys.argv[1]]['revision'])" "$1"; }
url() { python3 -c "import json,sys; print(json.load(open('third_party/manifest.json'))['dependencies'][sys.argv[1]]['url'])" "$1"; }
fetch() {  # name dir
  if [ ! -d "$2/.git" ]; then git clone "$(url "$1")" "$2"; fi
  git -C "$2" fetch --quiet origin && git -C "$2" checkout --quiet "$(rev "$1")"
  echo "$1 -> $2 @ $(git -C "$2" rev-parse --short HEAD)"
}
fetch libero third_party/libero
if [ "${1:-}" = "--capx" ]; then
  fetch cap_x third_party/cap-x
  git -C third_party/cap-x submodule update --init capx/third_party/sam3 || true
  sam3_rev=$(python3 -c "import json; print(json.load(open('third_party/manifest.json'))['dependencies']['cap_x']['experiment_submodules']['capx/third_party/sam3'])")
  git -C third_party/cap-x/capx/third_party/sam3 checkout --quiet "$sam3_rev" && echo "sam3 @ $sam3_rev"
fi
