#!/bin/bash
# Fetch MuJoCo Menagerie's robotstudio_so101 (Apache-2.0) at the pinned commit: shallow + partial (blobless) + sparse,
# so only that directory's files are downloaded (~20 MB, not the whole repository).
#   fetch_assets.sh <dest dir>  ->  <dest dir>/robotstudio_so101  (= SOARM_MENAGERIE)
set -euo pipefail
DEST=${1:?dest dir}
COMMIT=98a70c569c4c1b6da467a20012f766ee1f8dc521
mkdir -p "$DEST"
cd "$DEST"
[ -d .git ] || git init -q
git remote get-url origin >/dev/null 2>&1 || git remote add origin https://github.com/google-deepmind/mujoco_menagerie.git
git sparse-checkout set robotstudio_so101
git fetch -q --depth 1 --filter=blob:none origin "$COMMIT"
git checkout -q FETCH_HEAD
test -f robotstudio_so101/so101.xml
echo "SOARM_MENAGERIE=$DEST/robotstudio_so101 (commit $COMMIT)"
