"""Pin the asset files a teacher batch renders: copy every file the registry (assets/registry.jsonl) and the texture
list (assets/textures.jsonl) reference from $SOARM_ASSETS into $SOARM_ROOT/assets_frozen/<registry hash>/ and print that
directory. A rebuild of $SOARM_ASSETS (assets/build_cubes.py) during collection then cannot break running episodes.

    export SOARM_ASSETS=$(python tools/freeze_assets.py)      # teacher batches do this
Exits 1 (nothing printed) if a referenced file is missing in the source, e.g. while a rebuild is in progress.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from soarm.objects import REGISTRY_PATH, asset_root  # noqa: E402

TEXTURES = HERE / "assets" / "textures.jsonl"


def referenced() -> list[str]:
    files = []
    for line in REGISTRY_PATH.read_text().splitlines():
        if line.strip():
            vis = json.loads(line).get("visual") or {}
            files += [vis[k] for k in ("mesh", "texture") if vis.get(k)]
    if TEXTURES.exists():
        files += [json.loads(line)["file"] for line in TEXTURES.read_text().splitlines() if line.strip()]
    return sorted(set(files))


def main() -> int:
    digest = hashlib.sha256(REGISTRY_PATH.read_bytes() + (TEXTURES.read_bytes() if TEXTURES.exists() else b""))
    dest = Path(os.environ.get("SOARM_ROOT", "~/soarm_work")).expanduser() / "assets_frozen" / digest.hexdigest()[:12]
    files = referenced()
    if (dest / "COMPLETE").exists():
        print(dest)
        return 0
    src = asset_root()
    missing = [f for f in files if not (src / f).is_file()]
    if missing:
        print(f"freeze_assets: {len(missing)} files missing under {src} (e.g. {missing[0]}); "
              "run `python -m assets.build_cubes` or wait for the running build", file=sys.stderr)
        return 1
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    for f in files:
        (tmp / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / f, tmp / f)
    (tmp / "COMPLETE").write_text(f"{len(files)} files from {src}\n")
    tmp.rename(dest)
    print(dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
