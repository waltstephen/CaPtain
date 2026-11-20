"""Table / floor / wall textures for scene randomisation: CC0 material sets from ambientCG (ambientcg.com, CC0-1.0).

    python -m assets.fetch_textures [--out $SOARM_ASSETS] [--per-query 4]

Per query term: the most popular materials that have a 1K-JPG download; only the base-colour map is kept, resized to
512 px (<out>/textures/<assetId>.jpg, ~60 KB each). Rows go to assets/textures.jsonl (committed; the images are not):
{id, use: table|floor|wall, query, file, tile_m (physical size of one texture tile), source, licence, url}.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import time
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
API = "https://ambientcg.com/api/v2/full_json?type=Material&q={q}&limit={n}&sort=Popular&include=downloadData,dimensionsData"
# (query, use): table tops the arm works on; floor/wall = background seen past the table edge
QUERIES = (("wood", "table"), ("planks", "table"), ("fabric", "table"), ("plastic", "table"),
           ("marble", "table"), ("paper", "table"), ("leather", "table"), ("cardboard", "table"),
           # terrazzo chips read as small cubes on the table (contact sheet 11-10): background only
           ("concrete", "floor"), ("tiles", "floor"), ("carpet", "floor"), ("asphalt", "floor"), ("terrazzo", "floor"),
           ("plaster", "wall"), ("bricks", "wall"), ("painted", "wall"))
UA = {"User-Agent": "soarm-assets/1 (research; texture fetch)"}


def _get(url: str, timeout: float = 60) -> bytes:
    for k in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except Exception:                       # noqa: BLE001 (network retry, re-raised after 3 tries)
            if k == 2:
                raise
            time.sleep(2 + 3 * k)
    raise AssertionError


def fetch(out: Path, per_query: int) -> list[dict]:
    rows, seen = [], set()
    (out / "textures").mkdir(parents=True, exist_ok=True)
    for q, use in QUERIES:
        found = json.loads(_get(API.format(q=q, n=per_query * 3)))["foundAssets"]
        got = 0
        for a in found:
            aid = a["assetId"]
            if got >= per_query or aid in seen:
                continue
            dls = [d for f in a.get("downloadFolders", {}).values()
                   for d in f["downloadFiletypeCategories"].get("zip", {}).get("downloads", []) if d["attribute"] == "1K-JPG"]
            if not dls:
                continue
            dst = out / "textures" / f"{aid}.png"           # MuJoCo reads PNG (or its own binary), not JPEG
            if not dst.exists():
                z = zipfile.ZipFile(io.BytesIO(_get(dls[0]["fullDownloadPath"])))
                colour = [n for n in z.namelist() if n.endswith("_Color.jpg")]
                if not colour:
                    continue
                Image.open(z.open(colour[0])).convert("RGB").resize((512, 512), Image.LANCZOS).save(dst)
            dim = a.get("dimensionX")
            rows.append({"id": aid, "use": use, "query": q, "file": f"textures/{aid}.png",
                         "tile_m": round(float(dim) / 100, 3) if dim else None, "source": "ambientCG",
                         "licence": "CC0-1.0", "url": f"https://ambientcg.com/view?id={aid}"})
            seen.add(aid)
            got += 1
        print(f"{q:10s} {use:5s} {got}", flush=True)
    return rows


def main() -> None:
    root = os.environ.get("SOARM_ASSETS") or str(Path(os.environ.get("SOARM_ROOT", "~/soarm_work")) / "assets")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=root)
    ap.add_argument("--per-query", type=int, default=4)
    ap.add_argument("--rows", default=str(HERE / "textures.jsonl"))
    a = ap.parse_args()
    rows = fetch(Path(a.out).expanduser(), a.per_query)
    Path(a.rows).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    print(f"{len(rows)} textures -> {a.rows}")


if __name__ == "__main__":
    main()
