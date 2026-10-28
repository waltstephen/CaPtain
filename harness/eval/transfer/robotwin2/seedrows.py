"""shared loader: seed rows of one task, last row per seed wins, INFRA-error rows ignored (treated as unchecked)"""
import json
INFRA = ("OutOfMemoryError", "out of memory", "vk::", "ErrorInitializationFailed", "Offset increment outside graph capture",
         "cuRobo not installed", "CUDA error", "cannot create buffer", "CUBLAS", "cuDNN")
def is_infra(d): return d.get("reason") == "expert_error" and any(k in (d.get("error") or "") for k in INFRA)
def load(path):
    rows = {}
    try:
        for l in open(path):
            try: d = json.loads(l)
            except Exception: continue
            if is_infra(d):
                if d["seed"] in rows and not is_infra(rows[d["seed"]]): continue
                rows.pop(d["seed"], None); continue
            rows[d["seed"]] = d
    except FileNotFoundError: pass
    return rows
