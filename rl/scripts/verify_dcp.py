#!/usr/bin/env python3
"""verify_dcp.py <dcp dir>: a torch.distributed.checkpoint directory is complete iff .metadata reads and every shard file it
references exists and is at least as long as the furthest (offset + length) the metadata records for it. On success writes
<dir>/.shards_ok = {"files": {name: size}, ...} (an optional integrity check before copying or resuming a checkpoint)."""
import json, os, sys, time
from torch.distributed.checkpoint import FileSystemReader
d = sys.argv[1]
md = FileSystemReader(d).read_metadata()
need = {}
for info in md.storage_data.values():
    need[info.relative_path] = max(need.get(info.relative_path, 0), int(info.offset) + int(info.length))
bad = {f: (os.path.getsize(os.path.join(d, f)) if os.path.exists(os.path.join(d, f)) else None, n)
       for f, n in need.items() if not os.path.exists(os.path.join(d, f)) or os.path.getsize(os.path.join(d, f)) < n}
if bad:
    print(json.dumps({"dir": d, "ok": False, "bad": bad}))
    sys.exit(1)
files = {f: os.path.getsize(os.path.join(d, f)) for f in need}
files[".metadata"] = os.path.getsize(os.path.join(d, ".metadata"))
tmp = os.path.join(d, ".shards_ok.tmp")
json.dump({"files": files, "shards": len(need), "verified_at": time.time(), "host": os.uname().nodename}, open(tmp, "w"))
os.replace(tmp, os.path.join(d, ".shards_ok"))
print(json.dumps({"dir": d, "ok": True, "shards": len(need), "bytes": sum(files.values())}))
