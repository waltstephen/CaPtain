"""Immutable JSON bodies with independent bounded cache; no robot semantics."""
from collections import OrderedDict
from collections.abc import MutableMapping
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import sqlite3

from .read_pages import encode

class PublicRecordStore(MutableMapping):
    def __init__(self, root, *, cache_bytes=262144):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.root / "records.sqlite3"))
        self.db.execute("CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, digest TEXT NOT NULL, body TEXT NOT NULL)")
        self.db.commit()
        self.cache_limit = cache_bytes
        self.cache = OrderedDict()
        self.cache_size = 0

    def __setitem__(self, key, value):
        body = encode(value)
        digest = hashlib.sha256(body.encode()).hexdigest()
        existing = self.db.execute("SELECT digest FROM records WHERE id=?", (key,)).fetchone()
        if existing and existing[0] != digest:
            raise ValueError("immutable public record conflict")
        self.db.execute("INSERT OR IGNORE INTO records VALUES (?,?,?)", (key, digest, body))
        self.db.commit()

    def __getitem__(self, key):
        if key in self.cache:
            body = self.cache.pop(key)
            self.cache[key] = body
        else:
            row = self.db.execute("SELECT digest,body FROM records WHERE id=?", (key,)).fetchone()
            if row is None:
                raise KeyError(key)
            digest, body = row
            if hashlib.sha256(body.encode()).hexdigest() != digest:
                raise RuntimeError("public record integrity failure")
            size = len(body.encode())
            if size <= self.cache_limit:
                while self.cache and self.cache_size + size > self.cache_limit:
                    _, old = self.cache.popitem(last=False)
                    self.cache_size -= len(old.encode())
                self.cache[key] = body
                self.cache_size += size
        try:
            return json.loads(body)
        except (ValueError,UnicodeError) as exc:
            raise RuntimeError("invalid stored public JSON") from exc

    def __contains__(self,key):
        return isinstance(key,str) and self.db.execute("SELECT 1 FROM records WHERE id=?",(key,)).fetchone() is not None

    def __delitem__(self, key):
        raise TypeError("immutable evidence cannot be deleted")

    def __iter__(self):
        for row in self.db.execute("SELECT id FROM records ORDER BY rowid"):
            yield row[0]

    def __reversed__(self):
        for row in self.db.execute("SELECT id FROM records ORDER BY rowid DESC"):
            yield row[0]

    def __len__(self):
        return self.db.execute("SELECT COUNT(*) FROM records").fetchone()[0]

    def clear_cache(self):
        self.cache.clear()
        self.cache_size = 0

    def put(self, value):
        identity = "artifact://public/" + hashlib.sha256(encode(value).encode()).hexdigest()
        self[identity] = value
        return identity

    def describe(self, value):
        body = encode(value)
        return {"artifact_id": self.put(value), "type": type(value).__name__,
                "size_bytes": len(body.encode()), "length": len(value) if isinstance(value, (dict,list,str)) else None,
                "path": [], "read": "read_tool_result(artifact_id, path=[], offset=0, limit=128)",
                "page_units": "limit counts container items; text fills the bounded local RPC byte allowance",
                "scope": "immutable historical public value; reading does not refresh action validity"}

    def read_value(self, identity, *, path=(), offset=0, limit=128):
        from .read_errors import ToolInputError
        from .read_pages import artifact_page
        if not isinstance(identity,str) or not identity.startswith("artifact://public/") or identity not in self:
            raise ToolInputError("unknown public artifact; use an issued artifact_id",code="artifact_not_found")
        if type(offset) is not int or offset<0 or type(limit) is not int or not 1<=limit<=1024:
            raise ToolInputError("offset >=0 and container-item limit in [1,1024] required")
        if not isinstance(path,(list,tuple)) or len(path)>32:
            raise ToolInputError("path must have at most32 public keys/indices")
        if any(not isinstance(key,str) and type(key) is not int for key in path):
            raise ToolInputError("path components must be object keys or integer indices",code="artifact_path_not_found",
                read_tool="read_tool_result",read_args={"artifact_id":identity,"path":[],"offset":0,"limit":128})
        if len(encode(path).encode("utf-8"))>8192:
            raise ToolInputError("path is too large; read the parent value and inspect locally")
        value=self[identity]
        resolved=[]
        for key in path:
            if isinstance(value,dict) and isinstance(key,str) and key in value:
                value=value[key]
            elif isinstance(value,list) and type(key) is int and 0<=key<len(value):
                value=value[key]
            else:
                details={"artifact_id":identity,"resolved_path":resolved,"value_type":type(value).__name__,
                    "read_tool":"read_tool_result",
                    "read_args":{"artifact_id":identity,"path":resolved,"offset":0,"limit":128},
                    "guidance":"Use the nearest valid path and available fields/indices; this local read did not execute an action."}
                if isinstance(value,dict):
                    fields=[k for k in value if len(encode(k).encode("utf-8"))<=256][:32]
                    details.update(available_fields=fields,field_count=len(value),fields_complete=len(fields)==len(value))
                elif isinstance(value,list):
                    details.update(length=len(value),valid_index_range=[0,len(value)-1] if value else [])
                raise ToolInputError("artifact path does not select a value",code="artifact_path_not_found",**details)
            resolved.append(key)
        return artifact_page(value,identity=identity,path=path,offset=offset,limit=limit,describe=self.describe)
