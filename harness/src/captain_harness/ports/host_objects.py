"""Private durable snapshots of host-created public objects.

Only this process serializes objects. Policy can neither supply pickle bytes nor
choose filesystem paths. Digests are checked before restoring. This is never a
transport decoder or a model-visible data format.
"""
from pathlib import Path
import hashlib
import pickle
import sqlite3

class HostObjectStore:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(str(self.root/"objects.sqlite3"))
        self.db.execute("CREATE TABLE IF NOT EXISTS objects (id TEXT PRIMARY KEY,digest TEXT,body BLOB)")
        self.db.commit()

    def put(self,key,value):
        body=pickle.dumps(value,protocol=5);digest=hashlib.sha256(body).hexdigest()
        old=self.db.execute("SELECT digest FROM objects WHERE id=?",(key,)).fetchone()
        if old and old[0]!=digest:raise ValueError("immutable host snapshot conflict")
        self.db.execute("INSERT OR IGNORE INTO objects VALUES (?,?,?)",(key,digest,body))
        self.db.commit()

    def get(self,key):
        row=self.db.execute("SELECT digest,body FROM objects WHERE id=?",(key,)).fetchone()
        if row is None:raise KeyError(key)
        digest,body=row
        if hashlib.sha256(body).hexdigest()!=digest:raise ValueError("snapshot integrity failure")
        return pickle.loads(body)
