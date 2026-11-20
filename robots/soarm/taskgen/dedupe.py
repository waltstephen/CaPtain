"""Dedupe: goal-structure signature + instruction similarity against every existing family and earlier acceptances.

Signature = sorted goal atoms with roles replaced by their kind and count shape, plus layout constraint types and the
set of attribute constraints (colour same/not/distinct, size min/max). Two families with equal signatures are the same
task up to renaming. Instruction similarity = cosine of TF-IDF vectors over word uni+bigrams of all templates with slots
masked (no embedding endpoint on the gateway; numpy only) -- near-identical wording with a different signature is kept,
equal signature is always a duplicate, and so is cosine >= SIM_DUP with the same predicate set.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter

SIM_DUP = 0.80


def _kind(o) -> str:
    k = o["kind"] if isinstance(o["kind"], str) else "|".join(sorted(o["kind"]))
    return "container" if k in ("bowl", "box", "bin", "bowl|box", "box|bowl") else k


def _shape(o) -> str:
    c = o.get("count", 1)
    return "1" if c == 1 else "n" if isinstance(c, list) and c[1] > 1 else str(c)


def signature(spec: dict) -> str:
    roles = {o["role"]: o for o in spec["objects"]}
    goal_roles = set()

    def arg(a):
        if isinstance(a, str) and a.startswith("@"):
            r = re.match(r"@([a-z0-9_]+)", a).group(1)
            goal_roles.add(r)
            o = roles.get(r, {"kind": "?"})
            return f"{_kind(o)}{'*' if a.endswith('*') else ''}{_shape(o)}"
        return "$" if a == "$" else "var" if isinstance(a, str) and a.startswith("%") else str(a)

    atoms = []
    for g in spec["goal"]:
        if "count" in g:
            q = {k: g[k] for k in ("eq", "min", "max") if k in g}
            atoms.append(f"count[{g['count']['pred']}({','.join(arg(a) for a in g['count']['args'])})"
                         f"/{arg('@' + g['over'])}{q}]")
        else:
            fe = f"each:{arg('@' + g['for_each'])}:" if g.get("for_each") else ""
            params = ",".join(f"{k}={v}" for k, v in sorted((g.get("params") or {}).items()))
            atoms.append(f"{fe}{g['pred']}({','.join(arg(a) for a in g['args'])}){params}")
    cons = []
    for o in spec["objects"]:
        if o["role"] not in goal_roles:
            continue
        for attr, c in sorted((o.get("where") or {}).items()):
            if isinstance(c, dict):
                cons.append(f"{attr}:{','.join(sorted(k for k in c if k != 'distinct') + (['distinct'] if c.get('distinct') else []))}")
            else:
                cons.append(f"{attr}:fixed")
    lay = sorted(c["type"] for c in spec.get("layout") or [] if c["type"] != "region")
    return json.dumps([sorted(atoms), sorted(cons), lay])


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z]+", re.sub(r"\{[^}]*\}", " slot ", text.lower()))
    return words + [a + "_" + b for a, b in zip(words, words[1:])]


class Index:
    def __init__(self) -> None:
        self.docs: list[tuple[str, Counter]] = []
        self.sigs: dict[str, str] = {}
        self.preds: dict[str, frozenset] = {}

    def add(self, spec: dict) -> None:
        self.docs.append((spec["id"], Counter(t for s in spec["instructions"] for t in _tokens(s))))
        self.sigs.setdefault(signature(spec), spec["id"])
        self.preds[spec["id"]] = _preds(spec)

    def _vec(self, c: Counter, df: Counter, n: int) -> dict[str, float]:
        v = {t: (1 + math.log(k)) * math.log((n + 1) / (df[t] + 1)) for t, k in c.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def nearest(self, spec: dict) -> tuple[str | None, float]:
        c = Counter(t for s in spec["instructions"] for t in _tokens(s))
        docs = self.docs + [("_q", c)]
        df = Counter(t for _, d in docs for t in d)
        q = self._vec(c, df, len(docs))
        best, best_s = None, 0.0
        for sid, d in self.docs:
            v = self._vec(d, df, len(docs))
            s = sum(q[t] * v.get(t, 0.0) for t in q)
            if s > best_s:
                best, best_s = sid, s
        return best, best_s

    def check(self, spec: dict) -> tuple[bool, str]:
        sig = signature(spec)
        if sig in self.sigs:
            return False, f"same goal signature as {self.sigs[sig]}"
        near, sim = self.nearest(spec)
        if sim >= SIM_DUP and _preds(spec) == self.preds.get(near):
            return False, f"instructions cos {sim:.2f} to {near} with the same predicates"
        return True, f"nearest {near} cos {sim:.2f}"


def _preds(spec: dict) -> frozenset:
    return frozenset(g.get("pred") or "count:" + g["count"]["pred"] for g in spec["goal"])
