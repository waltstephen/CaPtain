#!/usr/bin/env python3
"""Code normalisation of agent SFT rows (teacher demonstrations, earlier agent data, RL self-successes).

Per episode (all rows of an episode share one rewrite, keyed by tool_call id, so history and targets stay consistent):
  1. the standard helper library (stdlib.PREAMBLE: val / need / downq / go) is prepended to the episode's first python cell,
     and its names are added to that cell's "Changed:" output line;
  2. every module-level helper definition that is BEHAVIOURALLY equivalent to a standard helper (checked by executing it
     against a recording mock robot / numeric probes, per call site) is deleted, and its call sites are rewritten to the
     standard helper with explicit arguments that reproduce exactly the same robot calls (same move_to/preview arguments,
     same return/raise behaviour);
  3. a non-equivalent definition that uses a standard name (go/val/need/downq) is renamed to <name>_custom (with its calls)
     so a standard name never has two meanings;
  4. other helpers are left untouched.
Equivalence is decided per definition over ALL of its call sites in the episode (any unmappable call -> the def is treated
as non-equivalent). Prints are not compared (tool outputs keep the teacher's recorded prints).

usage: normalize.py --out-dir OUT_DIR --report R.json [--workers N] IN.jsonl [IN2.jsonl ...]
       (rows of one episode may span files; each input is written, rows in input order, to OUT_DIR/<parent>__<basename>).
       Rows whose target cell introspects a helper (inspect.getsource(go), go.__defaults__, ...) get
       meta.nosup_reason = "introspect"; drop them before training (their turn stays as context of later rows).
"""
from __future__ import annotations
import argparse, ast, collections, copy, inspect, json, math, os, re, sys, io, contextlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stdlib import PREAMBLE, CHANGED, NAMES, GO_DEFAULTS, API_DEFAULTS  # noqa: E402

# ----------------------------------------------------------------------------------------------- mock robot
class Perm(dict):
    """permissive nested result value: any missing key is another empty Perm (falsy)"""
    def __missing__(self, k):
        return Perm()


def mkres(tag, ok):
    return Perm({"success": ok, "executed": ok, "failure_category": "none" if ok else "probe_fail", "message": tag,
                 "value": Perm(tag=tag), "diagnostics": Perm(), "budget": Perm(), "receipt": tag, "_tag": tag})


def nval(v):
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return tuple(nval(x) for x in v)
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, (int, float)):
        return round(float(v), 9)
    if isinstance(v, dict):
        return tuple(sorted((k, nval(x)) for k, x in v.items()))
    return ("?", type(v).__name__)


def nargs(a):
    a = dict(a or {})
    for k, d in API_DEFAULTS.items():
        a.setdefault(k, d)
    return tuple(sorted((k, nval(v)) for k, v in a.items()))


class Mock:
    def __init__(self, pok=True, mok=True):
        self.log, self.pok, self.mok, self.n = [], pok, mok, 0

    def preview_motion(self, action=None, arguments=None, **kw):
        self.n += 1
        self.log.append(("preview_motion", action, nargs(arguments)))
        return mkres(f"preview{self.n}", self.pok)

    def move_to(self, **kw):
        self.n += 1
        self.log.append(("move_to", None, nargs(kw)))
        return mkres(f"move{self.n}", self.mok)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        def f(*a, **k):
            self.n += 1
            self.log.append((name, None, repr((a, k))[:200]))
            return mkres(f"{name}{self.n}", True)
        return f


class Tag:  # registry of probe values -> source text
    def __init__(self):
        self.vals = []  # (value, src)

    def vec(self, n, src):
        i = len(self.vals)
        v = [1000.0 + 100 * i + k + 0.25 for k in range(n)]
        self.vals.append((nval(v), src))
        return np.array(v)

    def sca(self, src):
        i = len(self.vals)
        v = 7000.0 + i + 0.125
        self.vals.append((nval(v), src))
        return v

    def lookup(self, v):
        nv = nval(v)
        for tv, src in self.vals:
            if tv == nv:
                return src
        return None


def stdlib_ns(robot):
    ns = {"robot": robot, "np": np, "math": math, "json": json, "print": lambda *a, **k: None}
    exec(PREAMBLE, ns)
    return ns


PRINT_ONLY = {"brief", "summary", "report", "show", "pp", "pr", "log", "note", "describe", "short", "summarize", "dump", "info",
              "status", "printr", "show_result", "summ", "s"}


CAPS = {"move_to", "preview_motion", "observe", "hold", "recover", "release", "mark_point", "segment", "measure_object_geometry",
        "inspect_contact_geometry", "inspect_release_geometry", "compare_object_observations", "check_motion", "robot_geometry",
        "gripper_state", "status", "history", "finish", "transform_pose", "select_detection", "point"}


class _NS(dict):
    """globals for probes: print-only helper names the teacher defined elsewhere resolve to a no-op (they never call robot)"""
    def __missing__(self, k):
        if k in PRINT_ONLY:
            return lambda *a, **kw: None
        if k in CAPS and "robot" in self:   # every capability is also a bare function in the session
            return getattr(dict.__getitem__(self, "robot"), k)
        raise KeyError(k)


def base_ns(robot):
    ns = {"robot": robot, "np": np, "math": math, "json": json, "print": lambda *a, **k: None, "image": lambda *a, **k: None}
    try:
        from scipy.spatial.transform import Rotation
        ns.update(R=Rotation, Rot=Rotation, Rotation=Rotation)
    except Exception:
        pass
    std = stdlib_ns(robot)
    for n in NAMES:  # a helper that calls val/need/... inside resolves to the standard ones (probe only)
        ns[n] = std[n]
    return _NS(ns)


def run_probe(fn, args, kwargs, pok, mok, robot):
    robot.log.clear(); robot.n = 0; robot.pok, robot.mok = pok, mok
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            r = fn(*args, **kwargs)
        ret = ("ret", r.get("_tag") if isinstance(r, dict) else ("val", nval(r) if not callable(r) else "fn"))
    except Exception as e:  # noqa: BLE001
        ret = ("raise", type(e).__name__ in ("RuntimeError", "AssertionError", "ValueError", "Exception"))
    return list(robot.log), ret


# ----------------------------------------------------------------------------------------------- classification
SIMPLE_KINDS = ("val", "need", "downq")


def make_fn(defsrc, ns_extra=None, ctx=""):
    robot = Mock()
    ns = base_ns(robot)
    if ns_extra:
        ns.update(ns_extra)
    if ctx:
        try:
            exec(ctx, ns)
        except Exception:
            pass
    exec(defsrc, ns)
    return robot, ns


def classify_simple(fd, defsrc, ctx=""):
    """1-parameter helpers: val / need (result unwrap) or downq (top-down quaternion from yaw)."""
    a = fd.args
    if len(a.args) != 1 or a.vararg or a.kwarg or a.kwonlyargs or fd.decorator_list:
        return None
    try:
        robot, ns = make_fn(defsrc, None, ctx)
        fn = ns[fd.name]
    except Exception:
        return None
    body = defsrc
    if "success" in body:
        okr, failr = mkres("ok", True), mkres("bad", False)
        robot.log.clear()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                v1 = fn(okr)
        except Exception:
            return None
        if v1 is not okr["value"] or robot.log:
            return None
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                v2 = fn(failr)
        except RuntimeError:
            return "need" if not robot.log else None
        except Exception:
            return None
        return "val" if v2 is None and not robot.log else None
    if any(k in body for k in ("cos", "from_euler", "from_matrix", "from_rotvec", "Rot", "R.")):
        std = stdlib_ns(Mock())["downq"]
        try:
            for y in (0.0, 0.3, 1.2, -2.0, 3.0, -0.7):
                with contextlib.redirect_stdout(io.StringIO()):
                    v = fn(y)
                v = np.asarray(v, dtype=float).reshape(-1)
                s = np.asarray(std(y), dtype=float)
                if v.shape != (4,) or min(np.abs(v - s).max(), np.abs(v + s).max()) > 1e-6:
                    return None
            if robot.log:
                return None
        except Exception:
            return None
        return "downq"
    return None


_NOLIT = object()


def literal_src(node, src_b):
    try:
        return ast.literal_eval(node)
    except Exception:
        return _NOLIT


class GoDef:
    """A module-level definition that may be equivalent to the standard go(); call sites are mapped individually."""

    def __init__(self, fd, defsrc, consts, ctx=""):
        self.fd, self.defsrc, self.consts, self.ctx = fd, defsrc, consts, ctx
        a = fd.args
        self.ok = (len(a.args) >= 1 and not fd.decorator_list and not a.posonlyargs and "move_to" in defsrc)

    def map_call(self, call, seg):
        """-> list of (kind, text) for the canonical go() call args, or None if not equivalent for this call site.
        seg(node) returns the (already rewritten) source text of a child node."""
        if not self.ok:
            self.why = "L4"; return None
        if any(isinstance(x, ast.Starred) for x in call.args) or any(k.arg is None for k in call.keywords):
            self.why = "L6"; return None
        tags = Tag()
        params = [p.arg for p in self.fd.args.args]
        # defaults (evaluated at def time) -> names in default expressions become probe values
        free = {}
        default_tags = {}
        a = self.fd.args
        defaults = dict(zip(params[len(params) - len(a.defaults):], a.defaults)) if a.defaults else {}
        for kw, dn in zip(a.kwonlyargs, a.kw_defaults):
            if dn is not None:
                defaults[kw] = dn
        for pname, dn in defaults.items():
            if literal_src(dn, None) is _NOLIT:
                if not isinstance(dn, ast.Name):
                    return None  # complex default expression: give up
                idx = params.index(pname) if pname in params else 9
                is_vec = idx == 1 or "q" in pname.lower() or "quat" in pname.lower()
                free[dn.id] = tags.vec(4, dn.id) if is_vec else tags.sca(dn.id)
                default_tags[nval(free[dn.id])] = dn.id
        # free globals read inside the body at CALL time (e.g. `if q is None: q = qdown`) -> probe values named by the global
        locs = set(params) | {x.arg for x in a.kwonlyargs} | ({a.vararg.arg} if a.vararg else set()) | ({a.kwarg.arg} if a.kwarg else set())
        for n in ast.walk(self.fd):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                locs.add(n.id)
        known = set(dir(__builtins__) if not isinstance(__builtins__, dict) else __builtins__) | set(base_ns(Mock()).keys()) | PRINT_ONLY
        for n in ast.walk(self.fd):
            if (isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in locs and n.id not in known
                    and n.id not in free and n.id != self.fd.name):
                free[n.id] = tags.vec(4, n.id) if "q" in n.id.lower() else tags.sca(n.id)
        try:
            robot, ns = make_fn(self.defsrc, free, self.ctx)
            fn = ns[self.fd.name]
            sig = inspect.signature(fn)
        except Exception:
            self.why = "L40"; return None
        # call-site values
        lits = []
        args, kwargs = [], {}
        for i, x in enumerate(call.args):
            v = literal_src(x, None)
            if v is not _NOLIT and not isinstance(v, (dict, set)):
                args.append(v); lits.append((nval(v), seg(x)))
            else:
                pname = params[i] if i < len(params) else f"_{i}"
                args.append(tags.vec(3, seg(x)) if i == 0 else tags.vec(4, seg(x)) if i == 1 else tags.sca(seg(x)))
        for k in call.keywords:
            v = literal_src(k.value, None)
            if v is not _NOLIT and not isinstance(v, (dict, set)):
                kwargs[k.arg] = v; lits.append((nval(v), seg(k.value)))
            else:
                idx = params.index(k.arg) if k.arg in params else 9
                kwargs[k.arg] = (tags.vec(3, seg(k.value)) if idx == 0 else
                                 tags.vec(4, seg(k.value)) if (idx == 1 or "quat" in k.arg) else tags.sca(seg(k.value)))
        try:
            sig.bind(*args, **kwargs)
        except TypeError:
            self.why = "L62"; return None
        log, ret = run_probe(fn, args, kwargs, True, True, robot)
        if [e[0] for e in log] != ["preview_motion", "move_to"] or log[0][1] != "move_to" or log[0][2] != log[1][2]:
            self.why = "L65"; return None
        A = dict(log[1][2])

        def src_of(k, vec):
            v = A.get(k)
            s = tags.lookup(v)
            if s is not None:
                return s
            for lv, ls in lits:
                if lv == nval(v):
                    return ls
            if vec:
                self.why = "L77"; return None
            if isinstance(v, (int, float)) and (v in self.consts or v == API_DEFAULTS.get(k)):
                return repr(int(v)) if float(v).is_integer() else repr(v)
            if isinstance(v, str) and v in self.consts:
                return repr(v)
            if v is None and API_DEFAULTS.get(k, 0) is None:
                return "None"
            self.why = "L84"; return None
        p_src, q_src = src_of("position", True), src_of("quaternion", True)
        if p_src is None or q_src is None:
            self.why = "L87"; return None
        out = [("pos", p_src), ("pos", q_src)]
        canon = {"max_speed_m_s": ("speed", 0.05), "contact_mode": ("mode", "guarded"), "max_duration_s": ("duration", 20.0),
                 "max_angular_speed_rad_s": ("angular", 0.5)}
        korder = ["max_speed_m_s", "contact_mode", "max_duration_s", "max_angular_speed_rad_s"]
        for k, v in sorted(A.items(), key=lambda kv: (korder.index(kv[0]) if kv[0] in korder else 9, kv[0])):
            if k in ("position", "quaternion", "gripper") and k != "gripper":
                continue
            if k == "gripper":
                if v is not None:
                    self.why = "L97"; return None
                continue
            name, dflt = canon.get(k, (k, API_DEFAULTS.get(k)))
            if nval(v) == nval(dflt) and tags.lookup(v) is None:
                continue
            s = src_of(k, False)
            if s is None:
                self.why = "L104"; return None
            out.append(("kw", f"{name}={s}"))
        # verify: canonical go with the same probe values reproduces the robot calls and return/raise behaviour
        def dec(v):
            return list(v) if isinstance(v, tuple) else v
        cargs = [dec(A["position"]), dec(A["quaternion"])]
        ckw = {}
        for kname, (name, _) in list(canon.items()) + [(k, (k, None)) for k in A if k not in canon]:
            pass
        for kind, t in out[2:]:
            name = t.split("=", 1)[0]
            k = {v[0]: kk for kk, v in canon.items()}.get(name, name)
            ckw[name] = dec(A[k])
        self.used_defaults = {default_tags[nval(A[k])] for k in A if nval(A[k]) in default_tags}
        crobot = Mock()
        cfn = stdlib_ns(crobot)["go"]
        for pok, mok in ((True, True), (False, True), (True, False)):
            l1, r1 = run_probe(fn, args, kwargs, pok, mok, robot)
            l2, r2 = run_probe(cfn, cargs, ckw, pok, mok, crobot)
            if l1 != l2 or r1 != r2:
                self.why = "L124"; return None
        return out


def consts_of(fd):
    s = set()
    for n in ast.walk(fd):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float, str)) and not isinstance(n.value, bool):
            s.add(n.value if isinstance(n.value, str) else float(n.value))
    return s


# ----------------------------------------------------------------------------------------------- per-episode rewrite
class Cell:
    def __init__(self, cid, code):
        self.cid, self.code = cid, code
        self.b = code.encode("utf-8")
        try:
            self.tree = ast.parse(code)
        except SyntaxError:
            self.tree = None
        lines = self.b.split(b"\n")
        self.loff = [0]
        for ln in lines:
            self.loff.append(self.loff[-1] + len(ln) + 1)

    def off(self, lineno, col):
        return self.loff[lineno - 1] + col

    def span(self, n):
        return self.off(n.lineno, n.col_offset), self.off(n.end_lineno, n.end_col_offset)


def stmt_line_span(cell, st):
    """byte span of whole lines of a module-level statement, or None if it shares lines with other code."""
    s, e = cell.span(st)
    ls, le = cell.loff[st.lineno - 1], cell.loff[st.end_lineno] if st.end_lineno < len(cell.loff) else len(cell.b)
    head = cell.b[ls:s]
    tail = cell.b[e:min(le, len(cell.b))]
    if head.strip() or (tail.strip() and not tail.strip().startswith(b"#")):
        return None
    return ls, min(le, len(cell.b))


def episode_rewrite(cells, stats):
    """cells: ordered list of Cell. Returns {cid: (new_code, changed_edits)} where changed_edits describe Changed-line edits."""
    # candidate defs: (cell index, stmt index) -> classification
    cand = {}
    latest = {}   # name -> source of the latest module-level def (context for probing helpers that call it)

    def ctx_for(st):
        refs, out, todo = set(), [], [st]
        while todo:
            x = todo.pop()
            for n in ast.walk(x):
                if isinstance(n, ast.Name) and n.id in latest and n.id not in refs and n.id != st.name and n.id not in NAMES:
                    refs.add(n.id); out.append(latest[n.id][0]); todo.append(latest[n.id][1])
        return "\n".join(reversed(out))
    for ci, c in enumerate(cells):
        if c.tree is None:
            continue
        for si, st in enumerate(c.tree.body):
            if isinstance(st, ast.FunctionDef):
                defsrc = c.b[slice(*c.span(st))].decode("utf-8")
                ctx = ctx_for(st)
                kind = classify_simple(st, defsrc, ctx)
                if kind:
                    cand[(ci, si)] = ("simple", kind, st, None)
                elif "move_to" in defsrc and len(st.args.args) >= 1:
                    cand[(ci, si)] = ("go", "go", st, GoDef(st, defsrc, consts_of(st), ctx))
                latest[st.name] = (defsrc, st)
    stores = []
    for ci, c in enumerate(cells):
        if c.tree is None:
            continue
        for si, st in enumerate(c.tree.body):
            names = {n.id for n in ast.walk(st) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
            for n in ast.walk(st):
                if isinstance(n, (ast.Import, ast.ImportFrom)):
                    names |= {(al.asname or al.name).split(".")[0] for al in n.names}
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                    names.add(n.name)
            stores.append(((ci, si), names))
    for key, v in cand.items():
        if v[0] == "go":
            v[3].later_stores = set().union(*[nm for k2, nm in stores if k2 > key]) if stores else set()
    bad = set()   # defs found non-equivalent at some call site
    for _ in range(6):
        res = _pass(cells, cand, bad, stats)
        if res[0] == "ok":
            return res[1]
        bad |= res[1]
    stats["episode_fallback_unstable"] += 1
    return None


def _pass(cells, cand, bad, stats):
    binding = {}      # name -> (kind, target_name, key)   kind: 'canon'|'rename'
    out = {}
    newbad = set()
    for ci, c in enumerate(cells):
        edits = []    # (s, e, builder)
        chg = []      # Changed-line edits: ('del', name) / ('ren', old, new)
        if c.tree is None:
            # unparsable cell: rename-only of plain Name tokens is unsafe; leave as is
            out[c.cid] = (c.code, chg)
            continue
        defs_here = []
        for si, st in enumerate(c.tree.body):
            key = (ci, si)
            # rewrite references inside this statement with the CURRENT binding
            _collect(c, st, binding, cand, edits, newbad, stats)
            if isinstance(st, ast.FunctionDef):
                nm = st.name
                if key in cand and key not in bad:
                    kind, target = cand[key][0], cand[key][1]
                    sp = stmt_line_span(c, st)
                    if sp is None:
                        newbad.add(key)
                    else:
                        binding[nm] = ("canon", target, key)
                        edits.append((sp[0], sp[1], lambda: b""))
                        chg.append(("del", nm))
                        defs_here.append(nm)
                        continue
                if nm in NAMES:
                    binding[nm] = ("rename", nm + "_custom", key)
                    s = c.off(st.lineno, st.col_offset)
                    # rename the def name: find "def <nm>" after s
                    m = re.compile(rb"def\s+" + nm.encode() + rb"\b").search(c.b, s)
                    if m:
                        ns_, ne_ = m.end() - len(nm), m.end()
                        edits.append((ns_, ne_, (lambda nn=nm: (nn + "_custom").encode())))
                    chg.append(("ren", nm, nm + "_custom"))
                else:
                    binding.pop(nm, None)
            else:
                # plain assignment / import that rebinds a tracked name -> binding ends
                for n in ast.walk(st):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id in binding:
                        binding.pop(n.id, None)
                    elif isinstance(n, (ast.Import, ast.ImportFrom)):
                        for al in n.names:
                            binding.pop((al.asname or al.name).split(".")[0], None)
        if newbad:
            continue
        if getattr(c, "nosup", False):
            chg.append(("nosup", "", ""))
        out[c.cid] = (_render(c, edits), chg)
    if newbad - bad:
        return ("bad", newbad - bad)
    if newbad:  # already-bad keys reported again: should not happen
        return ("bad", newbad)
    return ("ok", out)


def _collect(c, st, binding, cand, edits, newbad, stats):
    """call/name rewrites inside statement st (not entering nested defs of tracked names is not needed: late binding is
    approximated by the binding at definition time)."""
    seen_calls = set()
    for n in ast.walk(st):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in binding:
            seen_calls.add(id(n.func))
            kind, target, key = binding[n.func.id]
            if kind == "rename":
                s, e = c.span(n.func)
                edits.append((s, e, (lambda t=target: t.encode())))
                continue
            ck = cand[key]
            if ck[0] == "simple":
                s, e = c.span(n.func)
                edits.append((s, e, (lambda t=target: t.encode())))
                continue
            gd = ck[3]
            seg = lambda node: _render_span(c, node, edits).decode("utf-8")  # noqa: E731
            # mapping needs the rewritten text of nested nodes -> computed lazily in the builder; probe now with raw text
            raw = gd.map_call(n, lambda node: c.b[slice(*c.span(node))].decode("utf-8"))
            if raw is None:
                newbad.add(key)
                stats["go_call_unmapped"] += 1
                if os.environ.get("NORM_DEBUG"):
                    print("UNMAPPED", getattr(gd, "why", "?"), repr(c.b[slice(*c.span(n))].decode()[:160]), "| DEF", repr(gd.defsrc[:300]), file=sys.stderr)
                continue
            if gd.used_defaults & getattr(gd, "later_stores", set()):
                newbad.add(key)
                stats["go_default_rebound"] += 1
                continue
            s, e = c.span(n)
            def build(n=n, gd=gd):
                m = gd.map_call(n, lambda node: _render_span(c, node, edits).decode("utf-8"))
                if m is None:  # nested rewrite changed a literal-looking arg: fall back to raw mapping texts
                    m = gd.map_call(n, lambda node: c.b[slice(*c.span(node))].decode("utf-8"))
                parts = [t for _, t in m]
                return ("go(" + ", ".join(parts) + ")").encode("utf-8")
            edits.append((s, e, build))
            stats["go_call_mapped"] += 1
    stsrc = c.b[slice(*c.span(st))].decode("utf-8", "replace")
    introspect = any(k in stsrc for k in ("getsource(", "__defaults__", "__code__", "signature(", "__doc__", "__name__"))
    for n in ast.walk(st):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in binding and id(n) not in seen_calls:
            kind, target, key = binding[n.id]
            if kind == "canon" and cand[key][0] == "go":
                if introspect:   # the teacher re-reads its own helper (after compaction): point it at go, don't supervise
                    c.nosup = True; stats["introspect_ref"] += 1
                else:
                    newbad.add(key); stats["go_nameref"] += 1; continue
            if target != n.id:
                s, e = c.span(n)
                edits.append((s, e, (lambda t=target: t.encode())))


def _render_span(c, node, edits):
    s, e = c.span(node)
    return _render_range(c, s, e, [x for x in edits if s <= x[0] and x[1] <= e and (x[0], x[1]) != (s, e)])


def _render_range(c, s, e, edits):
    # top-level edits only (not contained in another edit of the list)
    ed = sorted(edits, key=lambda x: (x[0], -(x[1] - x[0])))
    top, end = [], -1
    for x in ed:
        if x[0] >= end:
            top.append(x); end = x[1]
    outb, pos = [], s
    for (es, ee, b) in top:
        outb.append(c.b[pos:es]); outb.append(b()); pos = ee
    outb.append(c.b[pos:e])
    return b"".join(outb)


def _render(c, edits):
    if not edits:
        return c.code
    txt = _render_range(c, 0, len(c.b), edits).decode("utf-8")
    if not txt.strip():
        txt = "# (helpers above are the standard ones: val / need / downq / go)\n"
    return txt


# ----------------------------------------------------------------------------------------------- rows I/O
def call_code(tc):
    a = tc.get("function", {}).get("arguments")
    if isinstance(a, dict):
        return a.get("input"), "dict"
    if isinstance(a, str):
        try:
            d = json.loads(a)
            if isinstance(d, dict) and isinstance(d.get("input"), str):
                return d["input"], "json"
        except Exception:
            pass
        return a, "raw"
    return None, None


def set_code(tc, code, mode):
    if mode == "dict":
        tc["function"]["arguments"]["input"] = code
    elif mode == "json":
        d = json.loads(tc["function"]["arguments"]); d["input"] = code
        tc["function"]["arguments"] = json.dumps(d, ensure_ascii=False)
    else:
        tc["function"]["arguments"] = code


def ep_key(r):
    m = r.get("meta") or {}
    return m.get("episode") or r["id"].split("#")[0]


CHG_RE = re.compile(r"(^|\n)Changed: ([^\n]*)")


def edit_changed(text, chg, first):
    def fix(line):
        items = line.split("; ") if line else []
        out = []
        for it in items:
            nm = it.split("=", 1)[0]
            drop = any(k == "del" and nm == a for k, a, *_ in chg) and it[len(nm) + 1:].startswith("function(")
            if drop:
                continue
            for k, a, *rest in chg:
                if k == "ren" and nm == a and it[len(nm) + 1:].startswith("function("):
                    it = rest[0] + it[len(nm):]
            out.append(it)
        if first:
            out = [CHANGED] + out
        return "; ".join(out)
    m = CHG_RE.search(text)
    if m:
        new = fix(m.group(2))
        return text[:m.start(2)] + new + text[m.end(2):]
    if first:
        return text.rstrip("\n") + "\nChanged: " + CHANGED
    return text


def tool_text_apply(content, fn):
    if isinstance(content, str):
        return fn(content)
    if isinstance(content, list):
        # apply to the LAST text part (the Changed line closes the cell output)
        idx = [i for i, x in enumerate(content) if isinstance(x, dict) and x.get("type") == "text"]
        if idx:
            content = copy.copy(content)
            hit = [i for i in idx if CHG_RE.search(content[i]["text"])]
            for i in (hit[:1] or idx[-1:]):
                x = dict(content[i]); x["text"] = fn(x["text"]); content[i] = x
        return content
    return content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    # pass 1: cells per episode
    cells = collections.defaultdict(dict)   # ep -> cid -> (order, code)
    for f in args.inputs:
        for l in open(f):
            r = json.loads(l)
            ep = ep_key(r)
            ri = (r.get("meta") or {}).get("request_index") or 0
            msgs = r["messages"]
            # order: position of the assistant message among the row's assistant turns is history order; use the request
            # index of the row for the target turn and the min seen for history turns
            asst = [m for m in msgs if m.get("role") == "assistant"]
            for ai, m in enumerate(asst):
                for ti, tc in enumerate(m.get("tool_calls") or []):
                    if tc.get("function", {}).get("name") != "python":
                        continue
                    code, mode = call_code(tc)
                    if code is None:
                        continue
                    cid = tc.get("id")
                    est = (ri - (len(asst) - 1 - ai), ti)
                    prev = cells[ep].get(cid)
                    if prev is not None and prev[1] != code:
                        cells[ep][cid] = (prev[0], prev[1], True)   # inconsistent code for one id: never rewrite
                        continue
                    if prev is None or est < prev[0]:
                        cells[ep][cid] = (est, code, prev[2] if prev else False)
    stats = collections.Counter()
    plan = {}   # ep -> {cid: (new_code, chg, is_first)}
    from multiprocessing import Pool
    jobs = [(ep, d) for ep, d in cells.items()]
    with Pool(args.workers) as pool:
        for ep, res, st in pool.imap_unordered(_ep_job, jobs, chunksize=4):
            stats.update(st)
            plan[ep] = res
    # pass 2: apply
    os.makedirs(args.out_dir, exist_ok=True)
    for f in args.inputs:
        outp = os.path.join(args.out_dir, os.path.basename(os.path.dirname(f)) + "__" + os.path.basename(f))
        with open(f) as fi, open(outp, "w") as fo:
            for l in fi:
                r = json.loads(l)
                p = plan.get(ep_key(r))
                if p:
                    r = apply_row(r, p)
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("wrote", outp, file=sys.stderr)
    json.dump(dict(stats), open(args.report, "w"), indent=1)
    print(json.dumps(dict(stats), indent=1))


def _ep_job(job):
    ep, d = job
    st = collections.Counter()
    st["episodes"] += 1
    if any(v[2] for v in d.values()):
        st["episodes_inconsistent_ids"] += 1
    order = sorted(d.items(), key=lambda kv: kv[1][0])
    cl = [Cell(cid, v[1]) for cid, v in order]
    res = None
    try:
        res = episode_rewrite(cl, st)
    except Exception as e:  # noqa: BLE001
        st["episode_error"] += 1
        st["err:" + type(e).__name__] += 1
    first = order[0][0] if order else None
    plan = {}
    if res is None:
        st["episodes_preamble_only"] += 1
        # still prepend the preamble, and rename any def of a standard name (rename-only pass)
        res = {}
        try:
            res = _pass(cl, {}, set(), collections.Counter())[1]
            if not isinstance(res, dict):
                res = {}
        except Exception:
            res = {}
    for cid, v in d.items():
        if v[2]:
            continue
        new, chg = res.get(cid, (v[1], []))
        if cid == first:
            new = PREAMBLE + new
        if new != v[1] or chg or cid == first:
            plan[cid] = (new, chg, cid == first)
        st["defs_deleted"] += sum(1 for x in chg if x[0] == "del")
        st["defs_renamed"] += sum(1 for x in chg if x[0] == "ren")
        for x in chg:
            st[f"{x[0]}:{x[1]}"] += 1
    return ep, plan, st


def apply_row(r, p):
    r = copy.deepcopy(r)
    asst = [m for m in r["messages"] if m.get("role") == "assistant"]
    if asst and any(any(x[0] == "nosup" for x in p.get(tc.get("id"), ("", [], False))[1]) for tc in asst[-1].get("tool_calls") or []):
        r.setdefault("meta", {})["nosup_reason"] = "introspect"
    pending = {}
    for m in r["messages"]:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                cid = tc.get("id")
                if cid in p:
                    code, mode = call_code(tc)
                    set_code(tc, p[cid][0], mode)
                    pending[cid] = p[cid]
        elif m.get("role") == "tool" and m.get("tool_call_id") in pending:
            new, chg, first = pending[m["tool_call_id"]]
            if chg or first:
                m["content"] = tool_text_apply(m["content"], lambda t: edit_changed(t, chg, first))
    return r


if __name__ == "__main__":
    main()
