#!/usr/bin/env python3
"""verify.py OUT.jsonl [...]: static checks of normalised rows.
- every assistant python cell parses (or the original did not either -> counted separately)
- every call of a standard helper binds to its standard signature (positional count / keyword names)
- no definition of a standard name remains except inside the preamble
- Changed: lines never show a non-standard signature for a standard name"""
import ast, json, sys, collections, re, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stdlib import PREAMBLE, NAMES
API = {"position", "quaternion", "gripper", "max_speed_m_s", "max_angular_speed_rad_s", "max_duration_s", "position_tolerance_m",
       "orientation_tolerance_rad", "contact_mode", "load_budget_n"}
SIG = {"val": (1, 1, set()), "need": (1, 1, set()), "downq": (1, 1, {"yaw"}),
       "go": (2, 6, {"p", "q", "speed", "mode", "duration", "angular"} | API)}
st = collections.Counter(); ex = collections.defaultdict(list); seen = set()
for f in sys.argv[1:]:
    for l in open(f):
        r = json.loads(l)
        for m in r["messages"]:
            if m.get("role") == "tool":
                c = m["content"]; t = c if isinstance(c, str) else " ".join(x.get("text", "") for x in c if isinstance(x, dict))
                for mm in re.finditer(r"(?:Changed: |; )(go|val|need|downq)=function\(([^\n]*?)\)(?=; |\n|$)", t):
                    sig = mm.group(2)
                    good = {"go": "p, q, speed=0.05, mode='guarded', duration=20, angular=0.5, **kw", "val": "r", "need": "r", "downq": "yaw"}[mm.group(1)]
                    if sig != good:
                        st["changed_nonstd_sig"] += 1; ex["changed"].append(mm.group(0)[:120])
            if m.get("role") != "assistant":
                continue
            for tc in m.get("tool_calls") or []:
                a = tc.get("function", {}).get("arguments")
                code = a.get("input") if isinstance(a, dict) else None
                if not isinstance(code, str) or code in seen:
                    continue
                seen.add(code); st["cells"] += 1
                body = code[len(PREAMBLE):] if code.startswith(PREAMBLE) else code
                st["cells_with_preamble"] += code.startswith(PREAMBLE)
                try:
                    t = ast.parse(code)
                except SyntaxError:
                    st["syntax_error"] += 1; continue
                for n in ast.walk(ast.parse(body)):
                    if isinstance(n, ast.FunctionDef) and n.name in NAMES:
                        st["def_std_name_left"] += 1; ex["def"].append(n.name)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in SIG:
                        lo, hi, kws = SIG[n.func.id]
                        if any(isinstance(x, ast.Starred) for x in n.args) or any(k.arg is None for k in n.keywords):
                            st["call_star"] += 1; continue
                        npos = len(n.args); kn = [k.arg for k in n.keywords]
                        names = ["p", "q", "speed", "mode", "duration", "angular"] if n.func.id == "go" else ["r"] if n.func.id != "downq" else ["yaw"]
                        bound = set(names[:npos])
                        ok = npos <= (hi if n.func.id == "go" else 1) and not (bound & set(kn)) and set(kn) <= (kws | set(names))
                        ok = ok and (len(bound | set(kn) & set(names)) >= lo)
                        st[f"call_{n.func.id}"] += 1
                        if not ok:
                            st[f"call_{n.func.id}_BAD"] += 1; ex["call"].append(ast.get_source_segment(body, n)[:150])
print(json.dumps(dict(st), indent=1))
for k, v in ex.items():
    print(k, len(v), v[:8])
