#!/usr/bin/env python3
"""Render a benchmark config template (configs/benchmarks/*.json): replace ${VAR} and ${VAR:-default} with environment
values, refuse unset variables without a default, and write the result atomically.

    python3 eval/transfer/render_config.py configs/benchmarks/robotwin2_franka_dual.json runs/x/_config/robotwin2.json
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def render(text: str) -> str:
    missing: list[str] = []

    def sub(match: re.Match) -> str:
        name, default = match.group(1), match.group(2)
        value = os.environ.get(name)
        if value:
            return value
        if default is not None:
            return default
        missing.append(name)
        return match.group(0)

    out = PATTERN.sub(sub, text)
    if missing:
        raise SystemExit(f"set these environment variables first: {', '.join(sorted(set(missing)))}")
    json.loads(out)  # must still be valid JSON
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    src, dst = Path(argv[1]), Path(argv[2])
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
    tmp.write_text(render(src.read_text(encoding="utf-8")), encoding="utf-8")
    os.replace(tmp, dst)
    print(dst)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
