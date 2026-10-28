"""Codex's freeform apply_patch format (*** Begin Patch / Add / Update / Delete / End Patch)."""
from __future__ import annotations

from pathlib import Path


class PatchError(ValueError):
    pass


def _resolve(workspace: Path, name: str) -> Path:
    target = (workspace / name).resolve() if not Path(name).is_absolute() else Path(name).resolve()
    if workspace not in target.parents and target != workspace:
        raise PatchError(f"path {name!r} escapes the workspace")
    return target


def _apply_update(original: str, hunk_lines: list[str]) -> str:
    """Apply @@-separated context/change blocks to ``original`` (whole-line matching, in order)."""
    src = original.split("\n")
    if original.endswith("\n"):
        src = src[:-1]
    out: list[str] = []
    cursor = 0
    blocks: list[list[str]] = [[]]
    for line in hunk_lines:
        if line.startswith("@@"):
            blocks.append([])
        else:
            blocks[-1].append(line)
    for block in blocks:
        if not block:
            continue
        old = [l[1:] for l in block if l.startswith((" ", "-"))]
        new = [l[1:] for l in block if l.startswith((" ", "+"))]
        if old:
            index = _find(src, old, cursor)
            if index < 0:
                raise PatchError(f"context not found: {old[0]!r}")
            out.extend(src[cursor:index])
            out.extend(new)
            cursor = index + len(old)
        else:
            out.extend(new)
    out.extend(src[cursor:])
    text = "\n".join(out)
    return text + ("\n" if original.endswith("\n") or not original else "")


def _find(lines: list[str], pattern: list[str], start: int) -> int:
    for i in range(start, len(lines) - len(pattern) + 1):
        if lines[i:i + len(pattern)] == pattern:
            return i
    stripped = [p.rstrip() for p in pattern]
    for i in range(start, len(lines) - len(pattern) + 1):
        if [l.rstrip() for l in lines[i:i + len(pattern)]] == stripped:
            return i
    return -1


def apply_patch(patch: str, workspace: Path) -> str:
    """Apply the patch inside ``workspace``; returns Codex's summary text or a failure message."""
    workspace = Path(workspace).resolve()
    lines = patch.splitlines()
    try:
        if not lines or lines[0].strip() != "*** Begin Patch":
            raise PatchError("patch must start with *** Begin Patch")
        if "*** End Patch" not in {l.strip() for l in lines}:
            raise PatchError("patch must end with *** End Patch")
        i = 1
        summary: list[str] = []
        pending: list[tuple[str, Path, str | None]] = []
        while i < len(lines) and lines[i].strip() != "*** End Patch":
            line = lines[i]
            if line.startswith("*** Add File: "):
                path = _resolve(workspace, line[len("*** Add File: "):].strip())
                i += 1
                body: list[str] = []
                while i < len(lines) and not lines[i].startswith("*** "):
                    if not lines[i].startswith("+"):
                        raise PatchError(f"add hunk lines must start with '+': {lines[i]!r}")
                    body.append(lines[i][1:])
                    i += 1
                pending.append(("A", path, "\n".join(body) + "\n"))
            elif line.startswith("*** Delete File: "):
                path = _resolve(workspace, line[len("*** Delete File: "):].strip())
                if not path.is_file():
                    raise PatchError(f"cannot delete missing file {path.relative_to(workspace)}")
                pending.append(("D", path, None))
                i += 1
            elif line.startswith("*** Update File: "):
                path = _resolve(workspace, line[len("*** Update File: "):].strip())
                if not path.is_file():
                    raise PatchError(f"cannot update missing file {path.relative_to(workspace)}")
                i += 1
                move_to: Path | None = None
                if i < len(lines) and lines[i].startswith("*** Move to: "):
                    move_to = _resolve(workspace, lines[i][len("*** Move to: "):].strip())
                    i += 1
                hunk: list[str] = []
                while i < len(lines) and not lines[i].startswith("*** "):
                    hunk.append(lines[i])
                    i += 1
                if i < len(lines) and lines[i].strip() == "*** End of File":
                    i += 1
                updated = _apply_update(path.read_text(encoding="utf-8"), hunk)
                if move_to is not None:
                    pending.append(("D", path, None))
                    pending.append(("A", move_to, updated))
                else:
                    pending.append(("M", path, updated))
            else:
                raise PatchError(f"unexpected line: {line!r}")
        for kind, path, content in pending:
            if kind == "D":
                path.unlink()
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content or "", encoding="utf-8")
            summary.append(f"{kind} {path.relative_to(workspace)}")
    except (PatchError, OSError, UnicodeDecodeError) as exc:
        return f"apply_patch verification failed: {exc}"
    return "Success. Updated the following files:\n" + "\n".join(summary)


__all__ = ["PatchError", "apply_patch"]
