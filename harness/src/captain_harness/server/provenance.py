"""Source-tree provenance for episode results (moved from experiments.provenance in V1.1)."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Iterable

_SOURCE_ROOTS = ("src", "scripts", "configs", "tests")
_IMPLEMENTATION_ROOTS = ("src", "scripts")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git(project_root: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(("git", *arguments), cwd=project_root, capture_output=True, check=check)


def _files(project_root: Path, roots: Iterable[str]) -> Iterable[Path]:
    for root_name in roots:
        root = project_root / root_name
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}:
                yield path
    for name in ("pyproject.toml", "requirements.txt"):
        path = project_root / name
        if path.is_file():
            yield path


def _tree_digest(project_root: Path, paths: Iterable[Path]) -> tuple[str, int]:
    digest = hashlib.sha256()
    file_count = 0
    for path in paths:
        relative = path.relative_to(project_root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
        file_count += 1
    return digest.hexdigest(), file_count


def capture_source_provenance(project_root: Path, *, output_dir: Path | None = None) -> dict[str, object]:
    """Git commit, dirty state and content digests of the implementation tree.

    Works for a non-git checkout too (git fields become None). When ``output_dir`` is given the
    snapshot is written to ``source_snapshot.json`` and the dirty diff to ``source.patch``.
    """
    project_root = Path(project_root).resolve()
    commit = None
    dirty = None
    patch = b""
    try:
        commit = _git(project_root, "rev-parse", "HEAD").stdout.decode().strip()
        status = _git(project_root, "status", "--porcelain=v1").stdout.decode()
        dirty = bool(status.strip())
        patch = _git(project_root, "diff", "--binary", "HEAD", "--", *_SOURCE_ROOTS, "pyproject.toml").stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    source_tree_sha, source_file_count = _tree_digest(project_root, _files(project_root, _SOURCE_ROOTS))
    implementation_tree_sha, implementation_file_count = _tree_digest(project_root, _files(project_root, _IMPLEMENTATION_ROOTS))
    snapshot: dict[str, object] = {
        "schema_version": 2,
        "git_commit": commit,
        "git_dirty": dirty,
        "dirty_diff_sha256": _sha256_bytes(patch) if patch else None,
        "source_tree_sha256": source_tree_sha,
        "source_file_count": source_file_count,
        "implementation_tree_sha256": implementation_tree_sha,
        "implementation_file_count": implementation_file_count,
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if patch:
            (output_dir / "source.patch").write_bytes(patch)
            snapshot["source_patch_path"] = "source.patch"
        (output_dir / "source_snapshot.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return snapshot


def sha256_json(value: object) -> str:
    return _sha256_bytes(json.dumps(value, separators=(",", ":"), sort_keys=True).encode())


__all__ = ["capture_source_provenance", "sha256_json"]
