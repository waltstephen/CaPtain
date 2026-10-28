"""bubblewrap sandbox with Codex's workspace-write semantics.

Root read-only, workspace and /tmp writable, network shared, fresh pid namespace; the
episode's own output directory is hidden behind a tmpfs so command output, logs and audits
written by the host stay invisible to the agent.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class Sandbox:
    workspace: Path
    hidden_dirs: tuple[Path, ...] = ()
    writable_dirs: tuple[Path, ...] = (Path("/tmp"),)
    bwrap: str | None = None
    share_net: bool = True

    @classmethod
    def detect(cls, workspace: Path, *, hidden_dirs: Sequence[Path] = (), bwrap: str | None = None,
               require: bool = False) -> "Sandbox":
        binary = bwrap or shutil.which("bwrap")
        if require and not binary:
            raise RuntimeError("bwrap not found; refusing to run without a sandbox")
        return cls(workspace=Path(workspace).resolve(), hidden_dirs=tuple(Path(h).resolve() for h in hidden_dirs), bwrap=binary)

    @property
    def enabled(self) -> bool:
        return self.bwrap is not None

    def wrap(self, argv: Sequence[str], *, cwd: Path | None = None) -> list[str]:
        if not self.bwrap:
            return list(argv)
        cwd = Path(cwd or self.workspace)
        command = [self.bwrap, "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc"]
        for directory in self.writable_dirs:
            if directory.exists():
                command += ["--bind", str(directory), str(directory)]
        for hidden in self.hidden_dirs:
            if hidden.exists() and hidden != self.workspace and self.workspace not in hidden.parents:
                command += ["--tmpfs", str(hidden)]
        command += ["--bind", str(self.workspace), str(self.workspace)]
        command += ["--unshare-all"]
        if self.share_net:
            command += ["--share-net"]
        command += ["--die-with-parent", "--chdir", str(cwd), "--", *argv]
        return command


__all__ = ["Sandbox"]
