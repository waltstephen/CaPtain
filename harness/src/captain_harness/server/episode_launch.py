"""Start the environment server for one episode and build the agent workspace.

Shared by the react driver, the Codex CLI reference driver and the dry-run script. Knows the
repository layout (venv, third_party) and nothing about any agent loop.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Mapping

from captain_harness.server.task_markdown import task_markdown

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROMPT = ("Read TASK.md in this directory and complete the robot task it describes using robot.py. "
                  "Call robot.finish() when you are done.")


def harness_python(root: Path) -> Path:
    return Path(root) / ".venv-libero" / "bin" / "python"


def server_env(root: Path, *, mujoco_gl: str, egl_device: int | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(Path(root) / "src"),
        "MUJOCO_GL": mujoco_gl,
        "PYOPENGL_PLATFORM": mujoco_gl,
        "LIBERO_CONFIG_PATH": str(Path(root) / ".libero-config"),
    })
    if egl_device is not None:
        env["MUJOCO_EGL_DEVICE_ID"] = str(egl_device)
    # hosts whose system glvnd only lists Mesa: a user-level vendor file selects the NVIDIA EGL ICD
    nvidia_vendor = Path.home() / ".config" / "glvnd" / "10_nvidia.json"
    if mujoco_gl == "egl" and nvidia_vendor.is_file():
        env.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", str(nvidia_vendor))
    for secret in [k for k in env if k.startswith(("OPENAI_", "ANTHROPIC_", "HF_", "AWS_", "GOOGLE_", "WANDB_")) or "TOKEN" in k or "SECRET" in k or "API_KEY" in k]:
        env.pop(secret, None)
    return env


def http_json(url: str, method: str = "GET", payload: Mapping[str, Any] | None = None, timeout: float = 600.0) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def start_server(root: Path, output_dir: Path, *, suite: str, task_id: int, seed: int, images_dir: Path,
                 max_actions: int, max_tool_calls: int, wall_time_s: float, mujoco_gl: str,
                 egl_device: int | None = None, sam3_url: str | None = None, molmo_url: str | None = None,
                 extra_args: list[str] | None = None, libero_variant: str = "libero") -> subprocess.Popen:
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    if libero_variant == "libero":
        python = harness_python(root)
    else:  # LIBERO-PRO / LIBERO-plus: the fork's own venv (a drop-in `libero` package)
        from captain_harness.server.libero_variants import variant_python

        python = variant_python(root, libero_variant)
    command = [
        str(python), "-m", "captain_harness.server.env_server",
        "--suite", suite, "--task-id", str(task_id), "--seed", str(seed),
        "--output-dir", str(output_dir), "--harness-root", str(root),
        "--images-dir", str(images_dir),
        "--max-actions", str(max_actions), "--max-tool-calls", str(max_tool_calls),
        "--wall-time", str(wall_time_s), "--mujoco-gl", mujoco_gl,
    ]
    if egl_device is not None:
        command += ["--egl-device", str(egl_device)]
    if sam3_url:
        command += ["--sam3-url", sam3_url]
    if molmo_url:
        command += ["--molmo-url", molmo_url]
    env = server_env(root, mujoco_gl=mujoco_gl, egl_device=egl_device)
    if libero_variant != "libero":
        from captain_harness.server.libero_variants import variant_env

        command += ["--libero-variant", libero_variant]
        env.pop("LIBERO_CONFIG_PATH", None)  # the server writes <root>/.libero-config-<variant>
        env.update(variant_env(libero_variant))
    command += list(extra_args or [])
    stdout = (output_dir / "server.stdout").open("ab")
    return subprocess.Popen(command, cwd=str(root), env=env,
                            stdout=stdout, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)


def wait_for_server(output_dir: Path, timeout_s: float, process: subprocess.Popen) -> dict:
    output_dir = Path(output_dir)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if (output_dir / "server.json").is_file():
            time.sleep(0.5)
            return json.loads((output_dir / "server.json").read_text(encoding="utf-8"))
        if process.poll() is not None or (output_dir / "server_error.json").is_file():
            raise RuntimeError("env server exited during start-up; see server.stdout / server.log")
        time.sleep(1.0)
    raise RuntimeError("env server did not start within %.0f s" % timeout_s)


def build_workspace(output_dir: Path, info: Mapping[str, Any], manifest_payload: Mapping[str, Any], *,
                    wall_time_s: float, max_actions: int, max_tool_calls: int, standards: bool = True,
                    runtime: str = "shell") -> Path:
    """Write TASK.md, robot.py and .robot_server into <output_dir>/workspace (images/ already exists)."""
    workspace = Path(output_dir) / "workspace"
    workspace.mkdir(exist_ok=True)
    (workspace / "images").mkdir(exist_ok=True)
    (Path(output_dir) / "capability_manifest.json").write_text(
        json.dumps(manifest_payload["manifest"], indent=1, sort_keys=True), encoding="utf-8")
    shutil.copyfile(Path(__file__).with_name("robot.py"), workspace / "robot.py")
    (workspace / ".robot_server").write_text(str(info["url"]) + "\n", encoding="utf-8")
    (workspace / "TASK.md").write_text(
        task_markdown(info, manifest_payload, wall_time_s=wall_time_s, max_actions=max_actions,
                      max_tool_calls=max_tool_calls, standards=standards, runtime=runtime), encoding="utf-8")
    return workspace


INLINE_TASK_HEADER = "# TASK.md\n\n"
INLINE_TASK_NOTE = "\n\nTASK.md is reproduced in the developer instructions above and the initial camera images are attached."


def initial_image_files(workspace: Path) -> list[Path]:
    """The first observation's frames (images/obs_000_*.png), sorted by name."""
    return sorted((Path(workspace) / "images").glob("obs_000_*.png"))


def inline_task_items(items: list[dict[str, Any]], task_text: str, image_files: list[Path] | None = None) -> list[dict[str, Any]]:
    """Return a copy of the initial request items with TASK.md inlined as a developer message (right
    before the first user item) and the initial camera frames attached to the prompt user message.

    Used by ``react_episode.py --inline-task`` and, with the same strings, by the SFT exporter so
    that training samples and served requests agree byte for byte.
    """
    import base64
    import copy
    import mimetypes

    items = copy.deepcopy(items)
    if any(it.get("type") == "message" and it.get("role") == "developer"
           and any(str(c.get("text", "")).startswith(INLINE_TASK_HEADER) for c in it.get("content") or []) for it in items):
        return items
    first_user = next((i for i, it in enumerate(items) if it.get("type") == "message" and it.get("role") == "user"), len(items))
    developer = {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": INLINE_TASK_HEADER + task_text}]}
    items.insert(first_user, developer)
    last_user = max(i for i, it in enumerate(items) if it.get("type") == "message" and it.get("role") == "user")
    content = items[last_user].setdefault("content", [])
    for part in content:
        if part.get("type") == "input_text":
            part["text"] = str(part["text"]) + INLINE_TASK_NOTE
            break
    for path in image_files or []:
        mime = mimetypes.guess_type(str(path))[0] or "image/png"
        url = f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")
        content.append({"type": "input_image", "image_url": url, "detail": "high"})
    return items


def snapshot_workspace(workspace: Path) -> dict[str, bytes]:
    """Every file of the workspace as it is before the first request (TASK.md, NOTES.md, the SDK, the initial frames)."""
    root = Path(workspace)
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def restore_workspace(workspace: Path, snapshot: dict[str, bytes], *, keep: tuple[str, ...] = ()) -> dict[str, int]:
    """Put the workspace back to ``snapshot`` for a clean handoff: files the earlier actor created are deleted (unless
    listed in ``keep``, e.g. the handoff frames under images/), files it changed or deleted are restored, and empty
    directories it made are removed."""
    root = Path(workspace)
    removed = restored = 0
    for path in sorted(root.rglob("*"), reverse=True):
        relative = str(path.relative_to(root))
        if path.is_file() or path.is_symlink():
            if relative not in snapshot and relative not in keep:
                path.unlink()
                removed += 1
        elif path.is_dir() and not any(path.iterdir()) and not any(name.startswith(relative + "/") for name in snapshot):
            path.rmdir()
    for relative, data in snapshot.items():
        path = root / relative
        if not path.is_file() or path.read_bytes() != data:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            restored += 1
    return {"removed": removed, "restored": restored}


def stop_server(process: subprocess.Popen, url: str | None) -> None:
    if url:
        try:
            http_json(url + "/shutdown", "POST", {}, timeout=30)
        except Exception:
            pass
    try:
        process.wait(timeout=60)
    except Exception:
        try:
            os.killpg(process.pid, 9)
        except Exception:
            pass


__all__ = ["DEFAULT_PROMPT", "INLINE_TASK_HEADER", "INLINE_TASK_NOTE", "REPO_ROOT", "build_workspace", "harness_python", "http_json",
           "initial_image_files", "inline_task_items", "restore_workspace", "server_env", "snapshot_workspace", "start_server",
           "stop_server", "wait_for_server"]
