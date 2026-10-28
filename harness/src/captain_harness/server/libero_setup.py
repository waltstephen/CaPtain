"""LIBERO configuration for a checkout that vendors LIBERO under ``third_party/``."""
from __future__ import annotations

import os
from pathlib import Path


def configure_libero(root: Path, config_dir: Path | None = None, *, libero_checkout: Path | None = None) -> Path:
    """Write ``config.yaml`` pointing LIBERO at ``root/third_party`` and export LIBERO_CONFIG_PATH.

    ``libero_checkout``: repository root of a LIBERO fork (LIBERO-PRO / LIBERO-plus) to use instead of the vendored
    LIBERO; give it its own ``config_dir`` so the original config is never rewritten.
    Returns the config directory. Safe to call repeatedly.
    """
    root = Path(root)
    config_dir = Path(config_dir) if config_dir is not None else root / ".libero-config"
    config_dir.mkdir(parents=True, exist_ok=True)
    checkout = Path(libero_checkout) if libero_checkout is not None else root / "third_party/libero"
    package_root = checkout / "libero/libero"
    values = {
        "assets": package_root / "assets",
        "bddl_files": package_root / "bddl_files",
        "benchmark_root": package_root,
        "datasets": checkout / "libero/datasets",
        "init_states": package_root / "init_files",
    }
    # atomic replace: parallel env servers sharing a config dir must never read a half-written config.yaml
    text = "\n".join(f"{key}: {value.resolve()}" for key, value in values.items()) + "\n"
    target = config_dir / "config.yaml"
    try:
        current = target.read_text(encoding="utf-8")
    except OSError:
        current = None
    if current != text:
        tmp = config_dir / f".config.yaml.{os.getpid()}.tmp"
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, target)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir.resolve())
    return config_dir


_configure_libero = configure_libero

__all__ = ["configure_libero"]
