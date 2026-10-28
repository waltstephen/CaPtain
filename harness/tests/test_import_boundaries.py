"""Layer boundaries for the V1.1 layout.

* ``ports`` and ``react`` never import environment implementations.
* ``react`` is task-agnostic: no robot / LIBERO / task vocabulary, no import of ``server`` or ``adapters``.
* the capability backend and the environment server never read hidden simulator truth except at the
  single construction line that hands the host predicate to the adapter.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "captain_harness"

ENVIRONMENT_IMPLEMENTATION_PARTS = {
    "libero", "capx", "cap-x", "robotwin", "mujoco", "robosuite",
    "captain_harness.adapters", "captain_harness.server", "sam3", "molmo", "graspnet", "pyroki",
}
HIDDEN_TRUTH_NAMES = ("check_success", "reward", "object_poses", "contacts", "success_latch")
TASK_VOCABULARY = ("robot", "libero", "LIBERO", "bddl", "gripper", "observe", "TASK.md", "finish(")


def _python_files(package: str) -> list[Path]:
    root = SRC / package
    return sorted(root.rglob("*.py")) if root.exists() else []


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                package_parts = path.relative_to(SRC.parent).with_suffix("").parts[:-1]
                base = list(package_parts[: len(package_parts) - (node.level - 1)])
                modules.add(".".join(base + ([node.module] if node.module else [])))
            elif node.module:
                modules.add(node.module)
    return modules


def test_ports_and_react_do_not_import_environment_implementations() -> None:
    offenders = []
    for package in ("ports", "react"):
        for path in _python_files(package):
            for module in _imported_modules(path):
                lowered = module.lower()
                if any(part in lowered for part in ENVIRONMENT_IMPLEMENTATION_PARTS):
                    offenders.append(f"{path.relative_to(SRC)} imports {module}")
    assert not offenders, "\n".join(offenders)


def test_react_layer_is_task_agnostic() -> None:
    offenders = []
    for path in _python_files("react"):
        text = path.read_text(encoding="utf-8")
        for word in TASK_VOCABULARY:
            if word in text:
                offenders.append(f"{path.relative_to(SRC)} mentions {word!r}")
        for module in _imported_modules(path):
            if module.startswith(("captain_harness.server", "captain_harness.adapters")):
                offenders.append(f"{path.relative_to(SRC)} imports {module}")
    assert not offenders, "\n".join(offenders)


def test_tool_backend_does_not_read_hidden_simulator_truth() -> None:
    for name in ("adapters/capx/backend.py", "adapters/capx/services.py"):
        text = (SRC / name).read_text(encoding="utf-8")
        for forbidden in HIDDEN_TRUTH_NAMES:
            assert forbidden not in text, f"{name} references {forbidden}"


def test_server_touches_the_hidden_predicate_only_at_construction() -> None:
    for path in _python_files("server"):
        text = path.read_text(encoding="utf-8")
        for forbidden in HIDDEN_TRUTH_NAMES[1:]:
            assert forbidden not in text, f"{path.name} references {forbidden}"
        for line_number, line in enumerate(text.splitlines(), start=1):
            if "check_success" in line and not line.lstrip().startswith("#"):
                assert re.search(r"final_predicate\s*=\s*env\.check_success", line), (
                    f"{path.name}:{line_number} uses check_success outside adapter construction"
                )
