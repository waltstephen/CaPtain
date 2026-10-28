"""LIBERO-PRO / LIBERO-plus plumbing: config paths, interpreter choice, instruction source, robot-init override and
lazy sensor noise (no simulator needed)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from captain_harness.server import libero_variants as lv
from captain_harness.server.env_server import build_parser
from captain_harness.server.libero_setup import configure_libero


def _fake_checkout(tmp_path: Path, name: str) -> Path:
    root = tmp_path / name
    (root / "libero" / "libero" / "bddl_files").mkdir(parents=True)
    return root


def test_configure_libero_default_paths_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv("LIBERO_CONFIG_PATH", raising=False)
    config = configure_libero(tmp_path, tmp_path / "cfg")
    text = (config / "config.yaml").read_text()
    assert f"bddl_files: {(tmp_path / 'third_party/libero/libero/libero/bddl_files').resolve()}" in text
    assert f"datasets: {(tmp_path / 'third_party/libero/libero/datasets').resolve()}" in text


def test_configure_libero_fork_checkout(tmp_path, monkeypatch):
    monkeypatch.delenv("LIBERO_CONFIG_PATH", raising=False)
    checkout = _fake_checkout(tmp_path, "LIBERO-PRO")
    config = configure_libero(tmp_path, tmp_path / ".libero-config-pro", libero_checkout=checkout)
    text = (config / "config.yaml").read_text()
    assert f"init_states: {(checkout / 'libero/libero/init_files').resolve()}" in text
    assert "third_party" not in text


def test_variant_checkout_and_python(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPTAIN_BENCHMARKS_HOME", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        lv.variant_checkout("plus")
    _fake_checkout(tmp_path, "LIBERO-plus")
    assert lv.variant_checkout("plus") == tmp_path / "LIBERO-plus"
    with pytest.raises(FileNotFoundError):
        lv.variant_python(tmp_path / "repo", "pro")
    (tmp_path / ".venv-libero-pro" / "bin").mkdir(parents=True)
    (tmp_path / ".venv-libero-pro" / "bin" / "python").write_text("")
    assert lv.variant_python(tmp_path / "repo", "pro") == tmp_path / ".venv-libero-pro" / "bin" / "python"
    assert lv.variant_python(tmp_path / "repo", "libero") == tmp_path / "repo" / ".venv-libero" / "bin" / "python"


def test_parser_accepts_fork_suites():
    args = build_parser().parse_args(["--suite", "libero_goal_task", "--task-id", "7", "--output-dir", "/tmp/x",
                                      "--libero-variant", "pro", "--init-state-index", "0"])
    assert args.libero_variant == "pro" and args.suite == "libero_goal_task" and args.init_state_index == 0
    assert args.instruction_source == "auto" and args.robot_init_override == "auto"
    default = build_parser().parse_args(["--suite", "libero_goal", "--task-id", "0", "--output-dir", "/tmp/x"])
    assert default.libero_variant == "libero" and default.init_state_index is None


def test_bddl_language_prefers_parsed_problem():
    env = SimpleNamespace(language_instruction="  Turn  off the stove ", env=SimpleNamespace(parsed_problem={}))
    assert lv.bddl_language(env) == "Turn off the stove"
    env = SimpleNamespace(env=SimpleNamespace(parsed_problem={"language_instruction": ["put", "the", "bowl"]}))
    assert lv.bddl_language(env) == "put the bowl"


class _FakeSim:
    def __init__(self):
        self.data = SimpleNamespace(qpos=np.zeros(12), qvel=np.ones(12))
        self.forwarded = 0

    def forward(self):
        self.forwarded += 1


class _FakeEnv:
    def __init__(self, init_qpos):
        self.env = SimpleNamespace(sim=_FakeSim(), _get_observations=lambda: {"robot0_joint_pos": self.env.sim.data.qpos[:7].copy()})
        self.robots = [SimpleNamespace(robot_model=SimpleNamespace(init_qpos=np.asarray(init_qpos)),
                                       _ref_joint_pos_indexes=list(range(7)), _ref_joint_vel_indexes=list(range(7)))]
        self.noise = 0

    def set_init_state(self, state):
        self.env.sim.data.qpos[:] = state
        return {"robot0_joint_pos": self.env.sim.data.qpos[:7].copy()}

    def check_success(self):
        return False

    def _post_process(self):
        pass

    def _update_observables(self, force=False):
        pass


def test_robot_init_override_only_for_perturbed_robot():
    stock = _FakeEnv(lv.DEFAULT_PANDA_INIT_QPOS)
    assert lv.install_robot_init_override(stock) is None
    perturbed_q = np.asarray(lv.DEFAULT_PANDA_INIT_QPOS) + 0.1
    env = _FakeEnv(perturbed_q)
    assert lv.install_robot_init_override(env) is not None
    obs = env.set_init_state(np.full(12, 0.5))
    assert np.allclose(obs["robot0_joint_pos"], perturbed_q)
    assert np.allclose(env.env.sim.data.qpos[7:], 0.5)  # objects keep the saved state
    assert np.allclose(env.env.sim.data.qvel[:7], 0.0)


def test_lazy_noise_corrupts_once_on_access():
    calls = []

    def corrupt(image):
        calls.append(1)
        return image + 1

    obs = lv._LazyNoisyObservation({"agentview_image": np.zeros((2, 2, 3)), "robot0_eye_in_hand_image": np.zeros(1)}, corrupt)
    assert not calls
    assert obs["robot0_eye_in_hand_image"].sum() == 0 and not calls
    assert obs["agentview_image"].sum() == 12 and obs.get("agentview_image").sum() == 12
    assert len(calls) == 1


def test_lazy_noise_noop_without_noise():
    env = _FakeEnv(lv.DEFAULT_PANDA_INIT_QPOS)
    assert lv.install_lazy_sensor_noise(env) is None
