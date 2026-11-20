"""Gateway env-file secrets: TEACHER_* names, with the OPENAI_* aliases."""
from __future__ import annotations

import pytest

from soarm.gateway_proxy import gateway_secrets, read_env


def _env(tmp_path, text):
    path = tmp_path / "gateway.env"
    path.write_text(text, encoding="utf-8")
    return read_env(str(path))


def test_primary_names(tmp_path):
    env = _env(tmp_path, "TEACHER_BASE_URL=https://gw.example/v1\nTEACHER_API_KEY=k1\n")
    assert gateway_secrets(env) == ("https://gw.example/v1", "k1")


def test_openai_aliases(tmp_path):
    env = _env(tmp_path, "OPENAI_BASE_URL=https://proxy.example/v1/\nOPENAI_API_KEY=dummy-k2\n")
    assert gateway_secrets(env) == ("https://proxy.example/v1", "dummy-k2")


def test_missing_names_raise(tmp_path):
    with pytest.raises(KeyError):
        gateway_secrets(_env(tmp_path, "DEEPSEEK_API_KEY=only\n"))
