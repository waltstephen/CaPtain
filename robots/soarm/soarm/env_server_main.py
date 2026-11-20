"""`python -m soarm.env_server_main <env_server args>`: the harness env server with suite `soarm_sim` registered.

The harness's episode_builder() knows only its own suites; this wraps it instead of forking the harness.
"""
from __future__ import annotations

import sys

from OpenGL import GL  # noqa: F401  -- OSMesa: PyOpenGL must load before mujoco's renderer

from captain_harness.server import env_server

_builtin = env_server.episode_builder


def episode_builder(suite: str):
    if str(suite) == "soarm_sim":
        from .episode import build_soarm_episode
        return build_soarm_episode
    if str(suite) == "soarm_real":   # the real SO-101 on armhost (soarm/real_episode.py)
        from .real_episode import build_real_episode
        return build_real_episode
    return _builtin(suite)


env_server.episode_builder = episode_builder

if __name__ == "__main__":
    sys.exit(env_server.main())
