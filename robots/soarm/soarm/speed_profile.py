"""SO-101 speed profiles (eval/deploy only): one JSON per profile, selected with SOARM_PROFILE=<name | path.json>.

Unset = the release/student protocol, nothing changes. `configs/speed_<name>.json` sections map 1:1 onto harness
options (CaPtain-Harness docs/harness.md "Speed options"):
  driver                 scripts/react_episode.py flags (tools/run_episode.sh appends them after its own flags, so
                         they override TEMP / MAX_OUTPUT_TOKENS / MAX_IMAGES; extra run_episode.sh args override them)
  tool_backend_options   ToolCapabilityBackend keywords (move_to speed scaling / defaults)
  backend_options        XR1ActionBackend keywords (gripper_settle_min_steps, step_profile)
  services               perception options: sam3_cache_images, depth_resolution_level (tools/serve_perception.sh),
                         molmo_stop_after_coords
                         (CapXHttpToolServices)
The env server reads the same SOARM_PROFILE (soarm/episode.py), so one variable selects the whole stack.

    python -m soarm.speed_profile driver-args fast          # one flag or value per line (run_episode.sh)
    python -m soarm.speed_profile sam3-cache-images fast    # 0 when the profile has none
    python -m soarm.speed_profile depth-resolution-level fast   # empty when the profile has none (server default)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
DRIVER_KEYS = {"temperature": float, "top_p": float, "thinking_token_budget": int, "max_output_tokens": int,
               "max_images": int, "image_drop_chunk": int, "server_poll_s": float, "server_grace_s": float,
               # context budget (driver window compaction); needs a policy server with --max-model-len >= the window
               "context_limit": int, "context_hard_limit": int, "small_context_window": int, "keep_recent_tokens": int,
               # value-less switches: true passes the flag, false omits it
               "stream_exec": bool, "image_cap_sticky": bool}
TOOL_BACKEND_KEYS = {"motion_speed_scale": float, "default_max_speed_m_s": float,
                     "default_max_angular_speed_rad_s": float, "concurrent_label_check": bool}
BACKEND_KEYS = {"gripper_settle_min_steps": int, "step_profile": dict}   # step_profile keys: validated by the harness
SERVICE_KEYS = {"sam3_cache_images": int, "depth_resolution_level": int, "molmo_stop_after_coords": bool}
SECTIONS = {"driver": DRIVER_KEYS, "tool_backend_options": TOOL_BACKEND_KEYS, "backend_options": BACKEND_KEYS,
            "services": SERVICE_KEYS}


def profile_path(name: str) -> Path:
    return Path(name) if name.endswith(".json") else CONFIGS / f"speed_{name}.json"


def parse(raw: dict) -> dict:
    """Validated profile: every section present (possibly empty), values cast to their option types."""
    unknown = set(raw) - set(SECTIONS) - {"description"}
    if unknown:
        raise ValueError(f"speed profile: unknown sections {sorted(unknown)}")
    profile = {"description": str(raw.get("description", ""))}
    for section, keys in SECTIONS.items():
        values = raw.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"speed profile: {section} must be an object")
        bad = set(values) - set(keys)
        if bad:
            raise ValueError(f"speed profile: unknown {section} keys {sorted(bad)}")
        out = {}
        for key, value in values.items():
            kind = keys[key]
            if kind is dict:
                if not isinstance(value, dict):
                    raise ValueError(f"speed profile: {section}.{key} must be an object")
                out[key] = dict(value)
            elif kind is bool:
                if not isinstance(value, bool):
                    raise ValueError(f"speed profile: {section}.{key} must be bool, got {value!r}")
                out[key] = value
            elif isinstance(value, bool) or not isinstance(value, (int, float)) or (kind is int and value != int(value)):
                raise ValueError(f"speed profile: {section}.{key} must be {kind.__name__}, got {value!r}")
            else:
                out[key] = kind(value)
        profile[section] = out
    return profile


def load(name: str | None = None) -> dict | None:
    """The profile named by `name` (default $SOARM_PROFILE); None when unset."""
    name = os.environ.get("SOARM_PROFILE", "") if name is None else name
    if not name:
        return None
    path = profile_path(name)
    if not path.is_file():
        raise FileNotFoundError(f"SOARM_PROFILE={name!r}: no {path}")
    return {**parse(json.loads(path.read_text())), "name": name}


def driver_args(profile: dict | None) -> list[str]:
    args: list[str] = []
    for key, value in ((profile or {}).get("driver") or {}).items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            args += [flag] if value else []
        else:
            args += [flag, str(value)]
    return args


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] not in ("driver-args", "sam3-cache-images", "depth-resolution-level"):
        print(__doc__, file=sys.stderr)
        return 2
    profile = load(argv[1])
    services = (profile or {}).get("services", {})
    if argv[0] == "driver-args":
        print("\n".join(driver_args(profile)))
    elif argv[0] == "sam3-cache-images":
        print(services.get("sam3_cache_images", 0))
    else:
        print(services.get("depth_resolution_level", ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
