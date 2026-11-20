"""Speed profiles (soarm/speed_profile.py): parsing, driver flags, and that every option is a real harness option."""
import json
import os
from pathlib import Path

import pytest

from soarm import speed_profile as sp

HARNESS = Path(os.environ.get("HARNESS_DIR") or Path(__file__).resolve().parents[3] / "harness")


def test_unset_profile_is_the_release_protocol(monkeypatch):
    monkeypatch.delenv("SOARM_PROFILE", raising=False)
    assert sp.load() is None and sp.driver_args(None) == []
    monkeypatch.setenv("SOARM_PROFILE", "")
    assert sp.load() is None


def test_fast_profile_loads_by_name_and_path_and_maps_driver_keys_to_flags(monkeypatch):
    monkeypatch.setenv("SOARM_PROFILE", "fast")
    prof = sp.load()
    assert prof["name"] == "fast"
    assert {k: v for k, v in sp.load(str(sp.profile_path("fast"))).items() if k != "name"} == \
        {k: v for k, v in prof.items() if k != "name"}
    args = sp.driver_args(prof)
    flags = [a for a in args if a.startswith("--")]
    assert flags == ["--" + k.replace("_", "-") for k, v in prof["driver"].items() if v is not False]


def test_cli_prints_one_word_per_line(capsys):
    assert sp.main(["driver-args", "fast"]) == 0
    words = capsys.readouterr().out.split("\n")
    assert words[:2] == ["--temperature", "0.7"] and words[-1] == ""
    assert sp.main(["sam3-cache-images", "fast"]) == 0
    assert capsys.readouterr().out.strip() == str(sp.load("fast")["services"]["sam3_cache_images"])
    assert sp.main(["bogus", "fast"]) == 2


@pytest.mark.parametrize("raw, match", [
    ({"speed": {}}, "unknown sections"),
    ({"driver": {"max_image": 3}}, "unknown driver keys"),
    ({"driver": {"max_images": 12.5}}, "must be int"),
    ({"driver": {"temperature": "hot"}}, "must be float"),
    ({"backend_options": {"gripper_settle_min_steps": True}}, "must be int"),
    ({"backend_options": {"step_profile": 3}}, "must be an object"),
    ({"tool_backend_options": []}, "must be an object"),
])
def test_bad_profiles_are_rejected(raw, match):
    with pytest.raises(ValueError, match=match):
        sp.parse(raw)


def test_missing_profile_file_fails_loudly():
    with pytest.raises(FileNotFoundError):
        sp.load("no_such_profile")


def test_every_option_exists_in_the_harness():
    flags = (HARNESS / "scripts" / "react_episode.py").read_text()
    for key in sp.DRIVER_KEYS:
        assert f'"--{key.replace("_", "-")}"' in flags, key
    pytest.importorskip("numpy")
    from captain_harness.adapters.capx.motion_speed import MotionSpeedOptions
    tool = sp.load("fast")["tool_backend_options"]
    MotionSpeedOptions(tool["motion_speed_scale"], tool["default_max_speed_m_s"], tool["default_max_angular_speed_rad_s"])
    pytest.importorskip("mujoco")
    from captain_harness.adapters.hardware.backend import step_profile_attributes
    assert step_profile_attributes(sp.load("fast")["backend_options"]["step_profile"])["_speed_scale"] == 3.


def test_profile_json_is_plain(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"driver": {"thinking_token_budget": 512}}))
    prof = sp.load(str(path))
    assert sp.driver_args(prof) == ["--thinking-token-budget", "512"] and prof["backend_options"] == {}


def test_bool_driver_options_are_value_less_flags():
    on = sp.parse({"driver": {"stream_exec": True, "max_images": 12}})
    assert sp.driver_args(on) == ["--stream-exec", "--max-images", "12"]
    assert sp.driver_args(sp.parse({"driver": {"stream_exec": False}})) == []
    with pytest.raises(ValueError, match="must be bool"):
        sp.parse({"driver": {"stream_exec": 1}})
