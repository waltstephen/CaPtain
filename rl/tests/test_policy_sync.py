"""A new policy must not serve cached state from the previous weights."""
import json
from pathlib import Path
from unittest import mock

import pytest
import requests

import train as trainer


def response(status=200, payload=None):
    result = requests.Response()
    result.status_code = status
    result._content = b"" if payload is None else json.dumps(payload).encode()
    return result


@pytest.mark.parametrize("payload", [None, True, {"success": True}])
def test_resume_only_after_all_caches_confirm_reset(payload):
    calls = []

    def post(url, **kwargs):
        calls.append(url.rsplit("/", 1)[-1])
        result = payload if "reset_" in url else None
        if "reset_prefix_cache" in url and result is None:
            result = {"success": True}
        return response(payload=result)

    with mock.patch.object(trainer.requests, "post", side_effect=post):
        trainer._sync_one("http://policy", Path("/weights"), lambda _: None)
    assert calls == ["pause", "collective_rpc", "reset_prefix_cache", "reset_mm_cache", "reset_encoder_cache", "resume"]


@pytest.mark.parametrize("route", ["reset_prefix_cache", "reset_mm_cache", "reset_encoder_cache"])
@pytest.mark.parametrize("status,payload", [(500, None), (404, None), (200, False), (200, {"success": False}), (200, {"unknown": True})])
def test_cache_reset_failure_does_not_resume_or_report_success(route, status, payload):
    calls = []

    def post(url, **kwargs):
        name = url.rsplit("/", 1)[-1]
        calls.append(name)
        if name == route:
            return response(status, payload)
        return response(payload={"success": True} if name == "reset_prefix_cache" else None)

    with mock.patch.object(trainer.requests, "post", side_effect=post):
        with pytest.raises((requests.HTTPError, RuntimeError)):
            trainer._sync_one("http://policy", Path("/weights"), lambda _: None)
    assert calls[-1] == route
    assert "resume" not in calls


def test_non_json_cache_confirmation_is_rejected():
    bad = response()
    bad._content = b"not a success receipt"
    with mock.patch.object(trainer.requests, "post", side_effect=[response(), response(), bad]):
        with pytest.raises(RuntimeError, match="unverifiable"):
            trainer._sync_one("http://policy", Path("/weights"), lambda _: None)


def test_empty_prefix_cache_response_is_not_success():
    with mock.patch.object(trainer.requests, "post", return_value=response()):
        with pytest.raises(RuntimeError, match="no success receipt"):
            trainer._sync_one("http://policy", Path("/weights"), lambda _: None)


def test_any_policy_sync_failure_prevents_success_log():
    messages = []
    with mock.patch.object(trainer, "POLICY_URLS", ["http://a", "http://b"]), \
            mock.patch.object(trainer, "_sync_one", side_effect=[None, RuntimeError("cache not reset")]):
        with pytest.raises(RuntimeError, match="cache not reset"):
            trainer.sync_vllm(Path("/weights"), messages.append)
    assert not any("reloaded in" in message for message in messages)
