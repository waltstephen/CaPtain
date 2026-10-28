"""Render the first Responses request from a captured Codex CLI request template.

The templates under ``prompts/`` are real Codex 0.154 requests (captured through a dumping
proxy) with the per-episode values replaced by placeholders. Rendering swaps the workspace
path, date, timezone and prompt back in; everything else (instructions, developer/user
context items, tool schemas, sampling fields) is replayed byte for byte.
"""
from __future__ import annotations

import copy
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from captain_harness.react.catalog import ModelProfile

PROMPTS_DIR = Path(__file__).with_name("prompts")
WRITABLE_ROOTS_SENTENCE = re.compile(r"The writable roots? (?:is|are) .*?\.(?=\n|$)")


def load_template(profile: ModelProfile) -> dict[str, Any]:
    return json.loads((PROMPTS_DIR / profile.template).read_text(encoding="utf-8"))


def _writable_roots_sentence(workspace: str, extra_roots: tuple[str, ...]) -> str:
    roots = sorted({workspace, *extra_roots})
    if len(roots) == 1:
        return f"The writable root is `{roots[0]}`."
    return "The writable roots are " + ", ".join(f"`{root}`" for root in roots) + "."


def _substitute(value: Any, mapping: dict[str, str], workspace: str, extra_roots: tuple[str, ...]) -> Any:
    if isinstance(value, str):
        text = value
        for placeholder, actual in mapping.items():
            text = text.replace(placeholder, actual)
        if "<permissions instructions>" in text:
            text = WRITABLE_ROOTS_SENTENCE.sub(_writable_roots_sentence(workspace, extra_roots), text, count=1)
        return text
    if isinstance(value, list):
        return [_substitute(v, mapping, workspace, extra_roots) for v in value]
    if isinstance(value, dict):
        return {k: _substitute(v, mapping, workspace, extra_roots) for k, v in value.items()}
    return value


def render_request(profile: ModelProfile, *, model: str, workspace: Path | str, prompt: str,
                   current_date: str | None = None, timezone: str = "Etc/UTC",
                   extra_writable_roots: tuple[str, ...] = ("/tmp",)) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return (request body without ``input``, initial ``input`` items)."""
    template = load_template(profile)
    mapping = {
        "{{WORKSPACE}}": str(workspace),
        "{{CURRENT_DATE}}": current_date or datetime.now().strftime("%Y-%m-%d"),
        "{{TIMEZONE}}": timezone,
        "{{PROMPT}}": prompt,
    }
    body = _substitute(copy.deepcopy(template), mapping, str(workspace), extra_writable_roots)
    body["model"] = model
    items = body.pop("input")
    for item in items:
        if item.get("type") == "additional_tools":
            item["id"] = f"at_{uuid.uuid4()}"
    return body, items


def function_tool_names(body: dict[str, Any], items: list[dict[str, Any]]) -> list[str]:
    """Names of every tool the model may call (direct tools and code-mode namespaces)."""
    names = [t.get("name") or t.get("type") for t in body.get("tools", [])]
    for item in items:
        if item.get("type") == "additional_tools":
            for namespace in item.get("tools", []):
                names.extend(t.get("name") for t in namespace.get("tools", []))
    return [n for n in names if n]


__all__ = ["PROMPTS_DIR", "function_tool_names", "load_template", "render_request"]
