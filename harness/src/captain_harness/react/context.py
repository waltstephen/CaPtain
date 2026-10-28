"""Context management for the bare loop (V1.2): turn-atomic cut points, programmatic facts, incremental summaries.

Design (after Pi Agent's session compaction and Codex's local compaction):
* the history is split into *turns*: one model response (reasoning / message / tool calls) plus the tool
  outputs it produced; user messages are turns of their own. A cut never separates a call from its output.
* compaction keeps the initial context items, then a single user message describing what was dropped, then
  the most recent whole turns within ``keep_recent_tokens``.
* the drop notice always contains programmatic facts (turns / tool calls omitted, the commands that were run,
  the model's own last status notes, files now in the workspace) - deterministic, never delegated to the model.
* strategy "summary" adds an LLM summary generated incrementally (previous summary + newly dropped turns),
  bounded by ``summary_budget_tokens``; strategy "window" uses the facts only (small models);
  strategy "notes" (a compact-checkpoint turn) first gives the model one turn to update its own
  notes file in the workspace, then carries that file's contents in the notice instead of an LLM summary.
* token estimates: ``approx_tokens`` on the JSON of each item, images counted as IMAGE_TOKENS each.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from captain_harness.react.tools import approx_tokens

IMAGE_TOKENS = 1_000
DATA_URL = re.compile(r"^data:([^;]+);base64,")
CMD_RE = re.compile(r'"cmd"\s*:\s*"((?:[^"\\]|\\.)*)"')
CODE_MODE_CMD_RE = re.compile(r"exec_command\(\s*(?:\{[^}]*?cmd\s*:\s*)?(['\"`])(.*?)\1", re.S)
MAX_COMMANDS_IN_NOTICE = 12
MAX_NOTES_IN_NOTICE = 3
NOTICE_HEADER = "<context_compaction>"
NOTICE_FOOTER = "</context_compaction>"

SUMMARIZATION_PROMPT = (
    "You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume the task.\n\n"
    "Include:\n"
    "- Current progress and key decisions made\n"
    "- Important context, constraints, or user preferences\n"
    "- What remains to be done (clear next steps)\n"
    "- Any critical data, examples, or references needed to continue\n\n"
    "Be concise, structured, and focused on helping the next LLM seamlessly continue the work."
)
SUMMARY_PREFIX = (
    "Another language model started to solve this problem and produced a summary of its thinking process. You also have "
    "access to the state of the tools that were used by that language model. Use this to build on the work that has already "
    "been done and avoid duplicating work. Here is the summary produced by the other language model, use the information in "
    "this summary to assist with your own analysis: "
)
NO_SUMMARY = "(no summary available)"
DEFAULT_NOTES_FILE = "NOTES.md"
MAX_NOTES_CHARS = 16 * 1024
CONTINUE_PROMPT = (
    "<context_compaction_done>\nYour earlier turns were compacted; the <context_compaction> note above holds what was removed. "
    "Nothing else changed. Continue the task from your most recent turns.\n</context_compaction_done>"
)
CHECKPOINT_PROMPT = (
    "<context_checkpoint>\nYour conversation history is about to be compacted: everything except the task files, the "
    "images, your most recent turns and the file {notes} will be removed. Update {notes} now so that it alone lets you "
    "continue: the goal, what is confirmed (ids, numbers, poses you rely on), what is done, what failed and why, the current "
    "plan and the exact next step. Write it with a single shell command, do not take any other action in this turn, and end "
    "your turn immediately after the file is written.\n</context_checkpoint>"
)


def text_of(item: dict[str, Any]) -> str:
    content = item.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    return ""


def user_message(text: str) -> dict[str, Any]:
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def is_notice(item: dict[str, Any]) -> bool:
    return item.get("type") == "message" and item.get("role") == "user" and text_of(item).startswith(NOTICE_HEADER)


def estimate_item_tokens(item: dict[str, Any]) -> int:
    """approx_tokens over the serialised item, with image data URLs counted as IMAGE_TOKENS each (in tool outputs and
    in message content alike: --inline-task attaches the initial frames to the prompt message)."""
    images = 0
    for key in ("output", "content"):
        value = item.get(key)
        if not isinstance(value, list):
            continue
        parts = []
        for part in value:
            if isinstance(part, dict) and part.get("type") == "input_image":
                images += 1
                parts.append({k: v for k, v in part.items() if k != "image_url"})
            else:
                parts.append(part)
        item = {**item, key: parts}
    return approx_tokens(json.dumps(item, ensure_ascii=False)) + images * IMAGE_TOKENS


def estimate_tokens(items: list[dict[str, Any]]) -> int:
    return sum(estimate_item_tokens(it) for it in items)


def normalize_usage(usage: dict[str, Any] | None) -> dict[str, Any]:
    """Accept Responses usage (input_tokens/output_tokens/total_tokens) and chat-completions usage
    (prompt_tokens/completion_tokens/total_tokens); always return the Responses field names."""
    usage = dict(usage or {})
    if "input_tokens" not in usage and "prompt_tokens" in usage:
        usage["input_tokens"] = usage.get("prompt_tokens")
    if "output_tokens" not in usage and "completion_tokens" in usage:
        usage["output_tokens"] = usage.get("completion_tokens")
    if not usage.get("total_tokens") and (usage.get("input_tokens") or usage.get("output_tokens")):
        usage["total_tokens"] = int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)
    details = usage.get("prompt_tokens_details")
    if details and "input_tokens_details" not in usage:
        usage["input_tokens_details"] = {"cached_tokens": details.get("cached_tokens", 0)}
    return usage


@dataclass
class TokenEstimator:
    """bytes/4 estimate, corrected by the ratio real/estimated measured on previous requests (EMA).

    The ratio absorbs tokenizer differences (Qwen vs OpenAI, ~1.15-1.2x) and image token costs without shipping a
    tokenizer; it converges within a few requests. The raw estimate must cover the whole request (instructions and
    tool schemas included, see Loop._fixed_raw) so the ratio is a pure scale. Requests smaller than
    ``min_calibration_tokens`` are ignored (their relative noise is too high). Clamped so one bad usage report cannot
    swing the scale wildly."""
    scale: float = 1.0
    alpha: float = 0.5
    samples: int = 0
    min_scale: float = 0.5
    max_scale: float = 3.0
    min_calibration_tokens: int = 1_500

    def estimate(self, items: list[dict[str, Any]], fixed_raw: int = 0) -> int:
        return int(round((fixed_raw + estimate_tokens(items)) * self.scale))

    def calibrate(self, estimated_raw: int, real_input_tokens: int | None) -> None:
        if not real_input_tokens or estimated_raw < self.min_calibration_tokens:
            return
        ratio = max(self.min_scale, min(self.max_scale, real_input_tokens / estimated_raw))
        self.scale = ratio if self.samples == 0 else (1 - self.alpha) * self.scale + self.alpha * ratio
        self.samples += 1


def _is_output(item: dict[str, Any]) -> bool:
    return item.get("type") in ("function_call_output", "custom_tool_call_output")


def _is_user(item: dict[str, Any]) -> bool:
    return item.get("type") == "message" and item.get("role") == "user"


def split_turns(items: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group items into turns; a tool output always stays in the turn of its call."""
    turns: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    seen_output = False
    for item in items:
        if _is_user(item):
            if current:
                turns.append(current)
            turns.append([item])
            current, seen_output = [], False
            continue
        if _is_output(item):
            current.append(item)
            seen_output = True
            continue
        if seen_output and current:
            turns.append(current)
            current, seen_output = [], False
        current.append(item)
    if current:
        turns.append(current)
    return turns


def choose_cut(turns: list[list[dict[str, Any]]], keep_recent_tokens: int) -> int:
    """Index of the first turn to keep verbatim: newest turns within the budget, at least the newest one."""
    if not turns:
        return 0
    budget = keep_recent_tokens
    index = len(turns)
    while index > 0:
        cost = estimate_tokens(turns[index - 1])
        if cost > budget and index < len(turns):
            break
        budget -= cost
        index -= 1
    return index


def commands_in(items: list[dict[str, Any]]) -> list[str]:
    commands: list[str] = []
    for item in items:
        if item.get("type") == "function_call" and item.get("name") == "exec_command":
            match = CMD_RE.search(str(item.get("arguments") or ""))
            if match:
                commands.append(json.loads(f'"{match.group(1)}"'))
        elif item.get("type") == "custom_tool_call" and item.get("name") == "exec":
            for match in CODE_MODE_CMD_RE.finditer(str(item.get("input") or "")):
                commands.append(match.group(2))
        elif item.get("type") == "custom_tool_call" and item.get("name") == "python":
            code = str(item.get("input") or "").strip()
            if code:
                commands.append("python: " + _one_line(code, 160))
    return commands


def _one_line(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass
class CompactionState:
    """Carried across compactions: cumulative counts and the previous incremental summary."""
    compactions: int = 0
    omitted_turns: int = 0
    omitted_tool_calls: int = 0
    commands: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    summary: str | None = None

    def absorb(self, dropped_turns: list[list[dict[str, Any]]]) -> None:
        items = [it for turn in dropped_turns for it in turn if not is_notice(it)]
        self.omitted_turns += sum(1 for t in dropped_turns if not (len(t) == 1 and is_notice(t[0])))
        self.omitted_tool_calls += sum(1 for it in items if it.get("type") in ("function_call", "custom_tool_call"))
        self.commands.extend(commands_in(items))
        self.notes.extend(text_of(it) for it in items if it.get("type") == "message" and it.get("role") == "assistant" and text_of(it).strip())
        self.compactions += 1


def read_notes(workspace: Path | None, notes_file: str = DEFAULT_NOTES_FILE) -> str | None:
    if workspace is None:
        return None
    path = Path(workspace) / notes_file
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    if not text:
        return None
    return text if len(text) <= MAX_NOTES_CHARS else text[:MAX_NOTES_CHARS] + "\n…(truncated)"


def drop_notice(state: CompactionState, *, workspace: Path | None, summary: str | None, notes: str | None = None,
                notes_file: str = DEFAULT_NOTES_FILE) -> dict[str, Any]:
    lines = [NOTICE_HEADER,
             f"Earlier context was compacted ({state.compactions} time{'s' if state.compactions != 1 else ''}): {state.omitted_turns} of your "
             f"turns with {state.omitted_tool_calls} tool calls were removed from this conversation. The environment kept running; "
             "the world may have changed since your last kept turn, so look again before relying on old numbers."]
    if state.commands:
        recent = state.commands[-MAX_COMMANDS_IN_NOTICE:]
        lines.append(f"Commands you ran in the removed turns (last {len(recent)} of {len(state.commands)}):")
        lines.extend(f"  {i}. {_one_line(c, 160)}" for i, c in enumerate(recent, 1))
    if state.notes:
        lines.append("Your last status notes before the cut:")
        lines.extend(f"  - {_one_line(n, 300)}" for n in state.notes[-MAX_NOTES_IN_NOTICE:])
    if workspace is not None and Path(workspace).is_dir():
        files = sorted(p.relative_to(workspace).as_posix() for p in Path(workspace).rglob("*") if p.is_file() and not p.name.startswith("."))
        if files:
            shown = files[:40]
            lines.append(f"Files in the workspace now ({len(files)}): " + ", ".join(shown) + (" …" if len(files) > 40 else ""))
    if notes:
        lines.append(f"Contents of your {notes_file} at the time of compaction:\n{notes}")
    elif notes_file and workspace is not None and summary is None:
        lines.append(f"You had not written {notes_file}; keep one from now on so the next compaction preserves your state.")
    if summary:
        lines.append(f"{SUMMARY_PREFIX}\n{summary}")
    lines.append(NOTICE_FOOTER)
    return user_message("\n".join(lines))


def summary_request(body: dict[str, Any], initial_items: list[dict[str, Any]], previous_summary: str | None,
                    dropped_turns: list[list[dict[str, Any]]], budget_tokens: int) -> dict[str, Any]:
    """Incremental summary: <previous-summary> + <conversation> of newly dropped turns, no tools, bounded length."""
    request = {k: v for k, v in body.items() if k not in ("tools", "tool_choice", "parallel_tool_calls")}
    items: list[dict[str, Any]] = list(initial_items)
    if previous_summary:
        items.append(user_message(f"<previous-summary>\n{previous_summary}\n</previous-summary>"))
    items.append(user_message("<conversation>"))
    items.extend(it for turn in dropped_turns for it in turn if not is_notice(it))
    items.append(user_message("</conversation>"))
    words = max(150, budget_tokens * 3 // 4)
    items.append(user_message(f"{SUMMARIZATION_PROMPT}\n\nMerge <previous-summary> (if present) with <conversation> into one updated "
                              f"summary of at most {words} words. Keep concrete numbers (poses, offsets, detection ids, budgets) that the next "
                              "steps depend on."))
    request["input"] = items
    return request


def extract_summary(items: list[dict[str, Any]], budget_tokens: int) -> str:
    texts = [text_of(it) for it in items if it.get("type") == "message" and it.get("role") == "assistant"]
    texts = [t for t in texts if t.strip()]
    if not texts:
        return NO_SUMMARY
    summary = texts[-1].strip()
    limit = budget_tokens * 4
    return summary if len(summary.encode("utf-8")) <= limit else summary.encode("utf-8")[:limit].decode("utf-8", errors="ignore") + " …"


def plan_compaction(initial_items: list[dict[str, Any]], history: list[dict[str, Any]], keep_recent_tokens: int):
    """Return (dropped_turns, kept_turns) for the non-initial part of the history."""
    tail = history[len(initial_items):]
    turns = split_turns(tail)
    cut = choose_cut(turns, keep_recent_tokens)
    return turns[:cut], turns[cut:]


def is_continue(item: dict[str, Any]) -> bool:
    return item.get("type") == "message" and item.get("role") == "user" and text_of(item).startswith("<context_compaction_done>")


def rebuild(initial_items: list[dict[str, Any]], kept_turns: list[list[dict[str, Any]]], notice: dict[str, Any]) -> list[dict[str, Any]]:
    """initial context + drop notice + recent turns + a continuation cue, so the last user message always asks to go on."""
    kept = [it for turn in kept_turns for it in turn if not (is_notice(it) or is_continue(it))]
    return list(initial_items) + [notice] + kept + [user_message(CONTINUE_PROMPT)]


__all__ = ["CHECKPOINT_PROMPT", "CONTINUE_PROMPT", "TokenEstimator", "normalize_usage", "is_continue", "DEFAULT_NOTES_FILE", "read_notes", "CompactionState", "IMAGE_TOKENS", "NOTICE_HEADER", "SUMMARIZATION_PROMPT", "SUMMARY_PREFIX", "choose_cut", "commands_in",
           "drop_notice", "estimate_item_tokens", "estimate_tokens", "extract_summary", "is_notice", "plan_compaction", "rebuild",
           "split_turns", "summary_request", "text_of", "user_message"]
