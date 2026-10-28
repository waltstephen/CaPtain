#!/usr/bin/env python3
"""Raw-request loss boundaries and legacy Qwen3.5 archive inspection helpers.

New training samples use completion_mask only. Chat role parsing and history
alignment below are retained for inspecting archived segment-format rollouts.
"""
from __future__ import annotations

import math

IM_START, IM_END, NEWLINE = 248045, 248046, 198
ROLE_ASSISTANT = 74455
THINK_OPEN, THINK_CLOSE, NLNL = 248068, 248069, 271   # Qwen3.5 <think>, </think>, "\n\n"
SAMPLE_FORMAT = "per_request_v1"
RECORD_FORMAT = "raw_requests_v1"
NEUTRAL_SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0,
                    "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0}


def validate_sampling_settings(sampling: dict) -> None:
    """Reject transforms whose probabilities the original-token loss cannot score."""
    if not isinstance(sampling, dict):
        raise ValueError("missing sampling settings")
    for key, expected in NEUTRAL_SAMPLING.items():
        actual = sampling.get(key, expected)
        if type(actual) not in (int, float) or not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"unsupported transformed sampling distribution: {key}={actual}")
    for key in ("n", "best_of"):
        actual = sampling.get(key, 1)
        if key == "best_of" and actual is None:
            continue
        if type(actual) is not int or actual != 1:
            raise ValueError("multiple-choice sampling requires its own probability correction")
    minimum = sampling.get("min_tokens", 0)
    if type(minimum) is not int or minimum != 0:
        raise ValueError("min_tokens must be zero; EOS masking changes sampling probabilities")
    if sampling.get("ignore_eos", False) is not False:
        raise ValueError("ignore_eos must be false for unmodified policy sampling")
    for key in ("logit_bias", "allowed_token_ids", "bad_words", "logits_processors", "structured_outputs",
                "guided_json", "guided_regex", "guided_choice", "guided_grammar", "guided_json_object"):
        if sampling.get(key):
            raise ValueError(f"modified sampling support cannot use unmodified model log probabilities: {key}")
    response_format = sampling.get("response_format")
    if response_format is not None and response_format != {"type": "text"}:
        raise ValueError("structured response_format changes sampling support")
    if sampling.get("tool_choice", "auto") not in ("auto", "none"):
        raise ValueError("forced tool_choice changes sampling support")


def validate_token_ids(value, *, name: str, allow_empty: bool = True) -> list[int]:
    if not isinstance(value, list) or any(type(t) is not int or t < 0 for t in value):
        raise ValueError(f"{name} must be a list of non-negative integer token ids")
    if not allow_empty and not value:
        raise ValueError(f"{name} must not be empty")
    return list(value)


def completion_mask(prompt_length: int, completion_length: int) -> list[int]:
    """Mask the actual sampled suffix, without interpreting or adding any token."""
    if type(prompt_length) is not int or prompt_length < 1:
        raise ValueError("prompt_length must be a positive integer")
    if type(completion_length) is not int or completion_length < 0:
        raise ValueError("completion_length must be a non-negative integer")
    return [0] * prompt_length + [1] * completion_length


def assistant_mask(ids: list[int]) -> list[int]:
    """1 for tokens inside <|im_start|>assistant\\n ... <|im_end|> blocks (header excluded, <|im_end|> included)."""
    mask = [0] * len(ids)
    i = 0
    n = len(ids)
    while i < n:
        if ids[i] == IM_START and i + 2 < n and ids[i + 1] == ROLE_ASSISTANT and ids[i + 2] == NEWLINE:
            j = i + 3
            while j < n and ids[j] != IM_END:
                mask[j] = 1
                j += 1
            if j < n:
                mask[j] = 1  # <|im_end|>
            # The chat template's think prefix is part of the PROMPT (generation prompt "<think>\n" with thinking on,
            # "<think>\n\n</think>\n\n" with thinking off): the policy never samples those tokens, so they must not get
            # REINFORCE / KL gradients. Left in, the empty block drifted to log p(<think>) = -23 by step 60 and the KL term
            # exploded (attempt 5). Unmask <think> (+ its newline) and, for the empty block, </think> (+ its newline).
            k = i + 3
            if k < n and ids[k] == THINK_OPEN:
                mask[k] = 0; k += 1
                if k < n and ids[k] in (NLNL, NEWLINE):
                    mask[k] = 0; k += 1
                if k < n and ids[k] == THINK_CLOSE:
                    mask[k] = 0; k += 1
                    if k < n and ids[k] in (NLNL, NEWLINE):
                        mask[k] = 0
            i = j + 1
            continue
        i += 1
    return mask


def assistant_blocks(ids: list[int], mask: list[int]) -> list[tuple[int, int]]:
    """(start, end) index ranges of the masked (assistant) blocks, chronological."""
    blocks = []
    i = 0
    n = len(ids)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            blocks.append((i, j))
            i = j
        else:
            i += 1
    return blocks


def with_im_end(ids: list[int]) -> list[int]:
    ids = list(ids)
    if not ids or ids[-1] != IM_END:
        ids.append(IM_END)
    return ids


def load_tokenizer(path: str | None):
    """tokenizers.Tokenizer from <model dir>/tokenizer.json (the rollout service's env has no transformers); None if unavailable."""
    if not path:
        return None
    try:
        from pathlib import Path

        from tokenizers import Tokenizer
        p = Path(path)
        return Tokenizer.from_file(str(p / "tokenizer.json" if p.is_dir() else p))
    except Exception:  # noqa: BLE001
        return None


def same_text(tokenizer, a: list[int], b: list[int]) -> bool:
    """Legacy API: only identical sampled token sequences are equivalent."""
    return a == b


def align_history(ids: list[int], end_index: int, ok_indices: list[int], requests: list[dict], tokenizer=None) -> tuple[list[dict], bool]:
    """Map the assistant blocks inside one training sequence (prompt of the segment-ending request `end_index` (0-based proxy index)
    + its completion, <|im_end|>-terminated) to the earlier successful proxy requests, most recent first: the driver re-renders every
    earlier model turn as exactly one assistant message, so history block -j (from the end) is request `earlier[j]`.

    Returns (history, alignment_ok). `history` has one entry per history block, chronological: {"request": 1-based proxy index or None,
    "exact": block tokens == the completion tokens the policy actually sampled for that request, "block_len", "completion_len"}.
    This positional mapping is for archive diagnostics only, never new training.
    Non-exact blocks are the driver's canonical re-rendering of a parsed (or unparseable -> empty exec) tool call; training them as if
    they were the sampled tokens gives the gradient to text the policy never produced ."""
    mask = assistant_mask(ids)
    blocks = assistant_blocks(ids, mask)
    hist_blocks = blocks[:-1]
    earlier = [i for i in ok_indices if i < end_index][::-1]
    history: list[dict] = []
    ok = len(hist_blocks) <= len(earlier)
    for j, (a, b) in enumerate(reversed(hist_blocks)):
        if j >= len(earlier):
            history.append({"request": None, "exact": False, "block_len": b - a, "completion_len": None})
            continue
        r = earlier[j]
        comp = with_im_end(list(requests[r].get("token_ids") or []))
        block = ids[a:b]
        exact = block == comp
        entry = {"request": r + 1, "exact": exact, "block_len": b - a, "completion_len": len(comp)}
        history.append(entry)
    history.reverse()
    return history, ok
