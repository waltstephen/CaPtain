"""Recorded chat/completions bodies (tools/policy_proxy.py --record) -> token sequences for MTP drafter training.

Each JSONL line is one policy request. Within an episode the next body normally extends the previous one (append-only
prompt); a body is "maximal" when the next body of the same episode does not start with all of its messages (last body,
context compaction, image cap rewrites). Two selection modes, each assistant turn trained exactly once per epoch:
  exact: every maximal body; a turn is trained in the first maximal body that contains it (closest to the context it was
         sampled in). Costs one long target forward per history rewrite (rec_sB: 462 bodies x 42 k tokens).
  cover: walk an episode's maximal bodies from last to first and keep a body only if it holds turns not covered yet;
         a turn is trained in the latest body containing it (old images may already be capped away). rec_sB: 197
         bodies instead of 462 for the same 2057 turns; the KL target is the target's own distribution in that context.

Rendering mirrors vLLM 0.29's chat path: tool_call arguments JSON strings -> dicts (chat_utils._postprocess_messages),
image_url parts -> image placeholders, the model's chat_template.jinja with the body's chat_template_kwargs and tools.
Trained positions are the generated part of every assistant turn rendered with "<think>": reasoning + "\n</think>\n\n"
content + tool calls + <|im_end|> (the "<think>\n" opener is part of the generation prompt, not sampled).
"""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass

import torch

CWD_RE = re.compile(r"<cwd>(.*?)</cwd>", re.S)


def iter_bodies(path: str, limit: int | None = None) -> Iterator[tuple[int, dict]]:
    """(byte offset, body) per non-empty JSONL line."""
    with open(path, "rb") as fh:
        n = 0
        while True:
            off = fh.tell()
            line = fh.readline()
            if not line:
                return
            if line.strip():
                if limit is not None and n >= limit:
                    return
                n += 1
                yield off, json.loads(line)


def load_body(path: str, offset: int) -> dict:
    with open(path, "rb") as fh:
        fh.seek(offset)
        return json.loads(fh.readline())


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def episode_key(body: dict) -> str:
    """Episode id: the workspace cwd of the Codex environment context, else a hash of the opening messages."""
    for m in body["messages"][:4]:
        hit = CWD_RE.search(_text_of(m.get("content")))
        if hit:
            return hit.group(1)
    return hashlib.sha1(json.dumps(body["messages"][:3], sort_keys=True).encode()).hexdigest()


def _hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def turn_hash(msg: dict) -> str:
    return _hash({k: msg.get(k) for k in ("content", "reasoning_content", "reasoning", "tool_calls")})


@dataclass
class Sample:
    path: str
    offset: int
    episode: str
    train_turns: frozenset[str]  # turn_hash of the assistant turns trained in this body

    def body(self) -> dict:
        return load_body(self.path, self.offset)


def think_turn_indices(messages: list[dict]) -> list[int]:
    """Assistant turns the chat template renders with their reasoning (<think>): those after the last real user
    query. A later user message (e.g. <context_compaction_done>, output-limit notice) strips the reasoning of every
    earlier turn, so those turns are trained only in bodies from before that message."""
    last_query = -1
    for i, m in enumerate(messages):
        if m["role"] == "user":
            c = _text_of(m.get("content")).strip()
            if not (c.startswith("<tool_response>") and c.endswith("</tool_response>")):
                last_query = i
    return [i for i, m in enumerate(messages) if m["role"] == "assistant" and i > last_query]


def select_bodies(path: str, mode: str = "cover", limit: int | None = None) -> list[Sample]:
    """Pick the bodies to render (see module doc); light first pass keeps only hashes and offsets."""
    if mode not in ("exact", "cover"):
        raise ValueError(mode)
    last: dict[str, tuple[int, list[str], list[str]]] = {}
    maximal: dict[str, list[tuple[int, list[str]]]] = {}
    for off, body in iter_bodies(path, limit):
        key = episode_key(body)
        msg_hashes = [_hash(m) for m in body["messages"]]
        turns = [turn_hash(body["messages"][i]) for i in think_turn_indices(body["messages"])]
        prev = last.get(key)
        # superseded only if the next body extends it AND still renders its turns with reasoning (a new user message
        # changes the rendered tokens of all earlier turns although the message list is a prefix)
        if prev is not None and (msg_hashes[: len(prev[1])] != prev[1] or not set(prev[2]) <= set(turns)):
            maximal.setdefault(key, []).append((prev[0], prev[2]))
        last[key] = (off, msg_hashes, turns)
    for key, (off, _, turns) in last.items():
        maximal.setdefault(key, []).append((off, turns))

    out = []
    for key, bodies in maximal.items():
        done: set[str] = set()
        order = bodies if mode == "exact" else bodies[::-1]
        picked = []
        for off, turns in order:
            new = frozenset(t for t in turns if t not in done)
            done.update(new)
            if new:
                picked.append(Sample(path, off, key, new))
        out.extend(sorted(picked, key=lambda s: s.offset))
    return out


def _decode_image(url: str):
    from PIL import Image

    if not url.startswith("data:"):
        raise ValueError(f"only data: image urls are supported, got {url[:40]}")
    raw = base64.b64decode(url.split(",", 1)[1])
    return Image.open(io.BytesIO(raw)).convert("RGB")


def to_template_messages(body: dict) -> tuple[list[dict], list]:
    """OpenAI messages -> chat-template messages (image placeholders) + PIL images in prompt order."""
    msgs, images = [], []
    for m in body["messages"]:
        m = copy.copy(m)
        content = m.get("content")
        if isinstance(content, list):
            parts = []
            for p in content:
                if p.get("type") == "image_url":
                    images.append(_decode_image(p["image_url"]["url"]))
                    parts.append({"type": "image"})
                elif p.get("type") == "text":
                    parts.append({"type": "text", "text": p["text"]})
                else:
                    raise ValueError(f"unsupported content part {p.get('type')}")
            m["content"] = parts
        if m.get("role") == "assistant":
            if m.get("reasoning_content") is None and isinstance(m.get("reasoning"), str):
                m["reasoning_content"] = m["reasoning"]
            calls = m.get("tool_calls")
            if isinstance(calls, list) and not calls:
                m.pop("tool_calls")
            elif isinstance(calls, list):
                fixed = []
                for c in calls:
                    c = copy.deepcopy(c)
                    args = c.get("function", {}).get("arguments")
                    if isinstance(args, str):
                        c["function"]["arguments"] = json.loads(args) if args.strip() else {}
                    fixed.append(c)
                m["tool_calls"] = fixed
        msgs.append(m)
    return msgs, images


@dataclass
class Encoded:
    input_ids: torch.Tensor  # [L] long
    mm: dict  # processor outputs for the vision tower (pixel_values, image_grid_thw, mm_token_type_ids)
    gen_spans: list[tuple[int, int]]  # [start, end] inclusive token ranges of trained assistant generations
    n_images: int


class Renderer:
    def __init__(self, processor):
        self.processor = processor
        tok = processor.tokenizer
        self.im_start = tok.convert_tokens_to_ids("<|im_start|>")
        self.im_end = tok.convert_tokens_to_ids("<|im_end|>")
        self.gen_prefix = tok.encode("assistant\n<think>", add_special_tokens=False)
        self.nl = tok.encode("\n", add_special_tokens=False)

    def encode(self, sample: Sample, body: dict | None = None) -> Encoded:
        body = sample.body() if body is None else body
        msgs, images = to_template_messages(body)
        kwargs = dict(body.get("chat_template_kwargs") or {})
        text = self.processor.tokenizer.apply_chat_template(
            msgs, tools=body.get("tools") or None, tokenize=False, add_generation_prompt=False, **kwargs
        )
        out = self.processor(text=[text], images=images or None, return_tensors="pt")
        ids = out["input_ids"][0]
        mm = {k: v for k, v in out.items() if k not in ("input_ids", "attention_mask")}

        # assistant turns rendered with <think> (after the last user query), in message order
        think_idx = think_turn_indices(msgs)
        spans = self._gen_spans(ids)
        if len(spans) != len(think_idx):
            raise ValueError(f"found {len(spans)} assistant generations, template rendered {len(think_idx)}")
        gen = [s for s, i in zip(spans, think_idx) if turn_hash(body["messages"][i]) in sample.train_turns]
        return Encoded(ids, mm, gen, len(images))

    def _gen_spans(self, ids: torch.Tensor) -> list[tuple[int, int]]:
        """Generated part of each "<|im_start|>assistant\n<think>\n" turn. The "\n" after <think> belongs to the
        generation prompt; when the reasoning is empty the template's "\n\n</think>" merges it into one "\n\n" token,
        which is then counted as generated (the only rendering that cannot be split exactly)."""
        lst = ids.tolist()
        p = self.gen_prefix
        spans = []
        for i, t in enumerate(lst):
            if t == self.im_start and lst[i + 1 : i + 1 + len(p)] == p:
                start = i + 1 + len(p)
                if lst[start : start + len(self.nl)] == self.nl:
                    start += len(self.nl)
                end = lst.index(self.im_end, start)
                spans.append((start, end))
        return spans


def chain_starts(gen_spans: list[tuple[int, int]], step: int) -> list[int]:
    """MTP chain origins s whose step-`step` draft predicts a generated token: s+2+step in [g, end], s >= g-1.

    vLLM's first draft of a turn comes after prefill: h_{g-1} with the first sampled token x_g predicts x_{g+1}.
    """
    out = []
    for g, end in gen_spans:
        out.extend(range(g - 1, end - 1 - step))
    return out
