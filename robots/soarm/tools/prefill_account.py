"""Where do the uncached prompt tokens of each request come from? (soarm-speed, model-side prefill lever)

    python tools/prefill_account.py <recorded bodies.jsonl> --tokenizer <model dir> [--json out.json]

Bodies come from tools/policy_proxy.py --record, grouped into episodes (success_workload.episode_key) in order.
For request i of an episode the reusable prefix is the part of its messages identical to request i-1's messages plus
request i-1's final message is NOT reusable as generated (the chat template drops earlier reasoning, so the re-rendered
assistant turn differs from the generated tokens). New tokens = everything after the first differing message
(plus the differing message itself), counted with the tokenizer on the message text (+258 tokens per image, the
measured 512x512 view cost). Each request is classified by its first differing message:
  first      the episode's first request (system prompt + task; nothing to reuse across episodes past ~5 k tokens)
  append     every earlier message unchanged: new tokens = the previous answer (re-rendered) + tool output + images
  compaction the first changed message carries the driver's <context_compaction> summary
  image_cap  the first changed message lost images ("[image omitted" placeholder)
  other      any other rewrite of earlier history
Approximations: the chat template's role/tool-call markup is not counted (a few tokens per message); vLLM reuses
only whole 16-token KV blocks; GDN layers need the exact prefix (same rule).
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from success_workload import episode_key  # noqa: E402

IMAGE_TOKENS = 258


def parts(msg: dict) -> tuple[str, int]:
    """(text, images) of one chat message (string or OpenAI content list; tool calls as JSON)."""
    content = msg.get("content")
    text, images = [], 0
    if isinstance(content, str):
        text.append(content)
    elif isinstance(content, list):
        for item in content:
            if item.get("type") in ("image_url", "input_image", "image"):
                images += 1
            else:
                text.append(str(item.get("text", "")))
    for key in ("reasoning_content", "reasoning"):
        if msg.get(key):
            text.append(str(msg[key]))
    if msg.get("tool_calls"):
        text.append(json.dumps(msg["tool_calls"], ensure_ascii=False))
    return "\n".join(text), images


def classify(msg: dict) -> str:
    text, _ = parts(msg)
    if "<context_compaction" in text:
        return "compaction"
    if "[image omitted" in text:
        return "image_cap"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("record")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    cache: dict[str, int] = {}

    def ntok(msg: dict) -> int:
        text, images = parts(msg)
        if text not in cache:
            cache[text] = len(tok(text, add_special_tokens=False)["input_ids"])
        return cache[text] + IMAGE_TOKENS * images

    episodes: dict[str, list[dict]] = collections.OrderedDict()
    for line in Path(a.record).read_text().splitlines():
        if line.strip():
            body = json.loads(line)
            episodes.setdefault(episode_key(body), []).append(body)

    kinds = collections.Counter()
    new_tokens = collections.Counter()
    per_request, prompt_total = [], 0
    for bodies in episodes.values():
        prev: list[dict] | None = None
        for body in bodies:
            msgs = body["messages"]
            total = sum(ntok(m) for m in msgs)
            prompt_total += total
            if prev is None:
                kind, new = "first", total
            else:
                same = 0
                while same < min(len(msgs), len(prev)) and msgs[same] == prev[same]:
                    same += 1
                # the previous request's messages all kept -> append (its answer follows as a new message)
                kind = "append" if same == len(prev) else classify(msgs[same])
                new = sum(ntok(m) for m in msgs[same:])
            kinds[kind] += 1
            new_tokens[kind] += new
            per_request.append(new)
            prev = msgs
    n = len(per_request)
    out = {"episodes": len(episodes), "requests": n, "prompt_tokens_per_request": round(prompt_total / n),
           "new_tokens_per_request": round(sum(per_request) / n),
           "new_tokens_median": statistics.median(per_request),
           "by_kind": {k: {"requests": kinds[k], "new_tokens": new_tokens[k],
                           "share_of_new_tokens": round(new_tokens[k] / sum(new_tokens.values()), 3),
                           "new_tokens_per_request": round(new_tokens[k] / kinds[k])} for k in kinds}}
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
