"""Exact-prefix chains of a trajectory's selected original requests (torch-free: the env server and the trainer share it)."""
from __future__ import annotations

# the trainer applies the LM head to at most this many target positions per call (train.masked_logprobs); under FSDP
# every rank must make the same number of LM-head calls per pass, so a packed chain never exceeds one chunk of targets
LM_HEAD_CHUNK = 4096


def prefix_chains(selected: list[dict], max_targets: int | None = None) -> list[list[dict]]:
    """Group a trajectory's selected requests (request order) into chains in which each request's complete token sequence
    (prompt + completion) and image list are an EXACT prefix of the next one's prompt and images. A causal forward over the
    chain's last request then yields every member's target log-probabilities: no retokenization, no synthetic tokens, no
    truncation; compaction or any re-rendered history starts a new chain. Intervening unselected requests stay context.
    max_targets: a chain's total completion tokens stay <= this (a request that would exceed it starts a new chain)."""
    chains: list[list[dict]] = []
    for request in sorted(selected, key=lambda r: r["request_index"]):
        if chains:
            last = chains[-1][-1]
            ids = last["prompt_token_ids"] + last["token_ids"]
            hashes = list(last.get("image_hashes") or [])
            prompt, images = request["prompt_token_ids"], list(request.get("image_hashes") or [])
            has_topk = [isinstance(r.get("sampler_topk"), dict) for r in (last, request)]
            targets = sum(len(r["token_ids"]) for r in chains[-1]) + len(request["token_ids"])
            if (len(ids) <= len(prompt) and prompt[:len(ids)] == ids and images[:len(hashes)] == hashes
                    and has_topk[0] == has_topk[1] and (max_targets is None or targets <= max_targets)):
                chains[-1].append(request)
                continue
        chains.append([request])
    return chains


def packed_cost(selected: list[dict], quadratic_tokens: float = 140_000.0, max_targets: int | None = LM_HEAD_CHUNK) -> dict:
    """Learner work estimate of a trajectory after packing: one forward/backward per chain over its last request's context.
    cost = sum(n + n^2 / quadratic_tokens) (projections/MLP ~ n, full-attention layers ~ n^2); a scheduling proxy only."""
    lengths = [len(c[-1]["prompt_token_ids"]) + len(c[-1]["token_ids"]) for c in prefix_chains(selected, max_targets)]
    return {"packed_sequences": len(lengths), "packed_context_tokens": sum(lengths),
            "packed_cost": round(sum(n + n * n / quadratic_tokens for n in lengths), 1),
            "packed_lengths": lengths}       # per chain, in chain order: the trainer's pack-level rank assignment
