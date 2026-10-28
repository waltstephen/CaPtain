"""Strict loss samples from original policy requests, with unbiased subsampling."""
from __future__ import annotations

import base64
import hashlib
import io
import math

from PIL import Image
import torch

from prefix_chains import LM_HEAD_CHUNK, prefix_chains  # noqa: F401  (re-exported)

from chat_tokens import NEUTRAL_SAMPLING, validate_sampling_settings


class InvalidRollout(ValueError):
    pass


class OverLengthRollout(InvalidRollout):
    """Over-length guard: raised ONLY by the trainer's per-request length check (a selected request
    longer than the hard cap). A subclass, so every existing handler treats it exactly like InvalidRollout; only the trainer's
    --max-overlength-drops path (overlength_drop_set) may drop such a trajectory instead of refusing the batch."""


def overlength_drop_set(errors: list[dict], max_drops: int) -> set[int] | None:
    """crash fix A: the trajectories (result indices) to drop for the gathered build errors of all ranks (every rank passes the same
    all-gathered list, so every rank gets the same answer). An error is {"index": i, "overlength": bool, ...}. Returns the set of
    indices when EVERY error is an OverLengthRollout and they name at most max_drops distinct trajectories; None = refuse the
    batch (fail-closed as before): any other error, an error without its index, more than max_drops trajectories, or max_drops < 1."""
    if not errors or type(max_drops) is not int or max_drops < 1:
        return None
    if any(e.get("overlength") is not True or type(e.get("index")) is not int for e in errors):
        return None
    drop = {e["index"] for e in errors}
    return drop if len(drop) <= max_drops else None


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InvalidRollout(f"{name} must be a positive integer")
    return value


def validate_rollout(raw: dict) -> tuple[list[dict], int]:
    if raw.get("sample_format") != "per_request_v1":
        raise InvalidRollout("training requires per_request_v1; reconstructed history is forbidden")
    metadata = raw.get("eligible_requests")
    selected = raw.get("training_requests")
    if not isinstance(metadata, list) or not metadata or not isinstance(selected, list) or not selected:
        raise InvalidRollout("missing eligible or selected requests")
    by_id = {}
    for record in metadata:
        index = _positive_int(record.get("request_index"), "request_index")
        if index in by_id:
            raise InvalidRollout(f"duplicate eligible request {index}")
        _positive_int(record.get("prompt_length"), "prompt_length")
        _positive_int(record.get("completion_length"), "completion_length")
        by_id[index] = record
    total = sum(record["completion_length"] for record in metadata)
    if raw.get("full_completion_tokens") != total:
        raise InvalidRollout("full completion-token denominator does not match eligible requests")
    if raw.get("eligible_request_count") != len(metadata) or raw.get("selected_request_count") != len(selected):
        raise InvalidRollout("request count metadata mismatch")
    if len(selected) > len(metadata):
        raise InvalidRollout("selected request count exceeds candidates")
    probability = len(selected) / len(metadata)
    seen = set()
    for record in selected:
        index = _positive_int(record.get("request_index"), "request_index")
        if index not in by_id or index in seen:
            raise InvalidRollout(f"unknown or duplicate selected request {index}")
        seen.add(index)
        expected = by_id[index]
        sampling = record.get("sampling")
        try:
            validate_sampling_settings(sampling)
        except ValueError as exc:
            raise InvalidRollout(f"request {index}: {exc}") from exc
        for ids_name, length_name in (("prompt_token_ids", "prompt_length"), ("token_ids", "completion_length")):
            ids = record.get(ids_name)
            if not isinstance(ids, list) or not ids or any(type(t) is not int or t < 0 for t in ids):
                raise InvalidRollout(f"invalid raw token IDs for request {index}")
            if len(ids) != record.get(length_name) or len(ids) != expected[length_name]:
                raise InvalidRollout(f"{length_name} mismatch for request {index}")
        p = record.get("inclusion_probability")
        weight = record.get("loss_weight")
        if not isinstance(p, (int, float)) or not math.isclose(p, probability, rel_tol=1e-9):
            raise InvalidRollout(f"invalid sampling probability for request {index}")
        if not isinstance(weight, (int, float)) or not math.isfinite(weight) or not math.isclose(weight * p, 1.0, rel_tol=1e-9):
            raise InvalidRollout(f"invalid inclusion weight for request {index}")
    return selected, total


def build_request_samples(raw: dict, processor, *, image_pad_id: int, hard_max_tokens: int,
                          pack_prefixes: bool = False, chains: set[int] | None = None) -> tuple[list[dict], int]:
    """One training sample per selected original request; pack_prefixes: one per exact-prefix chain (prefix_chains), the
    chain's last request carrying every member's target range, inclusion weight and sampler record (token_weights).
    chains (with pack_prefixes): build only these chain indices (pack-level rank assignment; the rollout is still validated
    whole). Every sample carries its chain index as "chain"."""
    selected, total = validate_rollout(raw)
    if not pack_prefixes:
        if chains is not None:
            raise ValueError("chain selection needs pack_prefixes")
        return _request_samples(selected, processor, image_pad_id=image_pad_id, hard_max_tokens=hard_max_tokens), total
    samples = []
    for chain_index, chain in enumerate(prefix_chains(selected, LM_HEAD_CHUNK)):
        if chains is not None and chain_index not in chains:
            continue
        for request in chain[:-1]:          # members are not preprocessed; their image payloads must still match their hashes
            for url, image_hash in zip(request.get("images") or [], request.get("image_hashes") or []):
                if not isinstance(url, str) or hashlib.sha256(url.encode("utf-8")).hexdigest() != image_hash:
                    raise InvalidRollout("original image payload hash differs")
            if len(request.get("images") or []) != len(request.get("image_hashes") or []):
                raise InvalidRollout("image/hash count mismatch")
        (sample,) = _request_samples(chain[-1:], processor, image_pad_id=image_pad_id, hard_max_tokens=hard_max_tokens)
        sample["chain"] = chain_index
        if len(chain) == 1:
            samples.append(sample)
            continue
        mask = torch.zeros(sample["length"], dtype=torch.bool)
        weights, parts, credits, codes = [], [], [], []
        for request in chain:
            start, n = len(request["prompt_token_ids"]), len(request["token_ids"])
            mask[start:start + n] = True
            weights.append(torch.full((n,), float(request["loss_weight"]), dtype=torch.float32))
            credits.append(torch.full((n,), task_credit(request), dtype=torch.float32))
            codes.append(code_tensors(request, n))
            parts.append(sampler_tensors(request.get("sampler_topk"), n))
        packed = dict(sample, mask=mask, n_tokens=int(mask.sum()), token_weights=torch.cat(weights),
                      packed_requests=[r["request_index"] for r in chain])
        packed.pop("token_credit", None)
        if any("task_credit" in r for r in chain):
            packed["token_credit"] = torch.cat(credits)
        packed.pop("token_code", None)
        packed.pop("token_pure_error", None)
        if any(codes):
            if not all(codes):
                raise InvalidRollout("code-reward fields present on only part of a chain")
            packed["token_code"] = torch.cat([c["token_code"] for c in codes])
            packed["token_pure_error"] = torch.cat([c["token_pure_error"] for c in codes])
        for key in ("topk_ids", "topk_logprobs", "sampled_logprobs"):
            packed.pop(key, None)
            if parts and all(key in part for part in parts):
                packed[key] = torch.cat([part[key] for part in parts])
        samples.append(packed)
    return samples, total


def _request_samples(selected: list[dict], processor, *, image_pad_id: int, hard_max_tokens: int) -> list[dict]:
    samples = []
    for request in selected:
        prompt = request["prompt_token_ids"]
        completion = request["token_ids"]
        ids = prompt + completion
        if len(ids) > hard_max_tokens:
            raise OverLengthRollout(f"request {request['request_index']} has {len(ids)} tokens, exceeds {hard_max_tokens}")
        images = []
        urls = request.get("images") or []
        if len(urls) != len(request.get("image_hashes") or []):
            raise InvalidRollout("image/hash count mismatch")
        for url, image_hash in zip(urls, request.get("image_hashes") or []):
            if not isinstance(url, str) or not url.startswith("data:image/") or ";base64," not in url:
                raise InvalidRollout("missing original image data URL")
            if not isinstance(image_hash, str) or hashlib.sha256(url.encode("utf-8")).hexdigest() != image_hash:
                raise InvalidRollout("original image payload hash differs")
            try:
                content = base64.b64decode(url.split(",", 1)[1], validate=True)
                with Image.open(io.BytesIO(content)) as image:
                    images.append(image.convert("RGB"))
            except Exception as exc:
                raise InvalidRollout(f"invalid original image: {exc}") from exc
        mm = {}
        image_tokens = sum(t == image_pad_id for t in ids)
        if images:
            encoded = processor.image_processor(images=images, return_tensors="pt")
            grid = encoded["image_grid_thw"]
            merge = getattr(processor.image_processor, "merge_size", 2)
            expected = int((grid.prod(-1) // (merge * merge)).sum())
            if expected != image_tokens:
                raise InvalidRollout(f"image token count mismatch: {image_tokens} != {expected}")
            mm = {"pixel_values": encoded["pixel_values"], "image_grid_thw": grid}
        elif image_tokens:
            raise InvalidRollout("image tokens present without original images")
        mask = torch.zeros(len(ids), dtype=torch.bool)
        mask[len(prompt):] = True
        samples.append({"input_ids": torch.tensor(ids, dtype=torch.long), "mask": mask,
                        "length": len(ids), "n_tokens": len(completion),
                        "request_index": request["request_index"], "weight": float(request["loss_weight"]), **mm,
                        **sampler_tensors(request.get("sampler_topk"), len(completion)),
                        **({"token_credit": torch.full((len(completion),), task_credit(request), dtype=torch.float32)}
                           if "task_credit" in request else {}),
                        **code_tensors(request, len(completion))})
    return samples


def code_tensors(request: dict, n_tokens: int) -> dict:
    """python_exec_cost_v1 per-token tensors of one request (empty without code-reward fields): token_code = cost / n_tokens
    (the request's token mean, so a long erroring program is not weighted by its length) and token_pure_error = 1 for a
    pure error (no host effect), whose positive task credit the trainer withholds."""
    if "code_cost" not in request:
        return {}
    cost = request.get("code_cost")
    if type(cost) not in (int, float) or not math.isfinite(cost) or not 0.0 <= cost <= 1.0 or n_tokens < 1:
        raise InvalidRollout(f"request {request.get('request_index')}: code_cost must be a number in [0, 1]")
    if type(request.get("code_no_effect")) is not bool:
        raise InvalidRollout(f"request {request.get('request_index')}: code_no_effect must be a boolean")
    return {"token_code": torch.full((n_tokens,), float(cost) / n_tokens, dtype=torch.float32),
            "token_pure_error": torch.full((n_tokens,), 1.0 if request["code_no_effect"] else 0.0, dtype=torch.float32)}


def request_token_values(sample: dict, raw: dict, values: dict[int, float] | None) -> torch.Tensor:
    """A per-request scalar (e.g. the physics critic's request advantage) spread over a built sample's target tokens, in
    target order (a packed chain lists its members in prompt order). values None = a trajectory without values: zeros."""
    lengths = {int(r["request_index"]): len(r["token_ids"]) for r in raw["training_requests"]}
    members = sample.get("packed_requests") or [sample["request_index"]]
    parts = []
    for index in members:
        if values is not None and int(index) not in values:
            raise InvalidRollout(f"request {index} has no request value")
        parts.append(torch.full((lengths[int(index)],), 0.0 if values is None else float(values[int(index)]), dtype=torch.float32))
    out = torch.cat(parts)
    if out.numel() != int(sample["n_tokens"]):
        raise InvalidRollout(f"request values cover {out.numel()} tokens, the sample trains {sample['n_tokens']}")
    return out


def task_credit(request: dict) -> float:
    """The request's multiplier on the task advantage (env server: 0 on a successful trajectory's already-solved suffix)."""
    value = request.get("task_credit", 1.0)
    if type(value) not in (int, float) or value not in (0, 1):
        raise InvalidRollout(f"request {request.get('request_index')}: task_credit must be 0 or 1")
    return float(value)


def sampler_tensors(topk: dict | None, n_tokens: int) -> dict:
    """The sampler's per-token top-k ids/logprobs and sampled-token logprob, aligned with the completion (score centering);
    empty when the rollout carries none or they do not match the completion."""
    if not isinstance(topk, dict):
        return {}
    ids, lps, sampled = topk.get("ids"), topk.get("logprobs"), topk.get("sampled_logprob")
    k = topk.get("k")
    if (type(k) is not int or k < 1 or not isinstance(ids, list) or not isinstance(lps, list) or not isinstance(sampled, list)
            or not len(ids) == len(lps) == len(sampled) == n_tokens or any(len(r) != k for r in ids) or any(len(r) != k for r in lps)):
        return {}
    return {"topk_ids": torch.tensor(ids, dtype=torch.long), "topk_logprobs": torch.tensor(lps, dtype=torch.float32),
            "sampled_logprobs": torch.tensor(sampled, dtype=torch.float32)}


def sampled_objective(logp_sum, kl_sum, *, advantage: float, weight: float,
                      kl_beta: float, full_completion_tokens: int, world_size: int = 1):
    """Horvitz-Thompson numerator and a fixed pre-sampling token denominator."""
    if full_completion_tokens <= 0 or weight <= 0 or not math.isfinite(weight) or not math.isfinite(advantage):
        raise InvalidRollout("invalid objective normalizer or inclusion weight")
    return weight * (-advantage * logp_sum + kl_beta * kl_sum) * world_size / full_completion_tokens
