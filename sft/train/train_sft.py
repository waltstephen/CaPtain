#!/usr/bin/env python3
"""CaPtain SFT trainer: supervised fine-tuning of a vision-language student (Qwen3.5-4B) on CaPtain-Harness agent
trajectories (per-request chat views) plus packed visual-QA rows.

Input: one or more JSONL files, one row per training sample (schema in ../data/FORMAT.md):
  {"id": str, "images": [path, ...], "tools": [...optional function tools...],
   "messages": [{"role": "system"|"user"|"assistant"|"tool", "content": str | [{"type": "text", ...} | {"type": "image", ...}],
                 "reasoning_content"?: str, "tool_calls"?: [...], "tool_call_id"?: str}], "meta": {...}}
Images are referenced in order of appearance of image items.

Loss is computed on assistant turns only: every span from the token after the
``<|im_start|>assistant`` header up to and including its ``<|im_end|>`` is supervised. Per-request samples
(``meta.turn_index`` set) supervise only their last assistant turn, the earlier ones being context that other samples
already cover; prefix-packed rows (``meta.supervise_turns``, ../data/pack_sft_prefix.py) supervise exactly the listed
assistant turns. Works with Qwen2.5-VL / Qwen3-VL / Qwen3.5 style processors (anything whose processor exposes
apply_chat_template(+tools) + image inputs and ChatML special tokens).

Run through launch_sft.sh (torchrun, DeepSpeed ZeRO-3). Use --dry-run to build the dataset and one batch on CPU
without loading the model weights.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

IGNORE_INDEX = -100
PER_TOKEN_EXTRA = ("mm_token_type_ids",)


def load_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        with p.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def per_suite_cap(rows: list[dict], cap: int | None, seed: int = 0) -> list[dict]:
    """Stratified subsample: keep at most ``cap`` samples per ``meta.suite`` (libero_90 has 9x the tasks of the
    other suites and would otherwise dominate an epoch). Whole episodes are kept together: samples are grouped
    by ``meta.episode`` and episodes are drawn at random until the cap is reached."""
    if not cap:
        return rows
    import random
    by_suite: dict[str, dict[str, list[dict]]] = {}
    for row in rows:
        meta = row.get("meta") or {}
        by_suite.setdefault(str(meta.get("suite")), {}).setdefault(str(meta.get("episode") or row.get("id")), []).append(row)
    rng = random.Random(seed)
    kept: list[dict] = []
    for suite in sorted(by_suite):
        episodes = list(by_suite[suite].values())
        rng.shuffle(episodes)
        n = 0
        for samples in episodes:
            if n >= cap:
                break
            kept.extend(samples)
            n += len(samples)
        print(f"[sft] suite {suite}: {n} of {sum(len(s) for s in by_suite[suite].values())} samples kept (cap {cap})")
    return kept


def resolve_image(root: Path | None, rel: str) -> Image.Image:
    p = Path(rel)
    if root is not None and not p.is_absolute():
        p = root / p
    return Image.open(p).convert("RGB")


# ---------------------------------------------------------------------------------------------------------- text-hint mask
# Text-hint mask: with probability p, a row whose supervised turn is a perception/selection decision (it calls mark_point) is
# cut to end at that turn, only that turn is supervised, and every earlier textual position hint (mark_point x/y arguments,
# x_norm/y_norm and pixel-coordinate fields in code and tool outputs) is replaced by a placeholder, so the image is the only
# remaining source of the target point.
import random as _random
import re as _re

MASK_TOKEN = "<masked>"
_HINT_KEYS = _re.compile(r"""((?:["']|\b)(?:x_norm|y_norm|xy_norm|point_norm|pixel_xy|seed_pixel_xy|pixel|center_px|centroid_px|px|uv|"""
                         r"""bbox|bbox_px|bbox_norm|bbox_xyxy)["']?\s*[:=]\s*)(\[[^\]]*\]|-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?)""")


def _mark_arg_spans(text: str) -> list[tuple[int, int]]:
    """Char spans of the x and y positional arguments after the camera-name string of every mark_point( call."""
    out = []
    for m in _re.finditer(r"mark_point\(", text):
        depth, i = 0, m.end()
        while i < len(text):
            c = text[i]
            if c in "([{":
                depth += 1
            elif c in ")]}":
                if depth == 0:
                    break
                depth -= 1
            elif c == "," and depth == 0:
                break
            i += 1
        cam = _re.compile(r"""\s*,\s*(["'])[^"']+\1\s*,""").match(text, i)
        if not cam:
            continue
        j = cam.end()
        for _ in range(2):
            a = _re.compile(r"\s*([^,()\n]+?)\s*(?=[,)])").match(text, j)
            if not a:
                break
            out.append((a.start(1), a.end(1)))
            j = a.end() + 1
    return out


def mask_hints_text(text: str) -> tuple[str, int]:
    if '"required": [' in text or "Examples:" in text:
        return text, 0  # API documentation dumps: their example numbers are not scene hints
    spans = _mark_arg_spans(text)
    n = len(spans)
    for a, b in sorted(spans, reverse=True):
        text = text[:a] + MASK_TOKEN + text[b:]
    text, k = _HINT_KEYS.subn(lambda m: m.group(1) + MASK_TOKEN, text)
    return text, n + k


def _mask_message(m: dict) -> int:
    n = 0
    c = m.get("content")
    if isinstance(c, str):
        m["content"], k = mask_hints_text(c); n += k
    elif isinstance(c, list):
        for part in c:
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                part["text"], k = mask_hints_text(part["text"]); n += k
    if isinstance(m.get("reasoning_content"), str):
        m["reasoning_content"], k = mask_hints_text(m["reasoning_content"]); n += k
    for tc in m.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            fn["arguments"], k = mask_hints_text(args); n += k
        elif isinstance(args, dict):
            for key, v in args.items():
                if isinstance(v, str):
                    args[key], k = mask_hints_text(v); n += k
    return n


def _is_selection_turn(m: dict) -> bool:
    if m.get("role") != "assistant":
        return False
    blob = json.dumps(m.get("tool_calls") or []) + (m.get("content") if isinstance(m.get("content"), str) else "")
    return "mark_point(" in blob


def _count_images(messages: list[dict]) -> int:
    return sum(1 for m in messages if isinstance(m.get("content"), list) for p in m["content"] if p.get("type") == "image")


def maybe_mask_row(row: dict, p: float, rng: _random.Random) -> tuple[dict, bool]:
    """Returns (row', masked). row' supervises exactly one turn (meta.turn_index set) when masked."""
    if p <= 0 or rng.random() >= p:
        return row, False
    msgs = row["messages"]
    meta = row.get("meta") or {}
    a_idx = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
    if meta.get("supervise_turns") is not None:
        allowed = {a_idx[t] for t in meta["supervise_turns"] if t < len(a_idx)}
    elif meta.get("turn_index") is not None and a_idx:
        allowed = {a_idx[-1]}
    else:
        return row, False  # QA / general rows: no agent context to mask
    cands = [i for i in sorted(allowed) if _is_selection_turn(msgs[i])]
    if not cands:
        return row, False
    t = rng.choice(cands)
    import copy
    new_msgs = copy.deepcopy(msgs[: t + 1])
    n = sum(_mask_message(m) for m in new_msgs[:t])
    if n == 0:
        return row, False
    n_img = _count_images(new_msgs)
    new_meta = {k: v for k, v in meta.items() if k != "supervise_turns"}
    new_meta.update(turn_index=len([m for m in new_msgs if m.get("role") == "assistant"]) - 1, text_hints_masked=n)
    return {**row, "messages": new_msgs, "images": list(row.get("images") or [])[:n_img], "meta": new_meta}, True


def assistant_spans(input_ids: torch.Tensor, tokenizer) -> list[tuple[int, int]]:
    """[start, end) index ranges of supervised tokens: after ``<|im_start|>assistant\n`` up to and
    including the matching ``<|im_end|>`` (ChatML). Special tokens are atomic, so this is exact for
    any number of assistant turns, whatever the template did with <think> blocks or tool calls."""
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    ids = input_ids.tolist()
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(ids):
        if ids[i] == im_start and i + 1 < len(ids) and tokenizer.decode(ids[i + 1 : i + 2]).strip() == "assistant":
            j = i + 2
            while j < len(ids) and "\n" not in tokenizer.decode(ids[j : j + 1]):
                j += 1
            start = j + 1
            end = start
            while end < len(ids) and ids[end] != im_end:
                end += 1
            spans.append((start, min(end + 1, len(ids))))
            i = end + 1
        else:
            i += 1
    return spans


class RolloutSFTDataset(Dataset):
    def __init__(self, rows: list[dict], processor, image_root: Path | None, max_length: int, mask_p: float = 0.0, mask_seed: int = 0,
                 qa_loss_weight: float = 1.0):
        self.rows = rows
        self.qa_loss_weight = qa_loss_weight
        self.mask_p = mask_p
        self.mask_seed = mask_seed
        self.processor = processor
        self.image_root = image_root
        self.max_length = max_length
        if max_length < 2:
            raise ValueError("--max-length must be at least 2 for causal supervision")
        if not rows:
            raise ValueError("SFT dataset is empty; check the export filters and --data files")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        if self.mask_p > 0:
            row, _ = maybe_mask_row(row, self.mask_p, _random.Random(self.mask_seed * 1_000_003 + idx))
        images = [resolve_image(self.image_root, p) for p in row.get("images", [])]
        messages = row["messages"]
        tools = row.get("tools") or None
        full = self.processor.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=False)
        # No processor-side truncation: it would cut image placeholder tokens and break the
        # image/token count check. Invalid rows must not substitute a different
        # sample and silently change the sampler's population.
        enc = self.processor(text=[full], images=images or None, return_tensors="pt", padding=False)
        input_ids = enc["input_ids"][0]
        if input_ids.shape[0] > self.max_length:
            raise ValueError(
                f"SFT sample idx={idx} id={row.get('id')!r} encodes to {input_ids.shape[0]} tokens, "
                f"exceeding --max-length={self.max_length}. Re-export with --max-tokens={self.max_length} "
                "and the same processor, or choose a supported larger context. No sample was substituted or truncated."
            )
        attention = enc["attention_mask"][0]
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        spans = assistant_spans(input_ids, self.processor.tokenizer)
        meta = row.get("meta") or {}
        if meta.get("supervise_turns") is not None:
            # prefix-packed row (data/pack_sft_prefix.py): one forward over the longest view of a chain of
            # per-request samples whose rendered texts are exact prefixes of each other; each member's own target turn is supervised
            want = sorted(set(int(t) for t in meta["supervise_turns"]))
            if not want or want[-1] >= len(spans):
                raise ValueError(f"SFT sample idx={idx} id={row.get('id')!r}: supervise_turns {want} vs {len(spans)} assistant turns")
            spans = [spans[t] for t in want]
        elif meta.get("turn_index") is not None and spans:
            spans = spans[-1:]
        for start, end in spans:
            labels[start:end] = input_ids[start:end]
        if not torch.any(labels[1:] != IGNORE_INDEX):
            raise ValueError(f"SFT sample idx={idx} id={row.get('id')!r} has no supervised next-token targets; check its assistant turn and chat template")
        per_token = {k: enc[k][0] for k in PER_TOKEN_EXTRA if k in enc}  # e.g. mm_token_type_ids (Qwen3-VL M-RoPE)
        item = {k: v for k, v in enc.items() if k not in ("input_ids", "attention_mask", *PER_TOKEN_EXTRA)}
        item["input_ids"] = input_ids
        item["attention_mask"] = attention
        item["labels"] = labels
        item.update(per_token)
        # per-token loss weight of this row: meta.loss_weight if the mix set one (e.g. 2.0 for up-weighted demonstrations), else packed visual-QA /
        # general rows get qa_loss_weight and agent rows 1.0
        w = meta.get("loss_weight")
        if w is None and self.qa_loss_weight != 1.0:
            w = self.qa_loss_weight if meta.get("source") in ("qapack", "auxqa", "general") else 1.0
        if w is not None and float(w) != 1.0:
            item["loss_weight"] = torch.tensor(float(w), dtype=torch.float32)
        return item


@dataclass
class Collator:
    pad_token_id: int

    def __call__(self, batch: list[dict]) -> dict:
        out: dict = {}
        max_len = max(b["input_ids"].shape[0] for b in batch)
        ids, att, lab = [], [], []
        for b in batch:
            n = b["input_ids"].shape[0]
            pad = max_len - n
            ids.append(torch.cat([b["input_ids"], torch.full((pad,), self.pad_token_id, dtype=torch.long)]))
            att.append(torch.cat([b["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
            lab.append(torch.cat([b["labels"], torch.full((pad,), IGNORE_INDEX, dtype=torch.long)]))
        out["input_ids"] = torch.stack(ids)
        out["attention_mask"] = torch.stack(att)
        out["labels"] = torch.stack(lab)
        if all("loss_weight" in b for b in batch):
            out["loss_weight"] = torch.stack([b["loss_weight"] for b in batch])
        for key in PER_TOKEN_EXTRA:
            if all(key in b for b in batch):
                out[key] = torch.stack([torch.cat([b[key], torch.zeros(max_len - b[key].shape[0], dtype=b[key].dtype)]) for b in batch])
        # vision tensors: concatenate along the patch axis (Qwen-VL convention)
        for key in ("pixel_values", "image_grid_thw"):
            vals = [b[key] for b in batch if key in b]
            if vals:
                out[key] = torch.cat(vals, dim=0)
        return out


def trainer_loss_scale(trainer, num_items_in_batch) -> int:
    """Match HF Trainer's global-token compensation without changing its count policy."""
    uses_counts = bool(trainer.model_accepts_loss_kwargs or trainer.compute_loss_func)
    if not (trainer.args.average_tokens_across_devices and uses_counts and num_items_in_batch is not None):
        return 1
    scale = trainer.accelerator.num_processes
    parallelism = getattr(trainer.accelerator, "parallelism_config", None)
    if parallelism is not None:
        scale //= parallelism.tp_size
    return trainer.args.n_gpu if trainer.args.n_gpu > 1 else scale


def selective_causal_loss(model, inputs, *, num_items_in_batch=None, loss_scale=1):
    """Compute only supervised logits, preserving Trainer's token denominator."""
    labels = inputs.pop("labels")
    weight = inputs.pop("loss_weight", None)
    if labels.ndim != 2 or labels.shape[0] != 1:
        raise ValueError("selective loss requires per_device_train_batch_size=1")
    target = labels[0, 1:]
    keep = torch.nonzero(target != IGNORE_INDEX, as_tuple=False).squeeze(-1)
    if keep.numel() == 0:
        raise ValueError("SFT batch has no supervised next-token targets")
    out = model(**inputs, use_cache=False, logits_to_keep=keep)
    loss = torch.nn.functional.cross_entropy(out.logits[0].float(), target[keep], reduction="sum")
    if weight is not None:
        loss = loss * weight.reshape(-1)[0].to(loss.device, loss.dtype)
    if num_items_in_batch is None:
        denominator = keep.numel()
    else:
        denominator = torch.as_tensor(num_items_in_batch, device=loss.device, dtype=loss.dtype)
        if denominator.numel() != 1 or not torch.isfinite(denominator).all() or (denominator <= 0).any():
            raise ValueError("num_items_in_batch must be a positive finite token count")
        denominator = denominator.reshape(())
    return loss * loss_scale / denominator, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model dir (e.g. the untied Qwen3.5-4B, see scripts/untie_base.py)")
    ap.add_argument("--data", nargs="+", type=Path, required=True)
    ap.add_argument("--image-root", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--max-length", type=int, default=16384)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--per-device-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--deepspeed", type=str, default=None)
    ap.add_argument("--freeze-vision", action="store_true")
    ap.add_argument("--per-suite-cap", type=int, default=None, help="stratified subsample: at most N samples per meta.suite (whole episodes)")
    ap.add_argument("--subsample-seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mask-text-hints", type=float, default=0.0,
                    help="probability of the text-shortcut mask on rows whose supervised turn calls mark_point (default off)")
    ap.add_argument("--qa-loss-weight", type=float, default=1.0, help="per-token loss weight of packed QA / general rows (agent rows 1.0)")
    ap.add_argument("--loss-tokens-per-step", type=float, default=0.0,
                    help="C > 0: loss = sum(token NLL) / C on every step (constant global per-token weight; requires grad-accum 1)")
    args = ap.parse_args()
    if args.loss_tokens_per_step > 0 and args.grad_accum != 1:
        raise SystemExit("--loss-tokens-per-step requires --grad-accum 1")

    fraction = os.environ.get("SFT_MEMORY_FRACTION")
    if fraction and torch.cuda.is_available():
        # sharing a card with other (idle) processes: never take more than this share of its memory
        torch.cuda.set_per_process_memory_fraction(float(fraction), int(os.environ.get("LOCAL_RANK", 0)))
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    rows = per_suite_cap(load_rows(args.data), args.per_suite_cap, args.subsample_seed)
    ds = RolloutSFTDataset(rows, processor, args.image_root, args.max_length, mask_p=args.mask_text_hints, mask_seed=args.subsample_seed,
                           qa_loss_weight=args.qa_loss_weight)
    tok = processor.tokenizer
    collator = Collator(pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    print(f"[sft] {len(ds)} samples from {len(args.data)} file(s)")

    if args.dry_run:
        if args.mask_text_hints > 0:
            rng = _random.Random(0)
            stats = {"rows": 0, "masked": 0, "hints": 0}
            for r in rows[:2000]:
                r2, ok = maybe_mask_row(r, args.mask_text_hints, rng)
                stats["rows"] += 1; stats["masked"] += ok; stats["hints"] += (r2.get("meta") or {}).get("text_hints_masked", 0)
            print("[sft] dry-run mask stats (first 2000 rows):", stats)
        batch = collator([ds[i] for i in range(min(2, len(ds)))])
        n_sup = int((batch["labels"] != IGNORE_INDEX).sum())
        print("[sft] dry-run batch:", {k: tuple(v.shape) for k, v in batch.items()}, "supervised tokens:", n_sup)
        return

    from transformers import AutoModelForImageTextToText, Trainer, TrainingArguments

    class SelectiveLossTrainer(Trainer):
        """Compute lm_head + cross-entropy only at supervised positions via `logits_to_keep`.

        Full logits ([T, vocab]) for a 32k-token sample with a 248k vocab cost ~16 GB bf16 plus a
        32 GB float32 copy inside the stock loss, which OOMs an 80 GB card even under ZeRO-3.
        Supervised (assistant) tokens are a few thousand per sample. We stay on the wrapped
        model's forward so DeepSpeed ZeRO-3 parameter gathering keeps working; requires
        per-device batch size 1 (index tensors are per-sequence).
        """

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if args.loss_tokens_per_step > 0:
                # global per-token weighting ACROSS steps: every supervised token of the run weighs 1/C (C = mean supervised tokens
                # per optimizer step over the whole data), so a step with few tokens (short answers) gives a proportionally
                # small update. Gradients are averaged over ranks by the engine -> multiply by the world size.
                loss, out = selective_causal_loss(model, inputs, num_items_in_batch=args.loss_tokens_per_step,
                                                  loss_scale=self.accelerator.num_processes)
            else:
                loss, out = selective_causal_loss(model, inputs, num_items_in_batch=num_items_in_batch,
                                                  loss_scale=trainer_loss_scale(self, num_items_in_batch))
            return (loss, out) if return_outputs else loss

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa")
    if args.freeze_vision:
        for name, p in model.named_parameters():
            if "visual" in name or "vision" in name:
                p.requires_grad_(False)
    targs = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        bf16=True,
        gradient_checkpointing=True,
        # 40 GB GPUs: SFT_GC_OFFLOAD=1 keeps the per-layer checkpointed activations in pinned CPU memory
        # (transformers save_on_cpu around the same non-reentrant checkpoint); numerics unchanged.
        gradient_checkpointing_kwargs=({"offload": True} if os.environ.get("SFT_GC_OFFLOAD") == "1" else None),
        logging_steps=1,
        # SFT_SAVE_STEPS=N: periodic full (resumable) checkpoints instead of only at the epoch end; SFT_RESUME=<dir|1> resumes.
        save_strategy="steps" if os.environ.get("SFT_SAVE_STEPS") else "epoch",
        save_steps=int(os.environ.get("SFT_SAVE_STEPS") or 500),
        save_total_limit=2,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        weight_decay=0.0,
        # every supervised token of the whole global batch weighs the same (loss sum / global supervised-token count)
        average_tokens_across_devices=True,
        deepspeed=args.deepspeed,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        # SFT_SAMPLING=sequential: keep the file order (bucket_order.py / mix_order.py write world-size groups of similar length in random
        # group order, so a ZeRO-3 step does not wait for one 65k row while the other ranks hold short ones)
        train_sampling_strategy=os.environ.get("SFT_SAMPLING", "random"),
        report_to=[r for r in ("wandb",) if os.environ.get("WANDB_API_KEY")],
        run_name=args.output_dir.name,
    )
    callbacks = []
    trainer = SelectiveLossTrainer(model=model, args=targs, train_dataset=ds, data_collator=collator, callbacks=callbacks)
    resume = os.environ.get("SFT_RESUME") or None
    trainer.train(resume_from_checkpoint=True if resume == "1" else resume)
    trainer.save_model(str(args.output_dir / "final"))
    processor.save_pretrained(str(args.output_dir / "final"))


if __name__ == "__main__":
    main()
