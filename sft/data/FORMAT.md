# CaPtain SFT data format

`train/train_sft.py` reads JSONL files with one training row per line. Image paths are relative to `--image-root`, unless they are absolute.

```json
{
  "id": "<episode>#req-012#pack11",
  "images": ["images/<episode>__obs_001_head.png", "..."],
  "tools": [{"type": "function", "function": {"name": "python", "description": "...", "parameters": {"...": "..."}}}],
  "messages": [
    {"role": "system", "content": "<harness system prompt>"},
    {"role": "user", "content": "<environment context>"},
    {"role": "user", "content": [{"type": "image", "image": "images/...png"}, {"type": "text", "text": "Read TASK.md ..."}]},
    {"role": "assistant", "content": "<visible text>", "reasoning_content": "<thinking>",
     "tool_calls": [{"id": "call_1", "type": "function",
                     "function": {"name": "python", "arguments": {"input": "<python source>"}}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "Cell 1 completed in 4.8 s\nOutput:\n..."}
  ],
  "meta": {"episode": "...", "request_index": 12, "turn_index": 11, "supervise_turns": [1, 2, 11],
           "total_tokens": 48442, "suite": "libero_90", "source": "agent", "loss_weight": 2.0}
}
```

## Rendering and supervision

- The row is rendered with the processor's chat template, `apply_chat_template(messages, tools=tools)`, so the Qwen3.5 template is used. `reasoning_content` becomes the `<think>` block. Images are bound in order of appearance.
- **Supervised tokens.** Only assistant spans are supervised, from the token after `<|im_start|>assistant\n` up to and including `<|im_end|>`.
  - `meta.supervise_turns` (list of assistant-turn ordinals) supervises exactly those turns. `data/pack_sft_prefix.py` writes these rows: consecutive per-request views of one episode are packed into their longest view when they are exact text and image prefixes of each other.
  - Otherwise, `meta.turn_index` (set on per-request views) supervises only the last assistant turn. Earlier turns are context.
  - Otherwise every assistant turn is supervised. Visual-QA rows use this mode.
- **Length.** Rows are never truncated. A row longer than `--max-length` stops training with an error. Export rows with the same processor and keep them at or below the training context (65,536 tokens).
- **Per-row loss weight** comes from the first rule that applies:
  1. `meta.loss_weight`, if set (2.0 for up-weighted side-approach and pivot re-plan demonstrations);
  2. `--qa-loss-weight`, for rows with `meta.source` in `qapack`, `auxqa` or `general`;
  3. 1.0 otherwise.
- **Constant global per-token normaliser** (`--loss-tokens-per-step C`). The loss of a step is Σ(weight × token NLL) / C, multiplied by the world size; the engine then averages over ranks. Every supervised token therefore has the same weight for the whole run.
  - C is the mean weighted supervised-token count per optimizer step over the whole data.
  - Compute it with `QA_W=0.5 python train/sup_tokens.py <data.jsonl> <model dir> <world size>`; the value is `C_weighted_per_step`.
- **Text-hint mask** (`--mask-text-hints p`). With probability p, a row whose supervised turn calls `mark_point(...)` is changed as follows:
  - it is cut to end at that turn, and only that turn is supervised;
  - every earlier textual pixel hint is replaced by `<masked>`. Hints are the `mark_point` x/y arguments and `x_norm`, `y_norm`, `pixel`, `bbox` and similar fields in code and tool outputs.

  The image is then the only source of the target point.

## Step order

`SFT_SAMPLING=sequential` keeps the file order. Two tools write that order, both with one row per rank per optimizer step:

- `train/mix_order.py --world W` (used for CaPtain). Every step mixes agent and QA rows in the global ratio. Within each type, rows are length-sorted, so the lengths within a step are similar. The step order is shuffled.
- `train/bucket_order.py --world W`. Steps are pure length buckets in random order.

Both need `meta.total_tokens` and `meta.source` (`qapack` marks packed QA rows).

## Helper-library code normalisation (`data/normalize/`)

All agent rows in the stage-2 mix were rewritten to one small helper library. The library, `stdlib.PREAMBLE`, defines `val`, `need`, `downq` and `go` (preview, then `move_to`). It is prepended to the first python cell of each episode, so the model learns to write the library itself in its first request. No harness change is needed.

`normalize.py` handles the teacher's own helpers as follows:
- It checks each helper definition against a recording mock robot, per call site.
- If the definition is behaviourally equivalent to a standard helper, it is deleted and its calls are rewritten to the standard helper with explicit arguments.
- If a non-equivalent definition uses a standard name, it is renamed to `<name>_custom`.

`verify.py` runs static checks on the result.

```bash
python data/normalize/normalize.py --out-dir norm/ --report norm/report.json --workers 16 raw/*.jsonl
python data/normalize/verify.py norm/*.jsonl
```

## Stage-2 data filters (applied before mixing; documented, script not released)

The filters depend on host-side episode labels that are not part of this release:
- **Excluded tasks.** libero_10 task 5 is excluded: its success predicate has a 3.5 mm z offset.
- **Drawer task.** For libero_goal task 0, only episodes whose drawer ends pulled ≥ 0.1595 m are kept.
- **Unsupervised requests.** These requests stay as context of later rows but are not supervised:
  - requests after the first completion-gate rejection on libero_10 task 8;
  - "struggle" segments (retry loops of failed actions), about 3.6% of requests;
  - rows whose target cell introspects a helper (`meta.nosup_reason`).
- **Narrow-layout demonstrations.** These are capped at 10 per task when ≥ 20 wide-layout demonstrations exist.
- **RL self-successes.** These must meet all of: success, `finish()` called, no false finish, ≤ 3 failed cells, and no evaluation states. One is kept per (task, start).
