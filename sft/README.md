# CaPtain SFT

This directory has the supervised fine-tuning code that turns **Qwen/Qwen3.5-4B** into **CaPtain-4B-SFT**. CaPtain-4B-SFT is a code-as-policy agent that controls a robot by writing Python against the CaPtain-Harness tool API. It is the initialisation and the KL reference of the RL stage (`../rl/`).

```
sft/
  train/train_sft.py        trainer (HF Trainer + DeepSpeed ZeRO-3, supervised-position-only logits, text-hint mask,
                            per-row loss weights, constant global per-token normaliser)
  train/launch_sft.sh       multi-node torchrun launcher (env-var driven; same command on every node)
  train/ds_zero3.json       DeepSpeed config used (ds_zero3_offload.json: CPU-offloaded optimizer variant, not used)
  train/mix_order.py        per-step agent/QA mixing order for SFT_SAMPLING=sequential (used)
  train/bucket_order.py     alternative: length-bucketed step order
  train/sup_tokens.py       exact supervised-token count -> constant C for --loss-tokens-per-step
  train/test_token_loss.py  CPU unit test of the loss normalisation and row weights
  scripts/untie_base.py     untie lm_head from the embeddings of the base model (required by the RL trainer)
  configs/captain-4b-sft.yaml   the full two-stage configuration
  data/FORMAT.md            training-row schema, supervision rules, filters
  data/pack_sft_prefix.py   prefix-pack per-request views of an episode into one row (meta.supervise_turns)
  data/normalize/           helper-library code normalisation (stdlib.py preamble, normalize.py, verify.py)
```

## Install

The run used torch 2.10.0+cu128, transformers 5.17.0, deepspeed 0.19.6 and flash-linear-attention 0.5.2. Qwen3.5 is a hybrid Gated-DeltaNet model; it needs transformers with Qwen3.5 support (≥ 5.x), flash-linear-attention and causal-conv1d. causal-conv1d ships prebuilt wheels for torch 2.10.

```bash
python -m venv .venv-sft && . .venv-sft/bin/activate
pip install "torch==2.10.0" --index-url https://download.pytorch.org/whl/cu128
pip install "transformers==5.17.0" "deepspeed==0.19.6" "flash-linear-attention==0.5.2" causal-conv1d accelerate safetensors pillow
# optional: tilelang (on Hopper, if the Triton Gated-DeltaNet backward kernels fail in your Triton version)
```

## Data

The training mix is not released yet (about 861M tokens with images). Every row follows `data/FORMAT.md`: chat `messages` (system, user, assistant with `reasoning_content` and `python` tool calls, and tool results), image paths, the tool schema, and `meta`, which says which assistant turns are supervised and with what weight.

Stage-2 mix: 25,961 rows / 861.1M tokens, built from harness episodes.

| Source | Rows | Tokens | Share |
|---|---|---|---|
| Stage-1 agent data (code-normalised, hard tasks cleaned) | 7,903 | 348.9M | 40.5% |
| Hard-task teacher demonstrations, round 3 | 5,090 | 238.4M | 27.7% |
| Hard-task teacher demonstrations, round 2 | 2,138 | 99.3M | 11.5% |
| Hard-task teacher demonstrations, 60-request cap (≤ 10 per task) | 925 | 43.4M | 5.0% |
| RL self-successes (one per task and start) | 1,386 | 52.3M | 6.1% |
| Visual QA (loss weight 0.5) | 8,519 | 78.8M | 9.2% |

- The agent trajectories are synthetic. Most come from proprietary frontier teacher models running in CaPtain-Harness on LIBERO, with the same tools, context policy and caps as the student. The rest are the student's own RL successes.
- All agent code was normalised to one helper library (`data/normalize/`).
- The filters are listed in `data/FORMAT.md`.

## Train

SFT has two stages (`configs/captain-4b-sft.yaml`). Both use peak LR 1e-5 with 30 warmup steps and a cosine schedule, a frozen vision tower, text-hint mask p = 0.2, per-device batch 1 with no accumulation (16 ranks), bf16, gradient checkpointing, grad clip 1.0 and ZeRO-3. Only assistant turns are supervised.

```bash
# 0. untie the base model
python scripts/untie_base.py <Qwen3.5-4B dir> models/qwen35-4b-untied

# 1. order the rows for the world size and compute the loss constant C (CPU)
python train/mix_order.py --inp data/sft.jsonl --out data/sft_mix16.jsonl --world 16
QA_W=0.5 python train/sup_tokens.py data/sft_mix16.jsonl models/qwen35-4b-untied 16   # -> C_weighted_per_step

# 2. stage 1: from the untied base on the stage-1 mix; keep the step-1,700 checkpoint
#    (2-epoch cosine schedule, stopped early; standard global-batch token mean, all rows weight 1)
#    launch_sft.sh always passes --loss-tokens-per-step, so for stage 1 call train/train_sft.py directly with the
#    stage-1 values in configs/captain-4b-sft.yaml

# 3. stage 2 (= CaPtain-4B-SFT): from the stage-1 checkpoint, 1.5 epochs = 2,435 steps, C_w = 13,177.1, QA weight 0.5
NNODES=2 NODE_RANK=0 MASTER_ADDR=<node-0 address> MODEL=runs/stage1/checkpoint-1700 \
  DATA=data/sft_mix16.jsonl IMAGE_ROOT=data C=13177.1 RUN=captain-4b-sft bash train/launch_sft.sh
# ... and the same with NODE_RANK=1 on the second node
```

- The final model is written to `runs/captain-4b-sft/final`.
- Resume with `SFT_RESUME=1`, or with `SFT_RESUME=<checkpoint dir>`.
- A different world size changes both the step order and C: rerun `mix_order.py --world W` and `sup_tokens.py ... W`.
- Mid-training checkpoints are much worse than the final model. In stage 2, step 800 solved 12 of 80 held-out episodes (15%), and 43% of its code cells raised exceptions. The schedule recovers late in the cosine decay, so do not judge the run early.

Unit test (CPU only):
```bash
cd train && CUDA_VISIBLE_DEVICES= python test_token_loss.py   # prints TOKEN_LOSS_TEST_OK
```

## Hardware

- **Stage 2:** 16 × H100-80GB (2 nodes × 8), about 6–7 s per step, about 7.6 h wall time, about 120 H100-hours. Peak memory was ≤ 50 GB per GPU.
- **40 GB GPUs** (A100-40GB) need `SFT_GC_OFFLOAD=1`. This keeps checkpointed activations in pinned CPU memory, so 65k-token rows fit in 40 GB. Also set `OMP_NUM_THREADS=8`.
- The RAM must hold the ZeRO-3 states of a 4B model plus image decoding (4 dataloader workers per rank).

## Serve and evaluate

The `final/` directory is a standard Hugging Face model directory (`Qwen3_5ForConditionalGeneration`, bf16, untied `lm_head`).

- To serve it with vLLM and run LIBERO episodes in CaPtain-Harness, follow `../harness/README.md`. It uses the same command as for the released `CaPtain-4B-SFT` weights.
- RL from this checkpoint: `../rl/README.md`.

Released checkpoint results:

| Benchmark | Host success |
|---|---|
| LIBERO, official held-out states 11–16 (240 episodes) | 55.4% |
| LIBERO-PRO Pos+Task (6 cells) | 59.5 |
| LIBERO-Plus | 48.6 |
