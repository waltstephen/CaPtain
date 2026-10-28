# CaPtain RL

This directory holds the reinforcement-learning stage of **CaPtain**, the pipeline from "CaPtain: A Pipeline for Post-Training Code-as-Policy Robot Agents".
- It starts from **CaPtain-4B-SFT** and produces **CaPtain-4B**: round 300 of the reference run, 75 optimizer updates.
- The policy acts only through **CaPtain-Harness** (`../harness`): the same python tool, robot API, receipts and completion gate as in evaluation.

What the RL stage adds on top of plain outcome RL:

- **Simulator critic.** A value function V(s, t) reads *privileged* simulator state, recorded by the harness host before every model request (`hidden_request_marks.jsonl`).
  - Inputs: 36 physics features, the request index t, and a 40-way task one-hot.
  - It is never shown to the policy.
  - Architecture: 5 × MLP 3×256, members' logits averaged.
  - It starts from an offline warm start and is co-trained online (SAO-style; 2 critic updates per round).
  - Code: `critic_features.py`, `physics_critic.py`, `critic_cotrain.py`.
- **Advantage.** For episode *i* with live requests t = 1..T and immediate costs c_t ≤ 0, the value difference is taken at a fixed time index ("same-t"):
  - δ_t = c_t + V(s_{t+1}, t) − V(s_t, t), and δ_T = c_T + R_i − V(s_T, T).
  - A^GAE_t = Σ_k λ^k δ_{t+k}, with λ = 0.9.
  - Group-relative outcome over G = 4 rollouts of the same start: A_{i,t} = (G_i − mean_group G) + (A^GAE_{i,t} − mean_t A^GAE_{i,·}), where G_i = R_i + Σ_t c_t.
  - The result is scaled by 1/σ, where σ is an EMA of earlier rounds' advantage std. There is no mean-centering.
- **SAO-style update.**
  - Double-sided ratio mask: a token is kept iff 0.2 < π_θ/π_rollout < 4.0.
  - Score-centering correction (sampler top-k 20).
  - k3 KL to CaPtain-4B-SFT (β = 0.002), computed on the kept tokens only.
  - Trajectory-mean loss normalisation; ≤ 8 trained requests per trajectory (inclusion-weighted).
- **Rewards.**

  | Outcome | Return |
  |---|---|
  | Success + `finish()` | +1 |
  | Success without `finish()` | +0.25 |
  | False finish | −0.3 |
  | Budget exhausted | −0.1 |
  | Otherwise | 0 |

  - Action costs: 0.008 per physical action beyond a per-task reference, and 0.02 per failed action.
  - Post-success costs (0.005 per request, capped at 0.45).
  - A code-execution cost (weight 0.1).
  - A **staged one-shot subgoal reward** on 8 hard tasks (`configs/staged-reward-spec.json`).
- **Co-critic early stop.**
  - An episode whose V stays below 0.02 for 5 requests is cut. A 5% control slice is never cut.
  - The 8 hard tasks are exempt from the critic cut.
  - The `success_k` rule still applies to every task: after 15 consecutive requests with the goal already satisfied, the episode is settled as a missed finish.

## Layout

| Path | Contents |
|---|---|
| `fresh_run.py` | Run lifecycle: `init` (manifest + source hashes), `services` (env server + policy proxy on a rollout node), `train`, `status`, `stop`, `relocate` (layout changes of a live run). |
| `train.py` | Streaming trainer: FSDP2 full fine-tune with fp32 master weights; KL reference; critic co-training; gradient accumulation; checkpoint export; NCCL weight refit. |
| `env_server.py` | Rollout service. It runs one `harness/scripts/react_episode.py` episode per request against the policy proxy, scores it and writes the training sample. Also hosts the early-stop monitor. |
| `policy_proxy.py` | Recording proxy in front of the policy vLLMs: token ids, images, sampler top-k log-probs, and gating during weight swaps. |
| `physics_critic.py`, `critic_cotrain.py`, `critic_features.py` | Simulator critic: features, warm-start loading, GAE / group-relative advantages, rewards and costs, staged ladders. |
| `raw_request_training.py`, `prefix_chains.py`, `prefix_tree.py`, `chat_tokens.py` | Exact per-request training samples (prompt + sampled completion), prefix packing, assistant-token masks. |
| `segment_bank.py`, `segment_curriculum.py`, `fresh_start_bank.py` | Training-start pool: fresh initial states only (seeds ≥ 1000, never the evaluation states 11–16). Uniform sampling without replacement, plus the start-exclude list. |
| `code_reward.py`, `reward.py`, `reward_segment.py` | Code-execution cost, episode verdict and logged shaped score. |
| `grad_accum.py`, `stream_pool.py`, `infra_pipeline.py`, `rollout_storage.py`, `training_resume.py`, `attn_backends.py`, `nccl_refit.py`, `vllm_worker_ext.py`, `toolchain_preflight.py` | Infrastructure: accumulation, streaming pool, storage, resume checks, FA3, refit, sandbox preflight. |
| `make_start_weights.py` | Optional prioritized start sampling (not used by the reference config). |
| `configs/captain-4b.json` | Run config of the reference run: 16 GPUs = trainer node + rollout node. |
| `configs/captain-4b.8gpu.json` | The same training semantics on one 8-GPU node. |
| `configs/staged-reward-spec.json` | Hard-task ladders and early-stop exemptions, as in the config. |
| `configs/act-ref.json` | Per-task action reference for the action cost (1.5 × median physical actions of successful demonstrations). |
| `configs/start-exclude-thin-basket.json` | 649 libero_object starts never drawn. In these starts the LIBERO basket region is too thin, so a correct placement would score as failure. |
| `configs/start-pool.reference.json` | The reference run's start pool: 40 tasks × 184 starts, with per-file sha256 and bank sha256. |
| `configs/official-heldout-states.json` | Per task, the official LIBERO init-state indices used for training (0–10, 17–49) and the held-out evaluation indices (11–16). |
| `scripts/` | `env.sh` (common env), `rl_run.sh` (lifecycle on one node), `policies.sh` (policy vLLMs), `run_stream.sh` (supervisor), `prepare_starts.sh`, `export_checkpoint.sh`, `verify_dcp.py`. |
| `tests/` | CPU unit tests of the production path (`python -m pytest -q rl/tests`). |

## Data flow

```
                    rollout node                                                     trainer node
 SAM3 :8114 + Molmo2-8B :8122 (harness/services)
 policy vLLMs :8500.. (CaPtain weights, NCCL weight-transfer)  <--- NCCL refit ---  train.py (FSDP2, 8 ranks)
        ^                                                                               ^      |
 policy_proxy.py :8512 (records tokens/images/top-k, gates weight swaps)                |      | co-critic weights
        ^                                                                               |      v (cocritic_live.pt)
 env_server.py :8402  -- 80 concurrent CaPtain-Harness episodes (LIBERO on EGL)  --  samples + privileged marks
        (early stop: co-critic V on live marks; refuses the next request when hopeless)
```

Each round collects 256 episodes, as 64 starts × G = 4 rollouts of the same start. K = 4 rounds of gradients are accumulated per optimizer step, so one update covers 1,024 episodes. Rollouts stream continuously. A request generated by a policy more than 2 versions older than the current one gets no policy-gradient credit.

## Install

Environments of the reference run (Linux, NVIDIA H100):

| Env (`$CAPTAIN_WORKSPACE/envs/...`) | Used for | Key packages |
|---|---|---|
| `train` (`python_train`) | `train.py` | Python ≥ 3.10, torch 2.10.0+cu128, transformers 5.17.0, flash-linear-attention, causal-conv1d (torch 2.10 wheel), tilelang, safetensors, peft (imported, not used for full FT). Optional: FA3 from kernels-community/flash-attn3, as an overlay dir (`attn_overlay`). |
| `vllm` | policy servers | vLLM 0.28.0–0.29.0 (Qwen3.5 support, `--weight-transfer-config`), FlashInfer |
| `libero` (`python_env`) | `fresh_run.py`, `env_server.py`, `policy_proxy.py` | Python 3.10, fastapi, uvicorn, requests, numpy, torch (CPU is enough) |
| `harness/.venv-libero` | the episodes themselves | See `../harness/README.md`. The env server runs `<harness>/.venv-libero/bin/python scripts/react_episode.py`, so the venv (or a symlink to it) must live inside the harness checkout. |
| `student-py311` (`student_python`) | the model's own Python session in the sandbox | Python 3.11.16, numpy 1.26.4, scipy 1.13.1, Pillow 10.4.0 (checked before each episode) |

Other requirements:
- `bwrap` ≥ 0.5 (sandbox).
- EGL (`__EGL_VENDOR_LIBRARY_FILENAMES`).
- A shared filesystem for `$CAPTAIN_WORKSPACE`, mounted at the same path on all nodes.
- The perception services from `../harness/services/start_perception.sh` (SAM3 checkpoint, Molmo2-8B).

### Models and the critic warm start

```bash
export CAPTAIN_WORKSPACE=/shared/captain CAPTAIN_ROOT=/path/to/CaPtain
huggingface-cli download <hf-user>/CaPtain-4B-SFT --local-dir $CAPTAIN_WORKSPACE/models/CaPtain-4B-SFT
huggingface-cli download <hf-user>/CaPtain-artifacts --repo-type dataset --include 'critic/*' --local-dir $CAPTAIN_WORKSPACE
#  -> $CAPTAIN_WORKSPACE/critic/warm_start/{critic_default.pt, critic_split.pt, feature_spec.json, metrics_sameT.json}
```

The warm-start critic is an offline fit on "phase-0" rollouts of an earlier SFT student, all from fresh initial states, before RL. It is trained on the 0/1 host success; `feature_spec.json` documents the features, the standardisation and the same-t rule. `critic_dir` points at this directory.

### Training starts

Training draws only fresh initial states (harness seeds ≥ 1000, where seed 1000 + k selects state k of the harness's
`data/libero_fresh_init/`). The official evaluation states (seeds 11–16 → init states 11–16) can never be drawn.

**Exact reproduction (recommended).** The reference run used 7,360 starts: 184 per task = 140 wide-layout states with
lighting/camera/material variation + 44 official non-evaluation states (indices 0–10, 17–49). They are in
`CaPtain-artifacts/starts/libero_fresh_init_rl/`. Install them as the harness's fresh-init set, then build the pool:

```bash
huggingface-cli download <hf-user>/CaPtain-artifacts --repo-type dataset --include 'starts/*' --local-dir $CAPTAIN_WORKSPACE
mv $CAPTAIN_ROOT/harness/data/libero_fresh_init $CAPTAIN_ROOT/harness/data/libero_fresh_init.bundled
cp -r $CAPTAIN_WORKSPACE/starts/libero_fresh_init_rl $CAPTAIN_ROOT/harness/data/libero_fresh_init
bash rl/scripts/prepare_starts.sh       # -> $CAPTAIN_WORKSPACE/configs/start-bank.json, start-pool.json
# prints {"states": 7360, "tasks": 40, ..., "bank_sha256": "df69fab6b8ca1aa5ac2ea98c93aebab95d7360d2eb25a6a11fbfc406f0d2a5e9"}
```

The printed `bank_sha256` must equal `configs/start-pool.reference.json["bank_sha256"]`. This was verified for the
released files. The thin-basket exclude list (`start_exclude`) refers to starts of this pool.

**With the harness's bundled `data/libero_fresh_init`** (3,943 states, no visual variation), the pipeline also runs. In that case,
set `start_exclude` to `null`, because the exclude list names starts of the reference pool.

## Running

All config values may use `${VAR}` references; `fresh_run.load_config` expands them and refuses unresolved ones. Export these on every node:

```bash
export CAPTAIN_ROOT=/path/to/CaPtain CAPTAIN_WORKSPACE=/shared/captain CAPTAIN_NCCL_IFNAME=eth0
export TRAINER_HOSTNAME=<hostname -s of the trainer node> TRAINER_HOST=<its address>
export ROLLOUT_HOSTNAME=<hostname -s of the rollout node> ROLLOUT_HOST=<its address>
```

`hostname` must equal `socket.gethostname()` (short name) on that node: `fresh_run` refuses to start a role on the wrong host. For `configs/captain-4b.8gpu.json`, set the TRAINER_* and ROLLOUT_* variables to the same machine.

**Reference 16-GPU layout** (`configs/captain-4b.json`):
- Trainer node `a`: 8 GPUs, FSDP world 8.
- Rollout node `e`:
  - GPU 0: SAM3. GPU 1: Molmo2-8B (gpu-memory-utilization 0.55).
  - GPUs 2–7: six policy vLLMs (0.65 each, `max-num-seqs` 8).
  - EGL rendering on all 8 GPUs; 80 concurrent episodes.

Start the whole run with the supervisor, on the trainer node:

```bash
POLICY_LAYOUT="$ROLLOUT_HOST:2,3,4,5,6,7:0.65" PERC_LAYOUT="$ROLLOUT_HOST:0:1:0.55" \
  setsid nohup bash rl/scripts/run_stream.sh rl/configs/captain-4b.json > $CAPTAIN_WORKSPACE/logs/captain-4b-supervisor.log 2>&1 < /dev/null &
```

The supervisor brings up perception and the policy servers, runs `fresh_run init`, starts the env server and proxy on the rollout node, and then starts the trainer. It resumes a crashed trainer from the latest checkpoint (up to 3 times in 2 h) and exits at `max_steps`.
- **Stop:** `kill $(cat $CAPTAIN_WORKSPACE/state/captain-4b.supervisor.pid)`. The trap tears everything down.
- **Resume:** add `--resume` to `run_stream.sh`.
- **Move a live run to another layout:** stop it, edit only layout keys in a copy of the config, then run `bash rl/scripts/rl_run.sh relocate <new config> a` and resume. `relocate` refuses changes to any training key.

The same steps by hand, one node at a time:

```bash
bash rl/scripts/rl_run.sh init rl/configs/captain-4b.json a                       # trainer node: creates <run_dir>/manifest.json
INIT=$CAPTAIN_WORKSPACE/models/CaPtain-4B-SFT bash rl/scripts/policies.sh up "2 3 4 5 6 7" 0.65   # rollout node
bash rl/scripts/rl_run.sh services rl/configs/captain-4b.json e                   # rollout node: preflight, proxy :8512, env :8402
eval "$(source rl/scripts/env.sh; echo export $(default_train_env))"               # trainer env of the reference run
bash rl/scripts/rl_run.sh train rl/configs/captain-4b.json a [--resume]          # trainer node
```

**8 GPUs on one node** (`configs/captain-4b.8gpu.json`). This layout ran rounds 293–300 of the reference run, with identical training semantics:
- GPU 0: SAM3 + Molmo (0.55) + EGL.
- GPUs 1–3: three policies.
- GPUs 4–7: trainer, FSDP world 4. The trainer GPUs run close to 80 GB.

```bash
POLICY_LAYOUT="$ROLLOUT_HOST:1,2,3:0.65" PERC_LAYOUT="$ROLLOUT_HOST:0:0:0.55" SEQS=16 bash rl/scripts/run_stream.sh rl/configs/captain-4b.8gpu.json
```

### Trainer environment

The supervisor passes these to the trainer (`scripts/env.sh: default_train_env`):

| Variable | Effect |
|---|---|
| `RL_REFIT=nccl`, `RL_REFIT_NCCL_SO` | Weights go straight into the policy vLLMs. Point `RL_REFIT_NCCL_SO` at the vLLM env's libnccl. |
| `RL_KL_ON_KEPT=1` | KL only on ratio-mask kept tokens. |
| `RL_PAYLOAD_NCCL=1` | Batch payload broadcast over NCCL. |
| `RL_KEEP_PREV_FULL_STATE=1` | Keep the previous checkpoint's FSDP state. |
| `RL_ASYNC_OFFLOAD=1`, `RL_ACT_OFFLOAD_MIN_TOKENS=1000000` | Activation offload settings. |
| `RL_ASYNC_CKPT_PROCESS=0` | Thread-based checkpoint writes. The process-based writer leaks shared memory. |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | |
| `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600` | |

### Outputs, checkpoints, evaluation

`<run_dir>` holds:
- `manifest.json` (config + source hashes);
- `metrics.jsonl`, with per round: success, false/missed finish, KL, co-critic AUC/Brier/calibration, costs, staged-reward depth, early-stop stats;
- `episodes-<node>/` (one directory per episode: the harness outputs plus `rollout.json`);
- `checkpoint-<N>/` every 4 rounds.

Each `checkpoint-<N>` contains merged **bf16** weights. Turn one into a self-contained Hugging Face model with:

```bash
bash rl/scripts/export_checkpoint.sh $CAPTAIN_WORKSPACE/runs/captain-4b/checkpoint-300 $CAPTAIN_WORKSPACE/models/CaPtain-4B-SFT /out/CaPtain-4B
```

The fp32 FSDP/optimizer state lives in `full_state_dir` and is only needed to resume. To evaluate an exported model, serve it with vLLM and run the harness evaluation, as in `../harness/README.md` (task lists under `harness/eval/tasks/`).

The evaluation protocol differs from RL rollouts:
- T = 1, cap of 60 requests, 1080 s.
- 150 physical actions instead of 120.
- Official LIBERO states 11–16.

## Hyperparameters (CaPtain-4B; all set in `configs/captain-4b.json`)

| Item | Value |
|---|---|
| Init / KL reference | CaPtain-4B-SFT (`base_model`) |
| Fine-tuning | full, FSDP2, fp32 master weights, bf16 compute |
| Optimizer | AdamW β = (0.9, 0.99), weight decay 0, grad clip 0.3 |
| Learning rate | 4.24e-6, constant after a linear warm-up over 512 valid attempts |
| Batch | 256 episodes / round = 64 starts × G = 4; K = 4 rounds per update (1,024 episodes) |
| Length of the reference run | 300 rounds = 75 optimizer updates = 76,800 training episodes (CaPtain-4B = checkpoint-300; the run was later continued to 400, which is not part of the release) |
| KL | β = 0.002, k3, on ratio-mask kept tokens; reference forward on 1 pass in 8, weighted × 8 (`ref_every` 8) |
| Ratio mask | keep iff 1 − 0.8 < r < 1 + 3.0 |
| Score centering | on, sampler top-k 20 |
| Loss normalisation | trajectory_mean; zero-signal batches skipped |
| Trained requests / trajectory | ≤ 8, uniform without replacement, inclusion-weighted. Critic GAE runs over all requests. |
| Critic | warm start + co-training: lr 1e-4, 2 updates / round, `critic_hidden` none (physics + t only) |
| Advantage | GAE λ = 0.9 (same-t), group-relative outcome (G = 4), 1/σ scaling (EMA 0.9) |
| Returns | success + finish 1, missed finish 0.25, false finish −0.3, exhausted −0.1, else 0 |
| Costs | `act_cost` 0.008 beyond `act-ref`, `fail_cost` 0.02. Post-success: 0.005 / request + actions charged, capped at 0.45. Code cost `python_exec_cost_v1` weight 0.1, budget 60. |
| Staged reward | credit 0.15 × depth / (rungs + 1) on the 8 hard tasks' ladders; failures only |
| Early stop | τ 0.02, k 5, from request 5, control 5%; success rule `success_k` 15; hard tasks exempt from the critic cut |
| Caps (rollout) | 60 requests, 1080 s wall (+60 s grace), 120 physical actions, 600 tool calls, 61,440 tokens per request (`overlength_guard` drops ≤ 2 over-long trajectories per round) |
| Context policy | 65,536-token window, compaction at 60,000 (keep 16,000 recent), ≤ 4,096 output tokens, tool output ≤ 4,000 tokens (spilled), ≤ 12 images, thinking on |
| Sampling | T = 1.0, top-p 1.0; streaming, max staleness 2 |
| Starts | all 40 tasks of LIBERO spatial/object/goal/10, fresh starts only, uniform; thin-basket exclude list |
| Seed | 17; checkpoints every 4 rounds |
| Attention | FA3 for the language model's full-attention layers (`attn_backend`; set `null` for SDPA) |

## Hardware and time

- **Reference layout:** 16 × H100-80GB (two 8-GPU nodes). One round of 256 episodes took about 15–19 minutes (≈ 930–1,130 s), so 300 rounds took about 3.5–4 days of wall time.
- **Single node:** on 8 × H100 a round takes about 26 minutes (≈ 1,550–1,650 s). The trainer is then the bottleneck.
- **Memory and CPU:** the reference rollout node had 2 TB RAM and 192 cores for 80 concurrent episodes (LIBERO + harness host stores).
- **Disk:**
  - about 20 GB of run-dir data per round;
  - ≈ 230 GB per fp32 FSDP state (two are kept);
  - raw proxy records (`raw_record_dir`) grow by several GB per round on the rollout node's local disk; prune old `step-*` directories.

## Tests

```bash
cd rl && CUDA_VISIBLE_DEVICES= python -m pytest -q tests      # CPU only; 387 tests
```
