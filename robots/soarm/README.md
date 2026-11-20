# SO-ARM101 (SO-101) twin, data pipeline and fast real-arm deployment for CaPtain

Code-as-policy MuJoCo twin of the LeRobot SO-101 behind the unchanged CaPtain harness tool surface, the teacher-data and
SFT pipeline built on it, the fast serving stack for two consumer GPUs, and a real-arm backend. Status: simulation twin,
teacher data, SFT, fast serving and a real SO-101 deployment (2026-11). The harness submodule of this branch is pinned to
CaPtain-Harness `so101-fast` (opt-in speed options, release defaults unchanged).

| File | What |
|---|---|
| `soarm/objects.py`, `soarm/families.py`, `soarm/tasks.py` | object catalog, 16 task families (language templates + success atoms), layout sampler |
| `soarm/predicates.py` | geometric success predicates (pure numpy; `tests/test_predicates.py`) |
| `soarm/scene.py`, `soarm/sim.py`, `soarm/camera.py` | MuJoCo physics scene (Menagerie `robotstudio_so101`), reset/settle, RGB (+ rendered depth for tests/ablation) + segmentation |
| `soarm/twin.py`, `soarm/driver.py` | robosuite-named kinematic twin, `XR1Driver` (5-DoF DLS IK, contact-force proxy), host-only object audit |
| `soarm/perception.py` | perception mode (`SOARM_PERCEPTION`): `real` (default) = real SAM3 + Molmo2-8B services, the LIBERO paper protocol (replica URLs, logged calls in `soarm/perception.jsonl`); `privileged` = tests only. Depth (`SOARM_DEPTH`): `moge2` (default) = MoGe-2 from the RGB frame through the harness's real-arm `hardware/mono_depth.py` (table-plane scale); `sensor` = rendered depth, tests/ablation only |
| `soarm/perception_sim.py` | PRIVILEGED SAM3/Molmo stand-in (MuJoCo segmentation of the observed frame), only with `SOARM_PERCEPTION=privileged` (tests) |
| `soarm/taskspec.py`, `soarm/specs/{legacy,cube,cube_var,broad,broad_mesh,gen,suites}/`, `CONTRACT.md` | JSON task specs (schema + validator, declarative sampler with div2/div3 layout styles/clutter/platforms/size-variant kinds), fixed eval suites, seed ranges per machine |
| `soarm/mix.py`, `soarm/specs/mixes/`, `tools/make_mix.py`, `tools/make_variants.py` | training mixes (scene config + intro + per-task weights, held-out guard; `batch.py --mix`), written from the scripted-solver table; parameterised variant specs |
| `taskgen/` | LLM proposals of new task-family specs, scripted-solver verification, paraphrases |
| `tools/render_grid.py`, `tools/depth_eval.py` | render check (layouts x camera views, visibility); MoGe-2 vs rendered depth |
| `soarm/episode.py`, `soarm/env_server_main.py` | `soarm_sim` / `soarm_real` suite builders; env server with the suites registered |
| `soarm/gateway_proxy.py`, `soarm/data_learning.py` | chat/completions -> Responses proxy for teacher models (quota flag, per-episode cache key); data learning: the teacher writes its working out in a visible `log_reasoning_step` call, folded into `reasoning_content` (the SFT row's `<think>`) |
| `soarm/export_sft.py`, `soarm/splits.py` | episodes -> CaPtain SFT rows (`sft/data/FORMAT.md`); held-out families, train/eval seeds |
| `tools/run_episode.sh`, `tools/batch.py` | one episode with the CaPtain eval protocol (`harness/eval/run_episode.sh` flags); resumable batches (plan, adaptive per-family targets, mixes, fixed suites) |
| `tools/start_batch.sh`, `stop_batch.sh`, `kill_episodes.sh`, `proxy.sh`, `prune_runs.sh` | batch and teacher-proxy job control |
| `tools/build_sft.sh`, `tools/add_total_tokens.py`, `tools/train_sft.sh`, `tools/sft_probe.sh` | SFT data build (export -> prefix-pack -> length filter -> mix order -> C) and 24 GB-GPU training (ZeRO-3, full CPU offload) |
| `tools/sftopd_sft.sh`, `tools/sftopd_eval.sh`, `tools/sftopd_summary.py` | the 8-GPU SFT recipe of the fast policy and its fixed twin eval (div3_seen + held-out cube families) |
| `tools/summarize.py`, `tools/eval_summary.py`, `tools/success_stats.py`, `tools/success_workload.py` | per-spec success, tokens, wall, gateway reliability; eval tables; real-arm-equivalent time of strict successes |
| `tools/freeze_assets.py`, `tools/scripted_pick.py`, `tools/server_smoke.py` | pinned asset copies for teacher batches; physics and server smoke checks |
| `assets/` | procedural cube suite, GSO + YCB import, CC0 textures -> `assets/registry.jsonl` / `textures.jsonl` |
| `soarm/asset_mj.py`, `tools/gate_assets.py`, `tools/contact_sheet.py` | MuJoCo bodies of registry objects; physics gate (drop + scripted top-down grasp); scene previews |
| `soarm/speed_profile.py`, `configs/speed_fast*.json` | speed profiles (`SOARM_PROFILE`), see "Fast profile" below |
| `tools/quantize_policy.py`, `tools/extract_lora.py` | W4A16 (+ W8A16 lm_head) quantization of CaPtain-4B for vLLM; LoRA extraction |
| `tools/fetch_mtp.py`, `tools/drafter/` | MTP drafter: fetch the Qwen3.5-4B MTP head, self-distil it on the policy's own outputs (`train_mtp.py`), offline acceptance (`eval_accept.py`), model dir (`make_model_dir.py`) |
| `tools/vllm_patches/` | vLLM 0.29 runtime patches (`SPEC_VERIFY_PATCH=1` in `serve_policy.sh`): split-KV verify, FA2 prefill, FP8-prefill hybrid, draft vocabulary, no-eagle-drop; `build_draft_vocab.py` |
| `tools/serve_policy.sh`, `tools/serve_perception.sh`, `tools/warm_services.py` | policy (vLLM) and perception (SAM3, Molmo2-8B FP8, MoGe-2) servers; prefix-cache / service warm-up |
| `tools/policy_proxy.py`, `tools/policy_replay.py`, `tools/policy_bench.py`, `tools/speed_eval.py`, `tools/stack_eval.sh`, `tools/hop_trace.py`, `tools/timing_probe/` | request recording / replay, latency benchmarks and per-hop traces |
| `soarm/real_bus.py`, `soarm/real_calib.py`, `soarm/real_driver.py`, `soarm/real_depth.py`, `soarm/real_episode.py` | real SO-101 backend: Feetech bus, calibrated kinematics, `XR1Driver` with soft limits / table guard, MoGe-2 with a per-pixel affine table fit, `soarm_real` suite |
| `configs/hardware/so101_armhost_*.json`, `soarm101_prompt_depth.json` | hardware config, joint map, kinematic and camera calibration of the arm used here (geometry is specific to that arm and its camera mounts: recalibrate yours); prompt + depth fragment that must match the twin |
| `tools/arm_probe.py`, `tools/calib_*.py`, `tools/camcal.py`, `tools/flatcal.py`, `tools/flatrec.py`, `tools/camstream.py`, `tools/so101_ik.py`, `tools/real_arm_config.py` | real-arm probing and calibration (joint zeros, hand-eye, kinematics, fixed camera from a board lying on the table) |
| `tools/run_real_episode.sh`, `tools/real_arm_web.py`, `tools/real_arm_home.py`, `tools/real_arm_park.py`, `tools/real_arm_stop.sh` | real-arm episodes, web control, home / park (torque off) / stop |

Setup and teacher rollouts on any machine: **[ROLLOUT.md](ROLLOUT.md)** (paths via `env.sh` / `env.example`,
env-server wrapper `setup_wrapper.sh`, Menagerie `fetch_assets.sh`). Suite id: `--suite soarm_sim --task-id <spec tid
(CONTRACT.md 1)> --seed <layout seed>`. Assets (outside git, under `$SOARM_ASSETS`, default
`$SOARM_ROOT/assets`): `python -m assets.build_cubes` (~1.5 min) and `python -m assets.fetch_textures`; the committed
registry already carries the gate results.

## Fast version (SO-101)

**What it is.** CaPtain-4B fine-tuned on native SO-101 twin rollouts and served with a latency stack that fits two
RTX 4090s next to the perception services.

- Policy `ck300-r1113c`: CaPtain-4B ([YijiaFan/CaPtain-4B](https://huggingface.co/YijiaFan/CaPtain-4B)) SFT on 555
  successful native SO-101 twin episodes of GPT-6.x teacher agents (harness div3 mix, intro v5 / prompt profile
  `compact_v1`, real perception SAM3 + Molmo2 + MoGe-2, 64k context, compaction keeping 16k, 12 images, 100 requests,
  T 1.0). SFT: 8x H100, 2 epochs, 1,228 steps, LR 1e-5 cosine, frozen vision tower (`tools/sftopd_sft.sh`). Data:
  [YijiaFan/CaPtain-SO101-SFT-Data](https://huggingface.co/datasets/YijiaFan/CaPtain-SO101-SFT-Data) (built with
  `tools/build_sft.sh`).
- Serving stack ([YijiaFan/CaPtain-4B-SO101-Fast](https://huggingface.co/YijiaFan/CaPtain-4B-SO101-Fast)): W4A16 RTN
  weights + W8A16 lm_head (`tools/quantize_policy.py`), an MTP head trained on the policy's own outputs
  (`tools/drafter/`), MTP-5 probabilistic speculative decoding with a 16k draft vocabulary, FP8-prefill hybrid and the
  vLLM patches of `tools/vllm_patches/` (split-KV verify, FA2 prefill, draft vocabulary, no-eagle-drop); harness and
  perception speed options from `configs/speed_fast_v5.json`.

**Results.**

| eval | CaPtain-4B (ck300) | fast policy (ck300-r1113c) |
|---|---|---|
| twin, data harness, `div3_seen` strict success | 4/169 | **45/169** |
| twin, held-out cube families strict success | 0/40 | **10/40** |
| twin, realistic perception, fast stack on 2x RTX 4090: pick_lift / pick_place_mat / put_in_container strict success | – | **27/45** |
| s per request (real-arm-equivalent timeline) | – | **1.455 s** |
| strict-success episode median | – | **32.3 s** |

Strict = the simulator's goal holds, the agent called `finish()`, no false finish. The stack was then deployed on a
real SO-101 (2026-11-13/14) with the motion-safe profile below.

**How to serve** (one host, two 24 GB GPUs; policy on GPU 0 with SAM3, Molmo2-8B FP8 + MoGe-2 on GPU 1):

```bash
git clone --recurse-submodules -b so101-fast https://github.com/waltstephen/CaPtain.git && cd CaPtain/robots/soarm
cp env.example soarm.env && source env.sh               # venvs and services: ROLLOUT.md sections 2-3
huggingface-cli download YijiaFan/CaPtain-4B-SO101-Fast --local-dir "$SOARM_ROOT/models/CaPtain-4B-SO101-Fast"
MODEL_DIR=$SOARM_ROOT/models/CaPtain-4B-SO101-Fast

# policy on GPU 0, :8700 (W4A16 + W8 lm_head + MTP head; SPEC_VERIFY_PATCH=1 loads tools/vllm_patches into vLLM)
SPEC_VERIFY_PATCH=1 SOARM_DRAFT_VOCAB=$MODEL_DIR/draft_vocab/draft_vocab16384_r13.safetensors \
SOARM_VLLM_PATCH_NO_EAGLE_DROP=1 SOARM_VLLM_PATCH_FP8_PREFILL=1 VLLM_DISABLE_COMPILE_CACHE=1 POLICY_MEM=0.62 \
POLICY_ARGS='--attention-backend TRITON_ATTN --max-num-batched-tokens 8192 --speculative-config {"method":"mtp","num_speculative_tokens":5,"draft_sample_method":"probabilistic","attention_backend":"TRITON_ATTN"}' \
  bash tools/serve_policy.sh "$MODEL_DIR" 8700 0 captain-4b

# perception: SAM3 on GPU 0 :8114 (8-image encoder cache), Molmo2-8B FP8 on GPU 1 :8122, MoGe-2 on GPU 1 :8130 (level 0)
SOARM_PROFILE=fast_v5 bash tools/serve_perception.sh sam3 0
MOLMO_MEM=0.62 bash tools/serve_perception.sh molmo 1
SOARM_PROFILE=fast_v5 MOGE_CHECKPOINT=/path/to/moge-2-vitl-normal/model.pt bash tools/serve_perception.sh depth 1
python tools/warm_services.py --body <a recorded first request body>   # must print no error

# twin episode with the fast profile (prompt profile must match training: intro v5 / compact_v1)
SOARM_PROFILE=fast_v5 SOARM_INTRO=v5 MAX_REQUESTS=100 POLICY_URL=http://127.0.0.1:8700/v1 POLICY_MODEL=captain-4b \
  bash tools/run_episode.sh pick_lift 9000 "$RUNS_DIR/fast_v5"
# real arm (motion-safe profile fast_v5_real is the default here)
bash tools/run_real_episode.sh "pick up the blue cube" "$RUNS_DIR/real/ep001"
```

GPU memory (measured): GPU 0 = policy (~13 GB incl. FP8-prefill weight copies) + SAM3 (7.7 GB with the 8-image cache)
= 23.1 / 24 GB; do not raise the SAM3 cache next to the policy (it OOMs and returns HTTP 500). GPU 1 = Molmo2-8B FP8
(13.8 GB) + MoGe-2 (~3.8 GB). The real-arm hardware config is `configs/hardware/so101_armhost_real.json` merged with
`soarm101_prompt_depth.json` (re-measure `table_footprint_m` on your table); cameras are served by
`tools/camstream.py`, the arm's serial port is set in the config (`/dev/ttyACM0`).

**Motion safety.** `speed_fast_v5.json` contains motion speed-ups validated only in the twin (`motion_speed_scale 3`,
0.1 m/s and 1.0 rad/s defaults, the above-cap `step_profile`, `gripper_settle_min_steps 6`). They are **not signed off
for a real arm**. Real runs use `configs/speed_fast_v5_real.json`: every model-side and service speed-up, release motion
speeds (`motion_speed_scale 1`, 0.05 m/s, 0.5 rad/s), no step profile and no early gripper settle. The calibrated
kinematics are good inside the calibrated workspace (far held-out 1.2 mm mean) but not demonstrated at far/low poses
(shoulder_lift raw > 2750) or a strongly flexed wrist (wrist_flex raw < 2100): keep targets on the table footprint and
away from those poses. The driver clips every command to the joint soft limits and refuses goals below the table guard;
keep an e-stop at hand (`tools/real_arm_stop.sh` kills the episode, parks the arm and releases torque), run slowly
first, with someone watching the arm. Expect slower wall time on the real arm than the twin numbers (real frames,
release motion speeds).

## Fast profile (`SOARM_PROFILE`; eval/deploy only)

`SOARM_PROFILE=<name>` selects `configs/speed_<name>.json` (`soarm/speed_profile.py`); unset = the release/student
protocol, byte-for-byte. One variable drives every layer, all through CaPtain-Harness options (harness
`docs/harness.md` "Speed options"), no hooks. `fast` is the first stacked profile (for the earlier `sftv0` student),
`fast_v5` the one for models trained on the v5 / `compact_v1` data, `fast_v5_real` its motion-safe real-arm variant;
`fast_v5s`, `fast_v5se`, `fast_v5i24`, `fast_se` are A/B variants (sticky image cap, streamed cell execution, 24-image cap).

| Lever | Profile entry | Harness option |
|---|---|---|
| fast env-server start | `driver.server_poll_s 0.02`, `server_grace_s 0` | `react_episode.py --server-poll-s/--server-grace-s` |
| append-only image cap | `driver.max_images 12`, `image_drop_chunk 6` | `--max-images`, `--image-drop-chunk` (`react/chat_llm.py cap_images`) |
| thinking budget, output cap, temperature | `driver.thinking_token_budget`, `max_output_tokens`, `temperature` | `--thinking-token-budget`, `--max-output-tokens`, `--temperature` |
| move_to speed x3, defaults 0.1 m/s / 1 rad/s (cap = API maxima) | `tool_backend_options` | `motion_speed_scale`, `default_max_speed_m_s`, `default_max_angular_speed_rad_s` (`adapters/capx/motion_speed.py`) |
| distance-aware step profile, short settle | `backend_options.step_profile` | `XR1ActionBackend(step_profile=...)` per instance |
| gripper settle early stop | `backend_options.gripper_settle_min_steps 6` | `gripper_settle_min_steps` |
| SAM3 image-encoder cache | `services.sam3_cache_images 8` (each cached image holds ~0.1 GB of encoder features on the GPU; 32 OOMed next to the policy on a 24 GB card) | `services/sam3_server.py --cache-images` |
| mark_point label check concurrent with segmentation | `tool_backend_options.concurrent_label_check true` | `ToolCapabilityBackend(concurrent_label_check=...)` |
| Molmo stops after the first coordinates | `services.molmo_stop_after_coords true` | `CapXHttpToolServices(molmo_stop_after_coords=...)` |
| MoGe-2 resolution level | `services.depth_resolution_level 0` (ViT-L forward 61 -> 28 ms, 0.9 % median depth change) | harness `services/depth_server.py --resolution-level` via `tools/serve_perception.sh` |

Profile flags come after `tools/run_episode.sh`'s own flags (they override `TEMP`, `MAX_OUTPUT_TOKENS`); extra
arguments to `run_episode.sh` override the profile. The env server writes the active profile to
`<episode>/soarm/speed_profile.json`; move_to receipts carry `motion_speed` (requested and executed limits);
`<episode>/soarm/timing.json` records peak joint speed/acceleration for the STS3215 servo-limit check (no-load
~4.7 rad/s at 7.4 V; at 0.1 m/s the SO-101 joints stay below ~0.6 rad/s). Latency evaluation of a stack:
`SOARM_PROFILE=<name> tools/stack_eval.sh <name> <proxy port>` (15 timing + 30 success episodes; `tools/policy_proxy.py`
only records request bodies, the drafter's training data).
