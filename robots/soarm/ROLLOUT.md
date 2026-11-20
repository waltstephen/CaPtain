# SO-101 twin: teacher rollouts on a new machine

From a bare Linux box to CaPtain SFT rows. MuJoCo renders on the CPU (OSMesa) and the teacher model is a remote API;
perception is real (SAM3 + Molmo2-8B services on a GPU, §3.1, the LIBERO paper protocol), so a collection box needs
one or more GPUs for those services (or their URLs on another host). GPUs are also needed later for SFT
(`tools/train_sft.sh`) and for serving student checkpoints (`tools/serve_policy.sh`).

## 0. Requirements

- Linux x86_64 (tested: Ubuntu 22.04, kernel 5.15). Unprivileged user namespaces enabled (bubblewrap sandbox for the
  agent's Python): `bwrap --ro-bind / / --dev /dev true` must work as your user. Ubuntu 24.04 blocks this by default
  (`kernel.apparmor_restrict_unprivileged_userns=1`): ask an admin for
  `sysctl -w kernel.apparmor_restrict_unprivileged_userns=0`, or set `SOARM_NO_SANDBOX=1` to run without bwrap — the
  agent's Python is then unconfined and can read the run directory (hidden request marks included), so prefer the sysctl
  for collection and the flag only for smoke tests.
- CPU: ~1 core per parallel episode (+1 for the proxy). 6 parallel episodes ran on 32 cores with lots of headroom.
- RAM: ~1.5–2 GB per parallel episode (env server + driver + sandboxed Python).
- Disk: ~4 GB for venvs + OSMesa + Menagerie; a running episode can hold up to 0.5 GB of sensor stores, pruned to
  **~5–30 MB per finished episode** (`batch.py` prunes automatically).
- Network: the teacher API (Responses-API gateway or api.deepseek.com) and, for setup, GitHub + PyPI + conda-forge.
- Packages: `git`, `bubblewrap`, `curl`, a C toolchain is NOT needed. Python via `uv` (recommended) or conda; OSMesa via
  conda/micromamba (`mesalib<25` from conda-forge; Mesa ≥ 25.1 dropped OSMesa).

## 1. Code

```bash
git clone --recurse-submodules -b so101-fast https://github.com/waltstephen/CaPtain.git
cd CaPtain               # harness/ is pinned by the submodule (CaPtain-Harness so101-fast); do not update it
export REPO=$PWD
cp robots/soarm/env.example robots/soarm/soarm.env      # edit SOARM_ROOT if you do not want ~/soarm_work
source robots/soarm/env.sh && echo "$SOARM_ROOT $HARNESS_DIR"
```

`robots/soarm/env.sh` defines every path (see `env.example`); all tools source it. Settings file search order:
`$SOARM_ENV_FILE`, `robots/soarm/soarm.env`, `~/.soarm.env`. Never commit a filled-in env file.

## 2. Python environments, OSMesa, assets

```bash
source robots/soarm/env.sh
# uv: curl -LsSf https://astral.sh/uv/install.sh | sh   (or use conda for Python 3.11)
# (a) env server / driver / tools: Python 3.11, harness transfer requirements (mujoco 3.3.7)
uv venv -p 3.11 "$SOARM_VENV"
uv pip install -p "$SOARM_VENV/bin/python" -r "$HARNESS_DIR/requirements-transfer.txt" pytest
uv pip install -p "$SOARM_VENV/bin/python" --no-deps -e "$HARNESS_DIR"
# (b) the agent's sandboxed Python: exactly the versions CaPtain was trained with
uv venv -p 3.11 "$(dirname "$(dirname "$STUDENT_PYTHON")")"
uv pip install -p "$STUDENT_PYTHON" -r "$HARNESS_DIR/requirements-student.txt"
# (c) OSMesa (CPU rendering). Any conda works: conda / mamba / micromamba
conda create -y -p "$MESA_DIR" -c conda-forge --override-channels "mesalib<25"
ls "$MESA_DIR/lib/libOSMesa.so"
#     GPU box alternative: SOARM_GL=egl in soarm.env (NVIDIA EGL; skip (c))
# (d) robot model: MuJoCo Menagerie robotstudio_so101 at the pinned commit
bash robots/soarm/fetch_assets.sh "$SOARM_ROOT/menagerie"
# (d2) cube suite + CC0 textures (outside git, $SOARM_ASSETS = $SOARM_ROOT/assets; ~1.5 min, ~60 MB). Every cube-domain
#      episode needs them; teacher batches copy them to $SOARM_ROOT/assets_frozen/<hash> (tools/freeze_assets.py)
(cd robots/soarm && source env.sh && python -m assets.build_cubes && python -m assets.fetch_textures)
# (e) the env-server launcher the ReAct driver expects ($HARNESS_DIR/.venv-libero/bin/python -> wrapper)
bash robots/soarm/setup_wrapper.sh
# (f) toolchain preflight receipt (sandbox + student Python check) required by the python-runtime/v1 driver
PYTHONPATH="$HARNESS_DIR/src" "$SOARM_VENV/bin/python" rl/toolchain_preflight.py --harness "$HARNESS_DIR" \
  --contract python-runtime/v1 --student-python "$STUDENT_PYTHON" > "$FRESH_ENV_PREFLIGHT_PATH"
python3 -c "import json,os;print(json.load(open(os.environ['FRESH_ENV_PREFLIGHT_PATH']))['ok'])"   # True
```

## 3. Checks (5 min)

```bash
source robots/soarm/env.sh && cd robots/soarm
python -m pytest -q tests                                     # 309 passed
python tools/scripted_pick.py --family cube_pick_lift --seeds 0-4  # physics: "lifted 5 / 5 success 5" (≥ 4 is fine)
python tools/server_smoke.py --out /tmp/soarm_smoke | tail -3 # env server through the agent's client: "SMOKE OK"
```

### 3.1 Perception services (required; `soarm/perception.py`)

Episodes use real SAM3 and Molmo2-8B exactly as the LIBERO paper protocol (harness `services/start_perception.sh`,
`server/env_server.py` build_libero_episode): SAM3 = cap-x's server @53e9966 (sam3 @6fe87d6, `sam3.pt` sha256
9999e234…) on :8114, Molmo2-8B @e28fa28 served by vLLM 0.28.0 with `--max-model-len 16384 --max-num-batched-tokens
16384 --limit-mm-per-prompt '{"image":2}' --trust-remote-code` on :8122, the harness's own `CapXHttpToolServices`
(attempts=1).
```bash
export SOARM_SAM3_URLS=http://127.0.0.1:8114 SOARM_MOLMO_URLS=http://127.0.0.1:8122/v1,http://127.0.0.1:8123/v1
# one replica per episode (stable in the seed); unset = SAM3_URL / MOLMO_URL (env_server defaults :8114, :8122/v1)
```
Each episode writes `soarm/perception.json` (mode, replica URLs, tool revisions, depth) and `soarm/perception.jsonl`
(one row per service call: endpoint, prompt, latency, outcome, `"privileged": false`); `batch.jsonl` rows carry
`perception`, `perception_calls`, `perception_privileged_calls` (must be 0), `perception_errors`. `SOARM_PERCEPTION=
privileged` (MuJoCo-segmentation stand-in, `perception_sim.py`) is for tests only: `batch.py` refuses it for data.
Every SO-101 episode collected before 2026-11-11 used the privileged stand-in.

### 3.2 Depth: MoGe-2 from the RGB frame (required; `SOARM_DEPTH=moge2`)

The real SO-ARM has RGB cameras only, so episodes get depth the way the real arm does: the scene renders RGB only
(no simulator depth reaches the episode) and the harness's hardware monocular-depth path fills it --
`adapters/hardware/mono_depth.py` `MonoDepthSource` (MoGe-2 ViT-L `Ruicheng/moge-2-vitl-normal`@cb0e8bbd, `model.pt`
sha256 280741fd…, MoGe code @74fbce05, served by harness `services/depth_server.py` @f4abc69; the camera intrinsics
set MoGe's fov_x; one robust scale per frame puts the table plane at robot_base z = 0) for both cameras, plugged into
`XR1FacadeEnv(depth_estimators=...)` exactly as `hardware/episode.py` does for a `"depth": "mono"` camera.
```bash
harness/scripts/models/download_moge.sh      # or hf download Ruicheng/moge-2-vitl-normal model.pt --revision cb0e8bbd...
python harness/services/depth_server.py --checkpoint $MOGE_CHECKPOINT --port 8130 --device cuda:0   # ~2 GB GPU, 0.15 s/frame (A6000)
export SOARM_DEPTH=moge2 SOARM_DEPTH_URLS=http://127.0.0.1:8130,http://127.0.0.1:8131   # one replica per episode (by seed)
```
Episodes refuse to start unless their depth service answers `/health` as `moge2`; `perception.json` records `depth`,
`depth_url`, `depth_revision` (also `tool_revisions.depth`), alignment and the service's health answer;
`perception.jsonl` gets one `"kind": "depth"` row per estimate (latency, plane-fit scale/inliers, cache hits).
`batch.jsonl` rows and `episode_result.json` (`soarm_perception`) carry `depth_source`, `depth_revision`,
`depth_requests`; `batch.py` refuses `SOARM_DEPTH=sensor` and quarantines any episode whose depth source is not
moge2. `SOARM_DEPTH=sensor` (MuJoCo rendered depth, optional `SOARM_DEPTH_NOISE`) runs only inside pytest or with
`SOARM_DEPTH_ABLATION=1` (single ablation episodes through `tools/run_episode.sh`). Episodes before 2026-11-11 17:00 UTC
used rendered sensor depth (`depth_source` absent = sim_sensor). `tools/depth_eval.py` measures MoGe-2 against the
rendered depth on div2 frames (pixel errors, object centroid / top-height errors through the tools' back-projection).
Since div3c (2026-11-12) the plane fit uses only pixels whose ray meets the table plane inside the calibrated table
rectangle `perception.TABLE_FOOTPRINT_M` (x -0.04..0.39 m, |y| <= 0.33 m; every SO-ARM table covers it) and starts
from the densest cluster (harness `MonoDepthSource(table_footprint_m=...)`). The old fit locked onto the floor or wall
past the table edge in 12/160 div3 head frames; with the footprint 0/160 frames are off by more than 5 %, and head-camera
objects more than 2 cm off drop from 18.5 % to 5.6 %. `perception.json` records `depth_table_footprint_m`; depth rows
carry `align.footprint`. The real arm needs the same rectangle measured once on its table
(`configs/hardware/soarm101_prompt_depth.json`).

### 3.3 Prompt profile (intro v5 = harness `compact_v1`)

`SOARM_INTRO=v5` (teacher batches: `tools/batch.py --intro v5`) builds TASK.md with the harness prompt profile
`compact_v1` instead of the full LIBERO-protocol text:
- the API is listed as signatures with one-line summaries, and the full documentation is in workspace files (`API.md`:
  general rules, `api/<name>.md`: one capability);
- a NOTES.md STATUS rule is added;
- the standards section is SO-ARM-only (`episode.STANDARDS_V5`);
- capability results longer than 6000 characters of JSON (move_to with its trajectory, inspect_contact_geometry,
  recover, hold) come back summarised, with `details_file` naming the complete result in the workspace `values/`.

The reason: the full TASK.md was 13 k o200k tokens. Together with the verbatim Codex system prompt, the base context
was 36 k provider tokens against a 49.5 k compaction limit, so the driver compacted every 2-3 requests, and about
40 % of all teacher requests only rebuilt state (div3 diagnosis).
The Codex system prompt, the images and the context settings are unchanged. Rows record `intro`, `prompt_profile`,
`task_md_version` and `task_md_sha256` (`soarm/prompt.json`, `batch.jsonl`, `episode_result.json` `soarm_prompt`).
The real arm uses the same overrides: `tools/real_arm_config.py` writes `configs/hardware/soarm101_prompt_depth.json`
(merge it into the `hardware_real` config; harness `adapters/hardware/episode.py` reads `task_markdown_overrides` and
`mono_depth.table_footprint_m`).

## 4. Teacher credentials and proxy

The ReAct driver speaks OpenAI chat/completions (`--api chat`, the student's request shape). `soarm/gateway_proxy.py`
listens locally and forwards to the teacher:

- **Responses-API gateway** (Responses API only; `/v1/chat/completions` was unreliable): key file `$GATEWAY_ENV_FILE`
  (chmod 600) with `TEACHER_BASE_URL=https://…/v1` and `TEACHER_API_KEY=…`.
- **Official DeepSeek API** (chat/completions, `--provider deepseek`): key file `$DEEPSEEK_ENV_FILE` with
  `DEEPSEEK_API_KEY=…` (optional `DEEPSEEK_BASE_URL`, default https://api.deepseek.com); model `deepseek-flash`, add
  `--extra-body '{"thinking": {"type": "enabled"}, "reasoning_effort": "high"}'`. `--budget-cny N` stops when the
  account balance has dropped by N since the proxy started.

Recommended teacher: **deepseek-v4.1-flash-1017** (gateway) / **deepseek-flash** (official): returns its full reasoning
(stored as `reasoning_content` → the `<think>` block of the SFT rows), cheapest. gpt-6.1-sol is the strongest
teacher but, like gpt-6.1-terra, hides its reasoning behind a short summary title — enable §4.1 to supervise its working.

```bash
bash tools/proxy.sh start ds 8652 --env-file "$GATEWAY_ENV_FILE" --reasoning '{"effort":"high"}' \
  --concurrency 6 --daily-cap-usd 30 --quota-flag "$LOG_DIR/QUOTA_EXCEEDED"
# official API instead:
# bash tools/proxy.sh start ds 8652 --provider deepseek --env-file "$DEEPSEEK_ENV_FILE" --concurrency 4 --budget-cny 15 \
#   --extra-body '{"thinking": {"type": "enabled"}, "reasoning_effort": "high"}'
```

Cost guard: the proxy estimates every DeepSeek-Flash request at list prices (USD/M tokens: cache hit 0.003, miss 0.15,
output 0.60; ×2 at peak = 01–04 and 06–10 UTC Mon–Fri) into `$LOG_DIR/gateway_cost_<Beijing day>.json`. At
`--daily-cap-usd` it writes the quota flag and refuses requests; it also writes the flag when the gateway answers
`usage_limit_exceeded` (the gateway key has its own $50/day limit, reset 00:00 Beijing). While the flag exists batches
launch nothing, and episodes cut short by it are deleted so the seed is rerun. The estimate is conservative: 3 measured
official-API episodes cost CNY 0.90 total (≈ $0.04 each) where the list-price estimate said ≈ $0.10 each.

### 4.1 Data learning: the teacher's working in the record

Some Responses teachers act without returning their reasoning (`gpt-6.1-sol`, `gpt-6.1-terra`: only a one-line summary
title), so the exported SFT rows would supervise the action but not the thinking behind it. `--data-learning
forced|parallel` has the teacher write its working out in a visible `log_reasoning_step` call before it acts, then folds
those steps into `message.reasoning_content` — the channel `export_sft` supervises as the row's `<think>` block
(`sft/data/FORMAT.md`). Code: `soarm/data_learning.py`; the driver and the batch tools are unchanged.

| mode | Responses calls per model request | guarantees |
|---|---|---|
| `forced` | 2 (the second is a prefix-cache hit) | every request carries visible steps before its action |
| `parallel` | 1 | the scratchpad is offered next to the tools with a use-it-first instruction; the teacher may skip it (`thought: false` in the log) |

- Needs a **Responses** gateway (`--provider gateway`); the official DeepSeek API already returns its full reasoning, so
  `--data-learning` refuses `--provider deepseek`.
- Set it globally with `SOARM_DATA_LEARNING`, `SOARM_DATA_LEARNING_ROUNDS` (default 3) and `SOARM_RESPONSE_IMAGE_CAP`
  (send only the N most recent history images — the chat backend's / exporter's cap) in `soarm.env`; `tools/proxy.sh`
  passes them through. Each proxy log row then carries `data_learning` (`steps`, `visible_words`, `rounds`, `calls`).
- `--daily-cap-usd` estimates DeepSeek-Flash list prices only, so it does **not** cap a gpt teacher; watch the provider
  account instead. `forced` roughly doubles the input tokens per request (the prefix cache absorbs most of it).
- Example (OpenAI-compatible Responses account, gpt-6.1-sol):

```bash
export SOARM_DATA_LEARNING=forced
bash tools/proxy.sh start sol 8652 --env-file ~/.soarm_openai.env --concurrency 6 --max-output-tokens 16000
SOARM_INTRO=v2 CONTEXT_LIMIT=42000 POLICY_URL=http://127.0.0.1:8652/v1 POLICY_MODEL=gpt-6.1-sol \
  bash tools/run_episode.sh pick_place_mat 8100 "$RUNS_DIR/try"
# the exported row's last assistant turn then has a <think> block made of the logged steps
```

## 5. One episode, then batches

```bash
# Task specs (CONTRACT.md): run_episode.sh / batch.py take spec ids or tids. Teacher data = the 15 train cube specs
# (`--families cube`); legacy 16 families = `--families legacy` (their old runs stay in runs/scale_ds).
# one episode (intro v3 = cube collection text; context limit 42 k for deepseek, see tools/run_episode.sh).
# POLICY_MODEL is the upstream model id: deepseek-v4.1-flash-1017 on the gateway, deepseek-flash on the official API.
SOARM_INTRO=v3 CONTEXT_LIMIT=42000 WALL_TIME=3600 POLICY_URL=http://127.0.0.1:8652/v1 \
  POLICY_MODEL=deepseek-v4.1-flash-1017 bash tools/run_episode.sh cube_stack_two 8100 "$RUNS_DIR/try"
python3 -c "import json,glob;print(json.load(open(glob.glob('$RUNS_DIR/try/*/episode_result.json')[0]))['host_final_success'])"

# adaptive batch: launches specs in proportion to their measured strict yield (Laplace estimate, floor 0.05 so the
# hard specs keep getting starts) until each has --target-per-family; resumable (same command again).
# Episodes that end in infrastructure_error (env-server crash, `stop_batch now`) are discarded and do not count; a resumed
# batch deletes their partial dirs and reruns the seeds.
SOARM_INTRO=v3 CONTEXT_LIMIT=42000 bash tools/start_batch.sh scale_cube --model deepseek-v4.1-flash-1017 \
  --policy-url http://127.0.0.1:8652/v1 --families cube --target-per-family 60 --machine "$SOARM_MACHINE" \
  --seed-start 200 --lanes 6 --wall-time 3600
tail -f "$LOG_DIR/batch_scale_cube.log"         # one JSON row per finished episode
bash tools/stop_batch.sh scale_cube [now]       # stop launching ("now": also kill running episodes)
rm "$SOARM_ROOT/STOP_scale_cube"                # before resuming


# div2 training mix (CONTRACT.md 8): the mix file sets the scene (edge-mounted arm, mixed layouts, train camera
# views, clutter) and SOARM_INTRO=v4 for every episode; per-task strict targets = --target-total x weight, tasks
# below 15 % teacher yield after 10 attempts are retired. One 100k seed block per model batch (1,100,000+).
CONTEXT_LIMIT=42000 bash tools/start_batch.sh div2_sol --model gpt-6.1-sol \
  --policy-url http://127.0.0.1:8655/v1 --mix div2 --target-total 2000 --machine datahost \
  --seed-start 1100000 --lanes 12 --wall-time 3600
```

- Seeds (CONTRACT.md 1): set `SOARM_MACHINE` in soarm.env. armhost owns 0–999,999 (data 0–8,999, eval 9,000+), the
  datahost 1,000,000+ (data 1,000,000–1,499,999, eval 1,500,000+); `batch.py --machine` refuses other seeds,
  so on datahost use `--seed-start 1000200` (cube blocks) or a div2 block (1,100,000+). Held-out cube
  specs (swap, gather, count_place, unstack) and legacy gather/lay_down/swap are never collected for SFT.
- div2 checks: `python tools/make_variants.py --check` (variant specs current), scripted-solver table
  `python -m taskgen.verify specs.jsonl --out r.jsonl --layouts 10 --min-solved 5 --min-templates 1 --scene-config
  "$(cat div2_config.json)"` -> `python tools/make_mix.py --id div2 --results r.jsonl --config div2_config.json`;
  render check `python tools/render_grid.py --out DIR --mix div2 --specs a,b,c` (views x layouts PNG + visibility).
- div3 (CONTRACT.md 9): the same checks with `--scene-config "$(cat configs/div3_scene.json)" --max-draws 120` and
  `python tools/make_mix.py --id div3 --shares div3 --precise-factor 0.35 --results ... --config configs/div3_scene.json`;
  batches `--mix div3` (seed blocks 1,110,000 / 1,120,000 / 1,130,000 / 1,140,000).
- Eval: fixed suites (CONTRACT.md 6): `python tools/batch.py --out $RUNS_DIR/eval_<model>/<suite> --suite <suite>
  --machine armhost --model <model> --policy-url <url> --lanes 2` for each of cube_seen, cube_heldout_families,
  cube_heldout_appearance, broad, div2_seen, div2_heldout_views; then `python tools/eval_summary.py ...`.
- Episodes land in `$RUNS_DIR/<batch>/soarm_sim_<tid>_seed-<seed>/`; `batch.jsonl` has one row per episode
  (`family` = spec id); `soarm/task.json` records the spec id/tid/split and the scene config.
- `python tools/summarize.py $RUNS_DIR/scale_cube --usage $LOG_DIR/gateway_usage_ds_scale.jsonl` → per-spec strict
  success, tokens, wall, gateway latency/errors.
- Prune older runs by hand: `bash tools/prune_runs.sh $RUNS_DIR/*`.

## 6. Expected numbers (armhost, 11-09)

| teacher | strict success (success + finish) | median requests | median wall | input tokens (cache hits) | output tokens | cost |
|---|---|---|---|---|---|---|
| deepseek-v4.1-flash-1017, intro v1 (pilot, 45 ep.) | 29 % | 60 | 553 s | 2.0 M (90 %) | 44 k | — |
| deepseek-flash official, intro v2 (3 ep.) | 3/3 | 24–57 | 190–370 s | 0.9–2.2 M (87–89 %) | 14–32 k | CNY 0.30/ep. |
| gpt-6.1-terra (pilot + 1 h scale-up, 76 ep.) | 22–54 % | 18–44 | 130–400 s | 0.7–1.8 M (47–67 %) | 2.5–7.5 k | most of $50 |

Weak families: push_to_region (reworked:
square tile + push recipe in intro v2), line_up, multi_step, sort_by_*.

## 7. Export to SFT rows and copy back

```bash
TOKENIZER_MODEL=/path/to/CaPtain-4B bash tools/build_sft.sh "$DATA_DIR/sft_v1" 2 "$RUNS_DIR/scale_ds" [more run dirs]
```
Needs `$SFT_VENV` (torch 2.10 + transformers 5.17 + flash-linear-attention 0.5.2, see `sft/README.md`). Output:
`rows.jsonl` (per-request rows of strict successes), `rows_fail.jsonl`, `episodes.jsonl`, `images/`, `packed.jsonl`,
`packed_fit.jsonl` (≤ 65,536 tokens), `sft_mix2.jsonl`, `sup_tokens.json`.

To move data to the training machine, copy either the exported dataset dir (`rows*.jsonl`, `images/`; ~3 MB of images
and ~1.5 MB of rows per successful episode) or the raw episodes. For raw episodes the files that matter are
`episode_result.json`, `driver/` (views/items/responses/requests: the exact model views), `workspace/images/`,
`workspace/TASK.md`, `soarm/task.json`, `calls.jsonl`, `hidden_object_audit.jsonl`, `hidden_request_marks.jsonl`
(RL critic) — ~5–30 MB per episode after pruning:
```bash
rsync -a --include='*/' --include='episode_result.json' --include='driver/***' --include='workspace/images/***' \
  --include='workspace/TASK.md' --include='soarm/***' --include='calls.jsonl' --include='hidden_*.jsonl' \
  --include='batch.jsonl' --exclude='*' "$RUNS_DIR/scale_ds/" dest:/path/scale_ds/
```
