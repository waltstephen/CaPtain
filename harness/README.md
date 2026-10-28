# CaPtain-Harness

CaPtain-Harness is the policy-facing harness of **CaPtain: A Pipeline for Post-Training Code-as-Policy Robot Agents**.
A vision-language model controls a robot by writing Python against a fixed robot API of low-level perception and motion
primitives. CaPtain-4B and CaPtain-4B-SFT were trained in this harness on LIBERO and evaluated in it on LIBERO,
LIBERO-PRO, LIBERO-Plus, LIBERO-MAX, RoboTwin 2.0, RoboDojo, SimplerEnv and CaP-Bench. The same API runs on a real robot
through the hardware backend interface.

The model sees only:
- images;
- its own code and the code's output;
- receipts of the robot calls.

The simulator state and the success predicate never leave the host process.

```
 vLLM policy (OpenAI-compatible chat, thinking on)        SAM3 :8114        Molmo2-8B :8122
        ^  messages + images                                  ^ perception      ^
        |                                                     |                 |
 ReAct driver  (src/captain_harness/react, scripts/react_episode.py)
   - one tool: `python` = persistent Python 3.11 session in a bwrap sandbox (python-runtime/v1)
   - per-request view with context compaction (64k window), tool-output cap + spill, image budget
        |  robot.py (stdlib HTTP client): 22 capabilities + finish() installed as plain functions
        v
 environment server  (src/captain_harness/server/env_server.py), one episode per process
   - capability layer (src/captain_harness/adapters/capx): perception, IK, guarded Cartesian motion, contact / geometry evidence
   - receipts: one <=300-char line per robot call appended to the cell output
   - completion gate: finish() is checked against the agent's OWN public measurements (never simulator state)
   - LIBERO backend (src/captain_harness/adapters/libero), LIBERO-PRO / LIBERO-Plus / LIBERO-MAX variants (server/libero_variants.py)
   - transfer backends: RoboTwin 2.0, RoboDojo, SimplerEnv (adapters/robotwin2, robodojo, simplerenv), CaP-Bench (adapters/capgym)
   - hardware backend interface for real robots (adapters/hardware)
   - hidden state (host only): success predicate, object poses, per-request marks used by the RL simulator critic
```

## What the model sees

- **The `python` tool.** It is a persistent session that runs `import robot; from robot import *` at start-up.
  - Variables persist across calls.
  - Each cell times out after 600 s; a crash restarts the session.
  - There is no shell.
- **What a cell returns** (`react/python_repl.py`):
  - status and time;
  - stdout and the value of the last expression;
  - a trimmed traceback;
  - one receipt line per robot call;
  - the names of variables that changed;
  - up to 4 images attached with `image(path)`.
- **The robot API: 22 capabilities plus `finish(note)`.**
  - Capabilities: `observe`, `segment`, `select_detection`, `point`, `mark_point`, `track_target`, `read_motion_trajectory`, `get_robot_state`, `gripper_state`, `move_to` (an absolute EEF pose, or one jaw command), `hold`, `recover`, `release`, `preview_motion`, `measure_image_points`, `transform_pose`, `robot_geometry`, `check_motion`, `inspect_contact_geometry`, `compare_object_observations`, `inspect_release_geometry`, `measure_object_geometry`.
  - There are no grasp or place macros.
  - Results are dicts of the form `{success, executed, failure_category, message, value, diagnostics, budget}`; failures never raise.
- **TASK.md** is rendered by `server/task_markdown.py` and inlined into the first request. It contains:
  - the instruction;
  - how to act;
  - the standards the final scene is judged by;
  - the API signatures.
- **Budgets** (eval protocol): 60 model requests, 150 physical actions, 600 tool calls, 1080 s of wall time.
- **Completion gate** (`adapters/capx/completion_gate.py`).
  - `finish()` is accepted only if the agent's own recorded measurements support every goal: XY ≤ 2 cm, knob ≥ 0.35 rad, side ≤ 4 cm, and a fresh measurement per goal.
  - A rejection returns the numbers and the episode continues. After 2 rejections the finish passes.
  - The host then scores the final state once with the hidden LIBERO predicate. The agent never sees the verdict.
- **Context policy.**
  - The window is 65,536 tokens; compaction uses the deterministic `window` strategy: programmatic facts plus the most recent 16k tokens of whole turns.
  - Tool outputs are capped at 4,000 tokens; the full text is spilled to `.tool_output/` in the workspace.
  - At most 12 images are kept and at most 4096 output tokens are generated per request.

## Install

The harness was used on Linux x86_64 with NVIDIA GPUs and headless EGL rendering. You need:
- `git`;
- `bubblewrap` ≥ 0.5 (`bwrap`, for the sandbox; it needs `--clearenv`);
- an NVIDIA driver with EGL.

If your system glvnd lists only Mesa, point `__EGL_VENDOR_LIBRARY_FILENAMES` at an NVIDIA vendor file (`~/.config/glvnd/10_nvidia.json` is picked up automatically).

```bash
cd harness
scripts/setup_third_party.sh                 # LIBERO @ 8f1084e into third_party/libero (add --capx for the SAM3 service)

# 1. environment-server / driver interpreter (Python 3.10)
python3.10 -m venv .venv-libero
.venv-libero/bin/pip install -r requirements-libero.txt
.venv-libero/bin/pip install -e third_party/libero --no-deps
.venv-libero/bin/pip install -e . --no-deps

# 2. the agent's sandboxed interpreter (Python 3.11.16; exact versions matter, the driver probes them)
python3.11 -m venv ~/envs/student-py311 && ~/envs/student-py311/bin/pip install -r requirements-student.txt
export STUDENT_PYTHON=~/envs/student-py311/bin/python

# 3. perception services: an env with CaP-X + sam3 (third_party/cap-x, see its README) and a vLLM env for Molmo2-8B
#    SAM3 weights (facebook/sam3) are gated on Hugging Face: pass SAM3_CHECKPOINT=/path/sam3.pt or export HF_TOKEN.
TOOLS_PYTHON=/path/to/tools-env/bin/python VLLM_BIN=/path/to/vllm-env/bin/vllm SAM3_GPU=0 MOLMO_GPU=0 services/start_perception.sh
```

LIBERO writes its `config.yaml` to `<harness>/.libero-config` automatically on the first episode.

Optional benchmark forks (each needs its own venv):
- LIBERO-PRO and LIBERO-plus are drop-in `libero` packages. Install each into `.venv-libero-pro` / `.venv-libero-plus` at the harness root, or under `$CAPTAIN_BENCHMARKS_HOME`, which defaults to `~/benchmarks`. Their checkouts go to `~/benchmarks/LIBERO-PRO` / `~/benchmarks/LIBERO-plus`, or are passed with `--server-arg=--libero-checkout --server-arg=<dir>`.
- LIBERO-plus additionally needs ImageMagick (`wand`) for its sensor-noise perturbation.
- RoboTwin 2.0, RoboDojo and SimplerEnv: a second harness interpreter (`python3.11 -m venv .venv-transfer && .venv-transfer/bin/pip install -r requirements-transfer.txt`) plus each simulator in its own environment; CaP-Bench runs in `.venv-libero`. See [`eval/transfer/README.md`](eval/transfer/README.md).

## Run one evaluation episode with a served model

```bash
# policy: CaPtain-4B (or CaPtain-4B-SFT) with vLLM 0.29 (Qwen3.5 support), one GPU
services/serve_policy.sh <hf-user>/CaPtain-4B 8600          # or a local model dir

# one LIBERO episode with the paper protocol (T=1, top_p=1, 60-request cap, 1080 s, thinking on)
POLICY_URL=http://127.0.0.1:8600/v1 STUDENT_PYTHON=$STUDENT_PYTHON EGL_DEVICE=0 \
  eval/run_episode.sh libero_spatial 0 11 runs/demo
cat runs/demo/libero_spatial_0_seed-11/episode_result.json   # host_final_success, agent_finished, termination_reason, ...
```

An episode directory contains:
- `driver/transcript.jsonl`, `driver/views.jsonl` and `driver/requests.jsonl`: exactly what the model was sent and what it returned;
- `calls.jsonl`: every robot call with its receipt;
- `workspace/`: TASK.md, NOTES.md and the images;
- `server.log`;
- host-only audit files such as `hidden_*` (the model never sees them).

Seeds:
- **Seeds 11–16** select the official LIBERO init states `init_states[seed % 50]`. These are the held-out states of the paper; CaPtain training never used them.
- **Seeds ≥ 1000** select freshly sampled valid initial states from `data/libero_fresh_init/` (seed 1000+k gives state k). They are used for training-style rollouts.

## Reproduce the paper's evaluations

```bash
export POLICY_URL=http://127.0.0.1:8600/v1 STUDENT_PYTHON=~/envs/student-py311/bin/python EGL_DEVICE=0
python3 eval/run_jobs.py eval/tasks/libero_official_240.tsv     --out runs/captain-4b --parallel 8
python3 eval/run_jobs.py eval/tasks/libero_pro_postask_800.tsv  --out runs/captain-4b --parallel 8 --artifacts <CaPtain-artifacts>/eval
python3 eval/run_jobs.py eval/tasks/libero_plus700_pro160.tsv   --out runs/captain-4b --parallel 8
python3 eval/run_jobs.py eval/tasks/libero_max_160.tsv          --out runs/captain-4b --parallel 8 --artifacts <CaPtain-artifacts>/eval
python3 eval/summarize.py eval/tasks/libero_official_240.tsv runs/captain-4b
```

| Task list | Episodes | CaPtain-4B host / strict | CaPtain-4B-SFT host / strict |
|---|---|---|---|
| `libero_official_240.tsv`: LIBERO, 4 suites × 10 tasks × official states 11–16 | 240 | 73.8 / 67.5 | 55.4 / 52.1 |
| `libero_pro_postask_800.tsv`: LIBERO-PRO Pos+Task, 6-cell mean | 600 | 78.5 / 71.5 | 59.5 / 56.7 |
| same, 8-cell mean | 800 | 68.6 / 58.9 | 49.2 / 45.1 |
| `libero_plus700_pro160.tsv`: LIBERO-Plus, 7 categories | 700 | 66.1 / 59.0 | 48.6 / 44.6 |
| same: LIBERO-PRO, 4 categories | 160 | 68.8 / 60.6 | 48.8 / 48.1 |
| `libero_max_160.tsv`: LIBERO-MAX Dynamic / Base | 80 / 80 | 61.2 / 70.0 host | 40.0 / 51.2 host |

How to read the table:
- **Host success** means the hidden goal predicate holds at the end of the episode.
- **Strict success** additionally requires that the agent called `finish()` itself.
- Sampling is at T = 1, so a re-run gives numbers within the binomial noise of these values.
- `eval/summarize.py` reproduces the table from the original episode directories exactly.

Details of the LIBERO-PRO Pos+Task protocol:
- Pos uses the released `*_swap` files and Task the `*_task` files, with init states 1–10.
- 5 of the 800 starts are physically invalid and were replaced by the next valid state, by a rule fixed before any model was run.
- The frozen states and the rule are stored with sha256 hashes in `CaPtain-artifacts/eval/libero_pro_postask/manifest.json`.

Hardware and time:
- One 80 GB GPU serves the policy and runs ≤ 8 concurrent episodes. A second GPU hosts SAM3 + Molmo2-8B (≈ 30 GB) and EGL rendering.
- With CaPtain-4B an episode takes a median of ~6 min of wall time (p90 ~12 min). The official 240 episodes take about 3–4 h on 8 parallel lanes.
- Each episode also needs about one CPU core for MuJoCo and the sandboxed Python.

## Hooks used by RL training (default-off)

These hooks are used by `../rl` and change nothing when they are off:
- `--mark-requests` (driver): before each model request, the driver asks the server to log a host-only `hidden-request-mark/v1` row in `hidden_request_marks.jsonl`. The row holds the request index, action index, time and the hidden object state (`adapters/libero/hidden_object_audit.py`). The simulator critic is computed from these rows.
- **Live marks:** if a file `.live_request_marks` exists in the episode output directory, marks are also appended as they are taken. The RL early stop uses this.
- `CAPTAIN_FAST_LOOP=1` / `CAPTAIN_LAZY_RENDER=1`: a faster MuJoCo control loop and on-demand camera rendering. Physics, tool responses and depth are bit-identical; RGB differs by ≤ 1 intensity level on < 0.05% of pixels, the same as renderer nondeterminism.
- **Per-state visual variation** (light intensity and position, a small agentview camera jitter, material tints) is applied only for fresh-init states whose sidecar carries a `visual` list. It is used for the wide-layout RL start pool and never for official seeds.

## Tests

```bash
.venv-libero/bin/python -m pytest -q tests/test_react_*.py tests/test_server_env_server.py tests/test_import_boundaries.py tests/test_transfer_wiring.py   # driver + server + transfer wiring, seconds
MUJOCO_GL=osmesa .venv-libero/bin/python -m pytest -q                                                                          # everything, ~20 min CPU
```

Some adapter tests in the full suite have long-standing failures that already exist in the evaluated snapshot; see `../RELEASE_NOTES.md`.

## Design rules

1. **Three layers, one direction.**
   - `adapters`/`ports` know the environment; `server` wraps one episode; `react` knows only workspaces, tools, text, images and a model API.
   - `react` must not import `server` or `adapters` and must not use robot, LIBERO or task vocabulary (`tests/test_import_boundaries.py`).
2. **Hidden state stays hidden.**
   - `check_success` appears once, at adapter construction.
   - Nothing served over HTTP carries poses, contacts or the predicate value.
   - The episode output directory, which holds the hidden audit files, is hidden from the sandbox.
3. **Budgets live in the server.** Every request counts. The driver stops on the wall deadline, the request cap, or when the model stops calling tools.
4. **Do not change model-visible text** (TASK.md, tool descriptions, receipts, request template) when you evaluate the released checkpoints. CaPtain-4B was trained on exactly this text.
   - The request template `react/prompts/captain.request.json` is a captured Codex-CLI code-mode request, including its system prompt ("You are Codex, ...").
   - The released models expect this template.

## Backends

| Suite(s) | Backend | Used for |
|---|---|---|
| `libero_spatial/object/goal/10/90` (+ `--libero-variant pro\|plus`, `--max-case`) | `adapters/libero`, `server/libero_setup.py`, `server/libero_variants.py` | training and the LIBERO / LIBERO-PRO / LIBERO-Plus / LIBERO-MAX evaluations (`eval/`) |
| `robotwin2_sim` | `adapters/robotwin2` (SAPIEN 3 scene server in RoboTwin's env) | RoboTwin 2.0, official Hard protocol ([`eval/transfer`](eval/transfer/README.md)) |
| `robodojo_sim` | `adapters/robodojo` (Isaac Sim scene server in RoboDojo's env) | RoboDojo, official layouts and action budget |
| `simplerenv_sim` | `adapters/simplerenv` (SAPIEN 2.2 scene server in SimplerEnv's env) | SimplerEnv, WidowX visual matching (Google Robot supported) |
| `capgym_{lift,stack,restack,nut,wipe}` | `adapters/capgym` | CaP-Bench tasks rebuilt in robosuite 1.4 |
| `hardware_real` | `adapters/hardware` | a real robot arm through a user-supplied driver |

The transfer benchmarks, their install steps, official protocol settings and the extra inputs the harness uses on each
of them are described in [`eval/transfer/README.md`](eval/transfer/README.md).

## Hardware backend (real robots)

`adapters/hardware/facade.py` defines the driver protocol (`XR1Driver`): `reset()`, `read()` (measured joints, joint
velocities, gripper opening, optional wrist force in robot_base), `command(grip_site_goal_4x4, gripper_action)` for one
control tick, `hold()` and `frames(camera_names)` (RGB, metric depth or None, intrinsics, optional extrinsics).
`XR1FacadeEnv` wraps such a driver into the environment the LIBERO action backend drives, with a MuJoCo kinematic twin
(MJCF of the arm, robosuite joint/site names) that is updated from the measured joints, so the tools, receipts, guards
and TASK.md are exactly those of the simulation runs. The RoboTwin 2.0, RoboDojo and SimplerEnv backends are drivers of
this interface. For a real arm, write a driver and a config (twin MJCF path, cameras with their links and depth source,
control rate, workspace box, calibration profile) and run:

```bash
CAPTAIN_PYTHON=.venv-transfer/bin/python eval/run_episode.sh hardware_real 0 0 runs/robot --server-arg=--hardware-config --server-arg=robot.json \
  --server-arg=--hardware-task --server-arg=task.json --server-arg=--hardware-driver --server-arg=my_robot.driver:make_driver
```

Cameras without a depth sensor can use a monocular metric-depth service (`depth: "mono"`, `adapters/hardware/mono_depth.py`).
There is no success predicate on a real robot; the outcome is labelled after the episode. In the paper, the real-robot
episodes used a driver for an XR1 arm (a head RGB-D camera, a wrist camera, safety guards on the robot side) and a
checkpoint adapted to that embodiment in a coarse twin simulation; that robot driver and checkpoint are not part of
this release.

## License

MIT (see `../LICENSE`). Upstream components (LIBERO, CaP-X lineage, SAM 3, Molmo2, LIBERO-PRO/Plus/MAX task files, the
transfer benchmarks) keep their own licenses; see `../THIRD_PARTY_LICENSES.md`.
