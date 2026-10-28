# Zero-shot transfer benchmarks

CaPtain-4B is trained only on LIBERO. The paper evaluates it without further training on four benchmarks whose scenes and
tasks differ from LIBERO: RoboTwin 2.0, RoboDojo, SimplerEnv and CaP-Bench. This directory holds the evaluation drivers,
task lists, protocol tools and scorers for these runs. The backends are part of CaPtain-Harness:

| Benchmark | Suite | Backend (`src/captain_harness/...`) | Simulator (own environment) | Embodiment used |
|---|---|---|---|---|
| RoboTwin 2.0 | `robotwin2_sim` | `adapters/robotwin2` | RoboTwin 2.0 @ ea8b211, SAPIEN 3 | dual Franka Panda (RoboTwin's `franka-panda` embodiment) on the official layouts |
| RoboDojo | `robodojo_sim` | `adapters/robodojo` | RoboDojo @ cb2a37d, Isaac Sim 5.1 + IsaacLab | dual ARX X5 (RoboDojo's own embodiment) |
| SimplerEnv | `simplerenv_sim` | `adapters/simplerenv` | SimplerEnv @ eddb569 + ManiSkill2_real2sim @ ef7a4d4, SAPIEN 2.2.2 | WidowX 250s (Bridge); Google Robot also supported |
| CaP-Bench | `capgym_{lift,stack,restack,nut,wipe}` | `adapters/capgym` | robosuite 1.4 in the LIBERO venv | Franka Panda |

The policy-facing side is identical to the LIBERO evaluation: the same `python` tool, the same 22 capabilities plus
`finish()`, the same receipts, the same TASK.md (only the one-paragraph robot introduction, the relation line that depends
on the camera direction and, for RoboTwin and RoboDojo, a short benchmark guide differ), the same request template and the
same sampling (T = 1, top_p = 1, thinking on, 64k-token window). The RoboTwin, RoboDojo and SimplerEnv backends are drivers
of the hardware-backend interface (`adapters/hardware/facade.py`): the simulator runs in its own Python environment and
talks to the harness over a pipe; a MuJoCo kinematic twin built from the robot's own URDF gives forward kinematics and IK.

## Common setup

1. Install the harness, the student Python and the perception services as in [`../../README.md`](../../README.md).
2. Harness interpreter for RoboTwin / RoboDojo / SimplerEnv (the twins need MuJoCo 3; CaP-Bench uses `.venv-libero`):
   ```bash
   python3.11 -m venv .venv-transfer && .venv-transfer/bin/pip install -r requirements-transfer.txt
   ```
3. Serve the policy (`services/serve_policy.sh <hf-user>/CaPtain-4B 8600`) and start SAM3 + Molmo2-8B (`services/start_perception.sh`).
4. Run a task list and score it:
   ```bash
   export POLICY_URL=http://127.0.0.1:8600/v1 STUDENT_PYTHON=~/envs/student-py311/bin/python SIM_GPU=0
   python3 eval/transfer/run_jobs.py <benchmark> <tasks.jsonl | task dir> --out runs/captain-4b/<benchmark> --parallel 8
   python3 eval/transfer/summarize.py <benchmark> runs/captain-4b/<benchmark>
   ```
   `run_jobs.py` orders the jobs round-robin over tasks, skips finished episodes and retries infrastructure errors (they are
   never scored). `--limit N` runs a balanced prefix. A single episode is `eval/transfer/run_episode.sh <benchmark> <task.json> <out_root>`.

Budgets per episode: 60 model requests and 1,080 s of wall time (RoboTwin 2.0, SimplerEnv, CaP-Bench); 120 requests and
3,600 s (RoboDojo, whose Isaac Sim ticks are slower). Every benchmark allows 150 physical actions and 600 tool calls.

Benchmark config templates are in `configs/benchmarks/`; `run_episode.sh` renders them with this machine's paths
(`eval/transfer/render_config.py`, environment variables listed below). Calibration profiles are in `configs/calibration/`.

## Inputs beyond each benchmark's official policy observation

CaPtain-Harness gives the agent the same kinds of measurements on every backend. Compared with the official observation
of each benchmark, the agent additionally uses the following. It never receives object poses, the success predicate,
task progress or any other simulator ground truth; these are read only host-side, for scoring.

| Benchmark | Official policy observation | Additionally used by CaPtain-Harness |
|---|---|---|
| RoboTwin 2.0 | RGB (head + wrist cameras), camera intrinsics/extrinsics, joint positions, end-effector pose, commanded gripper value | simulated depth (head and wrist), joint velocities, measured finger opening, a wrist contact-force signal (summed SAPIEN contact impulses on the gripper links), SAM3 + Molmo2-8B perception services |
| RoboDojo | 3 RGB cameras, joint and end-effector state, commanded gripper value | simulated depth images, camera intrinsics/extrinsics, a simulated wrist force/torque reading (PhysX joint-6 wrench; the X5 has no wrist sensor), measured gripper aperture, SAM3 + Molmo2-8B |
| SimplerEnv | one RGB camera (with the real-image overlay in visual matching) | rendered depth and the intrinsics/extrinsics of the same camera, an added simulated wrist camera on the gripper (512x512, fovy 75 deg, RGB-D; not part of the official protocol), a wrist contact-force signal (contact impulses on the gripper links), SAM3 + Molmo2-8B |
| CaP-Bench | one RGB-D camera (`robot0_robotview`) and, depending on the tier, privileged poses or perception APIs | the LIBERO harness inputs: head RGB-D at CaP-Gym's camera pose, the robosuite wrist camera, calibrated wrist force; no privileged poses |

## RoboTwin 2.0

**Install** (Linux, NVIDIA GPU with Vulkan for SAPIEN rendering):
```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin && cd RoboTwin && git checkout ea8b21121ebb3cd201ff5b3fe361944ac94eda3f
git apply <harness>/eval/transfer/robotwin2/robotwin.patch          # rasterization switch, gripper-only planner without cuRobo
python3 <harness>/eval/transfer/robotwin2/patch_unstable.py .       # ROBOTWIN_ALLOW_UNSTABLE (see below)
python3.10 -m venv ../rtw-env && ../rtw-env/bin/pip install torch==2.4.1 torchvision==0.19.1 && ../rtw-env/bin/pip install -r scripts/requirements.txt
# apply the sapien urdf_loader and mplib source fixes that RoboTwin's own install script applies
# assets: embodiments.zip and objects.zip from the TianxingChen/RoboTwin2.0 dataset on Hugging Face, unzipped into assets/
# official seed check only (make_official_tasks.sh): cuRobo v0.7.8 built for your GPU (e.g. TORCH_CUDA_ARCH_LIST=8.0+PTX),
# then python scripts/update_embodiment_config_path.py
export ROBOTWIN_ROOT=$PWD ROBOTWIN_PYTHON=$PWD/../rtw-env/bin/python
```

**Official Hard protocol.** The paper uses RoboTwin's own evaluation protocol (`scripts/eval_policy_xpolicylab.py`):
task config `demo_randomized` (cluttered tables, random backgrounds and lighting, table height +-3 cm), seeds from 100000,
and a seed counts only if RoboTwin's expert program (`play_once` with the aloha-agilex embodiment and the cuRobo planner)
succeeds on it; unstable and expert-error seeds are skipped. The first 100 valid seeds per task give 5,000 episodes; the
instruction is RoboTwin's `build_instruction` (unseen templates for Hard). `robotwin2/official_hard_seeds.tsv` is the
resulting seed list. Build the task files (instructions included) with RoboTwin's own code:
```bash
eval/transfer/robotwin2/make_official_tasks.sh runs/robotwin2_official   # expert check per seed (GPU), then the first 100 valid seeds;
                                                                        # checks the result against official_hard_seeds.tsv
python3 eval/transfer/run_jobs.py robotwin2 runs/robotwin2_official/tasks --out runs/captain-4b/robotwin2 --parallel 8
python3 eval/transfer/summarize.py robotwin2 runs/captain-4b/robotwin2
```

**Embodiment and layout.** The agent controls two Franka Pandas (RoboTwin's `franka-panda` embodiment) instead of the
aloha-agilex arms; the instance is the official one. The harness builds the scene with the random state that the official
aloha-agilex setup would have at object placement, so objects, layout and instruction equal the official instance
(`adapters/robotwin2/sapien_server.py`). Both Franka bases are moved 0.10 m toward the table so the layouts are inside
the Panda's top-down reach. Task files set `allow_unstable`: such a scene was already accepted by the official expert
check, and the patched RoboTwin keeps it if objects settle slightly differently next to the Franka arms.

**Harness behaviour on this backend** (`configs/benchmarks/robotwin2_franka_dual.json`): an optional `arm` argument on the
single-arm tools, and calls without it go to the arm whose base is nearer the target unless the active arm holds an
object; contact detection recalibrated for the Franka gripper; a tilted re-approach when IK fails, the gripper closing
on contact and a grasp check in the receipt; an automatic hand-over for tasks that pass an object between arms; after a
contact stop the gripper holds the measured opening and the arm settles compliantly; an idle arm near the target is moved
home first (reported in the receipt); a short benchmark guide with generic procedures in TASK.md.

**Metrics.** ever = RoboTwin's `check_success` true after some physical action (the official evaluation loop stops at
the first success); host = final state; strict = host and the agent's own `finish()`. Single-arm subset = the 33 tasks whose
expert program moves one arm at a time and holds each object with one arm (`robotwin2/task_arm_classes.json`).
Paper (CaPtain-4B, 5,000 episodes): ever 22.6% (95% CI 21.5-23.8) on all 50 tasks and 30.0% on the single-arm subset;
host 19.5 / 26.5; strict 8.4 / 11.3. `summarize.py robotwin2 --rows <rows.json>` re-aggregates saved per-episode rows.

## RoboDojo

**Install.** Isaac Sim 5.1 in a Python 3.11 venv (`pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url
https://pypi.nvidia.com`), then RoboDojo (github.com/RoboDojo-Benchmark/RoboDojo @ cb2a37d) with the IsaacLab and cuRobo
revisions it pins, installed into that venv as described in RoboDojo's README, and RoboDojo's assets from Hugging Face
(including `Assets/Eval_Layout/RoboDojo/arx_x5/0`). Create a script that activates the Isaac Sim venv and puts the RoboDojo
checkout on `PYTHONPATH`, then:
```bash
export ROBODOJO_ROOT=/path/to/RoboDojo ROBODOJO_ENV_SH=/path/to/robodojo_env.sh
python3 eval/transfer/run_jobs.py robodojo eval/transfer/robodojo/official_2100.jsonl --out runs/captain-4b/robodojo --parallel 4
python3 eval/transfer/summarize.py robodojo runs/captain-4b/robodojo
```
The scene server (`adapters/robodojo/rd_server.py`, started through `rd_server.sh`) runs one episode per process under
RoboDojo's Isaac Sim Python and uses one GPU (`SIM_GPU`). Isaac Sim's camera annotators need a GPU with Vulkan raster
support; GPUs without it need a vendor-specific workaround that is not part of this release.

**Official protocol.** RoboDojo's own eval-client protocol (`src/eval_client/main.py`, `SeedManager`, the task's
`_task.yml`): for each of the 42 tasks the first `eval_nums` official layouts of `Eval_Layout/RoboDojo/arx_x5/0` in index
order, layout seed 0; 50 per task, and the 12 Generalization tasks use 25 fixed plus 25 `<task>_random` layouts; 2,100
episodes (`robodojo/official_2100.jsonl`). A layout that RoboDojo rejects as unstable is replaced by the task's next layout
index, as eval_client does (`robodojo/replace_unstable.py`). Scoring is RoboDojo's own: **official** = RoboDojo's verdict
and process Score at the task's native action budget (`step_lim`), taken from a snapshot of RoboDojo's reward state at the
budget; **unlimited** = the same episode continued up to 20x the native budget (the native budget is not enforced for
this column). Single-arm subset = the 32 tasks whose specification needs no simultaneous two-arm action, handover or navigation.

**Harness behaviour on this backend** (`configs/benchmarks/robodojo_x5.json`): the success check does not end the episode
(RoboDojo's official check is final and runs once, after the agent loop); wrist force estimated from the last arm joint;
speed limits and IK initialisation for the X5 arm; return to the home pose before the official check; a transparent grasp
retry; a finish instruction and a short guide in TASK.md; auto-arm routing, move_both, auto-tilt, grasp feedback and
auto-relay as on RoboTwin; generic guard fixes for the X5 wrist-force contact guards (`adapters/robodojo/r9_guard.py`, each
behind its own flag in `control.guard_fixes`).

## SimplerEnv

**Install** (NVIDIA GPU with Vulkan; SAPIEN 2.2 needs the NVIDIA Vulkan ICD and the glvnd EGL vendor file):
```bash
git clone https://github.com/simpler-env/SimplerEnv --recurse-submodules && cd SimplerEnv && git checkout eddb5690749a1e34267807c0f7b5eede323918d7
(cd ManiSkill2_real2sim && git checkout ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682)
python3.11 -m venv ../se-env && ../se-env/bin/pip install "setuptools<70" -r <harness>/eval/transfer/simplerenv/requirements-simplerenv.txt
../se-env/bin/pip install -e ManiSkill2_real2sim && ../se-env/bin/pip install -e .
export SIMPLERENV_ROOT=$PWD SIMPLERENV_PYTHON=$PWD/../se-env/bin/python   # optional: SIMPLERENV_EGL_VENDOR_FILE
python3 <harness>/eval/transfer/run_jobs.py simplerenv <harness>/eval/transfer/simplerenv/bridge_vm_96.jsonl --out runs/captain-4b/simplerenv
python3 <harness>/eval/transfer/summarize.py simplerenv runs/captain-4b/simplerenv
```

**Official protocol.** The episode list is SimplerEnv's own: the official scripts (`scripts/rt1x_bridge.sh`,
`scripts/rt1_*_{visual_matching,variant_agg}.sh`) were executed with a shim that records each `main_inference.py` argv
(`simplerenv/official/*.argv`), and every invocation is expanded with SimplerEnv's `get_args()` exactly like
`maniskill2_evaluator()` (`simplerenv/gen_tasks.py`). The paper reports the WidowX / Bridge visual-matching set: spoon on
towel, carrot on plate, stack the green block on the yellow block, eggplant in basket, 24 episodes each
(`simplerenv/bridge_vm_96.jsonl`), with the official `3rd_view_camera` and real-image overlay. `gen_tasks.py` also writes
the Google Robot visual-matching and variant-aggregation lists.

**Scoring.** host = SimplerEnv's own `evaluate()` at the end of the episode (the paper's number). Our episodes end at the
agent's `finish()`, the request cap or the wall budget, not at SimplerEnv's `max_episode_steps`; the verdict at the official
step horizon is reported as `horizon` (our agent moves slowly and deliberately, so it is near 0 by construction). Also
reported: ever (success at any official control period), strict, and Bridge grasp (`is_src_obj_grasped` at any step).
Aggregation follows SimplerEnv's `calc_metrics_evaluation_videos.py`. Paper (CaPtain-4B, 96 episodes): 83.3% (spoon 75.0,
carrot 79.2, stack 91.7, eggplant 87.5); 4.2% at the official 60/120-step horizon.

**Harness behaviour on this backend** (`configs/benchmarks/simplerenv.json`): the official delta end-effector controller
is bypassed; the harness commands arm joint targets through the robot's own PD drives (official passive-force
compensation and force limits, servo gains 1e4/5e2) and the official gripper joints; SimplerEnv's `evaluate()` runs once
per official control period of simulated time, so its counters keep their official timing. Images are letterboxed to
512x512 without cropping (intrinsics adjusted exactly). The two-stage apple task gives both official instructions at once.

## CaP-Bench (CaP-Gym tasks rebuilt in robosuite 1.4)

CaP-Bench (CaP-X) pins robosuite 1.5, which conflicts with LIBERO's robosuite 1.4. The five single-arm CaP-Gym tasks (Cube
Lift, Cube Stack, Cube Re-stack, Peg/Nut insertion, Spill Wipe) are rebuilt on robosuite 1.4 with CaP-Gym's arena,
objects, placement samplers and success predicates (`adapters/capgym/environment.py`); no extra install is needed.
```bash
python3 eval/transfer/run_jobs.py capbench eval/transfer/capbench/capbench_5x20.jsonl --out runs/captain-4b/capbench --parallel 4
python3 eval/transfer/summarize.py capbench runs/captain-4b/capbench
```
**Protocol.** 20 trials per task (seeds 0-19; trial k seeds the robosuite placement with k, so trials are reproducible);
the head camera is moved onto CaP-Gym's camera pose (`robot0_robotview`; birdview for Nut); task sentence = CaP-X's goal
sentence; CaP-Gym's wrist flip (`qpos[6] -= pi` at reset) is disabled (`CAPGYM_WRIST_FLIP=1` restores it).
**Scoring.** host = the env's `_check_success()` on the final state; **capped** = the predicate at CaP-Gym's step horizon
(1,500 control steps for Lift/Stack/Re-stack, counted after the harness's force calibration; Nut and Wipe as in CaP-Gym),
which is what CaP-Gym itself scores; strict = host and `finish()`. Paper (CaPtain-4B): Cube Lift 20/20 and Cube Stack 19/20
on the final state; at the horizon Lift 100 and Stack 50, mean 75.0. Re-stack 7/20 (host), Nut 0/20, Wipe 0/20: Wipe needs
a sustained press into the table, which the guarded motion primitives do not provide.
