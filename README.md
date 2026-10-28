# CaPtain: A Pipeline for Post-Training Code-as-Policy Robot Agents

**Project page:** coming soon | **Paper:** coming soon | **Models (Hugging Face):** coming soon | **Video:** coming soon

CaPtain is an open pipeline that post-trains a small open vision-language model into a code-as-policy (CaP) robot agent.
The agent controls a robot by writing Python against low-level perception and motion primitives, looks at images and at
the receipts of its own robot calls, and decides itself when the task is done. The pipeline has three steps:

1. **CaPtain-Harness** exposes only low-level perception and motion primitives, with no high-level skills, and the same API
   runs in simulation and on real robots.
2. **Supervised fine-tuning (SFT)** on synthetic agent trajectories collected in this harness gives a cold start.
3. **Reinforcement learning (RL)** with a **simulator critic**: a small critic reads the simulator's physical state (object
   poses, grasp state, sub-goal progress) and turns the change in its predicted success across each model request into
   that request's credit. The policy never sees this state.

Applied to Qwen3.5-4B, which scores 0% zero-shot, the pipeline gives **CaPtain-4B-SFT** and **CaPtain-4B**:
- held-out LIBERO success rises to 55.4% after SFT and 73.8% after RL;
- the gain persists on three perturbation suites (LIBERO-PRO, LIBERO-Plus, LIBERO-MAX; +17.6 to +21.2 points), with 78.5 on
  LIBERO-PRO Pos+Task;
- trained only on LIBERO, the agent transfers zero-shot to RoboTwin 2.0, RoboDojo, SimplerEnv and CaP-Bench;
- through a hardware backend, the pipeline extends to a real robot without real-world training data.

At test time CaPtain-4B uses neither a frontier model nor a LIBERO-trained VLA.

## Repository layout

| Part | What it is |
|---|---|
| [`harness/`](harness/README.md) | **CaPtain-Harness**: the `python` tool (a persistent sandboxed session), the 22-capability robot API plus `finish()`, one-line receipts, the completion gate, context compaction, perception services, and the backends: LIBERO (with LIBERO-PRO / LIBERO-Plus / LIBERO-MAX), RoboTwin 2.0, RoboDojo, SimplerEnv, CaP-Bench and a hardware backend interface for real robots. Also the evaluation runners, task lists and scorers of the paper. |
| [`harness/eval/transfer/`](harness/eval/transfer/README.md) | The zero-shot transfer evaluations: install notes, official protocol settings, task lists, run and scoring scripts per benchmark. |
| [`sft/`](sft/README.md) | SFT code for **CaPtain-4B-SFT**: data format, helper-library code normalisation, trainer (DeepSpeed ZeRO-3), launch script and the exact training configuration. |
| [`rl/`](rl/README.md) | RL code for **CaPtain-4B**: streaming rollout and environment servers, the simulator critic (privileged simulator state at request boundaries, co-trained online), advantages (GAE + group-relative outcome), the SAO-style update with KL to the SFT model, the trainer, the run lifecycle and the configuration of the final run. |

```
                 synthetic agent trajectories (collected in CaPtain-Harness)
                                   |
 Qwen3.5-4B --SFT (sft/)--> CaPtain-4B-SFT --RL in LIBERO with the simulator critic (rl/)--> CaPtain-4B
                                   |                                                          |
                                   +------------- evaluated in CaPtain-Harness (harness/) ----+
          LIBERO, LIBERO-PRO, LIBERO-Plus, LIBERO-MAX | RoboTwin 2.0, RoboDojo, SimplerEnv, CaP-Bench | real robot
```

## Models and artifacts

| Name | Contents |
|---|---|
| CaPtain-4B | RL model (round 300 = 75 optimizer updates), bf16, Qwen3.5-4B architecture |
| CaPtain-4B-SFT | SFT model; RL start point and KL reference |
| CaPtain-artifacts | simulator-critic warm start and feature spec; RL/SFT configs; the 7,360 RL start states and start-pool manifests; LIBERO-PRO Pos+Task frozen init states; LIBERO-MAX cases; example episodes |

The Hugging Face repositories are coming soon; the commands below use `<hf-user>/CaPtain-4B` as a placeholder. The SFT data
and RL rollouts will be released with them.

## Results (simulation, success %)

LIBERO family, host / strict (strict additionally requires that the agent called `finish()` itself):

| Benchmark | Episodes | Qwen3.5-4B | CaPtain-4B-SFT | **CaPtain-4B** |
|---|---|---|---|---|
| LIBERO, official held-out states 11-16 (4 suites x 10 tasks) | 240 | 0.0 / 0.0 | 55.4 / 52.1 | **73.8 / 67.5** |
| LIBERO-PRO Pos+Task, 6-cell mean | 600 | 0.0 / 0.0 | 59.5 / 56.7 | **78.5 / 71.5** |
| LIBERO-PRO Pos+Task, 8-cell mean | 800 | 0.0 / 0.0 | 49.2 / 45.1 | **68.6 / 58.9** |
| LIBERO-Plus, 7 categories | 700 | 0.0 / 0.0 | 48.6 / 44.6 | **66.1 / 59.0** |
| LIBERO-MAX Dynamic | 80 | 0.0 / 0.0 | 40.0 / 40.0 | **61.2 / 50.0** |
| LIBERO-MAX Base | 80 | 0.0 / 0.0 | 51.2 / 45.0 | **70.0 / 63.8** |

Zero-shot transfer of CaPtain-4B (trained only on LIBERO):

| Benchmark | Protocol | CaPtain-4B |
|---|---|---|
| RoboTwin 2.0 | official Hard protocol, 50 tasks x 100 trials, dual Franka Panda | 22.6 (all 50 tasks), 30.0 (33 single-arm tasks) |
| RoboDojo | 42 tasks, official layouts and native action budget, dual ARX X5; leaderboard Score / success | 1.7 / 0.8 (all), 2.0 / 1.0 (32 single-arm tasks) |
| SimplerEnv | WidowX (Bridge) visual matching, 4 tasks x 24 episodes | 83.3 |
| CaP-Bench | Cube Lift and Cube Stack, 20 trials each, at CaP-Gym's step horizon | 75.0 (CaPtain-4B-SFT 55.0, Qwen3.5-4B 5.0) |

Protocol: T = 1, top_p = 1, thinking on, a 64k-token context with compaction, SAM3 + Molmo2-8B perception, at most 60 model
requests and 1080 s per episode (RoboDojo: 120 requests, 3600 s). Host success means the hidden goal predicate holds at
the end of the episode. On the transfer benchmarks the harness also uses inputs that the benchmarks' official policy
observations do not include (simulated depth, camera calibration, a wrist force or contact signal, and on SimplerEnv an
added wrist camera); they are listed per benchmark in [`harness/eval/transfer/README.md`](harness/eval/transfer/README.md).
The agent never receives object poses, the success predicate or task progress.

## Installation

Linux x86_64 with NVIDIA GPUs (EGL for MuJoCo; Vulkan for the SAPIEN / Isaac Sim transfer benchmarks), `bubblewrap` >= 0.5.

```bash
git clone <this repository> CaPtain && cd CaPtain/harness
scripts/setup_third_party.sh                                   # LIBERO @ pinned revision into third_party/
python3.10 -m venv .venv-libero && .venv-libero/bin/pip install -r requirements-libero.txt \
  && .venv-libero/bin/pip install -e third_party/libero --no-deps && .venv-libero/bin/pip install -e . --no-deps
python3.11 -m venv ~/envs/student-py311 && ~/envs/student-py311/bin/pip install -r requirements-student.txt   # the agent's sandboxed Python
python3.11 -m venv .venv-transfer && .venv-transfer/bin/pip install -r requirements-transfer.txt              # only for RoboTwin / RoboDojo / SimplerEnv
```

Perception services (SAM3, Molmo2-8B) and the policy server (vLLM 0.29) are described in [`harness/README.md`](harness/README.md).
Each transfer benchmark's simulator is installed in its own environment; see [`harness/eval/transfer/README.md`](harness/eval/transfer/README.md).
Training environments are described in [`sft/README.md`](sft/README.md) and [`rl/README.md`](rl/README.md).

## Quick start: evaluate CaPtain-4B on one LIBERO episode

```bash
cd harness
services/start_perception.sh                                    # SAM3 :8114 + Molmo2-8B :8122
services/serve_policy.sh <hf-user>/CaPtain-4B 8600              # vLLM 0.29, OpenAI-compatible endpoint
STUDENT_PYTHON=~/envs/student-py311/bin/python eval/run_episode.sh libero_spatial 0 11 runs/demo
cat runs/demo/libero_spatial_0_seed-11/episode_result.json      # host_final_success, agent_finished, termination_reason, ...
```

The model endpoint is any OpenAI-compatible server (`POLICY_URL`, default `http://127.0.0.1:8600/v1`; `--auth bearer`
reads `CAPTAIN_API_KEY` or `OPENAI_API_KEY`). Full benchmark runs: [`harness/README.md`](harness/README.md) (LIBERO family)
and [`harness/eval/transfer/README.md`](harness/eval/transfer/README.md) (transfer). Training: [`sft/README.md`](sft/README.md), [`rl/README.md`](rl/README.md).

## Real robots

The harness reaches a robot through a driver that implements a small protocol (read measured state, command one control
tick with an absolute end-effector goal and a gripper action, hold, read camera frames with depth and intrinsics). A MuJoCo
kinematic twin updated from the measured joints provides kinematics, so the tools, receipts and TASK.md are those of the
simulation runs. See "Hardware backend" in [`harness/README.md`](harness/README.md).

## Hardware used for the paper

| Stage | Hardware | Time |
|---|---|---|
| SFT (stage 2) | 16 x H100-80GB | ~7.6 h (~120 GPU-hours) |
| RL (CaPtain-4B, 300 rounds of 256 episodes) | 16 x H100-80GB: 8 for the trainer; 8 for rollout (6 vLLM policies, SAM3, Molmo2-8B, EGL rendering) | ~15-20 min per round; about 1,290 GPU-hours |
| Evaluation | 1 GPU for the policy plus 1 for perception and rendering | ~6 min median per LIBERO episode |

## License

The code in this repository is released under the [MIT License](LICENSE). Third-party components keep their own licenses;
see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) (note that SAM 3 weights fall under the SAM License). The model
weights follow the license of the base model, Qwen3.5-4B (Apache-2.0). Known limitations of this release are listed in
[RELEASE_NOTES.md](RELEASE_NOTES.md).

## Citation

```bibtex
@article{fan2026captain,
  title   = {CaPtain: A Pipeline for Post-Training Code-as-Policy Robot Agents},
  author  = {Fan, Yijia and He, Xilin and Jiang, Yilei and Wu, Dongming and Wang, Jiaxu and Yue, Xiangyu},
  journal = {arXiv preprint},
  year    = {2026},
  note    = {arXiv identifier coming soon}
}
```
