# AGENTS.md — guide for coding agents working in this repository

This file is for AI coding agents (Codex, Claude Code, Cursor, …). Humans: see [README.md](README.md).

## What this repository is

CaPtain is a pipeline that turns a small vision-language model into a code-as-policy robot agent:
CaPtain-Harness (the environment the agent acts in) → SFT on agent trajectories → RL with a simulator critic.
It is meant to be used as infrastructure: run your own model in the harness, add a robot or benchmark, or retrain.

## Map

| You want to… | Go to |
|---|---|
| run one episode | `harness/eval/run_episode.sh <suite> <task_id> <seed> <out_dir>` |
| run / score a task list | `harness/eval/run_jobs.py <tasks.tsv> --out <dir>`, `harness/eval/summarize.py <tasks.tsv> <dir>` |
| change the agent loop, context compaction, model client | `harness/src/captain_harness/react/` |
| change what one episode does (budgets, gate, receipts plumbing) | `harness/src/captain_harness/server/` |
| change robot tools or a simulator backend | `harness/src/captain_harness/adapters/<backend>/`, contracts in `ports/` |
| add a robot or benchmark | new backend in `adapters/`, register the suite, task list in `harness/eval/`, `tests/test_transfer_wiring.py` |
| connect real hardware | implement the driver protocol of `adapters/hardware` (measured state, one control tick, hold, RGB-D frames + intrinsics) |
| run transfer benchmarks | `harness/eval/transfer/README.md` (one simulator environment per benchmark) |
| SFT | `sft/README.md`, `sft/train/launch_sft.sh`, `sft/configs/` |
| RL | `rl/README.md`, `rl/scripts/run_stream.sh rl/configs/<config>.json`; critic in `rl/physics_critic.py`, `rl/critic_cotrain.py` |

## Rules (do not break these)

1. **Layering.** `react` must not import `server` or `adapters` and must not contain robot, LIBERO or task vocabulary.
   Enforced by `harness/tests/test_import_boundaries.py`.
2. **Hidden state stays hidden.** Nothing served to the model may carry object poses, contacts or the success predicate.
   `check_success` is wired once, at adapter construction. The episode output directory (it holds hidden audit files) is
   hidden from the sandbox. The RL critic reads hidden state from host-only files, never through the model's tools.
3. **Budgets live in the server.** Request cap, wall time and action budget are enforced server-side; every request counts.
4. **Model-visible text is frozen for the released checkpoints.** Do not edit TASK.md templates, tool descriptions,
   receipts or `react/prompts/captain.request.json` when evaluating CaPtain-4B / CaPtain-4B-SFT. If you change them for a
   new model, say so and retrain.
5. **No hard-coded machines.** Paths, hosts and keys come from environment variables (`POLICY_URL`, `STUDENT_PYTHON`,
   `EGL_DEVICE`, `CAPTAIN_ROOT`, `CAPTAIN_WORKSPACE`, `CAPTAIN_API_KEY`, …). Never commit keys.
6. **RL hooks are default-off.** `--mark-requests`, live marks, `CAPTAIN_FAST_LOOP`, `CAPTAIN_LAZY_RENDER` must not change
   behaviour when off.

## Checks before you hand back

```bash
cd harness
.venv-libero/bin/python -m pytest -q tests/test_react_*.py tests/test_server_env_server.py \
  tests/test_import_boundaries.py tests/test_transfer_wiring.py        # seconds, CPU
MUJOCO_GL=osmesa .venv-libero/bin/python -m pytest -q                  # full suite, ~20 min CPU (known failures: RELEASE_NOTES.md)
```

For RL changes also run `rl/tests/` (needs the RL environment, see `rl/README.md`).

## Conventions

- Python 3.10 for the harness server/driver, Python 3.11 for the agent's sandboxed interpreter (exact versions matter).
- Success metrics: *host* = goal predicate on the final state; *strict* = host and the agent called `finish()`.
- Keep receipts one line and factual; the policy learns from them.
