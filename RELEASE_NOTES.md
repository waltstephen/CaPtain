# Release notes (v1.0.0)

## What was verified for this release

- **Evaluation tooling.**
  - `harness/eval/summarize.py`, run on the original evaluation episodes of CaPtain-4B with the task lists in `harness/eval/tasks/`, reproduces every number of the paper's main table:
    - LIBERO 73.8 / 67.5;
    - PRO Pos+Task 78.5 / 71.5 (6-cell) and 68.6 / 58.9 (8-cell);
    - Plus 66.1 / 59.0;
    - PRO 4-category 68.8 / 60.6;
    - MAX Dynamic / Base 61.2 / 70.0.
  - `harness/eval/run_episode.sh` uses exactly the command-line protocol of those evaluations.
- **Harness.**
  - The release harness is the evaluated snapshot with these changes:
    - packages and identifiers renamed;
    - the provider-specific Responses backend removed (`--auth none|bearer`);
    - the LIBERO-PRO / Plus / MAX variant options and the default-off RL hooks added. These are the request marks, live marks, fast loop and visual variation, and they were used by the RL run.
  - None of these changes touch model-visible text (TASK.md, tool descriptions, receipts, request template).
  - The driver and server tests (`tests/test_react_*.py`, `test_server_env_server.py`, `test_import_boundaries.py`) all pass.
- **RL.**
  - `rl/` is the code that trained CaPtain-4B. Apart from comments, help strings, module names and paths, the code is unchanged.
  - The 387 CPU unit tests pass.
  - `fresh_run init` with `rl/configs/captain-4b.json` and the reference start states from CaPtain-artifacts validates. The start bank it builds has exactly the reference sha256 (`df69fab6…`).
  - All training keys equal those used for CaPtain-4B; only paths, hosts and layout differ.
- **SFT.**
  - `sft/train/train_sft.py` is functionally identical to the trainer of CaPtain-4B-SFT: only comments changed, an unused vision-probe callback was removed, and `--model` became required.
  - The CPU loss test passes.
  - A synthetic end-to-end dry run (packing → ordering → loss constant → trainer `--dry-run`) works.
- **Weights.** The SHA-256 of the uploaded safetensors equals that of the evaluated checkpoints.
- **Transfer backends** (`harness/src/captain_harness/adapters/{robotwin2,robodojo,simplerenv,capgym,hardware}`, `harness/eval/transfer/`).
  - Each backend is the code of the evaluated harness snapshot for that benchmark, with the package renamed:
    - RoboTwin 2.0: the harness of the official Hard evaluation;
    - RoboDojo: the harness of the 42-task evaluation; its dual-arm tool adapter is kept as `adapters/robodojo/dual_arm.py`;
    - SimplerEnv: the harness of the WidowX evaluation;
    - CaP-Bench: the CaP-Gym backend of the evaluation.
  - The shared LIBERO/CaP-X modules gained the RoboDojo options as opt-in attributes (stiff-arm force model, step profile,
    gripper settle, re-tare, reach hints, an IK-branch check). They are inactive unless the RoboDojo builder sets them, so
    the LIBERO code path is unchanged; the CPU test subset that covers these modules has the same result as before the merge.
  - `harness/eval/transfer/summarize.py` reproduces the paper's RoboTwin 2.0 numbers (22.6 / 30.0 ever; 19.5 / 26.5 host;
    8.4 / 11.3 strict) from the saved per-episode rows, and the CaP-Bench numbers (Lift 100, Stack 50 at the horizon;
    mean 75.0) from the original episode directories.

## Released artifacts

- Code: this repository and [CaPtain-Harness](https://github.com/waltstephen/CaPtain-Harness) (MIT).
- Models: [CaPtain-4B](https://huggingface.co/YijiaFan/CaPtain-4B) and [CaPtain-4B-SFT](https://huggingface.co/YijiaFan/CaPtain-4B-SFT).
- Data: [CaPtain-SFT-Data](https://huggingface.co/datasets/YijiaFan/CaPtain-SFT-Data) (SFT trajectories and visual QA) and
  [CaPtain-artifacts](https://huggingface.co/datasets/YijiaFan/CaPtain-artifacts) (critic weights, configs, start states, example episodes).

## Notes

- **RoboTwin 2.0 task files.** The official seed list is shipped (`official_hard_seeds.tsv`); the task files with RoboTwin's
  instructions are rebuilt with RoboTwin's own expert check (`make_official_tasks.sh`, needs cuRobo and a GPU).
- **RoboDojo.** The shipped list is the full official 2,100-episode protocol. Unstable layouts are replaced with
  `replace_unstable.py`. RoboDojo needs a GPU with Vulkan raster support for Isaac Sim's camera annotators.
- **Real robot.** The hardware backend interface and an episode builder (`hardware_real`) are included; load your robot
  driver with `--hardware-driver`.
- **Request template.** `react/prompts/captain.request.json` is a captured Codex-CLI code-mode request, including its
  system prompt ("You are Codex, an agent based on GPT-6 ..."). CaPtain-4B was trained on this exact text, so it is kept verbatim.
