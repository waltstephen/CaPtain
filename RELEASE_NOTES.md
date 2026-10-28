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
  - The full test suite has the same failing set as the evaluated snapshot (99 failures and 26 errors, mostly legacy adapter tests for code paths the python-runtime contract no longer uses).
  - The driver and server tests (`tests/test_react_*.py`, `test_server_env_server.py`, `test_import_boundaries.py`) all pass.
- **RL.**
  - `rl/` is the code that trained CaPtain-4B. Apart from comments, help strings, module names and paths, the code is unchanged.
  - The 387 CPU unit tests pass.
  - `fresh_run init` with `rl/configs/captain-4b.json` and the reference start states from CaPtain-artifacts validates. The start bank it builds has exactly the reference sha256 (`df69fab6…`).
  - All training keys equal those of the reference run; only paths, hosts and layout differ.
- **SFT.**
  - `sft/train/train_sft.py` is functionally identical to the trainer of CaPtain-4B-SFT: only comments changed, an unused vision-probe callback was removed, and `--model` became required.
  - The CPU loss test passes.
  - A synthetic end-to-end dry run (packing → ordering → loss constant → trainer `--dry-run`) works.
- **Weights.** The SHA-256 of the uploaded safetensors equals that of the evaluated checkpoints.
- **Transfer backends** (`harness/src/captain_harness/adapters/{robotwin2,robodojo,simplerenv,capgym,hardware}`, `harness/eval/transfer/`).
  - Each backend is the code of the evaluated harness snapshot for that benchmark, with the package renamed:
    - RoboTwin 2.0: the frozen harness of the official Hard run (its source hash was re-computed and matches the run record);
    - RoboDojo: harness revision r9 of the 42-task run (the hashes of its changed files match the run record); its copy of
      the dual-arm tool adapter is kept as `adapters/robodojo/dual_arm.py`, because RoboTwin used a later revision;
    - SimplerEnv: the harness of the WidowX runs (after the last fix before the counted episodes);
    - CaP-Bench: the CaP-Gym backend of the counted runs.
  - The shared LIBERO/CaP-X modules gained the RoboDojo options as opt-in attributes (stiff-arm force model, step profile,
    gripper settle, re-tare, reach hints, an IK-branch check). They are inactive unless the RoboDojo builder sets them, so
    the LIBERO code path is unchanged; the CPU test subset that covers these modules has the same result as before the merge.
  - `harness/eval/transfer/summarize.py` reproduces the paper's RoboTwin 2.0 numbers (22.6 / 30.0 ever; 19.5 / 26.5 host;
    8.4 / 11.3 strict) from the saved per-episode rows, and the CaP-Bench numbers (Lift 100, Stack 50 at the horizon;
    mean 75.0) from the original episode directories.

## Known limitations

- **Transfer benchmarks were not re-run from this tree.** Their simulators (SAPIEN 3, Isaac Sim 5.1, SAPIEN 2.2) were not
  run during release preparation; the backends were checked by imports, CPU wiring tests and source hashes against the
  evaluated snapshots. RoboDojo episodes need a GPU with Vulkan raster support for Isaac Sim's camera annotators.
- **RoboTwin 2.0 task files.** The official seed list is shipped (`official_hard_seeds.tsv`); the task files with RoboTwin's
  instructions are rebuilt with RoboTwin's own expert check (`make_official_tasks.sh`, needs cuRobo and a GPU).
- **RoboDojo.** The paper reports the evaluated subset of layouts per task stated in its table; the shipped list is the full
  official 2,100-episode protocol. Unstable layouts are replaced with `replace_unstable.py`, not automatically.
- **Real robot.** The hardware backend interface and an episode builder (`hardware_real`, driver loaded from
  `--hardware-driver`) are included. The robot driver, the robot's twin model and the embodiment-adapted checkpoint used for
  the paper's real-robot episodes are not part of this release.

- **No GPU end-to-end run of the release tree.** The release code was not re-run on GPUs (vLLM serving, EGL episodes, RL rounds, SFT steps). Validation was done by unit tests, a config dry run, and reproducing the numbers from the original episodes.
- **Training data is not released yet.** This covers:
  - the SFT stage-2 mix (25,961 rows, 861M tokens, ~243k images);
  - the stage-1 mix;
  - the teacher episodes;
  - RL rollouts.
  
  The SFT stage-1 checkpoint (the init of stage 2) is also not released, so SFT is reproducible only once the data is published.
- **Software versions.** The trainer environments are documented from the recorded stage-1 / RL environments. The exact package set of every run was not frozen in a lock file.
- **Request template.** `react/prompts/captain.request.json` is a captured Codex-CLI code-mode request, including its system prompt ("You are Codex, an agent based on GPT-6 ..."). CaPtain-4B was trained on this exact text, so it is kept verbatim.
- **Benchmark licenses.** LIBERO-plus and LIBERO-MAX state no license. Before making the LIBERO-MAX case files (CaPtain-artifacts `eval/libero_max_cases/`) public, confirm their terms (see `THIRD_PARTY_LICENSES.md`).
- **LIBERO-10 task 5.** Its goal predicate has a ~3.5 mm offset in z, so a correct flat placement is scored as a failure. The task is kept in training and in all evaluations, as in the paper.
- **Held-out monitor.** The RL config disables the online held-out monitor of the reference run. It needed a second harness copy with held-out wide-layout states; it does not affect training.
- **Run length.** The reference RL run was continued past round 300. CaPtain-4B is the round-300 checkpoint; later rounds are not part of this release.
