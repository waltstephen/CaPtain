"""Checkpoint provenance checks and recoverable log trimming for RL restarts."""
from __future__ import annotations

import json
import os
import re
import stat
import tempfile
import time
import uuid
from pathlib import Path


class ResumeError(ValueError):
    pass


CRITICAL_ARGUMENTS = (
    "lr", "kl_beta", "adv_std_floor", "lora_r", "lora_alpha", "max_grad_norm",
    "warmup_steps", "prompts_per_step", "num_generations", "suites", "tasks",
    "seeds", "seed", "max_seq_tokens", "hard_max_tokens", "grad_accum_rounds",
    "act_cost", "fail_cost", "act_ref", "missed_finish_return", "exhaust_reward",
)
SEGMENT_ARGUMENTS = ("segment_bank", "segment_horizon", "curriculum_episode_fraction", "curriculum_frontier_step",
                     "curriculum_advance_fraction", "curriculum_mode", "curriculum_sourced_only", "curriculum_active_fraction",
                     "curriculum_active_min_delta", "curriculum_fresh_horizon", "curriculum_suites", "probe_set",
                     "curriculum_focus_set", "segment_budget", "segment_min_live", "segment_handoff", "score_centering",
                     "full_finetune", "fresh_unique_states", "curriculum_manifest", "boundary_fresh_only", "reward_mode",
                     "loss_normalization", "skip_zero_signal", "warmup_attempts", "code_reward", "code_reward_weight",
                     "code_reward_budget", "advantage_shuffle", "advantage_mode", "critic_dir", "gae_lambda", "ratio_mask_low",
                     "ratio_mask_high", "start_weights", "critic_cotrain", "critic_hidden", "critic_lr", "critic_updates")
# arguments added after runs started: a checkpoint that predates them resumes with these values
LATER_ARGUMENT_DEFAULTS = {"grad_accum_rounds": 1, "act_cost": 0.0, "fail_cost": 0.0, "act_ref": None, "missed_finish_return": None,
                           "exhaust_reward": None, "curriculum_mode": "reverse", "curriculum_sourced_only": False, "curriculum_active_fraction": 0.0,
                           "curriculum_active_min_delta": 0.0, "curriculum_fresh_horizon": None, "curriculum_suites": None,
                           "probe_set": None, "curriculum_focus_set": None, "segment_budget": None, "segment_min_live": 4,
                           "segment_handoff": "continuation", "score_centering": False, "full_finetune": False,
                           "fresh_unique_states": False, "curriculum_manifest": None, "boundary_fresh_only": False,
                           "reward_mode": None, "loss_normalization": "token_mean", "skip_zero_signal": False, "warmup_attempts": 0,
                           "code_reward": None, "code_reward_weight": 0.0, "code_reward_budget": 40,
                           "advantage_shuffle": False, "advantage_mode": "grpo", "critic_dir": None, "gae_lambda": 0.0,
                           "ratio_mask_low": None, "ratio_mask_high": None, "start_weights": None, "critic_cotrain": False,
                           "critic_hidden": "current", "critic_lr": 1e-4, "critic_updates": 2}
# Pinned resume-boundary switches of otherwise critical arguments, each
# accepted only along one pinned pair (either way round: forward / rollback); the same pins as fresh_run.TRAIN_KEY_ALLOWANCES /
# START_WEIGHTS_ALLOWANCES. kl_beta: saved -> requested value; start_weights: sha256 of the weights file in the saved curriculum state
# -> sha256 of the requested file (its path may change). Accepted switches are listed in state["resume_switches"] (logged by the trainer).
RESUME_SWITCHES: dict = {}   # pinned mid-run switches the resume check accepts, e.g. {"kl_beta": ((0.001, 0.002),)}


def resume_switch(name: str, old, new) -> str | None:
    for a, b in RESUME_SWITCHES.get(name, ()):
        if (old, new) == (a, b):
            return "forward"
        if (old, new) == (b, a):
            return "reverse"
    return None


def _pinned_resume_switch(name: str, saved, requested, state: dict) -> dict | None:
    if name == "kl_beta" and all(type(v) in (int, float) for v in (saved, requested)):
        direction = resume_switch("kl_beta", float(saved), float(requested))
        return direction and {"argument": "kl_beta", "from": saved, "to": requested, "direction": direction}
    if name == "start_weights" and saved and requested and Path(requested).is_file():
        import hashlib
        old_sha = (state.get("segment_curriculum") or {}).get("start_weights_sha")
        new_sha = hashlib.sha256(Path(requested).read_bytes()).hexdigest()
        direction = resume_switch("start_weights_sha256", old_sha, new_sha)
        return direction and {"argument": "start_weights", "from": {"path": str(saved), "sha256": old_sha},
                               "to": {"path": str(requested), "sha256": new_sha}, "direction": direction}
    return None


_MISSING = object()
CHECKPOINT_NAME = re.compile(r"checkpoint-([0-9]+)\Z")


def _nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def complete_checkpoint_step(path: Path) -> int | None:
    """The step of a complete checkpoint directory (state, trainer metadata, adapter), else None."""
    path = Path(path)
    match = CHECKPOINT_NAME.fullmatch(path.name)
    if not match or not path.is_dir():
        return None
    try:
        metadata = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    step = int(match.group(1))
    if not isinstance(metadata, dict) or type(metadata.get("global_step")) is not int or metadata["global_step"] != step:
        return None
    if not _nonempty_file(path / "training_state.pt"):
        return None
    if metadata.get("full_finetune"):
        # full fine-tuning: the sharded fp32 weights + AdamW state (torch.distributed.checkpoint) and the bf16 export
        return step if _nonempty_file(path / "fsdp" / ".metadata") and _nonempty_file(path / "model.safetensors.index.json") else None
    required = (path / "adapter" / "adapter_config.json",)
    weights = (path / "adapter" / "adapter_model.safetensors", path / "adapter" / "adapter_model.bin")
    if not all(_nonempty_file(p) for p in required) or not any(_nonempty_file(p) for p in weights):
        return None
    try:
        adapter_config = json.loads((path / "adapter" / "adapter_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(adapter_config, dict):
        return None
    return step


def latest_complete_checkpoint(output_dir: Path) -> Path:
    """Ignore temporary, superseded and partially written checkpoints."""
    root = Path(output_dir).resolve()
    if not root.is_dir():
        raise ResumeError(f"run directory does not exist: {root}")
    complete = [(step, path) for path in root.iterdir() if (step := complete_checkpoint_step(path)) is not None]
    if not complete:
        raise ResumeError(f"no complete checkpoint in {root}")
    return max(complete, key=lambda item: item[0])[1]


def _normal(value):
    if isinstance(value, (list, tuple)):
        return [_normal(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def expected_segment_prompts(bank: dict, args) -> list[str]:
    """The curriculum's prompt population for these arguments: sourced_only drops prompt-only trajectories, suites
    restricts, and a probe set's held-out trajectories are excluded."""
    from segment_curriculum import load_probe_set, population

    manifest = getattr(args, "curriculum_manifest", None)
    if manifest:
        return sorted(tr["track"] for tr in json.loads(Path(manifest).read_text(encoding="utf-8"))["tracks"])
    held_out = load_probe_set(getattr(args, "probe_set", None))["heldout_trajectories"]
    return sorted(population(bank, bool(getattr(args, "curriculum_sourced_only", False)), getattr(args, "curriculum_suites", None),
                             held_out, bool(getattr(args, "fresh_unique_states", False))))


def validate_resume(state: dict, args, world: int) -> None:
    """Require one complete checkpoint, matching adapter, and unchanged experiment."""
    if not isinstance(state, dict) or state.get("sample_format") != "per_request_v1":
        raise ResumeError("resume state must explicitly use per_request_v1")
    full = bool(getattr(args, "full_finetune", False))
    state_path = Path(args.resume_state).resolve() if args.resume_state else None
    adapter_path = Path(args.resume_adapter).resolve() if args.resume_adapter else None
    if state_path is None or (adapter_path is None and not full):
        raise ResumeError("full resume requires both --resume-state and --resume-adapter")
    checkpoint = state_path.parent
    if state_path.name != "training_state.pt" or checkpoint.parent != Path(args.output_dir).resolve():
        raise ResumeError("resume state must be training_state.pt in this run's checkpoint directory")
    if full:
        if adapter_path is not None:
            raise ResumeError("a full fine-tuning checkpoint has no adapter")
        if bool(state.get("full_finetune")) is not True or not (checkpoint / "fsdp" / ".metadata").is_file():
            raise ResumeError("checkpoint lacks the full fine-tuning training state (fsdp/)")
    elif adapter_path != (checkpoint / "adapter").resolve():
        raise ResumeError("resume adapter and optimizer state must belong to the same checkpoint")
    latest = latest_complete_checkpoint(args.output_dir)
    if checkpoint != latest:
        raise ResumeError(f"resume requires latest complete checkpoint {latest.name}; older rollback is forbidden")
    step = state.get("global_step")
    expected_step = int(CHECKPOINT_NAME.fullmatch(checkpoint.name).group(1))
    if type(step) is not int or step != args.start_step or step != expected_step:
        raise ResumeError("checkpoint name, saved global_step and --start-step disagree")
    saved_world = state.get("world_size")
    if type(saved_world) is not int or saved_world < 1 or (saved_world != world and not full):
        raise ResumeError("resume world size differs from checkpoint")   # full FT: DCP reshards the fsdp/ state
    if not isinstance(state.get("model"), str) or Path(state["model"]).resolve() != Path(args.model).resolve():
        raise ResumeError("resume base model differs from checkpoint")
    saved_args = state.get("args")
    if not isinstance(saved_args, dict):
        raise ResumeError("checkpoint does not contain training arguments")
    segment_bank = getattr(args, "segment_bank", None)
    critical = CRITICAL_ARGUMENTS + (SEGMENT_ARGUMENTS if segment_bank else ())
    switches = []
    for name in critical:
        saved = saved_args.get(name, LATER_ARGUMENT_DEFAULTS.get(name, _MISSING))
        if saved is _MISSING or not hasattr(args, name):
            raise ResumeError(f"resume argument is missing: {name}")
        if _normal(saved) != _normal(getattr(args, name)):
            switch = _pinned_resume_switch(name, saved, getattr(args, name), state)
            if not switch:
                raise ResumeError(f"resume would change {name}: saved={saved!r}, requested={getattr(args, name)!r}")
            switches.append(switch)
    if switches:
        state["resume_switches"] = switches
    optimizer = state.get("optimizer")
    if full:
        if optimizer != {"sharded": "fsdp"}:
            raise ResumeError("full fine-tuning optimizer state must live in the checkpoint's sharded fsdp/ state")
    elif not isinstance(optimizer, dict) or not isinstance(optimizer.get("state"), dict) or not isinstance(optimizer.get("param_groups"), list):
        raise ResumeError("checkpoint optimizer state is incomplete")
    prompts, order = state.get("prompts"), state.get("prompt_order")
    if segment_bank:
        from segment_bank import load_bank

        bank = load_bank(segment_bank)
        expected_prompts = expected_segment_prompts(bank, args)
        curriculum = state.get("segment_curriculum")
        if not isinstance(curriculum, dict) or curriculum.get("bank_sha256") != bank["sha256"]:
            raise ResumeError("checkpoint segment curriculum is missing or belongs to a different bank")
    else:
        expected_prompts = [(suite, task, seed) for suite in args.suites for task in args.tasks for seed in args.seeds]
    if _normal(prompts) != _normal(expected_prompts):
        raise ResumeError("checkpoint prompt population differs from this experiment")
    if not isinstance(order, list) or any(type(i) is not int for i in order) or sorted(order) != list(range(len(prompts))):
        raise ResumeError("checkpoint task order is not a permutation of its prompts")
    if type(state.get("prompt_cursor")) is not int or state["prompt_cursor"] < 0 or "prompt_rng" not in state:
        raise ResumeError("checkpoint task cursor or RNG state is missing")
    rank_rng = state.get("rank_rng")
    if not isinstance(rank_rng, list) or len(rank_rng) != saved_world:
        raise ResumeError("checkpoint rank RNG states are incomplete")
    for rank, rng in enumerate(rank_rng):
        if not isinstance(rng, dict) or any(key not in rng for key in ("python", "torch", "cuda")):
            raise ResumeError(f"checkpoint RNG state is incomplete for rank {rank}")


def load_resume(args, world: int) -> dict:
    """Load trusted local training state on CPU, then validate its provenance."""
    import torch

    if not args.resume_state:
        raise ResumeError("--resume-state is required")
    try:
        state = torch.load(args.resume_state, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ResumeError(f"cannot load resume state: {args.resume_state}") from exc
    validate_resume(state, args, world)
    return state


def _log_plan(path: Path, start_step: int) -> dict | None:
    if not path.exists():
        return None
    source = path.read_bytes()
    lines = source.splitlines(keepends=True)
    kept = []
    kept_rows = removed_rows = 0
    partial = False
    previous_step = -1
    trimming = False
    for i, line in enumerate(lines):
        if not line.strip():
            if not trimming:
                kept.append(line)
            continue
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError) as exc:
            if i == len(lines) - 1 and not line.endswith(b"\n"):
                partial = True
                trimming = True
                break
            raise ResumeError(f"invalid complete JSON row in {path.name}, line {i + 1}") from exc
        step = row.get("step") if isinstance(row, dict) else None
        # step 0 rows exist only as the probe of the initial policy (written before the first update)
        if type(step) is not int or step < 0 or (step == 0 and not row.get("probe_only")):
            raise ResumeError(f"invalid step in {path.name}, line {i + 1}")
        if step < previous_step:
            raise ResumeError(f"nonmonotonic steps in {path.name}; refusing to rewrite mixed lineages")
        previous_step = step
        if step > start_step:
            trimming = True
            removed_rows += 1
        else:
            kept.append(line)
            kept_rows += 1
    retained = b"".join(kept)
    added_newline = bool(retained and not retained.endswith(b"\n"))
    if added_newline:
        retained += b"\n"
    if not removed_rows and not partial and not added_newline:
        return None
    return {"path": path, "source": source, "retained": retained, "mode": stat.S_IMODE(path.stat().st_mode),
            "kept_rows": kept_rows, "removed_rows": removed_rows, "dropped_partial_line": partial, "added_final_newline": added_newline}


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def prepare_resume_logs(output_dir: Path, start_step: int) -> dict:
    """Rank zero only: archive originals before atomically trimming uncheckpointed tails."""
    if type(start_step) is not int or start_step < 0:
        raise ResumeError("resume log step must be a non-negative integer")
    root = Path(output_dir).resolve()
    plans = [p for name in ("metrics.jsonl", "rollouts.jsonl") if (p := _log_plan(root / name, start_step)) is not None]
    suffix = f"resume-step{start_step}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"
    for plan in plans:
        archive = plan["path"].with_name(f"{plan['path'].name}.{suffix}")
        with archive.open("xb") as fh:
            fh.write(plan["source"])
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(archive, plan["mode"])
        plan["archive"] = archive
    if plans:
        _sync_directory(root)
    summaries = []
    for plan in plans:
        path = plan["path"]
        if path.read_bytes() != plan["source"]:
            raise ResumeError(f"{path.name} changed while preparing resume; another writer may be active")
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.resume-", dir=root)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(plan["retained"])
                fh.flush()
                os.fsync(fh.fileno())
            os.chmod(temp_name, plan["mode"])
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        summaries.append({"file": path.name, "archive": str(plan["archive"]),
                          **{k: plan[k] for k in ("kept_rows", "removed_rows", "dropped_partial_line", "added_final_newline")}})
    if plans:
        _sync_directory(root)
    return {"start_step": start_step, "trimmed": summaries}
