#!/usr/bin/env python3
"""Manifest-driven, node-local lifecycle for the isolated raw-request RL run."""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid
from contextlib import contextmanager
from pathlib import Path

from training_resume import complete_checkpoint_step, latest_complete_checkpoint
from toolchain_preflight import PYTHON_RUNTIME_CONTRACT, RUNTIME_CONTRACT, SCHEMA_VERSION as PREFLIGHT_SCHEMA, validate_receipt

CONTRACTS = (RUNTIME_CONTRACT, PYTHON_RUNTIME_CONTRACT)

# Workspace for runs, environments and weights; every path below can be overridden in the run config (which may use ${VAR}
# references, expanded by load_config).
WORKSPACE = os.environ.get("CAPTAIN_WORKSPACE", str(Path.home() / "captain"))
REPO_ROOT = str(Path(__file__).resolve().parents[1])      # the CaPtain checkout (this file is <repo>/rl/fresh_run.py)
DEFAULT_RUN = f"{WORKSPACE}/runs/captain-4b"
DEFAULTS = {
    "run_dir": DEFAULT_RUN,
    "repo": REPO_ROOT,
    "harness": f"{REPO_ROOT}/harness",
    "base_model": None, "live_dir": None, "workspace": WORKSPACE,
    "python_env": f"{WORKSPACE}/envs/libero/bin/python",
    "python_train": f"{WORKSPACE}/envs/train/bin/python",
    "nnodes": 2, "single_node": "a", "rollout_nodes": None, "train_gpus": [4, 5, 6, 7], "minimum_free_gpu_gib": 60,
    # explicit training nodes (e.g. ["a", "b"] with rollout node "c"); None = the first nnodes of a, b, c, d
    "train_nodes": None,
    # the policy's chat_template_kwargs.enable_thinking (env server --enable-thinking); CaPtain-4B-SFT is trained with
    # reasoning, so its RL run uses "on"
    "enable_thinking": "off",
    "nodes": {"a": {"hostname": "node-a", "host": "node-a", "egl": 1},
              "b": {"hostname": "node-b", "host": "node-b", "egl": 3}},
    "proxy_port": 8512, "env_port": 8402, "policy_port": 8500, "master_port": 29641,
    # more policy servers with the same weights on each rollout node (the proxy pins each episode to one; the trainer syncs all)
    "extra_policy_ports": [],
    "nccl_socket_ifname": "eth00", "sam3_url": "http://127.0.0.1:8114", "molmo_url": "http://127.0.0.1:8122/v1",
    "reward_version": "v12a", "runtime_tool_contract": None, "prompts_per_step": 8, "num_generations": 4,
    "lr": 1e-5, "kl_beta": 0.05, "adv_std_floor": 0.2, "max_grad_norm": 0.3,
    "warmup_steps": 10, "save_steps": 10, "max_steps": 1000, "seed": 17,
    "lora_r": 64, "lora_alpha": 128, "max_seq_tokens": 76000, "hard_max_tokens": 76000,
    "suites": ["libero_spatial", "libero_object", "libero_goal"], "tasks": list(range(10)), "seeds": [1, 2, 3],
    "max_infrastructure_rate": 0.05, "max_training_requests": 4, "env_concurrency": 32,
    "max_requests": 40, "wall_time": 720, "grace_s": 60, "max_actions": 150, "max_tool_calls": 600,
    "context_window": 65536, "context_limit": 60000, "keep_recent_tokens": 16000,
    "max_output_tokens": 4096, "tool_output_tokens": 4000, "context_hard_limit": 65536, "max_images": 12,
    # per-call RL (python-runtime/v1): segment starts from rl/segment_bank.py; None = whole-episode GRPO
    "segment_bank": None, "segment_horizon": 1, "curriculum_episode_fraction": 0.25, "curriculum_frontier_step": 1,
    "curriculum_advance_fraction": 0.5, "max_infidelity_rate": 0.25, "inline_task": False,
    "curriculum_mode": "reverse", "curriculum_sourced_only": False, "curriculum_active_fraction": 0.0,
    "curriculum_active_min_delta": 0.0, "curriculum_fresh_horizon": None, "curriculum_suites": None,
    # fixed probe starts (rl/make_probe_set.py) rolled out without gradient every probe_every steps
    "probe_set": None, "probe_every": 10, "probe_generations": 2, "curriculum_focus_set": None,
    # remaining-task protocol (segment_budget B: a teacher start at K gets B-K live requests, unsolved starts only) and the
    # handoff at K (clean = physical state only); student_python pins the model's own interpreter (checked in the sandbox)
    "segment_budget": None, "segment_min_live": 4, "segment_handoff": "continuation", "student_python": None, "sandbox_hide": [],
    # score centering (Marek & Ryabinin 2026): the proxy records the sampler's top-k logprobs, the trainer subtracts the expected score
    "score_centering": False, "sampler_topk": 0,
    # full fine-tuning (FSDP2 over train_gpus) instead of LoRA; such runs cannot resume (no optimizer state is saved)
    "full_finetune": False,
    # fresh-only protocol helpers: one prompt per official initial state
    "fresh_unique_states": False,
    # start only from FRESH initial states (harness fresh-init-states, seeds >= 1000); fresh_start_pool is the
    # fresh-start-pool/v1 manifest of rl/fresh_start_bank.py (its bank is segment_bank; the manifest is hashed with the sources)
    "fresh_seeds_only": False, "fresh_start_pool": None,
    # every episode's complete proxy record (gzip JSON) on the rollout node's local disk, before training selection
    "raw_record_dir": None,
    # full fine-tuning: where the sharded fp32 weights + AdamW state of each checkpoint go (trainer node local disk)
    "full_state_dir": None,
    # remaining-task curriculum GRPO: boundary manifest, binary reward,
    # per-trajectory normalization, zero-signal skip, exposure-based warmup
    "curriculum_manifest": None, "boundary_fresh_only": False, "reward_mode": None, "loss_normalization": "token_mean",
    "skip_zero_signal": False, "warmup_attempts": 0,
    # policy servers on another node (URLs): the proxy routes to them and the trainer syncs them (live_dir must be shared)
    "policy_workers": [],
    # learner/pipeline: activation offload (auto = on with full FT), one-batch rollout
    # lookahead (0/1) and its truncated importance-weight cap (None = the plain on-policy score); taken from this run's config
    "offload_activations": "auto", "async_lookahead": 0, "is_cap": None,
    # one forward per exact-prefix chain of selected requests (same objective, fewer context tokens)
    "pack_prefixes": False,
    # learner/pipeline infra (same objective): rank assignment of packed
    # sequences ("pack") instead of whole trajectories, background checkpoint writes, and a lookahead that keeps collecting on
    # checkpoint rounds (the checkpoint stores the pre-draw curriculum state)
    "pack_assignment": "trajectory", "async_checkpoint": False, "lookahead_through_checkpoints": False,
    # streaming: continuous rollout pool + proxy-gated weight replacement (per-request versions, partial rollouts);
    # requests older than max_staleness optimizer versions lose their policy-gradient credit
    "streaming": False, "max_staleness": 2,
    # credit assignment: zero task advantage on a successful trajectory's
    # already-solved suffix (env server marks the requests; KL, records and denominators unchanged)
    "solved_suffix_credit": False,
    # code-execution reward: the env server labels every request's Python execution
    # (rl/code_reward.py); the trainer adds lambda * cost on the erroring request and withholds positive task credit from
    # pure errors. code_reward_weight is lambda; code_reward_budget the fixed request budget H of the normalizer
    "code_reward_mode": None, "code_reward_weight": 0.0, "code_reward_budget": 40,
    # diagnostic null control: permute task advantages within each group
    "advantage_shuffle": False,
    # a complete checkpoint of another run (same base model and experiment) that this run continues; see seed_from_checkpoint
    "continue_from": None,
    # simulator critic: per-request advantages from a physics-privileged critic (critic_dir with
    # feature_spec.json + critic_mlp.pt), GAE lambda over requests, final 0/1 reward; SAO double-sided ratio masking
    # (both bounds or neither; replaces is_cap)
    "advantage_mode": "grpo", "critic_dir": None, "gae_lambda": 0.0, "ratio_mask_low": None, "ratio_mask_high": None,
    # SAO-style critic co-training (critic_cotrain.py): physics + the current policy's detached boundary
    # hidden state, K updates per policy step at critic_lr; needs every request trained and trajectory-level rank assignment
    "critic_cotrain": False, "critic_hidden": "current", "critic_lr": 1e-4, "critic_updates": 2,
    # return of a false finish in the last-step advantage (success 1, otherwise 0); 0 = no false-finish penalty
    "false_finish_reward": 0.0,
    # prioritized start sampling for the segment curriculum (rl/make_start_weights.py; start-weights/v1), None = uniform
    "start_weights": None,
    # thin-basket fix: start-exclude/v1 file of training starts the curriculum never draws
    # (segment_curriculum.load_start_exclude); None = off. Changeable by relocate only for a pinned file (START_EXCLUDE_ALLOWED)
    "start_exclude": None,
    # gradient accumulation: one optimizer step per this many collection rounds (rl/grad_accum.py)
    "grad_accum_rounds": 1,
    # per-request reward costs (train --act-cost ...; all off by default)
    "act_cost": 0.0, "fail_cost": 0.0, "act_ref": None, "missed_finish_return": None, "exhaust_reward": None,
    # rollout speed-ups (all off by default; infrastructure keys read from the config on every start and resume,
    # changeable by relocate): artifact_root = local directory (e.g. /dev/shm/captain-artifacts) for the harness host object
    # store instead of the NFS episode dir (env server); ref_every = KL reference forward on 1 in N passes, KL weight x N (trainer)
    "artifact_root": None, "ref_every": 1,
    # early_stop = {"tau": 0.02, "k": 5, "min_request": 5, "control": 0.05}: co-critic early stop of hopeless episodes (env server
    # monitor + proxy refusal -> request_cap_exhausted; the trainer exports the co-critic to <run_dir>/cocritic_live.pt); None = off
    "early_stop": None,
    # attention kernel of the trainer's full-attention layers (rl/attn_backends.py: None = SDPA as before, "fa3", "sdpa_cudnn")
    # and the overlay directory of the prebuilt FA3 build
    "attn_backend": None, "attn_overlay": None,
    # torch.compile of the decoder layers' MLP + RMSNorms in the trainer (policy and reference); False = eager as before
    "compile_regional": False,
    # selective activation recompute: GiB per pass for keeping full-attention layers' activations (0 = every layer checkpointed)
    "keep_attn_act_gib": 0.0,
    # group sampling (train --stream-group-size / --group-advantage; 1 = off). Read from
    # the config on every start and resume, so a run can switch it at a checkpoint
    "stream_group_size": 1, "group_advantage": "group",
    # rl/prefix_tree.py: shared-prefix training, one prefix forward/backward per trajectory; needs pack_prefixes false and
    # pack_assignment trajectory (computation only: same objective and gradient); False = per-request passes
    "prefix_tree": False,
    # harness fast-loop gates (CAPTAIN_FAST_LOOP / CAPTAIN_LAZY_RENDER in the env role's environment); False = the old harness path
    "harness_fast_loop": False, "harness_lazy_render": False,
    # held-out monitor: {"every": 10, "ev12_seeds": [...], "official_seeds": [...], "ev12_harness": <harness copy with the
    # held-out wide-layout ("ev12") start data>}: every N-th streaming admission runs a held-out start, logged to <run>/monitor.jsonl, never trained; None = off
    "heldout_monitor": None,
    # missed-finish fix = reward_patch {"missed_finish_return": 0.25,
    # "post_success_act": true}: overrides the run's missed-finish return and
    # charges c_act on every physical action from the first request whose mark shows all goals satisfied (perception stays free).
    # Read from the config on every start and resume (the CRITICAL --missed-finish-return of the saved args stays as it was); None = off
    "reward_patch": None,
    # staged subgoal reward for hard tasks (both None = off, commands and objective as before):
    # staged_reward = {"credit": c < missed-finish return, "ladders": {suite:task_id: [rung, ...]}, "grasp": {...}}: a laddered task's
    # episode that is neither a host success nor a false finish and reaches d >= 1 of its n subgoal rungs (host request marks only, each
    # rung once, in order; physics_critic.stage_ladder) gets the return c x d / (n + 1) instead of 0 / exhaust (trainer --staged-reward,
    # from the config on every start and resume). es_exempt = {"tasks": [...]}: episodes of these tasks are never stopped by the co-critic
    # early-stop rule (env server exempt_tasks; the success rule success_k still applies). Not layout keys: relocate changes them only
    # together, only between off (both None) and the pinned STAGE_SWITCH_ALLOWANCES spec (sha256 of the canonical pair), either way.
    "staged_reward": None, "es_exempt": None,
    # over-length guard: overlength_guard = {"max_drops": N} turns on B1 (env --max-request-tokens
    # min(max_seq_tokens, hard_max_tokens): longer requests are not eligible for the training selection) and A (trainer
    # --max-overlength-drops N: up to N trajectories per round whose only build errors are OverLengthRollout are dropped on every
    # rank instead of refusing the batch). None = off (commands unchanged). Not a layout key: relocate changes it only between
    # None and the pinned OVERLENGTH_GUARD_ALLOWANCES value, either way.
    "overlength_guard": None,
}
TRAIN_KEYS = ("lr", "kl_beta", "adv_std_floor", "max_grad_norm", "warmup_steps", "save_steps", "max_steps", "seed",
              "lora_r", "lora_alpha", "max_seq_tokens", "hard_max_tokens", "prompts_per_step", "num_generations", "max_infrastructure_rate",
              "grad_accum_rounds", "act_cost", "fail_cost")
ENV_KEYS = ("max_requests", "wall_time", "grace_s", "max_actions", "max_tool_calls", "context_window", "context_limit",
            "keep_recent_tokens", "max_output_tokens", "tool_output_tokens", "context_hard_limit", "max_images", "max_training_requests")
SOURCE_FILES = ("train.py", "grad_accum.py", "raw_request_training.py", "env_server.py", "policy_proxy.py", "chat_tokens.py",
                "reward.py", "training_resume.py", "rollout_storage.py", "fresh_run.py", "toolchain_preflight.py",
                "reward_segment.py", "segment_bank.py",
                "segment_curriculum.py", "prefix_chains.py", "code_reward.py", "infra_pipeline.py", "physics_critic.py",
                "critic_features.py", "critic_cotrain.py", "nccl_refit.py", "stream_pool.py", "vllm_worker_ext.py",
                "fresh_start_bank.py")
SEGMENT_TRAIN_KEYS = ("segment_horizon", "curriculum_episode_fraction", "curriculum_frontier_step", "curriculum_advance_fraction",
                      "max_infidelity_rate", "curriculum_mode", "curriculum_active_fraction", "curriculum_active_min_delta",
                      "probe_every", "probe_generations", "segment_min_live", "segment_handoff")


class RunError(RuntimeError):
    pass


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(value, fh, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextmanager
def operation_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunError("another operation is already running") from exc
        yield


def expand_env(value):
    """${VAR} / $VAR references in every string of a config (os.path.expandvars); unresolved references are an error."""
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    if isinstance(value, str) and "$" in value:
        out = os.path.expandvars(value)
        if "${" in out:
            raise RunError(f"unresolved environment variable in config value {value!r}")
        return out
    return value


def load_config(path):
    """A run config file with ${VAR} references expanded from the environment (e.g. CAPTAIN_WORKSPACE, CAPTAIN_MODELS)."""
    return expand_env(json.loads(Path(path).read_text()))


def normalize_config(overrides):
    if not isinstance(overrides, dict) or set(overrides) - set(DEFAULTS):
        raise RunError("config must be an object containing only recognized keys")
    cfg = copy.deepcopy(DEFAULTS)
    cfg.update(overrides)
    # up to 8 nodes a-h; train_nodes (optional) names the training nodes explicitly
    if cfg["nnodes"] not in range(1, len(NODE_IDS) + 1) or cfg["single_node"] not in NODE_IDS:
        raise RunError(f"nnodes must be 1-{len(NODE_IDS)}; single_node must be one of {', '.join(NODE_IDS)}")
    tn = cfg["train_nodes"]
    if tn is not None and (not isinstance(tn, list) or len(tn) != cfg["nnodes"] or len(set(tn)) != len(tn)
                           or any(n not in NODE_IDS or n not in cfg["nodes"] for n in tn)):
        raise RunError("train_nodes must list nnodes distinct configured nodes")
    if not isinstance(cfg["train_gpus"], list) or not cfg["train_gpus"] or len(set(cfg["train_gpus"])) != len(cfg["train_gpus"]):
        raise RunError("train_gpus must be a nonempty list of unique GPU indices")
    if any(type(g) is not int or g < 0 for g in cfg["train_gpus"]):
        raise RunError("GPU indices must be non-negative integers")
    if cfg["rollout_nodes"] is None:
        cfg["rollout_nodes"] = active_nodes(cfg)
    rollout = cfg["rollout_nodes"]
    if not isinstance(rollout, list) or not rollout or any(n not in NODE_IDS for n in rollout) or len(set(rollout)) != len(rollout):
        raise RunError("rollout_nodes must be a nonempty list of unique configured node ids")
    if cfg.get("streaming") and (cfg["num_generations"] != 1 or type(cfg.get("max_staleness")) is not int or cfg["max_staleness"] < 0):
        raise RunError("streaming needs num_generations 1 and a non-negative integer max_staleness")
    g = cfg.get("stream_group_size", 1)
    if type(g) is not int or g < 1 or (g > 1 and (not cfg.get("streaming") or cfg["prompts_per_step"] % g)):
        raise RunError("stream_group_size must be an integer >= 1; > 1 needs streaming and prompts_per_step divisible by it")
    if cfg.get("group_advantage", "group") not in ("group", "critic"):
        raise RunError("group_advantage must be 'group' or 'critic'")
    minimum = cfg["minimum_free_gpu_gib"]
    if type(minimum) not in (int, float) or not math.isfinite(minimum) or minimum <= 0:
        raise RunError("minimum_free_gpu_gib must be a positive finite number")
    if not cfg["base_model"] or cfg["reward_version"] not in {"v12a", "v12b"}:
        raise RunError("base_model and an explicit supported reward version are required")
    if cfg["runtime_tool_contract"] not in {None, *CONTRACTS}:
        raise RunError("unsupported runtime tool contract")
    if cfg["reward_version"] == "v12b" and cfg["runtime_tool_contract"] not in CONTRACTS:
        raise RunError("reward v12b requires a versioned runtime contract and toolchain preflight")
    if cfg.get("segment_bank"):
        if cfg["runtime_tool_contract"] != PYTHON_RUNTIME_CONTRACT:
            raise RunError("segment RL requires python-runtime/v1")
        cfg["segment_bank"] = str(Path(cfg["segment_bank"]).resolve())
        if type(cfg["segment_horizon"]) is not int or cfg["segment_horizon"] < 1:
            raise RunError("segment_horizon must be a positive integer")
        if cfg["curriculum_mode"] not in ("reverse", "uniform") or type(cfg["curriculum_sourced_only"]) is not bool:
            raise RunError("curriculum_mode must be reverse or uniform; curriculum_sourced_only a boolean")
        suites = cfg["curriculum_suites"]
        if suites is not None and (not isinstance(suites, list) or not suites or any(type(x) is not str for x in suites)):
            raise RunError("curriculum_suites must be a nonempty list of suite names")
        fresh = cfg["curriculum_fresh_horizon"]
        if fresh is not None and (type(fresh) is not int or fresh < 1):
            raise RunError("curriculum_fresh_horizon must be a positive integer")
        if cfg["probe_set"]:
            cfg["probe_set"] = str(Path(cfg["probe_set"]).resolve())
        if cfg["curriculum_focus_set"] is not None and not (cfg["probe_set"] and type(cfg["curriculum_focus_set"]) is str):
            raise RunError("curriculum_focus_set names a set of the configured probe_set")
        if cfg.get("fresh_seeds_only"):
            if cfg["curriculum_episode_fraction"] != 1.0 or cfg["curriculum_sourced_only"] or cfg["probe_set"]:
                raise RunError("fresh_seeds_only: curriculum_episode_fraction 1.0, no curriculum_sourced_only, no probe_set")
            if not cfg.get("fresh_start_pool"):
                raise RunError("fresh_seeds_only needs fresh_start_pool (rl/fresh_start_bank.py manifest)")
            cfg["fresh_start_pool"] = str(Path(cfg["fresh_start_pool"]).resolve())
            pool = json.loads(Path(cfg["fresh_start_pool"]).read_text())
            bank_sha = json.loads(Path(cfg["segment_bank"]).read_text()).get("sha256")
            if pool.get("schema") != "fresh-start-pool/v1" or pool.get("bank_sha256") != bank_sha:
                raise RunError("fresh_start_pool is not the fresh-start-pool/v1 manifest of segment_bank")
        budget = cfg["segment_budget"]
        if budget is not None and (type(budget) is not int or budget <= cfg["segment_min_live"] or budget > cfg["max_requests"]):
            raise RunError("segment_budget must be an integer above segment_min_live and at most max_requests")
        if cfg["segment_handoff"] not in ("continuation", "clean"):
            raise RunError("segment_handoff must be continuation or clean")
        if cfg["segment_handoff"] == "clean" and not cfg["student_python"]:
            raise RunError("a clean handoff restarts the session with student_python; configure it")
    if cfg["student_python"]:
        if not os.path.isabs(cfg["student_python"]) or not os.access(cfg["student_python"], os.X_OK):
            raise RunError("student_python must be an absolute path to an executable interpreter")
    if type(cfg["score_centering"]) is not bool or type(cfg["sampler_topk"]) is not int or not 0 <= cfg["sampler_topk"] <= 20:
        raise RunError("score_centering must be a boolean and sampler_topk an integer in 0..20 (vLLM max_logprobs)")
    if cfg["score_centering"] and not cfg["sampler_topk"]:
        raise RunError("score_centering needs sampler_topk > 0")
    if not isinstance(cfg["sandbox_hide"], list) or any(not isinstance(p, str) or not os.path.isabs(p) for p in cfg["sandbox_hide"]):
        raise RunError("sandbox_hide must be a list of absolute directories")
    for key in ("run_dir", "repo", "harness", "base_model", "workspace"):
        cfg[key] = str(Path(cfg[key]).resolve())
    if cfg["continue_from"]:
        cfg["continue_from"] = str(Path(cfg["continue_from"]).resolve())
    # The invoked venv path determines pyvenv.cfg discovery, even for symlinked Python binaries.
    for key in ("python_env", "python_train"):
        cfg[key] = os.path.abspath(os.path.expanduser(cfg[key]))
    cfg["live_dir"] = str(Path(cfg["live_dir"] or (Path(cfg["run_dir"]) / "policy-live")).resolve())
    for key in ("proxy_port", "env_port", "policy_port", "master_port"):
        if type(cfg[key]) is not int or not 1 <= cfg[key] <= 65535:
            raise RunError(f"invalid {key}")
    extra = cfg["extra_policy_ports"]
    if not isinstance(extra, list) or any(type(x) is not int or not 1 <= x <= 65535 for x in extra):
        raise RunError("extra_policy_ports must be a list of ports")
    if len({cfg[key] for key in ("proxy_port", "env_port", "policy_port", "master_port")} | set(extra)) != 4 + len(extra):
        raise RunError("service, policy and rendezvous ports must be distinct")
    if cfg["code_reward_mode"] not in (None, "python_exec_cost_v1", "python_exec_contract_v2") or type(cfg["code_reward_weight"]) not in (int, float) \
            or cfg["code_reward_weight"] < 0 or (cfg["code_reward_mode"] is None) != (cfg["code_reward_weight"] == 0) \
            or type(cfg["code_reward_budget"]) is not int or cfg["code_reward_budget"] < 1:
        raise RunError("code_reward_mode (python_exec_cost_v1 or null) needs a positive code_reward_weight exactly when set, "
                       "and an integer code_reward_budget >= 1")
    if type(cfg["pack_prefixes"]) is not bool or type(cfg["solved_suffix_credit"]) is not bool:
        raise RunError("pack_prefixes and solved_suffix_credit must be booleans")
    if cfg["offload_activations"] not in ("auto", "on", "off") or cfg["async_lookahead"] not in (0, 1):
        raise RunError("offload_activations must be auto/on/off and async_lookahead 0 or 1")
    if cfg["pack_assignment"] not in ("trajectory", "pack") or (cfg["pack_assignment"] == "pack" and not cfg["pack_prefixes"]):
        raise RunError("pack_assignment must be trajectory or pack (pack needs pack_prefixes)")
    if type(cfg["async_checkpoint"]) is not bool or (cfg["async_checkpoint"] and not cfg["full_finetune"]):
        raise RunError("async_checkpoint must be a boolean (true needs full_finetune)")
    if type(cfg["lookahead_through_checkpoints"]) is not bool or (cfg["lookahead_through_checkpoints"] and not cfg["async_lookahead"]):
        raise RunError("lookahead_through_checkpoints must be a boolean (true needs async_lookahead)")
    if cfg["is_cap"] is not None and (type(cfg["is_cap"]) not in (int, float) or not cfg["is_cap"] > 1 or not cfg["score_centering"]):
        raise RunError("is_cap must be a number > 1 and needs score_centering (the sampler log-probabilities)")
    if cfg["enable_thinking"] not in ("off", "on", "server"):
        raise RunError("enable_thinking must be off, on or server")
    validate_advantage_config(cfg)
    if cfg["start_weights"] is not None:
        if not cfg.get("segment_bank"):
            raise RunError("start_weights apply to the segment curriculum (segment_bank)")
        cfg["start_weights"] = str(Path(cfg["start_weights"]).resolve())
        from segment_curriculum import load_start_weights
        try:
            load_start_weights(cfg["start_weights"])
        except (OSError, ValueError) as exc:
            raise RunError(f"start_weights: {exc}") from exc
    if cfg["start_exclude"] is not None:
        if not cfg.get("segment_bank") or cfg.get("curriculum_manifest"):
            raise RunError("start_exclude applies to the segment curriculum (segment_bank, no curriculum_manifest)")
        cfg["start_exclude"] = str(Path(cfg["start_exclude"]).resolve())
        from segment_curriculum import load_start_exclude
        try:
            spec = load_start_exclude(cfg["start_exclude"])
        except (OSError, ValueError) as exc:
            raise RunError(f"start_exclude: {exc}") from exc
        trajectories = json.loads(Path(cfg["segment_bank"]).read_text()).get("trajectories", {}).values()
        population = {f'{t.get("suite")}:{int(t.get("task_id"))}:{int(t.get("seed"))}' for t in trajectories}
        unknown = sorted(spec["exclude"] - population)
        if unknown:
            raise RunError(f"start_exclude: {len(unknown)} starts are not in the segment bank, e.g. {unknown[:3]}")
    if cfg["artifact_root"] is not None and not (isinstance(cfg["artifact_root"], str) and os.path.isabs(cfg["artifact_root"])):
        raise RunError("artifact_root must be null or an absolute path")
    if cfg["attn_backend"] not in (None, "sdpa", "fa3", "sdpa_cudnn") or (cfg["attn_backend"] == "fa3") != bool(cfg["attn_overlay"]):
        raise RunError("attn_backend must be null/sdpa/fa3/sdpa_cudnn; fa3 needs attn_overlay (and only fa3 uses it)")
    if type(cfg["keep_attn_act_gib"]) not in (int, float) or not 0 <= cfg["keep_attn_act_gib"] <= 60:
        raise RunError("keep_attn_act_gib must be a number in [0, 60]")
    if type(cfg["compile_regional"]) is not bool:
        raise RunError("compile_regional must be a boolean")
    hm = cfg["heldout_monitor"]
    if hm is not None and (not isinstance(hm, dict) or set(hm) - {"every", "ev12_seeds", "official_seeds", "ev12_harness"}
                           or type(hm.get("every")) is not int or hm["every"] < 2 or not cfg.get("streaming")
                           or (hm.get("ev12_seeds") and not hm.get("ev12_harness"))
                           or any(type(x) is not int for x in (hm.get("ev12_seeds") or []) + (hm.get("official_seeds") or []))
                           or not (hm.get("ev12_seeds") or hm.get("official_seeds"))):
        raise RunError("heldout_monitor must be null or {every >= 2, ev12_seeds, official_seeds, ev12_harness} on a streaming run")
    if type(cfg["harness_fast_loop"]) is not bool or type(cfg["harness_lazy_render"]) is not bool:
        raise RunError("harness_fast_loop and harness_lazy_render must be booleans")
    if type(cfg["prefix_tree"]) is not bool:
        raise RunError("prefix_tree must be a boolean")
    if cfg["prefix_tree"] and (cfg["pack_prefixes"] or cfg["pack_assignment"] != "trajectory" or not cfg["full_finetune"]):
        raise RunError("prefix_tree needs pack_prefixes false, pack_assignment trajectory and full_finetune")
    es = cfg["early_stop"]
    if es is not None:
        if not isinstance(es, dict) or set(es) - {"tau", "k", "min_request", "control", "success_k"} or not {"tau", "k"} <= set(es) \
                or ("success_k" in es and (type(es["success_k"]) is not int or es["success_k"] < 1)) \
                or type(es["tau"]) not in (int, float) or not 0 < es["tau"] < 1 or type(es["k"]) is not int or es["k"] < 1 \
                or type(es.get("min_request", 5)) is not int or es.get("min_request", 5) < 1 \
                or type(es.get("control", 0.0)) not in (int, float) or not 0 <= es.get("control", 0.0) < 1:
            raise RunError("early_stop must be null or {tau in (0,1), k >= 1, min_request >= 1, control in [0,1), success_k >= 1 (optional, d)}")
        if not cfg.get("critic_cotrain") or cfg.get("critic_hidden") not in ("none", None):
            raise RunError("early_stop needs the physics-only co-trained critic (critic_cotrain, critic_hidden none)")
    rp = cfg["reward_patch"]
    if rp is not None:
        if not isinstance(rp, dict) or not rp or set(rp) - {"missed_finish_return", "post_success_act", "post_success_req", "post_success_cap",
                                                          "critic_target"} \
                or ("post_success_cap" in rp and (not rp.get("post_success_req") or type(rp["post_success_cap"]) not in (int, float)
                                                  or not 0 < rp["post_success_cap"] <= 1)) \
                or ("missed_finish_return" in rp and (type(rp["missed_finish_return"]) not in (int, float) or not -1 <= rp["missed_finish_return"] <= 1)) \
                or ("post_success_act" in rp and type(rp["post_success_act"]) is not bool) \
                or ("post_success_req" in rp and (type(rp["post_success_req"]) not in (int, float) or not 0 <= rp["post_success_req"] <= 0.05)) \
                or rp.get("critic_target", "success") not in ("success", "success_finish"):
            raise RunError("reward_patch must be null or {missed_finish_return in [-1, 1], post_success_act bool, post_success_req in [0, 0.05], "
                           "post_success_cap in (0, 1] (with post_success_req), critic_target success|success_finish}")
        if rp.get("critic_target") == "success_finish" and (not cfg.get("critic_cotrain") or cfg.get("critic_hidden") not in ("none", None)):
            raise RunError("reward_patch critic_target needs the physics-only co-trained critic (critic_cotrain, critic_hidden none)")
        if not (cfg.get("act_cost") or cfg.get("fail_cost") or cfg.get("missed_finish_return") is not None or cfg.get("exhaust_reward") is not None):
            raise RunError("reward_patch needs the reward costs on (act_cost / missed_finish_return ...)")
    validate_staged(cfg)
    validate_overlength_guard(cfg)
    for key in ("prompts_per_step", "num_generations", "max_steps", "save_steps", "max_training_requests", "env_concurrency",
                "grad_accum_rounds", "ref_every"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise RunError(f"{key} must be a positive integer")
    return cfg


def drawn_task_keys(cfg) -> set:
    """the suite:task_id keys a run can draw (curriculum_suites or suites x tasks)"""
    return {f"{suite}:{int(t)}" for suite in (cfg.get("curriculum_suites") or cfg.get("suites") or []) for t in cfg.get("tasks") or []}


def validate_staged(cfg) -> None:
    """staged-reward keys (staged_reward, es_exempt; both None = off)"""
    known = drawn_task_keys(cfg)
    st = cfg.get("staged_reward")
    if st is not None:
        from physics_critic import staged_spec_error
        rp = cfg.get("reward_patch") or {}
        mfr = rp["missed_finish_return"] if "missed_finish_return" in rp else cfg.get("missed_finish_return")
        err = staged_spec_error(st, mfr)
        if err:
            raise RunError(err)
        if cfg.get("advantage_mode") != "physcritic" or not cfg.get("critic_cotrain") or cfg.get("critic_hidden") not in ("none", None):
            raise RunError("staged_reward needs the physics-only co-trained critic (advantage_mode physcritic, critic_cotrain, critic_hidden none)")
        if not (cfg.get("act_cost") or cfg.get("fail_cost") or cfg.get("missed_finish_return") is not None or cfg.get("exhaust_reward") is not None):
            raise RunError("staged_reward needs the reward costs on (act_cost / missed_finish_return ...)")
        unknown = sorted(set(st["ladders"]) - known)
        if unknown:
            raise RunError(f"staged_reward tasks not drawn by this run: {unknown}")
    ex = cfg.get("es_exempt")
    if ex is not None:
        if not isinstance(ex, dict) or set(ex) != {"tasks"} or not isinstance(ex["tasks"], list) or not ex["tasks"] \
                or len(set(ex["tasks"])) != len(ex["tasks"]) or not all(isinstance(k, str) for k in ex["tasks"]):
            raise RunError("es_exempt must be null or {tasks: non-empty list of distinct 'suite:task_id' keys}")
        if not cfg.get("early_stop"):
            raise RunError("es_exempt needs early_stop")
        unknown = sorted(set(ex["tasks"]) - known)
        if unknown:
            raise RunError(f"es_exempt tasks not drawn by this run: {unknown}")


def validate_advantage_config(cfg):
    """advantage_mode / critic / ratio-mask keys (the trainer re-checks the same rules in advantage_arg_error)."""
    if cfg["advantage_mode"] not in ("grpo", "physcritic"):
        raise RunError("advantage_mode must be grpo or physcritic")
    low, high = cfg["ratio_mask_low"], cfg["ratio_mask_high"]
    if (low is None) != (high is None):
        raise RunError("ratio_mask_low and ratio_mask_high go together")
    if low is not None:
        if type(low) not in (int, float) or type(high) not in (int, float) or not (0 < low < 1 and high > 0):
            raise RunError("ratio mask needs 0 < ratio_mask_low < 1 and ratio_mask_high > 0")
        if not cfg["score_centering"] or cfg["is_cap"] is not None:
            raise RunError("ratio masking needs score_centering (sampler log-probabilities) and replaces is_cap")
    lam = cfg["gae_lambda"]
    if type(cfg["critic_cotrain"]) is not bool or cfg["critic_hidden"] not in ("current", "none") \
            or type(cfg["critic_lr"]) not in (int, float) or not cfg["critic_lr"] > 0 \
            or type(cfg["critic_updates"]) is not int or cfg["critic_updates"] < 1:
        raise RunError("critic_cotrain boolean, critic_hidden current|none, critic_lr > 0, critic_updates >= 1")
    if cfg["advantage_mode"] == "grpo":
        if cfg["critic_dir"] is not None or lam != 0.0 or cfg["critic_cotrain"]:
            raise RunError("critic_dir / gae_lambda / critic_cotrain need advantage_mode physcritic")
        return
    if cfg["critic_cotrain"] and cfg["critic_hidden"] != "none":
        # the hidden-reading critic scores requests inside the training forward: every request trained, trajectory ranks
        if not cfg["full_finetune"] or cfg["pack_assignment"] != "trajectory":
            raise RunError("critic_cotrain needs full_finetune (separate KL reference) and pack_assignment trajectory")
        if cfg["max_training_requests"] < cfg["max_requests"]:
            raise RunError("critic_cotrain needs every request trained: max_training_requests >= max_requests")
    # critic_hidden none: advantages are precomputed on all request states from physics, so request
    # subsampling (inclusion-weighted) and pack-level rank assignment are allowed
    if not cfg["critic_dir"]:
        raise RunError("advantage_mode physcritic needs critic_dir")
    cfg["critic_dir"] = str(Path(cfg["critic_dir"]).resolve())
    from physics_critic import validate_critic_dir
    try:
        validate_critic_dir(cfg["critic_dir"])
    except ValueError as exc:
        raise RunError(str(exc)) from exc
    if type(lam) not in (int, float) or not 0.0 <= lam <= 1.0:
        raise RunError("gae_lambda must be a number in [0, 1]")
    if cfg["advantage_shuffle"]:
        raise RunError("advantage_shuffle is a GRPO diagnostic")
    ffr = cfg.get("false_finish_reward", 0.0)
    if type(ffr) not in (int, float) or not -1.0 <= ffr <= 0.0:
        raise RunError("false_finish_reward must be a number in [-1, 0] (e.g. -0.3)")
    if cfg["reward_mode"] != "final_success_binary_v1":
        raise RunError("physcritic trains on the final 0/1 reward: reward_mode final_success_binary_v1")


NODE_IDS = ("a", "b", "c", "d", "e", "f", "g", "h")   # up to 8 nodes


def active_nodes(cfg):
    if cfg.get("train_nodes"):
        return list(cfg["train_nodes"])
    return list(NODE_IDS[:cfg["nnodes"]]) if cfg["nnodes"] > 1 else [cfg["single_node"]]


def validate_node(cfg, node):
    expected = cfg["nodes"][node]["hostname"]
    actual = socket.gethostname().split(".")[0]
    if actual != expected:
        raise RunError(f"node {node} requires hostname {expected}; this host is {actual}")


def source_hashes(cfg):
    paths = [Path(cfg["repo"]) / "rl" / name for name in SOURCE_FILES]
    harness = Path(cfg["harness"])
    paths += [harness / "src/captain_harness/server/env_server.py",
              harness / "src/captain_harness/adapters/libero/hidden_object_audit.py",
              harness / "src/captain_harness/adapters/libero/backend.py", harness / "scripts/react_episode.py"]
    extensions = {".py", ".js", ".json", ".jinja", ".jinja2"}
    paths += [p for p in (harness / "src").rglob("*")
              if p.is_file() and p.suffix in extensions and "__pycache__" not in p.parts]
    if cfg.get("runtime_tool_contract") in CONTRACTS:
        paths += [harness / "configs/calibration/libero_franka_v3.json", harness / "third_party/manifest.json"]
    if cfg.get("segment_bank"):
        paths.append(Path(cfg["segment_bank"]))
    if cfg.get("continue_from"):
        paths += seed_files(Path(cfg["continue_from"]))
    if cfg.get("probe_set"):
        paths.append(Path(cfg["probe_set"]))
    if cfg.get("fresh_start_pool"):
        paths.append(Path(cfg["fresh_start_pool"]))
    if cfg.get("start_weights"):
        paths.append(Path(cfg["start_weights"]))
    if cfg.get("start_exclude"):
        paths.append(Path(cfg["start_exclude"]))
    if cfg.get("critic_dir"):
        # the default critic directory holds critic_default.pt (ensemble), not critic_mlp.pt
        paths += [Path(cfg["critic_dir"]) / "feature_spec.json"] + [Path(cfg["critic_dir"]) / name for name in
                                                                     ("critic_default.pt", "critic_mlp.pt", "cocritic_warm.pt")
                                                                     if (Path(cfg["critic_dir"]) / name).is_file()]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}


SEED_FILES = ("training_state.pt", "trainer_state.json")


def seed_files(checkpoint):
    """What a continued run needs from its source checkpoint (optimizer/curriculum state, metadata, adapter; full fine-tuning:
    the bf16 export index; the node-local fsdp/ state is copied separately and not hashed, every node must read these)."""
    adapter = [p for p in sorted((checkpoint / "adapter").glob("*")) if p.is_file()]
    full = [checkpoint / "model.safetensors.index.json"] if (checkpoint / "model.safetensors.index.json").exists() else []
    return [checkpoint / name for name in SEED_FILES] + adapter + full


def validate_continue_from(cfg):
    source = Path(cfg["continue_from"])
    step = complete_checkpoint_step(source)
    if step is None:
        raise RunError(f"continue_from is not a complete checkpoint: {source}")
    saved = json.loads((source / "trainer_state.json").read_text()).get("args")
    if not isinstance(saved, dict) or Path(saved.get("model", "")).resolve() != Path(cfg["base_model"]):
        raise RunError("continue_from was trained from a different base model")
    if Path(saved.get("output_dir", "")).resolve() == Path(cfg["run_dir"]):
        raise RunError("continue_from must belong to another run; use train --resume within a run")
    return step


def seed_from_checkpoint(cfg):
    """Copy the source checkpoint's resume state in as this run's first checkpoint and record the lineage.
    The merged model shards are not copied: the trainer re-merges into the live dir at start."""
    source = Path(cfg["continue_from"])
    step = validate_continue_from(cfg)
    root = Path(cfg["run_dir"])
    tmp = Path(tempfile.mkdtemp(prefix=f".seed-{source.name}-", dir=root))
    if (source / "fsdp").exists():
        # full fine-tuning: an independent copy of the sharded fp32/AdamW state (this run prunes its own states; the source
        # run keeps its resume point), on this run's full_state_dir when configured; the bf16 shards are only for serving
        state = Path(cfg["full_state_dir"]) / source.name if cfg.get("full_state_dir") else tmp / "fsdp"
        if state.exists():
            raise RunError(f"seed target already exists: {state}")
        shutil.copytree((source / "fsdp").resolve(), state)
        if state != tmp / "fsdp":
            (tmp / "fsdp").symlink_to(state)
        for path in sorted(source.glob("*")):
            if path.is_file() and not path.is_symlink() and path.name not in SEED_FILES:
                try:
                    os.link(path, tmp / path.name)          # bf16 shards: same filesystem, no extra space
                except OSError:
                    shutil.copy2(path, tmp / path.name)
    for path in seed_files(source):
        target = tmp / path.relative_to(source)
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    os.replace(tmp, root / source.name)
    atomic_json(root / "lineage.json", {"continued_from": str(source), "step": step, "seeded_at": time.time(),
                                        "sha256": {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                                                   for p in seed_files(source)}})
    return root / source.name


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": int(pid), "start_ticks": int(fields[19]), "ppid": int(fields[1]), "pgid": int(fields[2]),
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(), "hostname": socket.gethostname()}
    except (OSError, ValueError, IndexError):
        return None


def same_process(saved):
    current = process_identity(saved["pid"])
    return current is not None and all(current[k] == saved.get(k) for k in ("pid", "start_ticks", "boot_id", "hostname"))


def owned_processes(record):
    """Include separately sessioned episode descendants, never unrelated policy/tool servers."""
    found = {}
    owner = f"FRESH_RUN_OWNER={record['owner']}".encode()
    table = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        identity = process_identity(int(path.name))
        if identity is None:
            continue
        table[identity["pid"]] = identity
        try:
            if owner in (path / "environ").read_bytes().split(b"\0"):
                found[identity["pid"]] = identity
        except OSError:
            pass
    if same_process(record):
        found[record["pid"]] = table[record["pid"]]
    while True:
        added = {pid: item for pid, item in table.items() if item["ppid"] in found and pid not in found}
        if not added:
            break
        found.update(added)
    return list(found.values())


def stop_record(path, timeout=8):
    path = Path(path)
    if not path.exists():
        return {"stopped": False, "reason": "no process record"}
    record = json.loads(path.read_text())
    if record["hostname"] != socket.gethostname():
        raise RunError("refusing to signal a process record belonging to another host")
    current = process_identity(record["pid"])
    if current is not None and not same_process(record):
        raise RunError("PID has been reused; refusing to signal an unrelated process")
    targets = {p["pid"]: p for p in owned_processes(record)}
    record["stop_targets"] = list(targets.values())
    atomic_json(path, record)
    for identity in targets.values():
        if same_process(identity):
            try:
                os.kill(identity["pid"], signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + timeout
    while any(same_process(p) for p in targets.values()) and time.monotonic() < deadline:
        time.sleep(0.1)
        targets.update({p["pid"]: p for p in owned_processes(record)})
    for identity in targets.values():
        if same_process(identity):
            try:
                os.kill(identity["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 2
    while any(same_process(p) for p in targets.values()) and time.monotonic() < deadline:
        time.sleep(0.05)
    if any(same_process(p) for p in targets.values()):
        raise RunError("owned processes have not exited after termination; their records were retained")
    record.update(stopped_at=time.time(), stop_targets=list(targets.values()))
    atomic_json(path, record)
    return {"stopped": True, "pids": sorted(targets)}


def check_port(port):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", port))
        except OSError as exc:
            raise RunError(f"port {port} is occupied; refusing to replace its owner") from exc


def check_gpu_memory(cfg):
    indices = cfg["train_gpus"]
    command = ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits",
               "--id=" + ",".join(map(str, indices))]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
        free = {int(row[0].strip()): float(row[1].strip()) / 1024 for row in csv.reader(result.stdout.splitlines()) if row}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        raise RunError(f"cannot verify GPU free memory: {exc}") from exc
    minimum = cfg["minimum_free_gpu_gib"]
    for index in indices:
        if index not in free or not math.isfinite(free[index]):
            raise RunError(f"GPU {index} free memory is unavailable")
        if free[index] < minimum:
            raise RunError(f"GPU {index} has {free[index]:.1f} GiB free, requires {minimum} GiB; no process was evicted")
    return {index: free[index] for index in indices}


def json_get(url):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=5) as response:
        return json.load(response)


def expected_episode(cfg):
    return {"profile": "captain", "enable_thinking": cfg.get("enable_thinking", "off"),
            **({"runtime_tool_contract": cfg["runtime_tool_contract"]} if cfg.get("runtime_tool_contract") else {}),
            **({"inline_task": True} if cfg.get("inline_task") else {}),
            **{k: cfg[k] for k in ("context_window", "max_output_tokens", "max_images", "wall_time", "grace_s", "max_requests", "max_actions", "max_tool_calls")},
            "student_profile": {"strategy": "window", "context_limit": cfg["context_limit"], "keep_recent_tokens": cfg["keep_recent_tokens"],
                                "tool_output_tokens": cfg["tool_output_tokens"], "spill": True, "hard_limit": cfg["context_hard_limit"]}}


def validate_health(cfg, node):
    host = cfg["nodes"][node]["host"]
    proxy = json_get(f"http://{host}:{cfg['proxy_port']}/health")
    if proxy.get("ok") is not True or proxy.get("upstream_health") != 200:
        raise RunError(f"node {node} proxy or upstream policy is not healthy")
    sampling = proxy.get("sampling") or {}
    neutral = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0,
               "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0}
    if any(sampling.get(k) != v for k, v in neutral.items()):
        raise RunError(f"node {node} sampling is not the expected unmodified policy distribution")
    health = json_get(f"http://{host}:{cfg['env_port']}/health")
    if health.get("ok") is not True or health.get("sample_format") != "per_request_v1" or health.get("reward_version") != cfg["reward_version"]:
        raise RunError(f"node {node} environment is not the requested raw-request service")
    if health.get("goal_audit_schema") != "hidden-object-audit/v2" or health.get("max_training_requests") != cfg["max_training_requests"]:
        raise RunError(f"node {node} environment audit/sampling schema differs")
    if any((health.get("episode") or {}).get(k) != v for k, v in expected_episode(cfg).items()):
        raise RunError(f"node {node} episode budgets/context differ from the manifest")
    if cfg.get("runtime_tool_contract") in CONTRACTS:
        try:
            receipt = validate_receipt(health.get("environment_preflight"), hostname=cfg["nodes"][node]["hostname"],
                                       contract=cfg["runtime_tool_contract"])
            saved = validate_receipt(json.loads(preflight_path(cfg, node).read_text()), hostname=cfg["nodes"][node]["hostname"],
                                     contract=cfg["runtime_tool_contract"])
        except (OSError, ValueError) as exc:
            raise RunError(f"node {node} has no valid sandbox toolchain preflight: {exc}") from exc
        if receipt["fingerprint"] != saved["fingerprint"] or health.get("tool_audit_schema") != "tool-execution-audit/v1":
            raise RunError(f"node {node} environment toolchain/audit contract differs")
    return health


def child_environment(cfg):
    env = dict(os.environ)
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        env.pop(key, None)
    hosts = [n["host"] for n in cfg["nodes"].values()]
    env.update(no_proxy=",".join(["127.0.0.1", "localhost", *hosts]), NO_PROXY=",".join(["127.0.0.1", "localhost", *hosts]),
               RL_REWARD_VERSION=cfg["reward_version"], HF_HOME=str(Path(cfg["workspace"]) / "hf_cache"),
               TOKENIZERS_PARALLELISM="false", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               NCCL_SOCKET_IFNAME=cfg["nccl_socket_ifname"], GLOO_SOCKET_IFNAME=cfg["nccl_socket_ifname"],
               PYTHONPATH=str(Path(cfg["repo"]) / "rl"), OMP_NUM_THREADS="2",
               # rank 0 alone can run rollouts/probes for > 8 min; the trainer waits on a gloo group, and this keeps the
               # NCCL watchdog heartbeat (default 480 s) from aborting on any remaining long wait
               TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="3600")
    env["PATH"] = f"{cfg['workspace']}/tools/node/bin:{cfg['workspace']}/bin:{env.get('PATH', '')}"
    # opt-in harness speedups (harness_fastloop.patch: identity-tested, physics / responses / depth bit-identical): config keys, default off
    for key, var in (("harness_fast_loop", "CAPTAIN_FAST_LOOP"), ("harness_lazy_render", "CAPTAIN_LAZY_RENDER")):
        if cfg.get(key):
            env[var] = "1"
        else:
            env.pop(var, None)
    return env


def preflight_path(cfg, node):
    return Path(cfg["run_dir"]) / "runtime" / node / "toolchain_preflight.json"


def run_toolchain_preflight(cfg, output, *, env=None):
    """Probe before allocating a model or accepting rollouts; preserve failure evidence."""
    if cfg.get("runtime_tool_contract") not in CONTRACTS:
        return None
    # Probe the exact interpreter path used by both rollout and evaluation
    # episodes, including the new checkout's virtual-environment link.
    episode_python = str(Path(cfg["harness"]) / ".venv-libero/bin/python")
    argv = [episode_python, str(Path(cfg["repo"]) / "rl/toolchain_preflight.py"), "--harness", cfg["harness"],
            "--contract", cfg["runtime_tool_contract"]] + (["--student-python", cfg["student_python"]] if cfg.get("student_python") else [])
    probe_env = dict(env or child_environment(cfg), PYTHONPATH=str(Path(cfg["harness"]) / "src"))
    try:
        result = subprocess.run(argv, cwd=cfg["repo"], env=probe_env, stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, timeout=150)
        receipt = json.loads(result.stdout.strip().splitlines()[-1])
        if not isinstance(receipt, dict):
            raise ValueError("preflight output is not an object")
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        receipt = {"schema_version": PREFLIGHT_SCHEMA, "runtime_tool_contract": cfg["runtime_tool_contract"], "ok": False,
                   "environment_error": {"schema_version": "harness-environment-error/v1", "source": "tool_preflight",
                       "code": "sandbox_preflight_failed", "verified": True, "details": {"error": f"{type(exc).__name__}: {exc}"}}}
        atomic_json(output, receipt)
        raise RunError(f"sandbox toolchain preflight could not run; inspect {output}") from exc
    atomic_json(output, receipt)
    try:
        validate_receipt(receipt, hostname=socket.gethostname().split(".")[0], contract=cfg["runtime_tool_contract"])
        if result.returncode != 0:
            raise ValueError(f"preflight process exited {result.returncode}")
    except ValueError as exc:
        raise RunError(f"sandbox toolchain preflight failed; inspect {output}: {exc}") from exc
    return receipt


def policy_ports(cfg, node):
    """The node's policy servers: policy_port plus the node's extra_policy_ports (default: the top-level list)."""
    extra = (cfg["nodes"].get(node) or {}).get("extra_policy_ports", cfg.get("extra_policy_ports") or [])
    return [cfg["policy_port"], *extra]


def policy_urls(cfg, node):
    """Where the node's proxy sends policy requests: the node's own policy_workers (streaming: one proxy per env-server
    node, each with its share of the policies), else explicit policy_workers (another node), else the local ports."""
    own = (cfg["nodes"].get(node) or {}).get("policy_workers")
    return list(own or cfg.get("policy_workers") or []) or [f"http://127.0.0.1:{p}" for p in policy_ports(cfg, node)]


def role_commands(cfg, node, resume=False):
    rl = Path(cfg["repo"]) / "rl"
    node_cfg = cfg["nodes"].get(node) or {}     # per-node overrides: extra_policy_ports, sam3_url, molmo_url
    commands = {
        "proxy": [cfg["python_env"], str(rl / "policy_proxy.py"), "--host", "0.0.0.0", "--port", str(cfg["proxy_port"]),
                  "--upstream", ",".join(policy_urls(cfg, node)), "--temperature", "1.0", "--top-p", "1.0",
                  "--group-size", str(cfg.get("num_generations") or 4),
                  "--sampler-topk", str(cfg.get("sampler_topk") or 0)],
        "env": [cfg["python_env"], str(rl / "env_server.py"), "--host", "0.0.0.0", "--port", str(cfg["env_port"]),
                "--harness", cfg["harness"], "--root", str(Path(cfg["run_dir"]) / f"episodes-{node}"),
                "--proxy", f"http://127.0.0.1:{cfg['proxy_port']}", "--egl-devices", str(cfg["nodes"][node]["egl"]),
                "--max-concurrency", str(node_cfg.get("env_concurrency", cfg["env_concurrency"])), "--profile", "captain", "--student-profile",
                "--context-strategy", "window", "--enable-thinking", str(cfg.get("enable_thinking", "off")), "--sam3-url", node_cfg.get("sam3_url", cfg["sam3_url"]),
                "--molmo-url", node_cfg.get("molmo_url", cfg["molmo_url"])],
    }
    for key in ENV_KEYS:
        commands["env"] += ["--" + key.replace("_", "-"), str(cfg[key])]
    if cfg.get("runtime_tool_contract"):
        commands["env"] += ["--runtime-tool-contract", cfg["runtime_tool_contract"]]
    if cfg.get("inline_task"):
        commands["env"] += ["--inline-task"]
    if cfg.get("student_python"):
        commands["env"] += ["--student-python", cfg["student_python"]]
    if cfg.get("solved_suffix_credit"):
        commands["env"] += ["--solved-suffix-credit"]
    if cfg.get("artifact_root"):
        commands["env"] += ["--artifact-root", str(cfg["artifact_root"])]
    if (cfg.get("heldout_monitor") or {}).get("ev12_harness"):
        commands["env"] += ["--alt-harness", "ev12=" + str(cfg["heldout_monitor"]["ev12_harness"])]
    if cfg.get("early_stop"):
        commands["env"] += ["--early-stop", json.dumps({"min_request": 5, "control": 0.0, **cfg["early_stop"], "critic_dir": cfg["critic_dir"],
                                                        "critic_live": str(Path(cfg["run_dir"]) / "cocritic_live.pt"),
                                                        "snapshot_root": "/dev/shm/captain-earlystop",
                                                        **({"exempt_tasks": sorted(cfg["es_exempt"]["tasks"])} if cfg.get("es_exempt") else {})},
                                                       sort_keys=True)]
    if cfg.get("code_reward_mode"):
        commands["env"] += ["--code-reward", str(cfg["code_reward_mode"])]
    if cfg.get("overlength_guard"):            # over-length guard (env side): from the config on every start / resume (pinned switch)
        commands["env"] += ["--max-request-tokens", str(max_request_tokens(cfg))]
    if node_cfg.get("raw_record_dir", cfg.get("raw_record_dir")):
        commands["env"] += ["--raw-record-dir", str(node_cfg.get("raw_record_dir", cfg.get("raw_record_dir")))]
    for hidden in cfg.get("sandbox_hide") or []:
        commands["env"] += ["--sandbox-hide", hidden]
    nodes = active_nodes(cfg)
    if node not in nodes:
        return commands
    master = cfg["nodes"][nodes[0]]["host"]
    train = [cfg["python_train"], "-m", "torch.distributed.run", "--nnodes", str(len(nodes)), "--node-rank", str(nodes.index(node)),
             "--nproc-per-node", str(len(cfg["train_gpus"])), "--master-addr", master, "--master-port", str(cfg["master_port"]),
             "--max-restarts", "0", str(rl / "train.py")]
    values = cfg
    if resume:
        checkpoint = latest_complete_checkpoint(Path(cfg["run_dir"]))
        info = json.loads((checkpoint / "trainer_state.json").read_text())
        values = info.get("args")
        if not isinstance(values, dict):
            raise RunError("checkpoint trainer_state.json lacks saved args; refusing to guess resume parameters")
        seeded = cfg.get("continue_from") and checkpoint.name == Path(cfg["continue_from"]).name \
            and Path(values["output_dir"]).resolve() == Path(cfg["continue_from"]).parent
        if Path(values["model"]).resolve() != Path(cfg["base_model"]) or not (seeded or Path(values["output_dir"]).resolve() == Path(cfg["run_dir"])):
            raise RunError("checkpoint belongs to a different base model or run")
        train += ["--resume-state", str(checkpoint / "training_state.pt"),
                  *([] if values.get("full_finetune") else ["--resume-adapter", str(checkpoint / "adapter")]),
                  "--start-step", str(info["global_step"])]
    train += ["--model", cfg["base_model"], "--output-dir", cfg["run_dir"], "--live-dir", cfg["live_dir"]]
    # a pinned kl_beta switch (TRAIN_KEY_ALLOWANCES) is taken from the config on resume; otherwise the saved value as before
    kl_from_config = resume and values.get("kl_beta") != cfg.get("kl_beta") and kl_beta_switch(values.get("kl_beta"), cfg.get("kl_beta"))
    for key in TRAIN_KEYS:
        # max_steps / save_steps: the run's current horizon and checkpoint cadence (relocate may change them); every other
        # training key from the saved args
        train += ["--" + key.replace("_", "-"), str(cfg[key] if key in ("max_steps", "save_steps") or (key == "kl_beta" and kl_from_config) else
                                                     values.get(key, {"grad_accum_rounds": 1, "act_cost": 0.0, "fail_cost": 0.0}[key])
                                                     if key in ("grad_accum_rounds", "act_cost", "fail_cost") else values[key])]
    # optional reward-cost arguments (None = off): from the saved args on resume, else the config
    for key in ("act_ref", "missed_finish_return", "exhaust_reward"):
        v = values.get(key) if resume else cfg.get(key)
        if v is not None:
            train += ["--" + key.replace("_", "-"), str(v)]
    for key in ("suites", "tasks", "seeds"):
        train += ["--" + key, *map(str, values[key])]
    if cfg.get("segment_bank"):
        train += ["--segment-bank", cfg["segment_bank"]]
        for key in SEGMENT_TRAIN_KEYS:
            train += ["--" + key.replace("_", "-"), str(values.get(key, DEFAULTS[key]))]
        if values.get("curriculum_sourced_only"):
            train += ["--curriculum-sourced-only"]
        if values.get("curriculum_fresh_horizon"):
            train += ["--curriculum-fresh-horizon", str(values["curriculum_fresh_horizon"])]
        if values.get("curriculum_suites"):
            train += ["--curriculum-suites", *map(str, values["curriculum_suites"])]
        if values.get("probe_set"):
            train += ["--probe-set", str(values["probe_set"])]
        if values.get("curriculum_focus_set"):
            train += ["--curriculum-focus-set", str(values["curriculum_focus_set"])]
        if values.get("segment_budget"):
            train += ["--segment-budget", str(values["segment_budget"])]
        if values.get("score_centering"):
            train += ["--score-centering"]
    if values.get("full_finetune"):
        train += ["--full-finetune"]
    if values.get("fresh_unique_states"):
        train += ["--fresh-unique-states"]
    if values.get("fresh_seeds_only"):
        train += ["--fresh-seeds-only"]
    if int(cfg.get("ref_every") or 1) != 1:    # speed-up 2: an infrastructure key, from the config (not the saved args)
        train += ["--ref-every", str(int(cfg["ref_every"]))]
    if cfg.get("compile_regional"):
        train += ["--compile-regional"]
    if float(cfg.get("keep_attn_act_gib") or 0) > 0:
        train += ["--keep-attn-act-gib", str(float(cfg["keep_attn_act_gib"]))]
    if cfg.get("attn_backend") not in (None, "sdpa"):
        train += ["--attn-backend", str(cfg["attn_backend"])] + (["--attn-overlay", str(cfg["attn_overlay"])] if cfg.get("attn_overlay") else [])
    if cfg.get("prefix_tree"):                 # shared-prefix training: an infrastructure key, from the config (not the saved args)
        train += ["--prefix-tree"]
    if cfg.get("heldout_monitor"):
        hm = cfg["heldout_monitor"]
        train += ["--heldout-monitor", json.dumps({"every": hm["every"], "ev12_seeds": hm.get("ev12_seeds") or [],
                                                   "official_seeds": hm.get("official_seeds") or []}, sort_keys=True)]
    if cfg.get("early_stop"):                  # speed-up 3: the co-critic weights for the env servers' early-stop monitor
        train += ["--cocritic-live", str(Path(cfg["run_dir"]) / "cocritic_live.pt")]
    if cfg.get("reward_patch"):                # missed-finish fix part C: from the config (not the saved args)
        train += ["--reward-patch", json.dumps(cfg["reward_patch"], sort_keys=True)]
    if cfg.get("staged_reward"):               # staged subgoal reward: from the config on every start / resume (pinned switch)
        train += ["--staged-reward", json.dumps(cfg["staged_reward"], sort_keys=True)]
    if cfg.get("overlength_guard"):            # over-length guard (trainer side): from the config on every start / resume (pinned switch)
        train += ["--max-overlength-drops", str(int(cfg["overlength_guard"]["max_drops"]))]
    if cfg.get("full_state_dir"):            # where THIS run writes its node-local state (a layout key: not the saved args,
        train += ["--full-state-dir", str(cfg["full_state_dir"])]   # which on a continued run name the source run's directory)
    if values.get("curriculum_manifest"):
        train += ["--curriculum-manifest", str(values["curriculum_manifest"])]
    if values.get("boundary_fresh_only"):
        train += ["--boundary-fresh-only"]
    if values.get("reward_mode"):
        train += ["--reward-mode", str(values["reward_mode"])]
    if values.get("loss_normalization", "token_mean") != "token_mean":
        train += ["--loss-normalization", str(values["loss_normalization"])]
    if values.get("skip_zero_signal"):
        train += ["--skip-zero-signal"]
    if values.get("warmup_attempts"):
        train += ["--warmup-attempts", str(values["warmup_attempts"])]
    if cfg.get("offload_activations", "auto") != "auto":
        train += ["--offload-activations", str(cfg["offload_activations"])]
    if cfg.get("async_lookahead"):
        train += ["--async-lookahead", str(int(cfg["async_lookahead"]))]
    if cfg.get("is_cap") is not None:
        train += ["--is-cap", str(float(cfg["is_cap"]))]
    if cfg.get("advantage_shuffle"):
        train += ["--advantage-shuffle"]
    if cfg.get("advantage_mode", "grpo") != "grpo":
        train += ["--advantage-mode", str(cfg["advantage_mode"]), "--critic-dir", str(cfg["critic_dir"]),
                  "--gae-lambda", str(float(cfg["gae_lambda"]))]
    if cfg.get("false_finish_reward"):
        train += ["--false-finish-reward", str(float(cfg["false_finish_reward"]))]
    if cfg.get("start_weights"):
        train += ["--start-weights", str(cfg["start_weights"])]
    if cfg.get("start_exclude"):               # thin-basket fix: from the config (not the saved args), so relocate can set it
        train += ["--start-exclude", str(cfg["start_exclude"])]
    if cfg.get("critic_cotrain"):
        train += ["--critic-cotrain", "--critic-hidden", str(cfg["critic_hidden"]), "--critic-lr", str(float(cfg["critic_lr"])),
                  "--critic-updates", str(int(cfg["critic_updates"]))]
    if cfg.get("ratio_mask_low") is not None:
        train += ["--ratio-mask-low", str(float(cfg["ratio_mask_low"])), "--ratio-mask-high", str(float(cfg["ratio_mask_high"]))]
    if cfg.get("code_reward_mode"):
        train += ["--code-reward", str(cfg["code_reward_mode"]), "--code-reward-weight", str(float(cfg["code_reward_weight"])),
                  "--code-reward-budget", str(int(cfg["code_reward_budget"]))]
    if cfg.get("pack_prefixes"):
        train += ["--pack-prefixes"]
    if cfg.get("pack_assignment", "trajectory") != "trajectory":
        train += ["--pack-assignment", str(cfg["pack_assignment"])]
    if cfg.get("async_checkpoint"):
        train += ["--async-checkpoint"]
    if cfg.get("lookahead_through_checkpoints"):
        train += ["--lookahead-through-checkpoints"]
    if cfg.get("streaming"):
        train += ["--streaming", "--max-staleness", str(int(cfg["max_staleness"]))]
        if (cfg.get("stream_group_size") or 1) > 1:
            train += ["--stream-group-size", str(int(cfg["stream_group_size"])), "--group-advantage", str(cfg.get("group_advantage") or "group")]
    commands["train"] = train
    return commands


def record_path(cfg, node, role):
    return Path(cfg["run_dir"]) / "runtime" / node / f"{role}.json"


def start_role(manifest, node, role, argv, env):
    cfg = manifest["config"]
    path = record_path(cfg, node, role)
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["hostname"] != socket.gethostname():
            raise RunError("existing process record belongs to another host")
        if same_process(previous):
            if previous["argv"] != argv:
                raise RunError(f"owned {role} process has different arguments")
            return {"already_running": True, **previous}
        if process_identity(previous["pid"]) is not None or owned_processes(previous):
            raise RunError(f"stale {role} identity or surviving children; inspect/stop before launching")
    log = Path(cfg["run_dir"]) / "logs" / f"{node}-{role}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    owner = f"{manifest['run_id']}:{node}:{role}:{uuid.uuid4().hex}"
    env = dict(env, FRESH_RUN_OWNER=owner)
    with log.open("ab") as stream:
        proc = subprocess.Popen(argv, cwd=cfg["repo"], env=env, stdin=subprocess.DEVNULL,
                                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    identity = process_identity(proc.pid)
    if identity is None:
        raise RunError(f"{role} exited during launch; see {log}")
    result = {**identity, "node": node, "role": role, "owner": owner, "argv": argv, "log": str(log), "started_at": time.time()}
    atomic_json(path, result)
    time.sleep(0.2)
    if not same_process(result):
        raise RunError(f"{role} exited immediately; see {log}")
    return result


def training_history_exists(run_dir):
    root = Path(run_dir)
    return any((root / name).exists() for name in (
        "metrics.jsonl", "rollouts.jsonl", "training_state.pt", "trainer_state.json", "final-adapter",
    )) or any(root.glob("checkpoint-*")) or any(root.glob(".checkpoint-*"))


def initialize(path, overrides, node):
    cfg = normalize_config(overrides)
    validate_node(cfg, node)
    if Path(path).resolve().parent != Path(cfg["run_dir"]):
        raise RunError("manifest must live directly inside run_dir")
    if not (Path(cfg["base_model"]) / "config.json").is_file():
        raise RunError("base_model must be a complete local SFT model directory")
    if cfg.get("continue_from"):
        validate_continue_from(cfg)
    hashes = source_hashes(cfg)
    old = json.loads(Path(path).read_text()) if Path(path).exists() else None
    history_exists = training_history_exists(cfg["run_dir"])
    if old is None and history_exists:
        raise RunError("training history already exists without its manifest; refusing to initialize over it")
    if old is not None:
        changed = old["config"] != cfg or old["source_hashes"] != hashes
        if not changed:
            return old
        if history_exists:
            raise RunError("manifest config and sources are immutable once training history exists")
        # only the role records carry stopped_at (runtime/<node>/ also holds the toolchain preflight receipt)
        roles = [p for p in (Path(cfg["run_dir"]) / "runtime").glob("*/*.json") if p.stem in ("train", "env", "proxy")]
        if changed and any(not json.loads(p.read_text()).get("stopped_at") for p in roles):
            raise RunError("stop all owned roles before changing an initialized manifest")
    manifest = {"manifest_version": 1, "run_id": old["run_id"] if old else uuid.uuid4().hex,
                "created_at": old["created_at"] if old else time.time(), "config": cfg, "source_hashes": hashes}
    atomic_json(path, manifest)
    return manifest


# where the roles run (hosts, GPUs, service URLs, node-local directories); relocate may change only these
LAYOUT_KEYS = ("nnodes", "single_node", "rollout_nodes", "train_nodes", "train_gpus", "minimum_free_gpu_gib", "nodes", "extra_policy_ports",
               "policy_workers", "nccl_socket_ifname", "master_port", "sam3_url", "molmo_url", "env_concurrency",
               "raw_record_dir", "full_state_dir", "offload_activations", "pack_prefixes",   # computation only; objective unchanged
               "repo",      # a new copy of the code (e.g. resume onto another world size); changed files are recorded
               "artifact_root", "ref_every",   # speed-ups (rollout storage location; 1-in-N KL reference)
               "early_stop",                   # co-critic early stop (changes which requests are generated)
               "attn_backend", "attn_overlay",  # attention kernel (numerics within bf16 kernel noise; checked before use)
               "compile_regional", "keep_attn_act_gib",
               "stream_group_size", "group_advantage",   # group sampling
               "prefix_tree", "pack_assignment",   # shared-prefix computation / rank assignment (objective unchanged)
               "harness_fast_loop", "harness_lazy_render",
               "heldout_monitor",   # no-gradient held-out monitor episodes
               "reward_patch")      # missed-finish reward change; the harness fast-loop gates above are identity-tested


# Mid-run changes. relocate() normally accepts only LAYOUT_KEYS (and max_steps / save_steps / harness / start_exclude); every
# other config key is frozen for the lifetime of a run. A run may additionally switch a frozen key mid-run along a PINNED pair
# (forward = from -> to, reverse = the rollback). The tables below hold those pins; they are empty in the release (a fresh run
# simply starts with the final values in its config). Each accepted switch is logged in manifest.relocations.

# start-exclude/v1 files relocate may switch a run onto (sha256 -> description); null (switching it off again) is always allowed
START_EXCLUDE_ALLOWED: dict = {}


# harness files whose changes are host-side only (never visible to the model): live request marks (server/env_server.py) and
# the fast-loop / lazy-render gates (CAPTAIN_FAST_LOOP / CAPTAIN_LAZY_RENDER; default off = the original code path; with them on,
# physics, tool responses and depth are bit-identical and RGB within the renderer's own +-1 nondeterminism)
HARNESS_RELOCATABLE = ("harness/src/captain_harness/server/env_server.py",
                       "harness/src/captain_harness/ports/capability.py", "harness/src/captain_harness/adapters/capx/direct_programming.py",
                       "harness/src/captain_harness/adapters/libero/backend.py", "harness/src/captain_harness/adapters/libero/fast_loop.py")


# Model-visible harness versions relocate may switch a run onto (and back from). Keyed by the harness copy's COMMIT file (first line);
# "files" = exactly the changed source files, each pinned to its sha256 before ("base") and after ("new") the version change.
# Any other non-host-side difference is refused. Logged as harness_version_allowance (model_visible: true) in manifest.relocations.
HARNESS_VERSION_ALLOWANCES: dict = {}


# frozen training keys relocate may change along one pinned pair, e.g.
#   {"kl_beta": {"from": 0.001, "to": 0.002, "what": "...", "goal_ref": "..."}}
# role_commands passes a pinned kl_beta switch from the config on --resume (otherwise the saved args win), and the trainer's resume
# check (training_resume.RESUME_SWITCHES, same pins) must accept it.
TRAIN_KEY_ALLOWANCES: dict = {}


# start-weights/v1 files relocate may switch a run between (both sides pinned by sha256), e.g.
#   {"name": {"from": "<sha256>", "to": "<sha256>", "what": "...", "goal_ref": "..."}}
# The trainer's resume check and SegmentCurriculum.load_state_dict must accept the same pinned pair (training_resume.RESUME_SWITCHES).
START_WEIGHTS_ALLOWANCES: dict = {}


# staged_reward and es_exempt may change mid-run only TOGETHER, only between off (both None) and exactly one pinned spec:
# sha256 of the canonical JSON stage_spec_text({"es_exempt": ..., "staged_reward": ...}) (e.g. configs/staged-reward-spec.json), e.g.
#   {"name": {"sha256": "<sha256>", "what": "...", "goal_ref": "..."}}
STAGE_SWITCH_ALLOWANCES: dict = {}
STAGE_SWITCH_KEYS = ("es_exempt", "staged_reward")


# overlength_guard may change mid-run only between None (off) and a pinned value, e.g.
#   {"name": {"value": {"max_drops": 2}, "what": "...", "goal_ref": "..."}}
OVERLENGTH_GUARD_ALLOWANCES: dict = {}


def overlength_guard_switch(old_cfg: dict, new_cfg: dict) -> dict | None:
    """The OVERLENGTH_GUARD_ALLOWANCES record for old -> new overlength_guard, None when unchanged; RunError otherwise."""
    a, b = old_cfg.get("overlength_guard"), new_cfg.get("overlength_guard")
    if a == b:
        return None
    for name, entry in OVERLENGTH_GUARD_ALLOWANCES.items():
        for direction, off, on in (("forward", a, b), ("reverse", b, a)):
            if off is None and on == entry["value"]:
                return {"name": name, "direction": direction, "from": a, "to": b, "what": entry["what"], "goal_ref": entry["goal_ref"]}
    raise RunError(f"overlength_guard may change only between None and a pinned OVERLENGTH_GUARD_ALLOWANCES value; requested {a} -> {b}")


def validate_overlength_guard(cfg) -> None:
    og = cfg.get("overlength_guard")
    if og is None:
        return
    if not isinstance(og, dict) or set(og) != {"max_drops"} or type(og["max_drops"]) is not int or not 1 <= og["max_drops"] <= 8:
        raise RunError("overlength_guard must be null or {\"max_drops\": 1..8}")
    for key in ("max_seq_tokens", "hard_max_tokens"):
        if type(cfg.get(key)) is not int or cfg[key] < 1:
            raise RunError(f"overlength_guard needs a positive integer {key}")


def max_request_tokens(cfg) -> int:
    """the trainer's per-request hard cap (train: min(args.max_seq_tokens, args.hard_max_tokens))"""
    return min(int(cfg["max_seq_tokens"]), int(cfg["hard_max_tokens"]))


def stage_spec_text(pair: dict) -> str:
    """canonical JSON of {"es_exempt": ..., "staged_reward": ...} (the pinned spec file's exact bytes, no trailing newline)"""
    return json.dumps({k: pair.get(k) for k in STAGE_SWITCH_KEYS}, sort_keys=True, separators=(",", ":"))


def stage_switch(old_cfg: dict, new_cfg: dict) -> dict | None:
    """The STAGE_SWITCH_ALLOWANCES record for old -> new staged_reward / es_exempt, None when neither changes; RunError otherwise."""
    if all(old_cfg.get(k) == new_cfg.get(k) for k in STAGE_SWITCH_KEYS):
        return None
    off = lambda c: all(c.get(k) is None for k in STAGE_SWITCH_KEYS)  # noqa: E731
    sha = lambda c: hashlib.sha256(stage_spec_text(c).encode()).hexdigest()  # noqa: E731
    for name, entry in STAGE_SWITCH_ALLOWANCES.items():
        for direction, a, b in (("forward", old_cfg, new_cfg), ("reverse", new_cfg, old_cfg)):
            if off(a) and not off(b) and sha(b) == entry["sha256"]:
                return {"name": name, "direction": direction, "sha256": entry["sha256"], "from": {k: old_cfg.get(k) for k in STAGE_SWITCH_KEYS},
                        "to": {k: new_cfg.get(k) for k in STAGE_SWITCH_KEYS}, "what": entry["what"], "goal_ref": entry["goal_ref"]}
    raise RunError(f"staged_reward / es_exempt may change only together between off and a pinned STAGE_SWITCH_ALLOWANCES spec; "
                   f"requested sha256 {sha(old_cfg)[:12]} -> {sha(new_cfg)[:12]}")


def pinned_direction(entry, old, new):
    """forward / reverse when old -> new is the entry's pinned pair (either way round), else None."""
    if (old, new) == (entry["from"], entry["to"]):
        return "forward"
    if (old, new) == (entry["to"], entry["from"]):
        return "reverse"
    return None


def kl_beta_switch(old, new):
    """The TRAIN_KEY_ALLOWANCES kl_beta direction for old -> new (numbers only), else None."""
    entry = TRAIN_KEY_ALLOWANCES.get("kl_beta")
    numeric = all(type(v) in (int, float) for v in (old, new))
    return pinned_direction(entry, float(old), float(new)) if entry and numeric else None


def harness_commit(root):
    path = Path(root) / "COMMIT"
    return path.read_text().split("\n", 1)[0].strip() if path.is_file() else None


def harness_src_hashes(root):
    """harness/<rel> -> sha256 for the files source_hashes() takes from a harness copy's src tree (same extensions)."""
    root = Path(root)
    extensions = {".py", ".js", ".json", ".jinja", ".jinja2"}
    return {f"harness/{p.relative_to(root)}": hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root / "src").rglob("*"))
            if p.is_file() and p.suffix in extensions and "__pycache__" not in p.parts}


def harness_version_allowance(old_root, new_root, old_hashes, new_hashes):
    """None when the two harness copies differ only in HARNESS_RELOCATABLE; the allowance record when the non-host-side differences
    are exactly one pinned HARNESS_VERSION_ALLOWANCES version change (forward or reverse); RunError otherwise."""
    differ = sorted(k for k in set(old_hashes) | set(new_hashes) if k.startswith("harness/") and old_hashes.get(k) != new_hashes.get(k))
    extra = sorted(set(differ) - set(HARNESS_RELOCATABLE))
    if not extra:
        return None
    for version, entry in HARNESS_VERSION_ALLOWANCES.items():
        files = entry["files"]
        if set(extra) != set(files):
            continue
        for direction, root, src, dst in (("forward", new_root, "base", "new"), ("reverse", old_root, "new", "base")):
            if harness_commit(root) == version and all(old_hashes.get(f) == pin[src] and new_hashes.get(f) == pin[dst]
                                                       for f, pin in files.items()):
                return {"version": version, "direction": direction, "model_visible": True, "files": {f: {"from": pin[src], "to": pin[dst]}
                        for f, pin in files.items()}, "what": entry["what"], "goal_ref": entry["goal_ref"],
                        "old_commit": harness_commit(old_root), "new_commit": harness_commit(new_root)}
    raise RunError(f"a relocated harness may differ only in {HARNESS_RELOCATABLE} or by exactly one pinned HARNESS_VERSION_ALLOWANCES "
                   f"version; it differs in {extra}")


def relocate(path, overrides, node):
    """Move an initialized run (with training history) to other nodes/GPUs: every non-layout key must be unchanged and every
    owned role stopped; sources may change only with a new repo copy. The old layouts, changed source files and their
    old hashes stay in manifest["relocations"]. Resume with train --resume."""
    old = json.loads(Path(path).read_text())
    cfg = normalize_config(overrides)
    validate_node(cfg, node)
    if Path(path).resolve().parent != Path(cfg["run_dir"]) or Path(cfg["run_dir"]) != Path(old["config"]["run_dir"]):
        raise RunError("relocate keeps the run directory; the manifest must live directly inside it")
    changed = sorted(k for k in set(cfg) | set(old["config"]) if cfg.get(k) != old["config"].get(k))
    # save_steps: checkpoint cadence only, no schedule or objective depends on it
    # a new harness copy is relocatable when it differs only in host-side bookkeeping files (checked below)
    frozen = [k for k in changed if k not in LAYOUT_KEYS and k not in ("max_steps", "save_steps", "harness", "start_exclude")
              and k not in (("kl_beta",) if "kl_beta" in TRAIN_KEY_ALLOWANCES else ()) + (("start_weights",) if START_WEIGHTS_ALLOWANCES else ())
              + (STAGE_SWITCH_KEYS if STAGE_SWITCH_ALLOWANCES else ())
              + (("overlength_guard",) if OVERLENGTH_GUARD_ALLOWANCES else ())]
    allowance = train_key_allowance = start_weights_allowance = None
    stage_allowance = stage_switch(old["config"], cfg) if STAGE_SWITCH_ALLOWANCES else None   # (RunError if not pinned)
    guard_allowance = overlength_guard_switch(old["config"], cfg) if OVERLENGTH_GUARD_ALLOWANCES else None   # (RunError if not pinned)
    if "kl_beta" in changed and "kl_beta" in TRAIN_KEY_ALLOWANCES:     # (without the entry kl_beta stays frozen, as before)
        old_kl, new_kl = old["config"].get("kl_beta"), cfg.get("kl_beta")
        direction = kl_beta_switch(old_kl, new_kl)
        if direction is None:
            raise RunError(f"kl_beta may change only along the pinned TRAIN_KEY_ALLOWANCES pair; requested {old_kl} -> {new_kl}")
        entry = TRAIN_KEY_ALLOWANCES["kl_beta"]
        train_key_allowance = {"key": "kl_beta", "from": old_kl, "to": new_kl, "direction": direction, "what": entry["what"],
                               "goal_ref": entry["goal_ref"]}
    if "start_weights" in changed and START_WEIGHTS_ALLOWANCES:        # (without entries start_weights stays frozen, as before)
        old_sw, new_sw = old["config"].get("start_weights"), cfg.get("start_weights")
        if not old_sw or not new_sw:
            raise RunError("start_weights may only switch between two pinned START_WEIGHTS_ALLOWANCES files (not on or off)")
        old_sha = old.get("source_hashes", {}).get(str(Path(old_sw))) or hashlib.sha256(Path(old_sw).read_bytes()).hexdigest()
        new_sha = hashlib.sha256(Path(new_sw).read_bytes()).hexdigest()
        for name, entry in START_WEIGHTS_ALLOWANCES.items():
            direction = pinned_direction(entry, old_sha, new_sha)
            if direction:
                start_weights_allowance = {"name": name, "direction": direction, "from": {"path": old_sw, "sha256": old_sha},
                                           "to": {"path": new_sw, "sha256": new_sha}, "what": entry["what"], "goal_ref": entry["goal_ref"]}
                break
        else:
            raise RunError(f"start_weights {old_sw} (sha256 {old_sha}) -> {new_sw} (sha256 {new_sha}) is not a pinned "
                           f"START_WEIGHTS_ALLOWANCES switch")
    if "start_exclude" in changed and cfg.get("start_exclude"):
        sha = hashlib.sha256(Path(cfg["start_exclude"]).read_bytes()).hexdigest()
        if sha not in START_EXCLUDE_ALLOWED:
            raise RunError(f"start_exclude {cfg['start_exclude']} (sha256 {sha}) is not in START_EXCLUDE_ALLOWED")
        allowance = {"sha256": sha, "what": START_EXCLUDE_ALLOWED[sha]}
    if "save_steps" in changed and (type(cfg["save_steps"]) is not int or cfg["save_steps"] < 1):
        raise RunError("save_steps must be a positive integer")
    if "max_steps" in changed:
        # the horizon only bounds the loop (no schedule depends on it): extend it, never below the completed rounds
        done = max((int(json.loads(l).get("step") or 0) for l in (Path(cfg["run_dir"]) / "metrics.jsonl").read_text().splitlines()
                    if l.strip()), default=0) if (Path(cfg["run_dir"]) / "metrics.jsonl").exists() else 0
        if cfg["max_steps"] < old["config"]["max_steps"] or cfg["max_steps"] <= done:
            raise RunError(f"max_steps may only grow beyond the completed rounds ({done}); requested {cfg['max_steps']}")
    if frozen:
        raise RunError(f"relocate may only change the layout; these would change: {', '.join(frozen)}")
    hashes = source_hashes(cfg)
    def by_role(c, h):                         # repo/<rel> and harness/<rel> so a new copy of the same code compares equal
        roots = {"repo": Path(c["repo"]), "harness": Path(c["harness"])}
        out = {}
        for key, value in h.items():
            label = next((f"{name}/{Path(key).relative_to(root)}" for name, root in roots.items() if Path(key).is_relative_to(root)), key)
            out[label] = value
        return out
    new_rel, old_rel = by_role(cfg, hashes), by_role(old["config"], old["source_hashes"])
    code = sorted(k for k in set(new_rel) | set(old_rel) if new_rel.get(k) != old_rel.get(k))
    data = {p for c in (cfg, old["config"]) if c.get("start_exclude") for p in [c["start_exclude"]]}
    if start_weights_allowance:              # the pinned start-weights files are data, not code (no new repo copy needed)
        data |= {start_weights_allowance["from"]["path"], start_weights_allowance["to"]["path"]}
    if [c for c in code if c not in data] and "repo" not in changed:
        raise RunError("source hashes changed since initialization; relocate onto a new repo copy instead")
    harness_code = [c for c in code if c.startswith("harness/")]
    if harness_code and "harness" not in changed:
        raise RunError("harness sources changed in place; relocate onto a new harness copy instead")
    harness_allowance = monitor_allowance = None
    if "harness" in changed and set(harness_code) - set(HARNESS_RELOCATABLE):
        harness_allowance = harness_version_allowance(old["config"]["harness"], cfg["harness"],
                                                      {k: v for k, v in old_rel.items() if k.startswith("harness/")},
                                                      {k: v for k, v in new_rel.items() if k.startswith("harness/")})
    # the held-out monitor's ev12 harness copy is not hashed; it must switch together with a training-harness version change, and it
    # may carry a pinned version change only together with the same change of the training harness (series break marked in the record)
    old_ev = (old["config"].get("heldout_monitor") or {}).get("ev12_harness")
    new_ev = (cfg.get("heldout_monitor") or {}).get("ev12_harness")
    if harness_allowance and new_ev and new_ev == old_ev:
        raise RunError("the training harness changes version; heldout_monitor.ev12_harness must move to a copy of the same version")
    if old_ev and new_ev and old_ev != new_ev and (harness_allowance or (Path(old_ev).is_dir() and Path(new_ev).is_dir())):
        ev_old, ev_new = harness_src_hashes(old_ev), harness_src_hashes(new_ev)
        pinned = {f for entry in HARNESS_VERSION_ALLOWANCES.values() for f in entry["files"]}
        touched = {k for k in set(ev_old) | set(ev_new) if ev_old.get(k) != ev_new.get(k)}
        if harness_allowance or touched & pinned:    # otherwise unchanged behaviour (the ev12 copy is a layout key)
            try:
                monitor_allowance = harness_version_allowance(old_ev, new_ev, ev_old, ev_new)
            except RunError as exc:
                raise RunError(f"heldout_monitor.ev12_harness: {exc}") from None
            pick = lambda a: a and (a["version"], a["direction"])
            if pick(monitor_allowance) != pick(harness_allowance):
                raise RunError(f"heldout_monitor.ev12_harness version change {pick(monitor_allowance)} must match the training harness "
                               f"version change {pick(harness_allowance)}")
    roles = [p for p in (Path(cfg["run_dir"]) / "runtime").glob("*/*.json") if p.stem in ("train", "env", "proxy")]
    if any(not json.loads(p.read_text()).get("stopped_at") for p in roles):
        raise RunError("stop all owned roles on every node before relocating")
    if not changed:
        return old
    moved = {"at": time.time(), "changed": changed, "previous": {k: old["config"].get(k) for k in changed},
             "changed_sources": code, "previous_source_hashes": {k: old_rel.get(k) for k in code},
             **({"start_exclude_allowance": allowance} if allowance else {}),
             **({"train_key_allowance": train_key_allowance} if train_key_allowance else {}),
             **({"start_weights_allowance": start_weights_allowance} if start_weights_allowance else {}),
             **({"stage_switch_allowance": stage_allowance} if stage_allowance else {}),
             **({"overlength_guard_allowance": guard_allowance} if guard_allowance else {}),
             **({"harness_version_allowance": harness_allowance} if harness_allowance else {}),
             **({"heldout_monitor_harness_allowance": monitor_allowance} if monitor_allowance else {})}
    manifest = {**old, "config": cfg, "source_hashes": hashes, "relocations": [*old.get("relocations", []), moved]}
    atomic_json(path, manifest)
    return manifest


def run_action(manifest, node, action, resume=False):
    cfg = manifest["config"]
    validate_node(cfg, node)
    if action == "status":
        roles = {}
        for role in ("train", "env", "proxy"):
            path = record_path(cfg, node, role)
            saved = json.loads(path.read_text()) if path.exists() else None
            roles[role] = {"running": bool(saved and same_process(saved)), "record": saved}
        metrics = Path(cfg["run_dir"]) / "metrics.jsonl"
        last = None
        for line in reversed(metrics.read_text().splitlines() if metrics.exists() else []):
            try:
                candidate = json.loads(line)
                if isinstance(candidate, dict) and type(candidate.get("step")) is int:
                    last = candidate
                    break
            except ValueError:
                continue
        return {"node": node, "roles": roles, "last_metric": last, "completed": bool(last and last["step"] >= cfg["max_steps"])}
    if action == "stop":
        return {role: stop_record(record_path(cfg, node, role)) for role in ("train", "env", "proxy")}
    if source_hashes(cfg) != manifest["source_hashes"]:
        raise RunError("source hashes changed since initialization")
    commands = role_commands(cfg, node, resume=resume)
    env = child_environment(cfg)
    if action == "services":
        if node not in cfg["rollout_nodes"]:
            raise RunError(f"node {node} is not enabled for rollout services")
        if cfg.get("runtime_tool_contract") in CONTRACTS:
            path = preflight_path(cfg, node)
            run_toolchain_preflight(cfg, path, env=env)
            env["FRESH_ENV_PREFLIGHT_PATH"] = str(path)
        for role, port in (("proxy", cfg["proxy_port"]), ("env", cfg["env_port"])):
            path = record_path(cfg, node, role)
            if not path.exists() or not same_process(json.loads(path.read_text())):
                check_port(port)
        result = {role: start_role(manifest, node, role, commands[role], env) for role in ("proxy", "env")}
        deadline = time.monotonic() + 30
        while True:
            try:
                result["health"] = validate_health(cfg, node)
                return result
            except Exception as exc:
                if time.monotonic() >= deadline:
                    raise RunError(f"services did not become healthy: {exc}") from exc
                time.sleep(1)
    if action == "train":
        if node not in active_nodes(cfg):
            raise RunError(f"node {node} is not enabled for training")
        for other in cfg["rollout_nodes"]:
            validate_health(cfg, other)
        path = record_path(cfg, node, "train")
        if path.exists() and same_process(json.loads(path.read_text())):
            return {"already_running": True, **json.loads(path.read_text())}
        free_gpu_memory = check_gpu_memory(cfg)
        if not resume and training_history_exists(cfg["run_dir"]):
            raise RunError("existing training state requires --resume; fresh runs never reuse RL adapters")
        if node == active_nodes(cfg)[0]:
            path = record_path(cfg, node, "train")
            if not path.exists() or not same_process(json.loads(path.read_text())):
                check_port(cfg["master_port"])
        if not resume and cfg.get("continue_from"):
            seed_from_checkpoint(cfg)          # the first launch of a continued run
            resume = True
            commands = role_commands(cfg, node, resume=True)
        nodes = cfg["rollout_nodes"]
        trainer_host = active_nodes(cfg)[0]
        env["RL_ENV_SERVER"] = ",".join(f"http://{cfg['nodes'][n]['host']}:{cfg['env_port']}" for n in nodes)
        env["RL_POLICY_URL"] = ",".join(cfg.get("policy_workers") or
                                        [f"http://{'127.0.0.1' if n == trainer_host else cfg['nodes'][n]['host']}:{p}"
                                         for n in nodes for p in policy_ports(cfg, n)])
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cfg["train_gpus"]))
        if cfg.get("streaming"):     # the trainer replaces weights behind every env-server node's proxy gate
            env["RL_GATE_URLS"] = ",".join(f"http://{cfg['nodes'][n]['host']}:{cfg['proxy_port']}" for n in nodes)
        return {**start_role(manifest, node, "train", commands["train"], env), "gpu_free_gib_before_launch": free_gpu_memory}
    raise RunError(f"unsupported action: {action}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(DEFAULT_RUN) / "manifest.json")
    parser.add_argument("--node", choices=NODE_IDS, required=True)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("init", "relocate"):
        sub.add_parser(name).add_argument("--config", type=Path, required=True)
    for name in ("status", "services", "stop"):
        sub.add_parser(name)
    sub.add_parser("train").add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        lock = args.manifest.parent / (".manifest.lock" if args.action in ("init", "relocate") else f"runtime/{args.node}/.operation.lock")
        with operation_lock(lock):
            if args.action == "init":
                result = initialize(args.manifest, load_config(args.config), args.node)
            elif args.action == "relocate":
                result = relocate(args.manifest, load_config(args.config), args.node)
            else:
                result = run_action(json.loads(args.manifest.read_text()), args.node, args.action, getattr(args, "resume", False))
        print(json.dumps(result, indent=2))
    except (RunError, OSError, ValueError, KeyError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
