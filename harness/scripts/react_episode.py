#!/usr/bin/env python3
"""Run ONE episode with the V1.1 Codex-style driver against the environment server.

    .venv-libero/bin/python scripts/react_episode.py --suite libero_spatial --task-id 0 --seed 11 \
        --model captain-4b --api chat --endpoint http://127.0.0.1:8600/v1 --auth none --profile captain \
        --output-dir outputs/react-spatial0 --mujoco-gl egl   (see README.md for the full CaPtain-4B evaluation flags)

Steps: start the env server -> build the workspace (TASK.md, robot.py, initial PNGs) -> render the
captured Codex request template for the model -> run the bare loop inside a bwrap sandbox ->
when the loop ends or the wall budget is hit, call /finish if the agent did not -> write
episode_result.json in the Codex-reference shape (plus a `driver` block; `codex` is an alias).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from captain_harness.react.apply_patch import apply_patch  # noqa: E402
from captain_harness.react.catalog import profile_for, small_context_profile  # noqa: E402
from captain_harness.react.chat_llm import IMPLEMENTED_TOOLS, INCOMPLETE_NUDGE_TEXT, ChatCompletionsClient, convert_tools  # noqa: E402
from captain_harness.react.code_mode import run_exec_script  # noqa: E402
from captain_harness.react.llm import ResponsesClient  # noqa: E402
from captain_harness.react.loop import Loop  # noqa: E402
from captain_harness.react.python_repl import PythonRepl, ReplStartError, run_python_cell  # noqa: E402
from captain_harness.react.replay import ReplayClient, load_responses  # noqa: E402
from captain_harness.react.toolchain_probe import probe_student_runtime  # noqa: E402
from captain_harness.react.request import render_request  # noqa: E402
from captain_harness.react.sandbox import Sandbox  # noqa: E402
from captain_harness.react.tools import ToolRouter  # noqa: E402
from captain_harness.react.unified_exec import SessionManager, child_environment  # noqa: E402
from captain_harness.server import python_session  # noqa: E402
from captain_harness.server.python_session import PYTHON_PROMPT  # noqa: E402
from captain_harness.server.episode_launch import (  # noqa: E402
    DEFAULT_PROMPT, REPO_ROOT, build_workspace, http_json, initial_image_files, inline_task_items, restore_workspace, snapshot_workspace,
    start_server, stop_server, wait_for_server,
)

def default_endpoint() -> str | None:
    """No built-in endpoint: pass --endpoint or set REACT_ENDPOINT."""
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", required=True)
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--model", default="gpt-5.5")
    p.add_argument("--reasoning-effort", default=None, help="override the template's reasoning effort")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--root", type=Path, default=REPO_ROOT)
    p.add_argument("--mujoco-gl", choices=("egl", "osmesa"), default=os.environ.get("MUJOCO_GL", "egl"))
    p.add_argument("--egl-device", type=int, default=None)
    p.add_argument("--endpoint", default=os.environ.get("REACT_ENDPOINT") or default_endpoint(), help="chat: OpenAI-compatible base URL (e.g. vLLM http://host:port/v1); responses: full Responses URL (.../v1/responses)")
    p.add_argument("--auth", choices=("none", "bearer"), default=os.environ.get("REACT_AUTH", "none"),
                   help="bearer: send Authorization: Bearer $CAPTAIN_API_KEY (or $OPENAI_API_KEY)")
    p.add_argument("--api", choices=("responses", "chat"), default=os.environ.get("REACT_API", "responses"),
                   help="responses: OpenAI Responses API (default); chat: OpenAI-compatible /v1/chat/completions (vLLM students)")
    p.add_argument("--profile", default=None, help="driver profile (request template + tool mode) to borrow for models outside the catalog; default gpt-5.5 for --api chat")
    p.add_argument("--runtime-tool-contract", choices=("code-mode-runtime/v2", "python-runtime/v1"), default=None,
                   help="explicit versioned tool contract and private execution audit; omitted preserves captured legacy schemas. "
                        "python-runtime/v1: the only tool is a persistent `python` session with robot pre-imported")
    p.add_argument("--python-cell-timeout", type=float, default=600.0, help="python-runtime/v1: seconds before one python call is interrupted")
    p.add_argument("--replay-prefix", type=Path, default=None,
                   help="responses.jsonl-style file: re-issue these recorded model responses first (their tool calls run again), then the live model")
    p.add_argument("--handoff", choices=("continuation", "clean"), default="continuation",
                   help="with --replay-prefix: continuation keeps the prefix's Python session, workspace and conversation for the "
                        "live model; clean keeps only the physical world - a fresh Python session (--student-python), the "
                        "workspace as it was at the start, a fresh prompt with the current frames, and /history from here on")
    p.add_argument("--student-python", type=Path, default=None,
                   help="python-runtime/v1: the pinned interpreter of the model's own session (fresh episodes: from the start; "
                        "clean handoff: after it). Verified in the real sandbox before the episode; a failure is an environment error")
    p.add_argument("--sandbox-hide", type=Path, action="append", default=[],
                   help="directories the model's sandbox must not see (e.g. the source episodes of a replayed prefix)")
    p.add_argument("--mark-requests", action="store_true",
                   help="sample the host-only goal state before each request's tools run (hidden_request_marks.jsonl, for per-turn rewards)")
    p.add_argument("--reasoning-summary", choices=("auto", "concise", "detailed"), default=None,
                   help="responses: request reasoning summaries (body.reasoning.summary) so exported data carries the model's reasoning text")
    p.add_argument("--small-context-window", type=int, default=None,
                   help="serve a small-context model: catalog.small_context_profile(profile, context_window=N) - compaction at N - max(10%%, one turn), "
                        "keep N/4 of recent turns verbatim, 4000-token tool outputs spilled to files, strategy window (65536 for Qwen3.5 students; "
                        "the explicit --context-* / --keep-recent-tokens / --tool-output-tokens flags still override individual fields)")
    p.add_argument("--max-output-tokens", type=int, default=16384, help="chat: max_tokens per request (lowered automatically when the server reports the prompt+completion budget)")
    p.add_argument("--temperature", type=float, default=None, help="chat: sampling temperature (server default when omitted)")
    p.add_argument("--top-p", type=float, default=None, help="chat: top_p (server default when omitted)")
    p.add_argument("--enable-thinking", choices=("server", "on", "off"), default="server", help="chat: chat_template_kwargs.enable_thinking")
    p.add_argument("--max-images", type=int, default=16, help="chat: keep only the N most recent images in the replayed history (vLLM --limit-mm-per-prompt)")
    p.add_argument("--no-history-reasoning", action="store_true", help="chat: do not replay earlier reasoning_content (saves context; Qwen convention is to keep it within one agentic turn)")
    p.add_argument("--all-tools", action="store_true", help="chat: expose every tool of the template, not only the ones the driver implements")
    p.add_argument("--hoist-tool-images", action="store_true", help="chat: move images returned by tools into a user message after the tool messages (servers that only take text in tool messages)")
    p.add_argument("--incomplete-nudges", type=int, default=2, help="chat: how many times a length-truncated reply without a tool call gets a user nudge instead of ending the episode")
    p.add_argument("--stop-when-finished", choices=("auto", "on", "off"), default="auto",
                   help="stop the loop once the environment reports the episode finished (auto: on for --api chat, off for responses so reference rows keep Codex's behaviour)")
    p.add_argument("--api-key-env", default="REACT_API_KEY", help="chat: environment variable holding a bearer token (optional)")
    p.add_argument("--inline-task", action="store_true", help="inline TASK.md as a developer message and attach the initial camera frames to the prompt")
    p.add_argument("--wall-time", type=float, default=2400.0)
    p.add_argument("--max-actions", type=int, default=150)
    p.add_argument("--max-tool-calls", type=int, default=600)
    p.add_argument("--grace-s", type=float, default=90.0, help="extra seconds the loop may run past the wall budget")
    p.add_argument("--max-requests", type=int, default=None, help="debug cap on model requests")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--timezone", default="Etc/UTC")
    p.add_argument("--sam3-url", default=os.environ.get("SAM3_URL"))
    p.add_argument("--molmo-url", default=os.environ.get("MOLMO_URL"))
    p.add_argument("--server-arg", action="append", default=[], help="extra argument passed to env_server (repeatable)")
    p.add_argument("--libero-variant", choices=("libero", "pro", "plus"), default="libero",
                   help="benchmark fork for the env server: LIBERO-PRO (pro) or LIBERO-plus (plus); suites like libero_goal_task")
    p.add_argument("--no-sandbox", action="store_true", help="debug only: run tools without bwrap")
    p.add_argument("--no-compaction", action="store_true", help="stop with context_exhausted instead of compacting at the model's limit")
    p.add_argument("--context-limit", type=int, default=None, help="override the profile's auto-compact token limit (e.g. for small-context models)")
    p.add_argument("--context-limit-percent", type=float, default=None, help="auto-compact at this percent of the profile's context window (Codex default 90; 95 is the ceiling)")
    p.add_argument("--context-hard-limit", type=int, default=None, help="never let a request exceed this many real tokens (e.g. the training sample cap); the compaction trigger is derived as hard_limit - one turn")
    p.add_argument("--context-strategy", choices=("summary", "notes", "window"), default=None, help="drop notice with an incremental LLM summary (summary) or facts only (window)")
    p.add_argument("--keep-recent-tokens", type=int, default=None, help="budget of recent whole turns kept verbatim after a compaction")
    p.add_argument("--tool-output-tokens", type=int, default=None, help="per-tool-output token cap (Codex 10000)")
    p.add_argument("--spill-tool-output", action="store_true", help="save truncated tool outputs to <workspace>/.tool_output/ and tell the model")
    p.add_argument("--dry-run", action="store_true", help="start the server and build the workspace, then finish without a model")
    args = p.parse_args()

    out = args.output_dir.resolve()
    if (out / "episode_result.json").is_file():
        print(json.dumps(json.loads((out / "episode_result.json").read_text()), sort_keys=True))
        return 0
    out.mkdir(parents=True, exist_ok=True)
    (out / "workspace" / "images").mkdir(parents=True, exist_ok=True)
    started_wall = time.time()
    t0 = time.monotonic()
    if args.profile:
        profile = profile_for(args.profile)
    else:
        profile = profile_for(args.model, fallback="captain" if args.api == "chat" else None)
    if args.small_context_window:
        profile = small_context_profile(profile, context_window=int(args.small_context_window), keep_recent_tokens=args.keep_recent_tokens,
                                        truncation_tokens=args.tool_output_tokens or 4_000, strategy=args.context_strategy or "window")
    environment_preflight = None
    python_mode = args.runtime_tool_contract == "python-runtime/v1"
    if args.runtime_tool_contract is not None:
        if profile.tool_mode != "code_mode" and not python_mode:
            raise ValueError("--runtime-tool-contract requires a code-mode profile")
        receipt_path = os.environ.get("FRESH_ENV_PREFLIGHT_PATH")
        if not receipt_path:
            raise ValueError("runtime tool contract requires a verified host environment preflight")
        from captain_harness.react.toolchain_probe import validate_preflight_receipt

        environment_preflight = validate_preflight_receipt(json.loads(Path(receipt_path).read_text(encoding="utf-8")),
                                                           contract=args.runtime_tool_contract)
    if args.replay_prefix is not None and not python_mode:
        raise ValueError("--replay-prefix is supported with --runtime-tool-contract python-runtime/v1 only")
    if python_mode and args.prompt == DEFAULT_PROMPT:
        args.prompt = PYTHON_PROMPT
    result: dict = {"suite": args.suite, "task_id": args.task_id, "seed": args.seed, "model": args.model,
                    "driver_profile": profile.slug, "tool_mode": profile.tool_mode, "output_dir": str(out),
                    "started_wall_time": started_wall, "reasoning_effort": args.reasoning_effort or "template",
                    "api": args.api, "context_window": profile.context_window, "inline_task": bool(args.inline_task),
                    "runtime_tool_contract": args.runtime_tool_contract, "replay_prefix": str(args.replay_prefix) if args.replay_prefix else None,
                    "libero_variant": args.libero_variant}
    if environment_preflight is not None:
        result["environment_preflight"] = environment_preflight
    termination = "infrastructure_error"
    server = start_server(args.root, out, suite=args.suite, task_id=args.task_id, seed=args.seed,
                          images_dir=out / "workspace" / "images", max_actions=args.max_actions,
                          max_tool_calls=args.max_tool_calls, wall_time_s=args.wall_time, mujoco_gl=args.mujoco_gl,
                          egl_device=args.egl_device, sam3_url=args.sam3_url, molmo_url=args.molmo_url,
                          extra_args=list(args.server_arg) + (["--receipts"] if python_mode else []),
                          libero_variant=args.libero_variant)
    url = None
    sessions = None
    repl = None
    repls: dict = {}
    try:
        info = wait_for_server(out, 600.0, server)
        url = info["url"]
        manifest = http_json(url + "/manifest")
        workspace = build_workspace(out, info, manifest, wall_time_s=args.wall_time, max_actions=args.max_actions, max_tool_calls=args.max_tool_calls,
                                    runtime="python" if python_mode else "shell")
        result.update({"task": info["task"], "server_url": url, "server_pid": server.pid, "init_state_index": info.get("init_state_index")})
        if args.dry_run:
            http_json(url + "/finish", "POST", {"note": "dry run", "source": "driver"})
            termination = "dry_run"
        else:
            if not args.endpoint or (args.api == "chat" and args.endpoint == default_endpoint()):
                raise RuntimeError("no model endpoint: pass --endpoint (chat: http://host:port/v1) or set REACT_ENDPOINT")
            sandbox = Sandbox.detect(workspace, hidden_dirs=[out, *args.sandbox_hide], require=not args.no_sandbox)
            if args.no_sandbox:
                sandbox = Sandbox(workspace=workspace.resolve(), hidden_dirs=(), bwrap=None)
            env = child_environment(os.environ, {"ROBOT_SERVER_URL": url})
            from dataclasses import replace as _replace
            if args.context_limit or args.context_limit_percent:
                limit = int(args.context_limit) if args.context_limit else int(profile.context_window * args.context_limit_percent / 100.0)
                profile = _replace(profile, auto_compact_token_limit=limit)
            overrides = {}
            if args.context_strategy:
                overrides["context_strategy"] = args.context_strategy
            if args.keep_recent_tokens:
                overrides["keep_recent_tokens"] = int(args.keep_recent_tokens)
            if args.tool_output_tokens:
                overrides["truncation_tokens"] = int(args.tool_output_tokens)
            if args.spill_tool_output:
                overrides["spill_tool_output"] = True
            if overrides:
                profile = _replace(profile, **overrides)
            if args.context_hard_limit:
                from captain_harness.react.catalog import trigger_for_hard_limit
                profile = trigger_for_hard_limit(profile, int(args.context_hard_limit))
            result["context_policy"] = {"context_window": profile.context_window, "context_limit": profile.context_limit, "hard_limit": args.context_hard_limit, "strategy": profile.context_strategy,
                                        "keep_recent_tokens": profile.keep_recent_tokens, "tool_output_tokens": profile.truncation_tokens,
                                        "spill_tool_output": profile.spill_tool_output, "small_context_window": args.small_context_window,
                                        "max_images": args.max_images if args.api == "chat" else None}
            sessions = SessionManager(sandbox=sandbox, env=env, default_cwd=workspace, output_limit_tokens=profile.truncation_tokens,
                                      spill_dir=(workspace / ".tool_output") if profile.spill_tool_output else None)
            router = ToolRouter(sessions=sessions, workspace=workspace, apply_patch=lambda patch: apply_patch(patch, workspace))
            code_runner = None
            python_runner = None
            deadline = t0 + args.wall_time + args.grace_s
            student_python = str(args.student_python) if args.student_python else None
            if python_mode:
                python3 = ((environment_preflight or {}).get("executables") or {}).get("python3", {}).get("resolved")
                if student_python:
                    probe = probe_student_runtime(sandbox, env, student_python, startup=python_session.OFFLINE_PROBE_STARTUP,
                                                  cells=python_session.STUDENT_RUNTIME_CELLS)
                    result["student_runtime"] = {
                        "python": student_python, "ok": probe["ok"], "versions": probe.get("versions"), "error": probe.get("error"),
                        "failed": ([] if probe["worker"].get("ok") else ["worker"]) + [c["name"] for c in probe["cells"] if not c["ok"]]}
                    if not probe["ok"]:
                        router.environment_errors.append({"schema_version": "harness-environment-error/v1", "code": "student_runtime_unavailable",
                                                          "executable": student_python, "source": "tool_runtime", "verified": True,
                                                          "details": result["student_runtime"]})
                        raise RuntimeError(f"student python failed its sandbox check: {result['student_runtime']}")
                # a replayed prefix runs in the runtime it was recorded with; the model's own session uses the pinned one
                session_python = student_python if (student_python and args.replay_prefix is None) else python3
                result["session_python"] = session_python
                repl = PythonRepl(sandbox=sandbox, env=env, python=session_python, startup=python_session.STARTUP)
                repls["current"] = repl
                try:
                    repl.start()
                except ReplStartError as exc:
                    router.environment_errors.append({"schema_version": "harness-environment-error/v1", "code": "python_session_unavailable",
                                                      "executable": "python3", "source": "tool_runtime", "verified": True,
                                                      "details": {"error": str(exc)[-2000:]}})
                    raise
                router.closers.append(repl.close)
                spill_dir = (workspace / ".tool_output") if profile.spill_tool_output else None
                python_runner = lambda source: run_python_cell(  # noqa: E731
                    source, router=router, repl=repls["current"], timeout_s=args.python_cell_timeout, output_limit_tokens=profile.truncation_tokens,
                    spill_dir=spill_dir, deadline_monotonic=deadline)
            elif profile.tool_mode == "code_mode":
                code_runner = lambda source: run_exec_script(source, router=router, sandbox=sandbox, env=env, output_limit_tokens=profile.truncation_tokens)  # noqa: E731
            driver_dir = out / "driver"
            driver_dir.mkdir(exist_ok=True)
            provider_log = (driver_dir / "provider_events.jsonl").open("a", encoding="utf-8")

            def sink(event):
                provider_log.write(json.dumps({"t": time.time(), **event}) + "\n")
                provider_log.flush()

            if args.api == "chat":
                client = ChatCompletionsClient(
                    args.endpoint, api_key=os.environ.get(args.api_key_env) or None, max_output_tokens=args.max_output_tokens,
                    temperature=args.temperature, top_p=args.top_p,
                    enable_thinking=None if args.enable_thinking == "server" else (args.enable_thinking == "on"),
                    max_images=args.max_images, history_reasoning=not args.no_history_reasoning, all_tools=args.all_tools,
                    hoist_tool_images=args.hoist_tool_images, event_sink=sink)
            else:
                client = ResponsesClient(args.endpoint, auth=args.auth, event_sink=sink)
            if args.replay_prefix is not None:
                if profile.context_strategy not in ("window",) and not args.no_compaction:
                    raise ValueError("--replay-prefix needs --context-strategy window (a summary strategy would call the model mid-replay)")
                client = ReplayClient(client, load_responses(args.replay_prefix))
                result["replayed_requests"] = len(client.responses)
            body, items = render_request(profile, model=args.model, workspace=workspace, prompt=args.prompt, timezone=args.timezone)
            if args.reasoning_effort:
                body.setdefault("reasoning", {})["effort"] = args.reasoning_effort
            if args.reasoning_summary:
                body.setdefault("reasoning", {})["summary"] = args.reasoning_summary
            if args.inline_task:
                items = inline_task_items(items, (workspace / "TASK.md").read_text(encoding="utf-8"), initial_image_files(workspace))
            if args.api == "chat":
                # a chat server's parser emits whatever tool name the model writes; only the tools the
                # request exposed may run (a Responses API would never deliver a call to an undeclared tool)
                exposed, _custom = convert_tools(body, items, allow=None if args.all_tools else IMPLEMENTED_TOOLS)
                router.allowed_tools = {"python"} if python_mode else {tool["function"]["name"] for tool in exposed}
            episode_id = f"{args.suite}-{args.task_id}-s{args.seed}-{uuid.uuid4().hex[:12]}"
            before_tools = None
            if args.mark_requests:
                token = info.get("driver_token")
                if not token:
                    raise RuntimeError("--mark-requests needs an env server that issues a driver token")
                before_tools = lambda index: http_json(url + "/mark", "POST", {"request_index": index, "token": token}, timeout=120)  # noqa: E731
            loop = Loop(client=client, body=body, history=items, router=router, profile=profile, output_dir=driver_dir,
                        prompt_cache_key=episode_id, code_mode_runner=code_runner, deadline_monotonic=deadline, max_requests=args.max_requests,
                        compaction=not args.no_compaction, workspace=workspace,
                        runtime_tool_contract=args.runtime_tool_contract, environment_preflight=environment_preflight,
                        python_runner=python_runner, before_tools=before_tools,
                        tool_description=python_session.tool_description(args.python_cell_timeout) if python_mode else None,
                        incomplete_nudges=args.incomplete_nudges if args.api == "chat" else 0, incomplete_nudge_text=INCOMPLETE_NUDGE_TEXT)
            if args.handoff == "clean":
                if args.replay_prefix is None or not python_mode or not info.get("driver_token"):
                    raise ValueError("--handoff clean needs --replay-prefix, python-runtime/v1 and an env server that issues a driver token")
                prefix_requests = len(client.responses)
                initial_items = list(loop._initial_items)
                workspace_snapshot = snapshot_workspace(workspace)

                def handoff(loop_: Loop, index: int) -> None:
                    """Before the first live request: keep the world, replace the program state, workspace and conversation."""
                    if "handoff" in result or index != prefix_requests + 1:
                        return
                    reply = http_json(url + "/handoff", "POST", {"token": info["driver_token"]}, timeout=300)
                    if not reply.get("ok"):
                        raise RuntimeError(f"env server refused the handoff: {reply}")
                    repls["current"].close()
                    fresh = PythonRepl(sandbox=sandbox, env=env, python=student_python or python3, startup=python_session.STARTUP)
                    fresh.start()
                    repls["current"] = fresh
                    router.closers.append(fresh.close)
                    frames = [Path(image["path"]) for image in reply["images"]]
                    changes = restore_workspace(workspace, workspace_snapshot,
                                                keep=tuple(str(f.resolve().relative_to(workspace.resolve())) for f in frames))
                    left = args.max_requests - prefix_requests if args.max_requests else "unlimited"
                    loop_.reset_history(initial_items + [python_session.handoff_message(frames, requests=left, budget=reply["budget"])])
                    result["handoff"] = {"mode": "clean", "at_request": index, "prefix_requests": prefix_requests, "call_index": reply["call_index"],
                                         "python": student_python or python3, "frames": [f.name for f in frames], "workspace": changes,
                                         "requests_left": left, "budget": reply["budget"]}
                    loop_.events.write({"type": "handoff", **result["handoff"]})

                loop.before_request = handoff
            if args.stop_when_finished == "on" or (args.stop_when_finished == "auto" and args.api == "chat"):
                health_cache: dict = {"t": 0.0, "finished": False}

                def environment_finished() -> bool:
                    now = time.monotonic()
                    if not health_cache["finished"] and now - health_cache["t"] >= 2.0:
                        health_cache["t"] = now
                        try:
                            health_cache["finished"] = bool(http_json(url + "/health").get("finished"))
                        except Exception:  # noqa: BLE001 - a health hiccup must not stop the loop
                            pass
                    return health_cache["finished"]

                loop.stop_when = environment_finished
            holder: dict = {}

            def run_loop():
                try:
                    holder["result"] = loop.run()
                except Exception as exc:  # noqa: BLE001
                    holder["error"] = f"{type(exc).__name__}: {exc}"

            thread = threading.Thread(target=run_loop, daemon=True)
            thread.start()
            thread.join(timeout=max(1.0, deadline - time.monotonic()))
            loop_wall = time.monotonic() - t0
            killed = thread.is_alive()
            if killed:
                # the loop is still blocked (typically in a model request) at the deadline and will never reach its finally
                # in time: close the audit here; the footer is complete only if no tool call is still running
                router.finish_audit(complete=True)
            loop_result = holder.get("result")
            health = http_json(url + "/health")
            if health.get("finished"):
                termination = "agent_finished"
            elif killed or health.get("expired") or (time.monotonic() - t0) >= args.wall_time:
                termination = "wall_time_exhausted"
                http_json(url + "/finish", "POST", {"note": "driver: wall budget exhausted", "source": "driver_timeout"})
            elif loop_result is not None and loop_result.termination == "completed":
                termination = "agent_stopped_without_finish"
                http_json(url + "/finish", "POST", {"note": "driver: model ended the turn without finish()", "source": "driver_exit"})
            elif loop_result is not None and loop_result.termination == "context_exhausted":
                termination = "context_exhausted"
                http_json(url + "/finish", "POST", {"note": "driver: context window exhausted", "source": "driver_exit"})
            elif loop_result is not None and loop_result.termination == "request_cap":
                # a policy outcome (the model used its request budget), and the normal end of a segment rollout
                termination = "request_cap_exhausted"
                http_json(url + "/finish", "POST", {"note": f"driver: {loop_result.reason}", "source": "driver_exit"})
            else:
                termination = "infrastructure_error"
                result["error"] = holder.get("error") or (loop_result.reason if loop_result else "loop ended without result")
                http_json(url + "/finish", "POST", {"note": f"driver: {result['error']}", "source": "driver_exit"})
            provider_log.close()
            driver_block = {
                "requests": loop_result.requests if loop_result else None,
                "tool_calls": loop_result.tool_calls if loop_result else None,
                "usage": (loop_result.usage_total if loop_result else dict(loop.usage_total)),
                "loop_termination": loop_result.termination if loop_result else ("killed_at_deadline" if killed else "error"),
                "loop_reason": loop_result.reason if loop_result else holder.get("error"),
                "final_messages": [loop_result.final_message] if loop_result and loop_result.final_message else [],
                "commands": router.commands,
                "view_image_calls": router.view_image_calls,
                "rollout": {"view_image_calls": router.view_image_calls, "reasoning_items": loop_result.reasoning_items if loop_result else None,
                            "assistant_messages": loop_result.assistant_messages if loop_result else None,
                            "compactions": loop_result.compactions if loop_result else loop.compactions},
                "max_context_tokens": loop_result.max_context_tokens if loop_result else None,
                "killed_at_deadline": killed,
                "wall_s": round(loop_wall, 1),
                "sandbox": sandbox.enabled,
                "endpoint": args.endpoint,
                "auth": args.auth if args.api == "responses" else "none",
                "api": args.api,
                "chat": None if args.api != "chat" else {
                    "max_output_tokens": args.max_output_tokens, "temperature": args.temperature, "top_p": args.top_p,
                    "enable_thinking": args.enable_thinking, "max_images": args.max_images,
                    "history_reasoning": not args.no_history_reasoning, "all_tools": args.all_tools,
                    "hoist_tool_images": args.hoist_tool_images, "incomplete_nudges": args.incomplete_nudges},
                "stop_when_finished": loop.stop_when is not None,
            }
            usage = driver_block["usage"]
            if loop.tool_contract is not None:
                driver_block["tool_contract"] = loop.tool_contract
            usage["total_tokens"] = int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))
            result["driver"] = driver_block
            result["codex"] = driver_block  # compatibility with Codex-reference summarizers
            if router.environment_errors:
                result["environment_error"] = router.environment_errors[0]
                result["environment_errors"] = list(router.environment_errors)
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        termination = "infrastructure_error"
    finally:
        for session in {id(r): r for r in (repl, repls.get("current")) if r is not None}.values():
            session.close()
        if sessions is not None:
            sessions.close_all()
        stop_server(server, url)
    server_result = {}
    if (out / "result.json").is_file():
        server_result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    success = server_result.get("host_final_success")
    agent_finished = server_result.get("finish_source") == "agent"
    result.update({
        "termination_reason": termination,
        "host_final_success": success,
        "host_success_ever": server_result.get("host_success_ever"),
        "first_success_step": server_result.get("first_success_step"),
        "agent_finished": agent_finished,
        "false_finish": bool(agent_finished and success is False),
        "missed_finish": bool((not agent_finished) and success is True),
        "action_requests": server_result.get("action_requests"),
        "executed_actions": server_result.get("executed_actions"),
        "failed_actions": server_result.get("failed_actions"),
        "tool_requests": server_result.get("tool_requests"),
        "observation_exports": server_result.get("observation_exports"),
        "gate_rejections": server_result.get("gate_rejections"),
        "server_wall_time_s": server_result.get("wall_time_s"),
        "wall_time_s": round(time.monotonic() - t0, 1),
        "finish_note": server_result.get("finish_note"),
        "transcript": str(out / "driver" / "transcript.jsonl"),
    })
    if not (server_result or args.dry_run) and termination != "infrastructure_error":
        result["termination_reason"] = "infrastructure_error"
    if (out / "libero_variant.json").is_file():  # fork task identity (instruction, init override) for the result row
        try:
            result["libero_variant_info"] = {k: v for k, v in json.loads((out / "libero_variant.json").read_text(encoding="utf-8")).items()
                                             if k != "goal_state"}
        except (OSError, json.JSONDecodeError):
            pass
    (out / "episode_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: result.get(k) for k in ("suite", "task_id", "model", "termination_reason", "host_final_success",
                                                  "agent_finished", "gate_rejections", "action_requests", "tool_requests", "wall_time_s")}, sort_keys=True))
    return 0 if result["termination_reason"] != "infrastructure_error" else 2


if __name__ == "__main__":
    sys.exit(main())
