"""Export ReAct episodes (driver/views.jsonl + items.jsonl + responses.jsonl) as CaPtain SFT rows (sft/data/FORMAT.md):
one row per model request = exactly the view the model saw (translated with the harness's own chat translation and
image cap, as the student is served) + the teacher's reply as the supervised last assistant turn.

    python -m soarm.export_sft --episodes 'runs/*/soarm_sim_*' --out data/sft_v0 [--include-failures]

Writes <out>/rows.jsonl (+ rows_fail.jsonl), <out>/images/<episode>__<file>.png and <out>/episodes.jsonl (one summary per
episode). Pack with sft/data/pack_sft_prefix.py afterwards.
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path

from captain_harness.react.chat_llm import cap_images, convert_tools, responses_to_chat_messages

PLACEHOLDER = "http://soarm-image.invalid/{}"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def _swap_images(items: list[dict], paths: list[str]) -> list[dict]:
    """Image parts carry a stripped data URL + `path`; give each a unique valid URL so the chat translation keeps it."""
    out = []
    for item in items:
        item = dict(item)
        for key in ("content", "output"):
            parts = item.get(key)
            if not isinstance(parts, list):
                continue
            new = []
            for part in parts:
                if isinstance(part, dict) and part.get("type") == "input_image":
                    if not part.get("path"):
                        new.append({"type": "input_text", "text": "[image unavailable]"})
                        continue
                    paths.append(part["path"])
                    part = {**part, "image_url": PLACEHOLDER.format(len(paths) - 1)}
                new.append(part)
            item[key] = new
        out.append(item)
    return out


def _first_finish_request(responses: dict[int, dict]) -> int | None:
    """Request index of the first python cell that calls finish(); None if the episode never declares completion."""
    for k in sorted(responses):
        for item in responses[k].get("items") or []:
            if isinstance(item, dict) and "finish(" in str(item.get("input") or ""):
                return k
    return None


def _relocate(path: Path, ep: Path) -> Path:
    """Image paths as recorded on the collecting host (absolute, or relative to the agent workspace) -> the same file
    under this copy of the episode dir."""
    if not path.is_absolute():
        return ep / "workspace" / path
    if path.exists() or ep.name not in path.parts:
        return path
    parts = path.parts
    return ep.joinpath(*parts[len(parts) - parts[::-1].index(ep.name):])


def episode_rows(ep: Path, image_dir: Path, *, max_images: int = 12) -> tuple[list[dict], dict]:
    drv = ep / "driver"
    result = json.loads((ep / "episode_result.json").read_text())
    registry = {r["id"]: r["item"] for r in _jsonl(drv / "items.jsonl")}
    views = {v["request"]: v for v in _jsonl(drv / "views.jsonl")}
    responses = {r["request"]: r for r in _jsonl(drv / "responses.jsonl")}
    task = json.loads((ep / "soarm" / "task.json").read_text()) if (ep / "soarm" / "task.json").exists() else {}
    task["seed"] = int(ep.name.rsplit("seed-", 1)[1])      # episode (layout) seed; task.json holds the sampler sub-seed
    perc_path = ep / "soarm" / "perception.json"            # absent = episodes before SOARM_PERCEPTION (privileged)
    perc = json.loads(perc_path.read_text()) if perc_path.exists() else {"mode": "privileged"}
    perception = perc.get("mode")
    depth = perc.get("depth", "sim_sensor")                  # absent = episodes before SOARM_DEPTH (rendered depth)
    name = ep.name if not ep.parent.name else f"{ep.parent.name}__{ep.name}"
    rows = []
    tools = None
    for k in sorted(views):
        if k not in responses or responses[k].get("status") not in ("completed", None):
            continue
        ctx = [registry[i] for i in views[k]["items"]]
        reply = [it for it in responses[k]["items"] if isinstance(it, dict)]
        if not any(it.get("type") in ("custom_tool_call", "function_call") for it in reply):
            continue                                   # supervise acting turns only (a text-only reply ends the episode)
        paths: list[str] = []
        items = _swap_images(ctx, paths) + _swap_images(reply, paths)
        if tools is None:
            tools, _ = convert_tools({"input": ctx}, ctx, allow=None)
            tools = [t for t in tools if t["function"]["name"] == "python"]
        messages = responses_to_chat_messages(None, items, turn_groups=[(0, len(ctx)), (len(ctx), len(reply))])
        cap_images(messages, max_images)
        images = []
        for m in messages:
            if isinstance(m.get("content"), list):
                for i, part in enumerate(m["content"]):
                    if part.get("type") == "image_url":
                        src = _relocate(Path(paths[int(part["image_url"]["url"].rsplit("/", 1)[1])]), ep)
                        rel = f"images/{name}__{src.name}"
                        if not (image_dir.parent / rel).exists():
                            shutil.copyfile(src, image_dir.parent / rel)
                        m["content"][i] = {"type": "image", "image": rel}
                        images.append(rel)
            for call in m.get("tool_calls") or []:
                args = call["function"].get("arguments")
                if isinstance(args, str):
                    try:
                        call["function"]["arguments"] = json.loads(args)
                    except json.JSONDecodeError:
                        call["function"]["arguments"] = {"input": args}
        turns = sum(1 for m in messages if m["role"] == "assistant")
        usage = (responses[k].get("usage") or {})
        rows.append({"id": f"{name}#req-{k:03d}", "images": images, "tools": tools, "messages": messages,
                     "meta": {"episode": name, "request_index": k, "turn_index": turns - 1, "suite": "soarm_sim",
                              "family": task.get("family"), "seed": task.get("seed"), "source": "agent",
                              "perception": perception, "depth": depth,
                              "teacher": result.get("model"), "teacher_usage": usage,
                              "host_final_success": result.get("host_final_success"),
                              "agent_finished": result.get("agent_finished")}})
    summary = {"episode": name, "family": task.get("family"), "seed": task.get("seed"), "model": result.get("model"),
               "perception": perception, "depth": depth, "depth_revision": perc.get("depth_revision"),
               "host_final_success": result.get("host_final_success"), "agent_finished": result.get("agent_finished"),
               "false_finish": result.get("false_finish"), "termination": result.get("termination_reason"),
               "requests": len(views), "rows": len(rows)}
    # LIBERO r7 rule (docs/TRAINING.md): requests after a gate-rejected first finish are not supervised.
    gate_rejections = int(result.get("gate_rejections") or 0)
    cut = _first_finish_request(responses) if gate_rejections else None
    summary["gate_rejections"] = gate_rejections
    summary["gate_rejected_finish"] = cut is not None
    if cut is not None:
        before = len(rows)
        rows = [r for r in rows if r["meta"]["request_index"] <= cut]
        summary["rows_dropped_after_rejected_finish"] = before - len(rows)
        summary["rows"] = len(rows)
    return rows, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="+", required=True, help="episode dirs or globs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-images", type=int, default=12)
    ap.add_argument("--exclude-families", nargs="*", default=None,
                    help="families never exported (default: soarm.splits.HELD_OUT_FAMILIES)")
    a = ap.parse_args()
    from .splits import HELD_OUT_FAMILIES
    excluded = set(HELD_OUT_FAMILIES if a.exclude_families is None else a.exclude_families)
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    eps = sorted({Path(p) for pat in a.episodes for p in glob.glob(pat) if (Path(p) / "episode_result.json").exists()})
    with (out / "rows.jsonl").open("w") as ok_fh, (out / "rows_fail.jsonl").open("w") as fail_fh, \
            (out / "episodes.jsonl").open("w") as ep_fh:
        for ep in eps:
            task_path = ep / "soarm" / "task.json"
            family = json.loads(task_path.read_text()).get("family") if task_path.exists() else None
            if family in excluded:
                ep_fh.write(json.dumps({"episode": str(ep), "family": family, "excluded": "held-out family"}) + "\n")
                continue
            try:
                rows, summary = episode_rows(ep, out / "images", max_images=a.max_images)
            except Exception as exc:  # noqa: BLE001
                ep_fh.write(json.dumps({"episode": str(ep), "export_error": f"{type(exc).__name__}: {exc}"}) + "\n")
                continue
            good = bool(summary["host_final_success"]) and bool(summary["agent_finished"]) and not summary["false_finish"]
            summary["kept"] = good
            ep_fh.write(json.dumps(summary) + "\n")
            fh = ok_fh if good else fail_fh
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"exported {len(eps)} episodes -> {out}")


if __name__ == "__main__":
    main()
