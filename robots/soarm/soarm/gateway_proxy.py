"""Local OpenAI chat/completions endpoint that forwards to an OpenAI **Responses** API (the gateway used answers
/v1/responses reliably, /v1/chat/completions not). The ReAct driver runs unchanged with `--api chat`
(the student's request shape); this proxy translates each request and maps the reply back, including reasoning
(Responses `reasoning` items -> `message.reasoning_content`, kept by the driver as reasoning items for SFT).

    python -m soarm.gateway_proxy --port 8650 --env-file ~/.soarm_gateway.env --log usage.jsonl [--concurrency 3]

Data learning (`--data-learning`, `soarm/data_learning.py`): the proxy can make the teacher write its working out in a
visible `log_reasoning_step` call before it acts and fold those steps into `reasoning_content` (the channel the SFT
exporter supervises as the `<think>` block); `--response-image-cap N` sends only the N most recent history images,
matching the chat backend's cap.

Secrets: the key is read from the env file (TEACHER_API_KEY / TEACHER_BASE_URL, or DEEPSEEK_API_KEY /
DEEPSEEK_BASE_URL with --provider deepseek; OPENAI_BASE_URL / OPENAI_API_KEY are accepted as gateway aliases); it is
never logged. `--provider deepseek` forwards chat/completions to
the official DeepSeek API unchanged (it returns reasoning_content) and enforces a spend cap from GET /user/balance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .data_learning import DataLearningClient, cap_response_images


def read_env(path: str) -> dict[str, str]:
    out = {}
    for line in Path(path).expanduser().read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip().removeprefix("export ").strip()] = v.strip().strip('"').strip("'")
    return out


def gateway_secrets(env: dict[str, str]) -> tuple[str, str]:
    """(base url, api key) from the gateway env file; the OPENAI_* names are accepted as aliases."""
    base = env.get("TEACHER_BASE_URL") or env.get("OPENAI_BASE_URL")
    key = env.get("TEACHER_API_KEY") or env.get("OPENAI_API_KEY")
    if not base or not key:
        raise KeyError("gateway env file needs TEACHER_BASE_URL and TEACHER_API_KEY "
                       "(OPENAI_BASE_URL / OPENAI_API_KEY are accepted as aliases)")
    return base.rstrip("/"), key


def _text_parts(content) -> list[str]:
    if isinstance(content, str):
        return [content]
    return [p.get("text", "") for p in content or [] if isinstance(p, dict) and p.get("type") == "text"]


def _image_urls(content) -> list[str]:
    if isinstance(content, str):
        return []
    return [p["image_url"]["url"] for p in content or [] if isinstance(p, dict) and p.get("type") == "image_url"]


def chat_to_responses(req: dict[str, Any], *, reasoning: dict | None, max_output_tokens: int | None) -> dict[str, Any]:
    """chat/completions request -> Responses request. Tool-message images are re-sent as a user message right after
    the tool outputs (function_call_output carries text only)."""
    items: list[dict[str, Any]] = []
    pending_images: list[str] = []

    def flush_images():
        if pending_images:
            items.append({"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "[images returned by the tool call above]"},
                *({"type": "input_image", "image_url": u} for u in pending_images)]})
            pending_images.clear()

    for m in req.get("messages", []):
        role = m.get("role")
        if role != "tool":
            flush_images()
        if role in ("system", "developer"):
            items.append({"type": "message", "role": "developer",
                          "content": [{"type": "input_text", "text": t} for t in _text_parts(m.get("content")) if t]})
        elif role == "user":
            parts = [{"type": "input_text", "text": t} for t in _text_parts(m.get("content")) if t]
            parts += [{"type": "input_image", "image_url": u} for u in _image_urls(m.get("content"))]
            items.append({"type": "message", "role": "user", "content": parts or [{"type": "input_text", "text": ""}]})
        elif role == "assistant":
            text = "\n".join(_text_parts(m.get("content")))
            if text.strip():
                items.append({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]})
            for call in m.get("tool_calls") or []:
                args = call["function"].get("arguments")
                items.append({"type": "function_call", "call_id": call["id"], "name": call["function"]["name"],
                              "arguments": args if isinstance(args, str) else json.dumps(args)})
        elif role == "tool":
            items.append({"type": "function_call_output", "call_id": m["tool_call_id"],
                          "output": "\n".join(_text_parts(m.get("content")))})
            pending_images.extend(_image_urls(m.get("content")))
    flush_images()
    out: dict[str, Any] = {"model": req["model"], "input": items, "store": False}
    tools = [{"type": "function", "name": t["function"]["name"], "description": t["function"].get("description", ""),
              "parameters": t["function"].get("parameters", {"type": "object"})}
             for t in req.get("tools") or [] if t.get("type") == "function"]
    if tools:
        out["tools"] = tools
        out["tool_choice"] = req.get("tool_choice", "auto")
        out["parallel_tool_calls"] = bool(req.get("parallel_tool_calls", False))
    if max_output_tokens:
        out["max_output_tokens"] = int(max_output_tokens)
    if reasoning:
        out["reasoning"] = dict(reasoning)
    return out


def responses_to_chat(resp: dict[str, Any], model: str) -> dict[str, Any]:
    texts, reasoning, calls = [], [], []
    for item in resp.get("output") or []:
        kind = item.get("type")
        if kind == "reasoning":
            for part in (item.get("content") or []) + (item.get("summary") or []):
                if part.get("text"):
                    reasoning.append(part["text"])
        elif kind == "message":
            for part in item.get("content") or []:
                if part.get("type") in ("output_text", "text") and part.get("text"):
                    texts.append(part["text"])
        elif kind == "function_call":
            calls.append({"id": item.get("call_id") or item.get("id"), "type": "function",
                          "function": {"name": item["name"], "arguments": item.get("arguments") or "{}"}})
    message: dict[str, Any] = {"role": "assistant", "content": "\n".join(texts)}
    if reasoning:
        message["reasoning_content"] = "\n\n".join(reasoning)
    if calls:
        message["tool_calls"] = calls
    status = resp.get("status")
    finish = "length" if status == "incomplete" else ("tool_calls" if calls else "stop")
    usage = resp.get("usage") or {}
    return {"id": resp.get("id", ""), "object": "chat.completion", "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": usage.get("input_tokens", 0), "completion_tokens": usage.get("output_tokens", 0),
                      "total_tokens": usage.get("total_tokens", 0),
                      "prompt_tokens_details": {"cached_tokens": (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)},
                      "completion_tokens_details": {"reasoning_tokens": (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0)}}}

# DeepSeek-V4.1-Flash official list prices, USD per 1M tokens (api-docs.deepseek.com/quick_start/pricing, 11-09):
# (cache hit, cache miss, output); peak = 01-04 and 06-10 UTC Mon-Fri costs double. Used as the estimate for the gateway's
# deepseek-v4.1-flash-1017 too [INFERENCE: the gateway does not report per-model cost].
DEEPSEEK_FLASH_USD = (0.003, 0.15, 0.60)


def usage_tokens(usage: dict) -> tuple[int, int, int]:
    """(cache-hit input, cache-miss input, output) from chat-style usage (DeepSeek or translated Responses)."""
    prompt = int(usage.get("prompt_tokens") or 0)
    hit = usage.get("prompt_cache_hit_tokens")
    if hit is None:
        hit = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    return int(hit), max(0, prompt - int(hit)), int(usage.get("completion_tokens") or 0)


def estimate_usd(model: str, usage: dict, t: float) -> float | None:
    if "deepseek" not in model or "flash" not in model:
        return None
    g = time.gmtime(t)
    peak = g.tm_wday < 5 and (1 <= g.tm_hour < 4 or 6 <= g.tm_hour < 10)
    hit, miss, out = usage_tokens(usage)
    p = DEEPSEEK_FLASH_USD
    return (hit * p[0] + miss * p[1] + out * p[2]) / 1e6 * (2.0 if peak else 1.0)


def _dl_log(meta: dict | None) -> dict | None:
    """Data-learning metadata for the usage log (without the per-call response ids)."""
    if not meta:
        return None
    return {k: v for k, v in meta.items() if k != "calls"} | {"calls": len(meta.get("calls") or [])}



class Proxy:
    def __init__(self, a: argparse.Namespace) -> None:
        env = read_env(a.env_file)
        self.provider = a.provider
        if a.provider == "gateway":
            self.base, self.key = gateway_secrets(env)
        else:
            self.base = env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
            self.key = env["DEEPSEEK_API_KEY"]
        self.model_map = dict(kv.split("=", 1) for kv in a.model_map) if a.model_map else {}
        self.reasoning = json.loads(a.reasoning) if a.reasoning else None
        self.max_output_tokens = a.max_output_tokens
        self.timeout = a.timeout
        self.retries = a.retries
        self.sem = threading.BoundedSemaphore(a.concurrency)
        self.log_path = Path(a.log) if a.log else None
        self.lock = threading.Lock()
        self.budget_cny = a.budget_cny
        self.start_balance = self.balance() if a.provider == "deepseek" else None
        self.extra = json.loads(a.extra_body) if a.extra_body else {}
        self.quota_flag = a.quota_flag
        self.daily_cap_usd = a.daily_cap_usd
        self.cost_dir = Path(a.log).parent if a.log else Path(".")
        self.image_cap = a.response_image_cap
        self.data_learning = None
        if a.data_learning != "off":
            if a.provider != "gateway":
                raise ValueError("--data-learning needs --provider gateway: the official DeepSeek API already returns "
                                 "its reasoning_content, and the Responses scratchpad does not exist there")
            if self.image_cap:
                raise ValueError("--response-image-cap needs --provider gateway (chat requests keep images in messages)")
            self.data_learning = DataLearningClient(
                lambda body: self._post(self.base + "/responses", body),
                max_rounds=a.data_learning_rounds, mode=a.data_learning)
        elif self.image_cap and a.provider != "gateway":
            raise ValueError("--response-image-cap needs --provider gateway (chat requests keep images in messages)")

    def _post(self, url: str, body: dict) -> dict:
        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, method="POST", headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())

    def balance(self) -> float:
        req = urllib.request.Request(self.base + "/user/balance", headers={"Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            info = json.loads(r.read().decode())
        return float(sum(float(b.get("total_balance", 0)) for b in info.get("balance_infos", []) if b.get("currency") == "CNY"))

    def log(self, row: dict) -> None:
        if self.log_path:
            with self.lock, self.log_path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")

    # ------------------------------------------------------------------ cost (cap the gateway use per day)
    def day_file(self) -> Path:
        day = time.strftime("%Y%m%d", time.gmtime(time.time() + 8 * 3600))       # the gateway quota resets 00:00 Beijing
        return Path(self.cost_dir) / f"gateway_cost_{day}.json"

    def spent_today(self) -> float:
        f = self.day_file()
        return float(json.loads(f.read_text()).get("usd", 0.0)) if f.exists() else 0.0

    def add_cost(self, usd: float) -> float:
        import fcntl
        f = self.day_file()
        with open(str(f) + ".lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            cur = json.loads(f.read_text()) if f.exists() else {"usd": 0.0, "requests": 0}
            cur["usd"] = float(cur.get("usd", 0.0)) + usd
            cur["requests"] = int(cur.get("requests", 0)) + 1
            f.write_text(json.dumps(cur))
        return cur["usd"]

    def handle(self, req: dict) -> tuple[int, dict]:
        model = self.model_map.get(req.get("model"), req.get("model"))
        if self.provider == "deepseek" and self.start_balance is not None and self.budget_cny is not None:
            spent = self.start_balance - self.balance()
            if spent >= self.budget_cny:
                self.log({"t": time.time(), "event": "budget_stop", "spent_cny": spent})
                return 402, {"error": {"message": f"soarm budget cap reached: spent {spent:.2f} CNY"}}
        if self.daily_cap_usd is not None and self.spent_today() >= self.daily_cap_usd:
            msg = f"soarm daily cap: estimated ${self.spent_today():.2f} >= ${self.daily_cap_usd:.2f} today (Beijing day)"
            if self.quota_flag:
                Path(self.quota_flag).write_text(json.dumps({"t": time.time(), "model": model, "detail": msg}))
            self.log({"t": time.time(), "event": "daily_cap", "detail": msg})
            return 429, {"error": {"message": msg, "type": "usage_limit_exceeded"}}
        if self.provider == "gateway":
            body, url = chat_to_responses({**req, "model": model}, reasoning=self.reasoning,
                                          max_output_tokens=self.max_output_tokens), self.base + "/responses"
            # one cache key per episode (the first two messages carry the episode's workspace path): repeated prefixes of
            # one episode hit the provider's prompt cache (terra cached only ~47 % of input tokens without it)
            head = json.dumps(req.get("messages", [])[:2], sort_keys=True)
            body["prompt_cache_key"] = "soarm-" + hashlib.sha1(head.encode()).hexdigest()[:24]
            if self.image_cap:
                body["input"], _ = cap_response_images(body.get("input") or [], self.image_cap)
        else:
            body = {k: v for k, v in req.items() if k not in ("chat_template_kwargs",)}
            body["model"] = model
            if self.max_output_tokens:
                body["max_tokens"] = self.max_output_tokens
            # DeepSeek thinking mode: drop sampling params it ignores. reasoning_content of earlier assistant tool-call turns
            # MUST be passed back (HTTP 400 "The `reasoning_content` in the thinking mode must be passed back"; tested 11-09);
            # the driver keeps it in history (history_reasoning on), turns without it get an empty string.
            body.pop("temperature", None)
            body.pop("top_p", None)
            for m in body.get("messages", []):
                if m.get("role") == "assistant":
                    m.setdefault("reasoning_content", "")
            url = self.base + "/chat/completions"
        body.update(self.extra)
        last = None
        for attempt in range(self.retries):
            t0 = time.time()
            try:
                with self.sem:
                    t_sent = time.time()
                    raw = self.data_learning.respond(body) if self.data_learning else self._post(url, body)
                out = responses_to_chat(raw, model) if self.provider == "gateway" else raw
                u = out.get("usage", {})
                hit, miss, _ = usage_tokens(u)
                usd = estimate_usd(model, u, t_sent)
                day_usd = self.add_cost(usd) if usd is not None and self.provider == "gateway" else None
                self.log({"t": t0, "event": "ok", "model": model, "attempt": attempt, "queue_s": round(t_sent - t0, 2),
                          "latency_s": round(time.time() - t_sent, 2), "prompt_tokens": u.get("prompt_tokens"),
                          "completion_tokens": u.get("completion_tokens"), "cached_tokens": hit, "miss_tokens": miss,
                          "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                          "est_usd": round(usd, 6) if usd is not None else None,
                          "day_est_usd": round(day_usd, 4) if day_usd is not None else None,
                          "tool_calls": len(out["choices"][0]["message"].get("tool_calls") or []),
                          "data_learning": _dl_log(raw.get("data_learning")),
                          "finish": out["choices"][0].get("finish_reason")})
                return 200, out
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:300]
                last = (exc.code, detail)
                self.log({"t": t0, "event": "http_error", "model": model, "attempt": attempt, "code": exc.code,
                          "detail": detail, "elapsed_s": round(time.time() - t0, 2)})
                if "usage_limit_exceeded" in detail:      # daily cost quota of the key: no retry, raise the flag
                    if self.quota_flag:
                        Path(self.quota_flag).write_text(json.dumps({"t": time.time(), "model": model,
                                                                     "detail": detail[:200]}))
                    break
                # One upstream: a data-learning round fed the previous round's output items back to another upstream
                # connection (401 "input item ID does not belong to this connection", ~2/h at 40 lanes,
                # each one an infrastructure_error episode); a retry reruns the whole request
                if exc.code in (400, 401, 403, 404, 422) and "does not belong to this connection" not in detail:
                    break
            except Exception as exc:  # noqa: BLE001 - timeouts, resets
                last = (599, f"{type(exc).__name__}: {exc}")
                self.log({"t": t0, "event": "error", "model": model, "attempt": attempt, "error": last[1][:300],
                          "elapsed_s": round(time.time() - t0, 2)})
            time.sleep(min(120, 5 * 2 ** attempt))
        code, detail = last or (599, "unknown")
        return (code if code < 599 else 502), {"error": {"message": f"upstream failed: {detail}"}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8650)
    ap.add_argument("--env-file", required=True)
    ap.add_argument("--provider", choices=("gateway", "deepseek"), default="gateway")
    ap.add_argument("--model-map", nargs="*", default=None, help="alias=upstream model id")
    ap.add_argument("--reasoning", default=None, help='Responses reasoning object, e.g. {"effort":"medium","summary":"auto"}')
    ap.add_argument("--max-output-tokens", type=int, default=16000)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--budget-cny", type=float, default=None, help="deepseek: refuse requests once this much was spent")
    ap.add_argument("--extra-body", default=None, help="JSON merged into every upstream request")
    ap.add_argument("--quota-flag", default=None,
                    help="file written when the upstream reports usage_limit_exceeded (batches stop launching)")
    ap.add_argument("--daily-cap-usd", type=float, default=None,
                    help="gateway: stop (quota flag) once today's estimated spend (DeepSeek list prices) reaches this")
    ap.add_argument("--data-learning", choices=("off", "forced", "parallel"), default="off",
                    help="gateway: have the teacher write its reasoning in a visible log_reasoning_step call and fold it "
                         "into reasoning_content (forced = 2 calls/request, parallel = 1; soarm/data_learning.py)")
    ap.add_argument("--data-learning-rounds", type=int, default=3,
                    help="with --data-learning: scratchpad calls per request before the tool is withdrawn")
    ap.add_argument("--response-image-cap", type=int, default=None,
                    help="gateway: send only the N most recent history images (the chat backend's / SFT exporter's cap)")
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    proxy = Proxy(a)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def _send(self, code: int, obj: dict) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(200, {"object": "list", "data": [{"id": k, "object": "model"} for k in proxy.model_map] or []})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n).decode())
            if not self.path.rstrip("/").endswith("/chat/completions"):
                self._send(404, {"error": {"message": "only /v1/chat/completions"}})
                return
            code, out = proxy.handle(req)
            self._send(code, out)

    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
