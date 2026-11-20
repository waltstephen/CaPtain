"""Chat client for taskgen: deepseek-v4.1-flash-1017 through soarm.gateway_proxy (OpenAI chat API on localhost).

The proxy logs into $LOG_DIR, so taskgen spend lands in the same per-Beijing-day cost file as the teacher collection and
counts against the shared $30/day cap. This client additionally keeps its own ledger (taskgen usage jsonl) and stops at
`budget_usd` for one run.
"""
from __future__ import annotations

import http.client
import json
import os
import re
import threading
import time
import urllib.request
from pathlib import Path

from soarm.gateway_proxy import estimate_usd, usage_tokens

MODEL = "deepseek-v4.1-flash-1017"


class BudgetExceeded(RuntimeError):
    pass


class LLM:
    def __init__(self, url: str = "http://127.0.0.1:8661/v1/chat/completions", model: str = MODEL,
                 budget_usd: float = 2.0, ledger: str | None = None) -> None:
        self.url, self.model, self.budget = url, model, budget_usd
        self.ledger = Path(ledger or os.path.join(os.environ.get("LOG_DIR", "."), "taskgen_usage.jsonl"))
        self.spent = 0.0
        self.calls = 0
        self.tokens = {"hit": 0, "miss": 0, "out": 0}
        self._lock = threading.Lock()

    def chat(self, messages: list[dict], *, tag: str, temperature: float = 0.9, max_tokens: int = 12000) -> str:
        if self.spent >= self.budget:
            raise BudgetExceeded(f"taskgen run budget ${self.budget:.2f} reached (${self.spent:.3f})")
        body = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
        t0 = time.time()
        for attempt in range(3):          # a proxy restart drops the connection: retry instead of losing the run
            try:
                with urllib.request.urlopen(req, timeout=900) as r:
                    resp = json.loads(r.read())
                break
            except (OSError, http.client.HTTPException):
                if attempt == 2:
                    raise
                time.sleep(30)
        if "error" in resp:
            raise RuntimeError(f"llm error: {str(resp['error'])[:300]}")
        usage = resp.get("usage") or {}
        usd = estimate_usd(self.model, usage, time.time()) or 0.0
        hit, miss, out = usage_tokens(usage)
        with self._lock, open(self.ledger, "a") as f:
            self.spent += usd
            self.calls += 1
            for k, v in (("hit", hit), ("miss", miss), ("out", out)):
                self.tokens[k] += v
            f.write(json.dumps({"t": time.time(), "tag": tag, "usd": usd, "hit": hit, "miss": miss, "out": out,
                                "wall_s": round(time.time() - t0, 1)}) + "\n")
        return resp["choices"][0]["message"].get("content") or ""


def extract_json(text: str):
    """First JSON array/object in a reply (fenced or bare)."""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    body = m.group(1) if m else text
    start = min([i for i in (body.find("["), body.find("{")) if i >= 0], default=-1)
    if start < 0:
        raise ValueError("no JSON in reply")
    dec = json.JSONDecoder()
    obj, _ = dec.raw_decode(body[start:])
    return obj
