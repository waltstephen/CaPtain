"""Supervised (loss) token count of every row exactly as train_sft.py labels it (assistant spans; supervise_turns / turn_index);
images do not change assistant spans, so text-only tokenisation is exact. Prints C = total / number of optimizer steps."""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor
import torch
sys.argv_keep = sys.argv[:]
P = None
QA_W = float(os.environ.get("QA_W", "1.0"))   # per-token weight of qapack rows (CaPtain pipeline: 0.5); meta.loss_weight overrides
def init(tok):
    global P
    from transformers import AutoProcessor
    P = AutoProcessor.from_pretrained(tok)
def count(line):
    sys.argv = ["x"]
    from train_sft import assistant_spans
    r = json.loads(line)
    text = P.apply_chat_template(r["messages"], tools=r.get("tools") or None, tokenize=False, add_generation_prompt=False)
    ids = torch.tensor(P.tokenizer(text, add_special_tokens=False)["input_ids"])
    spans = assistant_spans(ids, P.tokenizer); m = r.get("meta") or {}
    if m.get("supervise_turns") is not None: spans = [spans[t] for t in sorted(set(int(t) for t in m["supervise_turns"]))]
    elif m.get("turn_index") is not None and spans: spans = spans[-1:]
    n = sum(e - s for s, e in spans)
    q = m.get("source") == "qapack"
    w = m.get("loss_weight")
    if w is None:
        w = QA_W if q else 1.0
    return q, n, n * float(w)
if __name__ == "__main__":
    data, tok, world = sys.argv_keep[1], sys.argv_keep[2], int(sys.argv_keep[3])
    lines = open(data).read().splitlines()
    with ProcessPoolExecutor(48, initializer=init, initargs=(tok,)) as ex:
        res = list(ex.map(count, lines, chunksize=16))
    steps = (len(lines) + world - 1) // world
    tot = sum(n for _, n, _ in res); qa = sum(n for q, n, _ in res if q); ag = tot - qa
    wt = sum(w for _, _, w in res); wqa = sum(w for q, _, w in res if q)
    per_step = [sum(n for _, n, _ in res[i:i + world]) for i in range(0, len(res), world)]
    print(json.dumps({"rows": len(lines), "steps": steps, "sup_tokens": tot, "agent_sup": ag, "qa_sup": qa,
                      "C_mean_per_step": round(tot / steps, 1), "step_min": min(per_step), "step_max": max(per_step),
                      "agent_row_mean": round(ag / max(1, sum(1 for q, *_ in res if not q)), 1), "qa_row_mean": round(qa / max(1, sum(1 for q, *_ in res if q)), 1),
                      "weighted_sup": round(wt), "C_weighted_per_step": round(wt / steps, 1), "agent_weighted_share": round((wt - wqa) / wt, 3)}))
