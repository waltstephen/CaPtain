"""Unit test for --loss-tokens-per-step: with W ranks (one row each, mixed agent-like long target and QA-like short target),
the rank-averaged gradient of selective_causal_loss(num_items_in_batch=C, loss_scale=W) equals the gradient of
(sum of ALL token NLLs in the global batch) / C, i.e. (mean NLL over all tokens) * N_tokens / C -- and two steps with different
token counts give per-token weights that are identical (1/C)."""
import sys, types, torch
sys.argv = ["x"]
from train_sft import selective_causal_loss, IGNORE_INDEX
torch.manual_seed(0)
V, H = 50, 16
class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.emb = torch.nn.Embedding(V, H); self.head = torch.nn.Linear(H, V)
    def forward(self, input_ids, use_cache=False, logits_to_keep=None, **kw):
        h = self.emb(input_ids); logits = self.head(h)
        if logits_to_keep is not None: logits = logits[:, logits_to_keep]
        return types.SimpleNamespace(logits=logits)
def row(n_ctx, n_tgt):
    ids = torch.randint(0, V, (1, n_ctx + n_tgt)); lab = ids.clone(); lab[0, :n_ctx] = IGNORE_INDEX
    return {"input_ids": ids, "labels": lab}
def grads(model):
    return torch.cat([p.grad.flatten() for p in model.parameters()])
def step_grad(model, rows, C):
    W = len(rows); model.zero_grad(); acc = None
    for r in rows:   # each "rank": own backward; the engine averages gradients over ranks
        model.zero_grad(); loss, _ = selective_causal_loss(model, dict(input_ids=r["input_ids"], labels=r["labels"].clone()),
                                                           num_items_in_batch=C, loss_scale=W)
        loss.backward(); g = grads(model); acc = g if acc is None else acc + g
    return acc / W
def ref_grad(model, rows, C):
    model.zero_grad(); tot = 0.0; n = 0
    for r in rows:
        lab = r["labels"][0, 1:]; keep = (lab != IGNORE_INDEX).nonzero().squeeze(-1)
        lg = model(r["input_ids"], logits_to_keep=keep).logits[0]
        tot = tot + torch.nn.functional.cross_entropy(lg, lab[keep], reduction="sum"); n += keep.numel()
    ((tot / n) * n / C).backward()
    return grads(model)
m = Toy(); C = 300.0
stepA = [row(200, 400), row(100, 300), row(50, 6), row(40, 5)]      # 2 agent-like + 2 short-answer rows
stepB = [row(60, 7), row(60, 5), row(60, 8), row(60, 6)]            # all short answers
for name, st in (("mixed", stepA), ("short-only", stepB)):
    g1, g2 = step_grad(m, st, C), ref_grad(m, st, C)
    err = (g1 - g2).abs().max().item() / g2.abs().max().item()
    print(f"{name}: tokens={sum(int((r['labels'][0,1:]!=IGNORE_INDEX).sum()) for r in st)} rel_err={err:.2e}")
    assert err < 1e-5
# per-token weight is the same in both steps: grad of a single token's NLL contributes 1/C regardless of step size
gA = step_grad(m, stepA[2:3] + stepA[:1], C) ; gA_ref = ref_grad(m, stepA[2:3] + stepA[:1], C)
assert (gA - gA_ref).abs().max() / gA_ref.abs().max() < 1e-5
# contrast: the old per-step normalisation (denominator = this step's token count) gives the short-only step a full-size update
def old_norm(rows):
    W = len(rows); n = sum(int((r['labels'][0, 1:] != IGNORE_INDEX).sum()) for r in rows); acc = None
    for r in rows:
        m.zero_grad(); loss, _ = selective_causal_loss(m, dict(input_ids=r["input_ids"], labels=r["labels"].clone()), num_items_in_batch=n, loss_scale=W)
        loss.backward(); g = grads(m); acc = g if acc is None else acc + g
    return acc / W
ratio_new = step_grad(m, stepB, C).norm() / step_grad(m, stepA, C).norm()
ratio_old = old_norm(stepB).norm() / old_norm(stepA).norm()
print(f"grad-norm(short-only step)/grad-norm(mixed step): per-step norm {ratio_old:.2f} -> constant C {ratio_new:.2f}")
# weighted: agent rows weight 1.0, QA rows weight w -> rank-averaged grad == grad of (sum_rows w_r * sum NLL_r) / C_w
def step_grad_w(model, rows, ws, C):
    W = len(rows); acc = None
    for r, w in zip(rows, ws):
        model.zero_grad(); loss, _ = selective_causal_loss(model, dict(input_ids=r["input_ids"], labels=r["labels"].clone(),
                                                                       loss_weight=torch.tensor([w])), num_items_in_batch=C, loss_scale=W)
        loss.backward(); g = grads(model); acc = g if acc is None else acc + g
    return acc / W
def ref_grad_w(model, rows, ws, C):
    model.zero_grad(); tot = 0.0
    for r, w in zip(rows, ws):
        lab = r["labels"][0, 1:]; keep = (lab != IGNORE_INDEX).nonzero().squeeze(-1)
        lg = model(r["input_ids"], logits_to_keep=keep).logits[0]
        tot = tot + w * torch.nn.functional.cross_entropy(lg, lab[keep], reduction="sum")
    (tot / C).backward(); return grads(model)
ws = [1.0, 1.0, 0.5, 0.5]
g1, g2 = step_grad_w(m, stepA, ws, C), ref_grad_w(m, stepA, ws, C)
err = ((g1 - g2).abs().max() / g2.abs().max()).item(); print(f"weighted (agent 1.0 / QA 0.5): rel_err={err:.2e}"); assert err < 1e-5
# and the QA rows' share of the gradient really halves: grad of QA rows alone at w=0.5 == 0.5 x at w=1.0
gq5 = step_grad_w(m, stepA[2:], [0.5, 0.5], C); gq1 = step_grad_w(m, stepA[2:], [1.0, 1.0], C)
assert ((gq5 - 0.5 * gq1).abs().max() / gq1.abs().max()).item() < 1e-6
print("TOKEN_LOSS_TEST_OK")
