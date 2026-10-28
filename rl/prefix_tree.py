"""Shared-prefix ("prefix tree") training for one trajectory's trained requests (opt-in).

All trained requests of one episode start with the same tokens (system prompt + tool docs + task + first frame, ~17.9k of
~41k tokens). The per-request trainer runs that prefix through the model once per request (ref forward, policy forward,
recompute and backward). Here it runs once per episode:

  1. plan_episode: P = longest common token prefix of the episode's samples, cut so that (a) no target token's predicting
     hidden state lies inside it (P <= first target index - 1 of every sample), (b) it never splits an image's pad run, and
     (c) it is a multiple of 64 (fla's chunk size: the GatedDeltaNet chunk boundaries then coincide with the per-request
     forward). Images fully inside the prefix must be the same images (hash / pixel identity checked) in every sample.
     M-RoPE positions are computed on every full sample exactly as the per-request forward does and sliced.
  2. prefix forward (capture): the HF model runs on the prefix with a PrefixCache in capture mode: every full-attention
     layer's (rotated, normed) K/V and every GatedDeltaNet layer's conv tail (last 4 pre-conv q/k/v inputs) and final
     recurrent state are recorded.
  3. branch forwards: each sample's suffix runs with the prefix states as cache (branch mode): attention keys/values are
     cat(prefix, own); GatedDeltaNet continues from the prefix conv tail and recurrent state (fla initial_state, which has
     a gradient). The branch attention is causal with the prefix in front (query i at absolute position P+i sees keys
     0..P+i): computed with the same SDPA causal kernel as the per-request forward on zero-padded queries (the P padding rows
     are dropped; their output gradient is 0, so they contribute nothing to dK/dV).
  4. gradient: the prefix states handed to the branches are detached leaves (requires_grad); each branch's loss is
     backpropagated on its own (memory = one branch at a time, like now), accumulating d(prefix states) in the leaves and
     the branch part of the parameter gradient. Finally ONE backward through the prefix graph with the accumulated state
     gradients (plus a zero gradient on the prefix's last hidden state so FSDP2's pre-backward hooks fire layer by layer,
     top-down, before any node of that layer runs). By the chain rule the sum equals the per-request gradients exactly.

The per-request loss code (train.sample_loss) is reused unchanged: a branch is handed to it as an `inputs`/`mask`
pair over the SUFFIX tokens plus a `core` that runs the branch forward (hidden index j = absolute position P+j predicts
suffix token j+1, so masked_logprobs' position logic is unchanged).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

ALIGN = 64


# ----------------------------------------------------------------------------- attention (branch: causal with prefix in front)
class PrefixBranchMask:
    """Sentinel passed as the full-attention layers' attention_mask in branch mode: P = prefix length in front of the queries."""

    def __init__(self, P: int):
        self.P = int(P)


_ORIG_SDPA = None
# "lower_right" (default): t x (P+t) attention; "padded": zero queries in front + the per-request causal flash kernel (P^2/2 waste)
import os as _os0
ATTN_MODE = _os0.environ.get("RL_PREFIX_TREE_ATTN", "lower_right")


def install_attention() -> None:
    """Wrap the 'sdpa' attention interface once: any call whose attention_mask is a PrefixBranchMask runs the zero-padded causal
    form; every other call is delegated unchanged to the original function (the default path stays bit-identical)."""
    global _ORIG_SDPA
    from transformers.modeling_utils import AttentionInterface
    if _ORIG_SDPA is not None:
        return
    _ORIG_SDPA = AttentionInterface._global_mapping["sdpa"]

    def tree_sdpa(module, query, key, value, attention_mask, *args, **kwargs):
        if not isinstance(attention_mask, PrefixBranchMask):
            return _ORIG_SDPA(module, query, key, value, attention_mask, *args, **kwargs)
        P = attention_mask.P
        b, h, t, d = query.shape
        if key.shape[2] != P + t:
            raise RuntimeError(f"prefix-tree attention: {key.shape[2]} keys != prefix {P} + {t} queries")
        if ATTN_MODE == "lower_right":
            # bottom-right aligned causal mask (query i at absolute P+i sees keys 0..P+i): memory-efficient SDPA kernel, only
            # t x (P+t) work; K/V heads repeated for GQA (that kernel has no enable_gqa)
            from torch.nn.attention.bias import causal_lower_right
            rep = getattr(module, "num_key_value_groups", 1)
            k = key.repeat_interleave(rep, dim=1) if rep > 1 else key
            v = value.repeat_interleave(rep, dim=1) if rep > 1 else value
            out = torch.nn.functional.scaled_dot_product_attention(query, k, v, attn_mask=causal_lower_right(t, P + t),
                                                                   scale=kwargs.get("scaling"))
            return out.transpose(1, 2).contiguous(), None
        q = torch.cat([query.new_zeros(b, h, P, d), query], dim=2)            # "padded": the per-request flash kernel
        out, _ = _ORIG_SDPA(module, q, key, value, None, *args, **kwargs)       # [b, P+t, h, d] (transposed by HF)
        return out[:, P:], None

    AttentionInterface._global_mapping["sdpa"] = tree_sdpa


# ----------------------------------------------------------------------------- cache (duck-types the HF Cache API used by Qwen3.5)
class _Slot:
    record_past = False

    def __init__(self, conv=None, rec=None):
        self.conv_states = {0: conv}
        self.recurrent_states = {0: rec}


class PrefixCache:
    """capture: record the prefix's per-layer states; branch: serve them (read-only) to one suffix forward."""

    def __init__(self, n_layers: int, mode: str, states: dict | None = None):
        assert mode in ("capture", "branch")
        self.mode = mode
        self.n_layers = n_layers
        # capture: frozen after the prefix forward. Activation-checkpoint recomputes in the backward call the layers again with
        # this cache; storing their (graph-carrying) recomputed states would keep every recomputed layer alive (+3.5 GB/layer)
        self.frozen = False
        self.states = states if states is not None else {"k": {}, "v": {}, "conv": {}, "rec": {}}
        if mode == "branch":
            self.layers = [_Slot(self.states["conv"].get(i), self.states["rec"].get(i)) for i in range(n_layers)]
            self.P = int(states["P"])
        else:
            self.layers = [_Slot() for _ in range(n_layers)]
            self.P = 0

    # GatedDeltaNet
    def has_previous_state(self, layer_idx=None, state_idx=None) -> bool:
        return self.mode == "branch"

    def update_conv_state(self, conv_states: torch.Tensor, layer_idx: int, conv_kernel_size: int | None = None, **kwargs):
        if self.mode == "capture":
            if not self.frozen:
                k = int(conv_kernel_size or 4)
                tail = conv_states[..., -k:].clone()     # not a view: a view would keep the whole [conv_dim, P] input alive
                if tail.shape[-1] < k:
                    tail = torch.nn.functional.pad(tail, (k - tail.shape[-1], 0))
                self.states["conv"][layer_idx] = tail
            return conv_states
        return torch.cat([self.states["conv"][layer_idx].to(conv_states.dtype), conv_states], dim=-1)

    def update_recurrent_state(self, recurrent_states: torch.Tensor, layer_idx: int, **kwargs):
        if self.mode == "capture" and not self.frozen:
            self.states["rec"][layer_idx] = recurrent_states
        return recurrent_states

    # full attention
    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, *args, **kwargs):
        if self.mode == "capture":
            if not self.frozen:
                self.states["k"][layer_idx] = key_states
                self.states["v"][layer_idx] = value_states
            return key_states, value_states
        return (torch.cat([self.states["k"][layer_idx], key_states], dim=-2),
                torch.cat([self.states["v"][layer_idx], value_states], dim=-2))

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.P


def allow_cache_under_checkpointing(model_core) -> None:
    """HF's GradientCheckpointingLayer drops past_key_values under activation checkpointing unless the layer opts in. Safe
    here: in capture mode a recompute only overwrites a state dict entry that was already handed to autograd; in branch mode
    the cache is read-only."""
    for layer in model_core.language_model.layers:
        layer._can_checkpoint_with_cache = True


# ----------------------------------------------------------------------------- planning
@dataclass
class Branch:
    sample: dict
    ids: torch.Tensor              # suffix token ids [T]
    mask: torch.Tensor             # suffix target mask [T]
    pos: torch.Tensor              # suffix M-RoPE positions [3, 1, T]
    pixel_values: torch.Tensor | None
    grid: torch.Tensor | None


@dataclass
class EpisodePlan:
    P: int
    ids: torch.Tensor              # prefix ids [P]
    pos: torch.Tensor              # [3, 1, P]
    pixel_values: torch.Tensor | None
    grid: torch.Tensor | None
    branches: list = field(default_factory=list)
    lcp: int = 0


def _pad_runs(ids: torch.Tensor, pad_id: int) -> list[tuple[int, int]]:
    is_pad = (ids == pad_id).to(torch.int8)
    if not bool(is_pad.any()):
        return []
    d = torch.diff(torch.cat([is_pad.new_zeros(1), is_pad, is_pad.new_zeros(1)]))
    starts = torch.nonzero(d == 1).flatten().tolist()
    ends = torch.nonzero(d == -1).flatten().tolist()
    return list(zip(starts, ends))


def _patch_counts(grid: torch.Tensor) -> list[int]:
    return [int(x) for x in grid.prod(-1).tolist()]


def plan_episode(samples: list[dict], core, image_pad_id: int, align: int = ALIGN, device=None) -> EpisodePlan:
    """Prefix/suffix split of one trajectory's samples (each = _request_samples output: input_ids, mask, optional pixel_values /
    image_grid_thw). Raises ValueError when the samples cannot share a prefix (then the caller must not use the tree)."""
    if not samples:
        raise ValueError("empty episode")
    ids = [s["input_ids"] for s in samples]
    n = min(len(x) for x in ids)
    lcp = n
    ref = ids[0]
    for x in ids[1:]:
        neq = torch.nonzero(ref[:n] != x[:n]).flatten()
        if neq.numel():
            lcp = min(lcp, int(neq[0]))
    first_target = min(int(torch.nonzero(s["mask"]).flatten()[0]) for s in samples)
    P = min(lcp, first_target - 1)
    P = (P // align) * align
    runs = [_pad_runs(x, image_pad_id) for x in ids]
    while True:
        moved = False
        for rr in runs:
            for a, b in rr:
                if a < P < b:
                    P = (a // align) * align
                    moved = True
        if not moved:
            break
    if P <= 0:
        raise ValueError(f"no shareable prefix (lcp {lcp}, first target {first_target})")
    # images fully inside the prefix: same count, grids and pixels in every sample
    n_pre = [sum(1 for a, b in rr if b <= P) for rr in runs]
    if len(set(n_pre)) != 1:
        raise ValueError(f"prefix image counts differ: {n_pre}")
    k = n_pre[0]
    pre_pix = pre_grid = None
    if k:
        g0 = samples[0]["image_grid_thw"][:k]
        rows = sum(_patch_counts(g0))
        pre_pix = samples[0]["pixel_values"][:rows]
        pre_grid = g0
        for s in samples[1:]:
            if not torch.equal(s["image_grid_thw"][:k], g0) or not torch.equal(s["pixel_values"][:rows], pre_pix):
                raise ValueError("an image inside the shared prefix differs between requests")
    dev = device or torch.device("cpu")
    plan = EpisodePlan(P=P, ids=ref[:P].clone(), pos=None, pixel_values=pre_pix, grid=pre_grid, lcp=lcp)
    for s, x, rr in zip(samples, ids, runs):
        if "pixel_values" in s:
            mm = (x == image_pad_id).long().unsqueeze(0).to(dev)
            pos, _ = core.get_rope_index(x.unsqueeze(0).to(dev), mm_token_type_ids=mm, image_grid_thw=s["image_grid_thw"].to(dev),
                                         attention_mask=torch.ones(1, len(x), dtype=torch.long, device=dev))
        else:
            pos = torch.arange(len(x), device=dev).view(1, 1, -1).expand(3, 1, -1)
        pos = pos.cpu()
        if plan.pos is None:
            plan.pos = pos[..., :P].clone()
        elif not torch.equal(plan.pos, pos[..., :P]):
            raise ValueError("prefix M-RoPE positions differ between requests")
        pix = grid = None
        if "pixel_values" in s and len(rr) > k:
            counts = _patch_counts(s["image_grid_thw"])
            start = sum(counts[:k])
            pix = s["pixel_values"][start:]
            grid = s["image_grid_thw"][k:]
        plan.branches.append(Branch(sample=s, ids=x[P:].clone(), mask=s["mask"][P:].clone(), pos=pos[..., P:].clone(),
                                    pixel_values=pix, grid=grid))
    return plan


# ----------------------------------------------------------------------------- forwards
def _call_core(core, ids, pos, pix, grid, cache, mask_full, device):
    kw = {"input_ids": ids.unsqueeze(0).to(device), "position_ids": pos.to(device), "past_key_values": cache,
          "attention_mask": {"full_attention": mask_full, "linear_attention": None}, "use_cache": False}
    if pix is not None:
        kw["pixel_values"] = pix.to(device)              # get_image_features casts to the vision dtype (same as the per-request path)
        kw["image_grid_thw"] = grid.to(device)
    return core(**kw)


class _Inject(torch.autograd.Function):
    """Identity on a decoder layer's prefix output; its backward also emits the gradients the branches accumulated for that
    layer's prefix states (holder['leaves'][key][layer].grad). The state gradients therefore enter layer i exactly when the
    backward reaches layer i's output: top-down order, one layer's recompute alive at a time, FSDP2 pre-backward hook (on the
    layer output) before any node of the layer runs."""

    @staticmethod
    def forward(ctx, out, holder, layer, keys, *states):
        ctx.holder, ctx.layer, ctx.keys = holder, layer, keys
        ctx.shapes = [(t.shape, t.dtype, t.device) for t in states]
        return out.clone()

    @staticmethod
    def backward(ctx, g):
        leaves = ctx.holder.get("leaves") or {}
        grads = []
        for key, (shape, dtype, device) in zip(ctx.keys, ctx.shapes):
            leaf = (leaves.get(key) or {}).get(ctx.layer)
            gr = None if leaf is None else leaf.grad
            grads.append(torch.zeros(shape, dtype=dtype, device=device) if gr is None else gr.to(dtype))
        return (g, None, None, None, *grads)


_CAPTURE = {"ctx": None}


def install_capture_hooks(model_core) -> None:
    """Forward hooks on the decoder layers: active only during a grad-mode prefix capture forward (not during its recompute,
    not for branches, not for the per-request path), where they wrap the layer output in _Inject."""
    for i, layer in enumerate(model_core.language_model.layers):
        if getattr(layer, "_prefix_tree_hook", False):
            continue

        def hook(module, args, output, i=i):
            ctx = _CAPTURE["ctx"]
            if ctx is None or not torch.is_grad_enabled():
                return None
            cache, holder = ctx
            keys, states = [], []
            for key in ("k", "v", "conv", "rec"):
                t = cache.states[key].get(i)
                if t is not None and t.requires_grad:
                    keys.append(key)
                    states.append(t)
            if not states:
                return None
            if isinstance(output, tuple):
                return (_Inject.apply(output[0], holder, i, tuple(keys), *states),) + tuple(output[1:])
            return _Inject.apply(output, holder, i, tuple(keys), *states)

        layer.register_forward_hook(hook)
        layer._prefix_tree_hook = True


def prefix_forward(core, plan: EpisodePlan, device, *, grad: bool):
    """Run the prefix; returns (states dict incl. P, last hidden state, holder). With grad the states keep their graph and the
    layer outputs carry _Inject nodes (install_capture_hooks) that feed holder['leaves'] gradients back in prefix_backward."""
    n_layers = len(core.language_model.layers)
    cache = PrefixCache(n_layers, "capture")
    holder = {"leaves": None}
    if grad:
        _CAPTURE["ctx"] = (cache, holder)
    try:
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            out = _call_core(core, plan.ids, plan.pos, plan.pixel_values, plan.grid, cache, None, device)
    finally:
        _CAPTURE["ctx"] = None
        cache.frozen = True
    st = cache.states
    st["P"] = plan.P
    st["_holder"] = holder
    return st, out.last_hidden_state


def detach_states(states: dict) -> dict:
    """Leaves for the branches: same values, own autograd roots (their .grad accumulates d loss / d prefix state)."""
    out = {"P": states["P"]}
    for key in ("k", "v", "conv", "rec"):
        out[key] = {i: t.detach().requires_grad_(t.requires_grad) for i, t in states[key].items()}
    return out


class BranchCore:
    """`core` for sample_loss/masked_logprobs: core(**inputs) runs branch b on top of the prefix states. `inputs` must be the
    branch inputs from branch_inputs() (suffix ids); only input_ids is read from it (the rest comes from the plan)."""

    def __init__(self, core, branch: Branch, states: dict, device):
        self.core, self.branch, self.states, self.device = core, branch, states, device

    def __call__(self, **inputs):
        n_layers = len(self.core.language_model.layers)
        cache = PrefixCache(n_layers, "branch", self.states)
        b = self.branch
        return _call_core(self.core, b.ids, b.pos, b.pixel_values, b.grid, cache, PrefixBranchMask(self.states["P"]), self.device)


def branch_inputs(branch: Branch, device) -> tuple[dict, torch.Tensor]:
    return {"input_ids": branch.ids.unsqueeze(0).to(device)}, branch.mask.unsqueeze(0).to(device)


def prefix_backward(states: dict, leaves: dict, last_hidden: torch.Tensor) -> None:
    """One backward through the prefix: the root is the prefix's last hidden state with a zero gradient; each layer's _Inject
    node adds the state gradients the branches accumulated (leaves[...].grad) on the way down."""
    states["_holder"]["leaves"] = leaves
    torch.autograd.backward([last_hidden], [torch.zeros_like(last_hidden)])


def dummy_branch_like(plan: EpisodePlan, n: int = 32) -> Branch:
    """A zero-weight FSDP filler branch: n text tokens continuing the prefix (one target), same collective sequence as a real one."""
    ids = torch.zeros(n, dtype=torch.long)          # a plain text token (never an image placeholder)
    mask = torch.zeros(n, dtype=torch.bool)
    mask[-1] = True
    last = int(plan.pos.max()) + 1 if plan.pos.numel() else 0
    pos = (torch.arange(n) + last).view(1, 1, -1).expand(3, 1, -1).clone()
    s = {"input_ids": ids, "mask": mask, "length": n, "n_tokens": 1, "weight": 1.0, "advantage": 0.0}
    return Branch(sample=s, ids=ids, mask=mask, pos=pos, pixel_values=None, grid=None)


# ----------------------------------------------------------------------------- one episode
import os as _os


def _mem(tag):
    if _os.environ.get("PT_DEBUG_MEM"):
        torch.cuda.synchronize()
        print(f"[mem] {tag}: alloc {torch.cuda.memory_allocated() / 2**30:.2f} GiB peak {torch.cuda.max_memory_allocated() / 2**30:.2f}", flush=True)

def run_episode(plan: EpisodePlan, *, core, lm_head, device, loss_fn, masked_logprobs, ref_core=None, ref_head=None,
                scales: list[float] | None = None, timer=None) -> list:
    """Reference forwards (prefix once, then each branch, no grad), policy prefix forward with grad, per-branch
    loss_fn(core_b, inputs, mask, sample, ref_lp, scale) -> loss then loss.backward(), finally the prefix backward.
    Returns the per-branch loss_fn results. The collective sequence depends only on len(plan.branches) (pad with
    dummy_branch_like for FSDP lockstep)."""
    import contextlib
    tm = timer or (lambda name: contextlib.nullcontext())
    scales = scales if scales is not None else [1.0] * len(plan.branches)
    ref_lps = [None] * len(plan.branches)
    if ref_core is not None:
        with torch.no_grad(), tm("t_ref_fwd"):
            rstates, _ = prefix_forward(ref_core, plan, device, grad=False)
            for j, b in enumerate(plan.branches):
                inputs, mask = branch_inputs(b, device)
                ref_lps[j] = masked_logprobs(BranchCore(ref_core, b, rstates, device), ref_head, inputs, mask, per_token=True)
            del rstates
    with tm("t_prefix_fwd"):
        states, last_hidden = prefix_forward(core, plan, device, grad=True)
        leaves = detach_states(states)
    results = []
    _mem("after prefix fwd")
    for j, b in enumerate(plan.branches):
        inputs, mask = branch_inputs(b, device)
        res = loss_fn(BranchCore(core, b, leaves, device), inputs, mask, b.sample, ref_lps[j], scales[j])
        results.append(res)
        _mem(f"after branch {j}")
    with tm("t_prefix_bwd"):
        prefix_backward(states, leaves, last_hidden)
    del states, leaves, last_hidden
    return results


def dummy_plan(n_branches: int, n: int = 64, token_id: int = 0) -> EpisodePlan:
    """A zero-weight FSDP filler episode (text only): same collective sequence as a real episode with n_branches branches."""
    ids = torch.full((n,), int(token_id), dtype=torch.long)
    pos = torch.arange(n).view(1, 1, -1).expand(3, 1, -1).clone()
    plan = EpisodePlan(P=n, ids=ids, pos=pos, pixel_values=None, grid=None, lcp=n)
    plan.branches = [dummy_branch_like(plan) for _ in range(n_branches)]
    return plan
