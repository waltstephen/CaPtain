"""soarm-speed runtime patches for vLLM 0.29 (serving only; a separate tool, not part of the harness or the episode
code). Loaded as `sitecustomize` when this directory is first on PYTHONPATH: tools/serve_policy.sh with
SPEC_VERIFY_PATCH=1 does that for the vLLM process only.

SOARM_VLLM_PATCH_3D=1 (default on when this file is on the path):
  Triton unified attention launches its split-KV ("3D", 16 softmax segments) kernel only when every query is a single
  token. A speculative-decoding verify step has 1+k query tokens per sequence, so it falls back to the 2D kernel with
  grid (q_blocks, kv_heads) = (2, 4) CTAs for our batch-1 policy: 1.9 ms per full-attention layer at 30 k context instead
  of 0.22 ms (measured, tools/profile_decode.py), which made every drafter (MTP, DFlash, suffix) slower than no
  speculation. The 3D kernel is correct for multi-token queries (per-row causal masking, per-row segment reduction);
  its segment buffers hold `seq_threshold_3D` query rows, so it is used only when the step has <= 16 query tokens per
  sequence and fits those rows. Use with --attention-backend TRITON_ATTN.

SOARM_VLLM_PATCH_FA_PREFILL=1 (default on when this file is on the path):
  the same backend computes PREFILL attention (steps with > 16 query tokens per sequence) with FlashAttention-2
  (vllm_flash_attn varlen, paged) directly on the TRITON_ATTN KV cache layout instead of the Triton 2D kernel:
  2048 new tokens on a 32 k context take 7.3 ms per full-attention layer instead of 13.7 ms on a 4090 (same output to
  2.4e-4, tools/vllm_patches/fa_prefill_check.py). Decode and verify steps keep the Triton split-KV path. Only for the
  plain case (decoder, no alibi/sinks/softcap/sliding window/mm-prefix/chunked-local attention). With an FP8 KV cache
  (FA2 has none) a one-sequence prefill step gathers its blocks, dequantizes them to bf16 and runs dense varlen FA2.

SOARM_DRAFT_VOCAB=<file.safetensors> (off by default): the Qwen3.5 MTP drafter scores only a fixed vocabulary subset
  (FR-Spec style; file = {"ids": int64 [K], "weight": bf16 [K, hidden]} rows of the lm_head, built by
  tools/vllm_patches/build_draft_vocab.py). Every draft step otherwise reads the full 248 k-row lm_head. Lossless: the
  target still verifies every draft token; a token outside the subset can only be produced by the target itself.

SOARM_VLLM_PATCH_NO_EAGLE_DROP=1 (off by default): keep the full prefix-cache hit with an EAGLE/MTP drafter. vLLM drops
  the last matched cache block for EAGLE/MTP because the drafter's KV at the last cached position may hold the
  lookahead token of an earlier continuation. On a hybrid (GDN) Qwen3.5 the cache block is 544 tokens, so every
  request re-prefills one extra block: measured 1050 vs 520 computed tokens per append request and +50 ms TTFT
  (rec_sB requests 16-24, MTP-4 vs no spec). The target's KV is exact either way, so outputs are unchanged; at worst
  one drafter KV entry per request is stale (acceptance, not correctness).

SOARM_VLLM_PATCH_FP8_PREFILL=1 (off by default): W4A16 (compressed-tensors, Marlin) linear layers also keep an FP8 copy
  of their dequantized weight (per-output-channel scale, made at load time by running the W4 kernel on an identity
  matrix) and run steps with more than SOARM_FP8_PREFILL_MIN_TOKENS (default 64) tokens - prefill chunks - as FP8 x FP8
  cutlass GEMMs with per-token dynamic activation scales (Ada FP8 tensor cores); decode / verify steps keep the W4
  Marlin path. Lossy (FP8 activations and weight rounding in prefill only). Layers wider than 100 k outputs (the
  lm_head) are left alone. Costs ~1 byte per weight of extra GPU memory. This patch changes the compiled model
  forward: serve with VLLM_DISABLE_COMPILE_CACHE=1 (a cached torch.compile artifact of the same model would silently
  keep the plain W4 path).

SOARM_VLLM_PATCH_MTP_NGRAM=1 (off by default; serve with num_speculative_tokens = K, e.g. 15): copy-aware drafting.
  The MTP head drafts only the first SOARM_MTP_CHAIN (default 5) tokens (the rest of the K slots are padding); when the
  last SOARM_MTP_NGRAM_N (default 6) tokens of the request (prompt + output) occurred earlier, all K drafts are replaced
  by the continuation of the most recent earlier occurrence (prompt lookup). The agent copies handles and earlier code
  from its own context (52.6 % of answer + code characters), so these drafts are long and usually accepted.
  Drafts are only proposals: n-gram / padding drafts carry one-hot draft probabilities, so probabilistic rejection
  sampling stays exact (lossless). The history is tracked on the CPU from the sampled tokens (one GPU sync per step)
  and kept across requests that extend the previous prompt (the agent's next request).
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys

_TARGETS = {"vllm.v1.attention.ops.triton_unified_attention", "vllm.v1.attention.backends.triton_attn",
            "vllm.model_executor.models.qwen3_5_mtp", "vllm.v1.core.kv_cache_coordinator",
            "vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16",
            "vllm.v1.spec_decode.llm_base_proposer", "vllm.v1.worker.gpu_model_runner"}


class _NgramHistory:
    """Token history of one request with an index: last-N-token key -> position of the token that followed it.
    Reused for the next request when its prompt extends this history (rollback to the common prefix)."""

    def __init__(self, n: int):
        self.n, self.tokens, self.index, self.undo = n, [], {}, []

    def _push(self, token: int) -> None:
        t = len(self.tokens)
        if t >= self.n:
            key = tuple(self.tokens[t - self.n:t])
            self.undo.append((key, self.index.get(key)))
            self.index[key] = t
        else:
            self.undo.append(None)
        self.tokens.append(token)

    def reset_to(self, prompt: list) -> None:
        common = 0
        limit = min(len(prompt), len(self.tokens))
        while common < limit and self.tokens[common] == prompt[common]:
            common += 1
        while len(self.tokens) > common:                      # roll back past the common prefix
            self.tokens.pop()
            entry = self.undo.pop()
            if entry is not None:
                key, previous = entry
                if previous is None:
                    self.index.pop(key, None)
                else:
                    self.index[key] = previous
        for token in prompt[common:]:
            self._push(token)

    def extend(self, tokens) -> None:
        for token in tokens:
            self._push(int(token))

    def continuation(self, k: int):
        if len(self.tokens) < self.n:
            return None
        start = self.index.get(tuple(self.tokens[-self.n:]))
        if start is None:
            return None
        cont = self.tokens[start:start + k]
        return cont if cont else None


def _patch_mtp_proposer(cls) -> None:
    """SpecDecodeBaseProposer.propose: draft only SOARM_MTP_CHAIN tokens with the model, pad to K (see module doc)."""
    import torch

    chain = int(os.environ.get("SOARM_MTP_CHAIN", "5"))
    original = cls.propose

    def propose(self, num_speculative_tokens, *args, **kwargs):
        k_total = int(num_speculative_tokens)
        if k_total <= chain:
            return original(self, num_speculative_tokens, *args, **kwargs)
        drafts = original(self, chain, *args, **kwargs)                 # [B, chain]
        self.num_speculative_tokens = k_total
        pad = drafts[:, -1:].expand(-1, k_total - chain)
        probs = self._last_draft_probs
        if probs is not None:                                           # [B, chain, V]: one-hot rows for padding
            extra = torch.zeros((probs.shape[0], k_total - chain, probs.shape[-1]), dtype=probs.dtype,
                                device=probs.device)
            extra.scatter_(2, pad.unsqueeze(-1).to(torch.int64), 1.0)
            self._last_draft_probs = torch.cat([probs, extra], dim=1).contiguous()
        return torch.cat([drafts, pad], dim=1).contiguous()

    propose.__wrapped__ = original
    cls.propose = propose


def _patch_mtp_runner(cls) -> None:
    """GPUModelRunner.propose_draft_token_ids: replace the drafts of requests whose recent tokens repeat earlier context."""
    import torch

    n = int(os.environ.get("SOARM_MTP_NGRAM_N", "6"))
    original = cls.propose_draft_token_ids

    def propose_draft_token_ids(self, scheduler_output, sampled_token_ids, *args, **kwargs):
        drafts = original(self, scheduler_output, sampled_token_ids, *args, **kwargs)
        if not isinstance(drafts, torch.Tensor) or drafts.dim() != 2:
            return drafts
        histories = self.__dict__.setdefault("_soarm_ngram", {})
        shared = self.__dict__.setdefault("_soarm_ngram_last", [None])
        sampled = sampled_token_ids.tolist() if isinstance(sampled_token_ids, torch.Tensor) else sampled_token_ids
        k = drafts.shape[1]
        rows, conts = [], []
        for row, req_id in enumerate(self.input_batch.req_ids[:drafts.shape[0]]):
            history = histories.get(req_id)
            if history is None:                       # new request: reuse the previous request's history if extended
                state = self.requests.get(req_id)
                prompt = list(getattr(state, "prompt_token_ids", None) or [])
                history = shared[0] if shared[0] is not None else _NgramHistory(n)
                history.reset_to(prompt)
                histories.clear()                     # batch-1 policy server: keep one live history
                histories[req_id] = history
                shared[0] = history
            history.extend(t for t in (sampled[row] if row < len(sampled) else []) if t >= 0)
            cont = history.continuation(k)
            if cont is not None:
                rows.append(row)
                conts.append(cont + [cont[-1]] * (k - len(cont)))
        if rows:
            index = torch.tensor(rows, device=drafts.device)
            values = torch.tensor(conts, device=drafts.device, dtype=drafts.dtype)
            drafts = drafts.clone()
            drafts[index] = values
            probs = self._draft_probs
            if probs is not None and probs.shape[1] == k:
                probs = probs.clone()
                probs[index] = 0.0
                probs[index.unsqueeze(1), torch.arange(k, device=drafts.device).unsqueeze(0), values.to(torch.int64)] = 1.0
                self._draft_probs = probs
        return drafts

    propose_draft_token_ids.__wrapped__ = original
    cls.propose_draft_token_ids = propose_draft_token_ids


_FP8_LAYERS: list = []


def _patch_fp8_prefill(cls) -> None:
    """CompressedTensorsWNA16: FP8 copy of each W4 weight for prefill-sized steps (see module doc)."""
    import torch

    min_tokens = int(os.environ.get("SOARM_FP8_PREFILL_MIN_TOKENS", "64"))

    # explicit schema: this module uses `from __future__ import annotations` (string annotations cannot be inferred)
    @torch.library.custom_op("soarm::hybrid_w4_fp8_linear", mutates_args=(), schema="(Tensor x, int index) -> Tensor")
    def hybrid_linear(x, index):
        layer, scheme = _FP8_LAYERS[index]
        x2 = x.reshape(-1, x.shape[-1])
        if x2.shape[0] > min_tokens:
            from vllm import _custom_ops as ops
            xq, xs = ops.scaled_fp8_quant(x2, use_per_token_if_dynamic=True)
            out = ops.cutlass_scaled_mm(xq, layer.soarm_fp8_weight, xs, layer.soarm_fp8_scale, out_dtype=x.dtype)
        else:
            out = scheme.kernel.apply_weights(layer, x2, None)
        return out.reshape(*x.shape[:-1], out.shape[-1])

    @hybrid_linear.register_fake
    def _(x, index):
        layer, _scheme = _FP8_LAYERS[index]
        return x.new_empty((*x.shape[:-1], layer.soarm_fp8_weight.shape[1]))

    original_process, original_apply = cls.process_weights_after_loading, cls.apply_weights

    def process_weights_after_loading(self, layer):
        original_process(self, layer)
        in_features = int(getattr(layer, "input_size_per_partition", 0) or 0)
        out_features = int(getattr(layer, "output_size_per_partition", 0) or 0)
        if not in_features or not out_features or out_features > 100_000:
            return
        with torch.no_grad():
            device = next(p.device for p in layer.parameters())
            eye = torch.eye(in_features, dtype=torch.bfloat16, device=device)
            w_t = self.kernel.apply_weights(layer, eye, None).float()        # [K, N] = dequantized W^T
            scale = w_t.abs().amax(dim=0, keepdim=True).clamp(min=1e-12) / 448.0     # per output channel [1, N]
            w8 = (w_t / scale).to(torch.float8_e4m3fn)                                # [K, N]
            layer.soarm_fp8_weight = w8.t().contiguous().t()                         # column-major K x N for cutlass
            layer.soarm_fp8_scale = scale.to(torch.float32)
            del eye, w_t
        layer.soarm_fp8_index = len(_FP8_LAYERS)
        _FP8_LAYERS.append((layer, self))

    def apply_weights(self, layer, x, bias=None):
        if getattr(layer, "soarm_fp8_index", None) is None:
            return original_apply(self, layer, x, bias)
        out = torch.ops.soarm.hybrid_w4_fp8_linear(x, layer.soarm_fp8_index)
        return out if bias is None else out + bias

    cls.process_weights_after_loading, cls.apply_weights = process_weights_after_loading, apply_weights
    cls._soarm_fp8_prefill = True


def _wrap_coordinator(original):
    """get_kv_cache_coordinator without the EAGLE last-block drop (no group flagged, use_eagle False)."""
    import dataclasses

    def get_kv_cache_coordinator(kv_cache_config, *args, **kwargs):
        if any(g.is_eagle_group for g in kv_cache_config.kv_cache_groups):
            kv_cache_config = dataclasses.replace(kv_cache_config, kv_cache_groups=[
                dataclasses.replace(g, is_eagle_group=False) for g in kv_cache_config.kv_cache_groups])
        if "use_eagle" in kwargs:
            kwargs["use_eagle"] = False
        elif len(args) >= 3:
            args = args[:2] + (False,) + args[3:]
        return original(kv_cache_config, *args, **kwargs)

    get_kv_cache_coordinator.__wrapped__ = original
    return get_kv_cache_coordinator


def _wrap(original):
    def unified_attention(*args, **kwargs):
        q = kwargs.get("q", args[0] if args else None)
        msq = kwargs.get("max_seqlen_q")
        rows = kwargs.get("seq_threshold_3D")
        if (msq is not None and 1 < msq <= 16 and rows is not None and q is not None and q.shape[0] <= rows
                and kwargs.get("mm_prefix_range") is None):
            kwargs["max_seqlen_q"] = 1      # only selects the split-KV launch; rows come from cu_seqlens_q
        return original(*args, **kwargs)

    unified_attention.__wrapped__ = original
    return unified_attention


def _wrap_forward(original):
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale=None,
                output_block_scale=None):
        md = attn_metadata
        fp8 = str(self.kv_cache_dtype).startswith("fp8")
        if (md is None or md.max_query_len <= 16 or output_scale is not None or output_block_scale is not None
                or self.attn_type != "decoder" or (self.kv_cache_dtype != "auto" and not fp8)
                or (fp8 and md.query_start_loc.shape[0] != 2)   # FP8: one sequence per step (our policy)
                or getattr(self, "_is_per_token_head_quant", False) or self.alibi_slopes is not None
                or self.sinks is not None or (self.logits_soft_cap or 0) != 0
                or tuple(self.sliding_window or (-1, -1)) != (-1, -1) or getattr(self, "chunk_lookback", -1) != -1
                or not md.causal or md.mm_prefix_range_tensor is not None
                or getattr(md, "rswa_prefix_lens", None) is not None):
            return original(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale,
                            output_block_scale)
        n = md.num_actual_tokens
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        if not fp8:
            flash_attn_varlen_func(q=query[:n], k=key_cache, v=value_cache, out=output[:n],
                                   cu_seqlens_q=md.query_start_loc, max_seqlen_q=md.max_query_len,
                                   seqused_k=md.seq_lens, max_seqlen_k=md.max_seq_len, softmax_scale=self.scale,
                                   causal=True, block_table=md.block_table, fa_version=2)
            return output
        # FP8 KV (FA2 has none): gather this sequence's blocks, dequantize to the output dtype, dense varlen FA2.
        # vLLM also hands over an FP8 query in this mode (layer._q_scale): dequantize it as well.
        import torch
        dt = output.dtype
        q = query[:n]
        if q.dtype != dt:
            q = q.to(dt) * layer._q_scale.to(dt)
        seq_len = md.max_seq_len
        block = key_cache.shape[1]
        blocks = md.block_table[0, :(seq_len + block - 1) // block]
        dense = []
        for cache, scale in ((key_cache, layer._k_scale), (value_cache, layer._v_scale)):
            c = cache.view(self.fp8_dtype)[blocks].reshape(-1, cache.shape[2], cache.shape[3])[:seq_len]
            dense.append(c.to(dt) * scale.to(dt))
        cu_k = torch.tensor([0, seq_len], dtype=torch.int32, device=q.device)
        flash_attn_varlen_func(q=q, k=dense[0], v=dense[1], out=output[:n], cu_seqlens_q=md.query_start_loc,
                               max_seqlen_q=md.max_query_len, cu_seqlens_k=cu_k, max_seqlen_k=seq_len,
                               softmax_scale=self.scale, causal=True, fa_version=2)
        return output

    forward.__wrapped__ = original
    return forward


def _patch_draft_vocab(cls, path: str) -> None:
    """Qwen3_5MTP: score only the subset rows (see module docstring)."""
    load_weights, compute_logits, get_top_tokens = cls.load_weights, cls.compute_logits, cls.get_top_tokens

    def load(self, weights):
        out = load_weights(self, weights)
        import torch
        from safetensors.torch import load_file
        sub = load_file(path)
        dev = next(self.parameters()).device
        self._soarm_ids = sub["ids"].to(dev)
        self._soarm_w = sub["weight"].to(device=dev, dtype=torch.bfloat16)
        return out

    def logits(self, hidden_states, spec_step_idx: int = 0):
        w = getattr(self, "_soarm_w", None)
        if w is None:
            return compute_logits(self, hidden_states, spec_step_idx)
        import torch
        full = torch.full((hidden_states.shape[0], self.config.vocab_size), float("-inf"),
                          device=hidden_states.device, dtype=hidden_states.dtype)
        full[:, self._soarm_ids] = hidden_states @ w.t().to(hidden_states.dtype)
        return full

    def top_tokens(self, hidden_states):
        w = getattr(self, "_soarm_w", None)
        if w is None:
            return get_top_tokens(self, hidden_states)
        return self._soarm_ids[(hidden_states @ w.t().to(hidden_states.dtype)).argmax(dim=-1)]

    cls.load_weights, cls.compute_logits, cls.get_top_tokens = load, logits, top_tokens
    cls._soarm_draft_vocab = True


def _patch(module) -> None:
    fn = getattr(module, "unified_attention", None)
    if os.environ.get("SOARM_VLLM_PATCH_3D", "1") == "1" and fn is not None and not hasattr(fn, "__wrapped__"):
        module.unified_attention = _wrap(fn)
    impl = getattr(module, "TritonAttentionImpl", None)
    if (os.environ.get("SOARM_VLLM_PATCH_FA_PREFILL", "1") == "1" and impl is not None
            and not hasattr(impl.forward, "__wrapped__")):
        impl.forward = _wrap_forward(impl.forward)
    mtp = getattr(module, "Qwen3_5MTP", None)
    if os.environ.get("SOARM_DRAFT_VOCAB") and mtp is not None and not getattr(mtp, "_soarm_draft_vocab", False):
        _patch_draft_vocab(mtp, os.environ["SOARM_DRAFT_VOCAB"])
    if os.environ.get("SOARM_VLLM_PATCH_MTP_NGRAM") == "1":
        proposer = getattr(module, "SpecDecodeBaseProposer", None)
        if proposer is not None and not hasattr(proposer.propose, "__wrapped__"):
            _patch_mtp_proposer(proposer)
        runner = getattr(module, "GPUModelRunner", None)
        if runner is not None and not hasattr(runner.propose_draft_token_ids, "__wrapped__"):
            _patch_mtp_runner(runner)
    scheme = getattr(module, "CompressedTensorsWNA16", None)
    if (os.environ.get("SOARM_VLLM_PATCH_FP8_PREFILL") == "1" and scheme is not None
            and not getattr(scheme, "_soarm_fp8_prefill", False)):
        _patch_fp8_prefill(scheme)
    coord = getattr(module, "get_kv_cache_coordinator", None)
    if (os.environ.get("SOARM_VLLM_PATCH_NO_EAGLE_DROP") == "1" and coord is not None
            and not hasattr(coord, "__wrapped__")):
        module.get_kv_cache_coordinator = _wrap_coordinator(coord)


class _Loader(importlib.abc.Loader):
    def __init__(self, inner):
        self.inner = inner

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        _patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in _TARGETS:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader)
        return spec


if "1" in (os.environ.get("SOARM_VLLM_PATCH_3D", "1"), os.environ.get("SOARM_VLLM_PATCH_FA_PREFILL", "1"),
           os.environ.get("SOARM_VLLM_PATCH_NO_EAGLE_DROP"), os.environ.get("SOARM_VLLM_PATCH_FP8_PREFILL"),
           os.environ.get("SOARM_VLLM_PATCH_MTP_NGRAM")) \
        or os.environ.get("SOARM_DRAFT_VOCAB"):
    sys.meta_path.insert(0, _Finder())
