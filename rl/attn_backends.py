"""Trainer attention backends for the language model's full-attention layers (the reference run used FA3).

  None / "sdpa"   transformers' default SDPA (PyTorch picks its FlashAttention-2 kernel on H100 for bf16 causal) = default
  "fa3"           FlashAttention-3: the prebuilt kernels-community/flash-attn3 build for this torch/CUDA, imported from an overlay
                  directory (`flash_attn_interface`), registered as the transformers attention function "captain_fa3" with the
                  flash mask interface (no materialised L x L mask). Causal, GQA native, one unpadded sequence per call (the
                  trainer's passes are batch 1 without padding); anything else (padding mask, packed varlen kwargs) is refused.
  "sdpa_cudnn"    SDPA with the cuDNN backend first in PyTorch's priority order (flash, efficient, math after it).
The vision tower keeps SDPA in every mode (frozen, bidirectional, per-image varlen)."""
from __future__ import annotations

import sys

BACKENDS = (None, "sdpa", "fa3", "sdpa_cudnn")
_FA3 = {}


def _fa3_attention(module, query, key, value, attention_mask, dropout: float = 0.0, scaling: float | None = None, **kwargs):
    if type(attention_mask).__name__ == "PrefixBranchMask":
        # prefix tree, branch mode: the queries are the last t of P + t keys; FA3's causal mask for seqlen_q < seqlen_k is
        # aligned to the bottom-right corner (query i sees keys 0..P+i) = prefix_tree's causal_lower_right (checked
        # against the SDPA reference before use)
        if key.shape[2] != attention_mask.P + query.shape[2]:
            raise RuntimeError(f"captain_fa3: {key.shape[2]} keys != prefix {attention_mask.P} + {query.shape[2]} queries")
        attention_mask = None
    if attention_mask is not None:
        raise ValueError("captain_fa3: padded batches are not supported (the trainer runs one unpadded sequence per pass)")
    for k in ("cu_seq_lens_q", "cu_seq_lens_k", "cu_seqlens_q", "cu_seqlens_k"):
        if kwargs.get(k) is not None:
            raise ValueError("captain_fa3: packed varlen inputs are not supported")
    if dropout:
        raise ValueError("captain_fa3: attention dropout is not supported")
    if not getattr(module, "is_causal", True):
        raise ValueError("captain_fa3: only causal attention")
    q, k, v = (t.transpose(1, 2) for t in (query, key, value))       # [B, H, L, D] -> [B, L, H, D]; GQA handled by the kernel
    out = _FA3["func"](q, k, v, softmax_scale=scaling, causal=True)
    if isinstance(out, tuple):
        out = out[0]
    return out, None                                                  # [B, L, H, D] like transformers' sdpa_attention_forward


def install(backend: str | None, overlay: str | None = None, flash_attn_func=None) -> dict | str:
    """Set up the backend before the models are built; returns the attn_implementation argument for from_pretrained.
    flash_attn_func: a stand-in kernel (unit tests on GPUs without FA3)."""
    if backend not in BACKENDS:
        raise ValueError(f"attention backend must be one of {BACKENDS}")
    if backend in (None, "sdpa"):
        return None
    if backend == "sdpa_cudnn":
        from torch.nn.attention import SDPBackend, sdpa_kernel
        _FA3["cudnn_ctx"] = sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
                                         SDPBackend.MATH], set_priority=True)
        _FA3["cudnn_ctx"].__enter__()                                  # process-wide flags (also seen by autograd threads)
        return None
    if flash_attn_func is None:
        if overlay and overlay not in sys.path:
            sys.path.insert(0, overlay)
        from flash_attn_interface import flash_attn_func                # noqa: F811 - the prebuilt FA3 build (overlay)
    _FA3["func"] = flash_attn_func
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, flash_attention_mask
    AttentionInterface.register("captain_fa3", _fa3_attention)
    AttentionMaskInterface.register("captain_fa3", flash_attention_mask)
    return {"text_config": "captain_fa3", "vision_config": "sdpa"}
