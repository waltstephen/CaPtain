"""Frozen Qwen3.5 target (HF, bf16): features the MTP head needs, computed on the fly (nothing cached to disk).

features(enc) -> dict(h [L,H] final hidden AFTER the model's final norm, emb [L,H] input embeddings with vision
embeddings merged at image positions, pos [3,L] M-RoPE position ids). The language model runs as a chunked prefill
(DynamicCache: full-attention KV + gated-delta-net states), which bounds activation memory for 50 k-token episodes.
"""
from __future__ import annotations

import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq


def use_torch_gdn() -> None:
    """CPU: route the gated-delta-net/conv kernels to transformers' reference torch code (fla/causal_conv1d are GPU)."""
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn",
                 "causal_conv1d_update"):
        fn = getattr(mq, name)
        while hasattr(fn, "__wrapped__"):
            fn = fn.__wrapped__
        setattr(mq, name, fn)


class Target:
    def __init__(self, model_dir: str, device: str = "cuda", dtype=torch.bfloat16):
        if not str(device).startswith("cuda"):
            use_torch_gdn()
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_dir, dtype=dtype, attn_implementation="sdpa"
        ).to(device).eval().requires_grad_(False)
        self.device = torch.device(device)
        self.text_config = self.model.config.text_config
        lm = self.model.model.language_model
        self.embed_tokens = lm.embed_tokens
        self.rotary = lm.rotary_emb
        self.lm_head = self.model.lm_head

    @torch.no_grad()
    def features(self, input_ids: torch.Tensor, mm: dict, chunk: int = 8192, aux_layers=None) -> dict:
        """aux_layers: target layer ids k (DFlash/DSpark target_layer_ids); returns feats["aux"] [L, len*H] = the
        residual stream after decoder layer k (vLLM aux hidden state k+1 = HF hidden_states[k+1]), concatenated."""
        m = self.model.model
        ids = input_ids.to(self.device)[None]
        emb = m.get_input_embeddings()(ids)
        mm = {k: v.to(self.device) for k, v in mm.items()}
        if "pixel_values" in mm:
            img = m.get_image_features(mm["pixel_values"].to(emb.dtype), mm["image_grid_thw"], return_dict=True)
            img = torch.cat(img.pooler_output, dim=0).to(emb.dtype)
            mask, _ = m.get_placeholder_mask(ids, inputs_embeds=emb, image_features=img)
            emb = emb.masked_scatter(mask, img)
        pos, _ = m.get_rope_index(
            ids,
            mm_token_type_ids=mm.get("mm_token_type_ids", torch.zeros_like(ids, dtype=torch.int32)),
            image_grid_thw=mm.get("image_grid_thw"),
        )
        L = ids.shape[1]
        H = emb.shape[-1]
        aux = None
        hooks = []
        if aux_layers:
            aux = torch.empty(L, len(aux_layers) * H, dtype=emb.dtype, device=self.device)
            span = [0, 0]

            def grab(slot):
                def hook(_mod, _inp, out):
                    out = out[0] if isinstance(out, tuple) else out
                    aux[span[0]:span[1], slot * H:(slot + 1) * H] = out[0]
                return hook

            for slot, k in enumerate(aux_layers):
                hooks.append(m.language_model.layers[k].register_forward_hook(grab(slot)))
        cache = DynamicCache(config=self.text_config)
        hs = []
        try:
            for a in range(0, L, chunk):
                b = min(a + chunk, L)
                if aux is not None:
                    span[0], span[1] = a, b
                out = m.language_model(
                    inputs_embeds=emb[:, a:b], position_ids=pos[:, :, a:b], past_key_values=cache, use_cache=True
                )
                hs.append(out.last_hidden_state[0])
        finally:
            for hk in hooks:
                hk.remove()
        del cache
        out = {"h": torch.cat(hs), "emb": emb[0], "pos": pos[:, 0]}
        if aux is not None:
            out["aux"] = aux
        return out

    def offload(self, on: bool) -> None:
        """Move the decoder layers + vision tower to CPU (on=True) or back, keeping embed/lm_head/norm/rotary on the
        GPU: frees ~7 GB while a large drafter trains on cached features of one body."""
        dev = torch.device("cpu") if on else self.device
        m = self.model.model
        m.language_model.layers.to(dev)
        m.visual.to(dev)
        if on:
            torch.cuda.empty_cache()

    def rope(self, pos3: torch.Tensor, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """pos3 [3, n] -> cos, sin [n, rot]."""
        x = torch.empty(0, device=pos3.device, dtype=dtype)
        cos, sin = self.rotary(x, pos3[:, None, :])
        return cos[0], sin[0]
