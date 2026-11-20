"""Check for sitecustomize.py SOARM_VLLM_PATCH_FA_PREFILL: FA2 varlen (paged) on vLLM's TRITON_ATTN KV layout
(num_blocks, H, block, 2*D) -> transpose(1,2).split, against a float reference and the Triton unified kernel, plus the
speed of both on one 2048-token prefill chunk over a 30 k context (Qwen3.5-4B full-attention shapes, block 528).

    CUDA_VISIBLE_DEVICES=<free gpu> $VLLM_VENV/bin/python tools/vllm_patches/fa_prefill_check.py
"""
import torch, time
from vllm.vllm_flash_attn import flash_attn_varlen_func
torch.manual_seed(0)
dev = "cuda"
H, Hkv, D, B = 16, 4, 256, 528
ctx, nq = 30000, 2048
nblocks = (ctx + nq) // B + 8
cache = torch.randn(nblocks, Hkv, B, 2 * D, device=dev, dtype=torch.bfloat16)
perm = torch.randperm(nblocks, device=dev)[: (ctx + nq + B - 1) // B].to(torch.int32)
block_table = perm[None, :]
kv = cache.transpose(1, 2)
k, v = kv.split(D, dim=-1)
print("k stride", k.stride(), "contig last", k.stride(-1) == 1)
q = torch.randn(nq, H, D, device=dev, dtype=torch.bfloat16)
out = torch.empty_like(q)
L = ctx + nq
cu_q = torch.tensor([0, nq], device=dev, dtype=torch.int32)
seqused = torch.tensor([L], device=dev, dtype=torch.int32)
scale = D ** -0.5
def fa():
    flash_attn_varlen_func(q=q, k=k, v=v, out=out, cu_seqlens_q=cu_q, max_seqlen_q=nq, seqused_k=seqused,
                           max_seqlen_k=L, softmax_scale=scale, causal=True, block_table=block_table, fa_version=2)
fa(); torch.cuda.synchronize()
# reference on a subset of query rows
kk = k[perm.long()].reshape(-1, Hkv, D)[:L].float(); vv = v[perm.long()].reshape(-1, Hkv, D)[:L].float()
rows = torch.tensor([0, 1, 777, nq - 1], device=dev)
g = H // Hkv
err = 0.0
for r in rows.tolist():
    pos = ctx + r
    qr = q[r].float()                                  # [H, D]
    kh = kk[: pos + 1].repeat_interleave(g, dim=1)      # [pos+1, H, D]
    vh = vv[: pos + 1].repeat_interleave(g, dim=1)
    s = torch.einsum("hd,nhd->hn", qr, kh) * scale
    ref = torch.einsum("hn,nhd->hd", s.softmax(-1), vh)
    err = max(err, (ref - out[r].float()).abs().max().item())
print("max abs err vs reference", err)
for _ in range(3): fa()
torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(20): fa()
torch.cuda.synchronize(); print("FA2 ms per layer call", (time.perf_counter() - t) / 20 * 1e3)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
out2 = torch.empty_like(q)
ones = torch.ones(1, Hkv, device=dev, dtype=torch.float32)
def tr():
    unified_attention(q=q, k=k, v=v, out=out2, cu_seqlens_q=cu_q, max_seqlen_q=nq, seqused_k=seqused, max_seqlen_k=L,
                      softmax_scale=scale, causal=True, window_size=(-1, -1), block_table=block_table, softcap=0,
                      q_descale=None, k_descale=ones, v_descale=ones)
tr(); torch.cuda.synchronize()
print("triton vs FA2 max abs diff", (out2.float() - out.float()).abs().max().item())
for _ in range(3): tr()
torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(20): tr()
torch.cuda.synchronize(); print("Triton ms per layer call", (time.perf_counter() - t) / 20 * 1e3)

# FP8 KV path of the patch (k_scale = v_scale = 1, what vLLM uses without calibrated scales): gather blocks,
# dequantize, dense varlen FA2 vs the Triton kernel on the FP8 cache
fp8 = torch.float8_e4m3fn
cache8 = (cache * 0.5).to(fp8)
k8, v8 = cache8.transpose(1, 2).split(D, dim=-1)
scale_t = torch.tensor(1.0, device=dev)
def fa8():
    dense = [c[perm.long()].reshape(-1, Hkv, D)[:L].to(torch.bfloat16) * scale_t.to(torch.bfloat16) for c in (k8, v8)]
    flash_attn_varlen_func(q=q, k=dense[0], v=dense[1], out=out, cu_seqlens_q=cu_q, max_seqlen_q=nq,
                           cu_seqlens_k=torch.tensor([0, L], device=dev, dtype=torch.int32), max_seqlen_k=L,
                           softmax_scale=scale, causal=True, fa_version=2)
def tr8():
    unified_attention(q=q, k=k8, v=v8, out=out2, cu_seqlens_q=cu_q, max_seqlen_q=nq, seqused_k=seqused, max_seqlen_k=L,
                      softmax_scale=scale, causal=True, window_size=(-1, -1), block_table=block_table, softcap=0,
                      q_descale=None, k_descale=ones, v_descale=ones)
fa8(); tr8(); torch.cuda.synchronize()
print("FP8 KV: FA2(dequant) vs Triton max abs diff", (out2.float() - out.float()).abs().max().item())
for name, fn in (("FA2 fp8-dequant", fa8), ("Triton fp8", tr8)):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(20): fn()
    torch.cuda.synchronize(); print(name, "ms per layer call", (time.perf_counter() - t) / 20 * 1e3)
# which FP8 result matches a float reference on the dequantized cache?
kk8 = k8[perm.long()].reshape(-1, Hkv, D)[:L].float() * 1; vv8 = v8[perm.long()].reshape(-1, Hkv, D)[:L].float() * 1
fa8(); torch.cuda.synchronize(); o_fa = out.clone(); tr8(); torch.cuda.synchronize(); o_tr = out2.clone()
for r in (0, 777, nq - 1):
    pos = ctx + r; qr = q[r].float()
    kh = kk8[: pos + 1].repeat_interleave(g, dim=1); vh = vv8[: pos + 1].repeat_interleave(g, dim=1)
    ref = torch.einsum("hn,nhd->hd", (torch.einsum("hd,nhd->hn", qr, kh) * scale).softmax(-1), vh)
    print("row", r, "ref absmax", round(ref.abs().max().item(), 4), "FA2 err", round((ref - o_fa[r].float()).abs().max().item(), 5),
          "Triton err", round((ref - o_tr[r].float()).abs().max().item(), 5))
