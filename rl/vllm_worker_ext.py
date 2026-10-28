"""vLLM worker extension for the policy servers (streaming RL).

vllm serve ... --worker-extension-cls vllm_worker_ext.RefitMemExt   (rl/ on the server's PYTHONPATH, VLLM_SERVER_DEV_MODE=1)

refit_empty_cache: after an NCCL weight update the receive buffers and the layerwise-reload staging tensors are freed into
PyTorch's caching allocator on the transfer's side streams, where the default stream cannot reuse them; nvidia-smi then
shows the server 4.5-9 GB larger than before the first refit (when the policies share GPUs with the
EGL renderers, those then failed to enumerate the near-full devices). Returning those blocks to the driver after every
update keeps the server's footprint flat. The KV cache and CUDA graph pools are allocated separately and are not touched.
"""
import gc


class RefitMemExt:
    def refit_mem(self) -> dict:
        import torch
        return {"allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
                "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 3)}

    def refit_empty_cache(self) -> dict:
        import torch
        gc.collect()
        torch.cuda.synchronize()
        before = torch.cuda.memory_reserved()
        torch.cuda.empty_cache()
        return {"reserved_before_gib": round(before / 2**30, 3), **self.refit_mem()}
