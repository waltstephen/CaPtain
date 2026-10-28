#!/usr/bin/env python3
"""Thin launcher around cap-x's SAM3 FastAPI server that accepts an explicit checkpoint.

cap-x's `capx.serving.launch_sam3_server.main` always calls
`build_sam3_image_model(enable_inst_interactivity=True)`, which downloads the gated
facebook/sam3 checkpoint from HF. We patch the builder so a local file can be used
and the request/response contract stays byte-identical to what the harness expects.
"""
from __future__ import annotations

import argparse
import functools
import os

import capx.serving.launch_sam3_server as srv
import sam3.model_builder as mb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8114)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint", default=None, help="local sam3.pt; if missing, fall back to HF download")
    args = ap.parse_args()

    ckpt = args.checkpoint if args.checkpoint and os.path.exists(args.checkpoint) else None
    if ckpt:
        srv.build_sam3_image_model = functools.partial(  # type: ignore[attr-defined]
            mb.build_sam3_image_model, checkpoint_path=ckpt, load_from_HF=False
        )
        print(f"[sam3_server] using local checkpoint {ckpt}")
    else:
        print("[sam3_server] no local checkpoint; sam3 will try HF download (needs HF_TOKEN with facebook/sam3 access)")
    srv.main(device=args.device, port=args.port, host=args.host)


if __name__ == "__main__":
    main()
