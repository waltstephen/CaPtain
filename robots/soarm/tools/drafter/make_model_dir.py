"""Build a vLLM model dir = deploy W4A16 + W8-lm_head policy + a (fine-tuned) MTP head, all symlinks.

    python make_model_dir.py --mtp $SOARM_ROOT/speed/drafter/run1/mtp.safetensors \
        --out $SOARM_ROOT/speed/models/sftv0-w4lmh8-mtpft

Same layout as models/sftv0-w4lmh8-mtp: every file of --base except config.json, the config.json of --config-from
(quantization ignore rules for the mtp prefix), and mtp.safetensors -> --mtp. Serve with
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'.
"""
from __future__ import annotations

import argparse
import os

from safetensors import safe_open


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=os.path.expandvars("$SOARM_ROOT/speed/models/sftv0-w4-lmhW8"))
    ap.add_argument("--config-from", default=os.path.expandvars("$SOARM_ROOT/speed/models/sftv0-w4lmh8-mtp"))
    a = ap.parse_args()
    mtp = os.path.abspath(a.mtp)
    with safe_open(mtp, "pt") as f:
        keys = list(f.keys())
    with safe_open(os.path.join(a.config_from, "mtp.safetensors"), "pt") as f:
        ref = set(f.keys())
    if set(keys) != ref:
        raise SystemExit(f"key set differs from the reference MTP file: {sorted(set(keys) ^ ref)}")
    os.makedirs(a.out, exist_ok=False)
    for name in sorted(os.listdir(a.base)):
        if name not in ("config.json", "mtp.safetensors"):
            os.symlink(os.path.join(a.base, name), os.path.join(a.out, name))
    os.symlink(os.path.realpath(os.path.join(a.config_from, "config.json")), os.path.join(a.out, "config.json"))
    os.symlink(mtp, os.path.join(a.out, "mtp.safetensors"))
    print("MODEL_DIR", a.out, len(keys), "mtp tensors")


if __name__ == "__main__":
    main()
