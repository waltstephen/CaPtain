#!/usr/bin/env bash
# Turn a run checkpoint into a standalone Hugging Face model directory (what was released as CaPtain-4B = step 300).
#   bash rl/scripts/export_checkpoint.sh <run_dir>/checkpoint-<N> <base_model dir (CaPtain-4B-SFT)> <out dir>
# Every checkpoint-<N> already holds the merged bf16 weights (model-*.safetensors + index), config, generation config,
# chat template and tokenizer; the image/video processor files and the BPE vocab are copied from the SFT base (unchanged by RL).
# The optimizer / FSDP state (full_state_dir) is not needed for inference.
set -euo pipefail
CK=${1:?checkpoint dir}; BASE=${2:?base model dir}; OUT=${3:?output dir}
mkdir -p "$OUT"
for f in "$CK"/*.safetensors "$CK"/model.safetensors.index.json "$CK"/config.json "$CK"/generation_config.json "$CK"/chat_template.jinja \
         "$CK"/tokenizer.json "$CK"/tokenizer_config.json "$CK"/processor_config.json; do
  [ -e "$f" ] && cp -n "$f" "$OUT/"
done
for f in preprocessor_config.json video_preprocessor_config.json merges.txt vocab.json; do
  [ -e "$OUT/$f" ] || cp "$BASE/$f" "$OUT/"
done
(cd "$OUT" && sha256sum *.safetensors *.json *.jinja *.txt > SHA256SUMS)
python3 - "$OUT" <<'P'
import json, os, struct, sys, glob
d = sys.argv[1]
for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; head = json.loads(fh.read(n))
    end = max(v["data_offsets"][1] for k, v in head.items() if k != "__metadata__")
    assert 8 + n + end == os.path.getsize(f), f"{f}: truncated"
print("exported", d)
P
