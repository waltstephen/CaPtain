#!/bin/bash
# Build an SFT dataset from teacher episodes: export (strict successes, held-out families excluded) -> prefix-pack ->
# total_tokens -> per-step order for <world> ranks -> loss constant C. Needs $SFT_VENV (transformers with Qwen3.5) and
# TOKENIZER_MODEL (any CaPtain-4B / Qwen3.5-4B model dir, for the chat template + tokenizer).
#   tools/build_sft.sh <out dir> <world> <run dir>...
set -euo pipefail
source "$(cd "$(dirname "$0")/.." && pwd)/env.sh"
OUT=$(realpath -m "${1:?out}"); WORLD=${2:?world}; shift 2
MODEL=${TOKENIZER_MODEL:?set TOKENIZER_MODEL to a CaPtain-4B / Qwen3.5-4B model dir}
PY=$SFT_VENV/bin/python
mkdir -p "$OUT"
globs=(); for r in "$@"; do globs+=("$(realpath -m "$r")/soarm_sim_*"); done
(cd "$SOARM_CODE" && python -m soarm.export_sft --episodes "${globs[@]}" --out "$OUT")
cd "$SFT_DIR"
$PY data/pack_sft_prefix.py --in "$OUT/rows.jsonl" --out "$OUT/packed.jsonl" --tokenizer "$MODEL"
$PY "$SOARM_CODE/tools/add_total_tokens.py" --in "$OUT/packed.jsonl" --out "$OUT/packed_tok.jsonl" \
  --model "$MODEL" --image-root "$OUT"
$PY - "$OUT" "${MAX_TOKENS:-65536}" <<'PY'
import json, sys
out, cap = sys.argv[1], int(sys.argv[2])
rows = [l for l in open(f"{out}/packed_tok.jsonl") if l.strip()]
# Image-less rows are dropped: with the frozen vision tower under ZeRO-3, a step where some ranks run the vision
# forward and others do not deadlocks the parameter all-gathers (rollout-1113: step 2 hung at 100 % GPU for 12 min).
keep = [l for l in rows if json.loads(l)["meta"]["total_tokens"] <= cap]
fit = [l for l in keep if json.loads(l).get("images")]
open(f"{out}/packed_fit.jsonl", "w").write("".join(fit))
print(json.dumps({"rows": len(rows), "kept_le_max_tokens": len(keep), "kept_with_images": len(fit), "max_tokens": cap}))
PY
$PY train/mix_order.py --inp "$OUT/packed_fit.jsonl" --out "$OUT/sft_mix$WORLD.jsonl" --world "$WORLD"
(cd train && $PY sup_tokens.py "$OUT/sft_mix$WORLD.jsonl" "$MODEL" "$WORLD") | tee "$OUT/sup_tokens.json"
python3 - "$OUT" <<'PY'
import json, sys, collections
out = sys.argv[1]
eps = [json.loads(l) for l in open(f"{out}/episodes.jsonl")]
kept = [e for e in eps if e.get("kept")]
fam = collections.Counter(e["family"] for e in kept)
print(json.dumps({"episodes_seen": len(eps), "kept": len(kept), "excluded": sum(1 for e in eps if e.get("excluded")),
                  "kept_by_family": dict(sorted(fam.items())),
                  "kept_by_teacher": dict(collections.Counter(e["model"] for e in kept))}, indent=1))
PY
