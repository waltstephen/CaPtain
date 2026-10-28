# Untie the Qwen3.5-4B base (sharded safetensors) for CaPtain SFT and RL: lm_head.weight = clone(embed_tokens), tie false, mtp.* dropped
# (HF Qwen3_5ForConditionalGeneration does not load them).   python untie_base.py <src dir> <dst dir>
import json, shutil, sys
from pathlib import Path
from safetensors.torch import load_file, save_file
src, dst = Path(sys.argv[1]), Path(sys.argv[2]); dst.mkdir(parents=True, exist_ok=True)
sd = {}
for f in sorted(src.glob("model*.safetensors")):
    sd.update(load_file(str(f)))
n_mtp = len([k for k in sd if k.startswith("mtp.")])
sd = {k: v for k, v in sd.items() if not k.startswith("mtp.")}
assert "lm_head.weight" not in sd
sd["lm_head.weight"] = sd["model.language_model.embed_tokens.weight"].clone()
save_file(sd, str(dst / "model.safetensors"), metadata={"format": "pt"})
cfg = json.load(open(src / "config.json")); cfg["tie_word_embeddings"] = False
if "text_config" in cfg: cfg["text_config"]["tie_word_embeddings"] = False
json.dump(cfg, open(dst / "config.json", "w"), indent=2)
for f in src.iterdir():
    if f.is_file() and not f.name.startswith("model") and f.name != "config.json":
        shutil.copy2(f, dst / f.name)
(dst / "UNTIED_FROM").write_text(f"{src}\nlm_head.weight = clone(embed_tokens); tie false; dropped {n_mtp} mtp.* tensors\n")
print("UNTIE_OK", len(sd), "tensors; dropped mtp", n_mtp)
