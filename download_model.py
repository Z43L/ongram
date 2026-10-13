from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "distilgpt2"
REVISION = "2290a62682d06624634c1f46a6ad5be0f47f38aa"
FILES = [
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]

if not (MODEL_DIR / "model.safetensors").exists():
    snapshot_download(
        "distilbert/distilgpt2",
        revision=REVISION,
        local_dir=MODEL_DIR,
        allow_patterns=FILES,
    )
else:
    print(f"Checkpoint already present: {MODEL_DIR}")
