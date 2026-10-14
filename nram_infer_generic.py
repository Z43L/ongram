"""Text-only inference using a generic NRA-M LoRA bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from nram_generic import load_nram_model, load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")

    manifest, model, _adapters = load_nram_model(
        args.bundle, local_files_only=args.local_files_only,
    )
    base = manifest["base_model"]
    tokenizer_source = base.get("tokenizer_source") or base["source"]
    tokenizer = load_tokenizer(
        tokenizer_source,
        revision=base.get("tokenizer_revision", base.get("revision")),
        local_files_only=args.local_files_only,
    )
    model.to(args.device).eval()
    encoded = tokenizer(args.prompt, return_tensors="pt").to(args.device)
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
