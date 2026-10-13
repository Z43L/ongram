"""Generate text using a DistilGPT2 base plus an accepted NRA-M bundle."""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nram_container import load_bundle
from nram_evolve import MODEL_DIR, attach_adapters, restore_adapters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent / "models" / "distilgpt2-nra-qa.nram")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--min-new-tokens", type=int, default=1)
    args = parser.parse_args()

    manifest, tensors = load_bundle(args.bundle)
    spec = manifest["adapter"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True).eval()
    adapters = attach_adapters(model, spec["targets"], int(spec["rank"]), float(spec["alpha"]))
    restore_adapters(adapters, tensors)
    encoded = tokenizer(args.prompt, return_tensors="pt")
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
