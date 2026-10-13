"""Create a zero-initialized, architecture-adapted NRAM bundle from an HF CausalLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from nram_container import save_bundle
from nram_generic import adapter_tensors, attach_adapters, discover_targets, load_base_model, weight_fingerprint


def convert(args: argparse.Namespace) -> dict:
    model = load_base_model(args.model, revision=args.revision,
                            trust_remote_code=args.trust_remote_code)
    revision = args.revision or getattr(model.config, "_commit_hash", None)
    # A trainable causal-LM needs a matching tokenizer; fail early if the source
    # is only a tensor dump without tokenizer/config metadata.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model, revision=args.tokenizer_revision or (revision if not args.tokenizer else None),
        trust_remote_code=args.trust_remote_code,
    )
    target_layouts = discover_targets(model, last_k_blocks=args.last_k_blocks,
                                      explicit=args.targets or None)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adapters = attach_adapters(model, target_layouts, args.rank, args.alpha)
    trainable = sum(parameter.numel() for adapter in adapters.values()
                    for parameter in adapter.parameters() if parameter.requires_grad)
    source = str(Path(args.model).resolve()) if Path(args.model).exists() else args.model
    manifest = {
        "model_family": "hf-causal-lm-dense-lora",
        "model_name": Path(args.output).stem,
        "version": 0,
        "base_model": {
            "source": source,
            "revision": revision,
            "model_type": getattr(model.config, "model_type", "unknown"),
            "architectures": getattr(model.config, "architectures", None),
            "weights_sha256": weight_fingerprint(args.model),
            "tokenizer_source": args.tokenizer or source,
            "tokenizer_revision": args.tokenizer_revision or (revision if not args.tokenizer else None),
        },
        "adapter": {
            "kind": "LoRA",
            "rank": args.rank,
            "alpha": args.alpha,
            "target_layouts": target_layouts,
            "targets": list(target_layouts),
            "selection": "explicit" if args.targets else "auto-last-block-projections",
            "last_k_blocks": args.last_k_blocks,
        },
        "graph": ["ENCODE", "READ", "TRANSFORM", "REFINE", "EVALUATE", "COMMIT", "DECODE", "HALT"],
        "tokenizer": {"class": tokenizer.__class__.__name__, "source": args.tokenizer or source},
        "evolution": {"objective": "lower held-out completion NLL", "optimizer_state_saved": False},
        "dataset": None,
        "last_validation": None,
        "journal": [],
    }
    save_bundle(args.output, manifest, adapter_tensors(adapters))
    return {"output": str(args.output), "model_type": manifest["base_model"]["model_type"],
            "revision": revision, "targets": list(target_layouts), "trainable_parameters": trainable,
            "rank": args.rank, "alpha": args.alpha, "format_version": 2}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face causal-LM ID or local Transformers checkpoint")
    parser.add_argument("--tokenizer", help="Tokenizer ID/path if different from --model")
    parser.add_argument("--tokenizer-revision", help="Pinned tokenizer revision when --tokenizer is separate")
    parser.add_argument("--revision", help="Pinned Hub branch, tag, or commit")
    parser.add_argument("--output", type=Path, required=True, help="Output .nram bundle")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--last-k-blocks", type=int, default=1)
    parser.add_argument("--targets", nargs="+", help="Exact projection names or module suffixes; overrides auto selection")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Allow executing custom model/tokenizer code from the model source")
    args = parser.parse_args()
    if args.rank < 1 or args.alpha <= 0:
        parser.error("rank must be positive and alpha must be greater than zero")
    print(json.dumps(convert(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
