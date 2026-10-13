"""Merge a generic NRAM bundle and export GGUF using llama.cpp converters."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from nram_merge_generic import merge

ROOT = Path(__file__).resolve().parent


def export(bundle: Path, prefix: Path, *, model: str | None = None,
           tokenizer: str | None = None, quantization: str | None = "Q4_K_M",
           trust_remote_code: bool = False) -> list[Path]:
    hf_dir = prefix.with_name(prefix.name + "-hf")
    f16_file = prefix.with_name(prefix.name + "-f16.gguf")
    merge(bundle, hf_dir, model, tokenizer, trust_remote_code)
    converter = ROOT / "third_party" / "llama.cpp" / "convert_hf_to_gguf.py"
    subprocess.run([sys.executable, str(converter), str(hf_dir), "--outfile", str(f16_file),
                    "--outtype", "f16"], cwd=ROOT, check=True)
    outputs = [f16_file]
    if quantization:
        quantizer = ROOT / "build" / "bin" / "llama-quantize"
        if not quantizer.is_file():
            raise FileNotFoundError("llama-quantize missing; build llama.cpp quantize target first")
        q_file = prefix.with_name(f"{prefix.name}-{quantization}.gguf")
        subprocess.run([str(quantizer), str(f16_file), str(q_file), quantization], cwd=ROOT, check=True)
        outputs.append(q_file)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True,
                        help="Output path prefix, e.g. models/my-nram")
    parser.add_argument("--model", help="Override base source recorded in the bundle")
    parser.add_argument("--tokenizer")
    parser.add_argument("--quantization", default="Q4_K_M",
                        help="llama-quantize method; use 'none' to emit only F16")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    quantization = None if args.quantization.lower() == "none" else args.quantization
    outputs = export(args.bundle, args.output_prefix, model=args.model,
                     tokenizer=args.tokenizer, quantization=quantization,
                     trust_remote_code=args.trust_remote_code)
    print("Exported:\n" + "\n".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
