#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

bundle=${1:-models/distilgpt2-nra-qa.nram}
name=${2:-distilgpt2-nra-export}
hf_dir="models/${name}-hf"
f16_file="models/${name}-f16.gguf"
q4_file="models/${name}-Q4_K_M.gguf"

.venv/bin/python nram_merge.py --bundle "$bundle" --output "$hf_dir"
.venv/bin/python third_party/llama.cpp/convert_hf_to_gguf.py \
  "$hf_dir" --outfile "$f16_file" --outtype f16
build/bin/llama-quantize "$f16_file" "$q4_file" Q4_K_M
printf 'Wrote %s and %s\n' "$f16_file" "$q4_file"
