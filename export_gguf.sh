#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -d third_party/llama.cpp/.git ]]; then
  mkdir -p third_party
  git clone https://github.com/ggml-org/llama.cpp.git third_party/llama.cpp
  git -C third_party/llama.cpp checkout 1b70350c92563449576bcbbf552994b1d1ed847d
fi

if ! rg -q 'yield from super\(\).modify_tensors\(data_torch, name, bid\)' third_party/llama.cpp/conversion/gpt2.py; then
  :
else
  (cd third_party/llama.cpp && git apply ../../patches/gpt2-skip-causal-mask.patch)
fi

.venv/bin/cmake -S third_party/llama.cpp -B build \
  -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_SERVER=OFF
.venv/bin/cmake --build build --target llama-completion llama-quantize -j4
.venv/bin/python third_party/llama.cpp/convert_hf_to_gguf.py \
  models/distilgpt2 --outfile models/distilgpt2-f16.gguf --outtype f16
build/bin/llama-quantize models/distilgpt2-f16.gguf \
  models/distilgpt2-Q4_K_M.gguf Q4_K_M
