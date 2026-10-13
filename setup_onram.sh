#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

LLAMA_CPP_COMMIT=1b70350c92563449576bcbbf552994b1d1ed847d
LLAMA_DIR=third_party/llama.cpp

if [[ ! -d "$LLAMA_DIR/.git" ]]; then
  mkdir -p third_party
  git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi
git -C "$LLAMA_DIR" checkout --detach "$LLAMA_CPP_COMMIT"

if grep -Fq 'yield from super().modify_tensors(data_torch, name, bid)' "$LLAMA_DIR/conversion/gpt2.py"; then
  git -C "$LLAMA_DIR" apply ../../patches/gpt2-skip-causal-mask.patch
fi

.venv/bin/cmake -S "$LLAMA_DIR" -B build \
  -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_SERVER=ON
.venv/bin/cmake --build build --target llama-server llama-completion llama-quantize -j4
