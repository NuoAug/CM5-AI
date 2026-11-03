#!/usr/bin/env bash
# 在 CM5 上构建 llama.cpp（CPU）。默认只构建 llama-server，避免 4GB 内存下全量构建 OOM。
set -euo pipefail

REF="${LLAMA_CPP_REF:-b11551}"
SRC="${LLAMA_CPP_SRC:-/opt/cm5-ai/runtime/llama.cpp-src}"
BUILD="${LLAMA_CPP_BUILD:-/opt/cm5-ai/runtime/llama.cpp/build}"
JOBS="${LLAMA_CPP_JOBS:-2}"

if [[ ! -d "$SRC/.git" ]]; then
  git clone https://github.com/ggml-org/llama.cpp "$SRC"
fi
git -C "$SRC" fetch --tags
git -C "$SRC" checkout "$REF"

cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DGGML_OPENMP=ON \
  -DLLAMA_CURL=OFF

cmake --build "$BUILD" --target llama-server -j "$JOBS"

echo "llama-server: $BUILD/bin/llama-server"
echo "如果 MiniCPM-V 4.7 加载失败，先确认该版本不低于 b11551；低于时回退 Qwen3-VL-2B。"
