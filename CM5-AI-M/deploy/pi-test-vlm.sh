#!/usr/bin/env bash
# Called with the dedicated EnvironmentFile by systemd; no global shell/SSH configuration changes.
set -euo pipefail
runner=/home/mylin/CM5-AI-M/runtime/llama.cpp/build/bin/llama-server
: "${VLM_MODEL:?VLM_MODEL missing}"
: "${VLM_MMPROJ:?VLM_MMPROJ missing}"
: "${VLM_API_KEY:?VLM_API_KEY missing}"
if [[ "$VLM_API_KEY" == REPLACE_WITH_RANDOM_LOCAL_KEY || "$VLM_API_KEY" == CHANGE_ME ]]; then
  echo 'Replace the local VLM key before starting; do not print it in logs.' >&2
  exit 2
fi
[[ -x "$runner" ]] || { echo 'Dedicated ARM64 llama-server is missing or not executable' >&2; exit 2; }
[[ -s "$VLM_MODEL" && -s "$VLM_MMPROJ" ]] || { echo 'MiniCPM language/vision assets are missing or empty' >&2; exit 2; }
exec "$runner" -m "$VLM_MODEL" --mmproj "$VLM_MMPROJ" --alias minicpm-v \
  -c "${VLM_CTX:-2048}" -t "${VLM_THREADS:-3}" -ngl 0 --parallel 1 \
  --host 127.0.0.1 --port 8091 --api-key "$VLM_API_KEY"
