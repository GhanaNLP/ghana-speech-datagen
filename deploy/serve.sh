#!/usr/bin/env bash
# Launch the Ghana NLP VoxCPM2 TTS server with vLLM-Omni.
#
# Prereqs (see deploy/README.md): a Linux GPU box with vLLM-Omni installed.
# The model is pulled from the Hugging Face Hub on first run and cached.
#
# Config via env vars (all optional):
#   MODEL        HF repo id or local path   (default: ghananlpcommunity/VoxCPM2-Ghana)
#   PORT         server port                (default: 8000)
#   API_KEY      require this bearer token  (default: none / open)
#   GPU_MEM      gpu-memory-utilization     (default: 0.9)
#   HF_TOKEN     HF token for faster/gated downloads (optional)
set -euo pipefail

MODEL="${MODEL:-ghananlpcommunity/VoxCPM2-Ghana}"
PORT="${PORT:-8000}"
GPU_MEM="${GPU_MEM:-0.9}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

args=(
  serve "$MODEL"
  --omni
  --deploy-config "$HERE/voxcpm2.yaml"
  --served-model-name voxcpm2
  --host 0.0.0.0 --port "$PORT"
  --gpu-memory-utilization "$GPU_MEM"
  --trust-remote-code
  --allowed-origins '["*"]'
  --allowed-methods '["*"]'
  --allowed-headers '["*"]'
)
if [ -n "${API_KEY:-}" ]; then
  args+=(--api-key "$API_KEY")
fi

echo "Starting VoxCPM2 TTS server: model=$MODEL port=$PORT" >&2
exec vllm "${args[@]}"
