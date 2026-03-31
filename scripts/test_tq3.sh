#!/bin/bash
# Quick test of TurboQuant TQ3 KV cache compression
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

export PATH="/opt/cuda/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/opt/cuda/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

cd "$PROJECT_ROOT"

echo ""
echo "  ⚡ TurboQuant TQ3 Test"
echo "  ════════════════════════"
echo ""

MODEL=$(ls models/*.gguf 2>/dev/null | head -1)

if [ -z "$MODEL" ]; then
  echo "  ❌ No model found in models/"
  echo "  💡 Download: bash scripts/download_model.sh smollm"
  exit 1
fi

echo "  📦 Model: $(basename "$MODEL")"
echo "  🔧 KV Cache: TQ3 (3-bit, 4.9x compression)"
echo "  📊 Context: 8192 tokens"
echo ""
echo "  Starting server..."
echo ""

exec engine/llama.cpp/build/bin/llama-server \
  -m "$MODEL" \
  --cache-type-k tq3 \
  --cache-type-v tq3 \
  -c 8192 \
  -ngl 99 \
  --host 0.0.0.0 \
  --port 11435
