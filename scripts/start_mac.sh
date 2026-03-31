#!/bin/bash
# QuantumLeap — Optimized LLM Inference (coexists with Ollama)
# macOS start script (Apple Silicon + Intel).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
QLP_PORT="${API_PORT:-11435}"

cd "$PROJECT_ROOT"

echo ""
echo "  QuantumLeap v0.5.0 — Optimized LLM Inference"
echo "  Built on llama.cpp | TurboQuant Engine"
echo "  ─────────────────────────────────────────────"
echo ""

if [ ! -f "engine/llama.cpp/build/bin/llama-server" ]; then
  echo "  [!] Engine not built. Run: bash scripts/setup.sh"
  exit 1
fi

# Detect hardware (macOS)
CPU_BRAND=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown CPU")
echo "  CPU: ${CPU_BRAND}"

RAM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
RAM_GB=$(( RAM_BYTES / 1073741824 ))
echo "  RAM: ${RAM_GB} GB"

GPU_DETECTED=false
if echo "$CPU_BRAND" | grep -qi "Apple"; then
  CHIP_NAME=$(echo "$CPU_BRAND" | sed 's/.*Apple /Apple /')
  GPU_CORES=$(system_profiler SPDisplaysDataType 2>/dev/null | grep "Total Number of Cores" | head -1 | awk '{print $NF}')
  if [ -n "$GPU_CORES" ]; then
    echo "  GPU: ${CHIP_NAME} (${GPU_CORES} GPU cores, ${RAM_GB}GB unified memory)"
  else
    echo "  GPU: ${CHIP_NAME} (Metal, ${RAM_GB}GB unified memory)"
  fi
  GPU_DETECTED=true
fi

if [ "$GPU_DETECTED" = false ]; then
  GPU_NAME=$(system_profiler SPDisplaysDataType 2>/dev/null | grep "Chipset Model" | head -1 | sed 's/.*: //')
  if [ -n "$GPU_NAME" ]; then
    VRAM=$(system_profiler SPDisplaysDataType 2>/dev/null | grep "VRAM" | head -1 | sed 's/.*: //')
    [ -n "$VRAM" ] && echo "  GPU: ${GPU_NAME} (${VRAM})" || echo "  GPU: ${GPU_NAME}"
    GPU_DETECTED=true
  fi
fi

[ "$GPU_DETECTED" = false ] && echo "  GPU: Not detected (CPU-only mode)"
echo ""

# Python venv
if [ ! -d ".venv" ] && [ ! -d "venv" ]; then
  echo "  Creating Python virtual environment..."
  python3 -m venv .venv
fi

[ -f ".venv/bin/activate" ] && source .venv/bin/activate || { [ -f "venv/bin/activate" ] && source venv/bin/activate; }

if ! python3 -c "import fastapi" 2>/dev/null; then
  echo "  Installing Python dependencies..."
  pip install --quiet -r api/requirements.txt
fi

MODEL_COUNT=$(find models/ -name "*.gguf" 2>/dev/null | wc -l | tr -d ' ')
if [ "$MODEL_COUNT" -eq 0 ]; then
  echo "  [!] No models found in models/"
  echo "      Use the Web UI to search and download models"
  echo ""
else
  echo "  Models: ${MODEL_COUNT} GGUF files found"
fi

# Stop any previous QuantumLeap on our port (NOT Ollama on 11434)
PID=$(lsof -ti:"${QLP_PORT}" 2>/dev/null || true)
[ -n "$PID" ] && kill "$PID" 2>/dev/null || true
PID=$(lsof -ti:8081 2>/dev/null || true)
[ -n "$PID" ] && kill "$PID" 2>/dev/null || true
sleep 1

echo ""
echo "  Starting QuantumLeap..."
echo ""
echo "  Web UI:     http://localhost:${QLP_PORT}"
echo "  Ollama API: http://localhost:${QLP_PORT}/api/"
echo "  OpenAI API: http://localhost:${QLP_PORT}/v1/"
echo ""
echo "  Ollama coexistence: Ollama stays on :11434, QuantumLeap on :${QLP_PORT}"
echo "  To replace Ollama:  API_PORT=11434 bash scripts/start_mac.sh"
echo ""
echo "  Press Ctrl+C to stop"
echo ""

exec python3 api/server.py "$@"
