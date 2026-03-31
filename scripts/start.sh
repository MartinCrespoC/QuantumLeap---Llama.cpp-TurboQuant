#!/bin/bash
# QuantumLeap — Optimized LLM Inference (coexists with Ollama)
# Linux start script. See start.bat (Windows), start_mac.sh (macOS).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
QLP_PORT="${API_PORT:-11435}"

export PATH="/opt/rocm/bin:/opt/cuda/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/opt/rocm/lib:/opt/cuda/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

cd "$PROJECT_ROOT"

echo ""
echo "  QuantumLeap v0.5.0 — Optimized LLM Inference"
echo "  Built on llama.cpp | TurboQuant Engine"
echo "  ─────────────────────────────────────────────"
echo ""

# Check engine
if [ ! -f "engine/llama.cpp/build/bin/llama-server" ]; then
  echo "  [!] Engine not built. Run: bash scripts/setup.sh"
  exit 1
fi

# Detect hardware
GPU_DETECTED=false

if command -v nvidia-smi &>/dev/null; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ -n "$GPU_NAME" ]; then
    echo "  GPU: ${GPU_NAME} (${VRAM} MB VRAM)"
    GPU_DETECTED=true
  fi
fi

if [ "$GPU_DETECTED" = false ] && command -v rocm-smi &>/dev/null; then
  GPU_INFO=$(rocm-smi --showproductname 2>/dev/null | grep "Card Series:" | head -1)
  if [ -n "$GPU_INFO" ]; then
    GPU_NAME=$(echo "$GPU_INFO" | sed 's/.*Card Series:[[:space:]]*//')
    VRAM_BYTES=$(cat /sys/class/drm/card*/device/mem_info_vram_total 2>/dev/null | head -1)
    if [ -n "$VRAM_BYTES" ]; then
      VRAM=$((VRAM_BYTES / 1024 / 1024))
      echo "  GPU: ${GPU_NAME} (${VRAM} MB VRAM)"
    else
      echo "  GPU: ${GPU_NAME}"
    fi
    GPU_DETECTED=true
  fi
fi

if [ "$GPU_DETECTED" = false ]; then
  GPU_NAME=$(lspci 2>/dev/null | grep -i "VGA.*AMD\|VGA.*ATI" | head -1 | sed 's/.*: //' | sed 's/\[//' | sed 's/\].*//')
  if [ -n "$GPU_NAME" ]; then
    echo "  GPU: ${GPU_NAME}"
    GPU_DETECTED=true
  fi
fi

if [ "$GPU_DETECTED" = false ]; then
  echo "  GPU: Not detected (CPU-only mode)"
fi
RAM_MB=$(free -m 2>/dev/null | awk '/^Mem:/{print $2}' || echo "unknown")
echo "  RAM: ${RAM_MB} MB"
echo ""

# Python venv
if [ ! -d ".venv" ] && [ ! -d "venv" ]; then
  echo "  Creating Python virtual environment..."
  python3 -m venv .venv
fi

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
fi

if ! python3 -c "import fastapi" 2>/dev/null; then
  echo "  Installing Python dependencies..."
  pip install --quiet -r api/requirements.txt
fi

MODEL_COUNT=$(find models/ -name "*.gguf" 2>/dev/null | wc -l)
if [ "$MODEL_COUNT" -eq 0 ]; then
  echo "  [!] No models found in models/"
  echo "      Use the Web UI to search and download models"
  echo ""
else
  echo "  Models: ${MODEL_COUNT} GGUF files found"
fi

# Stop any previous QuantumLeap instance on our port (NOT Ollama)
if command -v fuser &>/dev/null; then
  fuser -k "${QLP_PORT}/tcp" 2>/dev/null || true
elif command -v lsof &>/dev/null; then
  PID=$(lsof -ti:"${QLP_PORT}" 2>/dev/null || true)
  [ -n "$PID" ] && kill "$PID" 2>/dev/null || true
fi
# Also stop our internal llama-server on port 8081
if command -v fuser &>/dev/null; then
  fuser -k 8081/tcp 2>/dev/null || true
fi
sleep 1

echo ""
echo "  Starting QuantumLeap..."
echo ""
echo "  Web UI:     http://localhost:${QLP_PORT}"
echo "  Ollama API: http://localhost:${QLP_PORT}/api/"
echo "  OpenAI API: http://localhost:${QLP_PORT}/v1/"
echo ""
echo "  Ollama coexistence: Ollama stays on :11434, QuantumLeap on :${QLP_PORT}"
echo "  To replace Ollama:  API_PORT=11434 bash scripts/start.sh"
echo ""
echo "  Press Ctrl+C to stop"
echo ""

exec python3 api/server.py "$@"
