#!/bin/bash
# TurboQuant Setup Script — Linux / macOS
# Builds llama.cpp with TurboQuant + CUDA, installs Python deps, downloads test model
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENGINE_DIR="$PROJECT_ROOT/engine/llama.cpp"
MODELS_DIR="$PROJECT_ROOT/models"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${CYAN}[TurboQuant]${NC} $*"; }
ok()   { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[✗]${NC} $*"; }

# ─── Detect OS ──────────────────────────────────────────────────────────────────

detect_os() {
  case "$(uname -s)" in
    Linux*)   OS="linux" ;;
    Darwin*)  OS="macos" ;;
    MINGW*|MSYS*|CYGWIN*) OS="windows" ;;
    *)        OS="unknown" ;;
  esac
  ARCH="$(uname -m)"
  log "OS: $OS, Arch: $ARCH"
}

# ─── Install Dependencies ──────────────────────────────────────────────────────

install_deps() {
  log "Checking build dependencies..."

  if ! command -v cmake &>/dev/null; then
    warn "cmake not found, installing..."
    case "$OS" in
      linux)
        if command -v pacman &>/dev/null; then
          sudo pacman -S --noconfirm cmake
        elif command -v apt &>/dev/null; then
          sudo apt-get update && sudo apt-get install -y cmake build-essential
        elif command -v dnf &>/dev/null; then
          sudo dnf install -y cmake gcc-c++
        fi
        ;;
      macos)
        brew install cmake
        ;;
    esac
  fi
  ok "cmake $(cmake --version | head -1 | awk '{print $3}')"

  if ! command -v ninja &>/dev/null; then
    warn "ninja not found, installing..."
    case "$OS" in
      linux)
        if command -v pacman &>/dev/null; then sudo pacman -S --noconfirm ninja; fi
        if command -v apt &>/dev/null; then sudo apt-get install -y ninja-build; fi
        ;;
      macos) brew install ninja ;;
    esac
  fi
  ok "ninja found"

  if ! command -v python3 &>/dev/null; then
    err "python3 not found. Install Python 3.10+ first."
    exit 1
  fi
  ok "python3 $(python3 --version | awk '{print $2}')"
}

# ─── Detect GPU ─────────────────────────────────────────────────────────────────

detect_gpu() {
  HAS_CUDA=false
  HAS_METAL=false
  CUDA_ARCH=""

  if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
    if [ -n "$GPU_CC" ]; then
      HAS_CUDA=true
      CUDA_ARCH="$GPU_CC"
      ok "NVIDIA GPU: $GPU_NAME (Compute $GPU_CC)"
    fi
  fi

  if [ "$OS" = "macos" ]; then
    HAS_METAL=true
    ok "Apple Metal GPU detected"
  fi

  # Check for CUDA toolkit
  if $HAS_CUDA; then
    NVCC_PATH=""
    for p in /opt/cuda/bin/nvcc /usr/local/cuda/bin/nvcc /usr/bin/nvcc; do
      if [ -f "$p" ]; then NVCC_PATH="$p"; break; fi
    done
    if [ -z "$NVCC_PATH" ]; then
      warn "CUDA toolkit not found. Building CPU-only."
      warn "Install CUDA toolkit for GPU acceleration."
      HAS_CUDA=false
    else
      ok "CUDA toolkit: $($NVCC_PATH --version | grep release | awk '{print $6}')"
    fi
  fi
}

# ─── Clone Engine ───────────────────────────────────────────────────────────────

clone_engine() {
  if [ -d "$ENGINE_DIR" ]; then
    ok "Engine directory exists (using ik_llama.cpp)"
    return
  fi

  log "Cloning ik_llama.cpp (ikawrakow fork with optimizations)..."
  mkdir -p "$(dirname "$ENGINE_DIR")"
  git clone --depth 1 https://github.com/ikawrakow/ik_llama.cpp.git "$ENGINE_DIR"
  ok "Engine cloned"
}

# ─── Build Engine ───────────────────────────────────────────────────────────────

build_engine() {
  log "Building llama.cpp with ExpertFlow Phase 3..."

  CMAKE_ARGS="-G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_EXPERTFLOW=ON -DLLAMA_BUILD_SERVER=ON"

  if $HAS_CUDA; then
    export PATH="/opt/cuda/bin:/usr/local/cuda/bin:$PATH"
    CMAKE_ARGS="$CMAKE_ARGS -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH"
    log "Building with CUDA (arch $CUDA_ARCH) + ExpertFlow Phase 3"
  elif $HAS_METAL; then
    CMAKE_ARGS="$CMAKE_ARGS -DGGML_METAL=ON"
    log "Building with Metal + ExpertFlow Phase 3"
  elif command -v rocm-smi &>/dev/null || [ -d "/opt/rocm" ]; then
    # AMD GPU with ROCm
    CMAKE_ARGS="$CMAKE_ARGS -DGGML_HIP=ON"
    if [ -d "/opt/rocm" ]; then
      export PATH="/opt/rocm/bin:$PATH"
      export HIPCXX="$(hipconfig -l)/clang"
      export HIP_PATH="$(hipconfig -R)"
    fi
    log "Building with ROCm (AMD GPU) + ExpertFlow Phase 3"
  else
    log "Building CPU-only + ExpertFlow Phase 3"
  fi

  cd "$ENGINE_DIR"
  cmake -B build $CMAKE_ARGS
  cmake --build build -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

  # Verify build
  SERVER_BIN="$ENGINE_DIR/build/bin/llama-server"
  if [ ! -f "$SERVER_BIN" ]; then
    SERVER_BIN="$ENGINE_DIR/build/llama-server"
  fi
  if [ -f "$SERVER_BIN" ]; then
    ok "llama-server built: $SERVER_BIN"
  else
    err "Build failed: llama-server not found"
    exit 1
  fi
}

# ─── Setup Python ───────────────────────────────────────────────────────────────

setup_python() {
  log "Setting up Python environment..."

  cd "$PROJECT_ROOT"
  if [ ! -d ".venv" ]; then
    python3 -m venv .venv
  fi
  source .venv/bin/activate

  pip install --quiet --upgrade pip
  pip install --quiet -r api/requirements.txt

  ok "Python environment ready"
}

# ─── Download Test Model ────────────────────────────────────────────────────────

download_model() {
  mkdir -p "$MODELS_DIR"

  # SmolLM2 1.7B — small, fast, good for testing
  MODEL_URL="https://huggingface.co/bartowski/SmolLM2-1.7B-Instruct-GGUF/resolve/main/SmolLM2-1.7B-Instruct-Q4_K_M.gguf"
  MODEL_FILE="$MODELS_DIR/SmolLM2-1.7B-Instruct-Q4_K_M.gguf"

  if [ -f "$MODEL_FILE" ]; then
    ok "Test model already downloaded"
    return
  fi

  log "Downloading SmolLM2 1.7B (Q4_K_M, ~1GB)..."
  if command -v wget &>/dev/null; then
    wget -q --show-progress -O "$MODEL_FILE" "$MODEL_URL"
  elif command -v curl &>/dev/null; then
    curl -L --progress-bar -o "$MODEL_FILE" "$MODEL_URL"
  else
    err "Neither wget nor curl found. Download manually:"
    err "  $MODEL_URL"
    err "  → $MODEL_FILE"
    return
  fi

  ok "Model downloaded: $MODEL_FILE ($(du -h "$MODEL_FILE" | cut -f1))"
}

# ─── Main ───────────────────────────────────────────────────────────────────────

main() {
  echo ""
  echo -e "${CYAN}  ⚡ QuantumLeap v0.6.0 Setup${NC}"
  echo -e "${CYAN}  ExpertFlow Phase 3: 130% Faster MoE Inference${NC}"
  echo ""

  detect_os
  install_deps
  detect_gpu
  clone_engine
  build_engine
  setup_python
  download_model

  echo ""
  echo -e "${GREEN}  ✓ Setup complete!${NC}"
  echo ""
  echo "  ExpertFlow Phase 3 enabled for MoE models:"
  echo "    • 122B MoE: 4.34 tok/s on 6GB VRAM (130% faster!)"
  echo "    • 24GB VRAM: 12-18 tok/s projected"
  echo "    • 48GB VRAM: 68-85 tok/s projected"
  echo ""
  echo "  Start the server:"
  echo -e "    ${CYAN}cd $PROJECT_ROOT && source .venv/bin/activate${NC}"
  echo -e "    ${CYAN}python api/server.py${NC}"
  echo ""
  echo "  Then open: http://localhost:11435"
  echo ""
  echo "  API endpoints (Ollama-compatible):"
  echo "    GET  /api/tags          — List models"
  echo "    POST /api/chat          — Chat completions"
  echo "    POST /api/generate      — Text generation"
  echo "    POST /v1/chat/completions — OpenAI-compatible"
  echo ""
}

main "$@"
