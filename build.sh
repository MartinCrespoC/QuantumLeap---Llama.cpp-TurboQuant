#!/bin/bash
set -e

# QuantumLeap Build Script
# Automatically detects GPU (NVIDIA/AMD) and builds ExpertFlow + llama.cpp

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[QuantumLeap]${NC} $1"; }
warn() { echo -e "${YELLOW}[Warning]${NC} $1"; }
error() { echo -e "${RED}[Error]${NC} $1"; exit 1; }

log "QuantumLeap ExpertFlow Phase 3 Build Script"
echo "=============================================="

# Detect GPU
HAS_NVIDIA=false
HAS_AMD=false

if command -v nvidia-smi &> /dev/null; then
    HAS_NVIDIA=true
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    log "Detected NVIDIA GPU: $GPU_NAME"
elif command -v rocm-smi &> /dev/null || [ -d "/opt/rocm" ]; then
    HAS_AMD=true
    if command -v rocminfo &> /dev/null; then
        GPU_NAME=$(rocminfo | grep "Marketing Name" | head -1 | cut -d: -f2 | xargs)
        log "Detected AMD GPU: $GPU_NAME"
    else
        log "Detected AMD GPU (ROCm installed)"
    fi
else
    warn "No GPU detected - building CPU-only version"
fi

# Check dependencies
log "Checking dependencies..."
MISSING_DEPS=()

if ! command -v cmake &> /dev/null; then
    MISSING_DEPS+=("cmake")
fi

if ! command -v ninja &> /dev/null && ! command -v make &> /dev/null; then
    MISSING_DEPS+=("ninja or make")
fi

if ! command -v git &> /dev/null; then
    MISSING_DEPS+=("git")
fi

if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
    error "Missing dependencies: ${MISSING_DEPS[*]}\nInstall with: sudo apt install cmake ninja-build git (Ubuntu/Debian)"
fi

# Build ExpertFlow core library
log "Building ExpertFlow core library..."
cd core

if [ -d "build_ef" ]; then
    log "Cleaning previous build..."
    rm -rf build_ef
fi

CMAKE_ARGS="-B build_ef -GNinja -DCMAKE_BUILD_TYPE=Release"

if $HAS_NVIDIA; then
    log "Configuring for NVIDIA CUDA..."
    CMAKE_ARGS="$CMAKE_ARGS -DTURBOQUANT_GPU_BACKEND=cuda"
elif $HAS_AMD; then
    log "Configuring for AMD ROCm/HIP..."
    # Detect GPU architecture
    if command -v rocminfo &> /dev/null; then
        GPU_ARCH=$(rocminfo | grep "Name:" | grep "gfx" | head -1 | awk '{print $2}')
        if [ -n "$GPU_ARCH" ]; then
            log "Detected GPU architecture: $GPU_ARCH"
        else
            GPU_ARCH="gfx1010"
            warn "Could not detect GPU arch, using default: $GPU_ARCH"
        fi
    else
        GPU_ARCH="gfx1010"
        warn "rocminfo not found, using default arch: $GPU_ARCH"
    fi
    CMAKE_ARGS="$CMAKE_ARGS -DTURBOQUANT_GPU_BACKEND=hip"
else
    log "Configuring for CPU-only..."
    CMAKE_ARGS="$CMAKE_ARGS -DTURBOQUANT_GPU_BACKEND=none"
fi

cmake $CMAKE_ARGS || error "CMake configuration failed"
ninja -C build_ef test_expertflow || error "ExpertFlow build failed"

log "✓ ExpertFlow core library built successfully"
cd ..

# Build llama.cpp with ExpertFlow
log "Building llama.cpp with ExpertFlow Phase 3..."
cd engine/llama.cpp

if [ -d "build" ]; then
    log "Cleaning previous llama.cpp build..."
    rm -rf build
fi

CMAKE_ARGS="-B build -GNinja -DCMAKE_BUILD_TYPE=Release -DLLAMA_EXPERTFLOW=ON -DLLAMA_BUILD_SERVER=ON"

if $HAS_NVIDIA; then
    log "Configuring llama.cpp for NVIDIA CUDA..."
    CMAKE_ARGS="$CMAKE_ARGS -DGGML_CUDA=ON"
    if command -v nvcc &> /dev/null; then
        CUDA_ARCH=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1\2/p')
        if [ -n "$CUDA_ARCH" ]; then
            CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH"
        fi
    fi
elif $HAS_AMD; then
    log "Configuring llama.cpp for AMD ROCm/HIP..."
    export HIPCXX="/opt/rocm/llvm/bin/clang++"
    export HIP_PATH="/opt/rocm"
    CMAKE_ARGS="$CMAKE_ARGS -DGGML_HIP=ON -DGPU_TARGETS=$GPU_ARCH"
fi

cmake $CMAKE_ARGS || error "llama.cpp CMake configuration failed"
ninja -C build llama-server || error "llama.cpp build failed"

log "✓ llama-server built successfully"
cd ../..

# Verify binaries
log "Verifying binaries..."
LLAMA_SERVER="engine/llama.cpp/build/bin/llama-server"
EXPERTFLOW_TEST="core/build_ef/test_expertflow"

if [ ! -f "$LLAMA_SERVER" ]; then
    error "llama-server binary not found at $LLAMA_SERVER"
fi

if [ ! -f "$EXPERTFLOW_TEST" ]; then
    error "test_expertflow binary not found at $EXPERTFLOW_TEST"
fi

# Check GPU backend
if $HAS_NVIDIA || $HAS_AMD; then
    log "Checking GPU backend integration..."
    if $HAS_AMD; then
        if ldd "$LLAMA_SERVER" | grep -q "libggml-hip"; then
            log "✓ HIP backend linked successfully"
        else
            warn "HIP backend not detected in binary"
        fi
    elif $HAS_NVIDIA; then
        if ldd "$LLAMA_SERVER" | grep -q "cuda"; then
            log "✓ CUDA backend linked successfully"
        else
            warn "CUDA backend not detected in binary"
        fi
    fi
fi

# Success summary
echo ""
echo "=============================================="
log "Build completed successfully!"
echo "=============================================="
echo ""
echo "Binaries:"
echo "  - llama-server: $LLAMA_SERVER"
echo "  - test_expertflow: $EXPERTFLOW_TEST"
echo ""
echo "Next steps:"
echo "  1. Download a MoE model (e.g., Qwen3.5-122B-A10B-UD-IQ2_XXS)"
echo "  2. Run: ./run.sh <model_path>"
echo "  3. Or manually: $LLAMA_SERVER -m <model> -ngl 2"
echo ""
if $HAS_NVIDIA || $HAS_AMD; then
    echo "GPU detected - ExpertFlow will use hardware acceleration"
else
    echo "CPU-only build - consider adding a GPU for better performance"
fi
echo ""
