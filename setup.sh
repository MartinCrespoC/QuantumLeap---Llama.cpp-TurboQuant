#!/bin/bash
set -e

# TurboQuant Setup Script
# Automatic installation with hardware detection and validation

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BOLD}${BLUE}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║           ⚛️  QuantumLeap v0.6.0 Setup                    ║"
echo "║   ExpertFlow Phase 3: 130% Faster MoE Inference           ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if running as root
if [ "$EUID" -eq 0 ]; then
   echo -e "${RED}⚠️  Do not run this script as root${NC}"
   exit 1
fi

# ─── Hardware Detection ───────────────────────────────────────────────────────

echo -e "${BOLD}🔍 Detecting Hardware...${NC}"

# Detect GPU — NVIDIA, AMD ROCm, or Apple Silicon Metal
HAS_NVIDIA=false
HAS_AMD=false
HAS_METAL=false
VRAM_MB=0

if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    HAS_NVIDIA=true
    echo -e "  ${GREEN}✓${NC} GPU: $GPU_NAME (${VRAM_MB}MB VRAM, CUDA $CUDA_VERSION)"
elif command -v rocm-smi &> /dev/null || [ -d "/opt/rocm" ]; then
    GPU_NAME=$(rocm-smi --showproductname 2>/dev/null | grep "Card Series" | sed 's/.*: //' || lspci 2>/dev/null | grep -i "VGA.*AMD" | sed 's/.*\[//' | sed 's/\].*//' | head -1)
    VRAM_BYTES=$(rocm-smi --showmeminfo vram 2>/dev/null | grep "Total Memory" | awk '{print $NF}')
    if [ -n "$VRAM_BYTES" ] && [ "$VRAM_BYTES" -gt 0 ] 2>/dev/null; then
        VRAM_MB=$((VRAM_BYTES / 1048576))
    fi
    HAS_AMD=true
    echo -e "  ${GREEN}✓${NC} GPU: $GPU_NAME (${VRAM_MB}MB VRAM, ROCm)"
elif [ "$(uname -s)" = "Darwin" ]; then
    CPU_BRAND=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "")
    if echo "$CPU_BRAND" | grep -qi "Apple"; then
        CHIP_NAME=$(echo "$CPU_BRAND" | sed 's/.*Apple /Apple /')
        RAM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
        VRAM_MB=$((RAM_BYTES / 1048576))
        HAS_METAL=true
        GPU_CORES=$(system_profiler SPDisplaysDataType 2>/dev/null | grep "Total Number of Cores" | head -1 | awk '{print $NF}')
        if [ -n "$GPU_CORES" ]; then
            echo -e "  ${GREEN}✓${NC} GPU: ${CHIP_NAME} (${GPU_CORES} GPU cores, $((VRAM_MB / 1024))GB unified memory, Metal)"
        else
            echo -e "  ${GREEN}✓${NC} GPU: ${CHIP_NAME} ($((VRAM_MB / 1024))GB unified memory, Metal)"
        fi
    else
        echo -e "  ${YELLOW}⚠${NC} Intel Mac detected — check for discrete GPU"
    fi
else
    echo -e "  ${YELLOW}⚠${NC} No GPU detected (CPU-only mode)"
fi

# Detect RAM
RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
echo -e "  ${GREEN}✓${NC} RAM: ${RAM_GB}GB"

# Detect CPU
CPU_MODEL=$(lscpu | grep "Model name" | cut -d: -f2 | xargs)
CPU_CORES=$(nproc)
echo -e "  ${GREEN}✓${NC} CPU: $CPU_MODEL ($CPU_CORES cores)"

# Check AVX support
if grep -q avx512 /proc/cpuinfo; then
    AVX_SUPPORT="AVX-512"
elif grep -q avx2 /proc/cpuinfo; then
    AVX_SUPPORT="AVX2"
elif grep -q avx /proc/cpuinfo; then
    AVX_SUPPORT="AVX"
else
    AVX_SUPPORT="None"
fi
echo -e "  ${GREEN}✓${NC} SIMD: $AVX_SUPPORT"

# ─── Dependency Check ─────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}📦 Checking Dependencies...${NC}"

MISSING_DEPS=()

# Check Python 3.10+
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 10 ]; then
        echo -e "  ${GREEN}✓${NC} Python $PYTHON_VERSION"
    else
        echo -e "  ${RED}✗${NC} Python 3.10+ required (found $PYTHON_VERSION)"
        MISSING_DEPS+=("python3.10+")
    fi
else
    echo -e "  ${RED}✗${NC} Python 3 not found"
    MISSING_DEPS+=("python3")
fi

# Check CMake
if command -v cmake &> /dev/null; then
    CMAKE_VERSION=$(cmake --version | head -1 | awk '{print $3}')
    echo -e "  ${GREEN}✓${NC} CMake $CMAKE_VERSION"
else
    echo -e "  ${RED}✗${NC} CMake not found"
    MISSING_DEPS+=("cmake")
fi

# Check Ninja
if command -v ninja &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} Ninja build system"
else
    echo -e "  ${YELLOW}⚠${NC} Ninja not found (will use make, slower builds)"
fi

# Check Git
if command -v git &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} Git"
else
    echo -e "  ${RED}✗${NC} Git not found"
    MISSING_DEPS+=("git")
fi

# Check CUDA toolkit (if NVIDIA GPU present)
if [ "$HAS_NVIDIA" = true ]; then
    if command -v nvcc &> /dev/null; then
        NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d, -f1)
        echo -e "  ${GREEN}✓${NC} CUDA Toolkit $NVCC_VERSION"
    else
        echo -e "  ${YELLOW}⚠${NC} CUDA Toolkit not found (required for GPU acceleration)"
        MISSING_DEPS+=("cuda-toolkit")
    fi
fi

# Exit if missing critical dependencies
if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo ""
    echo -e "${RED}Missing dependencies: ${MISSING_DEPS[*]}${NC}"
    echo ""
    echo "Install them with:"
    echo "  Ubuntu/Debian: sudo apt install python3 python3-pip cmake ninja-build git"
    if [ "$HAS_NVIDIA" = true ]; then
        echo "  CUDA Toolkit: https://developer.nvidia.com/cuda-downloads"
    fi
    exit 1
fi

# ─── Python Virtual Environment ───────────────────────────────────────────────

echo ""
echo -e "${BOLD}🐍 Setting up Python Environment...${NC}"

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "  ${GREEN}✓${NC} Created virtual environment"
else
    echo -e "  ${GREEN}✓${NC} Virtual environment exists"
fi

source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip > /dev/null 2>&1
pip install -r api/requirements.txt > /dev/null 2>&1
echo -e "  ${GREEN}✓${NC} Installed Python dependencies"

# ─── Build llama.cpp ──────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}🔨 Building llama.cpp with optimizations...${NC}"

cd engine/llama.cpp
SKIP_BUILD=false

# Detect if already built
if [ -f "build/bin/llama-server" ] && [ -f "build/bin/llama-cli" ]; then
    echo -e "  ${YELLOW}⚠${NC} llama.cpp already built"
    read -p "  Rebuild? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        cd ../..
        echo -e "  ${GREEN}✓${NC} Skipped rebuild"
        SKIP_BUILD=true
    fi
fi

if [ "$SKIP_BUILD" != true ]; then
    rm -rf build
    mkdir -p build
    cd build

    # Configure CMake with optimizations
    CMAKE_ARGS=(
        -DCMAKE_BUILD_TYPE=Release
        -DGGML_NATIVE=ON
        -DLLAMA_EXPERTFLOW=ON
        -DLLAMA_BUILD_SERVER=ON
    )

    # Enable AVX if supported
    if [ "$AVX_SUPPORT" != "None" ]; then
        CMAKE_ARGS+=(-DGGML_AVX=ON)
        if [ "$AVX_SUPPORT" = "AVX2" ] || [ "$AVX_SUPPORT" = "AVX-512" ]; then
            CMAKE_ARGS+=(-DGGML_AVX2=ON)
        fi
        if [ "$AVX_SUPPORT" = "AVX-512" ]; then
            CMAKE_ARGS+=(
                -DGGML_AVX512=ON
                -DGGML_AVX512_VBMI=ON
                -DGGML_AVX512_VNNI=ON
            )
        fi
    fi

    # Enable GPU acceleration based on detected hardware
    if [ "$HAS_NVIDIA" = true ] && command -v nvcc &> /dev/null; then
        CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
        CMAKE_ARGS+=(
            -DGGML_CUDA=ON
            -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH
            -DGGML_CUDA_FA_ALL_QUANTS=ON
        )
        echo -e "  ${GREEN}✓${NC} CUDA enabled (arch $CUDA_ARCH)"
        echo -e "  ${GREEN}✓${NC} ExpertFlow Phase 3 enabled (MoE optimization)"
    elif [ "$HAS_AMD" = true ]; then
        CMAKE_ARGS+=(-DGGML_HIP=ON)
        if [ -d "/opt/rocm" ]; then
            export PATH="/opt/rocm/bin:$PATH"
            export HIPCXX="$(hipconfig -l)/clang"
            export HIP_PATH="$(hipconfig -R)"
        fi
        echo -e "  ${GREEN}✓${NC} ROCm/HIP enabled (AMD GPU)"
        echo -e "  ${GREEN}✓${NC} ExpertFlow Phase 3 enabled (MoE optimization)"
    elif [ "$HAS_METAL" = true ]; then
        CMAKE_ARGS+=(-DGGML_METAL=ON)
        echo -e "  ${GREEN}✓${NC} Metal enabled (Apple Silicon)"
        echo -e "  ${GREEN}✓${NC} ExpertFlow Phase 3 enabled (MoE optimization)"
    else
        echo -e "  ${GREEN}✓${NC} ExpertFlow Phase 3 enabled (CPU-only, MoE optimization)"
    fi

    # Use Ninja if available
    if command -v ninja &> /dev/null; then
        CMAKE_ARGS+=(-G Ninja)
        BUILD_CMD="ninja"
    else
        BUILD_CMD="make -j$CPU_CORES"
    fi

    echo -e "  ${BLUE}ℹ${NC} Building with: ${CMAKE_ARGS[*]}"
    cmake .. "${CMAKE_ARGS[@]}" > /dev/null 2>&1

    echo -e "  ${BLUE}ℹ${NC} Compiling (this may take 5-10 minutes)..."
    $BUILD_CMD llama-server llama-cli llama-quantize > /dev/null 2>&1

    cd ../../..
    echo -e "  ${GREEN}✓${NC} Built llama.cpp with optimizations"
fi

# ─── Create directories ───────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}📁 Creating directories...${NC}"

mkdir -p models benchmarks backups
echo -e "  ${GREEN}✓${NC} Created models/, benchmarks/, backups/"

# ─── Hardware Recommendations ─────────────────────────────────────────────────

echo ""
echo -e "${BOLD}💡 Recommendations for Your Hardware:${NC}"
echo ""

if [ "$HAS_NVIDIA" = true ] || [ "$HAS_AMD" = true ]; then
    VRAM_GB=$((VRAM_MB / 1024))

    if [ $VRAM_GB -le 6 ]; then
        echo -e "${YELLOW}6GB VRAM Setup (ExpertFlow Phase 3):${NC}"
        echo "  • ${BOLD}MoE 122B-A10B${NC} (IQ2_XXS) → ${GREEN}4.34 tok/s${NC} ⭐ (130% faster!)"
        echo "  • MoE 35B-A3B (IQ2_XXS) → 15+ tok/s"
        echo "  • SmolLM2 1.7B (Q4_K_M, full GPU) → 120 tok/s"
        echo "  • Dense 40B (IQ2_XXS) → 2.95 tok/s"
        echo ""
        echo "  Download 122B MoE model:"
        echo "    curl -L -o models/Qwen3.5-122B-A10B-UD-IQ2_XXS.gguf \\"
        echo "      'https://huggingface.co/unsloth/Qwen3.5-122B-A10B-GGUF/resolve/main/Qwen3.5-122B-A10B-UD-IQ2_XXS.gguf'"
    elif [ $VRAM_GB -le 12 ]; then
        echo -e "${GREEN}8-12GB VRAM Setup (ExpertFlow Phase 3):${NC}"
        echo "  • ${BOLD}MoE 122B-A10B${NC} → ${GREEN}6-8 tok/s${NC} (more layers on GPU)"
        echo "  • Dense 27B: Q4_K_M full GPU → 8-10 tok/s"
        echo "  • MoE 35B-A3B: More GPU layers → 20-25 tok/s"
        echo "  • 13B models: Q6_K/Q8_0 for best quality"
    elif [ $VRAM_GB -le 24 ]; then
        echo -e "${GREEN}24GB VRAM Setup (ExpertFlow Phase 3):${NC}"
        echo "  • ${BOLD}MoE 122B-A10B${NC} → ${GREEN}12-18 tok/s${NC} ⭐⭐ (6-9× baseline!)"
        echo "  • Dense 70B: Q4_K_M possible"
        echo "  • MoE models: Excellent performance"
        echo "  • Any 27B model: Full GPU with high quality quants"
    else
        echo -e "${GREEN}48GB+ VRAM Setup (ExpertFlow Phase 3):${NC}"
        echo "  • ${BOLD}MoE 122B-A10B${NC} → ${GREEN}68-85 tok/s${NC} ⭐⭐⭐ (15-19× baseline!)"
        echo "  • Dense 70B: Q8_0 full quality"
        echo "  • MoE models: Maximum performance"
        echo "  • Production-ready for high-volume use"
    fi
else
    echo -e "${YELLOW}CPU-Only Setup (ExpertFlow Phase 3):${NC}"
    echo "  • MoE 122B-A10B (IQ2_XXS) → 1.89 tok/s (baseline)"
    echo "  • MoE models recommended (3B-10B active params)"
    echo "  • Use IQ2_XXS or Q2_K quantizations"
    echo "  • Expected: 9-12 tok/s on MoE 35B-A3B"
fi

echo ""
echo -e "RAM: ${RAM_GB}GB"
if [ $RAM_GB -lt 16 ]; then
    echo -e "  ${YELLOW}⚠${NC} 16GB+ RAM recommended for large models"
elif [ $RAM_GB -ge 24 ]; then
    echo -e "  ${GREEN}✓${NC} Excellent for large model offloading"
fi

# ─── Model Management Guide ───────────────────────────────────────────────────

echo ""
echo -e "${BOLD}📚 Model Management:${NC}"
echo ""
echo "1. Download models to models/ directory"
echo "2. Quantization types:"
echo "   • ${GREEN}Requantizable${NC}: Q8_0, Q6_K, Q5_K, Q4_K, Q3_K, Q2_K"
echo "     → Use Web UI Requantize feature"
echo "   • ${YELLOW}Pre-quantized only${NC}: IQ1_S, IQ2_XXS, IQ2_M, IQ3_XXS, IQ4_NL"
echo "     → Download from HuggingFace (unsloth/bartowski repos)"
echo ""
echo "3. Model recommendations:"
echo "   • ${BOLD}MoE models${NC}: 3x faster than dense for same intelligence"
echo "   • ${BOLD}IQ2_XXS${NC}: Best for MoE on constrained hardware"
echo "   • ${BOLD}Q2_K${NC}: Best requantizable extreme compression"
echo ""
echo "See memory.md for complete optimization guide"

# ─── Final Steps ──────────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}${GREEN}✓ Setup Complete!${NC}"
echo ""
echo "Start TurboQuant:"
echo "  ${BOLD}bash scripts/start.sh${NC}"
echo ""
echo "Or manually:"
echo "  ${BOLD}source venv/bin/activate${NC}"
echo "  ${BOLD}python3 api/server.py${NC}"
echo ""
echo "Web UI will be available at: ${BLUE}http://localhost:11435${NC}"
echo ""
echo -e "${YELLOW}Note:${NC} First run will take longer as models are loaded"
echo ""
