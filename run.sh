#!/bin/bash

# QuantumLeap Run Script
# Simple wrapper to start llama-server with ExpertFlow

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[QuantumLeap]${NC} $1"; }
error() { echo -e "${RED}[Error]${NC} $1"; exit 1; }

LLAMA_SERVER="engine/llama.cpp/build/bin/llama-server"

if [ ! -f "$LLAMA_SERVER" ]; then
    error "llama-server not found. Run ./build.sh first"
fi

if [ $# -eq 0 ]; then
    echo "Usage: ./run.sh <model_path> [options]"
    echo ""
    echo "Examples:"
    echo "  ./run.sh models/Qwen3.5-122B-A10B-UD-IQ2_XXS.gguf"
    echo "  ./run.sh models/model.gguf -ngl 10 -c 8192"
    echo ""
    echo "Common options:"
    echo "  -ngl N     : Number of layers to offload to GPU (default: 2)"
    echo "  -c N       : Context size (default: 4096)"
    echo "  --port N   : Server port (default: 8080)"
    echo "  --host IP  : Server host (default: 127.0.0.1)"
    echo ""
    exit 1
fi

MODEL="$1"
shift

if [ ! -f "$MODEL" ]; then
    error "Model file not found: $MODEL"
fi

# Default settings
NGL=2
CONTEXT=4096
PORT=8080
HOST="127.0.0.1"

# Parse additional options
EXTRA_ARGS=""
while [ $# -gt 0 ]; do
    case "$1" in
        -ngl)
            NGL="$2"
            shift 2
            ;;
        -c)
            CONTEXT="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

log "Starting QuantumLeap ExpertFlow server..."
echo "Model: $MODEL"
echo "GPU layers: $NGL"
echo "Context: $CONTEXT"
echo "Server: http://$HOST:$PORT"
echo ""

exec "$LLAMA_SERVER" \
    -m "$MODEL" \
    -ngl "$NGL" \
    -c "$CONTEXT" \
    --host "$HOST" \
    --port "$PORT" \
    $EXTRA_ARGS
