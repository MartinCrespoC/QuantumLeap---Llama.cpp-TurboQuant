#!/bin/bash
# Benchmark script to verify ExpertFlow integration end-to-end

set -e

MODEL_PATH="${1:-models/Qwen3.5-122B-A10B-UD-IQ2_XXS.gguf}"
NGL="${2:-2}"
CTX="${3:-512}"

echo "=== ExpertFlow End-to-End Benchmark ==="
echo "Model: $MODEL_PATH"
echo "GPU layers: $NGL"
echo "Context: $CTX"
echo ""

# Start llama-server with ExpertFlow
echo "Starting llama-server with ExpertFlow..."
./engine/llama.cpp/build/bin/llama-server \
  -m "$MODEL_PATH" \
  -ngl "$NGL" \
  -c "$CTX" \
  --port 8082 \
  --log-disable &

SERVER_PID=$!
sleep 10

# Wait for server to be ready
echo "Waiting for server to load model..."
for i in {1..30}; do
  if curl -s http://localhost:8082/health > /dev/null 2>&1; then
    echo "Server ready!"
    break
  fi
  sleep 2
done

# Run benchmark
echo ""
echo "Running benchmark (100 tokens)..."
START_TIME=$(date +%s.%N)

RESPONSE=$(curl -s http://localhost:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Write a short poem about AI."}],
    "max_tokens": 100,
    "stream": false
  }')

END_TIME=$(date +%s.%N)
DURATION=$(echo "$END_TIME - $START_TIME" | bc)

# Extract tokens and calculate speed
TOKENS=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['usage']['completion_tokens'])" 2>/dev/null || echo "0")
SPEED=$(echo "scale=2; $TOKENS / $DURATION" | bc)

echo ""
echo "=== Results ==="
echo "Tokens generated: $TOKENS"
echo "Time: ${DURATION}s"
echo "Speed: ${SPEED} tok/s"
echo ""

# Cleanup
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo "Benchmark complete!"
