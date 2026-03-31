#!/bin/bash
# Stop QuantumLeap server (does NOT touch Ollama on port 11434)
QLP_PORT="${API_PORT:-11435}"

echo "Stopping QuantumLeap on port ${QLP_PORT}..."

# Kill QuantumLeap API on our port
PID=$(lsof -ti:"${QLP_PORT}" 2>/dev/null)
if [ -n "$PID" ]; then
    kill -9 $PID 2>/dev/null && echo "  stopped API on :${QLP_PORT} (PID: $PID)"
fi

# Kill internal llama-server on port 8081
PID=$(lsof -ti:8081 2>/dev/null)
if [ -n "$PID" ]; then
    kill -9 $PID 2>/dev/null && echo "  stopped llama-server on :8081 (PID: $PID)"
fi

sleep 1

if lsof -i:"${QLP_PORT}" >/dev/null 2>&1; then
    echo "  [!] Port ${QLP_PORT} still in use"
    exit 1
else
    echo "  Port ${QLP_PORT} is free"
fi

echo "Done (Ollama on :11434 untouched)"
