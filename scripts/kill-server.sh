#!/bin/bash
# Stop QuantumLeap server (does NOT touch Ollama on port 11434)
QLP_PORT="${API_PORT:-11435}"

echo "Stopping QuantumLeap on port ${QLP_PORT}..."

# Kill by port — QuantumLeap API
if command -v fuser &>/dev/null; then
    fuser -k "${QLP_PORT}/tcp" 2>/dev/null && echo "  stopped API on :${QLP_PORT}"
elif command -v lsof &>/dev/null; then
    PID=$(lsof -ti:"${QLP_PORT}" 2>/dev/null)
    [ -n "$PID" ] && kill -9 $PID 2>/dev/null && echo "  stopped API on :${QLP_PORT}"
elif command -v ss &>/dev/null; then
    PID=$(ss -tlnp 2>/dev/null | grep ":${QLP_PORT}" | grep -oP 'pid=\K[0-9]+' | head -1)
    [ -n "$PID" ] && kill -9 $PID 2>/dev/null && echo "  stopped API on :${QLP_PORT}"
fi

# Kill internal llama-server on port 8081
if command -v fuser &>/dev/null; then
    fuser -k 8081/tcp 2>/dev/null && echo "  stopped llama-server on :8081"
elif command -v lsof &>/dev/null; then
    PID=$(lsof -ti:8081 2>/dev/null)
    [ -n "$PID" ] && kill -9 $PID 2>/dev/null && echo "  stopped llama-server on :8081"
fi

sleep 1

# Verify
STILL_UP=false
if command -v ss &>/dev/null; then
    ss -tln 2>/dev/null | grep -q ":${QLP_PORT}" && STILL_UP=true
elif command -v lsof &>/dev/null; then
    lsof -ti:"${QLP_PORT}" &>/dev/null && STILL_UP=true
fi

if $STILL_UP; then
    echo "  [!] Port ${QLP_PORT} still in use"
    exit 1
else
    echo "  Port ${QLP_PORT} is free"
fi

echo "Done (Ollama on :11434 untouched)"
