# QuantumLeap — Quick Start

Get running in 5 minutes with **one command**.

## Prerequisites

- **Linux**: Ubuntu 20.04+ / Debian 11+ / Arch / Fedora
- **Dependencies**: cmake, ninja-build, git (auto-installed on most distros)
- **RAM**: 8GB+ (16GB+ for large models)
- **GPU** (optional): NVIDIA (CUDA) / AMD (ROCm)

## 1. Build (One Command)

```bash
git clone https://github.com/MartinCrespoC/QuantumLeap.git
cd QuantumLeap
./build.sh
```

**What it does**:
- ✅ Auto-detects GPU (NVIDIA CUDA / AMD ROCm / CPU-only)
- ✅ Builds ExpertFlow core library (~1 min)
- ✅ Builds llama.cpp with ExpertFlow Phase 3 (~2-4 min)
- ✅ Verifies GPU backend integration
- ✅ Shows next steps

**Build time**: 2-5 minutes total.

**Missing dependencies?** The script will tell you what to install:
```bash
# Ubuntu/Debian
sudo apt install cmake ninja-build git

# Arch
sudo pacman -S cmake ninja git

# Fedora
sudo dnf install cmake ninja-build git
```

## 2. Download a Model

**4GB VRAM** — MoE model (15+ tok/s):
```bash
curl -L -o models/Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf \
  "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/main/Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf"
```

**8GB+ VRAM** — Dense model:
```bash
curl -L -o models/Qwen3.5-27B-Q4_K_M.gguf \
  "https://huggingface.co/Qwen/Qwen3.5-27B-GGUF/resolve/main/qwen3.5-27b-q4_k_m.gguf"
```

**CPU-only** — Small fast model:
```bash
curl -L -o models/Qwen3.5-4B-Q2_K.gguf \
  "https://huggingface.co/Qwen/Qwen3.5-4B-GGUF/resolve/main/qwen3.5-4b-q2_k.gguf"
```

Or use the Web UI **Models** tab to search HuggingFace and download directly.

## 3. Run the Server

```bash
./run.sh models/your-model.gguf
```

**Examples**:
```bash
# Small model (fits on any GPU)
./run.sh models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf

# Large MoE model (auto-optimized)
./run.sh models/Qwen3.5-122B-A10B-UD-IQ2_XXS.gguf -ngl 2

# Custom settings
./run.sh models/model.gguf -ngl 10 -c 8192 --port 8082
```

Server starts at **http://127.0.0.1:8080** (or your custom port).

## 4. Use the API

### OpenAI-compatible API
```bash
curl http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing:",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Chat completion
```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 100
  }'
```

### Check server status
```bash
curl http://127.0.0.1:8080/health
# {"status":"ok"}
```

### Performance metrics
The API returns timing information in every response:
```json
{
  "timings": {
    "prompt_per_second": 5.08,
    "predicted_per_second": 4.14
  }
}
```

## Stopping the Server

Press `Ctrl+C` in the terminal where `./run.sh` is running.

Or find and kill the process:
```bash
pkill -f llama-server
```

## Troubleshooting

### Build Issues

**"cmake not found"**:
```bash
# Ubuntu/Debian
sudo apt install cmake ninja-build git

# Arch
sudo pacman -S cmake ninja git
```

**"No GPU detected" but you have one**:
```bash
# NVIDIA: Install CUDA Toolkit
# AMD: Install ROCm
# Then re-run ./build.sh
```

**Build fails with HIP errors (AMD)**:
```bash
# Check ROCm installation
rocm-smi
# If not found, install ROCm 5.7+
```

### Runtime Issues

| Problem | Solution |
|---------|----------|
| Model not found | Check path: `ls -lh models/*.gguf` |
| Out of memory | Lower `-ngl` value or use smaller model |
| Port already in use | Use `--port 8081` or kill existing server |
| Slow performance | Check GPU is being used: `nvidia-smi` or `rocm-smi` |
| Server won't start | Check logs, verify binary exists: `ls -lh engine/llama.cpp/build/bin/llama-server` |

## Performance Reference

| VRAM | Model | Speed |
|------|-------|-------|
| 4GB | MoE 35B-A3B IQ2_XXS | 15+ tok/s |
| 4GB | 4B Q2_K | 45 tok/s |
| 8GB | Dense 27B Q4_K_M | 8-10 tok/s |
| CPU | MoE 35B-A3B IQ2_XXS | 9-12 tok/s |
