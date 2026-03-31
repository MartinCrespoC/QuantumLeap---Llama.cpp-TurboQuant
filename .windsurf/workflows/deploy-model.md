---
description: Deploy a MoE model with QuantumLeap ExpertFlow Phase 3 + API server + Web UI
---

# Deploy MoE Model Workflow — ExpertFlow Phase 3

## 1. Download MoE Model

### Option A: 122B MoE (recommended for Phase 3 testing)
```bash
curl -L -o models/Qwen3.5-122B-A10B-UD-IQ2_XXS.gguf \
  'https://huggingface.co/unsloth/Qwen3.5-122B-A10B-GGUF/resolve/main/Qwen3.5-122B-A10B-UD-IQ2_XXS.gguf'
```

### Option B: 35B MoE (faster, smaller)
```bash
curl -L -o models/Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf \
  'https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/main/Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf'
```

### Option C: Mixtral 8x7B (classic MoE)
```bash
curl -L -o models/Mixtral-8x7B-Instruct-v0.1-IQ2_XXS.gguf \
  'https://huggingface.co/bartowski/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/Mixtral-8x7B-Instruct-v0.1-IQ2_XXS.gguf'
```

## 2. Verify QuantumLeap is built with ExpertFlow
```bash
ls -lh engine/llama.cpp/build/bin/llama-server
# Should be ~7.8 MB with ExpertFlow Phase 3
```

If not built, run:
```bash
bash setup.sh  # Auto-detects GPU and builds with ExpertFlow
```

## 3. Start QuantumLeap API Server
// turbo
```bash
source venv/bin/activate
python api/server.py
```

Expected output:
```
[ExpertFlow] Initialized for MoE model: 256 experts, top-8 routing
[ExpertFlow] Expert cache: 2.5 GB VRAM (3103 slots)
[ExpertFlow] Routing predictor: enabled (Markov chain)
[ExpertFlow] Transfer compression: 89.7% savings
Server running at http://localhost:11435
```

## 4. Verify API health
// turbo
```bash
curl -s http://localhost:11435/api/tags | python3 -m json.tool
```

## 5. Test generation with streaming
```bash
curl -s http://localhost:11435/api/generate -d '{
  "model": "Qwen3.5-122B-A10B-UD-IQ2_XXS.gguf",
  "prompt": "Explain quantum computing in simple terms",
  "stream": true
}'
```

## 6. Test OpenAI-compatible endpoint
```bash
curl -s http://localhost:11435/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen3.5-122B-A10B-UD-IQ2_XXS.gguf",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is ExpertFlow?"}
  ]
}' | python3 -m json.tool
```

## 7. Monitor ExpertFlow Performance

Check logs for metrics (logged every 100 tokens):
```
[ExpertFlow] Cache hit rate: 82.3% (target: 75-85%)
[ExpertFlow] Routing accuracy: 87.1% (target: 74-92%)
[ExpertFlow] Transfer compression: 89.7% savings
[ExpertFlow] Performance: 4.34 tok/s (2.3× baseline)
```

## 8. Open Web UI

Access the built-in Web UI:
```
http://localhost:11435
```

Features:
- Chat interface with streaming
- Model selection
- Temperature/top-p controls
- Token usage stats
- ExpertFlow metrics display

## 9. (Optional) Benchmark Performance
```bash
curl -s http://localhost:11435/api/generate -d '{
  "model": "Qwen3.5-122B-A10B-UD-IQ2_XXS.gguf",
  "prompt": "Write a detailed explanation of neural networks",
  "stream": false
}' | python3 -c "import sys, json; data=json.load(sys.stdin); print(f\"Tokens: {data.get('eval_count', 0)}, Time: {data.get('eval_duration', 0)/1e9:.2f}s, Speed: {data.get('eval_count', 0)/(data.get('eval_duration', 1)/1e9):.2f} tok/s\")"
```

## 10. (Optional) Production Deployment

### Systemd Service
Create `/etc/systemd/system/quantumleap.service`:
```ini
[Unit]
Description=QuantumLeap ExpertFlow Phase 3 API Server
After=network.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/path/to/QuantumLeap
ExecStart=/path/to/QuantumLeap/venv/bin/python api/server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable quantumleap
sudo systemctl start quantumleap
sudo systemctl status quantumleap
```

## Expected Performance (6GB VRAM)
- **122B MoE**: 4.34 tok/s ⭐ (130% faster than baseline)
- **35B MoE**: 15+ tok/s
- **Mixtral 8x7B**: 25+ tok/s
- **Cache hit rate**: 75-85%
- **Routing accuracy**: 74-92%

## Troubleshooting

### ExpertFlow not activating
```bash
# Check model metadata
strings models/Qwen3.5-122B-A10B-UD-IQ2_XXS.gguf | grep -i expert

# Verify build flags
ldd engine/llama.cpp/build/bin/llama-server | grep expertflow
```

### Low performance
```bash
# Check GPU detection
rocm-smi  # AMD
nvidia-smi  # NVIDIA

# Monitor VRAM usage
watch -n 1 'rocm-smi | grep -A5 "GPU use"'
```

### Server crashes
```bash
# Check logs
tail -f api/server.log

# Reduce ngl if OOM
# Edit api/server.py, set ngl=1 for 6GB VRAM
```
