---
description: Build QuantumLeap with ExpertFlow Phase 3 and run full test suite
---

# Build and Test Workflow — ExpertFlow Phase 3

## 1. Build ExpertFlow Core Library

Clean and configure CMake build with ExpertFlow enabled:
```bash
cd core && rm -rf build_ef && cmake -B build_ef -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DTURBOQUANT_AVX512=ON \
  -DTURBOQUANT_CUDA=ON
```

## 2. Build with all cores
// turbo
```bash
cd core && ninja -C build_ef
```

## 3. Run ExpertFlow tests (35/35 should pass)
```bash
cd core && ninja -C build_ef test_expertflow
```

## 4. Test with real MoE model
```bash
cd core && EXPERTFLOW_MODEL="../models/Qwen3.5-122B-A10B-UD-IQ2_XXS.gguf" \
  ./build_ef/test_expertflow
```

## 5. Build llama.cpp with ExpertFlow Phase 3

### AMD (ROCm/HIP)
```bash
cd engine/llama.cpp && rm -rf build && \
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
cmake -B build -G Ninja \
  -DGGML_HIP=ON -DGPU_TARGETS=gfx1010 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_EXPERTFLOW=ON \
  -DLLAMA_BUILD_SERVER=ON
```

### NVIDIA (CUDA)
```bash
cd engine/llama.cpp && rm -rf build && \
cmake -B build -G Ninja \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_EXPERTFLOW=ON \
  -DLLAMA_BUILD_SERVER=ON
```

## 6. Build llama-server
// turbo
```bash
cd engine/llama.cpp && ninja -C build llama-server
```

## 7. Verify binary
```bash
ls -lh engine/llama.cpp/build/bin/llama-server
# Should be ~7.8 MB with ExpertFlow
```

## 8. Test server with MoE model
```bash
source venv/bin/activate && python api/server.py
# Check logs for ExpertFlow initialization
```

## 9. Check for memory leaks (optional)
```bash
valgrind --leak-check=full --show-leak-kinds=all \
  core/build_ef/test_expertflow
```

## 10. Profile GPU kernels (optional)

### AMD (ROCm)
```bash
rocprof --stats core/build_ef/test_expertflow
```

### NVIDIA (CUDA)
```bash
nsys profile --stats=true core/build_ef/test_expertflow
```

## Expected Results
- ✅ ExpertFlow tests: 35/35 pass
- ✅ Binary size: ~7.8 MB
- ✅ Performance: 4.34 tok/s on 122B MoE (6GB VRAM)
- ✅ Cache hit rate: 75-85%
- ✅ Routing accuracy: 74-92%
- ✅ Transfer compression: 89.7% savings
- ✅ No memory leaks in Valgrind
