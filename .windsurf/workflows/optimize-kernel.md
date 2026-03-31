---
description: Profile and optimize a CUDA or CPU kernel for maximum performance
---

# Optimize Kernel Workflow — ExpertFlow Phase 3

## 1. Build in Profile Mode

### ExpertFlow Core
```bash
cd core && cmake -B build_profile -GNinja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DTURBOQUANT_AVX512=ON \
  -DTURBOQUANT_CUDA=ON \
  -DCMAKE_CXX_FLAGS="-g -fno-omit-frame-pointer"
```

### Build with debug symbols
// turbo
```bash
cd core && ninja -C build_profile
```

## 2. Profile CPU Hotspots (AVX-512/AVX2)

### Record with perf (high frequency sampling)
```bash
cd core && perf record -g -F 999 --call-graph dwarf \
  ./build_profile/test_expertflow
```

### Analyze hotspots
```bash
perf report --no-children --stdio | head -50
```

### Focus on specific functions
```bash
perf annotate --stdio hadamard_transform
perf annotate --stdio polar_decompose
perf annotate --stdio expert_cache_lookup
```

## 3. Check Assembly Output

### Verify SIMD vectorization
```bash
objdump -d -M intel -S core/build_profile/libexpertflow.a | \
  grep -A30 "hadamard_transform" | grep vmovaps
# Should see vmovaps (AVX-512) or vmovaps (AVX2)
```

### Check for unrolled loops
```bash
objdump -d -M intel core/build_profile/libexpertflow.a | \
  grep -A50 "expert_prefetch" | grep -c "vmovaps"
# Higher count = better unrolling
```

## 4. Profile GPU Kernels

### AMD (ROCm) — Detailed metrics
```bash
cd core && rocprof --stats --timestamp on \
  --hsa-trace --hip-trace \
  ./build_profile/test_expertflow
```

### Check occupancy and memory bandwidth
```bash
rocprof --stats ./build_profile/test_expertflow | \
  grep -E "Occupancy|MemoryBandwidth|Duration"
```

### NVIDIA (CUDA) — Nsight Systems
```bash
cd core && nsys profile --stats=true \
  --force-overwrite=true \
  -o expertflow_profile \
  ./build_profile/test_expertflow
```

### NVIDIA — Nsight Compute (kernel-level)
```bash
ncu --set full --export expertflow_kernel \
  ./build_profile/test_expertflow
```

## 5. Analyze ExpertFlow-Specific Metrics

### Cache hit rate profiling
```bash
cd core && EXPERTFLOW_MODEL="../models/Qwen3.5-122B-A10B-UD-IQ2_XXS.gguf" \
  EXPERTFLOW_PROFILE=1 \
  ./build_profile/test_expertflow 2>&1 | grep "Cache hit"
# Target: 75-85% hit rate
```

### Routing predictor accuracy
```bash
cd core && EXPERTFLOW_MODEL="../models/Qwen3.5-122B-A10B-UD-IQ2_XXS.gguf" \
  EXPERTFLOW_PROFILE=1 \
  ./build_profile/test_expertflow 2>&1 | grep "Routing accuracy"
# Target: 74-92% accuracy
```

### Transfer compression ratio
```bash
cd core && EXPERTFLOW_MODEL="../models/Qwen3.5-122B-A10B-UD-IQ2_XXS.gguf" \
  EXPERTFLOW_PROFILE=1 \
  ./build_profile/test_expertflow 2>&1 | grep "Compression"
# Target: 89.7% savings
```

## 6. Optimize Based on Findings

### CPU Optimization Checklist
- ✅ AVX-512 intrinsics for float arrays (16 elements/op)
- ✅ 64-byte alignment for SIMD loads/stores
- ✅ Loop unrolling (factor of 4-8)
- ✅ FMA instructions (_mm512_fmadd_ps)
- ✅ Minimize cache misses (prefetch with _mm_prefetch)

### GPU Optimization Checklist
- ✅ Block size multiple of 32 (warp size)
- ✅ Occupancy >70% (check with rocprof/ncu)
- ✅ Coalesced memory access (consecutive threads → consecutive addresses)
- ✅ Shared memory <48KB per block
- ✅ Warp-level primitives (__shfl_down_sync)
- ✅ Minimize kernel launches (<10 per token)

### ExpertFlow-Specific Optimizations
- ✅ Expert cache: LRU + frequency-weighted eviction
- ✅ Routing predictor: Markov chain with EMA
- ✅ Transfer compression: LZ77-style (89.7% savings)
- ✅ Pipeline overlap: 3 CUDA streams (compute, H2D, D2H)
- ✅ Fused kernels: dequant+GEMV for IQ2_XXS

## 7. Verify Correctness After Optimization
// turbo
```bash
cd core && ninja -C build_profile test_expertflow
# All 35 tests must pass
```

## 8. Benchmark Before/After

### Save baseline
```bash
cd core && ./build_profile/test_expertflow 2>&1 | \
  grep -E "tok/s|Cache hit|Routing accuracy" > /tmp/bench_before.txt
```

### Apply optimization, rebuild, test
```bash
cd core && ninja -C build_profile && \
  ./build_profile/test_expertflow 2>&1 | \
  grep -E "tok/s|Cache hit|Routing accuracy" > /tmp/bench_after.txt
```

### Compare results
```bash
echo "=== BEFORE ===" && cat /tmp/bench_before.txt
echo "=== AFTER ===" && cat /tmp/bench_after.txt
echo "=== DIFF ===" && diff /tmp/bench_before.txt /tmp/bench_after.txt || true
```

## 9. Memory Leak Check
```bash
cd core && valgrind --leak-check=full --show-leak-kinds=all \
  --suppressions=../valgrind.supp \
  ./build_profile/test_expertflow
# Should report: "All heap blocks were freed -- no leaks are possible"
```

## 10. Production Build with Optimizations

### Rebuild in Release mode
```bash
cd core && rm -rf build_ef && cmake -B build_ef -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DTURBOQUANT_AVX512=ON \
  -DTURBOQUANT_CUDA=ON \
  -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native"
```

### Build and verify
```bash
cd core && ninja -C build_ef && ninja -C build_ef test_expertflow
```

## Expected Performance Targets

### CPU (AVX-512)
- Hadamard Transform: 14-25× speedup vs scalar
- Polar Decompose: 10-15× speedup vs scalar
- Expert Dequant: 8-12× speedup vs scalar

### GPU (ROCm/CUDA)
- Occupancy: >70% for compute kernels
- Memory Bandwidth: >60% of peak
- Kernel Launch Overhead: <10 launches per token

### ExpertFlow Phase 3
- **Cache hit rate**: 75-85%
- **Routing accuracy**: 74-92%
- **Transfer compression**: 89.7% savings
- **Overall speedup**: 130% (2.3× baseline)

## Troubleshooting

### Low occupancy
```bash
# Reduce shared memory usage
# Increase block size (256 or 512 threads)
# Check register spilling with rocprof/ncu
```

### Poor cache hit rate
```bash
# Tune cache size (default 2.5 GB)
# Adjust frequency weight (default 0.3)
# Check routing predictor accuracy
```

### Memory leaks
```bash
# Use RAII for all GPU allocations
# Check CUDA stream cleanup
# Verify pinned memory pool destruction
```
