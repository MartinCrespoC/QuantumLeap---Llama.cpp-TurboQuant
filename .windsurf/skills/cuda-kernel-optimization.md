---
description: ExpertFlow Phase 3 — CUDA/HIP kernel optimization for MoE inference
---

# Skill: Optimize CUDA/HIP Kernels for ExpertFlow

## Target Hardware
- **AMD**: RX 5600 XT (RDNA1, gfx1010, 6GB VRAM, 288 GB/s)
- **NVIDIA**: RTX 3050/4090 (Ampere/Ada, 4-24GB VRAM)

## ExpertFlow Phase 3 Kernel Checklist

### Memory Access Patterns
- [ ] Coalesced global memory access (128-byte aligned)
- [ ] Consecutive threads → consecutive addresses
- [ ] Use `__restrict__` on all pointer params
- [ ] Align expert weights to 128 bytes for PCIe transfers
- [ ] Prefetch with async memcpy for H2D transfers

### Shared Memory Optimization
- [ ] Tile-based algorithms for expert GEMV
- [ ] Shared memory <48KB per block (RDNA2/Ampere limit)
- [ ] No bank conflicts (stride by 32 for float access)
- [ ] Use shared memory for hot expert cache

### Occupancy & Parallelism
- [ ] Occupancy >70% (`rocprof`/`ncu --metrics achieved_occupancy`)
- [ ] Block size multiple of 32 (warp size)
- [ ] Register pressure <64 per thread
- [ ] Minimize warp divergence in expert dispatch
- [ ] Target: 256-512 threads per block

### Warp-Level Primitives
- [ ] Use `__shfl_down_sync()` for warp reductions
- [ ] Avoid `__syncthreads()` in divergent branches
- [ ] Prefer warp-level ops over shared memory atomics
- [ ] Use warp-level reductions for GEMV

### Fused Kernels (ExpertFlow Specific)
- [ ] Fuse dequant+GEMV for IQ2_XXS experts
- [ ] Fuse SiLU activation with multiplication
- [ ] Minimize kernel launches (<10 per token)
- [ ] Combine expert accumulation with output projection

## ExpertFlow Phase 3 Kernel Patterns

### 1. Fused Dequant+GEMV (IQ2_XXS)
```cuda
__global__ void dequant_gemv_iq2xxs_kernel(
    const uint8_t* __restrict__ expert_weights,  // Quantized
    const float* __restrict__ input,              // Activation
    float* __restrict__ output,                   // Result
    const int expert_dim,
    const int input_dim
) {
    // Shared memory for input tile
    __shared__ float s_input[256];
    
    // Warp-level reduction
    float sum = 0.0f;
    for (int i = threadIdx.x; i < input_dim; i += blockDim.x) {
        // Dequantize on-the-fly
        float w = dequant_iq2xxs(expert_weights[i]);
        sum += w * input[i];
    }
    
    // Warp reduction
    sum = warp_reduce_sum(sum);
    
    if (threadIdx.x % 32 == 0) {
        atomicAdd(&output[blockIdx.x], sum);
    }
}
```

### 2. Expert Cache Lookup (Frequency-Weighted LRU)
```cuda
__device__ int expert_cache_lookup(
    const int layer_id,
    const int expert_id,
    CacheEntry* __restrict__ cache,
    const int cache_size
) {
    // Hash-based lookup with linear probing
    int slot = (layer_id * 256 + expert_id) % cache_size;
    
    // Check cache hit
    if (cache[slot].layer == layer_id && 
        cache[slot].expert == expert_id) {
        atomicAdd(&cache[slot].frequency, 1);
        return slot;  // Hit
    }
    
    return -1;  // Miss
}
```

### 3. Multi-Stream Pipeline Overlap
```cuda
// Stream 0: Compute (attention + expert dispatch)
// Stream 1: H2D (prefetch next experts)
// Stream 2: D2H (transfer results)

cudaStream_t streams[3];
for (int i = 0; i < 3; i++) {
    cudaStreamCreate(&streams[i]);
}

// Token loop
for (int t = 0; t < num_tokens; t++) {
    // Compute current token (stream 0)
    expert_dispatch_kernel<<<grid, block, 0, streams[0]>>>(/*...*/);
    
    // Prefetch next experts (stream 1)
    if (t + 1 < num_tokens) {
        cudaMemcpyAsync(d_experts, h_experts, size, 
                        cudaMemcpyHostToDevice, streams[1]);
    }
    
    // Transfer previous results (stream 2)
    if (t > 0) {
        cudaMemcpyAsync(h_output, d_output, size,
                        cudaMemcpyDeviceToHost, streams[2]);
    }
}
```

## Profiling Commands

### AMD (ROCm)
```bash
# Overall stats
rocprof --stats ./test_expertflow

# Detailed trace
rocprof --hsa-trace --hip-trace --timestamp on ./test_expertflow

# Specific metrics
rocprof --stats --timestamp on \
  -m pmc:SQ_WAVES,SQ_INSTS_VALU,TCC_HIT,TCC_MISS \
  ./test_expertflow
```

### NVIDIA (CUDA)
```bash
# Nsight Systems (timeline)
nsys profile --stats=true -o expertflow_profile ./test_expertflow

# Nsight Compute (kernel-level)
ncu --set full --export expertflow_kernel ./test_expertflow

# Legacy nvprof
nvprof --metrics achieved_occupancy,sm_efficiency,dram_utilization \
  ./test_expertflow
```

## Performance Targets

### Memory Bandwidth
- **AMD RX 5600 XT**: 288 GB/s theoretical
  - Good kernel: >173 GB/s (>60% utilization)
  - Expert transfer: Hidden behind compute (PCIe 3.0 x16 = 16 GB/s)

- **NVIDIA RTX 4090**: 1008 GB/s theoretical
  - Good kernel: >605 GB/s (>60% utilization)
  - Expert transfer: Hidden behind compute (PCIe 4.0 x16 = 32 GB/s)

### Compute Throughput
- **FP32 GEMV**: >70% of peak FLOPS
- **IQ2_XXS Dequant**: <5% overhead vs FP16
- **Expert Cache Lookup**: <10 cycles per lookup

### ExpertFlow Phase 3 Metrics
- **Cache hit rate**: 75-85% (target)
- **Routing accuracy**: 74-92% (target)
- **Transfer compression**: 89.7% savings (target)
- **Kernel launches**: <10 per token (target)
- **Overall speedup**: 130% (2.3× baseline)

## Optimization Workflow

### 1. Identify Hotspot
```bash
rocprof --stats ./test_expertflow | grep -E "Duration|Calls"
# Focus on kernels with Duration >10% total time
```

### 2. Analyze Occupancy
```bash
rocprof --stats ./test_expertflow | grep Occupancy
# Target: >70% for compute-bound kernels
```

### 3. Check Memory Bandwidth
```bash
rocprof --stats ./test_expertflow | grep -i bandwidth
# Target: >60% of peak
```

### 4. Optimize & Verify
```bash
# Rebuild with optimizations
ninja -C core/build_ef

# Re-profile
rocprof --stats ./test_expertflow

# Verify correctness
ninja -C core/build_ef test_expertflow
# All 35 tests must pass
```

## Common Issues & Fixes

### Low Occupancy (<50%)
- **Cause**: Too much shared memory or registers
- **Fix**: Reduce shared memory usage, use more blocks with fewer threads

### Poor Memory Bandwidth (<40%)
- **Cause**: Non-coalesced access or bank conflicts
- **Fix**: Ensure consecutive threads access consecutive addresses

### High Kernel Launch Overhead
- **Cause**: Too many small kernels
- **Fix**: Fuse kernels (e.g., dequant+GEMV+SiLU)

### Cache Thrashing
- **Cause**: Poor eviction policy
- **Fix**: Use frequency-weighted LRU instead of pure LRU

## ROCm/HIP vs CUDA Differences

### Memory Allocation
```cpp
// CUDA
cudaMalloc(&ptr, size);
cudaFree(ptr);

// HIP (works on both AMD and NVIDIA)
hipMalloc(&ptr, size);
hipFree(ptr);
```

### Kernel Launch
```cpp
// CUDA
kernel<<<grid, block>>>(args);

// HIP
hipLaunchKernelGGL(kernel, grid, block, 0, 0, args);
// Or use CUDA syntax (HIP supports it)
kernel<<<grid, block>>>(args);
```

### Profiling
```cpp
// CUDA
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventRecord(start);

// HIP
hipEvent_t start, stop;
hipEventCreate(&start);
hipEventRecord(start);
```

## Best Practices for ExpertFlow

1. **Fuse kernels**: Combine dequant+GEMV+activation
2. **Use streams**: Overlap compute, H2D, D2H
3. **Compress transfers**: LZ77-style (89.7% savings)
4. **Cache hot experts**: LRU + frequency-weighted
5. **Prefetch speculatively**: Markov chain predictor
6. **Minimize launches**: <10 kernels per token
7. **Profile first**: Don't optimize blindly
8. **Test always**: 35/35 tests must pass
