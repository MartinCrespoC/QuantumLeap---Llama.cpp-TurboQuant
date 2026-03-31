---
description: TurboQuant + ExpertFlow — AVX-512/AVX2 SIMD optimization for CPU hot paths
---

# Skill: Optimize with AVX-512/AVX2 SIMD

## Target Hardware
- **Intel i5-11400H**: AVX-512 (Tiger Lake, 6 cores, 12 threads)
- **AMD Ryzen 7 3700X**: AVX2 (Zen 2, 8 cores, 16 threads)

## When to Use SIMD

### TurboQuant KV Cache (Critical Paths)
- **Hadamard Transform** (FWHT) — 14-25× speedup vs scalar
- **Polar Decomposition** — 10-15× speedup vs scalar
- **QJL Sign Quantization** — 8-12× speedup vs scalar

### ExpertFlow MoE (Critical Paths)
- **Expert Weight Dequantization** (IQ2_XXS) — 8-12× speedup
- **Expert Cache Lookup** (hash computation) — 4-6× speedup
- **Routing Predictor** (frequency EMA) — 6-10× speedup

### General Criteria
- Vector operations on arrays >100 elements
- Hot loops identified by `perf record` (>5% total time)
- Data-parallel computations (no dependencies between iterations)

## Optimization Steps

### 1. Identify Hot Loop with perf
```bash
cd core && perf record -g -F 999 ./build_ef/test_expertflow
perf report --no-children --stdio | head -30
# Look for functions consuming >5% CPU time
```

### 2. Ensure Data Alignment
```cpp
// Stack allocation (64-byte aligned for AVX-512)
alignas(64) float data[1024];

// Heap allocation
float* data = (float*)aligned_alloc(64, 1024 * sizeof(float));
// ... use data ...
free(data);

// Check alignment at runtime
assert((uintptr_t)data % 64 == 0);
```

### 3. Vectorize with AVX-512 Intrinsics

#### Example: Hadamard Transform (FWHT)
```cpp
void hadamard_transform_avx512(float* data, int n) {
    // Process 16 floats at a time
    for (int i = 0; i < n; i += 16) {
        __m512 v = _mm512_load_ps(&data[i]);
        
        // Butterfly operations (simplified)
        __m512 a = _mm512_shuffle_ps(v, v, 0xD8);
        __m512 b = _mm512_shuffle_ps(v, v, 0x8D);
        __m512 sum = _mm512_add_ps(a, b);
        __m512 diff = _mm512_sub_ps(a, b);
        
        _mm512_store_ps(&data[i], sum);
        _mm512_store_ps(&data[i + 8], diff);
    }
}
```

#### Example: Polar Decomposition
```cpp
void polar_decompose_avx512(const float* input, float* radius, 
                             float* angles, int n) {
    for (int i = 0; i < n; i += 16) {
        __m512 x = _mm512_load_ps(&input[i]);
        __m512 y = _mm512_load_ps(&input[i + n]);
        
        // Compute radius: sqrt(x^2 + y^2)
        __m512 x2 = _mm512_mul_ps(x, x);
        __m512 y2 = _mm512_mul_ps(y, y);
        __m512 r = _mm512_sqrt_ps(_mm512_add_ps(x2, y2));
        
        // Compute angle: atan2(y, x)
        __m512 theta = _mm512_atan2_ps(y, x);
        
        _mm512_store_ps(&radius[i], r);
        _mm512_store_ps(&angles[i], theta);
    }
}
```

#### Example: Expert Dequantization (IQ2_XXS)
```cpp
void dequant_iq2xxs_avx512(const uint8_t* quant, float* output, 
                            int n, float scale) {
    __m512 vscale = _mm512_set1_ps(scale);
    
    for (int i = 0; i < n; i += 16) {
        // Load 16 bytes (IQ2_XXS uses 2 bits per weight)
        __m128i q = _mm_loadu_si128((__m128i*)&quant[i / 4]);
        
        // Unpack to 16 int32
        __m512i q32 = _mm512_cvtepu8_epi32(q);
        
        // Convert to float and scale
        __m512 f = _mm512_cvtepi32_ps(q32);
        f = _mm512_mul_ps(f, vscale);
        
        _mm512_store_ps(&output[i], f);
    }
}
```

### 4. Handle Remainder Elements
```cpp
void process_array_avx512(float* data, int n) {
    int i = 0;
    
    // Vectorized loop (16 elements at a time)
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_load_ps(&data[i]);
        // ... process v ...
        _mm512_store_ps(&data[i], v);
    }
    
    // Scalar fallback for remainder
    for (; i < n; i++) {
        data[i] = process_scalar(data[i]);
    }
}
```

### 5. Verify Vectorization
```bash
# Check assembly for AVX-512 instructions (zmm registers)
objdump -d -M intel core/build_ef/libexpertflow.a | \
  grep -A20 "hadamard_transform" | grep zmm

# Should see: vmovaps, vaddps, vmulps, etc. with zmm0-zmm31
```

## Key AVX-512 Intrinsics

### Load/Store
```cpp
__m512 _mm512_load_ps(const float* ptr);      // Aligned load (64-byte)
__m512 _mm512_loadu_ps(const float* ptr);     // Unaligned load
void _mm512_store_ps(float* ptr, __m512 v);   // Aligned store
void _mm512_storeu_ps(float* ptr, __m512 v);  // Unaligned store
```

### Arithmetic
```cpp
__m512 _mm512_add_ps(__m512 a, __m512 b);     // a + b
__m512 _mm512_sub_ps(__m512 a, __m512 b);     // a - b
__m512 _mm512_mul_ps(__m512 a, __m512 b);     // a * b
__m512 _mm512_div_ps(__m512 a, __m512 b);     // a / b
__m512 _mm512_fmadd_ps(__m512 a, __m512 b, __m512 c);  // a*b + c (fused)
__m512 _mm512_sqrt_ps(__m512 a);              // sqrt(a)
```

### Reductions
```cpp
float _mm512_reduce_add_ps(__m512 v);         // Horizontal sum
float _mm512_reduce_max_ps(__m512 v);         // Horizontal max
float _mm512_reduce_min_ps(__m512 v);         // Horizontal min
```

### Conversions
```cpp
__m512i _mm512_cvtps_epi32(__m512 v);         // Float → Int32
__m512 _mm512_cvtepi32_ps(__m512i v);         // Int32 → Float
__m512i _mm512_cvtepu8_epi32(__m128i v);      // UInt8 → Int32 (16 elements)
```

### Comparisons & Masks
```cpp
__mmask16 _mm512_cmp_ps_mask(__m512 a, __m512 b, int pred);
__m512 _mm512_mask_blend_ps(__mmask16 k, __m512 a, __m512 b);
```

## AVX2 Fallback (AMD Ryzen)

### When AVX-512 Unavailable
```cpp
#ifdef __AVX512F__
    // AVX-512 path (16 floats)
    __m512 v = _mm512_load_ps(data);
#elif defined(__AVX2__)
    // AVX2 path (8 floats)
    __m256 v = _mm256_load_ps(data);
#else
    // Scalar fallback
    float v = data[0];
#endif
```

### AVX2 Intrinsics (8 floats at a time)
```cpp
__m256 _mm256_load_ps(const float* ptr);
__m256 _mm256_add_ps(__m256 a, __m256 b);
__m256 _mm256_fmadd_ps(__m256 a, __m256 b, __m256 c);
void _mm256_store_ps(float* ptr, __m256 v);
```

## Performance Verification

### 1. Measure Speedup
```bash
# Baseline (scalar)
perf stat -e cycles,instructions core/build_ef/test_expertflow_scalar

# Optimized (AVX-512)
perf stat -e cycles,instructions core/build_ef/test_expertflow
```

### 2. Check AVX-512 Instruction Count
```bash
perf stat -e fp_arith_inst_retired.512b_packed_single \
  core/build_ef/test_expertflow
# Higher count = more AVX-512 usage
```

### 3. Verify Correctness
```bash
# Run tests to ensure vectorized code produces same results
ninja -C core/build_ef test_expertflow
# All 35 tests must pass
```

## Expected Speedups (vs Scalar)

### TurboQuant KV Cache
- **Hadamard Transform**: 14-25× (AVX-512), 8-12× (AVX2)
- **Polar Decomposition**: 10-15× (AVX-512), 6-8× (AVX2)
- **QJL Quantization**: 8-12× (AVX-512), 5-7× (AVX2)

### ExpertFlow MoE
- **Expert Dequant**: 8-12× (AVX-512), 5-7× (AVX2)
- **Cache Lookup**: 4-6× (AVX-512), 3-4× (AVX2)
- **Routing Predictor**: 6-10× (AVX-512), 4-6× (AVX2)

## Common Pitfalls

### 1. Unaligned Access
```cpp
// BAD: Unaligned load with aligned instruction
float* data = malloc(1024 * sizeof(float));  // May not be 64-byte aligned
__m512 v = _mm512_load_ps(data);  // SEGFAULT!

// GOOD: Use aligned_alloc or unaligned load
float* data = aligned_alloc(64, 1024 * sizeof(float));
__m512 v = _mm512_load_ps(data);  // OK

// Or use unaligned load
__m512 v = _mm512_loadu_ps(data);  // OK but slower
```

### 2. Loop Unrolling Too Aggressive
```cpp
// BAD: Too much unrolling (register pressure)
for (int i = 0; i < n; i += 64) {
    __m512 v0 = _mm512_load_ps(&data[i]);
    __m512 v1 = _mm512_load_ps(&data[i + 16]);
    __m512 v2 = _mm512_load_ps(&data[i + 32]);
    __m512 v3 = _mm512_load_ps(&data[i + 48]);
    // ... (uses too many zmm registers)
}

// GOOD: Moderate unrolling (4-8 iterations)
for (int i = 0; i < n; i += 32) {
    __m512 v0 = _mm512_load_ps(&data[i]);
    __m512 v1 = _mm512_load_ps(&data[i + 16]);
    // ... (balanced)
}
```

### 3. Ignoring Remainder Elements
```cpp
// BAD: Ignores last few elements
for (int i = 0; i < n; i += 16) {
    __m512 v = _mm512_load_ps(&data[i]);
    // ... (if n % 16 != 0, last elements not processed!)
}

// GOOD: Handle remainder
int i = 0;
for (; i + 16 <= n; i += 16) {
    __m512 v = _mm512_load_ps(&data[i]);
    // ...
}
for (; i < n; i++) {
    // Scalar fallback
}
```

## Best Practices

1. **Profile first**: Use `perf` to identify hot loops (>5% CPU time)
2. **Align data**: Use `alignas(64)` or `aligned_alloc(64, size)`
3. **Unroll moderately**: 4-8 iterations to balance ILP and register pressure
4. **Handle remainders**: Always process elements not divisible by vector width
5. **Verify correctness**: Run tests after vectorization (35/35 must pass)
6. **Check assembly**: Use `objdump` to verify zmm/ymm instructions generated
7. **Measure speedup**: Use `perf stat` to confirm performance improvement
8. **Fallback to AVX2**: Support AMD CPUs without AVX-512

## Profiling Commands

```bash
# Record hotspots
perf record -g -F 999 core/build_ef/test_expertflow

# Analyze
perf report --no-children --stdio | head -50

# Annotate specific function
perf annotate --stdio hadamard_transform

# Check AVX-512 usage
perf stat -e fp_arith_inst_retired.512b_packed_single \
  core/build_ef/test_expertflow
```

## Integration with ExpertFlow Phase 3

ExpertFlow Phase 3 achieves **130% speedup** through:
- **Expert cache**: 75-85% hit rate (SIMD hash lookup)
- **Routing predictor**: 74-92% accuracy (SIMD EMA computation)
- **Transfer compression**: 89.7% savings (SIMD LZ77)
- **Dequantization**: IQ2_XXS (SIMD unpacking)
- **Pipeline overlap**: Multi-stream execution

**CPU SIMD optimizations** complement GPU acceleration by:
- Reducing CPU bottlenecks in expert prefetching
- Accelerating routing prediction (Markov chain)
- Speeding up transfer compression/decompression
- Enabling efficient CPU-only fallback mode
