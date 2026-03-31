#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

// TurboQuant KV Cache Pipeline (Google Research, arXiv:2504.19874)
#include "turboquant/hadamard.h"
#include "turboquant/polarquant.h"
#include "turboquant/qjl.h"
#include "turboquant/turboquant_kv.h"

namespace turboquant {

// Quantization bit widths
enum class QuantBits : uint8_t {
  kInt2 = 2,
  kInt4 = 4,
  kInt8 = 8,
};

// Quantization method
enum class QuantMethod : uint8_t {
  kScalar,       // Simple scalar quantization
  kPolarQuant,   // PolarQuant (TurboQuant stage 1)
  kTurboQuant,   // Full TurboQuant (PolarQuant + Residual)
};

// Quantized tensor metadata
struct QuantMeta {
  size_t num_elements;
  size_t num_groups;
  size_t group_size;
  QuantBits bits;
  QuantMethod method;
  float* scales;        // Per-group scale factors
  float* zero_points;   // Per-group zero points
  float* magnitudes;    // PolarQuant magnitudes (if applicable)
  float* angles;        // PolarQuant angles (if applicable)
};

// Result of quantization
struct QuantResult {
  std::vector<uint8_t> data;     // Packed quantized data
  QuantMeta meta;
  float mse;                     // Mean squared error vs original
  float max_error;               // Maximum absolute error
};

// ============================================
// PolarQuant Transform (Stage 1)
// ============================================

// CPU implementation (scalar fallback) — defined in C++
void polar_transform_scalar(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ magnitude,
    float* __restrict__ angle,
    size_t n);

// Auto-dispatch (selects best implementation at runtime)
void polar_transform(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ magnitude,
    float* __restrict__ angle,
    size_t n);

// ============================================
// Residual Quantization (Stage 2)
// ============================================

QuantResult residual_quantize(
    const float* data,
    size_t n,
    QuantBits bits,
    size_t group_size = 128,
    int num_iterations = 3);

// ============================================
// Full TurboQuant Pipeline
// ============================================

QuantResult turboquant_encode(
    const float* data,
    size_t n,
    QuantBits bits = QuantBits::kInt2,
    size_t group_size = 128);

void turboquant_decode(
    const QuantResult& quantized,
    float* output,
    size_t n);

// ============================================
// CUDA Kernels (GPU)
// ============================================

// KV Cache compression on GPU
void kv_cache_compress_cuda(
    const float* kv_data,
    uint8_t* compressed,
    float* scales,
    size_t seq_len,
    size_t head_dim,
    QuantBits bits);

void kv_cache_decompress_cuda(
    const uint8_t* compressed,
    const float* scales,
    float* output,
    size_t seq_len,
    size_t head_dim,
    QuantBits bits);

// INT2/INT4 matrix multiply on GPU
void matmul_int2_cuda(
    const uint8_t* A,
    const uint8_t* B,
    float* C,
    const float* scales_a,
    const float* scales_b,
    int M, int N, int K);

void matmul_int4_cuda(
    const uint8_t* A,
    const uint8_t* B,
    float* C,
    const float* scales_a,
    const float* scales_b,
    int M, int N, int K);

// Compressed attention
void attention_compressed_cuda(
    const float* Q,
    const uint8_t* K_compressed,
    const uint8_t* V_compressed,
    const float* K_scales,
    const float* V_scales,
    float* output,
    int batch, int heads, int seq_len, int head_dim,
    QuantBits kv_bits);

// ============================================
// Fast Dequantization (Assembly)
// ============================================

// AVX-512 fast dequantize INT2 → FP32
void dequant_int2_avx512(
    const uint8_t* __restrict__ packed,
    const float* __restrict__ scales,
    float* __restrict__ output,
    size_t n);

// AVX-512 fast dequantize INT4 → FP32
void dequant_int4_avx512(
    const uint8_t* __restrict__ packed,
    const float* __restrict__ scales,
    float* __restrict__ output,
    size_t n);

// AVX2 fast dequantize INT2 → FP32 (fallback)
void dequant_int2_avx2(
    const uint8_t* __restrict__ packed,
    const float* __restrict__ scales,
    float* __restrict__ output,
    size_t n);

// AVX2 fast dequantize INT4 → FP32 (fallback)
void dequant_int4_avx2(
    const uint8_t* __restrict__ packed,
    const float* __restrict__ scales,
    float* __restrict__ output,
    size_t n);

// ============================================
// Lookup Tables
// ============================================

void init_lookup_tables();
void destroy_lookup_tables();

// ============================================
// Utility
// ============================================

// Check CPU feature support
bool has_avx512();
bool has_avx2();

// GPU device info (CUDA or HIP)
int get_gpu_device_count();
size_t get_gpu_free_memory(int device = 0);
int get_gpu_compute_capability(int device = 0);

}  // namespace turboquant
