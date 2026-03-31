#pragma once

#include <cstddef>
#include <cstdint>

#if defined(AVX512) || defined(AVX2)
#include <immintrin.h>
#endif

namespace turboquant {
namespace simd {

// ============================================
// AVX-512 Utility Functions
// ============================================

#ifdef AVX512

// Load 16 floats from aligned memory
inline __m512 load_aligned(const float* ptr) {
  return _mm512_load_ps(ptr);
}

// Load 16 floats from unaligned memory
inline __m512 load_unaligned(const float* ptr) {
  return _mm512_loadu_ps(ptr);
}

// Store 16 floats to aligned memory
inline void store_aligned(float* ptr, __m512 v) {
  _mm512_store_ps(ptr, v);
}

// Store 16 floats to unaligned memory
inline void store_unaligned(float* ptr, __m512 v) {
  _mm512_storeu_ps(ptr, v);
}

// Fused multiply-add: a * b + c
inline __m512 fmadd(__m512 a, __m512 b, __m512 c) {
  return _mm512_fmadd_ps(a, b, c);
}

// Horizontal sum of 16 floats
inline float hsum(__m512 v) {
  return _mm512_reduce_add_ps(v);
}

// Horizontal max of 16 floats
inline float hmax(__m512 v) {
  return _mm512_reduce_max_ps(v);
}

// Horizontal min of 16 floats
inline float hmin(__m512 v) {
  return _mm512_reduce_min_ps(v);
}

// Broadcast scalar to all 16 lanes
inline __m512 broadcast(float val) {
  return _mm512_set1_ps(val);
}

// Absolute value
inline __m512 abs(__m512 v) {
  return _mm512_abs_ps(v);
}

// Square root
inline __m512 sqrt(__m512 v) {
  return _mm512_sqrt_ps(v);
}

// Reciprocal square root (fast approximation)
inline __m512 rsqrt(__m512 v) {
  return _mm512_rsqrt14_ps(v);
}

// Clamp values to [min, max]
inline __m512 clamp(__m512 v, __m512 lo, __m512 hi) {
  return _mm512_min_ps(_mm512_max_ps(v, lo), hi);
}

// Quantize float to INT2 (0, 1, 2, 3)
inline __m512i quantize_int2(__m512 v, __m512 scale, __m512 zero) {
  __m512 scaled = _mm512_mul_ps(_mm512_add_ps(v, zero), scale);
  __m512 clamped = clamp(scaled, _mm512_setzero_ps(), broadcast(3.0f));
  return _mm512_cvtps_epi32(clamped);
}

// Quantize float to INT4 (0..15)
inline __m512i quantize_int4(__m512 v, __m512 scale, __m512 zero) {
  __m512 scaled = _mm512_mul_ps(_mm512_add_ps(v, zero), scale);
  __m512 clamped = clamp(scaled, _mm512_setzero_ps(), broadcast(15.0f));
  return _mm512_cvtps_epi32(clamped);
}

#endif  // AVX512

// ============================================
// AVX2 Utility Functions (fallback for non-AVX-512 CPUs)
// ============================================

#ifdef AVX2

// Load 8 floats from unaligned memory
inline __m256 load_unaligned_256(const float* ptr) {
  return _mm256_loadu_ps(ptr);
}

// Store 8 floats to unaligned memory
inline void store_unaligned_256(float* ptr, __m256 v) {
  _mm256_storeu_ps(ptr, v);
}

// Fused multiply-add: a * b + c
inline __m256 fmadd_256(__m256 a, __m256 b, __m256 c) {
  return _mm256_fmadd_ps(a, b, c);
}

// Broadcast scalar to all 8 lanes
inline __m256 broadcast_256(float val) {
  return _mm256_set1_ps(val);
}

// Square root
inline __m256 sqrt_256(__m256 v) {
  return _mm256_sqrt_ps(v);
}

// Horizontal sum of 8 floats
inline float hsum_256(__m256 v) {
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 lo = _mm256_castps256_ps128(v);
  lo = _mm_add_ps(lo, hi);
  lo = _mm_hadd_ps(lo, lo);
  lo = _mm_hadd_ps(lo, lo);
  return _mm_cvtss_f32(lo);
}

// Clamp values to [min, max]
inline __m256 clamp_256(__m256 v, __m256 lo, __m256 hi) {
  return _mm256_min_ps(_mm256_max_ps(v, lo), hi);
}

// Quantize float to INT2 (0, 1, 2, 3)
inline __m256i quantize_int2_256(__m256 v, __m256 scale, __m256 zero) {
  __m256 scaled = _mm256_mul_ps(_mm256_add_ps(v, zero), scale);
  __m256 clamped = clamp_256(scaled, _mm256_setzero_ps(), broadcast_256(3.0f));
  return _mm256_cvtps_epi32(clamped);
}

// Quantize float to INT4 (0..15)
inline __m256i quantize_int4_256(__m256 v, __m256 scale, __m256 zero) {
  __m256 scaled = _mm256_mul_ps(_mm256_add_ps(v, zero), scale);
  __m256 clamped = clamp_256(scaled, _mm256_setzero_ps(), broadcast_256(15.0f));
  return _mm256_cvtps_epi32(clamped);
}

#endif  // AVX2

// ============================================
// Alignment Utilities
// ============================================

// Check if pointer is aligned to N bytes
template <size_t N>
inline bool is_aligned(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & (N - 1)) == 0;
}

// Round up to next multiple of N
template <size_t N>
inline size_t align_up(size_t val) {
  return (val + N - 1) & ~(N - 1);
}

// Allocate aligned memory
inline void* aligned_alloc(size_t alignment, size_t size) {
  return ::aligned_alloc(alignment, align_up<64>(size));
}

}  // namespace simd
}  // namespace turboquant
