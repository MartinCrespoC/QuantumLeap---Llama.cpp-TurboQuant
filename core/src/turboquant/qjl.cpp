#include "turboquant/qjl.h"
#include "turboquant/simd_utils.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <random>

#ifdef AVX2
#include <immintrin.h>
#endif

namespace turboquant {

// ─── QJLContext ──────────────────────────────────────────────────────────────

std::unique_ptr<QJLContext> QJLContext::create(
    size_t input_dim, size_t proj_dim, uint64_t seed) {
  auto ctx = std::make_unique<QJLContext>();
  ctx->input_dim = input_dim;
  ctx->proj_dim = proj_dim;
  ctx->seed = seed;

  // Generate random Rademacher matrix: entries ±1/√proj_dim
  ctx->R.resize(proj_dim * input_dim);
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> dist(0, 1);
  float scale = 1.0f / std::sqrt(static_cast<float>(proj_dim));

  for (size_t i = 0; i < proj_dim * input_dim; ++i) {
    ctx->R[i] = dist(rng) ? scale : -scale;
  }

  return ctx;
}

// ─── AVX2-optimized dot product ─────────────────────────────────────────────

static inline float dot_product(const float* __restrict__ a,
                                const float* __restrict__ b,
                                size_t n) {
#ifdef AVX2
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  size_t i = 0;
  // Unrolled 2x: 16 elements/iteration to saturate FMA throughput
  for (; i + 15 < n; i += 16) {
    __m256 a0 = _mm256_loadu_ps(a + i);
    __m256 b0 = _mm256_loadu_ps(b + i);
    acc0 = _mm256_fmadd_ps(a0, b0, acc0);
    __m256 a1 = _mm256_loadu_ps(a + i + 8);
    __m256 b1 = _mm256_loadu_ps(b + i + 8);
    acc1 = _mm256_fmadd_ps(a1, b1, acc1);
  }
  acc0 = _mm256_add_ps(acc0, acc1);
  for (; i + 7 < n; i += 8) {
    __m256 a0 = _mm256_loadu_ps(a + i);
    __m256 b0 = _mm256_loadu_ps(b + i);
    acc0 = _mm256_fmadd_ps(a0, b0, acc0);
  }
  // Horizontal sum
  __m128 hi = _mm256_extractf128_ps(acc0, 1);
  __m128 lo = _mm256_castps256_ps128(acc0);
  __m128 sum = _mm_add_ps(lo, hi);
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  float result = _mm_cvtss_f32(sum);
  // Scalar tail
  for (; i < n; ++i) {
    result += a[i] * b[i];
  }
  return result;
#else
  float result = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    result += a[i] * b[i];
  }
  return result;
#endif
}

// ─── QJL Encode (AVX2 optimized) ────────────────────────────────────────────
// Project residual through random matrix R, then take sign bits
// Optimized: vectorized dot product + 64-bit word packing

QJLCompressed qjl_encode(
    const QJLContext& ctx,
    const float* __restrict__ residuals,
    size_t num_vectors) {
  QJLCompressed result;
  result.num_vectors = num_vectors;
  result.proj_dim = ctx.proj_dim;

  size_t bits_per_vec = ctx.proj_dim;
  size_t bytes_per_vec = (bits_per_vec + 7) / 8;
  result.sign_bits.resize(num_vectors * bytes_per_vec, 0);

  for (size_t v = 0; v < num_vectors; ++v) {
    const float* res = residuals + v * ctx.input_dim;
    uint8_t* out_bits = result.sign_bits.data() + v * bytes_per_vec;

    // Prefetch next vector's residual data
    if (v + 1 < num_vectors) {
      __builtin_prefetch(residuals + (v + 1) * ctx.input_dim, 0, 1);
    }

    // Process projections in groups of 64 for efficient bit packing
    size_t j = 0;
    for (; j + 63 < ctx.proj_dim; j += 64) {
      uint64_t word = 0;
      for (size_t b = 0; b < 64; ++b) {
        const float* row = ctx.R.data() + (j + b) * ctx.input_dim;
        float dot = dot_product(row, res, ctx.input_dim);
        if (dot >= 0.0f) {
          word |= (1ULL << b);
        }
      }
      // Store 8 bytes at once (little-endian)
      std::memcpy(out_bits + j / 8, &word, sizeof(uint64_t));
    }
    // Remainder: byte-level packing
    for (; j < ctx.proj_dim; ++j) {
      const float* row = ctx.R.data() + j * ctx.input_dim;
      float dot = dot_product(row, res, ctx.input_dim);
      if (dot >= 0.0f) {
        size_t bit_idx = j;
        out_bits[bit_idx / 8] |= (1 << (bit_idx % 8));
      }
    }
  }

  return result;
}

// ─── QJL Inner Product (AVX2 optimized) ──────────────────────────────────────
// Estimate <query, key_residual> using sign bits + random projection.
// Hot path: vectorized query projection + 64-bit popcount for sign accumulation

void qjl_inner_product(
    const QJLContext& ctx,
    const float* __restrict__ query,
    const QJLCompressed& compressed_keys,
    float* __restrict__ corrections,
    size_t num_vectors) {
  assert(compressed_keys.proj_dim == ctx.proj_dim);

  // Pre-compute query projections: qp[j] = R[j] · query (vectorized)
  std::vector<float> query_proj(ctx.proj_dim);
  for (size_t j = 0; j < ctx.proj_dim; ++j) {
    const float* row = ctx.R.data() + j * ctx.input_dim;
    query_proj[j] = dot_product(row, query, ctx.input_dim);
  }

  size_t bits_per_vec = ctx.proj_dim;
  size_t bytes_per_vec = (bits_per_vec + 7) / 8;
  float scale = static_cast<float>(M_PI) / (2.0f * ctx.proj_dim);

  // Pre-compute negated projections for branchless sign application
  // sign(bit) * qp[j] = bit ? qp[j] : -qp[j] = 2*bit*qp[j] - qp[j]
  // Sum over all j: 2 * (sum of qp where bit=1) - (sum of all qp)
  float total_sum = 0.0f;
  for (size_t j = 0; j < ctx.proj_dim; ++j) {
    total_sum += query_proj[j];
  }

  for (size_t v = 0; v < num_vectors; ++v) {
    const uint8_t* bits_ptr = compressed_keys.sign_bits.data()
                              + v * bytes_per_vec;

    // Prefetch next vector's sign bits
    if (v + 1 < num_vectors) {
      __builtin_prefetch(compressed_keys.sign_bits.data()
                         + (v + 1) * bytes_per_vec, 0, 1);
    }

    // Accumulate: sum of qp[j] where bit j is set
    float pos_sum = 0.0f;

#ifdef AVX2
    // Process 64 bits at a time using 64-bit word reads
    size_t j = 0;
    for (; j + 63 < ctx.proj_dim; j += 64) {
      uint64_t word;
      std::memcpy(&word, bits_ptr + j / 8, sizeof(uint64_t));

      // Extract each set bit and accumulate corresponding qp value
      // Use Brian Kernighan's bit trick for sparse words
      uint64_t tmp = word;
      while (tmp) {
        size_t bit_pos = __builtin_ctzll(tmp);  // Find lowest set bit
        pos_sum += query_proj[j + bit_pos];
        tmp &= tmp - 1;  // Clear lowest set bit
      }
    }
    // Remainder
    for (; j < ctx.proj_dim; ++j) {
      size_t bit_idx = j;
      int sign_bit = (bits_ptr[bit_idx / 8] >> (bit_idx % 8)) & 1;
      if (sign_bit) {
        pos_sum += query_proj[j];
      }
    }
#else
    for (size_t j = 0; j < ctx.proj_dim; ++j) {
      size_t bit_idx = j;
      int sign_bit = (bits_ptr[bit_idx / 8] >> (bit_idx % 8)) & 1;
      if (sign_bit) {
        pos_sum += query_proj[j];
      }
    }
#endif

    // correction = scale * (2 * pos_sum - total_sum)
    // This equals scale * Σ sign_j * qp[j]  since sign = 2*bit - 1
    corrections[v] = (2.0f * pos_sum - total_sum) * scale;
  }
}

// ─── Batch QJL Attention Correction ──────────────────────────────────────────

void qjl_attention_correction(
    const QJLContext& ctx,
    const float* __restrict__ queries,
    const QJLCompressed& compressed_keys,
    float* __restrict__ corrections,
    size_t num_queries,
    size_t num_keys) {
  for (size_t q = 0; q < num_queries; ++q) {
    qjl_inner_product(
        ctx,
        queries + q * ctx.input_dim,
        compressed_keys,
        corrections + q * num_keys,
        num_keys);
  }
}

}  // namespace turboquant
