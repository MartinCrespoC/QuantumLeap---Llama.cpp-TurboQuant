#pragma once
// ============================================================================
// QJL — Quantized Johnson-Lindenstrauss Transform
// Based on Google Research (AAAI 2025, used in TurboQuant)
//
// Stage 2 of TurboQuant: Applied to the residual error from PolarQuant.
// Uses a random JL projection followed by 1-bit sign quantization.
// This creates an UNBIASED inner product estimator with zero memory overhead.
//
// Key properties:
// - 1 bit per dimension (minimal storage)
// - Zero quantization constants (no scales, no zero-points)
// - Unbiased inner product estimation when combined with full-precision query
// - Corrects the bias introduced by MSE-optimal PolarQuant
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace turboquant {

// QJL random projection context
// Stores the random matrix R for consistent projection
struct QJLContext {
  size_t input_dim;       // Original vector dimension
  size_t proj_dim;        // Projection dimension (= input_dim for full)
  uint64_t seed;          // RNG seed for reproducibility
  std::vector<float> R;   // Random projection matrix [proj_dim, input_dim]
                          // Entries are ±1/√proj_dim (Rademacher)

  static std::unique_ptr<QJLContext> create(
      size_t input_dim, size_t proj_dim, uint64_t seed = 137);
};

// QJL compressed representation — just sign bits
struct QJLCompressed {
  std::vector<uint8_t> sign_bits;  // Packed sign bits, 8 signs per byte
  size_t num_vectors;
  size_t proj_dim;

  // Bits per original dimension
  float bits_per_element() const {
    return static_cast<float>(proj_dim) / proj_dim;  // 1 bit per proj dim
  }
};

// ─── QJL Encode ──────────────────────────────────────────────────────────────
// Project residual vector through random matrix, then take sign bits
// sign_bits[j] = sign(R[j] · residual)

QJLCompressed qjl_encode(
    const QJLContext& ctx,
    const float* __restrict__ residuals,  // [num_vectors, input_dim]
    size_t num_vectors);

// ─── QJL Inner Product Estimation ────────────────────────────────────────────
// Estimate <query, key_residual> using full-precision query and QJL-compressed key
// This is the unbiased correction term added to PolarQuant's attention score
//
// Formula: estimate = ||residual||_est * Σ_j sign_j * (R[j] · query) / proj_dim
// Simplified: we compute <query_projected, sign_bits> scaled appropriately

void qjl_inner_product(
    const QJLContext& ctx,
    const float* __restrict__ query,         // [dim] full precision
    const QJLCompressed& compressed_keys,
    float* __restrict__ corrections,         // [num_vectors] output
    size_t num_vectors);

// ─── Batch QJL Inner Product ─────────────────────────────────────────────────
// For computing attention: query against all compressed KV entries

void qjl_attention_correction(
    const QJLContext& ctx,
    const float* __restrict__ queries,       // [num_queries, dim]
    const QJLCompressed& compressed_keys,
    float* __restrict__ corrections,         // [num_queries, num_keys]
    size_t num_queries,
    size_t num_keys);

}  // namespace turboquant
