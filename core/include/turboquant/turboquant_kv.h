#pragma once
// ============================================================================
// TurboQuant KV Cache Pipeline
// Based on Google Research TurboQuant (arXiv:2504.19874, ICLR 2026)
//
// Full pipeline combining:
//   Stage 1: Hadamard random rotation (preconditioning)
//   Stage 2: PolarQuant (polar decomposition + angle quantization)
//   Stage 3: QJL (1-bit sign quantization on residual for unbiased correction)
//
// Results from paper:
//   - 3.5 bits/channel: absolute quality neutrality (zero loss)
//   - 2.5 bits/channel: marginal quality degradation
//   - 4-bit mode: 8x speedup over FP32 on H100 GPU
//   - KV cache reduced by 6x+ with perfect downstream accuracy
//
// This header provides the high-level API for compressing and using
// the KV cache during LLM inference.
// ============================================================================

#include "turboquant/hadamard.h"
#include "turboquant/polarquant.h"
#include "turboquant/qjl.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace turboquant {

// TurboQuant compression mode
enum class TQMode : uint8_t {
  kTQ2 = 2,   // ~2.5 bits/channel — marginal quality loss, max compression
  kTQ3 = 3,   // ~3.5 bits/channel — zero quality loss (recommended)
  kTQ4 = 4,   // ~4 bits/channel   — 8x speedup, excellent quality
};

// Full TurboQuant context — holds all state for encode/decode
struct TurboQuantContext {
  std::unique_ptr<HadamardContext> hadamard;
  std::unique_ptr<QJLContext> qjl;
  PolarQuantConfig polar_config;
  TQMode mode;
  size_t head_dim;        // Per-head dimension (e.g., 128)
  size_t padded_dim;      // Padded to power of 2 for Hadamard

  static std::unique_ptr<TurboQuantContext> create(
      size_t head_dim,
      TQMode mode = TQMode::kTQ3,
      uint64_t seed = 42);
};

// Compressed KV cache entry for one attention head
struct TQCompressedKV {
  PolarCompressed polar;       // Stage 2: PolarQuant compressed
  QJLCompressed qjl_residual;  // Stage 3: QJL sign bits on residual
  size_t seq_len;
  size_t head_dim;
  TQMode mode;

  // Memory usage in bytes
  size_t memory_bytes() const;

  // Compression ratio vs FP32
  float compression_ratio() const {
    size_t fp32_bytes = seq_len * head_dim * sizeof(float);
    return static_cast<float>(fp32_bytes) / memory_bytes();
  }

  // Effective bits per element
  float bits_per_element() const;

  // Pre-allocate internal buffers for max_seq_len tokens to avoid
  // reallocation during autoregressive generation (token-by-token append)
  void reserve(size_t max_seq_len, size_t head_dim, size_t proj_dim,
               uint8_t angle_bits, uint8_t radius_bits);
};

// ─── Encode: Compress KV cache ───────────────────────────────────────────────
// Input: raw KV embeddings [seq_len, head_dim] in FP32
// Output: TQCompressedKV with polar + QJL compressed data

TQCompressedKV turboquant_kv_encode(
    const TurboQuantContext& ctx,
    const float* __restrict__ kv_data,   // [seq_len, head_dim]
    size_t seq_len);

// ─── Decode: Reconstruct KV cache (for debugging/validation) ────────────────

void turboquant_kv_decode(
    const TurboQuantContext& ctx,
    const TQCompressedKV& compressed,
    float* __restrict__ output,          // [seq_len, head_dim]
    size_t seq_len);

// ─── Attention: Compute attention scores directly on compressed KV ───────────
// This is where the 8x speedup comes from.
// Computes: attention_logits[i] = <Q[i], K_compressed[j]> for all j
// Uses PolarQuant dot product + QJL correction for unbiased result.

void turboquant_attention_scores(
    const TurboQuantContext& ctx,
    const float* __restrict__ queries,         // [num_queries, head_dim]
    const TQCompressedKV& compressed_keys,
    float* __restrict__ attention_logits,      // [num_queries, seq_len]
    size_t num_queries);

// ─── Incremental: Append new KV token to compressed cache ───────────────────
// For autoregressive generation — compress one new token and append

void turboquant_kv_append(
    const TurboQuantContext& ctx,
    TQCompressedKV& compressed,
    const float* __restrict__ new_kv,   // [1, head_dim]
    size_t new_seq_pos);

// ─── Memory placement for mixed VRAM/RAM offload ─────────────────────────────

enum class KVPlacement : uint8_t {
  kCPU = 0,   // KV cache lives in system RAM (attention via SIMD)
  kGPU = 1,   // KV cache lives in GPU VRAM (attention via CUDA kernels)
};

// ─── CUDA variants (declared unconditionally for mixed-offload dispatch) ─────

#if defined(TURBOQUANT_CUDA) || defined(TURBOQUANT_HIP)

TQCompressedKV turboquant_kv_encode_cuda(
    const TurboQuantContext& ctx,
    const float* __restrict__ kv_data,
    size_t seq_len);

void turboquant_attention_scores_cuda(
    const TurboQuantContext& ctx,
    const float* __restrict__ queries,
    const TQCompressedKV& compressed_keys,
    float* __restrict__ attention_logits,
    size_t num_queries);

#endif

}  // namespace turboquant
