#include "turboquant/polarquant.h"
#include "turboquant/simd_utils.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

#ifdef AVX2
#include <immintrin.h>
#endif

namespace turboquant {

// ─── AVX2 vectorized squared norm ───────────────────────────────────────────

static inline float vec_norm_sq(const float* __restrict__ vec, size_t dim) {
#ifdef AVX2
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  size_t i = 0;
  for (; i + 15 < dim; i += 16) {
    __m256 v0 = _mm256_loadu_ps(vec + i);
    acc0 = _mm256_fmadd_ps(v0, v0, acc0);
    __m256 v1 = _mm256_loadu_ps(vec + i + 8);
    acc1 = _mm256_fmadd_ps(v1, v1, acc1);
  }
  acc0 = _mm256_add_ps(acc0, acc1);
  for (; i + 7 < dim; i += 8) {
    __m256 v0 = _mm256_loadu_ps(vec + i);
    acc0 = _mm256_fmadd_ps(v0, v0, acc0);
  }
  __m128 hi = _mm256_extractf128_ps(acc0, 1);
  __m128 lo = _mm256_castps256_ps128(acc0);
  __m128 sum = _mm_add_ps(lo, hi);
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  float result = _mm_cvtss_f32(sum);
  for (; i < dim; ++i) {
    result += vec[i] * vec[i];
  }
  return result;
#else
  float r_sq = 0.0f;
  for (size_t i = 0; i < dim; ++i) {
    r_sq += vec[i] * vec[i];
  }
  return r_sq;
#endif
}

// ─── Recursive Polar Decomposition ───────────────────────────────────────────
// Converts d-dimensional Cartesian vector to polar coordinates:
//   radius r = ||v||
//   angles θ_k = arccos(v_k / sqrt(v_k² + v_{k+1}² + ... + v_d²))
//
// After Hadamard preconditioning, θ_k ~ Beta((d-k)/2, (d-k)/2)
// centered at π/2 with tight concentration.

void polar_decompose(
    const float* __restrict__ vec,
    float* __restrict__ radius,
    float* __restrict__ angles,
    size_t dim) {
  assert(dim >= 2);

  // Compute radius = ||vec|| (AVX2 vectorized)
  float r_sq = vec_norm_sq(vec, dim);
  *radius = std::sqrt(r_sq);

  if (*radius < 1e-10f) {
    // Zero vector — all angles are π/2
    for (size_t i = 0; i < dim - 1; ++i) {
      angles[i] = static_cast<float>(M_PI) * 0.5f;
    }
    return;
  }

  // Recursive polar decomposition
  // For each dimension k (0..d-2):
  //   remaining_norm = sqrt(v_k² + v_{k+1}² + ... + v_{d-1}²)
  //   θ_k = arccos(v_k / remaining_norm)
  float remaining_sq = r_sq;
  for (size_t k = 0; k < dim - 1; ++k) {
    float remaining_norm = std::sqrt(remaining_sq);
    if (remaining_norm < 1e-10f) {
      // Remaining dimensions are zero
      for (size_t j = k; j < dim - 1; ++j) {
        angles[j] = static_cast<float>(M_PI) * 0.5f;
      }
      break;
    }
    float cos_theta = std::clamp(vec[k] / remaining_norm, -1.0f, 1.0f);
    angles[k] = std::acos(cos_theta);
    remaining_sq -= vec[k] * vec[k];
    if (remaining_sq < 0.0f) remaining_sq = 0.0f;
  }
}

// ─── Polar Reconstruction ────────────────────────────────────────────────────
// Inverse of polar_decompose: reconstruct vector from (radius, angles)
//   v_0 = r * cos(θ_0)
//   v_1 = r * sin(θ_0) * cos(θ_1)
//   v_2 = r * sin(θ_0) * sin(θ_1) * cos(θ_2)
//   ...
//   v_{d-1} = r * sin(θ_0) * sin(θ_1) * ... * sin(θ_{d-2})

void polar_reconstruct(
    float radius,
    const float* __restrict__ angles,
    float* __restrict__ vec,
    size_t dim) {
  assert(dim >= 2);

  float running_sin = radius;
  for (size_t k = 0; k < dim - 1; ++k) {
    vec[k] = running_sin * std::cos(angles[k]);
    running_sin *= std::sin(angles[k]);
  }
  vec[dim - 1] = running_sin;  // Last component: product of all sines
}

// ─── Angle Quantization ──────────────────────────────────────────────────────
// After Hadamard rotation, angle θ_k follows Beta((d-k)/2, (d-k)/2)
// centered at π/2.
//
// For uniform quantization on [0, π], concentration means most values
// are near π/2. We use a non-uniform quantizer that allocates more
// levels near the center. Specifically:
//   quantization_grid[i] = π/2 + (i - (levels-1)/2) * step_size
// where step_size adapts based on the Beta distribution's std dev.
//
// Std dev of Beta(a,a) on [0,π] = π / (2 * sqrt(2a+1))

static float beta_stddev(size_t k, size_t d) {
  // a = (d-k)/2 for the Beta distribution parameter
  float a = static_cast<float>(d - k) * 0.5f;
  if (a < 0.5f) a = 0.5f;
  // Std dev of Beta(a,a) mapped to [0, π]
  return static_cast<float>(M_PI) / (2.0f * std::sqrt(2.0f * a + 1.0f));
}

uint8_t quantize_angle(float angle, size_t k, size_t d, uint8_t bits) {
  int levels = 1 << bits;
  float center = static_cast<float>(M_PI) * 0.5f;
  float sigma = beta_stddev(k, d);
  // Cover ±3σ range centered at π/2
  float range = 6.0f * sigma;
  float lo = std::max(0.0f, center - range * 0.5f);
  float hi = std::min(static_cast<float>(M_PI), center + range * 0.5f);
  float step = (hi - lo) / (levels - 1);

  if (step < 1e-10f) return static_cast<uint8_t>(levels / 2);

  float clamped = std::clamp(angle, lo, hi);
  int q = static_cast<int>(std::round((clamped - lo) / step));
  return static_cast<uint8_t>(std::clamp(q, 0, levels - 1));
}

float dequantize_angle(uint8_t quantized, size_t k, size_t d, uint8_t bits) {
  int levels = 1 << bits;
  float center = static_cast<float>(M_PI) * 0.5f;
  float sigma = beta_stddev(k, d);
  float range = 6.0f * sigma;
  float lo = std::max(0.0f, center - range * 0.5f);
  float hi = std::min(static_cast<float>(M_PI), center + range * 0.5f);
  float step = (hi - lo) / (levels - 1);

  return lo + static_cast<float>(quantized) * step;
}

// ─── Radius Quantization ─────────────────────────────────────────────────────
// Simple min-max uniform quantization within a block

uint8_t quantize_radius(float radius, float rmin, float rmax, uint8_t bits) {
  int levels = (1 << bits) - 1;
  float range = rmax - rmin;
  if (range < 1e-10f) return 0;
  float normalized = (radius - rmin) / range;
  int q = static_cast<int>(std::round(normalized * levels));
  return static_cast<uint8_t>(std::clamp(q, 0, levels));
}

float dequantize_radius(uint8_t quantized, float rmin, float rmax, uint8_t bits) {
  int levels = (1 << bits) - 1;
  float range = rmax - rmin;
  return rmin + (static_cast<float>(quantized) / levels) * range;
}

// ─── Bit Packing Helpers ─────────────────────────────────────────────────────

static void pack_values(const std::vector<uint8_t>& values, uint8_t bits,
                        std::vector<uint8_t>& packed) {
  size_t total_bits = values.size() * bits;
  packed.resize((total_bits + 7) / 8, 0);

  size_t bit_pos = 0;
  for (size_t i = 0; i < values.size(); ++i) {
    uint8_t val = values[i] & ((1 << bits) - 1);
    size_t byte_idx = bit_pos / 8;
    size_t bit_offset = bit_pos % 8;

    // Write value across byte boundary if needed
    packed[byte_idx] |= (val << bit_offset) & 0xFF;
    if (bit_offset + bits > 8 && byte_idx + 1 < packed.size()) {
      packed[byte_idx + 1] |= val >> (8 - bit_offset);
    }
    bit_pos += bits;
  }
}

static uint8_t unpack_value(const std::vector<uint8_t>& packed, size_t index,
                            uint8_t bits) {
  size_t bit_pos = index * bits;
  size_t byte_idx = bit_pos / 8;
  size_t bit_offset = bit_pos % 8;
  uint8_t mask = (1 << bits) - 1;

  uint16_t raw = packed[byte_idx];
  if (byte_idx + 1 < packed.size()) {
    raw |= static_cast<uint16_t>(packed[byte_idx + 1]) << 8;
  }
  return static_cast<uint8_t>((raw >> bit_offset) & mask);
}

// ─── Full PolarQuant Encode ──────────────────────────────────────────────────

PolarCompressed polarquant_encode(
    const float* __restrict__ vectors,
    size_t num_vectors,
    const PolarQuantConfig& config) {
  PolarCompressed result;
  result.num_vectors = num_vectors;
  result.dim = config.dim;
  result.config = config;

  const size_t dim = config.dim;
  const size_t num_angles = dim - 1;

  // Temporary storage for all radii and angles
  std::vector<float> radii(num_vectors);
  std::vector<float> all_angles(num_vectors * num_angles);

  // Decompose each vector (with prefetch)
  for (size_t v = 0; v < num_vectors; ++v) {
    if (v + 1 < num_vectors) {
      __builtin_prefetch(vectors + (v + 1) * dim, 0, 1);
    }
    polar_decompose(
        vectors + v * dim,
        &radii[v],
        all_angles.data() + v * num_angles,
        dim);
  }

  // Quantize radii (per-block min-max)
  std::vector<uint8_t> quantized_radii(num_vectors);
  // Find global min/max radius (or per-block if block_size > 0)
  float rmin = *std::min_element(radii.begin(), radii.end());
  float rmax = *std::max_element(radii.begin(), radii.end());
  for (size_t v = 0; v < num_vectors; ++v) {
    quantized_radii[v] = quantize_radius(radii[v], rmin, rmax, config.radius_bits);
  }

  // Quantize angles (using Beta distribution knowledge)
  std::vector<uint8_t> quantized_angles(num_vectors * num_angles);
  for (size_t v = 0; v < num_vectors; ++v) {
    for (size_t k = 0; k < num_angles; ++k) {
      quantized_angles[v * num_angles + k] =
          quantize_angle(all_angles[v * num_angles + k], k, dim, config.angle_bits);
    }
  }

  // Pack into bit-packed format
  pack_values(quantized_angles, config.angle_bits, result.packed_angles);
  pack_values(quantized_radii, config.radius_bits, result.packed_radii);

  return result;
}

// ─── Full PolarQuant Decode ──────────────────────────────────────────────────

void polarquant_decode(
    const PolarCompressed& compressed,
    float* __restrict__ output,
    size_t num_vectors) {
  const size_t dim = compressed.dim;
  const size_t num_angles = dim - 1;
  const auto& config = compressed.config;

  // Find radius range (stored implicitly — we store rmin/rmax in first 8 bytes)
  // For now, decode all radii first to find range, then reconstruct
  // TODO: store rmin/rmax as metadata in PolarCompressed

  float rmin = 0.0f, rmax = 1.0f;  // Placeholder — will be stored in metadata

  for (size_t v = 0; v < num_vectors; ++v) {
    // Dequantize radius
    uint8_t qr = unpack_value(compressed.packed_radii, v, config.radius_bits);
    float radius = dequantize_radius(qr, rmin, rmax, config.radius_bits);

    // Dequantize angles
    float angles[512];  // Max dim supported
    for (size_t k = 0; k < num_angles; ++k) {
      uint8_t qa = unpack_value(compressed.packed_angles,
                                v * num_angles + k, config.angle_bits);
      angles[k] = dequantize_angle(qa, k, dim, config.angle_bits);
    }

    // Reconstruct vector
    polar_reconstruct(radius, angles, output + v * dim, dim);
  }
}

// ─── Compute Residual ────────────────────────────────────────────────────────

void polarquant_residual(
    const float* __restrict__ original,
    const PolarCompressed& compressed,
    float* __restrict__ residual,
    size_t num_vectors) {
  const size_t dim = compressed.dim;

  // Decode compressed
  std::vector<float> decoded(num_vectors * dim);
  polarquant_decode(compressed, decoded.data(), num_vectors);

  // Compute residual = original - decoded (AVX2 vectorized)
  size_t total = num_vectors * dim;
#ifdef AVX2
  size_t i = 0;
  for (; i + 7 < total; i += 8) {
    __m256 orig = _mm256_loadu_ps(original + i);
    __m256 dec = _mm256_loadu_ps(decoded.data() + i);
    _mm256_storeu_ps(residual + i, _mm256_sub_ps(orig, dec));
  }
  for (; i < total; ++i) {
    residual[i] = original[i] - decoded[i];
  }
#else
  for (size_t i = 0; i < total; ++i) {
    residual[i] = original[i] - decoded[i];
  }
#endif
}

}  // namespace turboquant
