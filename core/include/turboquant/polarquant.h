#pragma once
// ============================================================================
// PolarQuant — Multi-dimensional Polar Coordinate Quantization
// Based on Google Research PolarQuant (arXiv:2502.02617)
//
// After random preconditioning (Hadamard), vectors are converted to
// d-dimensional polar coordinates: radius r and (d-1) angles θ₁..θ_{d-1}.
// Key insight: after rotation, angles follow a concentrated Beta distribution
// with analytically known form → NO normalization constants needed.
// This eliminates the memory overhead of storing per-block scales/zero-points.
//
// Compression: KV cache from FP16 → 3-4 bits with near-zero quality loss
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <vector>

namespace turboquant {

// Quantization parameters for angle quantization
struct PolarQuantConfig {
  size_t dim;           // Vector dimension (must be power of 2)
  uint8_t angle_bits;   // Bits per angle (2-4 typical, 3 = sweet spot)
  uint8_t radius_bits;  // Bits for radius (4-8 typical)
  size_t block_size;    // Vectors processed together (for radius grouping)
};

// Compressed polar representation of a vector block
struct PolarCompressed {
  std::vector<uint8_t> packed_angles;   // Bit-packed quantized angles
  std::vector<uint8_t> packed_radii;    // Bit-packed quantized radii
  size_t num_vectors;
  size_t dim;
  PolarQuantConfig config;

  // Effective bits per element
  float bits_per_element() const {
    float angle_total = static_cast<float>(dim - 1) * config.angle_bits;
    float radius_total = static_cast<float>(config.radius_bits);
    return (angle_total + radius_total) / dim;
  }
};

// ─── Recursive Polar Decomposition ────────────────────────────────────────────
// Convert d-dimensional vector to polar coordinates:
//   v = [x₁, x₂, ..., x_d]
//   r = ||v||
//   θ₁ = arccos(x₁ / r)
//   θ₂ = arccos(x₂ / (r * sin(θ₁)))
//   ...recursive for higher dims
//
// This is the core of PolarQuant's compression.

// Scalar implementation
void polar_decompose(
    const float* __restrict__ vec,
    float* __restrict__ radius,      // scalar output
    float* __restrict__ angles,      // (dim-1) angles output
    size_t dim);

// Inverse: reconstruct vector from polar coordinates
void polar_reconstruct(
    float radius,
    const float* __restrict__ angles,
    float* __restrict__ vec,         // dim elements output
    size_t dim);

// ─── Angle Quantization ──────────────────────────────────────────────────────
// After Hadamard rotation, angle θ_k follows Beta((d-k)/2, (d-k)/2)
// distribution centered at π/2 with tight concentration.
// We use optimal scalar quantizer for this known distribution.

// Quantize a single angle given its index k and dimension d
uint8_t quantize_angle(float angle, size_t k, size_t d, uint8_t bits);

// Dequantize angle back to float
float dequantize_angle(uint8_t quantized, size_t k, size_t d, uint8_t bits);

// Quantize radius (simple min-max within block)
uint8_t quantize_radius(float radius, float rmin, float rmax, uint8_t bits);
float dequantize_radius(uint8_t quantized, float rmin, float rmax, uint8_t bits);

// ─── Full PolarQuant Encode/Decode ───────────────────────────────────────────
// Encodes a batch of vectors using PolarQuant method:
// 1. Hadamard preconditioning (caller must do this first)
// 2. Polar decomposition
// 3. Angle quantization using Beta distribution
// 4. Radius quantization

PolarCompressed polarquant_encode(
    const float* __restrict__ vectors,  // [num_vectors, dim] pre-rotated
    size_t num_vectors,
    const PolarQuantConfig& config);

void polarquant_decode(
    const PolarCompressed& compressed,
    float* __restrict__ output,         // [num_vectors, dim]
    size_t num_vectors);

// Get the residual error after PolarQuant (for QJL stage 2)
void polarquant_residual(
    const float* __restrict__ original,  // [num_vectors, dim] pre-rotated
    const PolarCompressed& compressed,
    float* __restrict__ residual,        // [num_vectors, dim]
    size_t num_vectors);

}  // namespace turboquant
