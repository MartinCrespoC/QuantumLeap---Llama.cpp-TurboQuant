#pragma once
// ============================================================================
// Randomized Hadamard Transform (Preconditioning)
// Based on Google Research TurboQuant (arXiv:2504.19874)
//
// Step 1 of TurboQuant: Randomly rotate input vectors using a fast
// Walsh-Hadamard Transform with random sign flips. This makes coordinates
// nearly independent with a concentrated Beta distribution, enabling
// efficient per-coordinate scalar quantization.
//
// Complexity: O(d log d) per vector, where d = dimension
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace turboquant {

// Random sign vector for Hadamard preconditioning
// D = diag(±1) random diagonal, applied before Hadamard
struct HadamardContext {
  size_t dim;                    // Must be power of 2
  std::vector<float> signs;     // Random ±1 signs, length = dim
  uint64_t seed;                // RNG seed for reproducibility

  // Create context with given dimension and seed
  static std::unique_ptr<HadamardContext> create(size_t dim, uint64_t seed = 42);
};

// Fast Walsh-Hadamard Transform (in-place, unnormalized)
// Transforms data[] of length n (must be power of 2)
// After transform, divide by sqrt(n) for orthonormal
void fwht_inplace(float* data, size_t n);

// Randomized Hadamard Transform: D * H * x
// Applies random sign flips then Hadamard transform
// This is the preconditioning step from the TurboQuant paper
void randomized_hadamard(
    const HadamardContext& ctx,
    const float* __restrict__ input,
    float* __restrict__ output,
    size_t n);

// Inverse: H^T * D^T * x = H * D * x (since H and D are symmetric)
// Used for decoding/reconstruction
void randomized_hadamard_inverse(
    const HadamardContext& ctx,
    const float* __restrict__ input,
    float* __restrict__ output,
    size_t n);

// Batch version: rotate multiple vectors at once
// input/output: [batch_size, dim]
void randomized_hadamard_batch(
    const HadamardContext& ctx,
    const float* __restrict__ input,
    float* __restrict__ output,
    size_t batch_size,
    size_t dim);

// SIMD-optimized variants
#ifdef AVX512
void fwht_avx512(float* data, size_t n);
#endif
#ifdef AVX2
void fwht_avx2(float* data, size_t n);
#endif

}  // namespace turboquant
