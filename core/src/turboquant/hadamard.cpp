#include "turboquant/hadamard.h"
#include "turboquant/simd_utils.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <random>

namespace turboquant {

// ─── HadamardContext ─────────────────────────────────────────────────────────

std::unique_ptr<HadamardContext> HadamardContext::create(size_t dim, uint64_t seed) {
  // dim must be power of 2
  assert((dim & (dim - 1)) == 0 && "Hadamard dimension must be power of 2");

  auto ctx = std::make_unique<HadamardContext>();
  ctx->dim = dim;
  ctx->seed = seed;
  ctx->signs.resize(dim);

  // Generate random ±1 signs (Rademacher distribution)
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> dist(0, 1);
  for (size_t i = 0; i < dim; ++i) {
    ctx->signs[i] = dist(rng) ? 1.0f : -1.0f;
  }

  return ctx;
}

// ─── Fast Walsh-Hadamard Transform ───────────────────────────────────────────
// In-place, iterative butterfly implementation
// Complexity: O(n log n)
// After transform, each element is the inner product with a Walsh function

void fwht_inplace(float* data, size_t n) {
  assert((n & (n - 1)) == 0 && "FWHT requires power-of-2 length");

  for (size_t half = 1; half < n; half <<= 1) {
    for (size_t i = 0; i < n; i += half << 1) {
      for (size_t j = i; j < i + half; ++j) {
        float a = data[j];
        float b = data[j + half];
        data[j]        = a + b;   // butterfly add
        data[j + half] = a - b;   // butterfly subtract
      }
    }
  }
}

#ifdef AVX2
// AVX2 optimized FWHT — processes 8 butterflies at once
void fwht_avx2(float* data, size_t n) {
  assert((n & (n - 1)) == 0);

  for (size_t half = 1; half < n; half <<= 1) {
    for (size_t i = 0; i < n; i += half << 1) {
      size_t j = i;
      // Vectorized butterflies when half >= 8
      for (; j + 7 < i + half && half >= 8; j += 8) {
        __m256 a = _mm256_loadu_ps(&data[j]);
        __m256 b = _mm256_loadu_ps(&data[j + half]);
        _mm256_storeu_ps(&data[j],        _mm256_add_ps(a, b));
        _mm256_storeu_ps(&data[j + half], _mm256_sub_ps(a, b));
      }
      // Scalar remainder
      for (; j < i + half; ++j) {
        float a = data[j];
        float b = data[j + half];
        data[j]        = a + b;
        data[j + half] = a - b;
      }
    }
  }
}
#endif

#ifdef AVX512
// AVX-512 optimized FWHT — processes 16 butterflies at once
void fwht_avx512(float* data, size_t n) {
  assert((n & (n - 1)) == 0);

  for (size_t half = 1; half < n; half <<= 1) {
    for (size_t i = 0; i < n; i += half << 1) {
      size_t j = i;
      // Vectorized butterflies when half >= 16
      for (; j + 15 < i + half && half >= 16; j += 16) {
        __m512 a = _mm512_loadu_ps(&data[j]);
        __m512 b = _mm512_loadu_ps(&data[j + half]);
        _mm512_storeu_ps(&data[j],        _mm512_add_ps(a, b));
        _mm512_storeu_ps(&data[j + half], _mm512_sub_ps(a, b));
      }
      // Scalar remainder
      for (; j < i + half; ++j) {
        float a = data[j];
        float b = data[j + half];
        data[j]        = a + b;
        data[j + half] = a - b;
      }
    }
  }
}
#endif

// ─── Randomized Hadamard Transform ───────────────────────────────────────────
// D * H * x where D = diag(random signs), H = Hadamard matrix
// Normalized by 1/sqrt(n) to make it orthonormal

void randomized_hadamard(
    const HadamardContext& ctx,
    const float* __restrict__ input,
    float* __restrict__ output,
    size_t n) {
  assert(n == ctx.dim);
  const float norm = 1.0f / std::sqrt(static_cast<float>(n));

  // Step 1: Apply random sign flips (D * x) — AVX2 vectorized
#ifdef AVX2
  {
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
      __m256 v = _mm256_loadu_ps(input + i);
      __m256 s = _mm256_loadu_ps(ctx.signs.data() + i);
      _mm256_storeu_ps(output + i, _mm256_mul_ps(v, s));
    }
    for (; i < n; ++i) {
      output[i] = input[i] * ctx.signs[i];
    }
  }
#else
  for (size_t i = 0; i < n; ++i) {
    output[i] = input[i] * ctx.signs[i];
  }
#endif

  // Step 2: Apply FWHT in-place
#ifdef AVX512
  if (n >= 32) {
    fwht_avx512(output, n);
  } else
#endif
#ifdef AVX2
  if (n >= 16) {
    fwht_avx2(output, n);
  } else
#endif
  {
    fwht_inplace(output, n);
  }

  // Step 3: Normalize — AVX2 vectorized
#ifdef AVX2
  {
    __m256 vnorm = _mm256_set1_ps(norm);
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
      __m256 v = _mm256_loadu_ps(output + i);
      _mm256_storeu_ps(output + i, _mm256_mul_ps(v, vnorm));
    }
    for (; i < n; ++i) {
      output[i] *= norm;
    }
  }
#else
  for (size_t i = 0; i < n; ++i) {
    output[i] *= norm;
  }
#endif
}

// Inverse is the same operation (Hadamard is its own inverse)
void randomized_hadamard_inverse(
    const HadamardContext& ctx,
    const float* __restrict__ input,
    float* __restrict__ output,
    size_t n) {
  assert(n == ctx.dim);
  const float norm = 1.0f / std::sqrt(static_cast<float>(n));

  // Step 1: Copy to output
  std::memcpy(output, input, n * sizeof(float));

  // Step 2: Apply FWHT in-place (H is symmetric and involutory up to scale)
#ifdef AVX512
  if (n >= 32) {
    fwht_avx512(output, n);
  } else
#endif
#ifdef AVX2
  if (n >= 16) {
    fwht_avx2(output, n);
  } else
#endif
  {
    fwht_inplace(output, n);
  }

  // Step 3: Normalize and undo sign flips (D^T = D since D is diagonal ±1)
#ifdef AVX2
  {
    __m256 vnorm = _mm256_set1_ps(norm);
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
      __m256 v = _mm256_loadu_ps(output + i);
      __m256 s = _mm256_loadu_ps(ctx.signs.data() + i);
      _mm256_storeu_ps(output + i, _mm256_mul_ps(_mm256_mul_ps(v, vnorm), s));
    }
    for (; i < n; ++i) {
      output[i] *= norm * ctx.signs[i];
    }
  }
#else
  for (size_t i = 0; i < n; ++i) {
    output[i] *= norm * ctx.signs[i];
  }
#endif
}

// Batch version for multiple vectors
void randomized_hadamard_batch(
    const HadamardContext& ctx,
    const float* __restrict__ input,
    float* __restrict__ output,
    size_t batch_size,
    size_t dim) {
  assert(dim == ctx.dim);
  for (size_t b = 0; b < batch_size; ++b) {
    randomized_hadamard(ctx, input + b * dim, output + b * dim, dim);
  }
}

}  // namespace turboquant
