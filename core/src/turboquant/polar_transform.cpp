#include "turboquant/turboquant.h"
#include "turboquant/simd_utils.h"

#include <cmath>
#include <cpuid.h>

// Assembly-implemented SIMD functions (unmangled C symbols from .S files)
#ifdef AVX512
extern "C" void polar_transform_avx512(
    const float* x, const float* y, float* magnitude, float* angle, size_t n);
#endif
#ifdef AVX2
extern "C" void polar_transform_avx2(
    const float* x, const float* y, float* magnitude, float* angle, size_t n);
#endif

namespace turboquant {

// Scalar fallback implementation
void polar_transform_scalar(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ magnitude,
    float* __restrict__ angle,
    size_t n) {
  for (size_t i = 0; i < n; ++i) {
    magnitude[i] = std::sqrt(x[i] * x[i] + y[i] * y[i]);
    angle[i] = std::atan2(y[i], x[i]);
  }
}

// Runtime CPU feature detection
bool has_avx512() {
  unsigned int eax, ebx, ecx, edx;
  if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
    return false;
  }
  // Check AVX-512 Foundation (bit 16 of EBX)
  return (ebx >> 16) & 1;
}

bool has_avx2() {
  unsigned int eax, ebx, ecx, edx;
  if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
    return false;
  }
  // Check AVX2 (bit 5 of EBX)
  return (ebx >> 5) & 1;
}

// Auto-dispatch: selects best implementation at runtime
// Priority: AVX-512 → AVX2 → scalar
void polar_transform(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ magnitude,
    float* __restrict__ angle,
    size_t n) {
#ifdef AVX512
  if (has_avx512() && n >= 16) {
    // Process 16 elements at a time with AVX-512
    size_t vec_n = (n / 16) * 16;
    polar_transform_avx512(x, y, magnitude, angle, vec_n);
    // Handle remainder with scalar
    if (vec_n < n) {
      polar_transform_scalar(x + vec_n, y + vec_n,
                             magnitude + vec_n, angle + vec_n,
                             n - vec_n);
    }
    return;
  }
#endif
#ifdef AVX2
  if (has_avx2() && n >= 8) {
    // Process 8 elements at a time with AVX2
    size_t vec_n = (n / 8) * 8;
    polar_transform_avx2(x, y, magnitude, angle, vec_n);
    if (vec_n < n) {
      polar_transform_scalar(x + vec_n, y + vec_n,
                             magnitude + vec_n, angle + vec_n,
                             n - vec_n);
    }
    return;
  }
#endif
  polar_transform_scalar(x, y, magnitude, angle, n);
}

}  // namespace turboquant
