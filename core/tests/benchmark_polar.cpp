#include "turboquant/turboquant.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace turboquant;

// Generate random float array
static void fill_random(float* data, size_t n, float min_val, float max_val) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(min_val, max_val);
  for (size_t i = 0; i < n; ++i) {
    data[i] = dist(rng);
  }
}

// Benchmark polar transform
static void bench_polar_transform(size_t n, int iterations) {
  // Allocate aligned buffers
  auto* x = static_cast<float*>(aligned_alloc(64, n * sizeof(float)));
  auto* y = static_cast<float*>(aligned_alloc(64, n * sizeof(float)));
  auto* mag = static_cast<float*>(aligned_alloc(64, n * sizeof(float)));
  auto* ang = static_cast<float*>(aligned_alloc(64, n * sizeof(float)));

  fill_random(x, n, -10.0f, 10.0f);
  fill_random(y, n, -10.0f, 10.0f);

  // Warmup
  polar_transform(x, y, mag, ang, n);

  // Benchmark scalar
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    polar_transform_scalar(x, y, mag, ang, n);
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  double scalar_ms = std::chrono::duration<double, std::milli>(t1 - t0).count()
                     / iterations;

  // Benchmark auto-dispatch (AVX-512 if available)
  t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    polar_transform(x, y, mag, ang, n);
  }
  t1 = std::chrono::high_resolution_clock::now();
  double auto_ms = std::chrono::duration<double, std::milli>(t1 - t0).count()
                   / iterations;

  // Verify correctness
  std::vector<float> mag_ref(n), ang_ref(n);
  polar_transform_scalar(x, y, mag_ref.data(), ang_ref.data(), n);
  polar_transform(x, y, mag, ang, n);

  float max_mag_err = 0, max_ang_err = 0;
  for (size_t i = 0; i < n; ++i) {
    max_mag_err = std::max(max_mag_err, std::abs(mag[i] - mag_ref[i]));
    max_ang_err = std::max(max_ang_err, std::abs(ang[i] - ang_ref[i]));
  }

  printf("PolarTransform [n=%zu, iters=%d]:\n", n, iterations);
  printf("  Scalar:      %.3f ms (%.1f Melems/s)\n",
         scalar_ms, n / (scalar_ms * 1000.0));
  printf("  Auto (AVX?): %.3f ms (%.1f Melems/s)\n",
         auto_ms, n / (auto_ms * 1000.0));
  printf("  Speedup:     %.1fx\n", scalar_ms / auto_ms);
  printf("  Max mag err: %.2e\n", max_mag_err);
  printf("  Max ang err: %.2e\n", max_ang_err);
  printf("  AVX-512:     %s\n", has_avx512() ? "YES" : "NO");
  printf("\n");

  free(x); free(y); free(mag); free(ang);
}

// Benchmark residual quantization
static void bench_residual_quant(size_t n, int iterations) {
  auto* data = static_cast<float*>(aligned_alloc(64, n * sizeof(float)));
  fill_random(data, n, -1.0f, 1.0f);

  // Benchmark INT2
  auto t0 = std::chrono::high_resolution_clock::now();
  QuantResult result;
  for (int i = 0; i < iterations; ++i) {
    result = residual_quantize(data, n, QuantBits::kInt2, 128, 3);
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  double int2_ms = std::chrono::duration<double, std::milli>(t1 - t0).count()
                   / iterations;

  printf("ResidualQuant INT2 [n=%zu, iters=%d]:\n", n, iterations);
  printf("  Time:       %.3f ms\n", int2_ms);
  printf("  MSE:        %.6f\n", result.mse);
  printf("  Max Error:  %.6f\n", result.max_error);
  printf("  Packed size: %zu bytes (%.2f bits/element)\n",
         result.data.size(), result.data.size() * 8.0 / n);
  printf("  Compression: %.1fx vs FP32\n", n * 4.0 / result.data.size());
  printf("\n");

  delete[] result.meta.scales;
  delete[] result.meta.zero_points;
  free(data);
}

int main(int argc, char* argv[]) {
  (void)argc;  // Unused
  (void)argv;  // Unused
  printf("=== TurboQuant Benchmark Suite ===\n");
  printf("CPU: %s\n", has_avx512() ? "AVX-512 DETECTED" : "No AVX-512");
  printf("\n");

  init_lookup_tables();

  // Polar transform benchmarks
  bench_polar_transform(1024, 10000);
  bench_polar_transform(1024 * 1024, 100);
  bench_polar_transform(16 * 1024 * 1024, 10);

  // Quantization benchmarks
  bench_residual_quant(1024 * 1024, 10);
  bench_residual_quant(16 * 1024 * 1024, 3);

  destroy_lookup_tables();

  printf("=== Benchmark Complete ===\n");
  return 0;
}
