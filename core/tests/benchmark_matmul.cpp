#include "turboquant/turboquant.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace turboquant;

int main() {
  printf("=== TurboQuant MatMul Benchmark ===\n");
#if defined(TURBOQUANT_CUDA) || defined(TURBOQUANT_HIP)
  printf("CUDA devices: %d\n", get_cuda_device_count());

  if (get_cuda_device_count() > 0) {
    size_t free_mem = get_cuda_free_memory(0);
    int cc = get_cuda_compute_capability(0);
    printf("GPU free memory: %.1f MB\n", free_mem / (1024.0 * 1024.0));
    printf("Compute capability: %d.%d\n", cc / 10, cc % 10);
  }
#else
  printf("GPU: not available (CPU-only build)\n");
#endif

  // TODO: Add CUDA matmul benchmarks
  // - INT2 matmul vs cuBLAS FP16
  // - INT4 matmul vs cuBLAS FP16
  // - Compressed attention vs standard attention
  // - KV cache compress/decompress throughput

  printf("\n[TODO] CUDA benchmarks not yet implemented.\n");
  printf("Build with CUDA support and implement kernel benchmarks.\n");

  return 0;
}
