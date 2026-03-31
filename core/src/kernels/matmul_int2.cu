#include "turboquant/cuda_utils.cuh"
#if defined(TURBOQUANT_HIP) || defined(__HIP_PLATFORM_AMD__)
  #include <hip/hip_fp16.h>
#else
  #include <cuda_fp16.h>
#endif

namespace turboquant {

// ============================================
// INT2 Matrix Multiply CUDA Kernel
// A (M x K) @ B (K x N) = C (M x N)
// A and B are packed INT2: 4 values per byte
// ============================================

#define TILE_M 32
#define TILE_N 32
#define TILE_K 128
#define THREADS_PER_BLOCK 256

// Unpack INT2 value from byte (values 0-3, centered at 1.5)
__device__ __forceinline__ float unpack_int2(uint8_t packed, int idx) {
  return static_cast<float>((packed >> (idx * 2)) & 0x3) - 1.5f;
}

// Shared memory tiled INT2 matmul kernel
__global__ void matmul_int2_kernel(
    const uint8_t* __restrict__ A,   // [M, K/4] packed INT2
    const uint8_t* __restrict__ B,   // [K, N/4] packed INT2
    float* __restrict__ C,           // [M, N] output FP32
    const float* __restrict__ sa,    // [M / group_size] scale A
    const float* __restrict__ sb,    // [N / group_size] scale B
    const int M, const int N, const int K,
    const int group_size) {

  // Block tile coordinates
  const int bx = blockIdx.x * TILE_N;
  const int by = blockIdx.y * TILE_M;
  const int tx = threadIdx.x % TILE_N;
  const int ty = threadIdx.x / TILE_N;

  // Shared memory for tiles
  __shared__ float As[TILE_M][TILE_K];
  __shared__ float Bs[TILE_K][TILE_N];

  // Accumulator
  float acc = 0.0f;

  // Loop over K tiles
  for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
    // Cooperative loading of A tile into shared memory
    for (int i = threadIdx.x; i < TILE_M * TILE_K; i += THREADS_PER_BLOCK) {
      int row = i / TILE_K;
      int col = i % TILE_K;
      int global_row = by + row;
      int global_col = k_tile + col;

      if (global_row < M && global_col < K) {
        int byte_idx = global_row * (K / 4) + global_col / 4;
        int sub_idx = global_col % 4;
        As[row][col] = unpack_int2(A[byte_idx], sub_idx);
      } else {
        As[row][col] = 0.0f;
      }
    }

    // Cooperative loading of B tile into shared memory
    for (int i = threadIdx.x; i < TILE_K * TILE_N; i += THREADS_PER_BLOCK) {
      int row = i / TILE_N;
      int col = i % TILE_N;
      int global_row = k_tile + row;
      int global_col = bx + col;

      if (global_row < K && global_col < N) {
        int byte_idx = global_row * (N / 4) + global_col / 4;
        int sub_idx = global_col % 4;
        Bs[row][col] = unpack_int2(B[byte_idx], sub_idx);
      } else {
        Bs[row][col] = 0.0f;
      }
    }

    __syncthreads();

    // Compute partial dot product
    int row = ty;
    int col = tx;
    if (by + row < M && bx + col < N) {
      #pragma unroll 8
      for (int k = 0; k < TILE_K; ++k) {
        acc += As[row][k] * Bs[k][col];
      }
    }

    __syncthreads();
  }

  // Write result with scale factors
  int out_row = by + ty;
  int out_col = bx + tx;
  if (out_row < M && out_col < N) {
    float scale_a = sa[out_row / group_size];
    float scale_b = sb[out_col / group_size];
    C[out_row * N + out_col] = acc * scale_a * scale_b;
  }
}

// Host function
void matmul_int2_cuda(
    const uint8_t* A, const uint8_t* B, float* C,
    const float* scales_a, const float* scales_b,
    int M, int N, int K) {

  dim3 block(THREADS_PER_BLOCK);
  dim3 grid(
    (N + TILE_N - 1) / TILE_N,
    (M + TILE_M - 1) / TILE_M
  );

  int group_size = 128;

  matmul_int2_kernel<<<grid, block>>>(
    A, B, C, scales_a, scales_b, M, N, K, group_size
  );

  CUDA_CHECK(cudaGetLastError());
}

}  // namespace turboquant
