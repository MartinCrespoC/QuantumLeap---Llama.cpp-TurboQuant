#pragma once

// ============================================
// HIP/CUDA Compatibility Layer
// When compiled with HIP (AMD ROCm), we need explicit mappings
// ============================================
#if defined(TURBOQUANT_HIP) || defined(__HIP_PLATFORM_AMD__)
  #include <hip/hip_runtime.h>
  #if __has_include(<hipblas/hipblas.h>)
    #include <hipblas/hipblas.h>
  #endif
  
  // HIP compatibility macros
  #define cudaError_t hipError_t
  #define cudaSuccess hipSuccess
  #define cudaGetErrorString hipGetErrorString
  #define cudaMalloc hipMalloc
  #define cudaFree hipFree
  #define cudaMallocHost hipHostMalloc
  #define cudaFreeHost hipHostFree
  #define cudaMemcpy hipMemcpy
  #define cudaMemcpyAsync hipMemcpyAsync
  #define cudaMemcpyHostToDevice hipMemcpyHostToDevice
  #define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
  #define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
  #define cudaMemset hipMemset
  #define cudaMemsetAsync hipMemsetAsync
  #define cudaGetDeviceCount hipGetDeviceCount
  #define cudaSetDevice hipSetDevice
  #define cudaGetDevice hipGetDevice
  #define cudaGetDeviceProperties hipGetDeviceProperties
  #define cudaMemGetInfo hipMemGetInfo
  #define cudaDeviceSynchronize hipDeviceSynchronize
  #define cudaStreamCreate hipStreamCreate
  #define cudaStreamCreateWithFlags hipStreamCreateWithFlags
  #define cudaStreamDestroy hipStreamDestroy
  #define cudaStreamSynchronize hipStreamSynchronize
  #define cudaStreamNonBlocking hipStreamNonBlocking
  #define cudaEventCreate hipEventCreate
  #define cudaEventCreateWithFlags hipEventCreateWithFlags
  #define cudaEventDestroy hipEventDestroy
  #define cudaEventRecord hipEventRecord
  #define cudaEventSynchronize hipEventSynchronize
  #define cudaEventQuery hipEventQuery
  #define cudaEventDisableTiming hipEventDisableTiming
  #define cudaStream_t hipStream_t
  #define cudaEvent_t hipEvent_t
  #define cudaDeviceProp hipDeviceProp_t
  #define __half _Float16
  #define cublasStatus_t hipblasStatus_t
  #define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
  #define cublasHandle_t hipblasHandle_t
  #define cublasCreate hipblasCreate
  #define cublasDestroy hipblasDestroy
  #define cublasSetStream hipblasSetStream
  #define cublasSgemm hipblasSgemm
  #define cublasSgemv hipblasSgemv
  #define CUBLAS_OP_N HIPBLAS_OP_N
  #define CUBLAS_OP_T HIPBLAS_OP_T
  
  // HIP warp sync uses 64-bit masks (AMD wavefront = 64 threads)
  #define __shfl_down_sync(mask, var, offset) __shfl_down(var, offset)
#else
  #include <cuda_runtime.h>
  #include <cublas_v2.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <memory>

namespace turboquant {
namespace cuda {

// ============================================
// Error Checking (works for both CUDA and HIP)
// ============================================

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
      fprintf(stderr, "GPU error at %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                        \
      abort();                                                                \
    }                                                                         \
  } while (0)

#define CUBLAS_CHECK(call)                                                    \
  do {                                                                        \
    cublasStatus_t err = (call);                                              \
    if (err != CUBLAS_STATUS_SUCCESS) {                                       \
      fprintf(stderr, "BLAS error at %s:%d: %d\n", __FILE__, __LINE__,       \
              static_cast<int>(err));                                          \
      abort();                                                                \
    }                                                                         \
  } while (0)

// ============================================
// RAII GPU Memory
// ============================================

struct CudaDeleter {
  void operator()(void* ptr) const {
    if (ptr) cudaFree(ptr);
  }
};

template <typename T>
using CudaUniquePtr = std::unique_ptr<T, CudaDeleter>;

template <typename T>
CudaUniquePtr<T> cuda_alloc(size_t count) {
  T* ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
  return CudaUniquePtr<T>(ptr);
}

// ============================================
// Memory Transfer
// ============================================

template <typename T>
void cuda_copy_h2d(T* dst, const T* src, size_t count) {
  CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void cuda_copy_d2h(T* dst, const T* src, size_t count) {
  CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T>
void cuda_copy_d2d(T* dst, const T* src, size_t count) {
  CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToDevice));
}

// ============================================
// Kernel Launch Utilities
// ============================================

// Tile sizes for matrix operations
constexpr int TILE_SIZE = 32;
constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS_PER_BLOCK = 1024;

// Calculate grid dimensions
inline dim3 grid_dim(int total, int block_size) {
  return dim3((total + block_size - 1) / block_size);
}

inline dim3 grid_dim_2d(int rows, int cols, int block_x, int block_y) {
  return dim3((cols + block_x - 1) / block_x, (rows + block_y - 1) / block_y);
}

// Get optimal block size for a kernel
template <typename KernelFunc>
int optimal_block_size(KernelFunc kernel) {
  int min_grid_size, block_size;
  cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel);
  return block_size;
}

// ============================================
// Device Info
// ============================================

inline int device_count() {
  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  return count;
}

inline size_t free_memory(int device = 0) {
  CUDA_CHECK(cudaSetDevice(device));
  size_t free_mem, total_mem;
  CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
  return free_mem;
}

inline int compute_capability(int device = 0) {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  return prop.major * 10 + prop.minor;
}

}  // namespace cuda
}  // namespace turboquant
