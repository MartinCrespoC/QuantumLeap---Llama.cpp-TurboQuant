cmake_minimum_required(VERSION 3.20)

# ============================================
# GPU detection: enable CUDA or HIP language as needed
# ============================================
set(TURBOQUANT_GPU_BACKEND "none" CACHE STRING "GPU backend: auto, cuda, hip, none")

if(TURBOQUANT_GPU_BACKEND STREQUAL "auto")
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    set(TURBOQUANT_GPU_BACKEND "cuda")
  elseif(EXISTS "/opt/rocm/bin/hipcc")
    set(TURBOQUANT_GPU_BACKEND "hip")
  else()
    set(TURBOQUANT_GPU_BACKEND "none")
  endif()
endif()

if(TURBOQUANT_GPU_BACKEND STREQUAL "cuda")
  project(TurboQuant LANGUAGES C CXX CUDA ASM)
elseif(TURBOQUANT_GPU_BACKEND STREQUAL "hip")
  # Force clang++ for HIP builds (required for --offload-arch flags)
  if(NOT CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "/opt/rocm/llvm/bin/clang++")
    set(CMAKE_C_COMPILER "/opt/rocm/llvm/bin/clang")
  endif()
  project(TurboQuant LANGUAGES C CXX HIP ASM)
else()
  project(TurboQuant LANGUAGES C CXX ASM)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(BUILD_TESTS "Build test suite" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
set(CUDA_ARCH "86" CACHE STRING "Target CUDA architecture")

# ============================================
# SIMD detection: auto-detect AVX-512 vs AVX2
# ============================================
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)

option(AVX512 "Enable AVX-512 optimizations (auto-detect if ON)" ON)
option(AVX2 "Enable AVX2 optimizations (auto-detect if ON)" ON)
option(FORCE_AVX512 "Force AVX-512 even if not detected" OFF)
option(FORCE_AVX2 "Force AVX2 even if not detected" OFF)

if(NOT AVX512)
  set(HAS_AVX512 FALSE)
  message(STATUS "AVX-512: DISABLED (explicitly disabled with -DAVX512=OFF)")
elseif(FORCE_AVX512 OR COMPILER_SUPPORTS_AVX512)
  # Skip runtime test when building as subdirectory (path resolution issues)
  if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    # Check if CPU actually has AVX-512 at build time
    try_run(AVX512_RUN_RESULT AVX512_COMPILE_RESULT
      ${CMAKE_BINARY_DIR}/avx512_test
      ${CMAKE_SOURCE_DIR}/cmake/check_avx512.cpp
      COMPILE_DEFINITIONS -mavx512f)
    if(AVX512_RUN_RESULT EQUAL 0 OR FORCE_AVX512)
      set(HAS_AVX512 TRUE)
      message(STATUS "AVX-512: ENABLED (CPU supports AVX-512)")
    else()
      set(HAS_AVX512 FALSE)
      message(STATUS "AVX-512: DISABLED (compiler supports it but CPU does not)")
    endif()
  else()
    # Building as subdirectory - trust compiler flag only
    set(HAS_AVX512 ${FORCE_AVX512})
    if(HAS_AVX512)
      message(STATUS "AVX-512: ENABLED (forced, skipped runtime test)")
    else()
      message(STATUS "AVX-512: DISABLED (skipped runtime test when building as subdirectory)")
    endif()
  endif()
else()
  set(HAS_AVX512 FALSE)
  message(STATUS "AVX-512: DISABLED (compiler does not support -mavx512f)")
endif()

if(NOT AVX2)
  set(HAS_AVX2 FALSE)
  message(STATUS "AVX2: DISABLED (explicitly disabled with -DAVX2=OFF)")
elseif(COMPILER_SUPPORTS_AVX2 OR FORCE_AVX2)
  set(HAS_AVX2 TRUE)
  message(STATUS "AVX2: ENABLED")
else()
  set(HAS_AVX2 FALSE)
  message(STATUS "AVX2: DISABLED (compiler does not support -mavx2)")
endif()

message(STATUS "GPU backend: ${TURBOQUANT_GPU_BACKEND}")

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

if(HAS_AVX512)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vbmi2")
  add_compile_definitions(AVX512=1)
endif()
if(HAS_AVX2)
  add_compile_definitions(AVX2=1)
endif()

if(ENABLE_ASAN)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
  set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=address")
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# ============================================
# GPU-specific setup
# ============================================
if(TURBOQUANT_GPU_BACKEND STREQUAL "cuda")
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -O3")
  include_directories(${CUDAToolkit_INCLUDE_DIRS})
  add_compile_definitions(TURBOQUANT_CUDA=1)
elseif(TURBOQUANT_GPU_BACKEND STREQUAL "hip")
  list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")
  find_package(hip REQUIRED)
  find_package(hipblas QUIET)
  add_compile_definitions(TURBOQUANT_HIP=1)
  add_compile_definitions(__HIP_PLATFORM_AMD__=1)
endif()

# ============================================
# TurboQuant Core Library
# ============================================

# Assembly sources — pick based on detected SIMD
set(ASM_SOURCES "")
if(HAS_AVX512)
  list(APPEND ASM_SOURCES
    src/turboquant/polar_transform_avx512.S
    src/kernels/dequant_fast.S
  )
endif()
if(HAS_AVX2)
  list(APPEND ASM_SOURCES
    src/turboquant/polar_transform_avx2.S
    src/kernels/dequant_fast_avx2.S
  )
endif()

# C++ sources
set(CPP_SOURCES
  src/turboquant/polar_transform.cpp
  src/turboquant/hadamard.cpp
  src/turboquant/polarquant.cpp
  src/turboquant/qjl.cpp
  src/turboquant/turboquant_kv.cpp
  src/turboquant/residual_quant.cpp
  src/turboquant/lookup_tables.cpp
  src/llama_integration/ggml_turboquant.cpp
  src/llama_integration/llama_cpp_patch.cpp
)

# GPU sources
set(GPU_SOURCES "")
if(TURBOQUANT_GPU_BACKEND STREQUAL "cuda")
  set(GPU_SOURCES
    src/turboquant/kv_cache_compress.cu
    src/turboquant/turboquant_kv_cuda.cu
    src/kernels/matmul_int4.cu
    src/kernels/matmul_int2.cu
    src/kernels/attention_compressed.cu
  )
elseif(TURBOQUANT_GPU_BACKEND STREQUAL "hip")
  # HIP can compile .cu files directly with hipcc
  set(GPU_SOURCES
    src/turboquant/kv_cache_compress.cu
    src/turboquant/turboquant_kv_cuda.cu
    src/kernels/matmul_int4.cu
    src/kernels/matmul_int2.cu
    src/kernels/attention_compressed.cu
  )
  # Tell CMake these are HIP sources
  set_source_files_properties(${GPU_SOURCES} PROPERTIES LANGUAGE HIP)
endif()

# Main library
add_library(turboquant STATIC ${CPP_SOURCES} ${GPU_SOURCES} ${ASM_SOURCES})
target_include_directories(turboquant PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(TURBOQUANT_GPU_BACKEND STREQUAL "cuda")
  target_link_libraries(turboquant CUDA::cudart CUDA::cublas)
elseif(TURBOQUANT_GPU_BACKEND STREQUAL "hip")
  target_link_libraries(turboquant hip::device)
  if(hipblas_FOUND)
    target_link_libraries(turboquant roc::hipblas)
  endif()
endif()

# ============================================
# ExpertFlow Library (MoE-Aware Inference Engine)
# ============================================
set(EXPERTFLOW_SOURCES
  src/expertflow/expert_map.cpp
  src/expertflow/expert_cache.cpp
  src/expertflow/expert_prefetcher.cpp
  src/expertflow/expert_compressor.cpp
  src/expertflow/routing_predictor.cpp
  src/expertflow/pipeline_controller.cpp
  src/expertflow/moe_dispatch.cpp
  src/expertflow/expertflow_backend.cpp
)

# GPU kernel sources for ExpertFlow
set(EXPERTFLOW_GPU_SOURCES "")
if(TURBOQUANT_GPU_BACKEND STREQUAL "cuda")
  set(EXPERTFLOW_GPU_SOURCES
    src/expertflow/moe_dispatch_cuda.cu
  )
elseif(TURBOQUANT_GPU_BACKEND STREQUAL "hip")
  set(EXPERTFLOW_GPU_SOURCES
    src/expertflow/moe_dispatch_cuda.cu
  )
  set_source_files_properties(${EXPERTFLOW_GPU_SOURCES} PROPERTIES LANGUAGE HIP)
endif()

add_library(expertflow STATIC ${EXPERTFLOW_SOURCES} ${EXPERTFLOW_GPU_SOURCES})
target_include_directories(expertflow PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(expertflow PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(TURBOQUANT_GPU_BACKEND STREQUAL "cuda")
  target_link_libraries(expertflow CUDA::cudart)
  target_compile_definitions(expertflow PRIVATE TURBOQUANT_CUDA=1)
elseif(TURBOQUANT_GPU_BACKEND STREQUAL "hip")
  target_link_libraries(expertflow hip::device)
  target_compile_definitions(expertflow PRIVATE TURBOQUANT_HIP=1)
endif()

# ============================================
# Tests
# ============================================
if(BUILD_TESTS)
  enable_testing()

  add_executable(test_accuracy tests/accuracy_test.cpp)
  target_link_libraries(test_accuracy turboquant)
  add_test(NAME AccuracyTest COMMAND test_accuracy)

  add_executable(test_turboquant_kv tests/test_turboquant_kv.cpp)
  target_link_libraries(test_turboquant_kv turboquant)
  add_test(NAME TurboQuantKVTests COMMAND test_turboquant_kv)

  add_executable(bench_polar tests/benchmark_polar.cpp)
  target_link_libraries(bench_polar turboquant)

  add_executable(bench_matmul tests/benchmark_matmul.cpp)
  target_link_libraries(bench_matmul turboquant)

  add_executable(test_expertflow tests/test_expertflow.cpp)
  target_link_libraries(test_expertflow expertflow)
  add_test(NAME ExpertFlowTests COMMAND test_expertflow)
endif()

# ============================================
# Install
# ============================================
install(TARGETS turboquant ARCHIVE DESTINATION lib)
install(DIRECTORY include/turboquant DESTINATION include)
