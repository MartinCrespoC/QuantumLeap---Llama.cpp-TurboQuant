// ExpertFlow — MoE-Aware Inference Engine
// moe_dispatch.h: Fused MoE expert dispatch kernel interface
//
// Performs the complete MoE FFN block in a single dispatch:
//   output = Σ(gate_weight_i × expert_i(input))
//   expert_i(x) = down_proj(SiLU(gate_proj(x)) ⊙ up_proj(x))
//
// Supports:
//   - Quantized weights from ExpertCache (IQ2_XXS, Q4_K, etc.)
//   - Variable number of active experts (top-K routing)
//   - Fused SiLU activation to avoid intermediate writes
//   - CPU reference and HIP/CUDA GPU implementations
//
// Dimensions are read from MoeArchitecture at runtime.
// Example: embed_dim=3072, expert_ffn_dim=1024, top_k=8 for Qwen MoE

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "expertflow/expert_cache.h"
#include "expertflow/expert_map.h"
#include "expertflow/pipeline_controller.h"

namespace expertflow {

// MoE dispatch parameters for a single layer
struct MoeDispatchParams {
    // Input hidden state (batch_size × embed_dim, FP32)
    const float* input;
    uint32_t batch_size;
    uint32_t embed_dim;

    // Expert FFN dimensions
    uint32_t ffn_dim;        // Intermediate dimension (from model metadata)

    // Active experts from routing
    uint32_t n_active;       // Number of active experts (top-K)
    const float* gate_weights;  // [n_active] gating weights from router

    // Expert weight pointers from ExpertCache
    // Each is a quantized weight matrix in GGUF format
    const uint8_t* const* gate_ptrs;  // [n_active] → gate_proj weights
    const uint8_t* const* up_ptrs;    // [n_active] → up_proj weights
    const uint8_t* const* down_ptrs;  // [n_active] → down_proj weights
    uint32_t quant_type;     // ggml quant type of the expert weights

    // Shared expert (optional, Qwen3.5 has one per layer)
    const uint8_t* shared_gate_ptr;   // Shared expert gate_proj (nullptr if none)
    const uint8_t* shared_up_ptr;     // Shared expert up_proj
    const uint8_t* shared_down_ptr;   // Shared expert down_proj

    // Output buffer (batch_size × embed_dim, FP32)
    float* output;

    // Scratch buffer for intermediate computations
    // Minimum size: batch_size × ffn_dim × sizeof(float) × 3
    float* scratch;
    size_t scratch_bytes;
};

// MoE dispatch statistics (per-call profiling)
struct MoeDispatchStats {
    double dequant_ms;     // Time dequantizing weights
    double matmul_ms;      // Time in matrix multiplications
    double activation_ms;  // Time in SiLU + element-wise ops
    double total_ms;       // Total dispatch time
    uint64_t flops;        // FLOPs for this dispatch
    double gflops;         // GFLOP/s achieved
};

// Compute minimum scratch buffer size for MoE dispatch
inline size_t moe_scratch_bytes(uint32_t batch_size, uint32_t ffn_dim) {
    // Need 3 temporary buffers of size (batch_size × ffn_dim):
    //   gate_out, up_out, hidden (for SiLU(gate) ⊙ up)
    return static_cast<size_t>(batch_size) * ffn_dim * sizeof(float) * 3;
}

// Compute FLOPs for a single MoE dispatch
inline uint64_t moe_flops(uint32_t batch_size, uint32_t embed_dim,
                           uint32_t ffn_dim, uint32_t n_active) {
    // Per expert: gate_proj + up_proj + down_proj matmuls + SiLU + hadamard
    // gate_proj: B × E × F (multiply-add)
    // up_proj:   B × E × F
    // down_proj: B × F × E
    // SiLU + hadamard: B × F (negligible vs matmul)
    uint64_t per_expert = static_cast<uint64_t>(batch_size) *
                          (2ULL * embed_dim * ffn_dim +  // gate + up
                           2ULL * ffn_dim * embed_dim);  // down
    return per_expert * n_active;
}

// ============================================================
// CPU Reference Implementation
// ============================================================

// Dequantize a quantized weight matrix to FP32.
// Supports IQ2_XXS, Q4_K, Q5_K, Q6_K, F16, F32.
// Returns number of elements dequantized (rows × cols).
size_t dequantize_weights(const uint8_t* src, float* dst,
                           uint32_t rows, uint32_t cols,
                           uint32_t quant_type);

// FP32 matrix multiply: C = A × B^T
// A: [M × K], B: [N × K] (row-major, B is transposed)
// C: [M × N]
void matmul_f32(const float* A, const float* B, float* C,
                uint32_t M, uint32_t K, uint32_t N);

// SiLU activation: x * sigmoid(x)
void silu_f32(float* x, uint32_t n);

// Element-wise multiply: dst = a ⊙ b
void hadamard_f32(const float* a, const float* b, float* dst, uint32_t n);

// Fused MoE dispatch — CPU reference implementation.
// Processes all active experts and accumulates the weighted output.
// This is the baseline for correctness verification.
MoeDispatchStats moe_dispatch_cpu(const MoeDispatchParams& params);

// ============================================================
// GPU Implementation (HIP/CUDA)
// ============================================================

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
using GpuStream_t_moe = hipStream_t;
#elif defined(__CUDACC__) || defined(EXPERTFLOW_CUDA)
#include <cuda_runtime.h>
using GpuStream_t_moe = cudaStream_t;
#else
using GpuStream_t_moe = void*;
#endif

// GPU MoE dispatch — fused kernel that processes all experts in one launch.
// Expert weights remain quantized in GPU memory; dequant happens on-the-fly.
// Uses shared memory tiling for the matmul.
MoeDispatchStats moe_dispatch_gpu(const MoeDispatchParams& params,
                                   GpuStream_t_moe stream);

// Convenience wrapper using PipelineController's ExpertPointers
MoeDispatchStats moe_dispatch(
    const float* input,
    float* output,
    float* scratch,
    size_t scratch_bytes,
    uint32_t batch_size,
    const MoeArchitecture& arch,
    const PipelineController::ExpertPointers& experts,
    uint32_t quant_type,
    GpuStream_t_moe stream = nullptr);

}  // namespace expertflow
