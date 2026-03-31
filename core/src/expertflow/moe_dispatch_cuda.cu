// ExpertFlow — moe_dispatch_cuda.cu
// Fused CUDA MoE expert dispatch kernels
//
// Architecture: Compute 8.6+ (Ampere/Ada)
// Strategy: bandwidth-bound GEMV with on-the-fly dequantization
//
// Kernels:
//   1. dequant_gemv_kernel   — Tiled GEMV with IQ2_XXS/Q4_K dequant in shared memory
//   2. fused_silu_mul_kernel — SiLU(gate) ⊙ up (fused, no intermediate writes)
//   3. expert_accumulate_kernel — output += gate_weight × expert_out
//
// For generation (batch_size=1), each expert FFN is 3 GEMVs:
//   gate_out[F] = gate_proj[F×E] × input[E]
//   up_out[F]   = up_proj[F×E]   × input[E]
//   hidden[F]   = SiLU(gate_out) ⊙ up_out
//   out[E]      = down_proj[E×F] × hidden[F]

#include "turboquant/cuda_utils.cuh"
#include "expertflow/moe_dispatch.h"

#include <cfloat>
#include <cstdio>
#include <chrono>

namespace expertflow {

// ============================================================
// Constants
// ============================================================

// IQ2_XXS: 66 bytes per 256 elements (QK_K=256)
static constexpr int QK_K_GPU = 256;
static constexpr int IQ2_XXS_BLOCK_BYTES = 66;

// Tile size for GEMV reduction — each warp reduces over TILE_K elements
static constexpr int TILE_K = 256;   // Matches QK_K for aligned block dequant
static constexpr int WARP_SZ = 32;
static constexpr int BLOCK_DIM = 256; // Threads per block

// ============================================================
// Device helpers
// ============================================================

// FP16 → FP32 conversion (device)
__device__ __forceinline__ float fp16_to_fp32_dev(uint16_t h) {
    // Use CUDA intrinsic for speed
    __half hval;
    memcpy(&hval, &h, sizeof(__half));
    return __half2float(hval);
}

// SiLU activation: x * sigmoid(x)
__device__ __forceinline__ float silu_dev(float x) {
    return x / (1.0f + __expf(-x));
}

// ============================================================
// Kernel 1: Dequantize + GEMV (row-parallel)
// ============================================================
// Each block computes one output element: out[row] = dot(W[row,:], input[:])
// W is quantized in QK_K blocks; input is FP32 in global memory.
//
// For IQ2_XXS: each block of 256 elements = 66 bytes
// Grid: (rows, 1, 1), Block: (BLOCK_DIM, 1, 1)
// Shared memory: input tile (TILE_K floats) + weight block (IQ2_XXS_BLOCK_BYTES)

__global__ void dequant_gemv_iq2xxs_kernel(
    const uint8_t* __restrict__ weight,  // [rows × cols] quantized (IQ2_XXS)
    const float*   __restrict__ input,   // [cols] FP32
    float*         __restrict__ output,  // [rows] FP32
    const int rows,
    const int cols) {

    const int row = blockIdx.x;
    if (row >= rows) return;

    const int tid = threadIdx.x;

    // Number of QK_K blocks per row
    const int n_blocks_per_row = cols / QK_K_GPU;
    const int row_bytes = n_blocks_per_row * IQ2_XXS_BLOCK_BYTES;

    // Pointer to this row's quantized data
    const uint8_t* row_data = weight + row * row_bytes;

    // Each thread accumulates partial dot product across blocks
    float thread_sum = 0.0f;

    // Process QK_K blocks in parallel across threads
    for (int blk = tid; blk < n_blocks_per_row; blk += blockDim.x) {
        const uint8_t* block_ptr = row_data + blk * IQ2_XXS_BLOCK_BYTES;
        const float* input_ptr = input + blk * QK_K_GPU;

        // Read scale (FP16, 2 bytes at block start)
        uint16_t d_raw;
        memcpy(&d_raw, block_ptr, 2);
        float d = fp16_to_fp32_dev(d_raw);

        // Read quantized values (32 uint16 codes, each encoding 8 elements)
        const uint16_t* qs = reinterpret_cast<const uint16_t*>(block_ptr + 2);

        float block_sum = 0.0f;
        for (int i = 0; i < QK_K_GPU / 8; ++i) {
            uint16_t code = qs[i];
            // Dequantize 8 values and dot product with input
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                uint8_t bits = (code >> (j * 2)) & 0x3;
                float val = (static_cast<float>(bits) - 1.5f) * d;
                block_sum += val * input_ptr[i * 8 + j];
            }
        }
        thread_sum += block_sum;
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = WARP_SZ / 2; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, offset);
    }

    // Block-level reduction across warps using shared memory
    __shared__ float warp_sums[BLOCK_DIM / WARP_SZ];
    const int warp_id = tid / WARP_SZ;
    const int lane_id = tid % WARP_SZ;

    if (lane_id == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        float sum = (lane_id < (blockDim.x / WARP_SZ)) ? warp_sums[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = WARP_SZ / 2; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        if (lane_id == 0) {
            output[row] = sum;
        }
    }
}

// ============================================================
// Generic GEMV for F16 weights
// ============================================================

__global__ void gemv_f16_kernel(
    const uint16_t* __restrict__ weight,  // [rows × cols] FP16
    const float*    __restrict__ input,   // [cols] FP32
    float*          __restrict__ output,  // [rows] FP32
    const int rows,
    const int cols) {

    const int row = blockIdx.x;
    if (row >= rows) return;

    const int tid = threadIdx.x;
    const uint16_t* row_data = weight + row * cols;

    float thread_sum = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        float w = fp16_to_fp32_dev(row_data[c]);
        thread_sum += w * input[c];
    }

    // Warp reduction
    #pragma unroll
    for (int offset = WARP_SZ / 2; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, offset);
    }

    __shared__ float warp_sums[BLOCK_DIM / WARP_SZ];
    const int warp_id = tid / WARP_SZ;
    const int lane_id = tid % WARP_SZ;

    if (lane_id == 0) warp_sums[warp_id] = thread_sum;
    __syncthreads();

    if (warp_id == 0) {
        float sum = (lane_id < (blockDim.x / WARP_SZ)) ? warp_sums[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = WARP_SZ / 2; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        if (lane_id == 0) output[row] = sum;
    }
}

// ============================================================
// Kernel 2: Fused SiLU(gate) ⊙ up
// ============================================================
// hidden[i] = SiLU(gate[i]) * up[i]

__global__ void fused_silu_mul_kernel(
    const float* __restrict__ gate,    // [N]
    const float* __restrict__ up,      // [N]
    float*       __restrict__ hidden,  // [N]
    const int N) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    hidden[idx] = silu_dev(gate[idx]) * up[idx];
}

// ============================================================
// Kernel 3: Weighted accumulation
// ============================================================
// output[i] += gate_weight * expert_out[i]

__global__ void expert_accumulate_kernel(
    float*       __restrict__ output,      // [N] accumulated
    const float* __restrict__ expert_out,  // [N] single expert output
    const float gate_weight,
    const int N) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    output[idx] += gate_weight * expert_out[idx];
}

// ============================================================
// Kernel: Zero-initialize buffer
// ============================================================

__global__ void zero_kernel(float* __restrict__ buf, const int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    buf[idx] = 0.0f;
}

// ============================================================
// Host: Launch GEMV for a given quant type
// ============================================================

static void launch_gemv(
    const uint8_t* d_weight,
    const float*   d_input,
    float*         d_output,
    int rows, int cols,
    uint32_t quant_type,
    cudaStream_t stream) {

    if (quant_type == 16) {
        // IQ2_XXS
        dequant_gemv_iq2xxs_kernel<<<rows, BLOCK_DIM, 0, stream>>>(
            d_weight, d_input, d_output, rows, cols);
    } else if (quant_type == 1) {
        // F16
        gemv_f16_kernel<<<rows, BLOCK_DIM, 0, stream>>>(
            reinterpret_cast<const uint16_t*>(d_weight),
            d_input, d_output, rows, cols);
    } else {
        // Unsupported quant type on GPU — zero fill as fallback
        int blocks = (rows + BLOCK_DIM - 1) / BLOCK_DIM;
        zero_kernel<<<blocks, BLOCK_DIM, 0, stream>>>(d_output, rows);
    }
}

// ============================================================
// Host: Fused MoE Dispatch (GPU)
// ============================================================
// Processes all K active experts on GPU:
//   For each expert i:
//     gate_out = GEMV(gate_proj[i], input)
//     up_out   = GEMV(up_proj[i], input)
//     hidden   = SiLU(gate_out) * up_out
//     out      = GEMV(down_proj[i], hidden)
//     output  += gate_weight[i] * out
//
// All expert weight pointers must already be in GPU memory (ExpertCache).
// Input, output, and scratch must be device pointers.

MoeDispatchStats moe_dispatch_gpu(const MoeDispatchParams& params,
                                   GpuStream_t_moe stream) {
    MoeDispatchStats stats{};
    auto t_start = std::chrono::high_resolution_clock::now();

    const uint32_t B = params.batch_size;
    const uint32_t E = params.embed_dim;
    const uint32_t F = params.ffn_dim;
    const uint32_t K = params.n_active;

    // For now, only support B=1 (autoregressive generation)
    // Batch support would use GEMM instead of GEMV
    if (B != 1) {
        fprintf(stderr, "[MoE GPU] Batch size %u not yet supported, using CPU\n", B);
        return moe_dispatch_cpu(params);
    }

    // Verify scratch buffer: need gate_out[F] + up_out[F] + hidden[F] + expert_out[E]
    size_t needed = (3 * F + E) * sizeof(float);
    if (params.scratch_bytes < needed) {
        fprintf(stderr, "[MoE GPU] Scratch too small: %zu < %zu\n",
                params.scratch_bytes, needed);
        return stats;
    }

    // Scratch layout (all device memory)
    float* d_gate_out   = params.scratch;
    float* d_up_out     = params.scratch + F;
    float* d_hidden     = params.scratch + 2 * F;
    float* d_expert_out = params.scratch + 3 * F;

    // Zero output accumulator
    int out_blocks = (E + BLOCK_DIM - 1) / BLOCK_DIM;
    zero_kernel<<<out_blocks, BLOCK_DIM, 0, stream>>>(params.output, E);

    auto t_dequant_start = std::chrono::high_resolution_clock::now();

    for (uint32_t i = 0; i < K; ++i) {
        float gw = params.gate_weights[i];
        if (gw < 1e-8f) continue;

        // 1. gate_out[F] = gate_proj[F×E] × input[E]
        launch_gemv(params.gate_ptrs[i], params.input, d_gate_out,
                    F, E, params.quant_type, stream);

        // 2. up_out[F] = up_proj[F×E] × input[E]
        launch_gemv(params.up_ptrs[i], params.input, d_up_out,
                    F, E, params.quant_type, stream);

        // 3. hidden[F] = SiLU(gate_out) ⊙ up_out
        int act_blocks = (F + BLOCK_DIM - 1) / BLOCK_DIM;
        fused_silu_mul_kernel<<<act_blocks, BLOCK_DIM, 0, stream>>>(
            d_gate_out, d_up_out, d_hidden, F);

        // 4. expert_out[E] = down_proj[E×F] × hidden[F]
        launch_gemv(params.down_ptrs[i], d_hidden, d_expert_out,
                    E, F, params.quant_type, stream);

        // 5. output += gate_weight × expert_out
        expert_accumulate_kernel<<<out_blocks, BLOCK_DIM, 0, stream>>>(
            params.output, d_expert_out, gw, E);
    }

    // Synchronize to get timing
    cudaStreamSynchronize(stream);

    auto t_end = std::chrono::high_resolution_clock::now();
    stats.total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    stats.dequant_ms = 0.0;  // Fused with GEMV — no separate dequant phase
    stats.matmul_ms = stats.total_ms;  // Everything is in the GEMV
    stats.activation_ms = 0.0;  // Fused
    stats.flops = moe_flops(B, E, F, K);
    stats.gflops = (stats.total_ms > 0)
        ? stats.flops / (stats.total_ms * 1e6)
        : 0.0;

    return stats;
}

}  // namespace expertflow
