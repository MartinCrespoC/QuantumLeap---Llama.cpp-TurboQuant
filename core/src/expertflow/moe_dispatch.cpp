// ExpertFlow — moe_dispatch.cpp
// Fused MoE expert dispatch: CPU reference + GPU dispatch wrapper
//
// CPU reference: dequantize IQ2_XXS → FP32, then matmul + SiLU + accumulate
// GPU path: delegates to ggml-compatible dequant kernels (Phase 4 integration)
//
// MoE FFN per expert:
//   gate = gate_proj(x)  → [B × F]  (embed_dim → ffn_dim)
//   up   = up_proj(x)    → [B × F]
//   hidden = SiLU(gate) ⊙ up
//   out  = down_proj(hidden) → [B × E]  (ffn_dim → embed_dim)
//   output += gate_weight × out

#include "expertflow/moe_dispatch.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace expertflow {

// ============================================================
// IQ2_XXS dequantization (from ggml-common.h format)
// ============================================================

// IQ2_XXS block: ggml_half (2 bytes) + QK_K/8 uint16 (64 bytes) = 66 bytes per 256 elements
// Each uint16 encodes 8 values via lookup table.
// For the CPU reference, we use a simplified dequant that extracts the scale
// and approximates the values. Full accuracy requires the ggml lookup tables.

static constexpr uint32_t QK_K = 256;

// Block sizes for supported quant types (type_size, block_size)
struct QuantInfo {
    uint32_t type_size;
    uint32_t block_size;
    float    bits_per_weight;
};

static QuantInfo get_quant_info(uint32_t type) {
    switch (type) {
        case 0:  return {4, 1, 32.0f};      // F32
        case 1:  return {2, 1, 16.0f};       // F16
        case 10: return {84, 256, 2.625f};   // Q2_K
        case 11: return {110, 256, 3.4375f}; // Q3_K
        case 12: return {144, 256, 4.5f};    // Q4_K
        case 13: return {176, 256, 5.5f};    // Q5_K
        case 14: return {210, 256, 6.5625f}; // Q6_K
        case 16: return {66, 256, 2.0625f};  // IQ2_XXS
        case 17: return {74, 256, 2.3125f};  // IQ2_XS
        case 22: return {82, 256, 2.5625f};  // IQ2_S
        default: return {2, 1, 16.0f};       // Fallback to F16
    }
}

// Simplified FP16 → FP32 conversion (IEEE 754 half-precision)
static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        // Denormalized
        float val = std::ldexp(static_cast<float>(mant), -24);
        return sign ? -val : val;
    }
    if (exp == 31) {
        if (mant == 0) return sign ? -INFINITY : INFINITY;
        return NAN;
    }

    float val = std::ldexp(static_cast<float>(mant + 1024), static_cast<int>(exp) - 25);
    return sign ? -val : val;
}

// Dequantize IQ2_XXS block to FP32
// Each block: 2 bytes scale (fp16) + 32 × 2 bytes (uint16 codebook indices)
// For reference implementation: extract scale and distribute approximately
static void dequant_iq2_xxs_block(const uint8_t* src, float* dst) {
    uint16_t d_raw;
    memcpy(&d_raw, src, 2);
    float d = fp16_to_fp32(d_raw);

    // The uint16 values encode 8 elements each via a codebook.
    // For the CPU reference, we approximate: value ≈ d × (code - mean) / range
    // Full accuracy requires the IQ2_XXS codebook from ggml-quants.c
    const uint16_t* qs = reinterpret_cast<const uint16_t*>(src + 2);

    for (uint32_t i = 0; i < QK_K / 8; ++i) {
        uint16_t code = qs[i];
        // Extract 8 values from the code (2 bits each, packed)
        for (uint32_t j = 0; j < 8; ++j) {
            // Approximate: map 2-bit values {0,1,2,3} to {-1.5, -0.5, 0.5, 1.5}
            uint8_t bits = (code >> (j * 2)) & 0x3;
            float val = (static_cast<float>(bits) - 1.5f) * d;
            dst[i * 8 + j] = val;
        }
    }
}

// Dequantize F16 to F32
static void dequant_f16(const uint8_t* src, float* dst, size_t n) {
    const uint16_t* fp16 = reinterpret_cast<const uint16_t*>(src);
    for (size_t i = 0; i < n; ++i) {
        dst[i] = fp16_to_fp32(fp16[i]);
    }
}

// Dequantize F32 (identity copy)
static void dequant_f32(const uint8_t* src, float* dst, size_t n) {
    memcpy(dst, src, n * sizeof(float));
}

size_t dequantize_weights(const uint8_t* src, float* dst,
                           uint32_t rows, uint32_t cols,
                           uint32_t quant_type) {
    size_t n_elements = static_cast<size_t>(rows) * cols;

    if (quant_type == 0) {  // F32
        dequant_f32(src, dst, n_elements);
        return n_elements;
    }
    if (quant_type == 1) {  // F16
        dequant_f16(src, dst, n_elements);
        return n_elements;
    }

    // Quantized types: process block by block
    auto qi = get_quant_info(quant_type);
    if (qi.block_size == 0 || qi.block_size == 1) {
        // Unknown type, zero-fill
        memset(dst, 0, n_elements * sizeof(float));
        return n_elements;
    }

    size_t n_blocks = n_elements / qi.block_size;
    const uint8_t* block_ptr = src;

    for (size_t b = 0; b < n_blocks; ++b) {
        if (quant_type == 16) {  // IQ2_XXS
            dequant_iq2_xxs_block(block_ptr, dst + b * qi.block_size);
        } else {
            // For other quant types, zero-fill in reference impl
            // Full support requires ggml dequantization functions
            memset(dst + b * qi.block_size, 0, qi.block_size * sizeof(float));
        }
        block_ptr += qi.type_size;
    }

    return n_elements;
}

// ============================================================
// FP32 matrix operations
// ============================================================

void matmul_f32(const float* A, const float* B, float* C,
                uint32_t M, uint32_t K, uint32_t N) {
    // C[M×N] = A[M×K] × B[N×K]^T
    // Simple reference implementation (O(MNK), no tiling)
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            const float* a_row = A + m * K;
            const float* b_row = B + n * K;
            for (uint32_t k = 0; k < K; ++k) {
                sum += a_row[k] * b_row[k];
            }
            C[m * N + n] = sum;
        }
    }
}

void silu_f32(float* x, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) {
        float val = x[i];
        x[i] = val / (1.0f + std::exp(-val));
    }
}

void hadamard_f32(const float* a, const float* b, float* dst, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) {
        dst[i] = a[i] * b[i];
    }
}

// ============================================================
// CPU Reference MoE Dispatch
// ============================================================

MoeDispatchStats moe_dispatch_cpu(const MoeDispatchParams& params) {
    MoeDispatchStats stats{};
    auto t_start = std::chrono::high_resolution_clock::now();

    const uint32_t B = params.batch_size;
    const uint32_t E = params.embed_dim;
    const uint32_t F = params.ffn_dim;
    const uint32_t K = params.n_active;

    // Verify scratch buffer is large enough
    size_t needed = moe_scratch_bytes(B, F);
    if (params.scratch_bytes < needed) {
        fprintf(stderr, "[MoE Dispatch] Scratch too small: %zu < %zu\n",
                params.scratch_bytes, needed);
        return stats;
    }

    // Scratch layout: gate_out[B×F], up_out[B×F], hidden[B×F]
    float* gate_out = params.scratch;
    float* up_out   = params.scratch + B * F;
    float* hidden   = params.scratch + 2 * B * F;

    // Zero output accumulator
    memset(params.output, 0, B * E * sizeof(float));

    // Temporary buffers for dequantized weights
    // gate_proj: [F × E], up_proj: [F × E], down_proj: [E × F]
    std::vector<float> w_gate(F * E);
    std::vector<float> w_up(F * E);
    std::vector<float> w_down(E * F);

    auto t_dequant_start = std::chrono::high_resolution_clock::now();

    for (uint32_t i = 0; i < K; ++i) {
        float gw = params.gate_weights[i];
        if (gw < 1e-8f) continue;  // Skip zero-weight experts

        // 1. Dequantize expert weights
        dequantize_weights(params.gate_ptrs[i], w_gate.data(),
                           F, E, params.quant_type);
        dequantize_weights(params.up_ptrs[i], w_up.data(),
                           F, E, params.quant_type);
        dequantize_weights(params.down_ptrs[i], w_down.data(),
                           E, F, params.quant_type);

        auto t_matmul_start = std::chrono::high_resolution_clock::now();

        // 2. gate_out = input × gate_proj^T  → [B × F]
        matmul_f32(params.input, w_gate.data(), gate_out, B, E, F);

        // 3. up_out = input × up_proj^T  → [B × F]
        matmul_f32(params.input, w_up.data(), up_out, B, E, F);

        auto t_act_start = std::chrono::high_resolution_clock::now();

        // 4. hidden = SiLU(gate_out) ⊙ up_out  → [B × F]
        silu_f32(gate_out, B * F);
        hadamard_f32(gate_out, up_out, hidden, B * F);

        auto t_down_start = std::chrono::high_resolution_clock::now();

        // 5. expert_out = hidden × down_proj^T  → [B × E]
        // Accumulate: output += gate_weight × expert_out
        // We reuse gate_out as temporary for expert_out (it's B×F, we need B×E)
        std::vector<float> expert_out(B * E);
        matmul_f32(hidden, w_down.data(), expert_out.data(), B, F, E);

        // 6. Weighted accumulation
        for (uint32_t j = 0; j < B * E; ++j) {
            params.output[j] += gw * expert_out[j];
        }

        auto t_expert_end = std::chrono::high_resolution_clock::now();

        stats.matmul_ms += std::chrono::duration<double, std::milli>(
            t_act_start - t_matmul_start).count();
        stats.matmul_ms += std::chrono::duration<double, std::milli>(
            t_expert_end - t_down_start).count();
        stats.activation_ms += std::chrono::duration<double, std::milli>(
            t_down_start - t_act_start).count();
    }

    auto t_dequant_end = std::chrono::high_resolution_clock::now();
    stats.dequant_ms = std::chrono::duration<double, std::milli>(
        t_dequant_end - t_dequant_start).count() - stats.matmul_ms - stats.activation_ms;

    auto t_end = std::chrono::high_resolution_clock::now();
    stats.total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    stats.flops = moe_flops(B, E, F, K);
    stats.gflops = (stats.total_ms > 0)
        ? stats.flops / (stats.total_ms * 1e6)  // GFLOP/s
        : 0.0;

    return stats;
}

// ============================================================
// GPU MoE Dispatch — CPU fallback when no GPU backend compiled
// ============================================================

#if !defined(TURBOQUANT_CUDA) && !defined(TURBOQUANT_HIP)
MoeDispatchStats moe_dispatch_gpu(const MoeDispatchParams& params,
                                   GpuStream_t_moe /*stream*/) {
    // No GPU backend — fall back to CPU reference
    fprintf(stderr, "[MoE Dispatch] No GPU backend, using CPU fallback\n");
    return moe_dispatch_cpu(params);
}
#endif

// ============================================================
// Convenience wrapper
// ============================================================

MoeDispatchStats moe_dispatch(
    const float* input,
    float* output,
    float* scratch,
    size_t scratch_bytes,
    uint32_t batch_size,
    const MoeArchitecture& arch,
    const PipelineController::ExpertPointers& experts,
    uint32_t quant_type,
    GpuStream_t_moe stream) {

    MoeDispatchParams params{};
    params.input       = input;
    params.batch_size  = batch_size;
    params.embed_dim   = arch.embed_dim;
    params.ffn_dim     = arch.expert_ffn_dim;
    params.n_active    = static_cast<uint32_t>(experts.gate_ptrs.size());
    params.gate_weights = experts.weights.data();
    params.gate_ptrs   = reinterpret_cast<const uint8_t* const*>(experts.gate_ptrs.data());
    params.up_ptrs     = reinterpret_cast<const uint8_t* const*>(experts.up_ptrs.data());
    params.down_ptrs   = reinterpret_cast<const uint8_t* const*>(experts.down_ptrs.data());
    params.quant_type  = quant_type;
    params.shared_gate_ptr = nullptr;
    params.shared_up_ptr   = nullptr;
    params.shared_down_ptr = nullptr;
    params.output      = output;
    params.scratch     = scratch;
    params.scratch_bytes = scratch_bytes;

    if (stream) {
        return moe_dispatch_gpu(params, stream);
    }
    return moe_dispatch_cpu(params);
}

}  // namespace expertflow
