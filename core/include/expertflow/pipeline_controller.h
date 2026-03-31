// ExpertFlow — MoE-Aware Inference Engine
// pipeline_controller.h: Orchestrates the full MoE inference pipeline
//
// Coordinates 3 GPU streams to overlap attention, expert compute, and
// expert prefetching. This is the top-level controller that ties together
// ExpertMap, ExpertCache, and ExpertPrefetcher.
//
// Stream 0: Attention + Router + Shared Expert (permanent GPU tensors)
// Stream 1: Expert matmul from cache (active experts)
// Stream 2: Expert H2D prefetch (next layer's experts)

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "expertflow/expert_cache.h"
#include "expertflow/expert_map.h"
#include "expertflow/expert_prefetcher.h"

// Forward-declare GPU stream/event types (same as prefetcher)
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#elif defined(__CUDACC__) || defined(EXPERTFLOW_CUDA)
#include <cuda_runtime.h>
#endif

namespace expertflow {

// Per-layer routing result from the router MLP
struct LayerRouting {
    uint32_t layer_id;
    std::vector<uint32_t> expert_ids;    // Top-K selected expert IDs
    std::vector<float>    gate_weights;  // Gating weights for each expert
};

// Per-token profiling data
struct TokenProfile {
    uint64_t token_id;
    double   attention_ms;       // Time in attention + router
    double   expert_compute_ms;  // Time in expert matmul
    double   expert_transfer_ms; // Time waiting for H2D transfer
    double   total_ms;           // Total token time
    uint32_t cache_hits;         // Expert cache hits this token
    uint32_t cache_misses;       // Expert cache misses this token
    uint32_t prefetch_hits;      // Prefetch predictions that were correct
};

// Pipeline performance summary
struct PipelineStats {
    uint64_t tokens_generated;
    double   total_time_ms;
    double   avg_tok_per_sec;
    double   peak_tok_per_sec;
    double   avg_cache_hit_rate;
    double   avg_prefetch_accuracy;
    uint64_t total_pipeline_stalls; // Times compute waited for transfer

    // Per-component breakdown (percentage of total time)
    double pct_attention;
    double pct_expert_compute;
    double pct_expert_transfer;
    double pct_overhead;
};

// Configuration for the full pipeline
struct PipelineConfig {
    // Memory budget
    size_t   expert_cache_vram_bytes;  // VRAM for expert cache (e.g., 4GB)
    size_t   staging_buffer_bytes;     // Pinned host staging buffer

    // Optimization flags
    bool     enable_prefetch;          // Layer-ahead expert prefetching
    bool     enable_coalescing;        // Coalesce contiguous expert DMA
    bool     enable_profiling;         // Per-token profiling (adds overhead)
    uint32_t speculative_top_k;        // Speculative load top-K (>= n_experts_used)
    uint32_t reserved_hot_per_layer;   // Hot experts pinned per layer

    // Cache tuning
    float    recency_weight;           // LRU vs frequency balance [0,1]

    // Generic defaults (works for any MoE model)
    static PipelineConfig defaults() {
        return PipelineConfig{
            .expert_cache_vram_bytes = 1ULL * 1024 * 1024 * 1024,  // 1 GB default
            .staging_buffer_bytes    = 64ULL * 1024 * 1024,         // 64 MB
            .enable_prefetch         = true,
            .enable_coalescing       = true,
            .enable_profiling        = false,
            .speculative_top_k       = 12,    // Load 12, need top-K → 50% slack
            .reserved_hot_per_layer  = 5,     // Pin top-5 per layer
            .recency_weight          = 0.6f,  // 60% LRU, 40% frequency
        };
    }

    // Auto-configure from model architecture and available VRAM.
    // Allocates VRAM for expert cache after reserving space for shared weights.
    //   vram_bytes: total GPU VRAM available (e.g. 6 GB)
    //   arch: model architecture from ExpertMap
    //   kv_compression_ratio: KV cache bytes after compression / bytes before
    //     1.0 = no compression (FP16 KV), ~0.22 = TQ3 (3.5/16), ~0.16 = TQ2
    //     Default 200 MB baseline KV is scaled by this ratio.
    static PipelineConfig auto_config(size_t vram_bytes,
                                       const MoeArchitecture& arch,
                                       float kv_compression_ratio = 1.0f) {
        auto cfg = defaults();

        // KV overhead: 200 MB baseline scaled by compression ratio
        // TQ3 (3.5 bits vs FP16 16 bits) → ratio ~0.22 → 44 MB instead of 200 MB
        size_t kv_overhead = static_cast<size_t>(
            200ULL * 1024 * 1024 * std::max(0.1f, kv_compression_ratio));

        // Reserve VRAM for shared weights + compressed KV
        size_t shared_overhead = arch.shared_weight_bytes + kv_overhead;
        if (vram_bytes > shared_overhead) {
            cfg.expert_cache_vram_bytes = vram_bytes - shared_overhead;
        } else {
            // Minimal cache: enough for 2 × top-K experts per layer
            cfg.expert_cache_vram_bytes = 2 * arch.n_experts_used *
                                          arch.expert_weight_bytes;
        }

        // Staging buffer: enough for 2 × top-K experts (double-buffered)
        if (arch.expert_weight_bytes > 0 && arch.n_experts_used > 0) {
            cfg.staging_buffer_bytes = std::max(
                cfg.staging_buffer_bytes,
                static_cast<size_t>(2 * arch.n_experts_used * arch.expert_weight_bytes));
        }

        // Speculative prefetch: 50% extra over top-K
        if (arch.n_experts_used > 0) {
            cfg.speculative_top_k = arch.n_experts_used + arch.n_experts_used / 2;
        }

        return cfg;
    }
};

// Pipeline Controller — top-level orchestrator for MoE inference
//
// Lifecycle:
//   1. Create PipelineController
//   2. Call init(gguf_path, config) → parses model, allocates cache + streams
//   3. For each token:
//      a. Call begin_token()
//      b. For each layer:
//         - Call process_layer(layer_id, routing) → returns expert GPU pointers
//         - Use pointers for matmul (or call our fused kernel)
//      c. Call end_token() → advances LRU clock, logs profiling
//   4. Call release() to free GPU resources
//
class PipelineController {
public:
    PipelineController() = default;
    ~PipelineController();

    // Non-copyable
    PipelineController(const PipelineController&) = delete;
    PipelineController& operator=(const PipelineController&) = delete;

    // Initialize the full pipeline from a GGUF model file.
    // 1. Parses GGUF → ExpertMap
    // 2. mmaps the file for CPU-side expert weight access
    // 3. Allocates GPU expert cache
    // 4. Creates HIP/CUDA streams
    // 5. Optionally profiles hot experts from a warmup pass
    bool init(const std::string& gguf_path, const PipelineConfig& config);

    // Release all resources (GPU memory, streams, mmap)
    void release();

    // Is the pipeline initialized and ready?
    bool is_ready() const { return ready_; }

    // Begin a new token generation step
    void begin_token();

    // Process one layer's MoE block.
    // Input: routing result from the router MLP
    // Output: GPU pointers for each active expert's 3 projections
    //
    // The returned pointers are valid until the next process_layer() call.
    // Format: [expert0_gate, expert0_up, expert0_down, expert1_gate, ...]
    //
    // This function:
    //   1. Looks up experts in cache
    //   2. Issues async H2D for misses
    //   3. Waits for transfers if needed (pipeline stall)
    //   4. Submits prefetch for layer+1
    //   5. Returns GPU pointers ready for matmul
    struct ExpertPointers {
        std::vector<uint8_t*> gate_ptrs;   // gate_proj GPU pointers
        std::vector<uint8_t*> up_ptrs;     // up_proj GPU pointers
        std::vector<uint8_t*> down_ptrs;   // down_proj GPU pointers
        std::vector<float>    weights;     // Gating weights
    };

    ExpertPointers process_layer(const LayerRouting& routing);

    // End current token generation step.
    // Advances LRU clock, records profiling, syncs streams.
    void end_token();

    // Run hot expert profiling warmup.
    // Generates n_warmup_tokens with random routing to measure expert frequency.
    // After profiling, reserves hot experts in cache.
    void warmup_profile(size_t n_warmup_tokens = 128);

    // Access sub-components
    const ExpertMap&        expert_map()  const { return map_; }
    const ExpertCache&      cache()       const { return cache_; }
    const ExpertPrefetcher& prefetcher()  const { return prefetcher_; }

    // Pipeline statistics
    PipelineStats compute_stats() const;

    // Per-token profiles (only if enable_profiling is on)
    const std::vector<TokenProfile>& token_profiles() const {
        return profiles_;
    }

    // Print full performance report
    void print_report() const;

    // Get the compute stream (for external use in attention/matmul)
    GpuStream_t compute_stream() const { return compute_stream_; }

    // Get the expert compute stream
    GpuStream_t expert_stream() const { return expert_stream_; }

private:
    bool ready_ = false;

    // Sub-components
    ExpertMap        map_;
    ExpertCache      cache_;
    ExpertPrefetcher prefetcher_;
    PipelineConfig   config_{};

    // mmap'd GGUF file
    uint8_t* mmap_base_ = nullptr;
    size_t   mmap_size_ = 0;
    int      mmap_fd_   = -1;

    // GPU streams
    GpuStream_t compute_stream_ = nullptr;  // Stream 0: attention
    GpuStream_t expert_stream_  = nullptr;  // Stream 1: expert matmul
    // Stream 2 is managed by ExpertPrefetcher

    // GPU events for stream synchronization
    GpuEvent_t expert_ready_event_ = nullptr;  // Signals expert matmul done

    // Profiling
    std::vector<TokenProfile> profiles_;
    TokenProfile              current_profile_{};

    // Hot expert reservation map: [layer_id] → list of hot expert_ids
    std::vector<std::vector<uint32_t>> hot_experts_;
};

}  // namespace expertflow
